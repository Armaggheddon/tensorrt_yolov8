from typing import List, Tuple
from enum import Enum
from dataclasses import dataclass, field

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

_logger_level = {
    "info": trt.Logger.INFO,
    "warning": trt.Logger.WARNING,
    "error": trt.Logger.ERROR
}

class DynamicType(Enum):
    NONE = 0
    BATCH = 1   # Only the batch size is dynamic

#####
# Manage dynamic batch with following strategy:
# 1. Set the batch size to 1 (or min)
# 2. Allocate memory for the minimum possible batch size 
# 3. Store dynamic sizes for each input and output tensor
#   by checking if binding is trt:TensorIOMode.INPUT/OUTPUT
# 4. For each shape store min/opt/max shapes from engine.get_tensor_profile_shape(0, binding)
#   where 0 is the optimization profile index
# 5. Before running inference, set the check if given batch size is valid.
#   If not, return error, if is, store the user shape so that when the output is ready
#   we can copy the output to the host based on the output shape/batch
# 6. Run inference
# 7. Copy the output to the host based on the user shape/batch
# ...


class EngineHelper():

    def __init__(
            self,
            engine_path:str,
            log_level:str="error"
    ) -> None: 

        ########################################
        # Load engine

        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()

        TRT_LOGGER = trt.Logger(_logger_level[log_level])
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")

        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            
        if not engine:
            raise ValueError(
                f"Failed loading engine from {self.engine_path}.\n"\
                f"Most likely the engine was built with a version of TensorRT different from {trt.__version__}.\n"\
                f"Please rebuild the engine with the correct version of TensorRT and try again")
        context = engine.create_execution_context()

        print(f"Input shape: {engine.get_tensor_shape(engine[0])}")
        # Retuns, based on the optimization profile, the possible shapes for the input tensor
        # as a list of min, opt, max shapes (x, x, x, x) for each dimension
        # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Engine.html#tensorrt.ICudaEngine.get_profile_shape
        print(f"Optimization profiles = {engine.get_profile_shape(0, engine[0])}")

        host_inputs  = []
        cuda_inputs  = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        self.input_shapes = []
        self.output_shapes = []

        self.is_dynamic, self.min_batch, self.opt_batch, self.max_batch = self._is_dynamic(engine)

        for binding in engine:
            binding_idx = engine[binding]
            binding_shape = context.get_tensor_shape(binding) # tuple with (1, 3, 640, 640) for example

            self.allocated_shape = (self.max_batch, *binding_shape[1:])

            print(f"Original shape: {context.get_tensor_shape(binding)}, allocated: {self.allocated_shape}")

            # if binding_shape[0] == -1:
            #     # This can happen if the engine is dynamic, but only the batch size is dynamic
            #     # in this case, we can set the batch size to 1 temporarily
            #     binding_shape[0] = 1

            size = trt.volume(self.allocated_shape)
            dtype = trt.nptype(engine.get_tensor_dtype(binding))
            # print(f"Using dtype {dtype} for {binding}")
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes) 

            bindings.append(int(cuda_mem))
            #print(engine.get_tensor_mode(binding))
            if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                context.set_input_shape(binding, self.allocated_shape)
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
                self.input_shapes.append(self.allocated_shape[1:])
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
                self.output_shapes.append(self.allocated_shape[1:])

                # print(f"[MEMALLOC] {binding}[{binding_idx}] {np.dtype(dtype)}: {size * np.dtype(dtype).itemsize / 1_000_000} MB")

            # print(f"[HOST+DEVICE] allocated input:{host_inputs[0].nbytes / 1_000_000} MB, output:{host_outputs[0].nbytes / 1_000_000} MB")
        
        self.stream = stream
        self.context = context 
        self.engine = engine

        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs

        self.bindings = bindings
    

    def _is_dynamic(self, engine:trt.ICudaEngine) -> Tuple[bool, int, int, int]:

        # If engine.get_tensor_shape(engine[x])[x] == -1, then the engine is dynamic
        # in this case the input shape has to be set before running inference
        # in the infer method, as well as pre-allocate the input and output buffers
        ret = False
        min_batch, opt_batch, max_batch = 0, 0, 0
        

        for binding in engine:
            if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                tensor_shape = engine.get_tensor_shape(binding)
                min_shape, opt_shape, max_shape = engine.get_profile_shape(0, binding)
                
                min_batch = min_shape[0]
                opt_batch = opt_shape[0]
                max_batch = max_shape[0]
                # if none of the dimensions have -1, then the engine is not dynamic,
                # min, opt, max will be the same
                if all([x != -1 for x in tensor_shape]):
                    # no dynamic shapes supported
                    break 
                if tensor_shape[0] == -1 and all([x != -1 for x in tensor_shape[1:]]):
                    ret = True
                    break
                else:
                    raise NotImplementedError(
                        "Engine has dynamic dimensions that are not the batch size. "\
                        "This is not supported, only dynamic batch size is. "\
                        "Please, set the dynamic dimensions to a fixed size and try again"
                    )
                
        return ret, min_batch, opt_batch, max_batch
    

    def infer(self, input_matrix : List[np.ndarray], batch_size: int, **kwargs) -> List[np.ndarray]:
        """
        Receives a pre-processed List of input 1D vectors, runs inference and returns the raw-outputs

        Args:
        - input_matrix (List[np.ndarray]): List of input matrices to run inference on. Each
                item in the list is a 1D contiguous numpy array, if runned in batch inference the 1D
                array must be concatenated to be a single 1D array. Each list item represents a different
                tensor input of the model. The order of the list must match the order of the input tensors
        - kwargs: Additional arguments to be passed to the inference engine

        Returns:
        - List[np.ndarray]: List of output matrices. The list contains the batch for each output of the model. 
            I.E. if a model has 2 outputs (output_0, and output_1) and has performed inference on a batch of 4 images, 
            the output will be a list of 2 numpy arrays where:
            - list[0] is a 1D array of the model's output_0 for the 4 images, 
            - list[1] is a 1D array of the output_1 for the 4 images.
        """

        # TODO: handle memory allocation for dynamic batch size

        self.cfx.push()

        stream = self.stream
        context = self.context

        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        
        # TODO: handle batch input, and different input shapes
        for i in range(min(batch_size, len(host_inputs))):
            # print(f"Copying input {input_matrix[0].shape} to device")
            
            if self.is_dynamic:
                if input_matrix[i].size > self.host_inputs[i].size:
                    raise ValueError(
                        f"Supplied size for input tensor {i} is larger than the maximum size supported by the engine. "\
                        f"Expected {self.host_inputs[i].size}, got {input_matrix[i].size}")
                elif input_matrix[i].size < self.host_inputs[i].size:
                    # allocate memory based on required batch size
                    # TODO: dynamically allocate batch based on required input size
                    # and update input shape in context
                    # It might happen that a previous batch size was used, and the new batch size is smaller
                    # in this case, we need to update the input shape in the context, reallocate memory in input and output buffers
                    self.context.set_input_shape(self.engine[i], (batch_size, *self.input_shapes[i]))
                    
            
            elif input_matrix[i].size != self.host_inputs[i].size:
                raise ValueError(
                    f"Engine does not support dynamic batches. Input tensor {i} has a different size than expected. "\
                    f"Expected {self.host_inputs[i].size}, got {input_matrix[i].size}"
                )

            np.copyto(host_inputs[i], input_matrix[i])
            cuda.memcpy_htod_async(cuda_inputs[i], host_inputs[i], stream)
        
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        
        for i in range(min(batch_size, len(host_outputs))):

            cuda.memcpy_dtoh_async(host_outputs[i], cuda_outputs[i], stream)

        stream.synchronize()
        self.cfx.pop()
        
        return host_outputs
    

    def __del__(self):
        try: self.cfx.pop()
        except: pass
        try: self.engine = [] # releases memory used by engine, otherwise will SegFault
        except: pass
        try: self.context.pop(); del self.context
        except: pass
        try: self.stream.destroy(); del self.stream
        except: pass
        self.cfx = None
        self.context = None
        self.engine = None
        self.stream = None
        self.host_inputs = None
        self.cuda_inputs = None
        self.host_outputs = None
        self.cuda_outputs = None
        self.bindings = None