from typing import List, Tuple
from enum import Enum

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
    BATCH = 1
    SIZE = 2
    ALL = 3

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

        self.is_dynamic, self.dynamic_type = self._is_dynamic(engine)

        for binding in engine:
            binding_idx = engine[binding]
            binding_shape = context.get_tensor_shape(binding) # tuple with (1, 3, 640, 640) for example
            # if binding_shape[0] == -1:
            #     # This can happen if the engine is dynamic, but only the batch size is dynamic
            #     # in this case, we can set the batch size to 1 temporarily
            #     binding_shape[0] = 1

            size = trt.volume(binding_shape)
            dtype = trt.nptype(engine.get_tensor_dtype(binding))
            # print(f"Using dtype {dtype} for {binding}")
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes) 

            bindings.append(int(cuda_mem))
            #print(engine.get_tensor_mode(binding))
            if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                context.set_input_shape(binding, engine.get_tensor_shape(binding))
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
                self.input_shapes.append(binding_shape)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
                self.output_shapes.append(binding_shape)

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
    

    def _is_dynamic(self, engine:trt.ICudaEngine) -> Tuple[bool, DynamicType]:

        # If engine.get_tensor_shape(engine[x])[x] == -1, then the engine is dynamic
        # in this case the input shape has to be set before running inference
        # in the infer method, as well as pre-allocate the input and output buffers
        ret = False
        dynamic_type = DynamicType.NONE
        for binding in engine:
            if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                tensor_shape = engine.get_tensor_shape(binding)
                for dim in tensor_shape:
                    if dim == -1:
                        if dynamic_type == DynamicType.NONE:
                            dynamic_type = DynamicType.SIZE
                        else:
                            dynamic_type = DynamicType.ALL
                        ret = True
                if ret: break

        return ret, dynamic_type
    

    def infer(self, input_matrix : List[np.ndarray], **kwargs) -> List[np.ndarray]:
        
        self.cfx.push()

        stream = self.stream
        context = self.context

        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        
        # TODO: handle batch input, and different input shapes
        for i in range(len(host_inputs)):
            np.copyto(host_inputs[i], input_matrix)
            cuda.memcpy_htod_async(cuda_inputs[i], host_inputs[i], stream)
        
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        
        for i in range(len(host_outputs)):
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