from typing import List

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

_logger_level = {
    "info": trt.Logger.INFO,
    "warning": trt.Logger.WARNING,
    "error": trt.Logger.ERROR
}

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
        context.set_input_shape(engine[0], engine.get_tensor_shape(engine[0]))

        host_inputs  = []
        cuda_inputs  = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        self.input_shapes = []
        self.output_shapes = []

        for binding in engine:
            binding_idx = engine[binding]
            binding_shape = context.get_tensor_shape(binding) # tuple with (1, 3, 640, 640) for example
            size = trt.volume(binding_shape)
            dtype = trt.nptype(engine.get_tensor_dtype(binding))
            # print(f"Using dtype {dtype} for {binding}")
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes) 

            bindings.append(int(cuda_mem))
            #print(engine.get_tensor_mode(binding))
            if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
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
    

    def infer(self, input_matrix : np.ndarray, **kwargs) -> List[np.ndarray]:
        
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