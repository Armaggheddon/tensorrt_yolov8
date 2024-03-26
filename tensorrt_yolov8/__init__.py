"""
tensorrt_yolov8 

A small wrapper library that allows to run YoloV8 
classification, detection, pose and segmentation 
models exported as TensorRT engine natively.
"""

import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import importlib

from tensorrt_yolov8.task import common


__version__ = "1.0"
__author__ = "Armaggheddon"


_model_types = [
    "classification",
    "detection",
    "pose",
    "segmentation",
    "obb", # TODO: implement
]

_logger_level = {
    "info": trt.Logger.INFO,
    "warning": trt.Logger.WARNING,
    "error": trt.Logger.ERROR
}

class TRTYoloV8():

    def available_models():
        return _model_types

    def __init__(
            self,
            model_type:str,
            engine_path:str,
            tensor_input=False,
            raw_output:bool=False,
            preprocess:callable=None,
            postprocess:callable=None,
            log_level:str="error"
    ) -> None:
        
        if model_type not in _model_types:
            raise ValueError(f"model_type must be one of {_model_types}")
        self.model_type = model_type

        if not os.path.exists(engine_path):
            raise ValueError(f"engine_path {engine_path} does not exist")
        
        if engine_path.endswith(".onnx"):
            raise ValueError(f"engine_path {engine_path} must be a .engine file. Use tensorrt_yolov8.utils.engine_builder to build an engine from an ONNX model.")
        if not engine_path.endswith(".engine"):
            raise ValueError(f"engine_path {engine_path} must be a .engine file. If you have an ONNX model, use tensorrt_yolov8.utils.engine_builder to build an engine from an ONNX model.")

        if log_level not in _logger_level.keys():
            raise ValueError(f"log_level must be one of {_logger_level.keys()}")

        self.engine_path = engine_path
        self.raw_output = raw_output

        if tensor_input:
            self._preprocess = None
        else:
            if preprocess is None:
                self._preprocess = common.preprocess #getattr(importlib.import_module("task", "common"), "preprocess")
            else:
                self._preprocess = preprocess

        if raw_output:
            self.postprocess = None
        else:
            if postprocess is None:
                self.postprocess = getattr(importlib.import_module(f"tensorrt_yolov8.task.{self.model_type}"), "postprocess")
            else:
                self.postprocess = postprocess

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
    

    def __call__(self, images : np.ndarray | list[np.ndarray], **kwargs) -> list:
        
        self.cfx.push()

        stream = self.stream
        context = self.context

        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        
        if self._preprocess is not None:
            if isinstance(images, np.ndarray):
                images = self._preprocess(images, self.input_shapes)
            elif isinstance(images, list) and isinstance(images[0], np.ndarray):
                images = self._preprocess(images, self.input_shapes)
            else:
                raise ValueError(f"Images was not of type numpy.ndarray or list of numpy.ndarray. Was {type(images)}")
        else:
            # check if given array length matches with model input size
            if isinstance(images, np.ndarray):
                if images.shape != len(host_inputs[0]):
                    raise ValueError(f"Input shape {images.shape} does not match model input shape {len(host_inputs[0])}")
                # if this point is reached input is, at least shape-wise correct
                # check if dtype is correct
                if images.dtype != host_inputs[0].dtype:
                    raise ValueError(f"Input dtype {images.dtype} does not match model input dtype {host_inputs[0].dtype}")
            else:
                raise ValueError(f"Images was not of type numpy.ndarray or list of numpy.ndarray. Was {type(images)}")

        np.copyto(host_inputs[0], images)
        
        # all yolo V8 models have a single input and a single output
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        for i in range(len(host_outputs)):
            cuda.memcpy_dtoh_async(host_outputs[i], cuda_outputs[i], stream)

        stream.synchronize()
        
        self.cfx.pop()
        # call model

        if self.postprocess is None:
            return host_outputs[0]
    
        results = self.postprocess(host_outputs, output_shape=self.output_shapes, **kwargs)
        
        return results
    

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

