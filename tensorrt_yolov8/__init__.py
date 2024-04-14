"""
tensorrt_yolov8 

A small wrapper library that allows to run YoloV8 
classification, detection, pose and segmentation 
models exported as TensorRT engine natively.
"""

import os
import numpy as np
import importlib
from typing import List, Union

from tensorrt_yolov8.engine_utils.engine_helper import EngineHelper
from tensorrt_yolov8.core.models.types import ModelResult
from tensorrt_yolov8.core.models.base import ModelBase


__version__ = "1.0"
__author__ = "Armaggheddon"


_model_types = [
    "classification",
    "detection",
    "pose",
    "segmentation",
    "obb",
]


class Pipeline():

    def get_model_types():
        return _model_types

    def __init__(
            self,
            model_type: str,
            model_path: str,
            warmup: bool = False,
            custom_model: ModelBase = None
    ):
        
        self.__engine = EngineHelper(model_path)

        self.__model: ModelBase = getattr(
            importlib.import_module(f"tensorrt_yolov8.models.yolo_v8.{model_type}"), model_type.capitalize()
        )(
            input_shapes=self.__engine.input_shapes, 
            output_shapes=self.__engine.output_shapes
        )

    def __call__(
            self,
            images: Union[np.ndarray, List[np.ndarray]],
            min_prob: float = 0.5,
            top_k: int = 300,
            **kwargs
    ) -> Union[List[ModelResult], List[List[ModelResult]]]:
        
        is_list = isinstance(images, list)
        if not is_list:
            images = [images]

        preprocess: List[np.ndarray] = self.__model.preprocess(images)
        predictions = self.__engine.infer(preprocess)
        postprocessed = self.__model.postprocess(predictions, min_prob, top_k, **kwargs)

        return postprocessed if is_list else postprocessed[0]
    
    def draw_results(
            self,
            images: Union[np.ndarray, List[np.ndarray]],
            results: Union[List[ModelResult], List[List[ModelResult]]],
    ) -> Union[np.ndarray, List[np.ndarray]]:

        is_list = isinstance(images, list)
        if is_list:
            assert len(images) == len(results), "The number of images and results must be the same"
        else:
            images = [images]
            results = [results]
        
        results = self.__model.draw_results(images, results)
        
        return results if is_list else results[0]
