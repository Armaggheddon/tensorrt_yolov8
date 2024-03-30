"""
tensorrt_yolov8 

A small wrapper library that allows to run YoloV8 
classification, detection, pose and segmentation 
models exported as TensorRT engine natively.
"""

import os
import numpy as np
import importlib
from typing import List

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
            image: np.ndarray,
            min_prob: float = 0.5,
            top_k: int = 300,
            **kwargs
    ) -> List[ModelResult]:
        
        preprocess = self.__model.preprocess(image)
        predictions = self.__engine.infer(preprocess)
        postprocessed = self.__model.postprocess(predictions, min_prob, top_k, **kwargs)

        return postprocessed
    
    def draw_results(
            self,
            image: np.ndarray,
            results: List[ModelResult],
    ) -> np.ndarray:
        return self.__model.draw_results(image, results)
