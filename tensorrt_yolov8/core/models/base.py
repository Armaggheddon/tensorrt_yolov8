from typing import Any, Tuple, List, Union, Protocol
import numpy as np

from .types import ModelResult


class ModelBase(Protocol):

    input_shapes: List[Tuple[int, int, int]]
    output_shapes: List[Tuple[int, int, int]]
    
    def preprocess(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Protocol hint"""
        ...
    
    def postprocess(self, output: np.ndarray, min_prob: float, top_k: int, **kwargs) -> Any:
        """Protocol hint"""
        ...

    def draw_results(self, image: np.ndarray, results: List[ModelResult], **kwargs) -> np.ndarray:
        """Protocol hint"""
        ...