from typing import Any, Tuple, List, Union, Protocol
import numpy as np

from .types import InputMetadata, ModelResult

# class ModelBase:
#     """
#     Base Model class for all models. 
#     Different model subclass this implementing
#     the required methods. A NotImplementedError
#     is otherwise raised.
#     """
#     def infer(self, image=np.ndarray, **kwargs) -> Any:

#         preproc_img = self.preprocess(image, **kwargs)
#         predictions = self.predict(preproc_img, **kwargs)
#         postprocessed = self.postprocess(predictions, **kwargs)

#         return postprocessed

#     def preprocess(self, image=np.ndarray, **kwargs) -> Tuple[np.ndarray, PreprocessMetadata]:
#         raise NotImplementedError
    
#     def postprocess(self, output=np.ndarray, **kwargs) -> Any:
#         raise NotImplementedError
    
#     def predict(self, predictions=np.ndarray, **kwargs) -> Tuple[np.ndarray, ...]:
#         raise NotImplementedError


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