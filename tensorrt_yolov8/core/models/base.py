from typing import Any, Tuple, List, Union, Protocol
import numpy as np

from .types import ModelResult


class ModelBase(Protocol):

    input_shapes: List[Tuple[int, int, int]]
    output_shapes: List[Tuple[int, int, int]]
    
    def preprocess(self, images: List[np.ndarray], **kwargs) -> List[np.ndarray]:
        """
        Handles preprocessing of the input images based on the model requirements.

        Arguments:
        - images -- list of images to preprocess
        - kwargs -- additional arguments to pass to the preprocessing function

        Returns:
        - preprocessed_images -- A list where each item is the preprocessed image/s as a 1D contiguous array.
            If the model has multiple inputs, then each item represents that input tensor that will be fed to the model.
        """
        ...
    
    def postprocess(self, output: List[np.ndarray], min_prob: float, top_k: int, **kwargs) -> List[List[ModelResult]]:
        """
        Handles postprocessing of the model output. 

        Arguments:
        - output -- The list of the model outputs where each item is a 1D contiguous array
        - min_prob -- the minimum confidence threshold to consider a detection
        - top_k -- the maximum number of detections to keep
        - kwargs -- additional arguments to pass to the post

        Returns:
        - results -- A list where each item is a list of ModelResult objects represents the 
            detections for each batch, e.g. results[0] are the detections for the first image given.
        """
        ...

    def draw_results(self, image: List[np.ndarray], results: List[List[ModelResult]], **kwargs) -> List[np.ndarray]:
        """
        Utility function that draws the results on the given images based on the results obtained.

        Arguments:
        - images -- list of images to draw the results on, if is a single image, then it should be a list with a single item
        - results -- list of lists of ModelResult objects representing the detections for each image/images (in case of batched input).
            It is the one that the user gets from the postprocess method.
        - kwargs -- additional arguments to pass to the drawing function

        Returns:
        - images -- A list of images with the results drawn on them. The original images are not modified.
        """
        ...