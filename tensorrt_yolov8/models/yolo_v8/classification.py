from typing import List, Tuple
import numpy as np
import cv2

from tensorrt_yolov8.core.models.base import ModelBase
from tensorrt_yolov8.core.models.types import ModelResult
from .common import yolo_preprocess
from .labels import CLASSIFICATION_LABELS


class Classification(ModelBase):

    model_type = "classification"

    def __init__(
            self, 
            input_shapes: List[Tuple[int, int, int]], 
            output_shapes: List[Tuple[int, int, int]],
            **kwargs
    ) -> None:

        # YoloV8 classification has only 1 input and 1 output
        # therefore we can safely assume that the first element
        # of the input_shapes and output_shapes list is the one
        # we are interested in
        self.input_shape = input_shapes[0]
        self.output_shape = output_shapes[0]

    def preprocess(self, image : np.ndarray, **kwargs) -> np.ndarray:
        return yolo_preprocess(image, to_shape=self.input_shape, swap_rb=True)


    def postprocess(self, output: np.ndarray, min_prob: float, top_k: int, **kwargs) -> List[ModelResult]:

        m_outputs = np.reshape(output[0], self.output_shape)

        if m_outputs.shape[0] > 1:
            # output is batched
            pass

        # TODO: see detection.py
        assert m_outputs.shape[0] == 1

        m_outputs = m_outputs[0]

        class_ids = np.argpartition(m_outputs, -top_k)[-top_k:]
        class_ids = class_ids[np.argsort(m_outputs[class_ids])][::-1]
        class_ids = class_ids[m_outputs[class_ids] > min_prob]

        results = [ 
            ModelResult(
                model_type=Classification.model_type, 
                label_id=idx,
                label_name=CLASSIFICATION_LABELS[idx],
                confidence=m_outputs[idx]
            ) for idx in class_ids
        ]
        
        return results
    
    def draw_results(self, image : np.ndarray, results : List[ModelResult], **kwargs) -> np.ndarray:
        
        for res in results:
            label = f"{res.label_name} @ {res.confidence*100:.1f}%"
            cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return image
