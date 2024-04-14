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

    def preprocess(self, images: List[np.ndarray], **kwargs) -> List[np.ndarray]:
        return [yolo_preprocess(images, to_shape=self.input_shape, swap_rb=True)]

    def __postprocess_batch(self, output: np.ndarray, batch_id: int, min_prob: float, top_k: int, **kwargs) -> List[ModelResult]:
        
        class_ids = np.argpartition(output, -top_k)[-top_k:]
        class_ids = class_ids[np.argsort(output[class_ids])][::-1]
        class_ids = class_ids[output[class_ids] > min_prob]

        results = [ 
            ModelResult(
                model_type=Classification.model_type, 
                label_id=idx,
                label_name=CLASSIFICATION_LABELS[idx],
                confidence=output[idx]
            ) for idx in class_ids
        ]
        
        return results

    def postprocess(self, output: List[np.ndarray], min_prob: float, top_k: int, **kwargs) -> List[List[ModelResult]]:

        m_outputs = np.reshape(output[0], self.output_shape)

        results = [
            self.__postprocess_batch(
                m_outputs[batch_id, :], 
                batch_id, 
                min_prob, 
                top_k, 
                **kwargs
            )
            for batch_id in range(m_outputs.shape[0])
        ]

        return results
    

    def __draw_result(self, image: np.ndarray, result: List[ModelResult], **kwargs) -> np.ndarray:

        img_overlay = image.copy()
        for res in result:
            if res.model_type != Classification.model_type: continue

            cv2.putText(
                img_overlay, 
                f"{res.label_name} @ {res.confidence*100:.1f}%", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2
            )

        return img_overlay


    def draw_results(self, images: List[np.ndarray], results: List[List[ModelResult]], **kwargs) -> List[np.ndarray]:
        
        imgs_overlay = [
            self.__draw_result(images[batch_id], batch, **kwargs)
            for batch_id, batch in enumerate(results)
        ]
        return imgs_overlay
