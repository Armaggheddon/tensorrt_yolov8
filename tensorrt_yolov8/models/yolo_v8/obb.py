import numpy as np
import cv2
from typing import List

from core.models.base import ModelBase
from core.models.types import ModelResult
from .common import yolo_preprocess
from .labels import OBB_LABELS

"""
OBB input is (X, 3, 1024, 1024)
OBB output is (X, 20, 21504) for each detection 20 represents:
- 0:4 X, Y, W, H
- 4:18 class probabilities
- 19 rotation in radians
"""

class Obb(ModelBase):

    model_type = "obb"

    def __init__(
            self,
            input_shapes: list[tuple[int, int, int]],
            output_shapes: list[tuple[int, int, int]],
    ):
        self.labels = OBB_LABELS

        self.input_shape = input_shapes[0]
        self.output_shape = output_shapes[0]

    def preprocess(self, image : np.ndarray, **kwargs) -> np.ndarray:
        self.src_img_h, self.src_img_w = image.shape[:2]
        return yolo_preprocess(image, self.input_shape, True)
    
    def postprocess(self, output: np.ndarray, min_prob: float, top_k: int, **kwargs) -> List[ModelResult]:
        
        nms_score = kwargs.get("nms_score", 0.4)

        m_outputs = np.reshape(output[0], self.output_shape)

        if m_outputs.shape[0] > 1:
            # output is batched
            pass

        assert m_outputs.shape[0] == 1, "Only batch size 1 is supported"

        m_outputs = m_outputs[0]

        m_outputs = m_outputs[:, np.amax(m_outputs[4:-1, :], axis=0) > min_prob]
        class_ids = np.argmax(m_outputs[4:-1, :], axis=0)
        scores = m_outputs[4+class_ids, np.arange(m_outputs.shape[-1])]

        # convert angle in degrees
        m_outputs[-1, :] *= 180.0/np.pi

        # Scale bboxes to target size
        m_outputs[[0, 2], :] *= self.src_img_w/self.input_shape[2]
        m_outputs[[1, 3], :] *= self.src_img_h/self.input_shape[3]

        rotated_boxes = [
            cv2.RotatedRect(
                m_outputs[:2, i],
                m_outputs[2:4, i],
                m_outputs[-1, i]
            )
            for i in range(m_outputs.shape[-1])
        ]

        indexes = cv2.dnn.NMSBoxesRotated(
            bboxes=rotated_boxes,
            scores=scores.astype(float).tolist(),
            score_threshold=min_prob,
            nms_threshold=nms_score,
            top_k=top_k
        )

        results = []

        for index in indexes:
            i = index
            if isinstance(i, list): i = i[0]

            r_box = rotated_boxes[i]
            angle = r_box.angle
            points = r_box.points()
            width, height = r_box.size
            center_x, center_y = r_box.center

            results.append(
                ModelResult(
                    model_type=Obb.model_type,
                    class_id=class_ids[i],
                    class_label=OBB_LABELS[class_ids[i]],
                    confidence=scores[i],
                    x1=int(points[0][0]),
                    y1=int(points[0][1]),
                    x2=int(points[1][0]),
                    y2=int(points[1][1]),
                    x3=int(points[2][0]),
                    y3=int(points[2][1]),
                    x4=int(points[3][0]),
                    y4=int(points[3][1]),
                    xc=center_x,
                    yc=center_y,
                    w=width,
                    h=height,
                    angle=angle
                )
            )
        
        return results