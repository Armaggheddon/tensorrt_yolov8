import numpy as np
import cv2
from typing import List

from core.models.base import ModelBase
from core.models.types import ModelResult, PosePoint
from .common import yolo_preprocess
from .labels import POSE_KEYPOINT_LABELS


class Pose(ModelBase):

    model_type = "pose"

    def __init__(
            self,
            input_shapes: list[tuple[int, int, int]],
            output_shapes: list[tuple[int, int, int]],
    ):
        self.labels = POSE_KEYPOINT_LABELS

        self.input_shape = input_shapes[0]
        self.output_shape = output_shapes[0]
    
    def preprocess(self, image : np.ndarray, **kwargs) -> np.ndarray:
        self.src_img_h, self.src_img_w = image.shape[:2]
        return yolo_preprocess(image, self.input_shape, True)
    
    def postprocess(self, output: np.ndarray, min_prob: float, top_k: int, **kwargs) -> List[ModelResult]:
        nms_score = kwargs.get("nms_score", 0.25)

        m_outputs = np.reshape(output[0], self.output_shape)

        if m_outputs.shape[0] > 1:
            # output is batched
            pass

        # TODO, see detection.py
        assert m_outputs.shape[0] == 1

        m_outputs = m_outputs[0]

        m_outputs = m_outputs[:, m_outputs[4, :] > min_prob]

        #class id is person for all keypoints
        class_id = 0
        scores = m_outputs[4, :]

        m_outputs[[0, 2], :] *= self.src_img_w / self.input_shape[2]
        m_outputs[[1, 3], :] *= self.src_img_h / self.input_shape[3]
        m_outputs[0, :] -= m_outputs[2, :] / 2
        m_outputs[1, :] -= m_outputs[3, :] / 2

        bboxes = m_outputs[:4, :].astype(int)

        indexes = cv2.dnn.NMSBoxes(
            bboxes=bboxes.T.tolist(),
            scores=scores.astype(float).tolist(),
            score_threshold=min_prob,
            nms_threshold=nms_score,
            top_k=top_k
        )

        results = []

        for index in indexes:
            if isinstance(index, list): index = index[0]

            x, y, w, h = bboxes[:, index]
            score, *keypoints = m_outputs[4:, index]

            results.append(
                ModelResult(
                    model_type=Pose.model_type,
                    class_id=class_id,
                    class_label="person",  # always person for pose model
                    confidence=score,
                    x1=x,
                    y1=y,
                    w=w,
                    h=h,
                    keypoints=[
                        PosePoint(
                            x=int(keypoints[i*3] * self.src_img_w / self.input_shape[2]),
                            y=int(keypoints[i*3+1] * self.src_img_h / self.input_shape[3]),
                            v=keypoints[i*3+2],
                            label=POSE_KEYPOINT_LABELS[i]
                        )
                        for i in range(0, len(POSE_KEYPOINT_LABELS))
                    ]
                )
            )

        return results