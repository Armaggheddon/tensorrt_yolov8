import numpy as np
import cv2
from typing import List, Tuple, Dict

from tensorrt_yolov8.core.models.base import ModelBase
from tensorrt_yolov8.core.models.types import ModelResult, PosePoint
from .common import yolo_preprocess
from .labels import POSE_KEYPOINT_LABELS


class Pose(ModelBase):

    model_type = "pose"

    def __init__(
            self,
            input_shapes: List[tuple[int, int, int]],
            output_shapes: List[tuple[int, int, int]],
    ):

        self.input_shape = input_shapes[0]
        self.output_shape = output_shapes[0]
    
    def preprocess(self, image : np.ndarray, **kwargs) -> np.ndarray:
        self.src_img_h, self.src_img_w = image.shape[:2]
        return yolo_preprocess(image, to_shape=self.input_shape, swap_rb=True)
    
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

        #class id is person for all kp
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
            # top_k=top_k
        )

        _take = min(top_k, len(indexes))
        if _take != len(indexes):
            indexes = indexes[:_take]

        results = []

        for index in indexes:
            if isinstance(index, list): index = index[0]

            x, y, w , h = bboxes[:, index]
            score, *kp = m_outputs[4:, index]

            # TODO find better handling for case of custom labels
            # for dictionary of keypoint labels keys
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
                            x=int(kp[i*3] * self.src_img_w / self.input_shape[2]),
                            y=int(kp[i*3+1] * self.src_img_h / self.input_shape[3]),
                            v=kp[i*3+2],
                            label=POSE_KEYPOINT_LABELS[i]
                        )
                        for i in range(0, len(POSE_KEYPOINT_LABELS))
                    ]
                )
            )

        return results
    
    # Can only work with the default YoloV8 POSE model
    # since in general the keypoint labels or what they represent
    # cannot be known
    def get_lines(self, kp: List[PosePoint], **kwargs) -> List[Tuple[int, int, int, int]]:
        
        """
        POSE_KEYPOINT_LABELS = [
            0 = "nose",
            1 = "left-eye",
            2 = "right-eye",
            3 = "left-ear",
            4 = "right-ear",

            5 = "left-shoulder",
            6 = "right-shoulder",
            7 = "left-elbow",
            8 = "right-elbow",
            9 = "left-wrist",
            10 = "right-wrist",
            11 = "left-hip",
            12 = "right-hip",            
            13 = "left-knee",
            14 = "right-knee",
            15 = "left-ankle",
            16 = "right-ankle",
        ]
        """

        lines = [
            # face
            (kp[4].x, kp[4].y, kp[2].x, kp[2].y),
            (kp[2].x, kp[2].y, kp[0].x, kp[0].y),
            (kp[0].x, kp[0].y, kp[1].x, kp[1].y),
            (kp[1].x, kp[1].y, kp[3].x, kp[3].y),
            # right side
            (kp[10].x, kp[10].y, kp[8].x, kp[8].y),
            (kp[8].x, kp[8].y, kp[6].x, kp[6].y),
            (kp[6].x, kp[6].y, kp[12].x, kp[12].y),
            (kp[12].x, kp[12].y, kp[14].x, kp[14].y),
            (kp[14].x, kp[14].y, kp[16].x, kp[16].y),
            # left side
            (kp[9].x, kp[9].y, kp[7].x, kp[7].y),
            (kp[7].x, kp[7].y, kp[5].x, kp[5].y),
            (kp[5].x, kp[5].y, kp[11].x, kp[11].y),
            (kp[11].x, kp[11].y, kp[13].x, kp[13].y),
            (kp[13].x, kp[13].y, kp[15].x, kp[15].y),

            # connect left and right hips and shoulders
            (kp[12].x, kp[12].y, kp[11].x, kp[11].y),
            (kp[6].x, kp[6].y, kp[5].x, kp[5].y),
        ]
        return lines
    
    def draw_results(self, image : np.ndarray, results : List[ModelResult], **kwargs) -> np.ndarray:
        
        img_overlay = image.copy()
        
        for result in results:
            x, y, w, h = result.x1, result.y1, result.w, result.h

            # Draw bbox
            cv2.rectangle(img_overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Draw label+confidence
            cv2.putText(
                img_overlay,
                f"{result.class_label} {result.confidence:.2f}",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Draw skeleton
            lines = self.get_lines(result.keypoints)
            for line in lines:
                cv2.line(img_overlay, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
            
            # Draw keypoints
            for point in result.keypoints:
                if point.v > 0:
                    cv2.circle(img_overlay, (point.x, point.y), 5, (0, 255, 0), -1)
            

        return img_overlay