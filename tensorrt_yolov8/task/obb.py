import numpy as np
import cv2
import math

from .utils import obb_labels as labels

"""
OBB input is (X, 3, 1024, 1024)
OBB output is (X, 20, 21504) for each detection 20 represents:
- 0:4 X, Y, W, H
- 4:18 class probabilities
- 19 rotation in radians
"""

def postprocess(outputs : np.ndarray, output_shape : np.ndarray, **kwargs) -> list['ObbResult']:    
    
    min_prob = kwargs.get("min_prob", 0.4)
    nms_score = kwargs.get("nms_score", 0.4)

    m_outputs = np.reshape(outputs[0], output_shape[0])

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
    )

    results = []

    for index in indexes:
        i = index
        if isinstance(i, list): i = i[0]

        results.append(
            ObbResult(
                class_id=class_ids[i],
                confidence=scores[i],
                rotated_rect = cv2.RotatedRect(
                    np.array(rotated_boxes[i].center) / (1024.0, 1024.0), # normalize output to 0-1
                    np.array(rotated_boxes[i].size) / (1024.0, 1024.0),
                    rotated_boxes[i].angle
                )
            )
        )
    
    return results


class ObbResult:
    def __init__(
            self,
            class_id : int,
            confidence : float,
            rotated_rect : cv2.RotatedRect,
    ):
        self.class_id = class_id
        self.class_label = labels[class_id]
        self.confidence = confidence
        self.rotated_rect = rotated_rect