import numpy as np
import cv2
import math

from .utils import obb_labels as labels

"""
OBB input is (X, 3, 1024, 1024)
OBB output is (X, 20, 21504) for each detection 20 represents:
- 0:4 X, Y, W, H
- 4:5 rotation
- 5:20 class probabilities
"""

# For x, w, w, h, angle to obb rectangle coordinates see:
# https://github.com/ultralytics/ultralytics/issues/9120#issuecomment-2008012496

def obb_to_corners(cx, cy, w, h, theta):
    print(f"cx: {cx}, cy: {cy}, w: {w}, h: {h}, theta: {theta}")
    corners = []
    # Define the four corners relative to the center (unrotated)
    offsets = [(w / 2, h / 2), (-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2)]
    
    for dx, dy in offsets:
        # Rotate each offset
        rx = dx * math.cos(theta) - dy * math.sin(theta)
        ry = dx * math.sin(theta) + dy * math.cos(theta)
        
        # Translate rotated offset to actual position
        x = cx + rx
        y = cy + ry
        
        corners.append((x,y))
  
    return corners


def postprocess(outputs : np.ndarray, output_shape : np.ndarray, **kwargs) -> list['ObbResult']:    
    
    min_prob = kwargs.get("min_prob", 0.4)
    nms_score = kwargs.get("nms_score", 0.4)

    m_outputs = np.reshape(outputs[0], output_shape[0])

    if m_outputs.shape[0] > 1:
        # output is batched
        pass

    assert m_outputs.shape[0] == 1, "Only batch size 1 is supported"

    m_outputs = m_outputs[0]

    m_outputs = m_outputs[:, np.amax(m_outputs[5:, :], axis=0) > min_prob]
    class_ids = np.argmax(m_outputs[5:, :], axis=0)
    scores = m_outputs[5+class_ids, np.arange(m_outputs.shape[-1])]

    # m_outputs[0, :] -= m_outputs[2, :] / 2
    # m_outputs[1, :] -= m_outputs[3, :] / 2

    # un-normalize degrees
    m_outputs[4, :] *= np.pi/2.0

    rotated_boxes = [
        cv2.RotatedRect(
            m_outputs[:2, i],
            m_outputs[2:4, i],
            m_outputs[4, i]
        )
        for i in range(m_outputs.shape[-1])
    ]
    # rotated_boxes = [
    #     obb_to_corners(
    #         m[0], m[1], 
    #         m[2], m[3], 
    #         m[4])
    #     for m in m_outputs
    # ]

    indexes = cv2.dnn.NMSBoxesRotated(
        bboxes=rotated_boxes,
        scores=scores.astype(float).tolist(),
        score_threshold=min_prob,
        nms_threshold=nms_score,
    )

    print(len(indexes))
    # print(f"Got {len(indexes)} detections")

    results = []

    for index in indexes:
        i = index
        if isinstance(i, list): i = i[0]

        box = m_outputs[:4, i]/1024.0
        results.append(
            ObbResult(
                class_id=class_ids[i],
                confidence=scores[i],
                x=box[0],
                y=box[1],
                w=box[2],
                h=box[3],
                rotated_rect = rotated_boxes[i]
            )
        )
    
    return results


class ObbResult:
    def __init__(
            self,
            class_id,
            confidence,
            x, y, w, h,
            rotated_rect,
    ):
        self.class_id = class_id
        self.class_label = labels[class_id]
        self.confidence = confidence
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.rotated_rect = rotated_rect