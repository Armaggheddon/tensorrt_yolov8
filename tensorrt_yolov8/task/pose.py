import tensorrt as trt
import numpy as np
import cv2

from .utils import pose_keypoint_labels


def postprocess(outputs : np.ndarray, output_shape : np.ndarray, **kwargs) -> list['PoseResult']:
    
    min_prob = kwargs.get("min_prob", 0.4)
    nms_score = kwargs.get("nms_score", 0.25)

    m_outputs = np.reshape(outputs[0], output_shape[0])

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

    m_outputs[0, :] -= m_outputs[2, :] / 2
    m_outputs[1, :] -= m_outputs[3, :] / 2

    indexes = cv2.dnn.NMSBoxes(
        bboxes=m_outputs[:4, :].astype(int).T.tolist(),
        scores=scores.astype(float).tolist(),
        score_threshold=min_prob,
        nms_threshold=nms_score,
    )

    results = []

    for index in indexes:
        if isinstance(index, list): index = index[0]

        x, y, w, h = m_outputs[:4, index] / 640.0
        score, *keypoints = m_outputs[4:, index]

        results.append(PoseResult(
            class_id,
            score,
            x, y, w, h,
            keypoints
        ))

    return results


class KeyPoint():
    def __init__(
        self,
        x,
        y,
        v,
        label,    
    ):
        self.x = x / 640
        self.y = y / 640
        self.v = v
        self.label = label

class PoseResult():

    def __init__(
            self,
            class_id,
            confidence,
            x, 
            y,
            w, 
            h,
            keypoints,
    ):
        self.class_id = class_id
        self.confidence = confidence
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.x2 = self.x + self.w
        self.y2 = self.y + self.h

        self.keypoints = {}
        for i in range(0, len(pose_keypoint_labels)):
            x = keypoints[i * 3]
            y = keypoints[i * 3 + 1]
            v = keypoints[i * 3 + 2]
            self.keypoints[pose_keypoint_labels[i]] = KeyPoint(x, y, v, pose_keypoint_labels[i])
        