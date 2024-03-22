import numpy as np
import cv2

from .utils import detection_labels as labels

"""
Segmentation output has 2 outputs:
- 0 -- (x, 32, 160, 160)
- 1 -- (x, 116, 8400)

See: https://github.com/ultralytics/ultralytics/issues/2953
"""

def postprocess(outputs : np.ndarray, output_shape : np.ndarray, **kwargs) -> list['SegmentationResult']:
    
    min_prob = kwargs.get("min_prob", 0.4)
    nms_score = kwargs.get("nms_score", 0.25)

    proposed_masks = outputs[0].reshape(output_shape[0])
    detections = outputs[1].reshape(output_shape[1])

    # they have to be the same size
    assert proposed_masks.shape[0] == detections.shape[0]

    if detections.shape[0] > 1:
        pass

    # TODO: add support for batch, see detection.py
    assert detections.shape[0] == 1, "Only batch size 1 is supported"
    assert proposed_masks.shape[0] == 1, "Only batch size 1 is supported"

    proposed_masks = proposed_masks[0]
    detections = detections[0]

    # print(f"Got {proposed_masks.shape=}, {detections.shape=}")

    # apply normal nms over detections
    detections = detections[:, np.amax(detections[4:80+4, :], axis=0) > min_prob]
    class_ids = np.argmax(detections[4:80+4, :], axis=0)

    scores = detections[4+class_ids, np.arange(detections.shape[-1])]

    detections[0, :] -= detections[2, :] / 2
    detections[1, :] -= detections[3, :] / 2

    indexes = cv2.dnn.NMSBoxes(
        bboxes=detections[:4, :].astype(int).T.tolist(), 
        scores=scores.astype(float).tolist(), 
        score_threshold=min_prob, 
        nms_threshold=nms_score
    )

    result = []

    for index in indexes:
        if isinstance(index, list): index = index[0]

        box = detections[:4, index] / 640.0
        mask_scores = detections[4+80:, index]
        class_id = class_ids.item(index)
        score = scores.item(index)

        # see https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
        # for explanation of einsum
        # mask scores has one dimension, i
        # proposed masks has 3 dimensions, i, j, k
        # we multiply each mask score by each mask, and sum them up in a matrix
        # the result is a 160x160 matrix
        obj_mask = np.einsum("i,ijk->jk", mask_scores, proposed_masks)
        # Equivalent to:
        # obj_mask = 0
        # for i in range(mask_scores.shape[0]):
        #     obj_mask += mask_scores[i] * proposed_masks[i]

        # apply sigmoid to mask output to have 0-1 range uniformly
        obj_mask = 1 / (1 + np.exp(-obj_mask))

        result.append(
            SegmentationResult(
                class_id=int(class_id), 
                confidence=float(score), 
                x=float(box[0]), 
                y=float(box[1]), 
                w=float(box[2]), 
                h=float(box[3]), 
                mask=obj_mask
        ))

    return result

class SegmentationResult():
    def __init__(
            self,
            class_id : int,
            confidence : float,
            x : float,
            y : float,
            w : float,
            h : float,
            mask : np.ndarray
    ):
        self.class_id = class_id
        self.class_label = labels[class_id]
        self.confidence = confidence
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.mask = mask

        self.x2 = self.x + self.w
        self.y2 = self.y + self.h