import numpy as np
import cv2
from typing import List

from tensorrt_yolov8.core.models.base import ModelBase
from tensorrt_yolov8.core.models.types import ModelResult
from .common import yolo_preprocess
from .labels import DETECTION_LABELS

"""
Segmentation output has 2 outputs:
- 0 -- (x, 32, 160, 160)
- 1 -- (x, 116, 8400)

See: https://github.com/ultralytics/ultralytics/issues/2953
"""

class Segmentation(ModelBase):

    model_type = "segmentation"

    def __init__(
            self,
            input_shapes: List[tuple[int, int, int]],
            output_shapes: List[tuple[int, int, int]],
    ):

        self.input_shape = input_shapes[0]
        self.output_shapes = output_shapes #2 outputs

    def preprocess(self, image : np.ndarray, **kwargs) -> np.ndarray:
        self.src_img_h, self.src_img_w = image.shape[:2]
        return yolo_preprocess(image, to_shape=self.input_shape, swap_rb=True)

    def postprocess(self, output: np.ndarray, min_prob: float, top_k: int, **kwargs) -> List[ModelResult]:

        nms_score = kwargs.get("nms_score", 0.25)

        proposed_masks = output[0].reshape(self.output_shapes[0])
        detections = output[1].reshape(self.output_shapes[1])

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

        detections[[0, 2], :] *= self.src_img_w / self.input_shape[2]
        detections[[1, 3], :] *= self.src_img_h / self.input_shape[3]
        detections[0, :] -= detections[2, :] / 2
        detections[1, :] -= detections[3, :] / 2

        bboxes = detections[:4, :].astype(int)

        # Returned indexes are sorted in descending
        # order of score (scores)
        indexes = cv2.dnn.NMSBoxes(
            bboxes=bboxes.T.tolist(), 
            scores=scores.astype(float).tolist(), 
            score_threshold=min_prob, 
            nms_threshold=nms_score
        )

        _take = min(top_k, len(indexes))
        if _take != len(indexes):
            indexes = indexes[:_take]

        result = []

        for index in indexes:
            if isinstance(index, list): index = index[0]

            box = bboxes[:, index]
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

                ModelResult(
                    model_type=Segmentation.model_type,
                    label_id=class_id,
                    label_name=DETECTION_LABELS[class_id],
                    confidence=score,
                    x=box[0],
                    y=box[1],
                    w=box[2],
                    h=box[3],
                    segmentation_mask=obj_mask
                )
            )

        return result

    def draw_results(self, image: np.ndarray, results: List[ModelResult], **kwargs) -> np.ndarray:

        img_overlay = image.copy()

        for res in results:
            if res.model_type != Segmentation.model_type: continue

            # draw bbox
            cv2.rectangle(
                img_overlay,
                (res.x, res.y),
                (res.x + res.w, res.y + res.h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                img_overlay,
                f"{res.label_name} {res.confidence:.2f}",
                (res.x, res.y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # get the mask specific for the object
            scale_x, scale_y = self.output_shapes[0][2]/self.src_img_w, self.output_shapes[0][3]/self.src_img_h
            n_x1, n_y1, n_w, n_h = res.x*scale_x, res.y*scale_y, res.w*scale_x, res.h*scale_y

            bbox_mask = res.segmentation_mask[
                int(n_y1):int(n_y1+n_h),
                int(n_x1):int(n_x1+n_w)
            ]

            # resize mask to original bbox size
            bbox_mask = cv2.resize(bbox_mask, (res.w, res.h), interpolation=cv2.INTER_LINEAR) # INTER_NEAREST, is faster but produces jagged edges in mask overlay
            bbox_mask = (bbox_mask > 0.5).astype(np.uint8)

            colored_mask = np.moveaxis(
                np.expand_dims(bbox_mask, axis=0).repeat(3, axis=0), 0, -1
            )

            masked = np.ma.MaskedArray(
                img_overlay[res.y:res.y+res.h, res.x:res.x+res.w],
                mask=colored_mask,
                fill_value=(0, 255, 0)).filled()
            
            img_overlay[res.y:res.y+res.h, res.x:res.x+res.w] = cv2.addWeighted(
                img_overlay[res.y:res.y+res.h, res.x:res.x+res.w], 0.7,
                masked, 0.3, 0
            )

        return img_overlay