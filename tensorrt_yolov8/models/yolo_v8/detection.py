import numpy as np
import cv2
from typing import List

from tensorrt_yolov8.core.models.base import ModelBase
from tensorrt_yolov8.core.models.types import ModelResult, ImgSize
from .common import yolo_preprocess
from .labels import DETECTION_LABELS


class Detection(ModelBase):

    model_type = "detection"

    def __init__(
            self,
            input_shapes: List[tuple[int, int, int]],
            output_shapes: List[tuple[int, int, int]],
            **kwargs
    ):
        self.input_shape = input_shapes[0] # has format (b, 3, 640, 640) if model is base
        self.output_shape = output_shapes[0] # has format (b, 84, 8400) if model is base
    
    def preprocess(self, images: List[np.ndarray], **kwargs) -> List[np.ndarray]:

        # TODO: check number of images, if too large throw error, if smaller, pad with zeros and save given number
        # so that in post processing only the batches that match real images are processed
        self.src_imgs_shape = [ ImgSize(h=img.shape[0], w=img.shape[1]) for img in images]
        return [yolo_preprocess(images, to_shape=self.input_shape, swap_rb=True)]
    

    def __postprocess_batch(self, output: np.ndarray, batch_id: int, min_prob: float, top_k: int, **kwargs) -> List[ModelResult]:
        
        nms_score = kwargs.get("nms_score", 0.4)

        # pre filter based on minimum probability, nms method will not 
        # discard any entry based on probability but only on nms score
        m_outputs = output[:, np.amax(output[4:, :], axis=0) > min_prob]      
        class_ids = np.argmax(m_outputs[4:, :], axis=0)

        # Use arange on second index instead of : to get "advanced indexing". 
        # if ":" is used, it will use 4+class_ids for each row instead of using the 
        # first item for the first row, second item for the second row and so on 
        # (which is the desired behavior).
        # This implementation is faster than using np.amax(m_outputs[4:, :], axis=0)
        # since we already have the argmax of the scores => 15% faster on 140 items
        # performance gains increase as the number of items increases ;)
        scores = m_outputs[4+class_ids, np.arange(m_outputs.shape[-1])] # range to number of boxes

        # move boxes coordinates to top left of image so that
        # 0:4 is like: [x1, y1, w, h]
        # Scale bboxes to target image size
        m_outputs[[0, 2], :] *= self.src_imgs_shape[batch_id].w/self.input_shape[2]
        m_outputs[[1, 3], :] *= self.src_imgs_shape[batch_id].h/self.input_shape[3]
        m_outputs[0, :] -= m_outputs[2, :] / 2
        m_outputs[1, :] -= m_outputs[3, :] / 2
        bboxes = m_outputs[:4, :].astype(int)

        # Returned indexes are sorted in descending 
        # order of confidence score (scores)
        indexes = cv2.dnn.NMSBoxes(
            bboxes=bboxes.T.tolist(),
            scores=scores.astype(float).tolist(),
            score_threshold=min_prob,
            nms_threshold=nms_score,
            # top_k=top_k # not the top_k that we want
        )     
        
        # filter out top_k
        _take = min(top_k, len(indexes))
        if _take != len(indexes):
            indexes = indexes[:_take]

        results = []

        for index in indexes:
            i = index
            if isinstance(i, list): i = i[0]

            box = bboxes[:, i]
            results.append(
                # TODO: handle case for custom yolo model where label might be more than 80
                # e.g. use label_name="" if label_id > 80
                ModelResult(
                    model_type=Detection.model_type,
                    label_id=class_ids.item(i),
                    label_name=DETECTION_LABELS[class_ids.item(i)],
                    confidence=float(scores.item(i)),
                    x1=box[0],
                    y1=box[1],
                    x2=box[0]+box[2],
                    y2=box[1]+box[3],
                    w=box[2],
                    h=box[3],
                )
            )

        return results


    def postprocess(self, output: List[np.ndarray], min_prob: float, top_k: int, **kwargs) -> List[List[ModelResult]]:

        m_outputs = np.reshape(output[0], self.output_shape)

        results = [
                self.__postprocess_batch(
                    m_outputs[batch_id, :, :], 
                    batch_id, 
                    min_prob, 
                    top_k, 
                    **kwargs
                )
                for batch_id in range(len(m_outputs))
        ]

        return results
    

    def __draw_result(self, image: np.ndarray, result: List[ModelResult], **kwargs) -> np.ndarray:
        
        img_overlay = image.copy()
        for res in result:
            if res.model_type != Detection.model_type: continue

            cv2.rectangle(
                img_overlay,
                (res.x1, res.y1),
                (res.x2, res.y2),
                (0, 255, 0),
                2
            )
            cv2.putText(
                img_overlay,
                f"{res.label_name} {res.confidence:.2f}",
                (res.x1, res.y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        
        return img_overlay


    def draw_results(self, images: List[np.ndarray], results: List[List[ModelResult]], **kwargs) -> List[np.ndarray]:
        
        imgs_overlay = [
            self.__draw_result(images[batch_id], batch, **kwargs)
            for batch_id, batch in enumerate(results)
        ]
        return imgs_overlay