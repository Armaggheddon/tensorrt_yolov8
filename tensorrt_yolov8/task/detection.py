import numpy as np
import cv2

from .utils import detection_labels as labels


def postprocess(outputs : np.ndarray, output_shape : np.ndarray, **kwargs) -> list['DetectionResult']:    
    
    min_prob = kwargs.get("min_prob", 0.4)
    nms_score = kwargs.get("nms_score", 0.25)

    m_outputs = np.reshape(outputs[0], output_shape[0])

    if m_outputs.shape[0] > 1:
        # output is batched
        pass
    
    # TODO: handle case where output is batched or handle it dynamically
    # tradeoff between numpy complexity and ease of implementation can be
    # made since batched is not common and batch size is usually small
    assert m_outputs.shape[0] == 1 # if not 1 crashes RN

    m_outputs = m_outputs[0]
    
    # pre filter based on minimum probability, nms method will not 
    # discard any entry based on probability but only on nms score
    m_outputs = m_outputs[:, np.amax(m_outputs[4:, :], axis=0) > min_prob]
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
    m_outputs[0, :] -= m_outputs[2, :] / 2
    m_outputs[1, :] -= m_outputs[3, :] / 2

    indexes = cv2.dnn.NMSBoxes(
        bboxes=m_outputs[:4, :].astype(int).T.tolist(),
        scores=scores.astype(float).tolist(),
        score_threshold=min_prob,
        nms_threshold=nms_score,
        # top_k=300
    )     
    
    results = []

    for index in indexes:
        i = index
        if isinstance(i, list): i = i[0]

        box = m_outputs[:4, i] / 640.0
        results.append(
            DetectionResult(
                class_id=int(class_ids.item(i)),
                confidence=float(scores.item(i)),
                x=float(box[0]),
                y=float(box[1]),
                w=float(box[2]),
                h=float(box[3]),
            )
        )

    return results
    

class DetectionResult():
    def __init__(
            self,
            class_id : int,
            confidence : float,
            x : float,
            y : float,
            w : float,
            h : float,
    ):
        self.class_id = class_id
        self.class_label = labels[class_id]
        self.confidence = confidence
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.x2 = self.x + self.w
        self.y2 = self.y + self.h