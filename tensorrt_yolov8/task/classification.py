import numpy as np

from .utils import classification_labels as labels


def postprocess(outputs : np.ndarray, output_shape : np.ndarray, **kwargs) -> list['ClassificationResult']:
    
    min_prob = kwargs.get("min_prob", 0.0)
    top_k = kwargs.get("top_k", 1000)

    m_outputs = np.reshape(outputs[0], output_shape[0])

    if m_outputs.shape[0] > 1:
        # output is batched
        pass

    # TODO: see detection.py
    assert m_outputs.shape[0] == 1

    m_outputs = m_outputs[0]

    class_ids = np.argpartition(m_outputs, -top_k)[-top_k:]
    class_ids = class_ids[np.argsort(m_outputs[class_ids])][::-1]
    class_ids = class_ids[m_outputs[class_ids] > min_prob]

    results = [ ClassificationResult(cls, m_outputs[cls]) for cls in class_ids]
    
    return results


class ClassificationResult():
    
    def __init__(
        self,
        class_id,
        probability,
    ):
        self.class_id = class_id
        self.class_label = labels[class_id]
        self.probability = probability