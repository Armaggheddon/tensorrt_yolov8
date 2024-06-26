from typing import Dict, List
import numpy as np
from dataclasses import dataclass


@dataclass
class ImgSize:
    w: int
    h: int


class BoundingBox(dict):
    w: int
    h: int
    xc: int
    yc: int
    x1: int
    y1: int
    x2: int
    y2: int
    x3: int
    y3: int
    x4: int
    y4: int
    angle: float # angle in degrees

    def __getattr__(self, attr):
        return self.get(attr, None)

class PosePoint(dict):
    x: int
    y: int
    v: int
    label: str

    def __getattr__(self, attr):
        return self.get(attr, None)


class ModelResult(dict):
    model_type: str
    label_id: int
    label_name: str
    confidence: float
    box: BoundingBox
    keypoints: List[PosePoint]
    segmentation_mask: np.ndarray

    def __getattr__(self, attr):
        return self.get(attr, None)