from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


Box = tuple[float, float, float, float]


@dataclass(slots=True)
class Detection:
    bbox: Box
    score: float
    appearance: np.ndarray | None = None
    roi_id: str = ""

    @property
    def center(self) -> tuple[float, float]:
        return box_center(self.bbox)


def box_center(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_iou(left: Box, right: Box) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def translate_box(box: Box, dx: float, dy: float) -> Box:
    x1, y1, x2, y2 = box
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def clip_box(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )


def appearance_histogram(frame: np.ndarray, box: Box, bins: int = 12) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clip_box(box, width, height)
    crop = frame[int(y1) : int(np.ceil(y2)), int(x1) : int(np.ceil(x2))]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten().astype(np.float32)
    return histogram


def appearance_distance(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.5
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.5
    cosine_similarity = float(np.dot(left, right) / denominator)
    return float(np.clip(1.0 - cosine_similarity, 0.0, 2.0) / 2.0)


def distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return float(np.hypot(left[0] - right[0], left[1] - right[1]))

