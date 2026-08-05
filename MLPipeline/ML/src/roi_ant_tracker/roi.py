from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True, frozen=True)
class RoiRect:
    roi_id: str
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y : self.y + self.height, self.x : self.x + self.width]

    def contains(self, point: tuple[float, float]) -> bool:
        x, y = point
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height

    def padded_bounds(self, padding: int, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        return (
            max(0, self.x - padding),
            max(0, self.y - padding),
            min(frame_width, self.x + self.width + padding),
            min(frame_height, self.y + self.height + padding),
        )


def middle_frame_index(video_path: str | Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if total <= 0:
        return 0
    return max(0, total // 2)


def read_frame(video_path: str | Path, frame_index: int | str = "middle") -> tuple[np.ndarray, int, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        resolved = middle_frame_index(video_path) if frame_index == "middle" else int(frame_index)
        resolved = min(max(0, resolved), max(0, total - 1))
        capture.set(cv2.CAP_PROP_POS_FRAMES, resolved)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {resolved} from video: {video_path}")
    return frame, resolved, fps


def load_rois(path: str | Path | None, frame_width: int, frame_height: int) -> list[RoiRect]:
    if path is None:
        return [RoiRect("full_frame", 0, 0, frame_width, frame_height)]
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    rectangles = payload.get("rectangles")
    if rectangles is None and {"crop_x", "crop_y", "crop_width", "crop_height"}.issubset(payload):
        rectangles = [
            {
                "id": "roi_001",
                "x": payload["crop_x"],
                "y": payload["crop_y"],
                "width": payload["crop_width"],
                "height": payload["crop_height"],
            }
        ]
    if not rectangles:
        raise ValueError(f"ROI JSON has no rectangles: {path}")
    rois = [
        RoiRect(
            roi_id=str(item.get("id") or f"roi_{index:03d}"),
            x=int(item["x"]),
            y=int(item["y"]),
            width=int(item["width"]),
            height=int(item["height"]),
        )
        for index, item in enumerate(rectangles, start=1)
    ]
    validate_rois(rois, frame_width, frame_height)
    return rois


def validate_rois(rois: list[RoiRect], frame_width: int, frame_height: int) -> None:
    seen: set[str] = set()
    for roi in rois:
        if roi.roi_id in seen:
            raise ValueError(f"Duplicate ROI id: {roi.roi_id}")
        seen.add(roi.roi_id)
        if roi.width <= 0 or roi.height <= 0:
            raise ValueError(f"ROI {roi.roi_id} width and height must be positive.")
        if roi.x < 0 or roi.y < 0:
            raise ValueError(f"ROI {roi.roi_id} x and y must be non-negative.")
        if roi.x + roi.width > frame_width or roi.y + roi.height > frame_height:
            raise ValueError(f"ROI {roi.roi_id} extends beyond the frame.")


def roi_payload(
    video_path: str | Path,
    frame_index: int,
    frame_width: int,
    frame_height: int,
    rectangles: list[RoiRect],
    representative_frame: str | None = None,
    preview_image: str | None = None,
) -> dict[str, Any]:
    return {
        "roi_version": 1,
        "source_video": str(Path(video_path).resolve()),
        "representative_frame_index": frame_index,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "representative_frame": representative_frame,
        "preview_image": preview_image,
        "rectangles": [
            {
                "id": roi.roi_id,
                "x": roi.x,
                "y": roi.y,
                "width": roi.width,
                "height": roi.height,
            }
            for roi in rectangles
        ],
    }


def draw_rois(frame: np.ndarray, rois: list[RoiRect]) -> np.ndarray:
    output = frame.copy()
    for roi in rois:
        cv2.rectangle(output, (roi.x, roi.y), (roi.x + roi.width, roi.y + roi.height), (0, 255, 255), 2)
        cv2.putText(
            output,
            roi.roi_id,
            (roi.x, max(18, roi.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
    return output
