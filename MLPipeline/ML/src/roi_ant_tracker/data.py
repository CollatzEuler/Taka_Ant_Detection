from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class CocoBoxDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    """Minimal COCO bounding-box reader; every non-crowd annotation is class ant."""

    def __init__(
        self,
        images_dir: str | Path,
        annotations_path: str | Path,
        training: bool = False,
        max_empty_fraction: float = 1.0,
    ) -> None:
        if not 0.0 <= max_empty_fraction <= 1.0:
            raise ValueError("max_empty_fraction must be in [0, 1].")
        self.images_dir = Path(images_dir)
        self.training = training
        payload = json.loads(Path(annotations_path).read_text(encoding="utf-8-sig"))
        self.images = sorted(payload["images"], key=lambda item: item["id"])
        self.annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            if not annotation.get("iscrowd", 0):
                self.annotations[int(annotation["image_id"])].append(annotation)
        if training and max_empty_fraction < 1.0:
            self.images = self.limit_empty_images(self.images, max_empty_fraction)
        self.positive_image_count = sum(1 for image in self.images if self.annotations.get(int(image["id"])))
        self.empty_image_count = len(self.images) - self.positive_image_count

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_info = self.images[index]
        image_path = self.images_dir / image_info["file_name"]
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        boxes: list[list[float]] = []
        areas: list[float] = []
        for annotation in self.annotations.get(int(image_info["id"]), []):
            x, y, box_width, box_height = [float(value) for value in annotation["bbox"]]
            x1 = max(0.0, min(float(width), x))
            y1 = max(0.0, min(float(height), y))
            x2 = max(0.0, min(float(width), x + box_width))
            y2 = max(0.0, min(float(height), y + box_height))
            if x2 - x1 <= 1.0 or y2 - y1 <= 1.0:
                continue
            boxes.append([x1, y1, x2, y2])
            areas.append((x2 - x1) * (y2 - y1))

        if self.training and random.random() < 0.5:
            image = image[:, ::-1].copy()
            boxes = [[width - x2, y1, width - x1, y2] for x1, y1, x2, y2 in boxes]
        if self.training and random.random() < 0.5:
            image = image[::-1, :].copy()
            boxes = [[x1, height - y2, x2, height - y1] for x1, y1, x2, y2 in boxes]

        box_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target = {
            "boxes": box_tensor,
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "image_id": torch.tensor([int(image_info["id"])]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return to_tensor(image), target

    def limit_empty_images(self, images: list[dict[str, Any]], max_empty_fraction: float) -> list[dict[str, Any]]:
        positive = [image for image in images if self.annotations.get(int(image["id"]))]
        empty = [image for image in images if not self.annotations.get(int(image["id"]))]
        if not positive:
            return images
        if max_empty_fraction <= 0.0:
            return positive
        allowed_empty = int((max_empty_fraction * len(positive)) / max(1.0 - max_empty_fraction, 1e-9))
        if len(empty) <= allowed_empty:
            return images
        rng = random.Random(7)
        kept_empty = rng.sample(empty, allowed_empty)
        kept_ids = {int(image["id"]) for image in [*positive, *kept_empty]}
        return [image for image in images if int(image["id"]) in kept_ids]


def collate_detection_batch(batch: list[Any]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    return tuple(zip(*batch))
