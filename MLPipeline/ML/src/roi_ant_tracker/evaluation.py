from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from .data import CocoBoxDataset, collate_detection_batch
from .inference import resolve_amp_dtype
from .model import choose_device, load_detector


@dataclass(slots=True, frozen=True)
class PredictionRecord:
    image_id: int
    score: float
    box: tuple[float, float, float, float]


def evaluate_detector(
    images_dir: str,
    annotations_path: str,
    checkpoint_path: str,
    output_dir: str,
    device_name: str = "auto",
    batch_size: int = 2,
    workers: int = 0,
    amp: str = "auto",
    score_threshold: float = 0.30,
    candidate_score_threshold: float = 0.05,
    iou_threshold: float = 0.50,
    center_distance_threshold: float = 12.0,
    inference_min_size: int | None = None,
    inference_max_size: int | None = None,
    progress_every: int = 25,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if not 0.0 <= candidate_score_threshold <= score_threshold <= 1.0:
        raise ValueError("Require 0 <= candidate-score-threshold <= score-threshold <= 1.")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou-threshold must be in (0, 1].")
    if center_distance_threshold <= 0:
        raise ValueError("center-distance-threshold must be positive.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = CocoBoxDataset(images_dir, annotations_path, training=False)
    if not dataset:
        raise ValueError("No images were found in the COCO annotation file.")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_detection_batch,
    )
    device = choose_device(device_name)
    load_started = time.perf_counter()
    model, model_config = load_detector(checkpoint_path, device)
    model_load_seconds = time.perf_counter() - load_started
    model.roi_heads.score_thresh = candidate_score_threshold
    if inference_min_size is not None:
        model.transform.min_size = (inference_min_size,)
    if inference_max_size is not None:
        model.transform.max_size = inference_max_size
    amp_dtype = resolve_amp_dtype(amp, device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    predictions: list[PredictionRecord] = []
    ground_truth: dict[int, list[tuple[float, float, float, float]]] = {}
    image_rows: list[dict[str, Any]] = []
    metadata_by_id = {int(item["id"]): item for item in dataset.images}
    model_seconds = 0.0
    evaluation_started = time.perf_counter()
    completed = 0
    print(
        f"[evaluate] images={len(dataset)} gt_boxes={sum(len(value) for value in dataset.annotations.values())} "
        f"device={device} batch_size={batch_size} amp={amp_dtype or 'off'}"
    )

    for images, targets in loader:
        device_images = [image.to(device) for image in images]
        synchronize(device)
        model_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            outputs = model(device_images)
        synchronize(device)
        model_seconds += time.perf_counter() - model_started

        for target, prediction in zip(targets, outputs):
            image_id = int(target["image_id"].item())
            gt_boxes = [tuple(float(value) for value in box) for box in target["boxes"].tolist()]
            ground_truth[image_id] = gt_boxes
            labels = prediction["labels"].detach().cpu().tolist()
            boxes = prediction["boxes"].detach().cpu().tolist()
            scores = prediction["scores"].detach().cpu().tolist()
            image_predictions = [
                PredictionRecord(image_id, float(score), tuple(float(value) for value in box))
                for box, score, label in zip(boxes, scores, labels)
                if int(label) == 1 and float(score) >= candidate_score_threshold
            ]
            predictions.extend(image_predictions)
            counts = match_image(image_predictions, gt_boxes, score_threshold, iou_threshold)
            info = metadata_by_id[image_id]
            image_rows.append(
                {
                    "image_id": image_id,
                    "file_name": info["file_name"],
                    "roi_id": str(info.get("roi_id") or "unknown"),
                    "source_frame_index": info.get("source_frame_index", ""),
                    "ground_truth": len(gt_boxes),
                    "predictions": counts["predictions"],
                    "true_positives": counts["true_positives"],
                    "false_positives": counts["false_positives"],
                    "false_negatives": counts["false_negatives"],
                }
            )
            completed += 1
        if progress_every > 0 and (completed % progress_every == 0 or completed == len(dataset)):
            elapsed = max(time.perf_counter() - evaluation_started, 1e-6)
            print(f"[evaluate] {completed}/{len(dataset)} images rate={completed / elapsed:.2f} images/s")

    ranked_scores, true_positive_flags = rank_predictions(predictions, ground_truth, iou_threshold)
    total_ground_truth = sum(len(boxes) for boxes in ground_truth.values())
    overall = metrics_at_threshold(ranked_scores, true_positive_flags, total_ground_truth, score_threshold)
    best = best_operating_point(ranked_scores, true_positive_flags, total_ground_truth)
    ap = average_precision(ranked_scores, true_positive_flags, total_ground_truth)
    center_scores, center_true_positive_flags = rank_predictions_by_center(
        predictions, ground_truth, center_distance_threshold
    )
    center_overall = metrics_at_threshold(
        center_scores, center_true_positive_flags, total_ground_truth, score_threshold
    )
    center_best = best_operating_point(center_scores, center_true_positive_flags, total_ground_truth)
    operating_thresholds = sorted({0.10, 0.20, 0.30, 0.40, 0.50, 0.70, score_threshold})
    per_roi = summarize_groups(image_rows, "roi_id")
    evaluation_seconds = max(time.perf_counter() - evaluation_started, 1e-6)
    summary: dict[str, Any] = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "annotations": str(Path(annotations_path).resolve()),
        "images_dir": str(Path(images_dir).resolve()),
        "evaluation_scope": (
            "These metrics are out-of-sample only when the evaluated images were excluded from detector training."
        ),
        "device": str(device),
        "amp": str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "off",
        "model_config": model_config,
        "image_count": len(dataset),
        "positive_image_count": dataset.positive_image_count,
        "empty_image_count": dataset.empty_image_count,
        "ground_truth_boxes": total_ground_truth,
        "candidate_score_threshold": candidate_score_threshold,
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        **overall,
        "average_precision": ap,
        "best_f1_operating_point": best,
        "operating_points": {
            f"{threshold:.2f}": metrics_at_threshold(
                ranked_scores, true_positive_flags, total_ground_truth, threshold
            )
            for threshold in operating_thresholds
        },
        "center_distance_evaluation": {
            "distance_threshold_pixels": center_distance_threshold,
            **center_overall,
            "average_precision": average_precision(
                center_scores, center_true_positive_flags, total_ground_truth
            ),
            "best_f1_operating_point": center_best,
            "operating_points": {
                f"{threshold:.2f}": metrics_at_threshold(
                    center_scores, center_true_positive_flags, total_ground_truth, threshold
                )
                for threshold in operating_thresholds
            },
        },
        "per_roi": per_roi,
        "model_load_seconds": model_load_seconds,
        "model_inference_seconds": model_seconds,
        "evaluation_seconds": evaluation_seconds,
        "model_images_per_second": len(dataset) / max(model_seconds, 1e-6),
        "end_to_end_images_per_second": len(dataset) / evaluation_seconds,
    }
    write_image_metrics(output / "per_image_metrics.csv", image_rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[evaluate] precision={overall['precision']:.3f} recall={overall['recall']:.3f} "
        f"f1={overall['f1']:.3f} ap@{iou_threshold:.2f}={ap:.3f} "
        f"best_threshold={best['score_threshold']:.3f} best_f1={best['f1']:.3f}"
    )
    print(f"[evaluate] summary={output / 'summary.json'}")
    return summary


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def match_image(
    predictions: list[PredictionRecord],
    ground_truth: list[tuple[float, float, float, float]],
    score_threshold: float,
    iou_threshold: float,
) -> dict[str, int]:
    selected = sorted(
        (prediction for prediction in predictions if prediction.score >= score_threshold),
        key=lambda prediction: prediction.score,
        reverse=True,
    )
    unmatched = set(range(len(ground_truth)))
    true_positives = 0
    for prediction in selected:
        match = best_unmatched_gt(prediction.box, ground_truth, unmatched, iou_threshold)
        if match is not None:
            unmatched.remove(match)
            true_positives += 1
    return {
        "predictions": len(selected),
        "true_positives": true_positives,
        "false_positives": len(selected) - true_positives,
        "false_negatives": len(unmatched),
    }


def best_unmatched_gt(
    prediction: tuple[float, float, float, float],
    ground_truth: list[tuple[float, float, float, float]],
    unmatched: set[int],
    iou_threshold: float,
) -> int | None:
    if not unmatched:
        return None
    candidates = sorted(unmatched)
    ious = box_iou(
        torch.tensor([prediction], dtype=torch.float32),
        torch.tensor([ground_truth[index] for index in candidates], dtype=torch.float32),
    )[0]
    best_local = int(torch.argmax(ious).item())
    return candidates[best_local] if float(ious[best_local]) >= iou_threshold else None


def rank_predictions(
    predictions: list[PredictionRecord],
    ground_truth: dict[int, list[tuple[float, float, float, float]]],
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    ranked = sorted(predictions, key=lambda prediction: prediction.score, reverse=True)
    unmatched = {image_id: set(range(len(boxes))) for image_id, boxes in ground_truth.items()}
    scores: list[float] = []
    flags: list[int] = []
    for prediction in ranked:
        scores.append(prediction.score)
        gt_boxes = ground_truth.get(prediction.image_id, [])
        available = unmatched.setdefault(prediction.image_id, set(range(len(gt_boxes))))
        match = best_unmatched_gt(prediction.box, gt_boxes, available, iou_threshold)
        if match is None:
            flags.append(0)
        else:
            available.remove(match)
            flags.append(1)
    return np.asarray(scores, dtype=np.float64), np.asarray(flags, dtype=np.int64)


def rank_predictions_by_center(
    predictions: list[PredictionRecord],
    ground_truth: dict[int, list[tuple[float, float, float, float]]],
    distance_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    ranked = sorted(predictions, key=lambda prediction: prediction.score, reverse=True)
    unmatched = {image_id: set(range(len(boxes))) for image_id, boxes in ground_truth.items()}
    scores: list[float] = []
    flags: list[int] = []
    for prediction in ranked:
        scores.append(prediction.score)
        gt_boxes = ground_truth.get(prediction.image_id, [])
        available = unmatched.setdefault(prediction.image_id, set(range(len(gt_boxes))))
        match = best_unmatched_gt_by_center(prediction.box, gt_boxes, available, distance_threshold)
        if match is None:
            flags.append(0)
        else:
            available.remove(match)
            flags.append(1)
    return np.asarray(scores, dtype=np.float64), np.asarray(flags, dtype=np.int64)


def best_unmatched_gt_by_center(
    prediction: tuple[float, float, float, float],
    ground_truth: list[tuple[float, float, float, float]],
    unmatched: set[int],
    distance_threshold: float,
) -> int | None:
    if not unmatched:
        return None
    prediction_center = ((prediction[0] + prediction[2]) / 2.0, (prediction[1] + prediction[3]) / 2.0)
    distances = {
        index: (
            (prediction_center[0] - (ground_truth[index][0] + ground_truth[index][2]) / 2.0) ** 2
            + (prediction_center[1] - (ground_truth[index][1] + ground_truth[index][3]) / 2.0) ** 2
        )
        ** 0.5
        for index in unmatched
    }
    best = min(distances, key=distances.get)
    return best if distances[best] <= distance_threshold else None


def metrics_at_threshold(
    ranked_scores: np.ndarray,
    true_positive_flags: np.ndarray,
    total_ground_truth: int,
    score_threshold: float,
) -> dict[str, float | int]:
    selected = ranked_scores >= score_threshold
    predicted = int(selected.sum())
    true_positives = int(true_positive_flags[selected].sum())
    false_positives = predicted - true_positives
    false_negatives = total_ground_truth - true_positives
    precision = safe_divide(true_positives, predicted)
    recall = safe_divide(true_positives, total_ground_truth)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    return {
        "predictions": predicted,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def best_operating_point(
    ranked_scores: np.ndarray,
    true_positive_flags: np.ndarray,
    total_ground_truth: int,
) -> dict[str, float | int]:
    if len(ranked_scores) == 0:
        return {"score_threshold": 1.0, **metrics_at_threshold(ranked_scores, true_positive_flags, total_ground_truth, 1.0)}
    best: dict[str, float | int] | None = None
    for index, score in enumerate(ranked_scores):
        if index + 1 < len(ranked_scores) and ranked_scores[index + 1] == score:
            continue
        metrics = metrics_at_threshold(ranked_scores, true_positive_flags, total_ground_truth, float(score))
        candidate = {"score_threshold": float(score), **metrics}
        if best is None or (candidate["f1"], candidate["precision"], candidate["score_threshold"]) > (
            best["f1"],
            best["precision"],
            best["score_threshold"],
        ):
            best = candidate
    assert best is not None
    return best


def average_precision(
    ranked_scores: np.ndarray,
    true_positive_flags: np.ndarray,
    total_ground_truth: int,
) -> float:
    if total_ground_truth == 0 or len(ranked_scores) == 0:
        return 0.0
    cumulative_tp = np.cumsum(true_positive_flags)
    cumulative_fp = np.cumsum(1 - true_positive_flags)
    recalls = cumulative_tp / total_ground_truth
    precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])
    changes = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[changes + 1] - recalls[changes]) * precisions[changes + 1]))


def summarize_groups(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, float | int]]:
    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        group = str(row[field])
        totals[group]["images"] += 1
        for key in ("ground_truth", "predictions", "true_positives", "false_positives", "false_negatives"):
            totals[group][key] += int(row[key])
    result: dict[str, dict[str, float | int]] = {}
    for group, values in sorted(totals.items()):
        precision = safe_divide(values["true_positives"], values["predictions"])
        recall = safe_divide(values["true_positives"], values["ground_truth"])
        result[group] = {
            **values,
            "precision": precision,
            "recall": recall,
            "f1": safe_divide(2.0 * precision * recall, precision + recall),
        }
    return result


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def write_image_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "image_id",
        "file_name",
        "roi_id",
        "source_frame_index",
        "ground_truth",
        "predictions",
        "true_positives",
        "false_positives",
        "false_negatives",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
