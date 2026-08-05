from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from roi_ant_tracker.geometry import Detection
from roi_ant_tracker.inference import RoiPredictor, resolve_frame_window
from roi_ant_tracker.roi import RoiRect, load_rois


@dataclass(slots=True)
class SuggestedFrame:
    video_path: str
    video_stem: str
    roi_json: str | None
    roi: RoiRect
    source_frame_index: int
    source_time_seconds: float
    detections: list[Detection]
    priority: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use a partially trained detector to prelabel high-value ROI frames for correction."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--roi-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=200, help="Number of ROI images to write for annotation.")
    parser.add_argument("--frame-step", type=int, default=30, help="Run the detector every Nth source frame.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--start-seconds", type=float)
    parser.add_argument("--end-seconds", type=float)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-threshold", type=float, default=0.12)
    parser.add_argument("--nms-threshold", type=float, default=0.70)
    parser.add_argument("--tile-size", type=int, default=0)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--tile-batch-size", type=int, default=4)
    parser.add_argument("--frame-batch-size", type=int, default=2)
    parser.add_argument("--amp", choices=["auto", "off", "fp16", "bf16"], default="auto")
    parser.add_argument("--inference-min-size", type=int)
    parser.add_argument("--inference-max-size", type=int)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--max-boxes-per-image", type=int, default=80)
    parser.add_argument("--min-frame-gap", type=int, default=0, help="Optional spacing between selected frames per ROI.")
    parser.add_argument("--progress-every", type=int, default=20)
    return parser


def main() -> None:
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    args = build_parser().parse_args()
    run_sampler(args)


def run_sampler(args: argparse.Namespace) -> None:
    if args.count <= 0:
        raise ValueError("count must be positive.")
    if args.frame_step <= 0:
        raise ValueError("frame-step must be positive.")
    if args.frame_batch_size <= 0:
        raise ValueError("frame-batch-size must be positive.")
    if args.max_boxes_per_image <= 0:
        raise ValueError("max-boxes-per-image must be positive.")

    output_dir = Path(args.output)
    frames_dir = output_dir / "selected_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame, final_frame = resolve_frame_window(
            fps=fps,
            total_frames=total_frames,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            start_seconds=args.start_seconds,
            end_seconds=args.end_seconds,
            duration_seconds=args.duration_seconds,
        )
        rois = load_rois(args.roi_json, frame_width, frame_height)
        predictor = RoiPredictor(
            checkpoint=args.checkpoint,
            roi_json=args.roi_json,
            frame_width=frame_width,
            frame_height=frame_height,
            device_name=args.device,
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            tile_batch_size=args.tile_batch_size,
            amp=args.amp,
            inference_min_size=args.inference_min_size,
            inference_max_size=args.inference_max_size,
            max_detections=args.max_detections,
        )
        print(
            f"[suggest] video={args.video} frames=[{start_frame}, {final_frame}) "
            f"seconds=[{start_frame / fps:.3f}, {final_frame / fps:.3f}) rois={len(rois)} "
            f"device={predictor.device} threshold={args.score_threshold}"
        )
        candidates = scan_candidates(
            capture=capture,
            video_path=args.video,
            roi_json=args.roi_json,
            rois=rois,
            predictor=predictor,
            fps=fps,
            start_frame=start_frame,
            final_frame=final_frame,
            frame_step=args.frame_step,
            frame_batch_size=args.frame_batch_size,
            max_boxes_per_image=args.max_boxes_per_image,
            progress_every=args.progress_every,
        )
    finally:
        capture.release()

    selected = select_suggestions(candidates, args.count, args.min_frame_gap)
    if not selected:
        raise RuntimeError(
            "No suggested boxes were found. Try lowering --score-threshold or scanning more frames."
        )
    write_outputs(
        selected=selected,
        output_dir=output_dir,
        frames_dir=frames_dir,
        source_video=args.video,
        frame_width=frame_width,
        frame_height=frame_height,
        fps=fps,
        args=args,
    )
    print(f"[suggest] candidates={len(candidates)} selected={len(selected)}")
    print(f"[suggest] frames={frames_dir}")
    print(f"[suggest] project={output_dir / 'annotations.json'}")


def scan_candidates(
    capture: cv2.VideoCapture,
    video_path: str,
    roi_json: str,
    rois: list[RoiRect],
    predictor: RoiPredictor,
    fps: float,
    start_frame: int,
    final_frame: int,
    frame_step: int,
    frame_batch_size: int,
    max_boxes_per_image: int,
    progress_every: int,
) -> list[SuggestedFrame]:
    frame_indices = list(range(start_frame, final_frame, frame_step))
    video_stem = sanitize_label(Path(video_path).stem)
    candidates: list[SuggestedFrame] = []
    started = time.time()
    processed = 0

    for batch_start in range(0, len(frame_indices), frame_batch_size):
        batch_indices = frame_indices[batch_start : batch_start + frame_batch_size]
        frames: list[np.ndarray] = []
        actual_indices: list[int] = []
        for frame_index in batch_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            frames.append(frame)
            actual_indices.append(frame_index)
        if not frames:
            continue
        detections_batch = predictor.predict_many(frames)
        for frame_index, detections in zip(actual_indices, detections_batch):
            by_roi: dict[str, list[Detection]] = {}
            for detection in detections:
                by_roi.setdefault(detection.roi_id, []).append(detection)
            for roi in rois:
                roi_detections = sorted(
                    by_roi.get(roi.roi_id, []),
                    key=lambda detection: detection.score,
                    reverse=True,
                )[:max_boxes_per_image]
                if not roi_detections:
                    continue
                candidates.append(
                    SuggestedFrame(
                        video_path=str(Path(video_path).resolve()),
                        video_stem=video_stem,
                        roi_json=str(Path(roi_json).resolve()),
                        roi=roi,
                        source_frame_index=frame_index,
                        source_time_seconds=frame_index / fps,
                        detections=roi_detections,
                        priority=suggestion_priority(roi_detections),
                    )
                )
        processed += len(actual_indices)
        if progress_every > 0 and (
            processed == len(actual_indices)
            or processed % progress_every == 0
            or batch_start + frame_batch_size >= len(frame_indices)
        ):
            report_progress("[suggest]", processed, len(frame_indices), started, f"candidates={len(candidates)}")
    return candidates


def suggestion_priority(detections: list[Detection]) -> float:
    # Prefer frames with several likely ants, and mildly boost scores near 0.5
    # because those are most useful for correcting the current decision boundary.
    count_score = min(len(detections), 20) * 2.0
    confidence_score = sum(float(detection.score) for detection in detections)
    uncertainty_score = sum(1.0 - min(abs(float(detection.score) - 0.5) / 0.5, 1.0) for detection in detections)
    return count_score + confidence_score + uncertainty_score


def select_suggestions(candidates: list[SuggestedFrame], count: int, min_frame_gap: int) -> list[SuggestedFrame]:
    selected: list[SuggestedFrame] = []
    selected_by_roi: dict[str, list[int]] = {}
    for candidate in sorted(candidates, key=lambda item: item.priority, reverse=True):
        prior_frames = selected_by_roi.setdefault(candidate.roi.roi_id, [])
        if min_frame_gap > 0 and any(abs(candidate.source_frame_index - frame) < min_frame_gap for frame in prior_frames):
            continue
        selected.append(candidate)
        prior_frames.append(candidate.source_frame_index)
        if len(selected) >= count:
            break
    return sorted(selected, key=lambda item: (item.source_frame_index, item.roi.roi_id))


def write_outputs(
    selected: list[SuggestedFrame],
    output_dir: Path,
    frames_dir: Path,
    source_video: str,
    frame_width: int,
    frame_height: int,
    fps: float,
    args: argparse.Namespace,
) -> None:
    capture = cv2.VideoCapture(source_video)
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen video: {source_video}")
    rows: list[dict[str, object]] = []
    images: list[dict[str, object]] = []
    try:
        for image_id, item in enumerate(selected, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, item.source_frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            crop = item.roi.crop(frame)
            image_name = (
                f"{item.video_stem}_{sanitize_label(item.roi.roi_id)}_"
                f"frame_{item.source_frame_index:07d}_suggested.png"
            )
            image_path = frames_dir / image_name
            cv2.imwrite(str(image_path), crop)
            rows.append(
                {
                    "image_name": image_name,
                    "image_path": str(image_path.resolve()),
                    "video_path": str(Path(source_video).resolve()),
                    "source_frame_index": item.source_frame_index,
                    "source_time_seconds": round(item.source_time_seconds, 6),
                    "width": item.roi.width,
                    "height": item.roi.height,
                    "roi_json": str(Path(args.roi_json).resolve()),
                    "roi_id": item.roi.roi_id,
                    "roi_x": item.roi.x,
                    "roi_y": item.roi.y,
                    "roi_width": item.roi.width,
                    "roi_height": item.roi.height,
                    "cluster_id": image_id,
                    "distance_to_cluster_center": round(item.priority, 8),
                    "suggested_box_count": len(item.detections),
                    "max_score": round(max(float(detection.score) for detection in item.detections), 6),
                }
            )
            images.append(
                {
                    "image_id": image_id,
                    "file_name": image_name,
                    "image_path": str(image_path.resolve()),
                    "width": item.roi.width,
                    "height": item.roi.height,
                    "video_path": str(Path(source_video).resolve()),
                    "source_frame_index": item.source_frame_index,
                    "source_time_seconds": item.source_time_seconds,
                    "roi_json": str(Path(args.roi_json).resolve()),
                    "roi_id": item.roi.roi_id,
                    "roi_x": item.roi.x,
                    "roi_y": item.roi.y,
                    "roi_width": item.roi.width,
                    "roi_height": item.roi.height,
                    "cluster_id": image_id,
                    "annotations": [suggested_annotation(detection, item.roi) for detection in item.detections],
                }
            )
    finally:
        capture.release()

    manifest_path = output_dir / "frame_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    project = {
        "project_version": 1,
        "project_type": "suggested_boxes",
        "manifest_path": str(manifest_path.resolve()),
        "images_dir": str(frames_dir.resolve()),
        "images": images,
    }
    (output_dir / "annotations.json").write_text(json.dumps(project, indent=2), encoding="utf-8")
    summary = {
        "video": str(Path(source_video).resolve()),
        "roi_json": str(Path(args.roi_json).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "fps": fps,
        "selected_count": len(images),
        "score_threshold": args.score_threshold,
        "frame_step": args.frame_step,
    }
    (output_dir / "suggested_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def suggested_annotation(detection: Detection, roi: RoiRect) -> dict[str, object]:
    x1, y1, x2, y2 = detection.bbox
    local_x1 = max(0.0, min(float(roi.width), x1 - roi.x))
    local_y1 = max(0.0, min(float(roi.height), y1 - roi.y))
    local_x2 = max(0.0, min(float(roi.width), x2 - roi.x))
    local_y2 = max(0.0, min(float(roi.height), y2 - roi.y))
    width = max(1.0, local_x2 - local_x1)
    height = max(1.0, local_y2 - local_y1)
    return {
        "bbox": [round(local_x1, 2), round(local_y1, 2), round(width, 2), round(height, 2)],
        "category": "ant",
        "identity": None,
        "suggested": True,
        "score": round(float(detection.score), 6),
    }


def report_progress(prefix: str, completed: int, total: int, started: float, detail: str) -> None:
    elapsed = max(time.time() - started, 1e-6)
    rate = completed / elapsed
    remaining = max(total - completed, 0)
    eta = remaining / max(rate, 1e-6)
    percent = (completed / total) * 100.0 if total > 0 else 100.0
    print(
        f"{prefix} {completed}/{total} frames ({percent:.1f}%) {detail} "
        f"rate={rate:.2f} fps eta={format_seconds(eta)}"
    )


def format_seconds(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def sanitize_label(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text.strip("_") or "item"


if __name__ == "__main__":
    main()
