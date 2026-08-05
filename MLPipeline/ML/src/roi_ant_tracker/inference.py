from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import nms
from torchvision.transforms.functional import to_tensor

from .geometry import (
    Box,
    Detection,
    appearance_distance,
    appearance_histogram,
    box_center,
    box_iou,
    clip_box,
    distance,
    translate_box,
)
from .model import choose_device, load_detector
from .roi import RoiRect, draw_rois, load_rois
from .tiling import iter_tiles


@dataclass(slots=True)
class Track:
    track_id: int
    bbox: Box
    score: float
    appearance: np.ndarray | None
    last_frame: int
    previous_center: tuple[float, float] | None = None
    previous_frame: int | None = None
    missed: int = 0
    age: int = 1
    observed_hits: int = 1
    last_assignment_cost: float = 0.0
    last_roi_id: str = ""

    @property
    def center(self) -> tuple[float, float]:
        return box_center(self.bbox)

    def predicted_box(self, frame_index: int) -> Box:
        if self.previous_center is None or self.previous_frame is None:
            return self.bbox
        gap = max(self.last_frame - self.previous_frame, 1)
        vx = (self.center[0] - self.previous_center[0]) / gap
        vy = (self.center[1] - self.previous_center[1]) / gap
        prediction_gap = max(frame_index - self.last_frame, 1)
        return translate_box(self.bbox, vx * prediction_gap, vy * prediction_gap)


@dataclass(slots=True)
class ReviewEvent:
    frame_index: int
    risk_score: float = 0.0
    reasons: set[str] = field(default_factory=set)
    track_ids: set[int] = field(default_factory=set)


@dataclass(slots=True)
class OverlayTarget:
    path: Path
    writer: cv2.VideoWriter
    width: int
    height: int
    roi: RoiRect | None = None


class RoiPredictor:
    def __init__(
        self,
        checkpoint: str,
        roi_json: str | None,
        frame_width: int,
        frame_height: int,
        device_name: str,
        score_threshold: float,
        nms_threshold: float,
        tile_size: int,
        tile_overlap: int,
        tile_batch_size: int,
        roi_padding: int,
        amp: str,
        inference_min_size: int | None,
        inference_max_size: int | None,
        max_detections: int,
        duplicate_center_distance: float,
        duplicate_overlap_threshold: float,
    ) -> None:
        self.device = choose_device(device_name)
        self.model, _ = load_detector(checkpoint, self.device)
        self.rois = load_rois(roi_json, frame_width, frame_height)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.tile_batch_size = tile_batch_size
        self.roi_padding = roi_padding
        self.max_detections = max_detections
        self.duplicate_center_distance = duplicate_center_distance
        self.duplicate_overlap_threshold = duplicate_overlap_threshold
        self.amp_dtype = resolve_amp_dtype(amp, self.device)
        if inference_min_size is not None:
            if inference_min_size <= 0:
                raise ValueError("inference-min-size must be positive.")
            self.model.transform.min_size = (inference_min_size,)
        if inference_max_size is not None:
            if inference_max_size <= 0:
                raise ValueError("inference-max-size must be positive.")
            self.model.transform.max_size = inference_max_size
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

    @torch.inference_mode()
    def predict_many(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        flat_tiles: list[tuple[int, RoiRect, np.ndarray, int, int]] = []
        for frame_index, frame in enumerate(frames):
            for roi in self.rois:
                search_x1, search_y1, search_x2, search_y2 = roi.padded_bounds(
                    self.roi_padding, frame.shape[1], frame.shape[0]
                )
                roi_frame = frame[search_y1:search_y2, search_x1:search_x2]
                for tile, tile_x, tile_y in iter_tiles(roi_frame, self.tile_size, self.tile_overlap):
                    flat_tiles.append((frame_index, roi, tile, search_x1 + tile_x, search_y1 + tile_y))

        boxes_by_frame: list[list[list[float]]] = [[] for _ in frames]
        scores_by_frame: list[list[float]] = [[] for _ in frames]
        roi_ids_by_frame: list[list[str]] = [[] for _ in frames]
        for start in range(0, len(flat_tiles), self.tile_batch_size):
            batch = flat_tiles[start : start + self.tile_batch_size]
            tensors = [
                to_tensor(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)).to(self.device)
                for _, _, tile, _, _ in batch
            ]
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_dtype is not None,
            ):
                outputs = self.model(tensors)
            for output, (frame_index, roi, _tile, offset_x, offset_y) in zip(outputs, batch):
                for box, score in zip(output["boxes"].cpu().tolist(), output["scores"].cpu().tolist()):
                    if score < self.score_threshold:
                        continue
                    x1, y1, x2, y2 = box
                    full_box = [
                        x1 + offset_x,
                        y1 + offset_y,
                        x2 + offset_x,
                        y2 + offset_y,
                    ]
                    clipped = clip_box(tuple(full_box), frames[frame_index].shape[1], frames[frame_index].shape[0])
                    if clipped[2] - clipped[0] <= 1.0 or clipped[3] - clipped[1] <= 1.0:
                        continue
                    if not roi.contains(box_center(clipped)):
                        continue
                    boxes_by_frame[frame_index].append(list(clipped))
                    scores_by_frame[frame_index].append(float(score))
                    roi_ids_by_frame[frame_index].append(roi.roi_id)

        return [
            finalize_detections(
                frame,
                boxes,
                scores,
                roi_ids,
                self.nms_threshold,
                self.max_detections,
                self.duplicate_center_distance,
                self.duplicate_overlap_threshold,
            )
            for frame, boxes, scores, roi_ids in zip(frames, boxes_by_frame, scores_by_frame, roi_ids_by_frame)
        ]


class OpenWorldTracker:
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        max_distance: float,
        max_missed: int,
        match_threshold: float,
        ambiguity_margin: float,
        close_distance: float,
        border_margin: float,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.match_threshold = match_threshold
        self.ambiguity_margin = ambiguity_margin
        self.close_distance = close_distance
        self.border_margin = border_margin
        self.tracks: dict[int, Track] = {}
        self.next_track_id = 1
        self.events: dict[int, ReviewEvent] = {}

    def update(self, frame_index: int, detections: list[Detection]) -> list[dict[str, object]]:
        track_ids = list(self.tracks)
        costs = self._cost_matrix(frame_index, track_ids, detections)
        matches, unmatched_tracks, unmatched_detections = assign(costs, self.match_threshold)
        rows: list[dict[str, object]] = []

        for track_index, detection_index in matches:
            track = self.tracks[track_ids[track_index]]
            detection = detections[detection_index]
            margin = assignment_margin(costs, track_index, detection_index)
            appearance_jump = appearance_distance(track.appearance, detection.appearance)
            predicted_center = box_center(track.predicted_box(frame_index))
            jump = distance(predicted_center, detection.center)
            frame_gap = max(frame_index - track.last_frame, 1)
            risk, reasons = assignment_risk(
                float(costs[track_index, detection_index]),
                margin,
                appearance_jump,
                jump,
                self.ambiguity_margin,
                self.max_distance * frame_gap,
            )
            self._update_track(track, detection, frame_index, float(costs[track_index, detection_index]))
            if reasons:
                self.add_event(frame_index, risk, reasons, [track.track_id])
            rows.append(track_row(frame_index, track, True, margin, risk, "|".join(sorted(reasons))))

        ended: list[int] = []
        for track_index in unmatched_tracks:
            track = self.tracks[track_ids[track_index]]
            track.age += 1
            track.missed += 1
            if track.missed > self.max_missed:
                if not near_border(track.center, self.frame_width, self.frame_height, self.border_margin):
                    self.add_event(frame_index, 0.8, {"interior_track_end"}, [track.track_id])
                ended.append(track.track_id)
                continue
            track.bbox = track.predicted_box(frame_index)
            track.last_frame = frame_index
            risk = min(0.35 + track.missed / max(self.max_missed, 1) * 0.45, 0.9)
            self.add_event(frame_index, risk, {"missing_detection"}, [track.track_id])
            rows.append(track_row(frame_index, track, False, 0.0, risk, "missing_detection"))
        for track_id in ended:
            del self.tracks[track_id]

        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            track = Track(
                track_id=self.next_track_id,
                bbox=detection.bbox,
                score=detection.score,
                appearance=detection.appearance.copy() if detection.appearance is not None else None,
                last_frame=frame_index,
                last_roi_id=detection.roi_id,
            )
            self.tracks[track.track_id] = track
            self.next_track_id += 1
            risk = 0.1 if near_border(track.center, self.frame_width, self.frame_height, self.border_margin) else 0.7
            reason = "" if risk < 0.5 else "interior_track_birth"
            if reason:
                self.add_event(frame_index, risk, {reason}, [track.track_id])
            rows.append(track_row(frame_index, track, True, 1.0, risk, reason))

        self._flag_close_encounters(frame_index)
        return sorted(rows, key=lambda row: int(row["track_id"]))

    def _cost_matrix(self, frame_index: int, track_ids: list[int], detections: list[Detection]) -> np.ndarray:
        if not track_ids or not detections:
            return np.empty((len(track_ids), len(detections)), dtype=np.float32)
        costs = np.full((len(track_ids), len(detections)), 1e6, dtype=np.float32)
        for track_index, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            predicted_box = track.predicted_box(frame_index)
            predicted_center = box_center(predicted_box)
            frame_gap = max(frame_index - track.last_frame, 1)
            allowed = self.max_distance * frame_gap * (1.0 + 0.5 * min(track.missed, self.max_missed))
            for detection_index, detection in enumerate(detections):
                if track.last_roi_id != detection.roi_id:
                    continue
                motion = distance(predicted_center, detection.center)
                if motion > allowed:
                    continue
                appearance = appearance_distance(track.appearance, detection.appearance)
                overlap = box_iou(predicted_box, detection.bbox)
                costs[track_index, detection_index] = (
                    0.65 * min(motion / max(allowed, 1.0), 1.0)
                    + 0.25 * appearance
                    + 0.10 * (1.0 - overlap)
                )
        return costs

    def _update_track(self, track: Track, detection: Detection, frame_index: int, cost: float) -> None:
        track.previous_center = track.center
        track.previous_frame = track.last_frame
        track.bbox = detection.bbox
        track.score = detection.score
        track.last_frame = frame_index
        track.last_roi_id = detection.roi_id
        track.missed = 0
        track.age += 1
        track.observed_hits += 1
        track.last_assignment_cost = cost
        if detection.appearance is not None:
            track.appearance = (
                detection.appearance.copy()
                if track.appearance is None
                else normalized(0.8 * track.appearance + 0.2 * detection.appearance)
            )

    def _flag_close_encounters(self, frame_index: int) -> None:
        observed = [track for track in self.tracks.values() if track.missed == 0]
        for index, left in enumerate(observed):
            for right in observed[index + 1 :]:
                separation = distance(left.center, right.center)
                if separation < self.close_distance:
                    risk = 0.45 + 0.35 * (1.0 - separation / max(self.close_distance, 1.0))
                    self.add_event(frame_index, risk, {"close_encounter"}, [left.track_id, right.track_id])

    def add_event(self, frame_index: int, risk: float, reasons: set[str], track_ids: list[int]) -> None:
        event = self.events.setdefault(frame_index, ReviewEvent(frame_index))
        event.risk_score = max(event.risk_score, float(risk))
        event.reasons.update(reasons)
        event.track_ids.update(track_ids)


def track_video(
    video_path: str,
    checkpoint_path: str,
    output_dir: str,
    roi_json: str | None = None,
    device_name: str = "auto",
    score_threshold: float = 0.30,
    nms_threshold: float = 0.70,
    tile_size: int = 0,
    tile_overlap: int = 64,
    tile_batch_size: int = 4,
    frame_batch_size: int = 1,
    frame_step: int = 1,
    roi_padding: int = 0,
    amp: str = "auto",
    write_overlay: bool = True,
    overlay_mode: str = "roi",
    inference_min_size: int | None = None,
    inference_max_size: int | None = None,
    max_detections: int = 300,
    duplicate_center_distance: float = 0.0,
    duplicate_overlap_threshold: float = 0.80,
    max_distance: float = 60.0,
    max_missed: int = 5,
    match_threshold: float = 0.80,
    ambiguity_margin: float = 0.12,
    close_distance: float = 80.0,
    border_margin: float = 80.0,
    minimum_track_hits: int = 3,
    review_threshold: float = 0.60,
    start_frame: int = 0,
    end_frame: int | None = None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    duration_seconds: float | None = None,
    progress_every: int = 100,
) -> None:
    if frame_batch_size <= 0 or tile_batch_size <= 0 or frame_step <= 0:
        raise ValueError("frame-batch-size, tile-batch-size, and frame-step must be positive.")
    if roi_padding < 0:
        raise ValueError("roi-padding must be non-negative.")
    if overlay_mode not in {"roi", "full", "both"}:
        raise ValueError("overlay-mode must be one of: roi, full, both.")
    if duplicate_center_distance < 0:
        raise ValueError("duplicate-center-distance must be non-negative.")
    if not 0.0 <= duplicate_overlap_threshold <= 1.0:
        raise ValueError("duplicate-overlap-threshold must be in [0, 1].")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame, final_frame = resolve_frame_window(
        fps=fps,
        total_frames=total_frames,
        start_frame=start_frame,
        end_frame=end_frame,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        duration_seconds=duration_seconds,
    )
    if start_frame >= final_frame:
        capture.release()
        raise ValueError("Selected video window is empty.")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    predictor = RoiPredictor(
        checkpoint=checkpoint_path,
        roi_json=roi_json,
        frame_width=width,
        frame_height=height,
        device_name=device_name,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        tile_batch_size=tile_batch_size,
        roi_padding=roi_padding,
        amp=amp,
        inference_min_size=inference_min_size,
        inference_max_size=inference_max_size,
        max_detections=max_detections,
        duplicate_center_distance=duplicate_center_distance,
        duplicate_overlap_threshold=duplicate_overlap_threshold,
    )
    tracker = OpenWorldTracker(
        frame_width=width,
        frame_height=height,
        max_distance=max_distance,
        max_missed=max_missed,
        match_threshold=match_threshold,
        ambiguity_margin=ambiguity_margin,
        close_distance=close_distance,
        border_margin=border_margin,
    )
    print(
        f"[track] device={predictor.device} rois={len(predictor.rois)} "
        f"frame_batch_size={frame_batch_size} tile_batch_size={tile_batch_size} frame_step={frame_step} "
        f"roi_padding={roi_padding} "
        f"amp={predictor.amp_dtype or 'off'} "
        f"frames=[{start_frame}, {final_frame}) seconds=[{start_frame / fps:.3f}, {final_frame / fps:.3f})"
    )

    overlay_targets = (
        open_overlay_targets(output, fps / frame_step, width, height, predictor.rois, overlay_mode)
        if write_overlay
        else []
    )

    detections_path = output / "detections.csv"
    raw_path = output / "raw_trajectories.csv"
    started = time.time()
    sampled_frame_total = len(range(start_frame, final_frame, frame_step))
    processed_count = 0
    last_processed_frame: int | None = None
    with detections_path.open("w", newline="", encoding="utf-8") as detections_handle, raw_path.open(
        "w", newline="", encoding="utf-8"
    ) as tracks_handle:
        detections_writer = csv.DictWriter(detections_handle, fieldnames=detection_fields())
        tracks_writer = csv.DictWriter(tracks_handle, fieldnames=trajectory_fields())
        detections_writer.writeheader()
        tracks_writer.writeheader()
        next_frame_index = start_frame
        stream_ended = False
        while next_frame_index < final_frame:
            frames: list[np.ndarray] = []
            frame_indices: list[int] = []
            for _ in range(frame_batch_size):
                if next_frame_index >= final_frame:
                    break
                frame_index = next_frame_index
                ok, frame = capture.read()
                if not ok:
                    stream_ended = True
                    break
                frames.append(frame)
                frame_indices.append(frame_index)
                skipped = 1
                while skipped < frame_step and frame_index + skipped < final_frame:
                    if not capture.grab():
                        stream_ended = True
                        break
                    skipped += 1
                next_frame_index = frame_index + frame_step
                if stream_ended:
                    break
            if not frames:
                break
            detections_batch = predictor.predict_many(frames)
            for frame_index, frame, detections in zip(frame_indices, frames, detections_batch):
                processed_count += 1
                last_processed_frame = frame_index
                for detection in detections:
                    detections_writer.writerow(detection_row(frame_index, fps, detection))
                rows = tracker.update(frame_index, detections)
                for row in rows:
                    row["time_seconds"] = frame_index / fps
                    tracks_writer.writerow(row)
                for target in overlay_targets:
                    target.writer.write(draw_overlay_frame(frame, rows, predictor.rois, target, tracker.events.get(frame_index)))
                if progress_every > 0 and (
                    processed_count == 1
                    or processed_count % progress_every == 0
                    or processed_count == sampled_frame_total
                ):
                    report_progress(
                        prefix="[track]",
                        completed=processed_count,
                        total=sampled_frame_total,
                        started=started,
                        detail=(
                            f"frame={frame_index}/{final_frame - 1} detections={len(detections)} "
                            f"active_tracks={len(tracker.tracks)}"
                        ),
                    )
            if stream_ended:
                break

    capture.release()
    for target in overlay_targets:
        target.writer.release()
    processing_seconds = max(time.time() - started, 1e-6)
    requested_window_completed = processed_count == sampled_frame_total
    if not requested_window_completed:
        print(
            f"[track] WARNING: video decoding ended before the requested window completed: "
            f"processed={processed_count}/{sampled_frame_total} last_source_frame={last_processed_frame} "
            f"reported_total_frames={total_frames}"
        )
    valid_ids = write_filtered_trajectories(raw_path, output / "trajectories.csv", minimum_track_hits)
    write_review_outputs(output, tracker.events, valid_ids, review_threshold)
    manifest = {
        "video": str(Path(video_path).resolve()),
        "roi_json": str(Path(roi_json).resolve()) if roi_json else None,
        "detector_checkpoint": str(Path(checkpoint_path).resolve()),
        "frame_width": width,
        "frame_height": height,
        "fps": fps,
        "start_frame": start_frame,
        "end_frame": final_frame,
        "start_seconds": start_frame / fps,
        "end_seconds": final_frame / fps,
        "duration_seconds": (final_frame - start_frame) / fps,
        "frame_step": frame_step,
        "reported_total_frames": total_frames,
        "requested_sample_count": sampled_frame_total,
        "processed_frame_count": processed_count,
        "last_processed_frame": last_processed_frame,
        "requested_window_completed": requested_window_completed,
        "decode_ended_early": not requested_window_completed,
        "processing_seconds": processing_seconds,
        "processing_fps": processed_count / processing_seconds,
        "source_realtime_factor": ((processed_count * frame_step) / fps) / processing_seconds,
        "device": str(predictor.device),
        "amp": str(predictor.amp_dtype).replace("torch.", "") if predictor.amp_dtype is not None else "off",
        "roi_padding": roi_padding,
        "roi_count": len(predictor.rois),
        "rois": [
            {
                "id": roi.roi_id,
                "x": roi.x,
                "y": roi.y,
                "width": roi.width,
                "height": roi.height,
            }
            for roi in predictor.rois
        ],
        "minimum_track_hits": minimum_track_hits,
        "score_threshold": score_threshold,
        "nms_threshold": nms_threshold,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "tile_batch_size": tile_batch_size,
        "frame_batch_size": frame_batch_size,
        "inference_min_size": inference_min_size,
        "inference_max_size": inference_max_size,
        "max_detections": max_detections,
        "duplicate_center_distance": duplicate_center_distance,
        "duplicate_overlap_threshold": duplicate_overlap_threshold,
        "overlay_mode": overlay_mode if write_overlay else "none",
        "overlay_paths": [str(target.path) for target in overlay_targets],
        "valid_track_count": len(valid_ids),
        "review_frame_count": sum(
            1 for event in tracker.events.values() if event.risk_score >= review_threshold
        ),
    }
    (output / "tracking_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[track] processed={processed_count} frames seconds={processing_seconds:.2f} "
        f"rate={processed_count / processing_seconds:.2f} fps "
        f"source_realtime={((processed_count * frame_step) / fps) / processing_seconds:.2f}x"
    )
    print(f"[track] detections={detections_path}")
    print(f"[track] trajectories={output / 'trajectories.csv'}")
    print(f"[track] review={output / 'review_events.csv'}")
    for target in overlay_targets:
        print(f"[track] overlay={target.path}")


def resolve_frame_window(
    fps: float,
    total_frames: int,
    start_frame: int,
    end_frame: int | None,
    start_seconds: float | None,
    end_seconds: float | None,
    duration_seconds: float | None,
) -> tuple[int, int]:
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration-seconds must be positive.")
    if end_seconds is not None and duration_seconds is not None:
        raise ValueError("Use either end-seconds or duration-seconds, not both.")
    first_frame = int(round(start_seconds * fps)) if start_seconds is not None else start_frame
    first_frame = max(0, min(total_frames, first_frame))
    if duration_seconds is not None:
        final_frame = first_frame + int(round(duration_seconds * fps))
    elif end_seconds is not None:
        final_frame = int(round(end_seconds * fps))
    elif end_frame is not None:
        final_frame = end_frame
    else:
        final_frame = total_frames
    final_frame = max(0, min(total_frames, final_frame))
    return first_frame, final_frame


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


def finalize_detections(
    frame: np.ndarray,
    boxes: list[list[float]],
    scores: list[float],
    roi_ids: list[str],
    nms_threshold: float,
    max_detections: int,
    duplicate_center_distance: float = 0.0,
    duplicate_overlap_threshold: float = 0.80,
) -> list[Detection]:
    if not boxes:
        return []
    box_tensor = torch.tensor(boxes, dtype=torch.float32)
    score_tensor = torch.tensor(scores, dtype=torch.float32)
    keep: list[int] = []
    for roi_id in dict.fromkeys(roi_ids):
        roi_indices = [index for index, value in enumerate(roi_ids) if value == roi_id]
        local_keep = nms(box_tensor[roi_indices], score_tensor[roi_indices], nms_threshold).tolist()
        roi_keep = [roi_indices[index] for index in local_keep]
        keep.extend(
            suppress_same_ant_duplicates(
                boxes=box_tensor,
                keep=roi_keep,
                center_distance_threshold=duplicate_center_distance,
                overlap_threshold=duplicate_overlap_threshold,
            )
        )
    keep = sorted(keep, key=lambda index: scores[index], reverse=True)[:max_detections]
    return [
        Detection(
            bbox=tuple(float(value) for value in box_tensor[index].tolist()),
            score=float(scores[index]),
            appearance=appearance_histogram(frame, tuple(box_tensor[index].tolist())),
            roi_id=roi_ids[index],
        )
        for index in keep
    ]


def suppress_same_ant_duplicates(
    boxes: torch.Tensor,
    keep: list[int],
    center_distance_threshold: float,
    overlap_threshold: float,
) -> list[int]:
    if not keep:
        return []
    selected: list[int] = []
    for candidate_index in keep:
        candidate = tuple(float(value) for value in boxes[candidate_index].tolist())
        if any(
            same_ant_duplicate(
                candidate,
                tuple(float(value) for value in boxes[selected_index].tolist()),
                center_distance_threshold,
                overlap_threshold,
            )
            for selected_index in selected
        ):
            continue
        selected.append(candidate_index)
    return selected


def same_ant_duplicate(
    candidate: Box,
    selected: Box,
    center_distance_threshold: float,
    overlap_threshold: float,
) -> bool:
    if center_distance_threshold > 0 and distance(box_center(candidate), box_center(selected)) <= center_distance_threshold:
        return True
    if overlap_threshold > 0 and intersection_over_smaller_box(candidate, selected) >= overlap_threshold:
        return True
    return False


def intersection_over_smaller_box(left: Box, right: Box) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    smaller = min(left_area, right_area)
    return intersection / smaller if smaller > 0 else 0.0


def resolve_amp_dtype(amp: str, device: torch.device) -> torch.dtype | None:
    if amp == "off":
        return None
    if device.type != "cuda":
        if amp != "auto":
            raise RuntimeError("Mixed-precision inference requires a CUDA device.")
        return None
    if amp == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if amp == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA device does not support bfloat16.")
        return torch.bfloat16
    if amp == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported AMP mode: {amp}")


def assign(costs: np.ndarray, threshold: float) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    unmatched_tracks = set(range(costs.shape[0]))
    unmatched_detections = set(range(costs.shape[1]))
    if costs.size == 0:
        return [], unmatched_tracks, unmatched_detections
    rows, columns = linear_sum_assignment(costs)
    matches = []
    for row, column in zip(rows.tolist(), columns.tolist()):
        if costs[row, column] > threshold:
            continue
        matches.append((row, column))
        unmatched_tracks.discard(row)
        unmatched_detections.discard(column)
    return matches, unmatched_tracks, unmatched_detections


def assignment_margin(costs: np.ndarray, row: int, column: int) -> float:
    value = float(costs[row, column])
    alternatives = [
        *(float(costs[row, index]) for index in range(costs.shape[1]) if index != column),
        *(float(costs[index, column]) for index in range(costs.shape[0]) if index != row),
    ]
    finite = [alternative for alternative in alternatives if alternative < 1e5]
    return min(finite) - value if finite else 1.0


def assignment_risk(
    cost: float,
    margin: float,
    appearance_jump: float,
    jump: float,
    ambiguity_margin: float,
    max_distance: float,
) -> tuple[float, set[str]]:
    risk = min(cost, 1.0) * 0.4
    reasons: set[str] = set()
    if margin < ambiguity_margin:
        reasons.add("ambiguous_assignment")
        risk = max(risk, 0.65 + 0.3 * (1.0 - max(margin, 0.0) / max(ambiguity_margin, 1e-6)))
    if appearance_jump > 0.45:
        reasons.add("appearance_jump")
        risk = max(risk, min(appearance_jump + 0.25, 1.0))
    if jump > max_distance * 0.7:
        reasons.add("large_motion_jump")
        risk = max(risk, min(jump / max(max_distance, 1.0), 1.0))
    return min(risk, 1.0), reasons


def detection_fields() -> list[str]:
    return [
        "frame_index",
        "time_seconds",
        "roi_id",
        "center_x",
        "center_y",
        "x1",
        "y1",
        "x2",
        "y2",
        "score",
    ]


def detection_row(frame_index: int, fps: float, detection: Detection) -> dict[str, object]:
    x1, y1, x2, y2 = detection.bbox
    center_x, center_y = detection.center
    return {
        "frame_index": frame_index,
        "time_seconds": round(frame_index / fps, 6),
        "roi_id": detection.roi_id,
        "center_x": round(center_x, 4),
        "center_y": round(center_y, 4),
        "x1": round(x1, 4),
        "y1": round(y1, 4),
        "x2": round(x2, 4),
        "y2": round(y2, 4),
        "score": round(detection.score, 6),
    }


def track_row(
    frame_index: int,
    track: Track,
    observed: bool,
    ambiguity_margin: float,
    switch_risk: float,
    review_reason: str,
) -> dict[str, object]:
    x1, y1, x2, y2 = track.bbox
    center_x, center_y = track.center
    return {
        "frame_index": frame_index,
        "time_seconds": 0.0,
        "track_id": track.track_id,
        "roi_id": track.last_roi_id,
        "center_x": round(center_x, 4),
        "center_y": round(center_y, 4),
        "x1": round(x1, 4),
        "y1": round(y1, 4),
        "x2": round(x2, 4),
        "y2": round(y2, 4),
        "score": round(track.score, 6),
        "observed": int(observed),
        "track_age": track.age,
        "missed_frames": track.missed,
        "assignment_cost": round(track.last_assignment_cost, 6),
        "ambiguity_margin": round(ambiguity_margin, 6),
        "switch_risk": round(switch_risk, 6),
        "review_reason": review_reason,
    }


def trajectory_fields() -> list[str]:
    return [
        "frame_index",
        "time_seconds",
        "track_id",
        "roi_id",
        "center_x",
        "center_y",
        "x1",
        "y1",
        "x2",
        "y2",
        "score",
        "observed",
        "track_age",
        "missed_frames",
        "assignment_cost",
        "ambiguity_margin",
        "switch_risk",
        "review_reason",
    ]


def write_filtered_trajectories(raw_path: Path, output_path: Path, minimum_hits: int) -> set[int]:
    hits: dict[int, int] = {}
    with raw_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if int(row["observed"]) == 1:
                track_id = int(row["track_id"])
                hits[track_id] = hits.get(track_id, 0) + 1
    valid = {track_id for track_id, count in hits.items() if count >= minimum_hits}
    with raw_path.open("r", newline="", encoding="utf-8-sig") as source, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames or trajectory_fields())
        writer.writeheader()
        for row in reader:
            if int(row["track_id"]) in valid:
                writer.writerow(row)
    return valid


def write_review_outputs(
    output: Path,
    events: dict[int, ReviewEvent],
    valid_ids: set[int],
    threshold: float,
) -> None:
    selected = [
        event
        for event in sorted(events.values(), key=lambda item: item.frame_index)
        if event.risk_score >= threshold and event.track_ids.intersection(valid_ids)
    ]
    fields = ["event_id", "frame_index", "risk_score", "reasons", "track_ids"]
    with (output / "review_events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event_id, event in enumerate(selected, start=1):
            writer.writerow(
                {
                    "event_id": event_id,
                    "frame_index": event.frame_index,
                    "risk_score": round(event.risk_score, 6),
                    "reasons": "|".join(sorted(event.reasons)),
                    "track_ids": "|".join(str(value) for value in sorted(event.track_ids.intersection(valid_ids))),
                }
            )


def open_overlay_targets(
    output: Path,
    fps: float,
    frame_width: int,
    frame_height: int,
    rois: list[RoiRect],
    overlay_mode: str,
) -> list[OverlayTarget]:
    targets: list[tuple[Path, tuple[int, int], RoiRect | None]] = []
    if overlay_mode in {"roi", "both"}:
        for roi in rois:
            path = (
                output / "tracking_overlay.mp4"
                if len(rois) == 1
                else output / f"tracking_overlay_{safe_file_stem(roi.roi_id)}.mp4"
            )
            targets.append((path, (roi.width, roi.height), roi))
    if overlay_mode == "full":
        targets.append((output / "tracking_overlay.mp4", (frame_width, frame_height), None))
    elif overlay_mode == "both":
        targets.append((output / "tracking_overlay_full.mp4", (frame_width, frame_height), None))

    opened: list[OverlayTarget] = []
    for path, (width, height), roi in targets:
        output_width, output_height = even_video_dimensions(width, height)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, output_height))
        if not writer.isOpened():
            for target in opened:
                target.writer.release()
            raise RuntimeError(f"Could not create overlay writer: {path}")
        opened.append(OverlayTarget(path=path, writer=writer, width=output_width, height=output_height, roi=roi))
    return opened


def safe_file_stem(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)
    return cleaned.strip("_") or "roi"


def draw_overlay_frame(
    frame: np.ndarray,
    rows: list[dict[str, object]],
    rois: list[RoiRect],
    target: OverlayTarget,
    event: ReviewEvent | None,
) -> np.ndarray:
    if target.roi is None:
        return fit_overlay_frame(draw_frame(frame, rows, rois, event), target.width, target.height)
    return fit_overlay_frame(draw_roi_frame(frame, rows, target.roi, event), target.width, target.height)


def even_video_dimensions(width: int, height: int) -> tuple[int, int]:
    return (width + width % 2, height + height % 2)


def fit_overlay_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    output = np.zeros((height, width, 3), dtype=frame.dtype)
    copy_height = min(height, frame.shape[0])
    copy_width = min(width, frame.shape[1])
    output[:copy_height, :copy_width] = frame[:copy_height, :copy_width]
    return output


def draw_roi_frame(
    frame: np.ndarray,
    rows: list[dict[str, object]],
    roi: RoiRect,
    event: ReviewEvent | None,
) -> np.ndarray:
    output = roi.crop(frame).copy()
    roi_track_ids: set[int] = set()
    for row in rows:
        if str(row.get("roi_id", "")) != roi.roi_id:
            continue
        track_id = int(row["track_id"])
        roi_track_ids.add(track_id)
        color = id_color(track_id) if int(row["observed"]) else (110, 110, 110)
        x1, y1, x2, y2 = [int(round(float(row[key]))) for key in ("x1", "y1", "x2", "y2")]
        x1 -= roi.x
        x2 -= roi.x
        y1 -= roi.y
        y2 -= roi.y
        if x2 < 0 or y2 < 0 or x1 >= roi.width or y1 >= roi.height:
            continue
        x1 = max(0, min(roi.width - 1, x1))
        x2 = max(0, min(roi.width - 1, x2))
        y1 = max(0, min(roi.height - 1, y1))
        y2 = max(0, min(roi.height - 1, y2))
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)
        cv2.putText(output, str(track_id), (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.putText(output, roi.roi_id, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    if event is not None and event.risk_score >= 0.5 and event.track_ids.intersection(roi_track_ids):
        cv2.putText(
            output,
            f"REVIEW risk={event.risk_score:.2f}",
            (8, min(42, max(18, roi.height - 8))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )
    return output


def draw_frame(
    frame: np.ndarray,
    rows: list[dict[str, object]],
    rois: list[RoiRect],
    event: ReviewEvent | None,
) -> np.ndarray:
    output = draw_rois(frame, rois)
    for row in rows:
        track_id = int(row["track_id"])
        color = id_color(track_id) if int(row["observed"]) else (110, 110, 110)
        x1, y1, x2, y2 = [int(round(float(row[key]))) for key in ("x1", "y1", "x2", "y2")]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)
        cv2.putText(output, str(track_id), (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    if event is not None and event.risk_score >= 0.5:
        cv2.putText(
            output,
            f"REVIEW risk={event.risk_score:.2f}",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    return output


def normalized(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    return values / norm if norm > 1e-12 else values


def near_border(center: tuple[float, float], width: int, height: int, margin: float) -> bool:
    return center[0] <= margin or center[1] <= margin or center[0] >= width - margin or center[1] >= height - margin


def id_color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id * 7919)
    return tuple(int(value) for value in rng.integers(70, 256, size=3))
