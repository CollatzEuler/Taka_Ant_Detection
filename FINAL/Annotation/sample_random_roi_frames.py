from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True)
class PatchCandidate:
    candidate_id: int
    selection_group: str
    frame_index: int
    time_seconds: float
    x: int
    y: int
    width: int
    height: int
    feature: np.ndarray
    motion_score: float
    contrast_score: float
    sharpness_score: float
    exposure_score: float
    excluded_fraction: float
    proposal_mode: str
    quality_score: float = 0.0
    cluster_id: int = -1
    distance_to_cluster_center: float = 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select fast, diverse annotation crops from random times and random allowed locations. "
            "Crop sizes stay near reference ROIs; excluded mask pixels are avoided."
        )
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-roi-json", required=True)
    parser.add_argument("--exclusion-json", help="JSON from select-exclusions. Omit to allow the full frame.")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument(
        "--beginning-count",
        type=int,
        help="Selections reserved for the beginning window. Default: one third of count.",
    )
    parser.add_argument("--beginning-seconds", type=float, default=600.0)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--end-seconds", type=float)
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=8,
        help="Candidate crops per final crop. Higher improves diversity but costs more decoding.",
    )
    parser.add_argument("--patches-per-frame", type=int, default=8)
    parser.add_argument(
        "--size-jitter",
        type=float,
        default=0.08,
        help="Maximum proportional size change from a randomly chosen reference ROI.",
    )
    parser.add_argument("--max-excluded-fraction", type=float, default=0.01)
    parser.add_argument("--motion-gap-frames", type=int, default=6)
    parser.add_argument(
        "--motion-guided-fraction",
        type=float,
        default=0.50,
        help="Fraction of candidates centered on strong localized motion; the rest use random allowed positions.",
    )
    parser.add_argument("--thumbnail-size", type=int, nargs=2, default=(40, 40), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--pca-components", type=int, default=12)
    parser.add_argument(
        "--quality-weight",
        type=float,
        default=0.35,
        help="Preference for motion, focus, contrast, and usable exposure within each PCA cluster.",
    )
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def main() -> None:
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    run_sampler(build_parser().parse_args())


def run_sampler(args: argparse.Namespace) -> None:
    validate_args(args)
    started = time.time()
    output = Path(args.output)
    frames_dir = output / "selected_frames"
    output.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    video_path = Path(args.video)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    first_frame = max(0, min(total_frames, int(round(args.start_seconds * fps))))
    final_frame = (
        max(0, min(total_frames, int(round(args.end_seconds * fps))))
        if args.end_seconds is not None
        else total_frames
    )
    if first_frame >= final_frame:
        raise ValueError("The selected video time window is empty.")
    beginning_final = min(final_frame, first_frame + int(round(args.beginning_seconds * fps)))
    beginning_count = args.beginning_count if args.beginning_count is not None else args.count // 3
    beginning_count = max(0, min(args.count, beginning_count))
    random_count = args.count - beginning_count
    reference_sizes = load_reference_sizes(args.reference_roi_json)
    validate_reference_sizes(reference_sizes, frame_width, frame_height)
    allowed_mask = load_allowed_mask(args.exclusion_json, frame_width, frame_height)
    blocked_integral = cv2.integral((allowed_mask == 0).astype(np.uint8))

    pools: list[tuple[str, int, int, int]] = []
    if beginning_count > 0:
        pools.append(("beginning", beginning_count, first_frame, beginning_final))
    if random_count > 0:
        random_first = beginning_final if beginning_final < final_frame else first_frame
        pools.append(("random", random_count, random_first, final_frame))

    selected: list[PatchCandidate] = []
    selection_stats: dict[str, Any] = {}
    next_candidate_id = 1
    for pool_index, (group, selection_count, pool_first, pool_final) in enumerate(pools):
        target_candidates = selection_count * args.candidate_multiplier
        candidates = collect_candidates(
            video_path=video_path,
            fps=fps,
            first_frame=pool_first,
            final_frame=pool_final,
            target_candidates=target_candidates,
            patches_per_frame=args.patches_per_frame,
            reference_sizes=reference_sizes,
            size_jitter=args.size_jitter,
            max_excluded_fraction=args.max_excluded_fraction,
            blocked_integral=blocked_integral,
            allowed_mask=allowed_mask,
            thumbnail_size=tuple(args.thumbnail_size),
            motion_gap_frames=args.motion_gap_frames,
            motion_guided_fraction=args.motion_guided_fraction,
            rng=rng,
            group=group,
            first_candidate_id=next_candidate_id,
            progress_every=args.progress_every,
        )
        if len(candidates) < selection_count:
            raise RuntimeError(
                f"Only {len(candidates)} valid {group} candidates were found for {selection_count} selections. "
                "Paint a larger allowed area, increase max-excluded-fraction, or reduce count."
            )
        next_candidate_id += len(candidates)
        assign_quality_scores(candidates)
        chosen, explained = select_diverse_candidates(
            candidates,
            count=selection_count,
            max_components=args.pca_components,
            quality_weight=args.quality_weight,
            random_seed=args.random_seed + pool_index,
        )
        selected.extend(chosen)
        selection_stats[group] = {
            "candidate_count": len(candidates),
            "selected_count": len(chosen),
            "frame_range": [pool_first, pool_final],
            "time_range_seconds": [pool_first / fps, pool_final / fps],
            "pca_components_used": min(args.pca_components, len(candidates), candidates[0].feature.size),
            "pca_explained_variance_ratio_sum": explained,
        }

    rows = export_selected_crops(
        video_path=video_path,
        fps=fps,
        selected=selected,
        frames_dir=frames_dir,
        reference_roi_json=args.reference_roi_json,
        exclusion_json=args.exclusion_json,
    )
    if not rows:
        raise RuntimeError("The selected source frames could not be decoded.")
    manifest_path = output / "frame_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    preview_path = output / "selection_preview.jpg"
    write_contact_sheet(rows, preview_path)

    sizes = [(row["roi_width"], row["roi_height"]) for row in rows]
    summary = {
        "sampler": "random_roi_pca_motion_v1",
        "video": str(video_path.resolve()),
        "reference_roi_json": str(Path(args.reference_roi_json).resolve()),
        "exclusion_json": str(Path(args.exclusion_json).resolve()) if args.exclusion_json else None,
        "output_dir": str(output.resolve()),
        "selected_frames_dir": str(frames_dir.resolve()),
        "frame_manifest_csv": str(manifest_path.resolve()),
        "selection_preview": str(preview_path.resolve()),
        "reported_total_frames": total_frames,
        "fps": fps,
        "selected_time_window_seconds": [first_frame / fps, final_frame / fps],
        "beginning_seconds": args.beginning_seconds,
        "requested_count": args.count,
        "exported_count": len(rows),
        "beginning_count": sum(row["selection_group"] == "beginning" for row in rows),
        "random_count": sum(row["selection_group"] == "random" for row in rows),
        "reference_sizes": [list(size) for size in reference_sizes],
        "selected_width_range": [min(size[0] for size in sizes), max(size[0] for size in sizes)],
        "selected_height_range": [min(size[1] for size in sizes), max(size[1] for size in sizes)],
        "size_jitter": args.size_jitter,
        "max_excluded_fraction": args.max_excluded_fraction,
        "motion_gap_frames": args.motion_gap_frames,
        "motion_guided_fraction": args.motion_guided_fraction,
        "candidate_multiplier": args.candidate_multiplier,
        "patches_per_frame": args.patches_per_frame,
        "thumbnail_size": list(args.thumbnail_size),
        "pca_components": args.pca_components,
        "quality_weight": args.quality_weight,
        "random_seed": args.random_seed,
        "selection_groups": selection_stats,
        "seconds": time.time() - started,
    }
    summary_path = output / "sampler_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[sample-random-roi] candidates={sum(value['candidate_count'] for value in selection_stats.values())} "
        f"selected={len(rows)} beginning={summary['beginning_count']} random={summary['random_count']}"
    )
    print(
        f"[sample-random-roi] size_range={summary['selected_width_range']}x"
        f"{summary['selected_height_range']} seconds={summary['seconds']:.1f}"
    )
    print(f"[sample-random-roi] frames={frames_dir}")
    print(f"[sample-random-roi] manifest={manifest_path}")
    print(f"[sample-random-roi] preview={preview_path}")


def validate_args(args: argparse.Namespace) -> None:
    if args.count <= 0:
        raise ValueError("count must be positive.")
    if args.beginning_count is not None and not 0 <= args.beginning_count <= args.count:
        raise ValueError("beginning-count must be between 0 and count.")
    if args.beginning_seconds <= 0:
        raise ValueError("beginning-seconds must be positive.")
    if args.candidate_multiplier < 1 or args.patches_per_frame < 1:
        raise ValueError("candidate-multiplier and patches-per-frame must be positive.")
    if not 0.0 <= args.size_jitter <= 0.25:
        raise ValueError("size-jitter must be in [0, 0.25].")
    if not 0.0 <= args.max_excluded_fraction <= 1.0:
        raise ValueError("max-excluded-fraction must be in [0, 1].")
    if args.motion_gap_frames < 0:
        raise ValueError("motion-gap-frames must be non-negative.")
    if not 0.0 <= args.motion_guided_fraction <= 1.0:
        raise ValueError("motion-guided-fraction must be in [0, 1].")
    if args.pca_components <= 0:
        raise ValueError("pca-components must be positive.")
    if args.quality_weight < 0:
        raise ValueError("quality-weight must be non-negative.")


def load_reference_sizes(path: str | Path) -> list[tuple[int, int]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    rectangles = payload.get("rectangles", [])
    sizes = [(int(item["width"]), int(item["height"])) for item in rectangles]
    if not sizes:
        raise ValueError(f"Reference ROI JSON has no rectangles: {path}")
    return sizes


def validate_reference_sizes(sizes: list[tuple[int, int]], frame_width: int, frame_height: int) -> None:
    for width, height in sizes:
        if width <= 1 or height <= 1:
            raise ValueError("Reference ROI sizes must be larger than one pixel.")
        if width > frame_width or height > frame_height:
            raise ValueError(
                f"Reference ROI {width}x{height} is larger than the source video {frame_width}x{frame_height}."
            )


def load_allowed_mask(path: str | None, frame_width: int, frame_height: int) -> np.ndarray:
    if path is None:
        return np.full((frame_height, frame_width), 255, dtype=np.uint8)
    json_path = Path(path)
    payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
    mask_path = Path(payload["mask_path"])
    if not mask_path.is_absolute():
        mask_path = json_path.parent / mask_path
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read exclusion mask: {mask_path}")
    if mask.shape != (frame_height, frame_width):
        raise ValueError(
            f"Exclusion mask is {mask.shape[1]}x{mask.shape[0]}, but video is {frame_width}x{frame_height}."
        )
    return np.where(mask >= 128, 255, 0).astype(np.uint8)


def collect_candidates(
    video_path: Path,
    fps: float,
    first_frame: int,
    final_frame: int,
    target_candidates: int,
    patches_per_frame: int,
    reference_sizes: list[tuple[int, int]],
    size_jitter: float,
    max_excluded_fraction: float,
    blocked_integral: np.ndarray,
    allowed_mask: np.ndarray,
    thumbnail_size: tuple[int, int],
    motion_gap_frames: int,
    motion_guided_fraction: float,
    rng: np.random.Generator,
    group: str,
    first_candidate_id: int,
    progress_every: int,
) -> list[PatchCandidate]:
    if first_frame >= final_frame:
        return []
    requested_frame_count = max(1, math.ceil(target_candidates / patches_per_frame * 1.25))
    frame_indices = stratified_random_frames(first_frame, final_frame, requested_frame_count, rng)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    candidates: list[PatchCandidate] = []
    started = time.time()
    try:
        for scan_index, frame_index in enumerate(frame_indices, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            future = read_future_frame(capture, frame, frame_index, final_frame, motion_gap_frames)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            future_gray = cv2.cvtColor(future, cv2.COLOR_BGR2GRAY)
            motion_points = localized_motion_points(frame_gray, future_gray, allowed_mask)
            accepted_boxes: list[tuple[int, int, int, int]] = []
            attempts = 0
            while len(accepted_boxes) < patches_per_frame and attempts < patches_per_frame * 40:
                attempts += 1
                width, height = random_roi_size(reference_sizes, size_jitter, frame.shape[1], frame.shape[0], rng)
                proposal_mode = "motion" if len(motion_points) > 0 and rng.random() < motion_guided_fraction else "random"
                if proposal_mode == "motion":
                    center_x, center_y = motion_points[int(rng.integers(0, len(motion_points)))]
                    x = max(0, min(frame.shape[1] - width, int(round(center_x - width / 2))))
                    y = max(0, min(frame.shape[0] - height, int(round(center_y - height / 2))))
                else:
                    x = int(rng.integers(0, frame.shape[1] - width + 1))
                    y = int(rng.integers(0, frame.shape[0] - height + 1))
                excluded_fraction = rectangle_sum(blocked_integral, x, y, width, height) / float(width * height)
                if excluded_fraction > max_excluded_fraction:
                    continue
                box = (x, y, width, height)
                if any(rectangle_iou(box, other) > 0.70 for other in accepted_boxes):
                    continue
                patch = frame_gray[y : y + height, x : x + width]
                future_patch = future_gray[y : y + height, x : x + width]
                feature = patch_feature(patch, future_patch, thumbnail_size)
                motion, contrast, sharpness, exposure = patch_quality_metrics(patch, future_patch)
                candidates.append(
                    PatchCandidate(
                        candidate_id=first_candidate_id + len(candidates),
                        selection_group=group,
                        frame_index=frame_index,
                        time_seconds=frame_index / fps,
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                        feature=feature,
                        motion_score=motion,
                        contrast_score=contrast,
                        sharpness_score=sharpness,
                        exposure_score=exposure,
                        excluded_fraction=excluded_fraction,
                        proposal_mode=proposal_mode,
                    )
                )
                accepted_boxes.append(box)
            if progress_every > 0 and (
                scan_index == 1 or scan_index % progress_every == 0 or scan_index == len(frame_indices)
            ):
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"[sample-random-roi] group={group} frames={scan_index}/{len(frame_indices)} "
                    f"candidates={len(candidates)}/{target_candidates} rate={scan_index / elapsed:.2f} frames/s"
                )
            if len(candidates) >= target_candidates:
                break
    finally:
        capture.release()
    return candidates[:target_candidates]


def read_future_frame(
    capture: cv2.VideoCapture,
    current: np.ndarray,
    frame_index: int,
    final_frame: int,
    motion_gap_frames: int,
) -> np.ndarray:
    usable_gap = min(motion_gap_frames, max(0, final_frame - frame_index - 1))
    if usable_gap <= 0:
        return current
    for _ in range(max(0, usable_gap - 1)):
        if not capture.grab():
            return current
    ok, future = capture.read()
    return future if ok else current


def stratified_random_frames(
    first_frame: int,
    final_frame: int,
    count: int,
    rng: np.random.Generator,
) -> list[int]:
    length = final_frame - first_frame
    count = max(1, min(count, length))
    edges = np.linspace(first_frame, final_frame, count + 1, dtype=np.int64)
    frames: list[int] = []
    for left, right in zip(edges[:-1], edges[1:]):
        upper = max(int(left) + 1, int(right))
        frames.append(int(rng.integers(int(left), upper)))
    return sorted(set(min(final_frame - 1, frame) for frame in frames))


def random_roi_size(
    reference_sizes: list[tuple[int, int]],
    size_jitter: float,
    frame_width: int,
    frame_height: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    reference_width, reference_height = reference_sizes[int(rng.integers(0, len(reference_sizes)))]
    scale = float(rng.uniform(1.0 - size_jitter, 1.0 + size_jitter))
    width = max(2, min(frame_width, int(round(reference_width * scale))))
    height = max(2, min(frame_height, int(round(reference_height * scale))))
    return width, height


def localized_motion_points(
    current: np.ndarray,
    future: np.ndarray,
    allowed_mask: np.ndarray,
    downsample: int = 8,
) -> np.ndarray:
    difference = cv2.absdiff(current, future)
    small_width = max(1, current.shape[1] // downsample)
    small_height = max(1, current.shape[0] // downsample)
    small_difference = cv2.resize(difference, (small_width, small_height), interpolation=cv2.INTER_AREA)
    small_allowed = cv2.resize(allowed_mask, (small_width, small_height), interpolation=cv2.INTER_NEAREST) > 0
    values = small_difference[small_allowed]
    if values.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    threshold = float(np.percentile(values, 99.5))
    if threshold <= 0:
        return np.empty((0, 2), dtype=np.float32)
    rows, columns = np.where(small_allowed & (small_difference >= threshold))
    if len(rows) == 0:
        return np.empty((0, 2), dtype=np.float32)
    scale_x = current.shape[1] / small_width
    scale_y = current.shape[0] / small_height
    return np.column_stack(((columns + 0.5) * scale_x, (rows + 0.5) * scale_y)).astype(np.float32)


def rectangle_sum(integral: np.ndarray, x: int, y: int, width: int, height: int) -> int:
    x2, y2 = x + width, y + height
    return int(integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x])


def rectangle_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    intersection = max(0, min(lx + lw, rx + rw) - max(lx, rx)) * max(
        0, min(ly + lh, ry + rh) - max(ly, ry)
    )
    union = lw * lh + rw * rh - intersection
    return intersection / union if union else 0.0


def patch_feature(current: np.ndarray, future: np.ndarray, thumbnail_size: tuple[int, int]) -> np.ndarray:
    current_small = cv2.resize(current, thumbnail_size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    motion_small = cv2.resize(cv2.absdiff(current, future), thumbnail_size, interpolation=cv2.INTER_AREA).astype(
        np.float32
    ) / 255.0
    current_values = current_small.reshape(-1)
    current_values -= float(current_values.mean())
    standard_deviation = float(current_values.std())
    if standard_deviation > 1e-6:
        current_values /= standard_deviation
    return np.concatenate((current_values, motion_small.reshape(-1) * 2.0)).astype(np.float32)


def patch_quality_metrics(current: np.ndarray, future: np.ndarray) -> tuple[float, float, float, float]:
    difference = cv2.absdiff(current, future).reshape(-1).astype(np.float32)
    top_count = max(20, difference.size // 100)
    motion = float(np.partition(difference, difference.size - top_count)[-top_count:].mean() / 255.0)
    contrast = float(current.std())
    sharpness = float(cv2.Laplacian(current, cv2.CV_32F).var())
    unusable = float(np.mean((current <= 5) | (current >= 250)))
    return motion, contrast, sharpness, 1.0 - unusable


def assign_quality_scores(candidates: list[PatchCandidate]) -> None:
    motion = percentile_ranks([candidate.motion_score for candidate in candidates])
    contrast = percentile_ranks([candidate.contrast_score for candidate in candidates])
    sharpness = percentile_ranks([candidate.sharpness_score for candidate in candidates])
    exposure = percentile_ranks([candidate.exposure_score for candidate in candidates])
    for index, candidate in enumerate(candidates):
        candidate.quality_score = float(
            0.55 * motion[index] + 0.15 * contrast[index] + 0.15 * sharpness[index] + 0.15 * exposure[index]
        )


def percentile_ranks(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if len(array) <= 1:
        return np.ones((len(array),), dtype=np.float64)
    order = np.argsort(array, kind="stable")
    ranks = np.empty_like(array)
    ranks[order] = np.linspace(0.0, 1.0, len(array))
    return ranks


def select_diverse_candidates(
    candidates: list[PatchCandidate],
    count: int,
    max_components: int,
    quality_weight: float,
    random_seed: int,
) -> tuple[list[PatchCandidate], float]:
    if count >= len(candidates):
        for cluster_id, candidate in enumerate(candidates):
            candidate.cluster_id = cluster_id
        return list(candidates), 1.0
    features = np.stack([candidate.feature for candidate in candidates], axis=0)
    components = max(1, min(max_components, features.shape[0] - 1, features.shape[1]))
    if components == 1:
        embeddings = features[:, :1]
        explained = 1.0
    else:
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(features, mean=None, maxComponents=components)
        embeddings = cv2.PCAProject(features, mean, eigenvectors)
        total_variance = float(np.var(features, axis=0).sum())
        explained = float(eigenvalues.sum() / max(total_variance, 1e-9))
    cv2.setRNGSeed(random_seed)
    _compactness, labels, centers = cv2.kmeans(
        embeddings.astype(np.float32),
        count,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4),
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    cluster_ids = labels.reshape(-1)
    differences = embeddings - centers[cluster_ids]
    distances = np.sqrt(np.sum(differences * differences, axis=1))
    minimum, maximum = float(distances.min()), float(distances.max())
    normalized_distances = (distances - minimum) / max(maximum - minimum, 1e-9)
    selected_indices: list[int] = []
    for cluster_id in range(count):
        indices = np.where(cluster_ids == cluster_id)[0]
        if len(indices) == 0:
            continue
        best = min(
            indices.tolist(),
            key=lambda index: normalized_distances[index] - quality_weight * candidates[index].quality_score,
        )
        candidates[best].cluster_id = cluster_id
        candidates[best].distance_to_cluster_center = float(distances[best])
        selected_indices.append(best)
    if len(selected_indices) < count:
        remaining = sorted(
            (index for index in range(len(candidates)) if index not in set(selected_indices)),
            key=lambda index: candidates[index].quality_score,
            reverse=True,
        )
        selected_indices.extend(remaining[: count - len(selected_indices)])
    return [candidates[index] for index in selected_indices[:count]], explained


def export_selected_crops(
    video_path: Path,
    fps: float,
    selected: list[PatchCandidate],
    frames_dir: Path,
    reference_roi_json: str,
    exclusion_json: str | None,
) -> list[dict[str, object]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen video: {video_path}")
    rows: list[dict[str, object]] = []
    video_stem = sanitize_label(video_path.stem)
    try:
        for output_index, candidate in enumerate(
            sorted(selected, key=lambda item: (item.frame_index, item.selection_group, item.candidate_id)), start=1
        ):
            capture.set(cv2.CAP_PROP_POS_FRAMES, candidate.frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            crop = frame[candidate.y : candidate.y + candidate.height, candidate.x : candidate.x + candidate.width]
            roi_id = f"random_roi_{output_index:04d}"
            image_name = (
                f"{video_stem}_{candidate.selection_group}_{roi_id}_frame_{candidate.frame_index:09d}_"
                f"x{candidate.x}_y{candidate.y}.png"
            )
            image_path = frames_dir / image_name
            if not cv2.imwrite(str(image_path), crop):
                continue
            rows.append(
                {
                    "image_name": image_name,
                    "image_path": str(image_path.resolve()),
                    "video_path": str(video_path.resolve()),
                    "source_frame_index": candidate.frame_index,
                    "source_time_seconds": round(candidate.frame_index / fps, 6),
                    "width": candidate.width,
                    "height": candidate.height,
                    "roi_json": "",
                    "roi_id": roi_id,
                    "roi_x": candidate.x,
                    "roi_y": candidate.y,
                    "roi_width": candidate.width,
                    "roi_height": candidate.height,
                    "selection_group": candidate.selection_group,
                    "reference_roi_json": str(Path(reference_roi_json).resolve()),
                    "exclusion_json": str(Path(exclusion_json).resolve()) if exclusion_json else "",
                    "candidate_id": candidate.candidate_id,
                    "quality_score": round(candidate.quality_score, 8),
                    "motion_score": round(candidate.motion_score, 8),
                    "contrast_score": round(candidate.contrast_score, 8),
                    "sharpness_score": round(candidate.sharpness_score, 8),
                    "exposure_score": round(candidate.exposure_score, 8),
                    "excluded_fraction": round(candidate.excluded_fraction, 8),
                    "proposal_mode": candidate.proposal_mode,
                    "cluster_id": candidate.cluster_id,
                    "distance_to_cluster_center": round(candidate.distance_to_cluster_center, 8),
                }
            )
    finally:
        capture.release()
    return rows


def write_contact_sheet(rows: list[dict[str, object]], output_path: Path, maximum: int = 64) -> None:
    selected_rows = rows[:maximum]
    if not selected_rows:
        return
    tile_width, tile_height = 180, 165
    columns = min(8, max(1, math.ceil(math.sqrt(len(selected_rows)))))
    rows_count = math.ceil(len(selected_rows) / columns)
    sheet = np.zeros((rows_count * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, row in enumerate(selected_rows):
        image = cv2.imread(str(row["image_path"]))
        if image is None:
            continue
        image = fit_thumbnail(image, tile_width, tile_height - 28)
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        sheet[y : y + image.shape[0], x : x + image.shape[1]] = image
        label = f"{row['selection_group'][0]} t={float(row['source_time_seconds']):.1f} q={float(row['quality_score']):.2f}"
        cv2.putText(sheet, label, (x + 3, y + tile_height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
    cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def fit_thumbnail(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    output = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    output[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return output


def sanitize_label(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text.strip("_") or "item"


if __name__ == "__main__":
    main()
