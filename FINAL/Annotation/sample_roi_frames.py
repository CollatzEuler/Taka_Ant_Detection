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

from roi_ant_tracker.roi import RoiRect, load_rois


@dataclass(slots=True)
class CandidateFrame:
    candidate_index: int
    video_path: str
    video_stem: str
    roi_json: str | None
    roi_id: str
    roi_label: str
    roi_x: int
    roi_y: int
    source_frame_index: int
    source_time_seconds: float
    width: int
    height: int
    feature: np.ndarray
    frame: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample diverse ROI crops for ant box annotation without preprocessing the video."
    )
    parser.add_argument("--videos", nargs="+", required=True, help="One or more source video paths.")
    parser.add_argument(
        "--roi-jsons",
        nargs="*",
        default=[],
        help=(
            "ROI JSONs from select-roi. With one video, all listed JSONs are sampled. "
            "With many videos, pass one JSON for all videos or one per video. Omit for full-frame samples."
        ),
    )
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--count", type=int, default=300, help="Total number of ROI frames to select.")
    parser.add_argument("--frame-step", type=int, default=90, help="Sample every Nth frame as a candidate.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--start-seconds", type=float, help="Start sampling at this video time.")
    parser.add_argument("--end-seconds", type=float, help="Stop sampling at this video time.")
    parser.add_argument("--duration-seconds", type=float, help="Sample this many seconds from the start time.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print ETA every N sampled source frames. Use 0 to disable.",
    )
    parser.add_argument("--thumbnail-size", type=int, nargs=2, default=(96, 96), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--pca-components", type=int, default=16, help="Maximum PCA components.")
    parser.add_argument("--random-seed", type=int, default=7, help="Random seed for clustering.")
    return parser


def main() -> None:
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    args = build_parser().parse_args()
    run_sampler(args)


def run_sampler(args: argparse.Namespace) -> None:
    started = time.time()
    output_dir = Path(args.output)
    frames_dir = output_dir / "selected_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    video_roi_pairs = expand_video_roi_pairs(args.videos, args.roi_jsons)
    candidates = collect_candidates(
        video_roi_pairs=video_roi_pairs,
        frame_step=args.frame_step,
        thumbnail_size=tuple(args.thumbnail_size),
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        start_seconds=args.start_seconds,
        end_seconds=args.end_seconds,
        duration_seconds=args.duration_seconds,
        progress_every=args.progress_every,
    )
    if not candidates:
        raise RuntimeError("No candidate ROI frames were collected.")

    selected_count = min(args.count, len(candidates))
    embeddings, explained = pca_embeddings(candidates, args.pca_components)
    cluster_ids, distances = cluster_candidates(embeddings, selected_count, args.random_seed)

    rows: list[dict[str, object]] = []
    for candidate, cluster_id, distance_to_center in select_cluster_representatives(candidates, cluster_ids, distances):
        image_name = (
            f"{candidate.video_stem}_{candidate.roi_label}_"
            f"frame_{candidate.source_frame_index:07d}.png"
        )
        image_path = frames_dir / image_name
        cv2.imwrite(str(image_path), candidate.frame)
        rows.append(
            {
                "image_name": image_name,
                "image_path": str(image_path.resolve()),
                "video_path": candidate.video_path,
                "source_frame_index": candidate.source_frame_index,
                "source_time_seconds": round(candidate.source_time_seconds, 6),
                "width": candidate.width,
                "height": candidate.height,
                "roi_json": candidate.roi_json or "",
                "roi_id": candidate.roi_id,
                "roi_x": candidate.roi_x,
                "roi_y": candidate.roi_y,
                "roi_width": candidate.width,
                "roi_height": candidate.height,
                "cluster_id": cluster_id,
                "distance_to_cluster_center": round(float(distance_to_center), 8),
            }
        )

    manifest_path = output_dir / "frame_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "videos": args.videos,
        "roi_jsons": args.roi_jsons,
        "output_dir": str(output_dir.resolve()),
        "selected_frames_dir": str(frames_dir.resolve()),
        "frame_manifest_csv": str(manifest_path.resolve()),
        "candidate_count": len(candidates),
        "selected_count": len(rows),
        "frame_step": args.frame_step,
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "start_seconds": args.start_seconds,
        "end_seconds": args.end_seconds,
        "duration_seconds": args.duration_seconds,
        "thumbnail_size": list(args.thumbnail_size),
        "pca_components_used": int(embeddings.shape[1]),
        "pca_explained_variance_ratio_sum": explained,
        "random_seed": args.random_seed,
    }
    summary_path = output_dir / "sampler_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[sample-roi] candidate_frames={len(candidates)} selected_frames={len(rows)}")
    print(f"[sample-roi] frames={frames_dir}")
    print(f"[sample-roi] manifest={manifest_path}")
    print(f"[sample-roi] summary={summary_path}")
    print(f"[sample-roi] done in {format_seconds(time.time() - started)}")


def expand_video_roi_pairs(videos: list[str], roi_jsons: list[str]) -> list[tuple[str, str | None]]:
    if not roi_jsons:
        return [(video, None) for video in videos]
    if len(videos) == 1:
        return [(videos[0], roi_json) for roi_json in roi_jsons]
    if len(roi_jsons) == 1:
        return [(video, roi_jsons[0]) for video in videos]
    if len(roi_jsons) == len(videos):
        return list(zip(videos, roi_jsons))
    raise ValueError("Pass either one ROI JSON, one ROI JSON per video, or one video with many ROI JSONs.")


def collect_candidates(
    video_roi_pairs: list[tuple[str, str | None]],
    frame_step: int,
    thumbnail_size: tuple[int, int],
    start_frame: int,
    end_frame: int | None,
    start_seconds: float | None,
    end_seconds: float | None,
    duration_seconds: float | None,
    progress_every: int,
) -> list[CandidateFrame]:
    candidates: list[CandidateFrame] = []
    candidate_index = 0
    for video_path, roi_json in video_roi_pairs:
        path = Path(video_path)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rois = load_rois(roi_json, frame_width, frame_height)
        roi_source_label = sanitize_label(Path(roi_json).stem) if roi_json else "full_frame"
        first_frame, final_frame = resolve_frame_window(
            fps=fps,
            total_frames=total_frames,
            start_frame=start_frame,
            end_frame=end_frame,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            duration_seconds=duration_seconds,
        )
        if first_frame >= final_frame:
            capture.release()
            raise ValueError("start-frame must be smaller than end-frame.")
        sampled_source_frames = list(range(first_frame, final_frame, max(1, frame_step)))
        started = time.time()
        print(
            f"[sample-roi] scanning {video_path} frames=[{first_frame}, {final_frame}) "
            f"seconds=[{first_frame / fps:.3f}, {final_frame / fps:.3f}) rois={len(rois)}"
        )
        try:
            for scan_index, frame_index in enumerate(sampled_source_frames, start=1):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    continue
                for roi in rois:
                    roi_frame = roi.crop(frame)
                    height, width = roi_frame.shape[:2]
                    if width <= 0 or height <= 0:
                        continue
                    feature = frame_feature(roi_frame, thumbnail_size)
                    label = sanitize_label(f"{roi_source_label}_{roi.roi_id}")
                    candidates.append(
                        CandidateFrame(
                            candidate_index=candidate_index,
                            video_path=str(path.resolve()),
                            video_stem=sanitize_label(path.stem),
                            roi_json=str(Path(roi_json).resolve()) if roi_json else None,
                            roi_id=roi.roi_id,
                            roi_label=label,
                            roi_x=roi.x,
                            roi_y=roi.y,
                            source_frame_index=frame_index,
                            source_time_seconds=frame_index / fps,
                            width=width,
                            height=height,
                            feature=feature,
                            frame=roi_frame,
                        )
                    )
                    candidate_index += 1
                if progress_every > 0 and (
                    scan_index == 1
                    or scan_index % progress_every == 0
                    or scan_index == len(sampled_source_frames)
                ):
                    report_progress(
                        prefix="[sample-roi]",
                        completed=scan_index,
                        total=len(sampled_source_frames),
                        started=started,
                        detail=f"frame={frame_index} candidates={candidate_index}",
                    )
        finally:
            capture.release()
    return candidates


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


def frame_feature(frame: np.ndarray, thumbnail_size: tuple[int, int]) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, thumbnail_size, interpolation=cv2.INTER_AREA)
    feature = small.astype(np.float32).reshape(-1) / 255.0
    feature -= float(feature.mean())
    std = float(feature.std())
    if std > 1e-6:
        feature /= std
    return feature


def pca_embeddings(candidates: list[CandidateFrame], max_components: int) -> tuple[np.ndarray, float]:
    features = np.stack([candidate.feature for candidate in candidates], axis=0)
    components = max(1, min(max_components, features.shape[0], features.shape[1]))
    if components == 1:
        embedding = features[:, :1]
        explained = 1.0
    else:
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(features, mean=None, maxComponents=components)
        embedding = cv2.PCAProject(features, mean, eigenvectors)
        total_variance = float(np.var(features, axis=0).sum())
        explained = float(eigenvalues.sum() / max(total_variance, 1e-9))
    return embedding, explained


def cluster_candidates(
    embeddings: np.ndarray,
    count: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if count <= 1:
        center = embeddings.mean(axis=0, keepdims=True)
        return np.zeros((embeddings.shape[0],), dtype=np.int32), np.linalg.norm(embeddings - center, axis=1)
    cv2.setRNGSeed(random_seed)
    _compactness, labels, centers = cv2.kmeans(
        embeddings.astype(np.float32),
        count,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4),
        5,
        cv2.KMEANS_PP_CENTERS,
    )
    cluster_ids = labels.reshape(-1)
    differences = embeddings - centers[cluster_ids]
    distances = np.sqrt(np.sum(differences * differences, axis=1))
    return cluster_ids, distances


def select_cluster_representatives(
    candidates: list[CandidateFrame],
    cluster_ids: np.ndarray,
    distances: np.ndarray,
) -> list[tuple[CandidateFrame, int, float]]:
    by_cluster: dict[int, list[tuple[CandidateFrame, float]]] = {}
    for candidate, cluster_id, distance_to_center in zip(candidates, cluster_ids.tolist(), distances.tolist()):
        by_cluster.setdefault(cluster_id, []).append((candidate, float(distance_to_center)))
    selected: list[tuple[CandidateFrame, int, float]] = []
    for cluster_id in sorted(by_cluster):
        candidate, distance_to_center = min(by_cluster[cluster_id], key=lambda item: item[1])
        selected.append((candidate, cluster_id, distance_to_center))
    selected.sort(key=lambda item: (Path(item[0].video_path).name, item[0].roi_label, item[0].source_frame_index))
    return selected


def sanitize_label(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text.strip("_") or "item"


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


if __name__ == "__main__":
    main()
