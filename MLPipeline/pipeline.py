from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent

TOOLS: dict[str, list[str]] = {
    "select-roi": [str(PIPELINE_ROOT / "Preprocessing" / "roi_select.py"), "select"],
    "middle-frame": [str(PIPELINE_ROOT / "Preprocessing" / "roi_select.py"), "middle-frame"],
    "preview-roi": [str(PIPELINE_ROOT / "Preprocessing" / "roi_select.py"), "preview"],
    "select-exclusions": [str(PIPELINE_ROOT / "Preprocessing" / "exclusion_select.py"), "select"],
    "preview-exclusions": [str(PIPELINE_ROOT / "Preprocessing" / "exclusion_select.py"), "preview"],
    "sample-roi": [str(PIPELINE_ROOT / "Annotation" / "sample_roi_frames.py")],
    "sample-random-roi": [str(PIPELINE_ROOT / "Annotation" / "sample_random_roi_frames.py")],
    "sample-suggested": [str(PIPELINE_ROOT / "Annotation" / "sample_suggested_boxes.py")],
    "annotate": [str(PIPELINE_ROOT / "Annotation" / "annotate_boxes.py")],
    "export-coco": [str(PIPELINE_ROOT / "Annotation" / "export_coco.py")],
    "merge-coco": [str(PIPELINE_ROOT / "Annotation" / "merge_coco_labels.py")],
    "split-coco": [str(PIPELINE_ROOT / "Annotation" / "split_coco.py")],
    "detector": ["-m", "roi_ant_tracker.cli"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ROI-first ant ML pipeline: ROI selection, annotation, training, and tracking."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train-detector", help="Train the ROI ant detector.")
    train.add_argument("--images", required=True)
    train.add_argument("--annotations", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--architecture", choices=["mobilenet_v3_fpn", "resnet50_fpn_v2"], default="mobilenet_v3_fpn")
    train.add_argument("--min-size", type=int, default=768)
    train.add_argument("--max-size", type=int, default=1280)
    train.add_argument("--initialization", choices=["detector", "backbone", "random"], default="detector")
    train.add_argument("--resume-checkpoint")
    train.add_argument("--reset-training-history", action="store_true")
    train.add_argument("--box-detections-per-img", type=int, default=300)
    train.add_argument("--lr-patience", type=int, default=3)
    train.add_argument("--lr-factor", type=float, default=0.5)
    train.add_argument("--min-learning-rate", type=float, default=1e-7)
    train.add_argument("--plateau-delta", type=float, default=0.001)
    train.add_argument("--log-every", type=int, default=10)
    train.add_argument("--max-grad-norm", type=float, default=1.0)
    train.add_argument("--max-empty-fraction", type=float, default=1.0)
    train.add_argument("--device", default="auto")
    train.add_argument("--workers", type=int, default=0)

    track = commands.add_parser("track-video", help="Track ants by searching only inside saved ROIs.")
    track.add_argument("--video", required=True)
    track.add_argument("--checkpoint", required=True)
    track.add_argument("--output", required=True)
    track.add_argument("--roi-json")
    track.add_argument("--device", default="auto")
    track.add_argument("--score-threshold", type=float, default=0.30)
    track.add_argument("--nms-threshold", type=float, default=0.70)
    track.add_argument("--tile-size", type=int, default=0)
    track.add_argument("--tile-overlap", type=int, default=64)
    track.add_argument("--tile-batch-size", type=int, default=4)
    track.add_argument("--frame-batch-size", type=int, default=1)
    track.add_argument("--frame-step", type=int, default=1)
    track.add_argument("--roi-padding", type=int, default=0)
    track.add_argument("--amp", choices=["auto", "off", "fp16", "bf16"], default="auto")
    track.add_argument("--no-overlay", action="store_true")
    track.add_argument("--overlay-mode", choices=["roi", "full", "both"], default="roi")
    track.add_argument("--inference-min-size", type=int)
    track.add_argument("--inference-max-size", type=int)
    track.add_argument("--max-detections", type=int, default=300)
    track.add_argument("--duplicate-center-distance", type=float, default=0.0)
    track.add_argument("--duplicate-overlap-threshold", type=float, default=0.80)
    track.add_argument("--max-distance", type=float, default=60.0)
    track.add_argument("--max-missed", type=int, default=5)
    track.add_argument("--match-threshold", type=float, default=0.80)
    track.add_argument("--ambiguity-margin", type=float, default=0.12)
    track.add_argument("--close-distance", type=float, default=80.0)
    track.add_argument("--border-margin", type=float, default=80.0)
    track.add_argument("--minimum-track-hits", type=int, default=3)
    track.add_argument("--review-threshold", type=float, default=0.60)
    track.add_argument("--start-frame", type=int, default=0)
    track.add_argument("--end-frame", type=int)
    track.add_argument("--start-seconds", type=float)
    track.add_argument("--end-seconds", type=float)
    track.add_argument("--duration-seconds", type=float)
    track.add_argument("--progress-every", type=int, default=100)

    evaluate = commands.add_parser("evaluate-detector", help="Evaluate the detector against COCO ant boxes.")
    evaluate.add_argument("--images", required=True)
    evaluate.add_argument("--annotations", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--batch-size", type=int, default=2)
    evaluate.add_argument("--workers", type=int, default=0)
    evaluate.add_argument("--amp", choices=["auto", "off", "fp16", "bf16"], default="auto")
    evaluate.add_argument("--score-threshold", type=float, default=0.30)
    evaluate.add_argument("--candidate-score-threshold", type=float, default=0.05)
    evaluate.add_argument("--iou-threshold", type=float, default=0.50)
    evaluate.add_argument("--center-distance-threshold", type=float, default=12.0)
    evaluate.add_argument("--inference-min-size", type=int)
    evaluate.add_argument("--inference-max-size", type=int)
    evaluate.add_argument("--progress-every", type=int, default=25)

    for name in TOOLS:
        tool = commands.add_parser(name, help=f"Forward arguments to the {name} stage.")
        tool.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train-detector":
        run_python(
            [
                "-m",
                "roi_ant_tracker.cli",
                "train",
                "--images",
                str(Path(args.images).resolve()),
                "--annotations",
                str(Path(args.annotations).resolve()),
                "--output",
                str(Path(args.output).resolve()),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--learning-rate",
                str(args.learning_rate),
                "--architecture",
                args.architecture,
                "--min-size",
                str(args.min_size),
                "--max-size",
                str(args.max_size),
                "--initialization",
                args.initialization,
                "--box-detections-per-img",
                str(args.box_detections_per_img),
                "--lr-patience",
                str(args.lr_patience),
                "--lr-factor",
                str(args.lr_factor),
                "--min-learning-rate",
                str(args.min_learning_rate),
                "--plateau-delta",
                str(args.plateau_delta),
                "--log-every",
                str(args.log_every),
                "--max-grad-norm",
                str(args.max_grad_norm),
                "--max-empty-fraction",
                str(args.max_empty_fraction),
                "--device",
                args.device,
                "--workers",
                str(args.workers),
                *optional_arg("--resume-checkpoint", args.resume_checkpoint),
                *(["--reset-training-history"] if args.reset_training_history else []),
            ]
        )
    elif args.command == "track-video":
        run_python(
            [
                "-m",
                "roi_ant_tracker.cli",
                "track",
                "--video",
                str(Path(args.video).resolve()),
                "--checkpoint",
                str(Path(args.checkpoint).resolve()),
                "--output",
                str(Path(args.output).resolve()),
                "--device",
                args.device,
                "--score-threshold",
                str(args.score_threshold),
                "--nms-threshold",
                str(args.nms_threshold),
                "--tile-size",
                str(args.tile_size),
                "--tile-overlap",
                str(args.tile_overlap),
                "--tile-batch-size",
                str(args.tile_batch_size),
                "--frame-batch-size",
                str(args.frame_batch_size),
                "--frame-step",
                str(args.frame_step),
                "--roi-padding",
                str(args.roi_padding),
                "--amp",
                args.amp,
                "--overlay-mode",
                args.overlay_mode,
                "--max-detections",
                str(args.max_detections),
                "--duplicate-center-distance",
                str(args.duplicate_center_distance),
                "--duplicate-overlap-threshold",
                str(args.duplicate_overlap_threshold),
                "--max-distance",
                str(args.max_distance),
                "--max-missed",
                str(args.max_missed),
                "--match-threshold",
                str(args.match_threshold),
                "--ambiguity-margin",
                str(args.ambiguity_margin),
                "--close-distance",
                str(args.close_distance),
                "--border-margin",
                str(args.border_margin),
                "--minimum-track-hits",
                str(args.minimum_track_hits),
                "--review-threshold",
                str(args.review_threshold),
                "--start-frame",
                str(args.start_frame),
                "--progress-every",
                str(args.progress_every),
                *optional_arg("--roi-json", args.roi_json),
                *optional_arg("--end-frame", args.end_frame),
                *optional_arg("--start-seconds", args.start_seconds),
                *optional_arg("--end-seconds", args.end_seconds),
                *optional_arg("--duration-seconds", args.duration_seconds),
                *optional_arg("--inference-min-size", args.inference_min_size),
                *optional_arg("--inference-max-size", args.inference_max_size),
                *(["--no-overlay"] if args.no_overlay else []),
            ]
        )
    elif args.command == "evaluate-detector":
        run_python(
            [
                "-m",
                "roi_ant_tracker.cli",
                "evaluate",
                "--images",
                str(Path(args.images).resolve()),
                "--annotations",
                str(Path(args.annotations).resolve()),
                "--checkpoint",
                str(Path(args.checkpoint).resolve()),
                "--output",
                str(Path(args.output).resolve()),
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--workers",
                str(args.workers),
                "--amp",
                args.amp,
                "--score-threshold",
                str(args.score_threshold),
                "--candidate-score-threshold",
                str(args.candidate_score_threshold),
                "--iou-threshold",
                str(args.iou_threshold),
                "--center-distance-threshold",
                str(args.center_distance_threshold),
                "--progress-every",
                str(args.progress_every),
                *optional_arg("--inference-min-size", args.inference_min_size),
                *optional_arg("--inference-max-size", args.inference_max_size),
            ]
        )
    else:
        forwarded = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
        run_python([*TOOLS[args.command], *forwarded])


def optional_arg(name: str, value: object | None) -> list[str]:
    if value is None:
        return []
    return [name, str(value)]


def run_python(arguments: list[str]) -> None:
    environment = os.environ.copy()
    source_root = str(PIPELINE_ROOT / "ML" / "src")
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join([source_root, existing] if existing else [source_root])
    command = [sys.executable, *arguments]
    print("[pipeline] " + subprocess.list2cmdline(command))
    subprocess.run(command, check=True, cwd=str(WORKSPACE_ROOT), env=environment)


if __name__ == "__main__":
    main()
