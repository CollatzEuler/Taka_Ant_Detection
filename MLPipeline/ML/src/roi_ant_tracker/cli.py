from __future__ import annotations

import argparse

from .evaluation import evaluate_detector
from .inference import track_video
from .model import ARCHITECTURES
from .training import train_detector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROI-limited ant detector and open-world tracker.")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train an ant detector from COCO boxes.")
    train.add_argument("--images", required=True)
    train.add_argument("--annotations", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--architecture", choices=ARCHITECTURES, default="mobilenet_v3_fpn")
    train.add_argument("--min-size", type=int, default=768)
    train.add_argument("--max-size", type=int, default=1280)
    train.add_argument(
        "--initialization",
        choices=["detector", "backbone", "random"],
        default="detector",
        help="Starting weights. 'detector' fine-tunes a fully COCO-pretrained Faster R-CNN.",
    )
    train.add_argument("--resume-checkpoint", help="Continue fine-tuning from a prior checkpoint.")
    train.add_argument(
        "--reset-training-history",
        action="store_true",
        help="Load resume-checkpoint weights but start a fresh epoch count, training log, and best-loss baseline.",
    )
    train.add_argument("--box-detections-per-img", type=int, default=300)
    train.add_argument("--lr-patience", type=int, default=3, help="Epochs without meaningful loss improvement before lowering LR.")
    train.add_argument("--lr-factor", type=float, default=0.5, help="Multiplier applied when the loss plateaus.")
    train.add_argument("--min-learning-rate", type=float, default=1e-7)
    train.add_argument("--plateau-delta", type=float, default=0.001, help="Relative loss improvement required to reset plateau count.")
    train.add_argument("--log-every", type=int, default=10, help="Print training loss/LR every N batches. Use 0 for epoch summaries only.")
    train.add_argument("--max-grad-norm", type=float, default=1.0, help="Clip gradients to this norm. Use 0 to disable.")
    train.add_argument(
        "--max-empty-fraction",
        type=float,
        default=1.0,
        help="Maximum fraction of empty images kept during training. Use 0.25 or 0.5 when annotations are sparse.",
    )
    train.add_argument("--device", default="auto")
    train.add_argument("--workers", type=int, default=0)

    track = commands.add_parser("track", help="Detect ants only inside saved ROIs and export trajectories.")
    track.add_argument("--video", required=True)
    track.add_argument("--checkpoint", required=True)
    track.add_argument("--output", required=True)
    track.add_argument("--roi-json", help="ROI JSON from MLPipeline select-roi. Omit for full-frame inference.")
    track.add_argument("--device", default="auto")
    track.add_argument("--score-threshold", type=float, default=0.30)
    track.add_argument("--nms-threshold", type=float, default=0.70)
    track.add_argument("--tile-size", type=int, default=0, help="Use 0 for one inference crop per ROI.")
    track.add_argument("--tile-overlap", type=int, default=64)
    track.add_argument("--tile-batch-size", type=int, default=4)
    track.add_argument("--frame-batch-size", type=int, default=1)
    track.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Run inference on every Nth source frame. Useful for long high-frame-rate videos.",
    )
    track.add_argument(
        "--roi-padding",
        type=int,
        default=0,
        help="Add context pixels around each ROI, but keep only detections centered inside the original ROI.",
    )
    track.add_argument(
        "--amp",
        choices=["auto", "off", "fp16", "bf16"],
        default="auto",
        help="CUDA mixed precision. auto prefers BF16 when supported, otherwise FP16.",
    )
    track.add_argument("--no-overlay", action="store_true", help="Skip tracking-overlay video encoding.")
    track.add_argument(
        "--overlay-mode",
        choices=["roi", "full", "both"],
        default="roi",
        help="Write ROI-cropped overlays by default. Use 'full' for the old whole-frame overlay or 'both' for both.",
    )
    track.add_argument("--inference-min-size", type=int)
    track.add_argument("--inference-max-size", type=int)
    track.add_argument("--max-detections", type=int, default=300)
    track.add_argument(
        "--duplicate-center-distance",
        type=float,
        default=0.0,
        help="Drop lower-score detections whose centers are within this many pixels of a higher-score detection.",
    )
    track.add_argument(
        "--duplicate-overlap-threshold",
        type=float,
        default=0.80,
        help="Drop lower-score detections when this fraction of the smaller box overlaps a higher-score detection.",
    )
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

    evaluate = commands.add_parser("evaluate", help="Evaluate a detector against held-out COCO ant boxes.")
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        train_detector(
            images_dir=args.images,
            annotations_path=args.annotations,
            output_path=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            architecture=args.architecture,
            min_size=args.min_size,
            max_size=args.max_size,
            initialization=args.initialization,
            device_name=args.device,
            workers=args.workers,
            resume_checkpoint=args.resume_checkpoint,
            box_detections_per_img=args.box_detections_per_img,
            lr_patience=args.lr_patience,
            lr_factor=args.lr_factor,
            min_learning_rate=args.min_learning_rate,
            plateau_delta=args.plateau_delta,
            log_every=args.log_every,
            max_grad_norm=args.max_grad_norm,
            max_empty_fraction=args.max_empty_fraction,
            reset_training_history=args.reset_training_history,
        )
    elif args.command == "track":
        track_video(
            video_path=args.video,
            checkpoint_path=args.checkpoint,
            output_dir=args.output,
            roi_json=args.roi_json,
            device_name=args.device,
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            tile_batch_size=args.tile_batch_size,
            frame_batch_size=args.frame_batch_size,
            frame_step=args.frame_step,
            roi_padding=args.roi_padding,
            amp=args.amp,
            write_overlay=not args.no_overlay,
            overlay_mode=args.overlay_mode,
            inference_min_size=args.inference_min_size,
            inference_max_size=args.inference_max_size,
            max_detections=args.max_detections,
            duplicate_center_distance=args.duplicate_center_distance,
            duplicate_overlap_threshold=args.duplicate_overlap_threshold,
            max_distance=args.max_distance,
            max_missed=args.max_missed,
            match_threshold=args.match_threshold,
            ambiguity_margin=args.ambiguity_margin,
            close_distance=args.close_distance,
            border_margin=args.border_margin,
            minimum_track_hits=args.minimum_track_hits,
            review_threshold=args.review_threshold,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            start_seconds=args.start_seconds,
            end_seconds=args.end_seconds,
            duration_seconds=args.duration_seconds,
            progress_every=args.progress_every,
        )
    elif args.command == "evaluate":
        evaluate_detector(
            images_dir=args.images,
            annotations_path=args.annotations,
            checkpoint_path=args.checkpoint,
            output_dir=args.output,
            device_name=args.device,
            batch_size=args.batch_size,
            workers=args.workers,
            amp=args.amp,
            score_threshold=args.score_threshold,
            candidate_score_threshold=args.candidate_score_threshold,
            iou_threshold=args.iou_threshold,
            center_distance_threshold=args.center_distance_threshold,
            inference_min_size=args.inference_min_size,
            inference_max_size=args.inference_max_size,
            progress_every=args.progress_every,
        )


if __name__ == "__main__":
    main()
