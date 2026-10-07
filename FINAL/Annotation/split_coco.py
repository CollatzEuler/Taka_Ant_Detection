from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split COCO frames into temporal training and validation blocks without adjacent-frame leakage."
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument(
        "--block-frames",
        type=int,
        default=600,
        help="Keep source frames in blocks of this size together. At 60 fps, 600 frames is 10 seconds.",
    )
    parser.add_argument("--seed", type=int, default=17)
    return parser


def split_coco_payload(
    payload: dict[str, Any],
    validation_fraction: float,
    block_frames: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation-fraction must be between 0 and 1.")
    if block_frames <= 0:
        raise ValueError("block-frames must be positive.")
    images = list(payload.get("images", []))
    if len(images) < 2:
        raise ValueError("At least two images are required for a train/validation split.")

    blocks_by_video: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for image in images:
        video = str(image.get("video_path") or image.get("source_video") or "unknown_video")
        frame = int(image.get("source_frame_index", image["id"]))
        blocks_by_video[video][frame // block_frames].append(image)

    validation_ids: set[int] = set()
    for video, blocks in blocks_by_video.items():
        block_ids = sorted(blocks)
        if len(block_ids) < 2:
            raise ValueError(
                f"Video {video!r} has only one temporal block. Reduce --block-frames or label a wider time range."
            )
        random.Random(f"{seed}:{video}").shuffle(block_ids)
        validation_count = min(len(block_ids) - 1, max(1, round(len(block_ids) * validation_fraction)))
        selected_blocks = set(block_ids[:validation_count])
        validation_ids.update(
            int(image["id"])
            for block_id in selected_blocks
            for image in blocks[block_id]
        )

    train_ids = {int(image["id"]) for image in images} - validation_ids
    return subset_payload(payload, train_ids), subset_payload(payload, validation_ids)


def subset_payload(payload: dict[str, Any], image_ids: set[int]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in payload.items()
        if key not in {"images", "annotations"}
    }
    result["images"] = [image for image in payload.get("images", []) if int(image["id"]) in image_ids]
    result["annotations"] = [
        annotation
        for annotation in payload.get("annotations", [])
        if int(annotation["image_id"]) in image_ids
    ]
    return result


def main() -> None:
    args = build_parser().parse_args()
    payload = json.loads(Path(args.annotations).read_text(encoding="utf-8-sig"))
    train, validation = split_coco_payload(
        payload,
        validation_fraction=args.validation_fraction,
        block_frames=args.block_frames,
        seed=args.seed,
    )
    train_output = Path(args.train_output)
    validation_output = Path(args.validation_output)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    validation_output.parent.mkdir(parents=True, exist_ok=True)
    train_output.write_text(json.dumps(train, indent=2), encoding="utf-8")
    validation_output.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(
        f"[split] train_images={len(train['images'])} train_boxes={len(train['annotations'])} "
        f"validation_images={len(validation['images'])} validation_boxes={len(validation['annotations'])}"
    )
    print(f"[split] train={train_output}")
    print(f"[split] validation={validation_output}")


if __name__ == "__main__":
    main()
