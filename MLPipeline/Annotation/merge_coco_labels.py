from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge multiple single-class COCO detector datasets into one JSON."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=2,
        metavar=("COCO_JSON", "IMAGE_PREFIX"),
        required=True,
        help=(
            "Input COCO JSON and path prefix to prepend to each image file_name. "
            "Repeat once per dataset. Example: --dataset labels1_coco.json labels1/selected_frames"
        ),
    )
    parser.add_argument("--output", required=True, help="Merged COCO JSON output path.")
    parser.add_argument(
        "--category-name",
        default="ant",
        help="Single detector category name to use in the merged file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    merged = merge_coco_datasets(args.dataset, args.category_name)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(
        f"[merge] datasets={len(args.dataset)} images={len(merged['images'])} "
        f"annotations={len(merged['annotations'])} output={output}"
    )


def merge_coco_datasets(
    dataset_specs: list[list[str]],
    category_name: str,
) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    seen_file_names: set[str] = set()
    next_image_id = 1
    next_annotation_id = 1

    for coco_path_text, image_prefix_text in dataset_specs:
        coco_path = Path(coco_path_text)
        image_prefix = normalize_prefix(image_prefix_text)
        payload = json.loads(coco_path.read_text(encoding="utf-8-sig"))
        old_to_new_image_id: dict[int, int] = {}

        for image in payload.get("images", []):
            old_image_id = int(image["id"])
            file_name = normalize_join(image_prefix, str(image["file_name"]))
            if file_name in seen_file_names:
                raise ValueError(f"Duplicate merged image file_name: {file_name}")
            seen_file_names.add(file_name)
            old_to_new_image_id[old_image_id] = next_image_id

            merged_image = dict(image)
            merged_image["id"] = next_image_id
            merged_image["file_name"] = file_name
            images.append(merged_image)
            next_image_id += 1

        for annotation in payload.get("annotations", []):
            old_image_id = int(annotation["image_id"])
            if old_image_id not in old_to_new_image_id:
                raise ValueError(f"Annotation references missing image_id={old_image_id} in {coco_path}")

            merged_annotation = dict(annotation)
            merged_annotation["id"] = next_annotation_id
            merged_annotation["image_id"] = old_to_new_image_id[old_image_id]
            merged_annotation["category_id"] = 1
            annotations.append(merged_annotation)
            next_annotation_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": category_name}],
    }


def normalize_prefix(prefix: str) -> str:
    return prefix.replace("\\", "/").strip("/")


def normalize_join(prefix: str, file_name: str) -> str:
    normalized_name = file_name.replace("\\", "/").lstrip("/")
    if not prefix:
        return normalized_name
    if normalized_name.startswith(prefix + "/"):
        return normalized_name
    return f"{prefix}/{normalized_name}"


if __name__ == "__main__":
    main()
