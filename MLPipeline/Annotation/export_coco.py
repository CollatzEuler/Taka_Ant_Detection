from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Annotation project JSON to COCO format."
    )
    parser.add_argument("--project", required=True, help="Annotation project JSON path.")
    parser.add_argument("--output", required=True, help="Output COCO JSON path.")
    parser.add_argument("--skip-empty", action="store_true", help="Skip images with no boxes.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_path = Path(args.project)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    project = json.loads(project_path.read_text(encoding="utf-8-sig"))
    images = []
    annotations = []
    annotation_id = 1
    for image in project.get("images", []):
        boxes = image.get("annotations", [])
        if args.skip_empty and not boxes:
            continue
        images.append(
            {
                "id": int(image["image_id"]),
                "file_name": image["file_name"],
                "width": int(image["width"]),
                "height": int(image["height"]),
                "video_path": image.get("video_path", ""),
                "source_frame_index": image.get("source_frame_index"),
                "source_time_seconds": image.get("source_time_seconds"),
                "roi_json": image.get("roi_json", ""),
                "roi_id": image.get("roi_id", "full_frame"),
                "roi_x": image.get("roi_x", 0),
                "roi_y": image.get("roi_y", 0),
            }
        )
        for box in boxes:
            x, y, width, height = [float(value) for value in box["bbox"]]
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": int(image["image_id"]),
                    "category_id": 1,
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                    "identity": box.get("identity"),
                }
            )
            annotation_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "ant"}],
    }
    output_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    print(f"[export] images={len(images)} annotations={len(annotations)} output={output_path}")


if __name__ == "__main__":
    main()
