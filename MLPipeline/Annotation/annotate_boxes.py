from __future__ import annotations

import argparse
import csv
import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import cv2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate ant bounding boxes on sampled ROI frames."
    )
    parser.add_argument("--images-dir", required=True, help="Directory containing sampled frame images.")
    parser.add_argument("--manifest", required=True, help="frame_manifest.csv from sample-roi.")
    parser.add_argument("--project", required=True, help="Annotation project JSON path.")
    parser.add_argument("--canvas-max-width", type=int, default=1100)
    parser.add_argument("--canvas-max-height", type=int, default=800)
    return parser


class AnnotationApp:
    def __init__(
        self,
        root: tk.Tk,
        images_dir: Path,
        manifest_path: Path,
        project_path: Path,
        canvas_max_width: int,
        canvas_max_height: int,
    ) -> None:
        self.root = root
        self.images_dir = images_dir
        self.manifest_path = manifest_path
        self.project_path = project_path
        self.canvas_max_width = canvas_max_width
        self.canvas_max_height = canvas_max_height
        self.project = self.load_or_create_project()
        self.index = 0
        self.scale = 1.0
        self.display_width = 1
        self.display_height = 1
        self.offset_x = 0
        self.offset_y = 0
        self.photo = None
        self.current_cv_image = None
        self.drag_start: tuple[int, int] | None = None
        self.drag_current: tuple[int, int] | None = None
        self.selected_box_index: int | None = None
        self.active_identity = tk.StringVar(value="")
        self.status_text = tk.StringVar(value="")

        self.root.title("Ant ROI Annotation")
        self.root.protocol("WM_DELETE_WINDOW", self.handle_quit)

        self.build_ui()
        self.bind_keys()
        self.render_current_image()

    def build_ui(self) -> None:
        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        toolbar = tk.Frame(left)
        toolbar.pack(fill=tk.X, padx=6, pady=6)
        for label, command in [
            ("Prev", self.prev_image),
            ("Next", self.next_image),
            ("Save", self.save_project),
            ("Export COCO", self.export_coco),
            ("Delete Box", self.delete_selected_box),
            ("Undo", self.undo_last_box),
        ]:
            tk.Button(toolbar, text=label, command=command).pack(side=tk.LEFT, padx=4)

        identity_frame = tk.Frame(toolbar)
        identity_frame.pack(side=tk.LEFT, padx=12)
        tk.Label(identity_frame, text="Active ID").pack(side=tk.LEFT)
        tk.Entry(identity_frame, width=4, textvariable=self.active_identity).pack(side=tk.LEFT, padx=4)

        self.canvas = tk.Canvas(left, width=self.canvas_max_width, height=self.canvas_max_height, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        right = tk.Frame(main, width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        self.image_label = tk.Label(right, justify=tk.LEFT, anchor="w")
        self.image_label.pack(fill=tk.X, padx=6, pady=6)

        tk.Label(right, text="Boxes").pack(anchor="w", padx=6)
        self.box_list = tk.Listbox(right, height=18)
        self.box_list.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.box_list.bind("<<ListboxSelect>>", self.on_select_box)

        help_text = "\n".join(
            [
                "Mouse drag: draw box",
                "Delete: remove selected box",
                "u: undo last box",
                "n / p: next or prev image",
                "1-9: set active identity",
                "0: clear active identity",
                "s: save",
                "e: export COCO",
                "q: save and quit",
            ]
        )
        tk.Label(right, text=help_text, justify=tk.LEFT, anchor="w").pack(fill=tk.X, padx=6, pady=6)

        status = tk.Label(self.root, textvariable=self.status_text, anchor="w")
        status.pack(fill=tk.X, padx=6, pady=4)

    def bind_keys(self) -> None:
        self.root.bind("<KeyPress-n>", lambda _event: self.next_image())
        self.root.bind("<KeyPress-p>", lambda _event: self.prev_image())
        self.root.bind("<KeyPress-s>", lambda _event: self.save_project())
        self.root.bind("<KeyPress-e>", lambda _event: self.export_coco())
        self.root.bind("<KeyPress-u>", lambda _event: self.undo_last_box())
        self.root.bind("<KeyPress-q>", lambda _event: self.handle_quit())
        self.root.bind("<Delete>", lambda _event: self.delete_selected_box())
        for digit in "123456789":
            self.root.bind(f"<KeyPress-{digit}>", self.make_identity_handler(digit))
        self.root.bind("<KeyPress-0>", lambda _event: self.set_active_identity(""))

    def make_identity_handler(self, digit: str):
        return lambda _event: self.set_active_identity(digit)

    def set_active_identity(self, value: str) -> None:
        self.active_identity.set(value)
        if self.selected_box_index is not None:
            annotations = self.current_image_record()["annotations"]
            if 0 <= self.selected_box_index < len(annotations):
                annotations[self.selected_box_index]["identity"] = value or None
                self.render_current_image()
                return
        self.update_status()

    def load_or_create_project(self) -> dict:
        if self.project_path.exists():
            return json.loads(self.project_path.read_text(encoding="utf-8-sig"))
        rows = []
        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for image_id, row in enumerate(reader, start=1):
                rows.append(
                    {
                        "image_id": image_id,
                        "file_name": row["image_name"],
                        "image_path": row["image_path"],
                        "width": int(float(row["width"])),
                        "height": int(float(row["height"])),
                        "video_path": row["video_path"],
                        "source_frame_index": int(float(row["source_frame_index"])),
                        "source_time_seconds": float(row["source_time_seconds"]),
                        "roi_json": row.get("roi_json", ""),
                        "roi_id": row.get("roi_id", "full_frame"),
                        "roi_x": int(float(row.get("roi_x", 0) or 0)),
                        "roi_y": int(float(row.get("roi_y", 0) or 0)),
                        "roi_width": int(float(row.get("roi_width", row["width"]) or row["width"])),
                        "roi_height": int(float(row.get("roi_height", row["height"]) or row["height"])),
                        "cluster_id": int(float(row.get("cluster_id", image_id) or image_id)),
                        "annotations": [],
                    }
                )
        return {
            "project_version": 1,
            "manifest_path": str(self.manifest_path.resolve()),
            "images_dir": str(self.images_dir.resolve()),
            "images": rows,
        }

    @property
    def images(self) -> list[dict]:
        return self.project["images"]

    def current_image_record(self) -> dict:
        return self.images[self.index]

    def render_current_image(self) -> None:
        record = self.current_image_record()
        path = Path(record["image_path"])
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        self.current_cv_image = frame
        height, width = frame.shape[:2]
        self.scale = min(
            self.canvas_max_width / max(width, 1),
            self.canvas_max_height / max(height, 1),
            1.0,
        )
        self.display_width = max(1, int(round(width * self.scale)))
        self.display_height = max(1, int(round(height * self.scale)))
        resized = cv2.resize(frame, (self.display_width, self.display_height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        ppm = cv2.imencode(".ppm", rgb)[1].tobytes()
        self.photo = tk.PhotoImage(data=ppm)

        self.canvas.delete("all")
        self.canvas.config(width=self.display_width, height=self.display_height)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.draw_boxes()
        self.refresh_box_list()
        self.image_label.config(
            text=(
                f"Image {self.index + 1}/{len(self.images)}\n"
                f"{record['file_name']}\n"
                f"video={Path(record['video_path']).name}\n"
                f"frame={record['source_frame_index']} time={record['source_time_seconds']:.2f}s\n"
                f"roi={record.get('roi_id', 'full_frame')} "
                f"origin=({record.get('roi_x', 0)}, {record.get('roi_y', 0)})"
            )
        )
        self.update_status()

    def draw_boxes(self) -> None:
        record = self.current_image_record()
        for index, annotation in enumerate(record["annotations"]):
            x, y, width, height = annotation["bbox"]
            x1 = int(round(x * self.scale))
            y1 = int(round(y * self.scale))
            x2 = int(round((x + width) * self.scale))
            y2 = int(round((y + height) * self.scale))
            selected = index == self.selected_box_index
            suggested = bool(annotation.get("suggested", False))
            color = "#00aaff" if suggested else "#00ff66"
            if selected:
                color = "#ffcc00"
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
            identity = annotation.get("identity") or "-"
            self.canvas.create_text(
                x1 + 4,
                max(12, y1 + 12),
                text=f"id {identity}",
                fill=color,
                anchor=tk.NW,
            )
        if self.drag_start is not None and self.drag_current is not None:
            x1, y1 = self.drag_start
            x2, y2 = self.drag_current
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#00ffff", width=2)

    def refresh_box_list(self) -> None:
        self.box_list.delete(0, tk.END)
        for index, annotation in enumerate(self.current_image_record()["annotations"]):
            x, y, width, height = annotation["bbox"]
            identity = annotation.get("identity") or "-"
            score = annotation.get("score")
            score_text = f" score={float(score):.2f}" if score is not None else ""
            suggested_text = " suggested" if annotation.get("suggested", False) else ""
            self.box_list.insert(
                tk.END,
                f"{index + 1}: x={x:.1f} y={y:.1f} w={width:.1f} h={height:.1f} id={identity}{score_text}{suggested_text}",
            )
        if self.selected_box_index is not None and self.selected_box_index < self.box_list.size():
            self.box_list.selection_set(self.selected_box_index)

    def on_select_box(self, _event) -> None:
        selection = self.box_list.curselection()
        self.selected_box_index = selection[0] if selection else None
        self.render_current_image()

    def on_mouse_down(self, event) -> None:
        self.drag_start = (event.x, event.y)
        self.drag_current = (event.x, event.y)

    def on_mouse_drag(self, event) -> None:
        if self.drag_start is None:
            return
        self.drag_current = (event.x, event.y)
        self.render_current_image()

    def on_mouse_up(self, event) -> None:
        if self.drag_start is None:
            return
        self.drag_current = (event.x, event.y)
        bbox = self.canvas_bbox_to_image_bbox(self.drag_start, self.drag_current)
        self.drag_start = None
        self.drag_current = None
        if bbox is not None:
            self.current_image_record()["annotations"].append(
                {
                    "bbox": bbox,
                    "category": "ant",
                    "identity": self.active_identity.get() or None,
                }
            )
            self.selected_box_index = len(self.current_image_record()["annotations"]) - 1
        self.render_current_image()

    def canvas_bbox_to_image_bbox(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> list[float] | None:
        x1 = max(0, min(start[0], end[0]))
        y1 = max(0, min(start[1], end[1]))
        x2 = min(self.display_width, max(start[0], end[0]))
        y2 = min(self.display_height, max(start[1], end[1]))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        image_x = x1 / self.scale
        image_y = y1 / self.scale
        image_width = (x2 - x1) / self.scale
        image_height = (y2 - y1) / self.scale
        return [round(image_x, 2), round(image_y, 2), round(image_width, 2), round(image_height, 2)]

    def prev_image(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.selected_box_index = None
            self.render_current_image()

    def next_image(self) -> None:
        if self.index < len(self.images) - 1:
            self.index += 1
            self.selected_box_index = None
            self.render_current_image()

    def delete_selected_box(self) -> None:
        if self.selected_box_index is None:
            return
        annotations = self.current_image_record()["annotations"]
        if 0 <= self.selected_box_index < len(annotations):
            annotations.pop(self.selected_box_index)
        self.selected_box_index = None
        self.render_current_image()

    def undo_last_box(self) -> None:
        annotations = self.current_image_record()["annotations"]
        if annotations:
            annotations.pop()
        self.selected_box_index = None
        self.render_current_image()

    def save_project(self) -> None:
        self.project_path.parent.mkdir(parents=True, exist_ok=True)
        self.project_path.write_text(json.dumps(self.project, indent=2), encoding="utf-8")
        self.update_status("Project saved.")

    def export_coco(self) -> None:
        output_path = self.project_path.with_name(self.project_path.stem + "_coco.json")
        images = []
        annotations = []
        annotation_id = 1
        for image in self.images:
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
            for box in image["annotations"]:
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
        self.update_status(f"COCO exported to {output_path}")

    def update_status(self, message: str | None = None) -> None:
        labeled = sum(1 for image in self.images if image["annotations"])
        current_boxes = len(self.current_image_record()["annotations"])
        base = (
            f"Labeled images: {labeled}/{len(self.images)} | "
            f"Current boxes: {current_boxes} | Active ID: {self.active_identity.get() or '-'}"
        )
        if message:
            base = f"{base} | {message}"
        self.status_text.set(base)

    def handle_quit(self) -> None:
        self.save_project()
        self.root.destroy()


def main() -> None:
    args = build_parser().parse_args()
    root = tk.Tk()
    app = AnnotationApp(
        root=root,
        images_dir=Path(args.images_dir),
        manifest_path=Path(args.manifest),
        project_path=Path(args.project),
        canvas_max_width=args.canvas_max_width,
        canvas_max_height=args.canvas_max_height,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
