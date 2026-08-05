from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from roi_ant_tracker.roi import RoiRect, draw_rois, load_rois, read_frame, roi_payload, validate_rois


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and preview per-video search ROIs from a representative middle frame."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser("select", help="Draw one or more rectangles and save an ROI JSON.")
    select.add_argument("--video", required=True)
    select.add_argument("--output", required=True)
    select.add_argument("--frame", default="middle", help="Frame index or 'middle'.")
    select.add_argument(
        "--rect",
        action="append",
        default=[],
        metavar="X,Y,W,H",
        help="Non-interactive rectangle. May be passed multiple times.",
    )
    select.add_argument("--window-title", default="Select ant search ROIs")
    select.add_argument("--representative-output", help="Representative frame PNG path.")
    select.add_argument("--preview-output", help="ROI preview PNG path.")

    middle = commands.add_parser("middle-frame", help="Write the video's representative middle frame to PNG.")
    middle.add_argument("--video", required=True)
    middle.add_argument("--output", required=True)
    middle.add_argument("--frame", default="middle", help="Frame index or 'middle'.")

    preview = commands.add_parser("preview", help="Render saved ROI rectangles onto a representative frame.")
    preview.add_argument("--video", required=True)
    preview.add_argument("--roi-json", required=True)
    preview.add_argument("--output", required=True)
    preview.add_argument("--frame", default="middle", help="Frame index or 'middle'.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "select":
        run_select(args)
    elif args.command == "middle-frame":
        run_middle_frame(args)
    elif args.command == "preview":
        run_preview(args)


def run_select(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame, frame_index, _fps = read_frame(args.video, parse_frame_arg(args.frame))
    frame_height, frame_width = frame.shape[:2]

    if args.rect:
        rois = [
            RoiRect(f"roi_{index:03d}", x, y, width, height)
            for index, (x, y, width, height) in enumerate((parse_rect(value) for value in args.rect), start=1)
        ]
    else:
        rois = interactive_rois(frame, args.window_title)
    if not rois:
        print("[select-roi] no ROI saved")
        return
    validate_rois(rois, frame_width, frame_height)

    representative_path = Path(args.representative_output) if args.representative_output else output.with_name(
        output.stem + "_representative.png"
    )
    preview_path = Path(args.preview_output) if args.preview_output else output.with_name(output.stem + "_preview.png")
    representative_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(representative_path), frame)
    cv2.imwrite(str(preview_path), draw_rois(frame, rois))

    payload = roi_payload(
        video_path=args.video,
        frame_index=frame_index,
        frame_width=frame_width,
        frame_height=frame_height,
        rectangles=rois,
        representative_frame=str(representative_path.resolve()),
        preview_image=str(preview_path.resolve()),
    )
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[select-roi] rois={len(rois)} output={output}")
    print(f"[select-roi] representative={representative_path}")
    print(f"[select-roi] preview={preview_path}")


def run_middle_frame(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame, frame_index, _fps = read_frame(args.video, parse_frame_arg(args.frame))
    cv2.imwrite(str(output), frame)
    print(f"[middle-frame] frame={frame_index} output={output}")


def run_preview(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame, frame_index, _fps = read_frame(args.video, parse_frame_arg(args.frame))
    height, width = frame.shape[:2]
    rois = load_rois(args.roi_json, width, height)
    cv2.imwrite(str(output), draw_rois(frame, rois))
    print(f"[preview-roi] frame={frame_index} rois={len(rois)} output={output}")


def interactive_rois(frame, window_title: str) -> list[RoiRect]:
    state = {
        "origin": None,
        "current": None,
        "dragging": False,
        "rectangles": [],
    }

    def mouse_callback(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["origin"] = (x, y)
            state["current"] = (x, y)
            state["dragging"] = True
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["current"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["dragging"]:
            state["current"] = (x, y)
            state["dragging"] = False
            rect = normalize_rectangle(state["origin"], state["current"])
            if rect is not None:
                state["rectangles"].append(rect)
            state["origin"] = None
            state["current"] = None

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_title, mouse_callback)
    saved = False
    try:
        while True:
            preview = frame.copy()
            rois = [
                RoiRect(f"roi_{index:03d}", x, y, width, height)
                for index, (x, y, width, height) in enumerate(state["rectangles"], start=1)
            ]
            if rois:
                preview = draw_rois(preview, rois)
            if state["dragging"] and state["origin"] is not None and state["current"] is not None:
                rect = normalize_rectangle(state["origin"], state["current"])
                if rect is not None:
                    x, y, width, height = rect
                    cv2.rectangle(preview, (x, y), (x + width, y + height), (255, 255, 0), 2)
            cv2.putText(
                preview,
                "Drag ROI rectangles | s save | u undo | c clear | q quit",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
            )
            cv2.imshow(window_title, preview)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("s"):
                saved = True
                break
            if key == ord("u") and state["rectangles"]:
                state["rectangles"].pop()
            if key == ord("c"):
                state["rectangles"].clear()
            if key == ord("q") or key == 27:
                break
    finally:
        cv2.destroyWindow(window_title)
    if not saved:
        return []
    return [
        RoiRect(f"roi_{index:03d}", x, y, width, height)
        for index, (x, y, width, height) in enumerate(state["rectangles"], start=1)
    ]


def parse_frame_arg(value: str) -> int | str:
    return "middle" if str(value).lower() == "middle" else int(value)


def parse_rect(value: str) -> tuple[int, int, int, int]:
    parts = [int(round(float(part.strip()))) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--rect must be X,Y,W,H")
    return tuple(parts)  # type: ignore[return-value]


def normalize_rectangle(
    start: tuple[int, int] | None,
    end: tuple[int, int] | None,
) -> tuple[int, int, int, int] | None:
    if start is None or end is None:
        return None
    x1, y1 = start
    x2, y2 = end
    left = min(x1, x2)
    top = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    if width <= 1 or height <= 1:
        return None
    return left, top, width, height


if __name__ == "__main__":
    main()

