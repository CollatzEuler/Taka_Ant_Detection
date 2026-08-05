from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from roi_ant_tracker.roi import read_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paint a reusable mask of locations that random ROI sampling must avoid."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser("select", help="Paint excluded areas black and save a mask JSON.")
    select.add_argument("--video", required=True)
    select.add_argument("--output", required=True, help="Output exclusion JSON path.")
    select.add_argument("--frame", default="middle", help="Representative frame index or 'middle'.")
    select.add_argument("--brush-size", type=int, default=100, help="Brush diameter in source pixels.")
    select.add_argument("--display-width", type=int, default=1600)
    select.add_argument("--display-height", type=int, default=900)
    select.add_argument(
        "--rect",
        action="append",
        default=[],
        metavar="X,Y,W,H",
        help="Non-interactive excluded rectangle. May be repeated.",
    )
    select.add_argument("--window-title", default="Black out areas random ROIs must avoid")
    select.add_argument("--mask-output")
    select.add_argument("--preview-output")
    select.add_argument("--representative-output")

    preview = commands.add_parser("preview", help="Render an exclusion mask over a video frame.")
    preview.add_argument("--video", required=True)
    preview.add_argument("--exclusion-json", required=True)
    preview.add_argument("--output", required=True)
    preview.add_argument("--frame", default="middle")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "select":
        run_select(args)
    else:
        run_preview(args)


def run_select(args: argparse.Namespace) -> None:
    if args.brush_size <= 0:
        raise ValueError("brush-size must be positive.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame, frame_index, _fps = read_frame(args.video, parse_frame_arg(args.frame))
    height, width = frame.shape[:2]
    allowed_mask = np.full((height, width), 255, dtype=np.uint8)
    for value in args.rect:
        x, y, rect_width, rect_height = parse_rect(value)
        block_rectangle(allowed_mask, x, y, rect_width, rect_height)

    if not args.rect:
        saved = paint_exclusions(
            frame,
            allowed_mask,
            args.window_title,
            args.brush_size,
            args.display_width,
            args.display_height,
        )
        if not saved:
            print("[select-exclusions] no mask saved")
            return

    mask_path = Path(args.mask_output) if args.mask_output else output.with_name(output.stem + "_mask.png")
    preview_path = Path(args.preview_output) if args.preview_output else output.with_name(output.stem + "_preview.png")
    representative_path = (
        Path(args.representative_output)
        if args.representative_output
        else output.with_name(output.stem + "_representative.png")
    )
    for path in (mask_path, preview_path, representative_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), allowed_mask)
    cv2.imwrite(str(preview_path), render_exclusions(frame, allowed_mask))
    cv2.imwrite(str(representative_path), frame)

    blocked_pixels = int(np.count_nonzero(allowed_mask == 0))
    payload = {
        "exclusion_version": 1,
        "source_video": str(Path(args.video).resolve()),
        "representative_frame_index": frame_index,
        "frame_width": width,
        "frame_height": height,
        "mask_path": str(mask_path.resolve()),
        "preview_image": str(preview_path.resolve()),
        "representative_frame": str(representative_path.resolve()),
        "mask_semantics": "255=allowed, 0=excluded",
        "blocked_pixel_count": blocked_pixels,
        "blocked_fraction": blocked_pixels / float(width * height),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[select-exclusions] blocked_fraction={payload['blocked_fraction']:.3f} "
        f"mask={mask_path} output={output}"
    )
    print(f"[select-exclusions] preview={preview_path}")


def run_preview(args: argparse.Namespace) -> None:
    frame, frame_index, _fps = read_frame(args.video, parse_frame_arg(args.frame))
    mask = load_exclusion_mask(args.exclusion_json, frame.shape[1], frame.shape[0])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), render_exclusions(frame, mask))
    print(f"[preview-exclusions] frame={frame_index} output={output}")


def paint_exclusions(
    frame: np.ndarray,
    allowed_mask: np.ndarray,
    window_title: str,
    brush_size: int,
    display_width: int,
    display_height: int,
) -> bool:
    height, width = frame.shape[:2]
    scale = min(1.0, display_width / width, display_height / height)
    state: dict[str, object] = {
        "painting": False,
        "restore": False,
        "last": None,
        "brush_size": brush_size,
        "history": [],
    }

    def source_point(x: int, y: int) -> tuple[int, int]:
        return min(width - 1, int(round(x / scale))), min(height - 1, int(round(y / scale)))

    def begin_stroke(x: int, y: int, restore: bool) -> None:
        history = state["history"]
        assert isinstance(history, list)
        history.append(allowed_mask.copy())
        if len(history) > 8:
            history.pop(0)
        point = source_point(x, y)
        state["painting"] = True
        state["restore"] = restore
        state["last"] = point
        paint_line(allowed_mask, point, point, int(state["brush_size"]), restore)

    def mouse_callback(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            begin_stroke(x, y, False)
        elif event == cv2.EVENT_RBUTTONDOWN:
            begin_stroke(x, y, True)
        elif event == cv2.EVENT_MOUSEMOVE and bool(state["painting"]):
            current = source_point(x, y)
            previous = state["last"]
            assert isinstance(previous, tuple)
            paint_line(allowed_mask, previous, current, int(state["brush_size"]), bool(state["restore"]))
            state["last"] = current
        elif event in {cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP}:
            state["painting"] = False
            state["last"] = None

    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_title, mouse_callback)
    saved = False
    try:
        while True:
            preview = render_exclusions(frame, allowed_mask)
            display = cv2.resize(preview, (int(round(width * scale)), int(round(height * scale))))
            cv2.putText(
                display,
                "Left: block | Right: restore | [ ] brush | u undo | c clear | s save | q quit",
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                display,
                f"brush={int(state['brush_size'])} source px",
                (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
            cv2.imshow(window_title, display)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("s"):
                saved = True
                break
            if key == ord("u"):
                history = state["history"]
                assert isinstance(history, list)
                if history:
                    allowed_mask[:] = history.pop()
            if key == ord("c"):
                allowed_mask[:] = 255
            if key == ord("["):
                state["brush_size"] = max(10, int(state["brush_size"]) - 20)
            if key == ord("]"):
                state["brush_size"] = min(max(width, height), int(state["brush_size"]) + 20)
            if key == ord("q") or key == 27:
                break
    finally:
        cv2.destroyWindow(window_title)
    return saved


def paint_line(
    allowed_mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    brush_size: int,
    restore: bool,
) -> None:
    cv2.line(allowed_mask, start, end, 255 if restore else 0, thickness=max(1, brush_size))


def block_rectangle(mask: np.ndarray, x: int, y: int, width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("Excluded rectangle width and height must be positive.")
    x1 = max(0, min(mask.shape[1], x))
    y1 = max(0, min(mask.shape[0], y))
    x2 = max(0, min(mask.shape[1], x + width))
    y2 = max(0, min(mask.shape[0], y + height))
    if x1 < x2 and y1 < y2:
        mask[y1:y2, x1:x2] = 0


def load_exclusion_mask(path: str | Path, frame_width: int, frame_height: int) -> np.ndarray:
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


def render_exclusions(frame: np.ndarray, allowed_mask: np.ndarray) -> np.ndarray:
    output = frame.copy()
    output[allowed_mask == 0] = 0
    return output


def parse_frame_arg(value: str) -> int | str:
    return "middle" if str(value).lower() == "middle" else int(value)


def parse_rect(value: str) -> tuple[int, int, int, int]:
    parts = [int(round(float(part.strip()))) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--rect must be X,Y,W,H")
    return tuple(parts)  # type: ignore[return-value]


if __name__ == "__main__":
    main()
