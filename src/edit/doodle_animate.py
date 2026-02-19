#!/usr/bin/env python3
"""
Doodle animation engine: convert a static image into a whiteboard-style
hand-drawing animation video.

Reveal algorithms:
  zigzag — Smooth pen path with scrub radius. Angle parameter (0–90°)
           controls stroke tilt: 0=horizontal, 90=vertical, 30=default.
  rows   — Cell-grid reveal, row by row (left↔right, top→bottom).
  knn    — Nearest-neighbor traversal of content cells only.

Usage:
  from src.edit.doodle_animate import animate_image
  animate_image(image_path, output_path, duration=8.0, reveal="zigzag", angle=15)
"""

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_HAND_PATH = _ASSETS_DIR / "hand-draw.png"


def _load_hand(scale: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Load RGBA hand image, crop to content, scale, return (bgr, alpha)."""
    hand = cv2.imread(str(_HAND_PATH), cv2.IMREAD_UNCHANGED)
    if hand is None:
        raise FileNotFoundError(f"Hand asset not found: {_HAND_PATH}")

    # Extract alpha and crop to non-transparent bounding box
    alpha = hand[:, :, 3]
    ys, xs = np.where(alpha > 127)
    if len(ys) > 0:
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        hand = hand[y0:y1, x0:x1]

    h, w = hand.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    hand = cv2.resize(hand, (new_w, new_h), interpolation=cv2.INTER_AREA)

    alpha_out = hand[:, :, 3].astype(np.float32) / 255.0
    return hand[:, :, :3], alpha_out


def _detect_bg_color(img: np.ndarray) -> np.ndarray:
    """Detect dominant background color by sampling border pixels.

    Returns a (3,) BGR array suitable for np.full fill.
    """
    h, w = img.shape[:2]
    strip = max(1, int(min(h, w) * 0.03))
    pixels = np.concatenate(
        [
            img[:strip, :].reshape(-1, 3),
            img[-strip:, :].reshape(-1, 3),
            img[:, :strip].reshape(-1, 3),
            img[:, -strip:].reshape(-1, 3),
        ]
    )
    return np.median(pixels, axis=0).astype(np.uint8)


def _overlay_hand(
    canvas: np.ndarray,
    hand_bgr: np.ndarray,
    hand_alpha: np.ndarray,
    cx: int,
    cy: int,
) -> np.ndarray:
    """Composite hand image onto canvas at (cx, cy) = pen tip position.

    The pen tip is at the top-center of the hand image (marker held upright).
    """
    frame = canvas.copy()
    hh, hw = hand_bgr.shape[:2]
    fh, fw = frame.shape[:2]

    # Pen tip is at top-center of the cropped hand image
    tip_x = int(hw * 0.50)
    tip_y = 0

    x0 = cx - tip_x
    y0 = cy - tip_y
    x1 = x0 + hw
    y1 = y0 + hh

    # Clip to frame bounds
    src_x0 = max(0, -x0)
    src_y0 = max(0, -y0)
    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = min(fw, x1)
    dst_y1 = min(fh, y1)
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return frame

    roi = frame[dst_y0:dst_y1, dst_x0:dst_x1]
    h_crop = hand_bgr[src_y0:src_y1, src_x0:src_x1]
    a_crop = hand_alpha[src_y0:src_y1, src_x0:src_x1, np.newaxis]

    blended = h_crop.astype(np.float32) * a_crop + roi.astype(np.float32) * (
        1.0 - a_crop
    )
    frame[dst_y0:dst_y1, dst_x0:dst_x1] = blended.astype(np.uint8)
    return frame


def _build_zigzag_path(
    w: int, h: int, num_strokes: int = 15, angle: float = 15.0
) -> list[tuple[int, int]]:
    """Build a zigzag pen path as (x, y) waypoints at a given angle.

    angle controls the tilt of strokes (degrees, 0–90):
      0  → horizontal rows, advancing top to bottom
      90 → vertical columns, advancing left to right
      30 → strokes tilted 30° from horizontal (default, natural for text)

    Each stroke is a line at `angle` degrees from horizontal. Strokes are
    evenly spaced along the perpendicular (advance) direction and alternate
    direction for the zigzag effect.
    """
    import math

    theta = math.radians(max(0.5, min(89.5, angle)))
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)

    # Total sweep distance to cover the full image along the advance direction
    sweep_dist = w * sin_t + h * cos_t
    spacing = sweep_dist / num_strokes

    points: list[tuple[int, int]] = []

    for i in range(num_strokes):
        d = (i + 0.5) * spacing

        # Find parameter t range where the stroke line is inside the image.
        # Stroke line: x = d*sin_t + t*cos_t, y = d*cos_t - t*sin_t
        t_lo, t_hi = -1e9, 1e9

        if cos_t > 1e-6:
            t_lo = max(t_lo, -d * sin_t / cos_t)
            t_hi = min(t_hi, (w - d * sin_t) / cos_t)
        if sin_t > 1e-6:
            t_lo = max(t_lo, (d * cos_t - h) / sin_t)
            t_hi = min(t_hi, d * cos_t / sin_t)

        if t_lo >= t_hi:
            continue

        x0 = int(d * sin_t + t_lo * cos_t)
        y0 = int(d * cos_t - t_lo * sin_t)
        x1 = int(d * sin_t + t_hi * cos_t)
        y1 = int(d * cos_t - t_hi * sin_t)

        p0, p1 = (x0, y0), (x1, y1)
        if i % 2 == 0:
            points.extend([p0, p1])
        else:
            points.extend([p1, p0])

    return points


def _interpolate_path(
    waypoints: list[tuple[int, int]], num_points: int
) -> list[tuple[int, int]]:
    """Evenly sample num_points along a polyline defined by waypoints."""
    if len(waypoints) < 2:
        return waypoints * num_points

    # Compute cumulative arc length
    segments = []
    total_len = 0.0
    for i in range(1, len(waypoints)):
        dx = waypoints[i][0] - waypoints[i - 1][0]
        dy = waypoints[i][1] - waypoints[i - 1][1]
        seg_len = (dx * dx + dy * dy) ** 0.5
        segments.append(seg_len)
        total_len += seg_len

    if total_len == 0:
        return [waypoints[0]] * num_points

    result: list[tuple[int, int]] = []
    for p in range(num_points):
        t = p / max(1, num_points - 1) * total_len
        cumulative = 0.0
        for i, seg_len in enumerate(segments):
            if cumulative + seg_len >= t or i == len(segments) - 1:
                frac = (t - cumulative) / seg_len if seg_len > 0 else 0
                frac = max(0.0, min(1.0, frac))
                x = int(
                    waypoints[i][0] + frac * (waypoints[i + 1][0] - waypoints[i][0])
                )
                y = int(
                    waypoints[i][1] + frac * (waypoints[i + 1][1] - waypoints[i][1])
                )
                result.append((x, y))
                break
            cumulative += seg_len

    return result


def _build_rows_order(h: int, w: int, cell_size: int) -> list[tuple[int, int]]:
    """Row-by-row: left→right on even rows, right→left on odd rows, top→bottom."""
    rows = h // cell_size
    cols = w // cell_size
    order: list[tuple[int, int]] = []

    for r in range(rows):
        if r % 2 == 0:
            for c in range(cols):
                order.append((r, c))
        else:
            for c in range(cols - 1, -1, -1):
                order.append((r, c))

    return order


def _build_knn_order(
    thresh: np.ndarray,
    cell_size: int,
    black_threshold: int = 10,
) -> list[tuple[int, int]]:
    """Nearest-neighbor traversal of content cells only.

    Visits cells that have dark pixels in nearest-neighbor order, skipping
    blank regions. Looks like tracing outlines.
    """
    h, w = thresh.shape
    rows = h // cell_size
    cols = w // cell_size

    has_content: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            cell = thresh[
                r * cell_size : (r + 1) * cell_size,
                c * cell_size : (c + 1) * cell_size,
            ]
            if np.any(cell < black_threshold):
                has_content.append((r, c))

    if not has_content:
        return []

    remaining = np.array(has_content, dtype=np.float32)
    order: list[tuple[int, int]] = []
    current_idx = 0

    while len(remaining) > 1:
        current = remaining[current_idx].copy()
        order.append((int(current[0]), int(current[1])))
        remaining = np.delete(remaining, current_idx, axis=0)
        dists = np.sum((remaining - current) ** 2, axis=1)
        current_idx = int(np.argmin(dists))

    order.append((int(remaining[0][0]), int(remaining[0][1])))
    return order


# ---------------------------------------------------------------------------
# Frame renderers
# ---------------------------------------------------------------------------


def _render_zigzag_frames(
    proc: subprocess.Popen,
    img: np.ndarray,
    w: int,
    h: int,
    draw_frames: int,
    hold_frames: int,
    hand_bgr: np.ndarray | None,
    hand_alpha: np.ndarray | None,
    show_hand: bool,
    num_strokes: int = 15,
    angle: float = 15.0,
) -> None:
    """Render frames using a smooth zigzag pen path with a scrub radius."""
    import math

    waypoints = _build_zigzag_path(w, h, num_strokes, angle=angle)
    dense_path = _interpolate_path(waypoints, draw_frames)

    # Spacing along advance direction; scrub radius covers the gap between strokes
    theta = math.radians(max(0.5, min(89.5, angle)))
    sweep_dist = w * math.sin(theta) + h * math.cos(theta)
    stroke_spacing = sweep_dist / num_strokes
    scrub_radius = int(stroke_spacing * 0.65)

    bg = np.full_like(img, _detect_bg_color(img))
    mask = np.zeros((h, w), dtype=np.uint8)
    prev_pt = dense_path[0]

    total_frames = draw_frames + hold_frames

    for frame_idx in range(total_frames):
        if frame_idx < draw_frames:
            cur_pt = dense_path[frame_idx]
            cv2.line(mask, prev_pt, cur_pt, 255, thickness=2 * scrub_radius)
            cv2.circle(mask, cur_pt, scrub_radius, 255, -1)
            prev_pt = cur_pt

            mask_f = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
            canvas = (img * mask_f + bg * (1.0 - mask_f)).astype(np.uint8)

            if show_hand and hand_bgr is not None:
                frame = _overlay_hand(
                    canvas, hand_bgr, hand_alpha, cur_pt[0], cur_pt[1]
                )
            else:
                frame = canvas
        else:
            frame = img

        proc.stdin.write(frame.tobytes())


def _render_cell_frames(
    proc: subprocess.Popen,
    img: np.ndarray,
    w: int,
    h: int,
    cell_size: int,
    draw_frames: int,
    hold_frames: int,
    hand_bgr: np.ndarray | None,
    hand_alpha: np.ndarray | None,
    show_hand: bool,
    reveal: str,
) -> None:
    """Render frames using cell-grid progressive reveal (rows / knn)."""
    if reveal == "knn":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
        )
        visit_order = _build_knn_order(thresh, cell_size)
    else:
        visit_order = _build_rows_order(h, w, cell_size)

    total_cells = len(visit_order)
    if total_cells == 0:
        visit_order = [(0, 0)]
        total_cells = 1

    cells_per_frame = max(1, total_cells / max(1, draw_frames))
    total_frames = draw_frames + hold_frames

    canvas = np.full((h, w, 3), _detect_bg_color(img), dtype=np.uint8)
    cells_revealed = 0
    last_cx, last_cy = w // 2, h // 2

    for frame_idx in range(total_frames):
        if frame_idx < draw_frames:
            target_cells = min(int((frame_idx + 1) * cells_per_frame), total_cells)

            while cells_revealed < target_cells:
                r, c = visit_order[cells_revealed]
                y0 = r * cell_size
                x0 = c * cell_size
                y1 = min(y0 + cell_size, h)
                x1 = min(x0 + cell_size, w)
                canvas[y0:y1, x0:x1] = img[y0:y1, x0:x1]
                last_cx = x0 + cell_size // 2
                last_cy = y0 + cell_size // 2
                cells_revealed += 1

            if show_hand and hand_bgr is not None:
                frame = _overlay_hand(canvas, hand_bgr, hand_alpha, last_cx, last_cy)
            else:
                frame = canvas
        else:
            frame = canvas

        proc.stdin.write(frame.tobytes())


def animate_image(
    image_path: Path,
    output_path: Path,
    duration: float = 8.0,
    fps: int = 25,
    cell_size: int = 8,
    target_w: int | None = None,
    target_h: int | None = None,
    show_hand: bool = True,
    end_hold_fraction: float = 0.15,
    reveal: str = "zigzag",
    angle: float = 15.0,
) -> Path:
    """Convert a static image into a whiteboard hand-drawing animation.

    Args:
        image_path: Input image (PNG/JPG).
        output_path: Output video (MP4).
        duration: Total video duration in seconds.
        fps: Frames per second.
        cell_size: Grid cell size for progressive reveal (smaller = finer detail).
        target_w/target_h: Resize image to these dimensions (default: use original).
        show_hand: Whether to overlay the hand cursor.
        end_hold_fraction: Fraction of duration to hold the completed drawing.
        reveal: "zigzag" (path-based with scrub radius), "rows" (cell grid L↔R),
                or "knn" (cell grid nearest-neighbor).
        angle: Zigzag stroke tilt in degrees (0=horizontal rows, 90=vertical cols,
               30=slight diagonal). Only used when reveal="zigzag".

    Returns:
        output_path
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    if target_w and target_h:
        bg_fill = _detect_bg_color(img)
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = np.full((target_h, target_w, 3), bg_fill, dtype=np.uint8)
        y_off = (target_h - new_h) // 2
        x_off = (target_w - new_w) // 2
        img[y_off : y_off + new_h, x_off : x_off + new_w] = resized
        h, w = target_h, target_w

    # Make dimensions divisible by cell_size
    h = (h // cell_size) * cell_size
    w = (w // cell_size) * cell_size
    img = img[:h, :w]

    # Make dimensions even for h264
    h = h & ~1
    w = w & ~1
    img = img[:h, :w]

    total_frames = int(duration * fps)
    hold_frames = int(total_frames * end_hold_fraction)
    draw_frames = total_frames - hold_frames

    # Load hand overlay
    hand_bgr, hand_alpha = None, None
    if show_hand:
        hand_scale = min(0.9, 3 * max(0.15, min(0.35, 200 / max(w, h))))
        hand_bgr, hand_alpha = _load_hand(scale=hand_scale)

    # Start ffmpeg process for piping frames
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "bgr24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    use_path_reveal = reveal == "zigzag"

    try:
        if use_path_reveal:
            _render_zigzag_frames(
                proc,
                img,
                w,
                h,
                draw_frames,
                hold_frames,
                hand_bgr,
                hand_alpha,
                show_hand,
                angle=angle,
            )
        else:
            _render_cell_frames(
                proc,
                img,
                w,
                h,
                cell_size,
                draw_frames,
                hold_frames,
                hand_bgr,
                hand_alpha,
                show_hand,
                reveal,
            )
    finally:
        proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}): {stderr[-500:]}")

    print(
        f"  Doodle animation: {output_path} ({total_frames} frames, {duration}s)",
        flush=True,
    )
    return output_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert image to whiteboard drawing animation"
    )
    parser.add_argument("image", type=Path, help="Input image (PNG/JPG)")
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Output video path"
    )
    parser.add_argument(
        "--duration", type=float, default=8.0, help="Duration in seconds"
    )
    parser.add_argument("--fps", type=int, default=25, help="Frames per second")
    parser.add_argument("--cell-size", type=int, default=8, help="Grid cell size")
    parser.add_argument("--width", type=int, default=None, help="Target width")
    parser.add_argument("--height", type=int, default=None, help="Target height")
    parser.add_argument("--no-hand", action="store_true", help="Disable hand cursor")
    parser.add_argument(
        "--reveal",
        choices=["zigzag", "rows", "knn"],
        default="zigzag",
        help="Reveal: zigzag (path + scrub radius), rows (cell grid), knn (nearest-neighbor)",
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=30.0,
        help="Zigzag stroke angle in degrees (0=horizontal, 90=vertical, default=15)",
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Error: {args.image} not found")
        return

    output = args.output or args.image.parent / f"{args.image.stem}_doodle.mp4"
    animate_image(
        args.image,
        output,
        duration=args.duration,
        fps=args.fps,
        cell_size=args.cell_size,
        target_w=args.width,
        target_h=args.height,
        show_hand=not args.no_hand,
        reveal=args.reveal,
        angle=args.angle,
    )


if __name__ == "__main__":
    main()
