#!/usr/bin/env python3
"""
Face-aware 9:16 portrait crop.

Detects the face on the first frame of a video and returns crop parameters
so the 9:16 window is centered on the face (or center of frame if no face).
Uses MediaPipe Face Detection (Tasks API; more robust than Haar).

Usage:
  from src.edit.face_crop import get_portrait_crop_params
  x, w, h = get_portrait_crop_params(Path("segment.mp4"))
  # ffmpeg: crop=w:h:x:0
"""

import subprocess
import urllib.request
from pathlib import Path
from typing import Tuple

import cv2

# Type alias for (height_percent, anchor) for title placement
TitlePlacement = Tuple[int, str]
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions

# Short-range BlazeFace model (good for talking head). Downloaded on first use.
_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
_MODEL_PATH = Path(__file__).resolve().parent / "face_detection_short_range.tflite"


def _ensure_model() -> Path:
    if not _MODEL_PATH.exists():
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_FACE_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


def _detect_faces_mediapipe(
    frame: cv2.typing.MatLike, ih: int, iw: int
) -> list[tuple[int, int, int, int, float]]:
    """Run MediaPipe Face Detection; return list of (x, y, w, h, score)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if not rgb.flags.c_contiguous:
        rgb = rgb.copy(order="C")
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

    model_path = _ensure_model()
    base_opts = base_options_module.BaseOptions(model_asset_path=str(model_path))
    opts = FaceDetectorOptions(
        base_options=base_opts,
        min_detection_confidence=0.5,
    )
    detector = FaceDetector.create_from_options(opts)
    result = detector.detect(mp_image)
    detector.close()

    faces_px: list[tuple[int, int, int, int, float]] = []
    if not result.detections:
        return faces_px
    for det in result.detections:
        b = det.bounding_box
        x = max(0, min(b.origin_x, iw - 1))
        y = max(0, min(b.origin_y, ih - 1))
        w = max(1, min(b.width, iw - x))
        h = max(1, min(b.height, ih - y))
        score = det.categories[0].score if det.categories else 0.0
        faces_px.append((x, y, w, h, score))
    return faces_px


def get_portrait_crop_params(video_path: Path) -> Tuple[int, int, int]:
    """Compute 9:16 portrait crop (width, height, x_offset) from first frame.

    Uses MediaPipe Face Detection on the first frame. If a face is found,
    the crop is horizontally centered on the face (clamped to frame). If not,
    the crop is center-based.

    Returns:
        (crop_x, crop_w, crop_h) for ffmpeg crop=w:h:x:0. All even.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise ValueError(f"Could not read first frame: {video_path}")

    ih, iw = frame.shape[:2]
    crop_w = int(ih * 9 / 16) & ~1
    crop_h = ih & ~1

    faces = _detect_faces_mediapipe(frame, ih, iw)

    if len(faces) == 0:
        crop_x = (iw - crop_w) // 2
    else:
        # Pick highest-confidence face (avoids false positives like bottles)
        (x, y, w, h, _) = max(faces, key=lambda r: r[4])
        face_center_x = x + w // 2
        crop_x = face_center_x - crop_w // 2
        crop_x = max(0, min(crop_x, iw - crop_w))
    crop_x -= crop_x % 2
    return crop_x, crop_w, crop_h


def get_title_placement_from_face(
    video_path: Path,
    time_s: float = 1.0,
    title_box_height_pct: int = 12,
    gap_below_face_pct: int = 2,
    min_height_pct: int = 8,
) -> TitlePlacement:
    """Recommend title placement so the title sits just below the face.

    Uses MediaPipe face detection on a frame at time_s (default 1s, when
    title is visible). Returns (height_percent, anchor) for add_title:
    anchor="bottom" so the title box grows upward from height_percent.

    Args:
        video_path: Path to the video (e.g. 03_jumpcut.mp4).
        time_s: Timestamp to sample (default 1.0).
        title_box_height_pct: Approximate title box height as % of frame (default 12).
        gap_below_face_pct: Gap between face bottom and title top (default 2).
        min_height_pct: Minimum height_percent so title stays above platform chrome (default 8).

    Returns:
        (height_percent, "bottom") for add_title --height and --anchor.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return (20, "bottom")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_num = int(time_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return (20, "bottom")

    ih, iw = frame.shape[:2]
    faces = _detect_faces_mediapipe(frame, ih, iw)
    if not faces:
        return (20, "bottom")

    (x, y, w, h, _) = max(faces, key=lambda r: r[4])
    # Face bottom in pixel row (from top); percent from bottom = 0 at bottom, 100 at top
    face_bottom_px = y + h
    face_bottom_pct = 100 * (ih - face_bottom_px) / ih
    # We want title TOP just below face: title_bottom = face_bottom - gap - title_height
    want_bottom_pct = face_bottom_pct - gap_below_face_pct - title_box_height_pct
    height_percent = max(min_height_pct, min(int(want_bottom_pct), 50))
    return (height_percent, "bottom")


def get_face_crop_x_at_time(
    video_path: Path, time_s: float, crop_w: int, iw: int, ih: int
) -> int:
    """Detect face position at a specific timestamp and return the crop x offset.

    Args:
        video_path: Path to the video.
        time_s: Timestamp (seconds) to sample.
        crop_w: Target crop width (9:16 portrait).
        iw: Video width in pixels.
        ih: Video height in pixels.

    Returns:
        crop_x offset (even, clamped to frame bounds).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return (iw - crop_w) // 2 & ~1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_num = int(time_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        crop_x = (iw - crop_w) // 2
        return crop_x & ~1

    faces = _detect_faces_mediapipe(frame, ih, iw)

    if not faces:
        crop_x = (iw - crop_w) // 2
    else:
        (x, y, w, h, _) = max(faces, key=lambda r: r[4])
        face_center_x = x + w // 2
        crop_x = face_center_x - crop_w // 2
        crop_x = max(0, min(crop_x, iw - crop_w))

    return crop_x & ~1


def face_crop_per_cut(
    video_path: Path,
    output_path: Path,
    boundaries: list[float],
) -> Path:
    """Crop video to 9:16 portrait with per-cut face detection.

    Splits the video at boundary timestamps, detects the face at the start
    of each section, applies a section-specific crop, and concats the results
    in a single ffmpeg pass.

    Args:
        video_path: Input video (landscape).
        output_path: Output video (portrait 9:16).
        boundaries: Timestamps (in the input video timeline) where cuts
                     join — from apply_cuts boundaries.json.

    Returns:
        output_path.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    iw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ih = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()

    crop_w = int(ih * 9 / 16) & ~1
    crop_h = ih & ~1

    # Build sections from boundaries: [(start, end), ...]
    sections: list[tuple[float, float]] = []
    prev = 0.0
    for b in boundaries:
        if b > prev:
            sections.append((prev, b))
        prev = b
    if prev < duration:
        sections.append((prev, duration))

    if not sections:
        sections = [(0.0, duration)]

    # Detect face x-offset at the start of each section
    crop_xs: list[int] = []
    for start, _ in sections:
        crop_x = get_face_crop_x_at_time(video_path, start, crop_w, iw, ih)
        crop_xs.append(crop_x)
        print(f"  Face crop: t={start:.2f}s → crop_x={crop_x}")

    # If all crop_x values are the same, use a simple single-pass crop
    if len(set(crop_xs)) == 1:
        print(f"  All sections have same crop_x={crop_xs[0]}, using simple crop")
        crop_vf = f"crop={crop_w}:{crop_h}:{crop_xs[0]}:0"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                crop_vf,
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-c:a",
                "copy",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
        return output_path

    # Build per-section trim+crop filter, then concat
    video_parts: list[str] = []
    audio_parts: list[str] = []
    for i, ((start, end), crop_x) in enumerate(zip(sections, crop_xs)):
        video_parts.append(
            f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS,"
            f"crop={crop_w}:{crop_h}:{crop_x}:0[v{i}]"
        )
        audio_parts.append(
            f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}]"
        )

    n = len(sections)
    v_concat = "".join(f"[v{i}]" for i in range(n))
    a_concat = "".join(f"[a{i}]" for i in range(n))
    concat_filter = (
        f"{v_concat}concat=n={n}:v=1:a=0[vout];" f"{a_concat}concat=n={n}:v=0:a=1[aout]"
    )

    filter_complex = ";".join(video_parts + audio_parts) + ";" + concat_filter

    print(f"  Per-cut face crop: {n} sections, {len(set(crop_xs))} distinct positions")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )

    return output_path


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Get face-centered 9:16 crop params (MediaPipe)"
    )
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--draw",
        type=Path,
        help="Optional: save first frame with face box (green) and crop (red)",
    )
    args = parser.parse_args()
    x, w, h = get_portrait_crop_params(args.video)
    print(f"crop={w}:{h}:{x}:0")
    if args.draw:
        cap = cv2.VideoCapture(str(args.video))
        _, frame = cap.read()
        cap.release()
        ih, iw = frame.shape[:2]
        faces = _detect_faces_mediapipe(frame, ih, iw)
        for fx, fy, fw, fh, _ in faces:
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, 0), (x + w, h), (0, 0, 255), 2)
        cv2.imwrite(str(args.draw), frame)
        print(f"Debug image: {args.draw}", file=sys.stderr)


if __name__ == "__main__":
    main()
