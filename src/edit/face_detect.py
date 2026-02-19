"""
Face detection on a single frame using MediaPipe BlazeFace (short-range).

Used by portrait_crop (9:16 crop centering) and add_whiteboard_overlay (bubble placement).
"""

import urllib.request
from pathlib import Path

import cv2
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


def detect_faces_mediapipe(
    frame: cv2.typing.MatLike, ih: int, iw: int
) -> list[tuple[int, int, int, int, float]]:
    """Run MediaPipe Face Detection on a BGR frame; return list of (x, y, w, h, score)."""
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
