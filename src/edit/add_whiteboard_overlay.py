#!/usr/bin/env python3
"""
Add a whiteboard animation overlay to a talking-head video.

Pipeline:
  1. Get word-level transcript (Deepgram) if not already cached.
  2. LLM picks the best 10-second window + generates an image prompt.
  3. Generate a whiteboard image (Gemini) and doodle-animate it.
  4. Composite: whiteboard overlay + circle-cropped face bubble.
  5. QC with Gemini vision.

Usage:
  dotenvx run -f .env -- uv run python -m src.edit.add_whiteboard_overlay \\
      projects/HunterZier/.../segment_01/06_music.mp4
"""

import argparse
import asyncio
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from google import genai
from google.genai import types
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from PIL import Image, ImageDraw

from .doodle_animate import animate_image
from .face_detect import detect_faces_mediapipe
from .get_transcript import get_transcript, read_word_transcript_file


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _get_video_info(video_path: Path) -> dict:
    """Get video width, height, duration, fps via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    w, h = int(stream["width"]), int(stream["height"])
    num, den = map(int, stream["r_frame_rate"].split("/"))
    fps = num / den if den else 30.0
    duration = float(data["format"]["duration"])
    return {"width": w, "height": h, "fps": fps, "duration": duration}


def _get_words(video_path: Path) -> tuple[str, list[dict]]:
    """Load word-level transcript from cached JSON, or generate via Deepgram."""
    candidates = [
        video_path.parent / f"{video_path.stem}-words.json",
        video_path.parent / "05_captioned-words.json",
    ]
    for p in candidates:
        if p.exists():
            print(f"  Using cached transcript: {p}", flush=True)
            return read_word_transcript_file(p)

    print("  No cached transcript found — running Deepgram...", flush=True)
    result = asyncio.get_event_loop().run_until_complete(get_transcript(video_path))
    words = result["words"]

    out_path = video_path.parent / f"{video_path.stem}-words.json"
    out_path.write_text(json.dumps(words, indent=2))
    print(f"  Transcript saved: {out_path}", flush=True)

    transcript = " ".join(w["word"] for w in words)
    return transcript, words


async def _get_words_async(video_path: Path) -> tuple[str, list[dict]]:
    """Async version: load word-level transcript or generate via Deepgram."""
    candidates = [
        video_path.parent / f"{video_path.stem}-words.json",
        video_path.parent / "05_captioned-words.json",
    ]
    for p in candidates:
        if p.exists():
            print(f"  Using cached transcript: {p}", flush=True)
            return read_word_transcript_file(p)

    print("  No cached transcript found — running Deepgram...", flush=True)
    result = await get_transcript(video_path)
    words = result["words"]

    out_path = video_path.parent / f"{video_path.stem}-words.json"
    out_path.write_text(json.dumps(words, indent=2))
    print(f"  Transcript saved: {out_path}", flush=True)

    transcript = " ".join(w["word"] for w in words)
    return transcript, words


# ---------------------------------------------------------------------------
# Step 1 — Find the best overlay window via LLM
# ---------------------------------------------------------------------------


async def find_whiteboard_window(
    transcript_text: str,
    words: list[dict],
    duration: float,
    overlay_seconds: float = 12.0,
) -> dict:
    """Use an LLM to pick the best window for a whiteboard animation.

    Returns dict with keys: start, end, prompt, explanation.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    llm = ChatOpenAI(
        model="google/gemini-3-flash-preview",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=90.0,
    )

    words_with_ts = "\n".join(
        f"[{w['start']:.2f}-{w['end']:.2f}] {w['word']}" for w in words
    )

    prompt_guidance = (
        "Also write an image generation prompt for a whiteboard-style diagram "
        "that illustrates the concept being discussed.\n\n"
        "CRITICAL — VISUAL STYLE:\n"
        "- Generate the diagram AS the whiteboard: a flat graphic of the drawn "
        "content only. The viewer is looking directly at the drawing on a flat "
        "white surface.\n"
        "- Do NOT generate a photograph of a physical whiteboard. No whiteboard "
        "frame, no wall, no marker tray, no room, no perspective, no 3D context.\n"
        "- Output must look like a flat infographic or slide: just the hand-drawn "
        "content on a clean white background, as if the whiteboard surface fills "
        "the entire image.\n\n"
        "STYLE REQUIREMENTS:\n"
        "- Hand-drawn marker style (black with colored highlights: green, blue, red).\n"
        "- Clear text labels, headings, numbers, and short phrases.\n"
        "- Arrows, boxes, circles, checkmarks, icons, and simple diagrams.\n"
        "- Portrait orientation (4:5 aspect ratio, taller than wide).\n"
        "- Dense enough to be informative but not cluttered; 3-6 key visual elements.\n\n"
        "The image will be animated with a hand-drawing reveal effect, so design "
        "it as a finished diagram that looks good when drawn progressively.\n\n"
    )

    messages = [
        SystemMessage(
            content=(
                "You are an expert video editor. Given a word-level transcript with "
                "timestamps, find the single best contiguous window (exactly "
                f"{overlay_seconds:.0f} seconds) where a visual diagram would be "
                "most helpful to the viewer.\n\n"
                "Pick a section where the speaker explains a concept, compares "
                "options, lists items, or describes a process — something that would "
                "benefit from a visual diagram.\n\n"
                + prompt_guidance
                + f"Video duration: {duration:.1f}s\n"
                f"Overlay duration: {overlay_seconds:.0f}s\n\n"
                "Return ONLY valid JSON (no markdown fences) with these keys:\n"
                '  "start": <float seconds>,\n'
                '  "end": <float seconds>,\n'
                '  "prompt": "<generation prompt>",\n'
                '  "explanation": "<why this window was chosen>"\n'
            )
        ),
        HumanMessage(content=f"Word-level transcript:\n{words_with_ts}"),
    ]

    print("Finding best whiteboard overlay window...", flush=True)
    response = await llm.ainvoke(messages)
    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(text)

    result["start"] = float(result["start"])
    result["end"] = float(result["end"])
    if result["end"] - result["start"] > overlay_seconds + 1:
        result["end"] = result["start"] + overlay_seconds

    print(f"  Window: {result['start']:.1f}s – {result['end']:.1f}s", flush=True)
    print(f"  Explanation: {result['explanation']}", flush=True)
    return result


# ---------------------------------------------------------------------------
# Step 2 — Generate whiteboard image + doodle animation
# ---------------------------------------------------------------------------


async def generate_whiteboard(
    prompt: str,
    video_output_path: Path,
    duration: float = 12.0,
    reveal: str = "zigzag",
    angle: float = 15.0,
    image_model: str = "pro",
) -> Path:
    """Generate a whiteboard image (4:5) with Gemini and doodle-animate it.

    image_model: "pro" (gemini-3-pro, with flash fallback on transient errors)
    or "fast" (gemini-2.5-flash-image only).
    Output video is 4:5; compositor letterboxes it at the top of the 9:16 frame.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    image_path = video_output_path.parent / "whiteboard_image.png"

    if image_model == "pro":
        models_to_try = ["gemini-3-pro-image-preview", "gemini-2.5-flash-image"]
    else:
        models_to_try = ["gemini-2.5-flash-image"]
    max_retries = 3
    last_error: Exception | None = None
    generated = False

    for model_name in models_to_try:
        for attempt in range(1, max_retries + 1):
            try:
                print(
                    f"  Generating whiteboard image with {model_name} (attempt {attempt})...",
                    flush=True,
                )
                if attempt == 1:
                    print(f"  Prompt: {prompt[:150]}...", flush=True)

                style_prefix = (
                    "Flat graphic only: draw the diagram content on a white background. "
                    "Do not draw a photograph of a physical whiteboard—no frame, no wall, "
                    "no marker tray, no room. "
                )
                response = client.models.generate_content(
                    model=model_name,
                    contents=[f"Generate an image: {style_prefix}{prompt}"],
                    config=types.GenerateContentConfig(
                        image_config=types.ImageConfig(
                            aspect_ratio="4:5",
                        ),
                    ),
                )

                for part in response.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.mime_type.startswith(
                        "image/"
                    ):
                        raw = part.inline_data.data
                        img = Image.open(io.BytesIO(raw))
                        img.save(str(image_path))
                        print(
                            f"  Whiteboard image saved: {image_path} ({img.size})",
                            flush=True,
                        )
                        generated = True
                        break
                else:
                    text_parts = [
                        p.text for p in response.candidates[0].content.parts if p.text
                    ]
                    raise RuntimeError(
                        f"No image returned. Text: {' '.join(text_parts)[:300]}"
                    )

                if generated:
                    break
            except Exception as e:
                last_error = e
                is_transient = "503" in str(e) or "UNAVAILABLE" in str(e)
                if is_transient and attempt < max_retries:
                    wait = attempt * 5
                    print(f"  Transient error, retrying in {wait}s...", flush=True)
                    await asyncio.sleep(wait)
                elif is_transient:
                    print(
                        f"  {model_name} unavailable after {max_retries} attempts, "
                        "trying fallback...",
                        flush=True,
                    )
                    break
                else:
                    raise
        if generated:
            break

    if not generated:
        raise RuntimeError(
            f"All image generation models failed. Last error: {last_error}"
        )

    # Keep image 4:5 so the doodle hand only draws within that area.
    print(
        f"  Creating doodle animation (reveal={reveal}, angle={angle})...", flush=True
    )
    animate_image(
        image_path=image_path,
        output_path=video_output_path,
        duration=duration,
        fps=25,
        cell_size=8,
        show_hand=True,
        end_hold_fraction=0.42,
        reveal=reveal,
        angle=angle,
    )

    return video_output_path


def _reanimate_image(
    image_path: Path,
    video_output_path: Path,
    reveal: str = "zigzag",
    angle: float = 15.0,
) -> Path:
    """Re-run doodle animation on an existing 4:5 whiteboard image."""
    print(
        f"  Creating doodle animation (reveal={reveal}, angle={angle})...", flush=True
    )
    animate_image(
        image_path=image_path,
        output_path=video_output_path,
        duration=12.0,
        fps=25,
        cell_size=8,
        show_hand=True,
        end_hold_fraction=0.42,
        reveal=reveal,
        angle=angle,
    )
    return video_output_path


# ---------------------------------------------------------------------------
# Step 3 — Composite: whiteboard overlay + circle-cropped face bubble
# ---------------------------------------------------------------------------


def _detect_face_rect(
    video_path: Path, time_s: float
) -> tuple[int, int, int, int] | None:
    """Detect face bounding box at a given timestamp. Returns (x, y, w, h) or None."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_s * fps))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    ih, iw = frame.shape[:2]
    faces = detect_faces_mediapipe(frame, ih, iw)
    if not faces:
        return None
    x, y, w, h, _ = max(faces, key=lambda f: f[4])
    return (x, y, w, h)


def _create_circle_mask(size: int) -> Path:
    """Create a circle alpha mask PNG of given size."""
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    draw.ellipse([0, 0, size - 1, size - 1], fill=255)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    img.save(tmp.name)
    return Path(tmp.name)


def _create_face_bubble_overlay(size: int, border_width: int = 4) -> Path:
    """Create a RGBA PNG with a white circle border ring (transparent center + outside)."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse(
        [0, 0, size - 1, size - 1],
        fill=None,
        outline=(255, 255, 255, 255),
        width=border_width,
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    img.save(tmp.name)
    return Path(tmp.name)


def _detect_bg_color(image_path: Path) -> str:
    """Detect dominant background color from a whiteboard image.

    Samples thin border strips, takes the median color. Returns an ffmpeg-
    compatible hex string like '0xF5F3EE'.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return "white"
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
    median = np.median(pixels, axis=0).astype(int)
    b, g, r = median
    hex_color = f"0x{r:02X}{g:02X}{b:02X}"
    print(f"  Detected whiteboard BG color: {hex_color}", flush=True)
    return hex_color


def _create_letterbox_from_bottom_strip(
    wb_image_path: Path, vw: int, vh: int, strip_rows: int = 5
) -> Path:
    """Create a letterbox image by replicating the bottom strip of the 4:5 image.

    The whiteboard image often has non-uniform BG; averaging the bottom few
    rows and replicating them vertically gives a letterbox that matches.
    """
    img = cv2.imread(str(wb_image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {wb_image_path}")
    ih, iw = img.shape[:2]
    strip = img[-strip_rows:, :].astype(np.float32).mean(axis=0)  # (iw, 3)
    strip = np.clip(strip, 0, 255).astype(np.uint8)
    # One row at source width, then resize to (vw, 1)
    row = strip.reshape(1, iw, 3)
    row = cv2.resize(row, (vw, 1), interpolation=cv2.INTER_AREA)
    content_h = int(vw * 5 / 4)
    letterbox_h = vh - content_h
    if letterbox_h <= 0:
        letterbox_h = 1
    letterbox = np.tile(row, (letterbox_h, 1, 1))
    out_path = Path(tempfile.mktemp(suffix=".letterbox.png"))
    cv2.imwrite(str(out_path), letterbox)
    return out_path


def composite_whiteboard_with_face(
    talking_head_path: Path,
    whiteboard_path: Path,
    output_path: Path,
    overlay_start: float,
    overlay_end: float,
    face_bubble_size_pct: float = 0.30,
    face_padding_pct: float = 0.8,
    margin_right_pct: float = 0.09,
    margin_bottom_pct: float = 0.16,
    bg_color: str = "white",
    letterbox_path: Path | None = None,
) -> Path:
    """Overlay whiteboard on talking head with circle-cropped face bubble.

    Two-pass approach:
      Pass 1: Create a short face-bubble video (only the overlay duration).
      Pass 2: Overlay whiteboard + face-bubble onto the talking head.

    When whiteboard is 4:5 and letterbox_path is set, the letterbox is built
    from the image's bottom strip (replicated) and stacked below the scaled
    whiteboard. Otherwise the bottom is padded with bg_color.
    """
    info = _get_video_info(talking_head_path)
    vw, vh = info["width"], info["height"]
    overlay_dur = overlay_end - overlay_start

    face_rect = _detect_face_rect(talking_head_path, overlay_start + 1.0)

    bubble_d = int(vw * face_bubble_size_pct)
    bubble_d = bubble_d if bubble_d % 2 == 0 else bubble_d + 1
    margin_x = int(vw * margin_right_pct)
    margin_y = int(vh * margin_bottom_pct)

    bubble_x = vw - bubble_d - margin_x
    bubble_y = vh - bubble_d - margin_y

    if face_rect:
        fx, fy, fw, fh = face_rect
        pad = int(max(fw, fh) * face_padding_pct)
        crop_cx = fx + fw // 2
        crop_cy = fy + fh // 2
        crop_size = max(fw, fh) + 2 * pad
        crop_size = min(crop_size, vw, vh)
        crop_x = max(0, crop_cx - crop_size // 2)
        crop_y = max(0, crop_cy - crop_size // 2)
        if crop_x + crop_size > vw:
            crop_x = vw - crop_size
        if crop_y + crop_size > vh:
            crop_y = vh - crop_size
        print(
            f"  Face detected: ({fx},{fy},{fw},{fh}), bubble crop: ({crop_x},{crop_y},{crop_size})",
            flush=True,
        )
    else:
        crop_size = bubble_d
        crop_x = (vw - crop_size) // 2
        crop_y = (vh - crop_size) // 2
        print("  WARNING: No face detected, using center crop for bubble", flush=True)

    mask_path = _create_circle_mask(bubble_d)
    border_w = max(3, bubble_d // 30)
    ring_path = _create_face_bubble_overlay(bubble_d, border_width=border_w)

    try:
        face_bubble_path = Path(tempfile.mktemp(suffix=".mov"))

        face_filter = (
            f"[0:v]trim=start={overlay_start}:end={overlay_end},setpts=PTS-STARTPTS,"
            f"crop={crop_size}:{crop_size}:{crop_x}:{crop_y},"
            f"scale={bubble_d}:{bubble_d}[face_s];"
            f"[1:v]scale={bubble_d}:{bubble_d}[mask_s];"
            f"[face_s][mask_s]alphamerge[face_c];"
            f"[2:v]scale={bubble_d}:{bubble_d}[ring_s];"
            f"[ring_s][face_c]overlay=0:0[bubble]"
        )

        cmd_pass1 = [
            "ffmpeg",
            "-y",
            "-i",
            str(talking_head_path),
            "-i",
            str(mask_path),
            "-i",
            str(ring_path),
            "-filter_complex",
            face_filter,
            "-map",
            "[bubble]",
            "-an",
            "-c:v",
            "png",
            "-t",
            str(overlay_dur),
            str(face_bubble_path),
        ]

        print(
            f"  Pass 1: Creating face bubble clip ({overlay_dur:.1f}s)...", flush=True
        )
        r1 = subprocess.run(cmd_pass1, capture_output=True, text=True)
        if r1.returncode != 0:
            print(f"  FFmpeg pass1 stderr:\n{r1.stderr}", flush=True)
            raise subprocess.CalledProcessError(r1.returncode, cmd_pass1)
        print(f"  Pass 1 done: {face_bubble_path}", flush=True)

        fade_dur = 0.4

        wb_info = _get_video_info(whiteboard_path)
        wb_w, wb_h = wb_info["width"], wb_info["height"]
        wb_ratio = wb_w / wb_h if wb_h else 1
        use_letterbox = (
            letterbox_path is not None
            and letterbox_path.exists()
            and 0.75 <= wb_ratio <= 0.85
        )

        if use_letterbox:
            content_h = int(vw * 5 / 4)
            letterbox_h = vh - content_h
            if letterbox_h < 1:
                letterbox_h = 1
            # [1]=wb video → scale exactly to (vw, content_h); no pad (black pad caused a visible seam)
            wb_scale = f"[1:v]scale={vw}:{content_h}," f"format=rgba[wb_scaled];"
            lb_scale = (
                f"[2:v]scale={vw}:{letterbox_h},format=rgba[lb];"
                f"[wb_scaled][lb]vstack[wb_full];"
            )
            wb_fade = (
                f"[wb_full]fade=t=in:st=0:d={fade_dur}:alpha=1,"
                f"fade=t=out:st={overlay_dur - fade_dur}:d={fade_dur}:alpha=1,"
                f"setpts=PTS+{overlay_start}/TB[wb];"
            )
            comp_filter = (
                wb_scale + lb_scale + wb_fade + f"[3:v]format=rgba,"
                f"fade=t=in:st=0:d={fade_dur}:alpha=1,"
                f"fade=t=out:st={overlay_dur - fade_dur}:d={fade_dur}:alpha=1,"
                f"setpts=PTS+{overlay_start}/TB[bub];"
                f"[0:v][wb]overlay=0:0:eof_action=pass[v1];"
                f"[v1][bub]overlay={bubble_x}:{bubble_y}:eof_action=pass[vout]"
            )
            cmd_pass2 = [
                "ffmpeg",
                "-y",
                "-i",
                str(talking_head_path),
                "-i",
                str(whiteboard_path),
                "-loop",
                "1",
                "-t",
                str(overlay_dur),
                "-i",
                str(letterbox_path),
                "-i",
                str(face_bubble_path),
                "-filter_complex",
                comp_filter,
                "-map",
                "[vout]",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-c:a",
                "copy",
                str(output_path),
            ]
        else:
            # 4:5 without letterbox or non-4:5 → scale and pad with bg_color
            if 0.75 <= wb_ratio <= 0.85:
                wb_scale = (
                    f"[1:v]scale={vw}:-1:force_original_aspect_ratio=decrease,"
                    f"pad={vw}:{vh}:0:0:color={bg_color},"
                    f"format=rgba,"
                )
            else:
                wb_scale = (
                    f"[1:v]scale={vw}:{vh}:force_original_aspect_ratio=decrease,"
                    f"pad={vw}:{vh}:(ow-iw)/2:(oh-ih)/2:color={bg_color},"
                    f"format=rgba,"
                )
            comp_filter = (
                wb_scale + f"fade=t=in:st=0:d={fade_dur}:alpha=1,"
                f"fade=t=out:st={overlay_dur - fade_dur}:d={fade_dur}:alpha=1,"
                f"setpts=PTS+{overlay_start}/TB[wb];"
                f"[2:v]format=rgba,"
                f"fade=t=in:st=0:d={fade_dur}:alpha=1,"
                f"fade=t=out:st={overlay_dur - fade_dur}:d={fade_dur}:alpha=1,"
                f"setpts=PTS+{overlay_start}/TB[bub];"
                f"[0:v][wb]overlay=0:0:eof_action=pass[v1];"
                f"[v1][bub]overlay={bubble_x}:{bubble_y}:eof_action=pass[vout]"
            )
            cmd_pass2 = [
                "ffmpeg",
                "-y",
                "-i",
                str(talking_head_path),
                "-i",
                str(whiteboard_path),
                "-i",
                str(face_bubble_path),
                "-filter_complex",
                comp_filter,
                "-map",
                "[vout]",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-c:a",
                "copy",
                str(output_path),
            ]

        print(f"  Pass 2: Compositing final video...", flush=True)
        r2 = subprocess.run(cmd_pass2, capture_output=True, text=True)
        if r2.returncode != 0:
            print(f"  FFmpeg pass2 stderr:\n{r2.stderr}", flush=True)
            raise subprocess.CalledProcessError(r2.returncode, cmd_pass2)

        print(f"  Output: {output_path}", flush=True)
        return output_path
    finally:
        mask_path.unlink(missing_ok=True)
        ring_path.unlink(missing_ok=True)
        if face_bubble_path.exists():
            face_bubble_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Step 4 — QC with Gemini
# ---------------------------------------------------------------------------


async def qc_video(video_path: Path) -> str:
    """Upload video to Gemini and get QC feedback."""
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    print("Uploading video for QC review...")
    uploaded_file = client.files.upload(file=str(video_path))

    deadline = time.monotonic() + 5 * 60
    while True:
        state = getattr(uploaded_file.state, "name", str(uploaded_file.state))
        if state == "ACTIVE":
            break
        if state not in {"PROCESSING", "PROCESSING_UPLOAD"}:
            raise RuntimeError(f"File processing failed: {state}")
        if time.monotonic() >= deadline:
            raise TimeoutError("File processing timed out")
        await asyncio.sleep(2)
        uploaded_file = client.files.get(name=uploaded_file.name)

    prompt = (
        "You are a video QC reviewer. This video is a social media short (9:16 portrait) "
        "featuring a talking head with a whiteboard animation overlay that appears for "
        "about 12 seconds during the video.\n\n"
        "During the overlay section:\n"
        "- The whiteboard animation should take up most of the screen\n"
        "- A circle-cropped face of the speaker should appear in the bottom-right corner\n"
        "- The speaker's audio should continue playing\n\n"
        "Evaluate the video on these criteria:\n"
        "1. TIMING: Does the whiteboard appear at a moment that makes sense for the content?\n"
        "2. VISUAL QUALITY: Does the whiteboard animation look professional and relevant?\n"
        "3. FACE BUBBLE: Is the circle-cropped face visible and well-positioned?\n"
        "4. TRANSITION: Are the transitions in/out of the overlay smooth?\n"
        "5. OVERALL: Does it look like a polished social media video?\n\n"
        "Rate each criterion 1-5 and provide specific, actionable feedback.\n"
        "Give an overall score (1-5) and list any critical issues that must be fixed."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=types.Content(
            parts=[
                types.Part(file_data=types.FileData(file_uri=uploaded_file.uri)),
                types.Part(text=prompt),
            ]
        ),
    )

    feedback = response.text or "No feedback generated."
    print(f"\n{'='*60}")
    print("QC FEEDBACK")
    print(f"{'='*60}")
    print(feedback)
    print(f"{'='*60}\n")
    return feedback


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def run(
    video_path: Path,
    output_path: Path | None = None,
    skip_generation: bool = False,
    overlay_start: float | None = None,
    overlay_end: float | None = None,
    reveal: str = "zigzag",
    angle: float = 15.0,
    image_model: str = "pro",
) -> Path:
    """Full pipeline: transcript → window → generate → composite → QC."""
    info = _get_video_info(video_path)
    print(f"Input: {video_path}", flush=True)
    print(
        f"  Size: {info['width']}x{info['height']}, {info['duration']:.1f}s, {info['fps']:.1f}fps",
        flush=True,
    )

    transcript, words = await _get_words_async(video_path)
    print(f"  Transcript: {len(words)} words, {len(transcript)} chars", flush=True)

    wb_path = video_path.parent / "whiteboard_overlay.mp4"

    if overlay_start is not None and overlay_end is not None:
        window = {"start": overlay_start, "end": overlay_end}
        print(
            f"  Using provided window: {overlay_start:.1f}s–{overlay_end:.1f}s",
            flush=True,
        )
    else:
        window = await find_whiteboard_window(
            transcript, words, info["duration"], overlay_seconds=12.0
        )

    wb_image = video_path.parent / "whiteboard_image.png"

    if skip_generation and wb_path.exists():
        print(f"  Skipping generation (using existing {wb_path})", flush=True)
    elif "prompt" in window:
        await generate_whiteboard(
            window["prompt"],
            wb_path,
            reveal=reveal,
            angle=angle,
            image_model=image_model,
        )
    elif wb_image.exists():
        # Re-animate existing image (e.g. when --start/--end override the window)
        print(f"  Re-animating existing image: {wb_image}", flush=True)
        _reanimate_image(wb_image, wb_path, reveal=reveal, angle=angle)
    else:
        raise ValueError(
            "No prompt available and no existing whiteboard image to re-animate. "
            "Run without --start/--end first to generate the image."
        )

    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_whiteboard.mp4"

    bg_color = _detect_bg_color(wb_image) if wb_image.exists() else "white"
    vw, vh = info["width"], info["height"]
    letterbox_path: Path | None = None
    if wb_image.exists():
        letterbox_path = _create_letterbox_from_bottom_strip(wb_image, vw, vh)

    try:
        composite_whiteboard_with_face(
            talking_head_path=video_path,
            whiteboard_path=wb_path,
            output_path=output_path,
            overlay_start=window["start"],
            overlay_end=window["end"],
            bg_color=bg_color,
            letterbox_path=letterbox_path,
        )
    finally:
        if letterbox_path is not None and letterbox_path.exists():
            letterbox_path.unlink(missing_ok=True)

    await qc_video(output_path)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add whiteboard animation overlay to talking-head video"
    )
    parser.add_argument("video", type=Path, help="Input talking-head video (9:16)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <input>_whiteboard.mp4)",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation if whiteboard_overlay.mp4 already exists",
    )
    parser.add_argument(
        "--reveal",
        choices=["zigzag", "rows", "knn"],
        default="zigzag",
        help="Doodle reveal: zigzag (path + scrub radius), rows, or knn",
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=15.0,
        help="Zigzag stroke angle 0-90 (0=horizontal, 90=vertical, default=15)",
    )
    parser.add_argument(
        "--image-model",
        choices=["fast", "pro"],
        default="pro",
        help="Gemini image model: pro (pro with flash fallback) or fast (flash only)",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Override overlay start (seconds)",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="Override overlay end (seconds)",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: {args.video} not found")
        sys.exit(1)

    asyncio.run(
        run(
            args.video,
            args.output,
            args.skip_generation,
            args.start,
            args.end,
            reveal=args.reveal,
            angle=args.angle,
            image_model=args.image_model,
        )
    )


if __name__ == "__main__":
    main()
