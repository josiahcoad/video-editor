#!/usr/bin/env python3
"""
Analyze video frames to suggest optimal title and caption placement.

Extracts frames at specified timestamps, annotates them with a percentage
grid so the LLM has spatial reference, sends them to Gemini, and returns
placement recommendations.

Usage:
  python check_placement.py <video_file>
  python check_placement.py <video_file> --timestamps 0 3 10
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from google import genai
from google.genai import types


def extract_frame(video_path: Path, timestamp: float, output_path: Path) -> None:
    """Extract a single frame from a video at the given timestamp."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(timestamp),
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )


def annotate_frame_with_grid(
    frame_path: Path,
    output_path: Path,
    step: int = 10,
) -> None:
    """Draw horizontal percentage grid lines on a frame.

    Lines are labelled with "% from bottom" so the LLM can visually map
    vertical positions to the percentage values we use in our pipeline.
    """
    img = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    # Try to load a readable font; fall back to default
    font_size = max(14, h // 60)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except OSError:
        font = ImageFont.load_default()

    for pct in range(step, 100, step):
        y = int(h * (1 - pct / 100))  # 0% = bottom, 100% = top
        # Semi-transparent line
        color = (255, 50, 50, 140) if pct % 20 == 0 else (255, 150, 150, 100)
        draw.line([(0, y), (w, y)], fill=color, width=2 if pct % 20 == 0 else 1)
        # Label on the right edge
        label = f"{pct}%"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (w - tw - 6, y - font_size - 2), label, fill=(255, 255, 255, 220), font=font
        )

    img.save(output_path, quality=92)


def analyze_placement(
    video_path: Path,
    timestamps: list[float],
    model: str = "gemini-3-flash-preview",
) -> str:
    """Extract frames, annotate with grid, and ask Gemini for placement advice."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable must be set")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Extract and annotate frames
        annotated_paths: list[Path] = []
        for ts in timestamps:
            raw = tmp / f"frame_{ts}s_raw.jpg"
            annotated = tmp / f"frame_{ts}s.jpg"
            print(f"Extracting frame at {ts}s...")
            extract_frame(video_path, ts, raw)
            annotate_frame_with_grid(raw, annotated, step=5)
            annotated_paths.append(annotated)

        # Build multimodal prompt
        parts: list[types.Part] = []
        for ts, fp in zip(timestamps, annotated_paths):
            img_bytes = fp.read_bytes()
            parts.append(
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/jpeg",
                        data=img_bytes,
                    )
                )
            )
            parts.append(types.Part(text=f"[Frame at {ts}s]"))

        parts.append(
            types.Part(
                text="""You are a video production expert. You're looking at frames from a short-form vertical video (720×1280, 9:16). Each frame has horizontal guide lines labelled with "% from bottom" (0 % = bottom edge, 100 % = top edge). Use these grid lines to give precise answers — read the nearest line and interpolate.

I need to place two text elements:
1. **TITLE** — a white rounded-rectangle badge with black text, shown for the first ~6 s.
2. **CAPTIONS** — white text with a black outline, shown throughout the video.

### How placement works
- The percentage value sets the **bottom edge** of the text element, measured from the bottom of the frame.
- Example: CAPTION_HEIGHT_PERCENT: 40 means the bottom of the caption text sits at the 40 % line. The text block is ~4 % tall, so it occupies roughly the 40–44 % band.

### Step 1: Identify landmarks in EACH frame
For each frame, identify these vertical zones using the grid lines:
- **Platform chrome zone**: bottom 0–8 % (always hidden by TikTok/Reels UI)
- **Speaker's face**: what % range does the face occupy? (e.g., "25–40 %")
- **Speaker's body/torso**: what % range?
- **On-screen content** (chart, slide, graphic): what % range does it occupy? If split-screen, where is the divider?
- **Empty/safe gaps**: what bands have no important visual content?

### Step 2: Find the optimal caption zone
Captions must:
- Be ABOVE the platform chrome (above 10 %)
- NOT overlap the speaker's face in ANY frame
- NOT overlap on-screen content (charts, graphics) in ANY frame
- Sit in the largest visual gap between content regions

For split-screen layouts (e.g., chart on top, speaker on bottom), the best position is usually right at the dividing line / seam between the two sections. Captions can slightly overlap the very edge of content — a 2-3 % overlap with a chart axis label or the top of the speaker's head is acceptable and preferable to placing captions far from the action. This typically means captions land in the 35–50 % range.

For full-screen talking head, captions go in the chest/neck area — below the chin but well above platform chrome. This is usually the 30–50 % range.

IMPORTANT: Do NOT default to placing captions near the bottom (10–25 %). That wastes vertical space and puts text in the platform chrome danger zone. Captions should be as close to the visual center as possible while respecting face/content constraints.

### Step 3: Title placement
The title appears for the first ~6 s only. Place it where the early frames have clear space. For talking-head frames, placing the title around 15–25 % (bottom anchor) often works well — it sits in the lower-middle area without covering the face.

### Response format (exactly this, no extra lines)
LANDMARKS:
- Frame at Xs: face Y1–Y2%, content Y3–Y4%, gap Y5–Y6%
- Frame at Xs: face Y1–Y2%, content Y3–Y4%, gap Y5–Y6%
- ...
TITLE_HEIGHT_PERCENT: <integer>
TITLE_ANCHOR: top|bottom
CAPTION_HEIGHT_PERCENT: <integer>
REASONING: <2-3 sentences explaining your placement using landmark analysis>"""
            )
        )

        print(f"Analyzing {len(timestamps)} frames with {model}...")
        response = client.models.generate_content(
            model=model,
            contents=types.Content(parts=parts),
            config=types.GenerateContentConfig(
                temperature=0.0,
                media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            ),
        )

        text = response.text or ""

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            print(
                f"  Tokens: {um.prompt_token_count} prompt, "
                f"{um.candidates_token_count} completion, "
                f"{um.total_token_count} total"
            )

        return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze video frames for title/caption placement"
    )
    parser.add_argument("video", type=Path, help="Video file to analyze")
    parser.add_argument(
        "--timestamps",
        nargs="+",
        type=float,
        default=[0, 3, 10],
        help="Timestamps (in seconds) to extract frames at (default: 0 3 10)",
    )
    parser.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="Gemini model (default: gemini-3-flash-preview)",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    result = analyze_placement(args.video, args.timestamps, args.model)
    print(f"\n{result}")


if __name__ == "__main__":
    main()
