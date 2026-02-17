#!/usr/bin/env python3
"""
Add a title overlay to a video.

If no title is provided, automatically generates one from the video transcript using an LLM.
The title will be displayed centered on the screen for the first 2 seconds.
Long titles will automatically wrap to multiple lines.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from deepgram import DeepgramClient
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .encode_common import H264_SOCIAL_COLOR_ARGS
from .settings_loader import load_settings


# Try to load from .env file if available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

API_KEY = "37e776c73c0de03eeacfaa9635e26ce6787bcf74"


async def transcribe_video(audio_path: Path) -> str:
    """Transcribe audio file using Deepgram and return transcript."""
    client = DeepgramClient(api_key=API_KEY)

    with audio_path.open("rb") as audio_file:
        audio_bytes = audio_file.read()

    response = client.listen.v1.media.transcribe_file(
        request=audio_bytes,
        model="nova-2",
        smart_format=True,
        punctuate=True,
    )

    if not response or not response.results:
        raise RuntimeError("Deepgram returned empty response")

    channel = response.results.channels[0]
    alternative = channel.alternatives[0]

    return alternative.transcript or ""


async def generate_title_from_transcript(
    transcript: str, count: int = 1, voice_file: Path | None = None
) -> str | list[str]:
    """Generate title(s) from video transcript using LLM.

    Args:
        transcript: Video transcript text
        count: Number of title suggestions to generate (default: 1)
        voice_file: Path to a voice/style markdown file (optional)

    Returns:
        Single title string if count=1, list of titles if count>1
    """
    import time

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")

    # Load voice file if provided
    voice_instructions = ""
    if voice_file and voice_file.exists():
        voice_instructions = (
            "\n\nVOICE & STYLE GUIDE (follow these rules strictly):\n"
            + voice_file.read_text()
        )
        print(f"🔧 DEBUG [title]: Loaded voice file: {voice_file}")

    print(f"🔧 DEBUG [title]: Initializing LLM for title generation (count={count})...")
    llm_start = time.time()
    llm = ChatOpenAI(
        model="google/gemini-3-flash-preview",
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        timeout=90.0,  # Increased to 90 seconds
    )
    print(f"🔧 DEBUG [title]: LLM initialized ({time.time() - llm_start:.2f}s)")

    if count == 1:
        print(f"🔧 DEBUG [title]: Building prompt for single title...")
        messages = [
            SystemMessage(
                content="You are a video title generator. Create a concise title "
                "for a video based on its transcript. The title should be clear "
                "and suitable for social media. Keep it under 10 words. "
                "Return ONLY the title text, nothing else." + voice_instructions
            ),
            HumanMessage(
                content=f"Generate a title for this video transcript:\n\n{transcript}"
            ),
        ]
        print(
            f"🔧 DEBUG [title]: Calling LLM (transcript length: {len(transcript)} chars)..."
        )
        title_call_start = time.time()
        try:
            response = await llm.ainvoke(messages)
            title_call_duration = time.time() - title_call_start
            print(f"🔧 DEBUG [title]: LLM call completed ({title_call_duration:.1f}s)")
        except Exception as e:
            title_call_duration = time.time() - title_call_start
            print(f"❌ ERROR [title]: LLM call failed after {title_call_duration:.1f}s")
            print(f"❌ ERROR [title]: Exception: {type(e).__name__}: {str(e)}")
            raise
        title = response.content.strip()
        # Remove quotes if LLM added them
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1]
        if title.startswith("'") and title.endswith("'"):
            title = title[1:-1]
        return title
    else:
        print(f"🔧 DEBUG [title]: Building prompt for {count} titles...")
        messages = [
            SystemMessage(
                content="You are a video title generator. Create multiple concise titles "
                "for a video based on its transcript. Each title should be clear "
                "and suitable for social media. Keep each title under 10 words. "
                "Return ONLY a numbered list of titles (1. Title 1, 2. Title 2, etc.), nothing else."
                + voice_instructions
            ),
            HumanMessage(
                content=f"Generate {count} different titles for this video transcript:\n\n{transcript}"
            ),
        ]
        print(
            f"🔧 DEBUG [title]: Calling LLM for {count} titles (transcript length: {len(transcript)} chars)..."
        )
        title_call_start = time.time()
        try:
            response = await llm.ainvoke(messages)
            title_call_duration = time.time() - title_call_start
            print(f"🔧 DEBUG [title]: LLM call completed ({title_call_duration:.1f}s)")
        except Exception as e:
            title_call_duration = time.time() - title_call_start
            print(f"❌ ERROR [title]: LLM call failed after {title_call_duration:.1f}s")
            print(f"❌ ERROR [title]: Exception: {type(e).__name__}: {str(e)}")
            raise
        content = response.content.strip()

        # Parse numbered list
        titles = []
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove numbering (1., 2., etc.)
            for prefix in [f"{i}. " for i in range(1, count + 1)]:
                if line.startswith(prefix):
                    title = line[len(prefix) :].strip()
                    # Remove quotes
                    if title.startswith('"') and title.endswith('"'):
                        title = title[1:-1]
                    if title.startswith("'") and title.endswith("'"):
                        title = title[1:-1]
                    titles.append(title)
                    break

        # If parsing failed, try splitting by newlines and taking first N
        if len(titles) < count:
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            titles = lines[:count]
            # Clean up titles
            titles = [t.lstrip("0123456789. ").strip("\"'") for t in titles]

        return titles[:count]


def _get_video_size(video_path: Path) -> tuple[int, int]:
    """Get video width and height via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    # Output is two lines: width, then height
    lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
    if len(lines) >= 2:
        return int(lines[0]), int(lines[1])
    raise ValueError(f"ffprobe did not return width,height: {result.stdout!r}")


def _create_rounded_title_image(
    video_w: int,
    video_h: int,
    lines: list[str],
    height_percent: int,
    anchor: str = "top",
    dark: bool = False,
) -> Path:
    """Create a PNG overlay: rounded rect + text. Returns path to temp file.

    Args:
        dark: If True, uses black background with white text (good for screen content overlays).
              If False (default), uses white background with black text.
    """
    roboto_bold_path = "/Users/apple/Downloads/Roboto/static/Roboto-Bold.ttf"
    # Keep title well inside video bounds: use 80% width with side margins
    margin_pct = 0.10  # 10% margin each side
    box_w = int(video_w * (1 - 2 * margin_pct))
    box_left = int(video_w * margin_pct)
    box_right = box_left + box_w

    # Scale font/line spacing with video height (base 1080p)
    scale = video_h / 1080.0
    padding = int(20 * scale)
    radius = int(20 * scale)
    line_spacing = int(48 * scale)
    num_lines = len(lines)
    box_h = num_lines * line_spacing + 2 * padding
    # height_percent: 0 = bottom, 100 = top
    # anchor="top"   → height_percent positions the TOP edge of the box
    # anchor="bottom" → height_percent positions the BOTTOM edge (box grows upward)
    # anchor="center" → box is vertically centered (height_percent ignored)
    if anchor == "bottom":
        # Bottom edge at height_percent from bottom → box grows upward
        box_bottom = video_h - (height_percent * video_h // 100)
        box_top = box_bottom - box_h
        # Clamp so box doesn't go above the frame
        if box_top < 0:
            box_top = 0
            box_bottom = box_h
    elif anchor == "center":
        # Vertically centered
        box_top = (video_h - box_h) // 2
        box_bottom = box_top + box_h
    else:
        # Top edge at height_percent from bottom (legacy behavior)
        top_from_bottom = video_h * (100 - height_percent) // 100
        box_top = top_from_bottom
        box_bottom = box_top + box_h
        # Clamp so box doesn't go below the frame
        if box_bottom > video_h:
            box_bottom = video_h
            box_top = video_h - box_h

    # Choose font size so ALL lines fit inside box width (inner = box_w - 2*padding)
    inner_w = box_w - 2 * padding
    font_size = max(36, int(60 * scale))
    try:
        font = ImageFont.truetype(roboto_bold_path, font_size)
    except OSError:
        font = ImageFont.load_default()

    draw_temp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    # Allow shrinking down to 12 so small videos (e.g. 202x360) still fit
    while font_size > 12:
        all_fit = True
        for line in lines:
            bbox = draw_temp.textbbox((0, 0), line, font=font)
            if (bbox[2] - bbox[0]) > inner_w:
                all_fit = False
                break
        if all_fit:
            break
        font_size -= 2
        try:
            font = ImageFont.truetype(roboto_bold_path, font_size)
        except OSError:
            break
    # Recreate font at final size for drawing
    try:
        font = ImageFont.truetype(roboto_bold_path, font_size)
    except OSError:
        font = ImageFont.load_default()

    img = Image.new("RGBA", (video_w, video_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Color scheme
    bg_color = (0, 0, 0, 255) if dark else (255, 255, 255, 255)
    text_color = (255, 255, 255, 255) if dark else (0, 0, 0, 255)

    # Draw rounded rectangle within bounds
    draw.rounded_rectangle(
        [box_left, box_top, box_right, box_bottom],
        radius=radius,
        fill=bg_color,
        outline=None,
    )

    # Draw each line centered in the box; clamp x so text stays inside box
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        line_h = bbox[3] - bbox[1]
        x = box_left + (box_w - line_w) // 2
        x = max(box_left + padding, min(x, box_right - padding - line_w))
        y = box_top + padding + i * line_spacing + (line_spacing - line_h) // 2
        draw.text((x, y), line, font=font, fill=text_color)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    img.save(tmp.name)
    return Path(tmp.name)


def add_title(
    video_path: Path,
    title_text: str,
    output_path: Path,
    duration: float = 20.0,
    height_percent: int = 10,
    anchor: str = "top",
    dark: bool = False,
) -> None:
    """Add title overlay to video.

    Args:
        video_path: Input video file
        title_text: Title text to display
        output_path: Output video file
        duration: How long to show title (in seconds, default 2.0)
        height_percent: Vertical position (0-100, where 0 is bottom, 100 is top, default: 10)
        anchor: Which edge of the box height_percent refers to ('top', 'bottom', or 'center' for vertically centered)
        dark: If True, black background with white text
    """
    title_text = title_text.strip()
    # Keep lines short so title fits in safe zone and doesn't overflow (platform chrome avoids bottom ~25%)
    max_chars_per_line = 20

    # If user entered multiple lines (e.g. from text_area), respect them
    if "\n" in title_text:
        raw_lines = [
            line.strip().upper() for line in title_text.split("\n") if line.strip()
        ]
        lines = []
        for raw_line in raw_lines:
            # Optionally wrap long lines
            if len(raw_line) <= max_chars_per_line:
                lines.append(raw_line)
            else:
                words = raw_line.split()
                current_line = []
                current_length = 0
                for word in words:
                    word_len = len(word)
                    if (
                        current_length + word_len + 1 > max_chars_per_line
                        and current_line
                    ):
                        lines.append(" ".join(current_line))
                        current_line = [word]
                        current_length = word_len
                    else:
                        current_line.append(word)
                        current_length += word_len + (1 if current_line else 0)
                if current_line:
                    lines.append(" ".join(current_line))
    else:
        # Single line: word-wrap at max_chars_per_line
        words = title_text.upper().split()
        lines = []
        current_line = []
        current_length = 0
        for word in words:
            word_len = len(word)
            if current_length + word_len + 1 > max_chars_per_line and current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = word_len
            else:
                current_line.append(word)
                current_length += word_len + (1 if current_line else 0)
        if current_line:
            lines.append(" ".join(current_line))

    video_w, video_h = _get_video_size(video_path)
    overlay_path = _create_rounded_title_image(
        video_w, video_h, lines, height_percent, anchor, dark
    )

    try:
        # Overlay the rounded-title PNG on the video (FFmpeg drawtext has no rounded box)
        vf = f"[0:v][1:v]overlay=0:0:enable='between(t,0,{duration})'[v]"
        print(f"Adding title overlay: '{title_text.upper()}' ({duration}s)")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(overlay_path),
                "-filter_complex",
                vf,
                "-map",
                "[v]",
                "-map",
                "0:a?",
                "-c:v",
                "h264_videotoolbox",
                *H264_SOCIAL_COLOR_ARGS,
                "-c:a",
                "copy",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error (stderr): {e.stderr}")
        print(f"FFmpeg error (stdout): {e.stdout}")
        raise
    finally:
        overlay_path.unlink(missing_ok=True)


def read_transcript_file(transcript_path: Path) -> str:
    """Read transcript from file.

    Supports both JSON (utterances format) and plain text formats.
    """
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    content = transcript_path.read_text()

    # Try to parse as JSON (utterances format)
    try:
        import json

        data = json.loads(content)
        # If it's a list of utterances, extract text from each
        if isinstance(data, list) and len(data) > 0 and "text" in data[0]:
            return "\n".join([u["text"] for u in data])
        # If it's a single object with transcript field
        if isinstance(data, dict) and "transcript" in data:
            return data["transcript"]
    except (json.JSONDecodeError, KeyError):
        pass

    # Otherwise treat as plain text
    return content


async def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print(
            "Usage: python add_title.py <video_file> [title_text] [output_file] [--duration <seconds>] [--transcript <file>] [--dry-run]"
        )
        print("  If title_text is not provided, it will be generated from transcript.")
        print(
            "  --transcript: Path to transcript file (JSON or TXT). If not provided, will transcribe video."
        )
        print("  --duration:  How long to show title (default: 20.0 seconds)")
        print("  --height:    Vertical position 0-100 (default: 10)")
        print(
            "  --anchor:    Which edge height positions — 'top', 'bottom', or 'center' (default: top)"
        )
        print(
            "  --dark:      Use dark style (black box, white text) instead of classic (white box)"
        )
        print("  --dry-run:   Only generate and print title, don't process video")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    # Defaults from project/session settings.json (CLI flags override)
    settings = load_settings(video_path)
    title_cfg = settings.get("title") or {}
    duration = float(title_cfg.get("duration", 20.0))
    height_percent = int(title_cfg.get("height_percent", 10))
    anchor = (title_cfg.get("anchor") or "top").lower()
    if anchor not in ("top", "bottom", "center"):
        anchor = "top"
    style = (title_cfg.get("style") or "classic").lower()
    dark = style == "dark"

    # Parse flags
    dry_run = "--dry-run" in sys.argv
    transcript_file = None
    if "--transcript" in sys.argv:
        idx = sys.argv.index("--transcript")
        if idx + 1 < len(sys.argv):
            transcript_file = Path(sys.argv[idx + 1])

    # Parse title (optional)
    title_text = None
    arg_idx = 2
    if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
        title_text = sys.argv[2]
        arg_idx = 3

    # Parse output path
    output_path = video_path.parent / f"{video_path.stem}-with-title.mp4"
    if len(sys.argv) > arg_idx and not sys.argv[arg_idx].startswith("--"):
        output_path = Path(sys.argv[arg_idx])
        arg_idx += 1

    # Override with CLI --duration
    if "--duration" in sys.argv:
        idx = sys.argv.index("--duration")
        if idx + 1 < len(sys.argv):
            try:
                duration = float(sys.argv[idx + 1])
            except ValueError:
                print(f"Error: Invalid duration value: {sys.argv[idx + 1]}")
                sys.exit(1)

    # Override with CLI --height
    if "--height" in sys.argv:
        idx = sys.argv.index("--height")
        if idx + 1 < len(sys.argv):
            try:
                height_percent = int(sys.argv[idx + 1])
            except ValueError:
                print(f"Error: Invalid height value: {sys.argv[idx + 1]}")
                sys.exit(1)

    # Parse voice file
    voice_file = None
    if "--voice" in sys.argv:
        idx = sys.argv.index("--voice")
        if idx + 1 < len(sys.argv):
            voice_file = Path(sys.argv[idx + 1])

    # Override with CLI --anchor
    if "--anchor" in sys.argv:
        idx = sys.argv.index("--anchor")
        if idx + 1 < len(sys.argv):
            anchor = sys.argv[idx + 1].lower()
            if anchor not in ("top", "bottom", "center"):
                print(
                    f"Error: --anchor must be 'top', 'bottom', or 'center', got '{anchor}'"
                )
                sys.exit(1)

    # Override with CLI --dark
    if "--dark" in sys.argv:
        dark = True

    # Generate title from transcript if not provided
    if not title_text:
        transcript = None

        if transcript_file:
            print(f"Reading transcript from file: {transcript_file}")
            transcript = read_transcript_file(transcript_file)
        else:
            print(
                "No title or transcript file provided. Generating title from video transcript..."
            )

            # Extract audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_tmp:
                audio_path = Path(audio_tmp.name)

            try:
                print("Extracting audio from video...")
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(video_path),
                        "-vn",
                        "-acodec",
                        "pcm_s16le",
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        str(audio_path),
                    ],
                    check=True,
                    capture_output=True,
                )

                print("Transcribing with Deepgram...")
                transcript = await transcribe_video(audio_path)

            finally:
                audio_path.unlink(missing_ok=True)

        print("Generating title with LLM (Gemini 3 Flash)...")
        if dry_run:
            titles = await generate_title_from_transcript(
                transcript, count=3, voice_file=voice_file
            )
            print("\n🔍 DRY RUN: Top 3 title suggestions:")
            for i, title in enumerate(titles, 1):
                print(f"   {i}. {title}")
            return
        else:
            title_text = await generate_title_from_transcript(
                transcript, voice_file=voice_file
            )
            print(f"Generated title: '{title_text}'")

    if not dry_run:
        add_title(
            video_path, title_text, output_path, duration, height_percent, anchor, dark
        )
        print(f"✅ Output video: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
