#!/usr/bin/env python3
"""
Add a title overlay to a video.

If no title is provided, automatically generates one from the video transcript using an LLM.
The title will be displayed centered on the screen for the first 2 seconds.
Long titles will automatically wrap to multiple lines.
"""

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from deepgram import DeepgramClient
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

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
    transcript: str, count: int = 1
) -> str | list[str]:
    """Generate title(s) from video transcript using LLM.

    Args:
        transcript: Video transcript text
        count: Number of title suggestions to generate (default: 1)

    Returns:
        Single title string if count=1, list of titles if count>1
    """
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")

    llm = ChatOpenAI(
        model="google/gemini-3-flash-preview",
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )

    if count == 1:
        messages = [
            SystemMessage(
                content="You are a video title generator. Create a compelling, concise title "
                "for a video based on its transcript. The title should be engaging, "
                "clear, and suitable for social media. Keep it under 10 words. "
                "Return ONLY the title text, nothing else."
            ),
            HumanMessage(
                content=f"Generate a title for this video transcript:\n\n{transcript}"
            ),
        ]
        response = await llm.ainvoke(messages)
        title = response.content.strip()
        # Remove quotes if LLM added them
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1]
        if title.startswith("'") and title.endswith("'"):
            title = title[1:-1]
        return title
    else:
        messages = [
            SystemMessage(
                content="You are a video title generator. Create multiple compelling, concise titles "
                "for a video based on its transcript. Each title should be engaging, "
                "clear, and suitable for social media. Keep each title under 10 words. "
                "Return ONLY a numbered list of titles (1. Title 1, 2. Title 2, etc.), nothing else."
            ),
            HumanMessage(
                content=f"Generate {count} different titles for this video transcript:\n\n{transcript}"
            ),
        ]
        response = await llm.ainvoke(messages)
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


def add_title(
    video_path: Path,
    title_text: str,
    output_path: Path,
    duration: float = 5.0,
    height_percent: int = 10,
) -> None:
    """Add title overlay to video.

    Args:
        video_path: Input video file
        title_text: Title text to display
        output_path: Output video file
        duration: How long to show title (in seconds, default 2.0)
        height_percent: Vertical position (0-100, where 0 is bottom, 100 is top, default: 10)
    """
    # Split long titles into multiple lines (max ~30 chars per line)
    words = title_text.upper().split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        word_len = len(word)
        # If adding this word would exceed ~30 chars, start new line
        if current_length + word_len + 1 > 30 and current_line:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = word_len
        else:
            current_line.append(word)
            current_length += word_len + (1 if current_line else 0)

    if current_line:
        lines.append(" ".join(current_line))

    # Use multiple drawtext filters (one per line) since \n doesn't work reliably
    font_size = 50
    line_spacing = 60  # Spacing between lines
    num_lines = len(lines)
    # Total height of all lines
    total_height = num_lines * line_spacing

    filter_parts = []

    for i, line in enumerate(lines):
        # Escape special characters for ffmpeg drawtext
        # Use double quotes for text parameter to avoid apostrophe issues
        # When using double quotes, only need to escape: backslash, double quote, colon, percent, and square brackets
        # Apostrophes don't need escaping when using double quotes
        escaped_line = (
            line.replace("\\", "\\\\")  # Escape backslashes first
            .replace('"', '\\"')  # Escape double quotes
            .replace(":", "\\:")  # Escape colons
            .replace("%", "\\%")  # Escape percent signs
            .replace("[", "\\[")  # Escape square brackets
            .replace("]", "\\]")  # Escape square brackets
        )

        # Calculate y position for this line
        # height_percent: 0 = bottom, 100 = top
        # Position the TOP of the text block at height_percent from bottom
        # y = h * (100 - height_percent) / 100 + (i * line_spacing)
        # No need to subtract total_height/2 since we're positioning the top
        height_factor = 100 - height_percent
        y_offset_pixels = i * line_spacing
        # Use parentheses to group expression properly and avoid parsing issues
        if y_offset_pixels == 0:
            y_pos = f"h*{height_factor}/100"
        else:
            y_pos = f"h*{height_factor}/100+{y_offset_pixels}"

        # Create drawtext filter for this line
        # Use double quotes for text parameter to handle apostrophes correctly
        # For enable parameter, use escaped single quotes or double quotes
        title_filter = (
            f'drawtext=text="{escaped_line}":'
            f"font=Helvetica-Bold:"
            f"fontsize={font_size}:"
            f"fontcolor=black:"
            f"box=1:"
            f"boxcolor=white@1.0:"
            f"boxborderw=15:"
            f"text_align=center:"
            f"x=(w-text_w)/2:"  # Center horizontally
            f"y={y_pos}:"  # Position vertically
            f"enable='between(t,0,{duration})'"  # Show for specified duration - quotes needed to avoid parsing issues
        )
        filter_parts.append(title_filter)

    # Combine filters
    vf_filter = ",".join(filter_parts)

    # Apply title overlay to video
    print(f"Adding title overlay: '{title_text.upper()}' ({duration}s)")
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                vf_filter,
                "-c:v",
                "h264_videotoolbox",
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
        print("  --duration: How long to show title (default: 2.0 seconds)")
        print("  --dry-run: Only generate and print title, don't process video")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

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

    # Parse duration
    duration = 2.0
    if "--duration" in sys.argv:
        idx = sys.argv.index("--duration")
        if idx + 1 < len(sys.argv):
            try:
                duration = float(sys.argv[idx + 1])
            except ValueError:
                print(f"Error: Invalid duration value: {sys.argv[idx + 1]}")
                sys.exit(1)

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
            titles = await generate_title_from_transcript(transcript, count=3)
            print("\n🔍 DRY RUN: Top 3 title suggestions:")
            for i, title in enumerate(titles, 1):
                print(f"   {i}. {title}")
            return
        else:
            title_text = await generate_title_from_transcript(transcript)
            print(f"Generated title: '{title_text}'")

    if not dry_run:
        add_title(video_path, title_text, output_path, duration)
        print(f"✅ Output video: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
