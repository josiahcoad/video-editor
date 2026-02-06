#!/usr/bin/env python3
"""
Generate title suggestions for a video.

Runs independently of the main workflow:
1. Quick silence detection
2. Extract first 30 seconds of non-silent audio
3. Transcribe
4. Generate title suggestions via LLM
"""

import asyncio
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class TitleOptions(BaseModel):
    """Response model for title options."""

    titles: list[str] = Field(description="List of 3 title options")


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def detect_silence_ranges(video_path: Path) -> list[tuple[float, float]]:
    """Detect silence ranges in video. Returns list of (start, end) tuples."""
    # Extract low-quality audio for fast detection
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = Path(tmp.name)

    try:
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
                "8000",
                "-ac",
                "1",
                str(audio_path),
            ],
            check=True,
            capture_output=True,
        )

        # Detect silence
        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(audio_path),
                "-af",
                "silencedetect=noise=-30dB:d=0.3",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
        )

        # Parse silence ranges
        silence_ranges: list[tuple[float, float]] = []
        silence_start_pattern = re.compile(r"silence_start: ([\d.]+)")
        silence_end_pattern = re.compile(r"silence_end: ([\d.]+)")

        current_start: float | None = None
        for line in result.stderr.split("\n"):
            if match := silence_start_pattern.search(line):
                current_start = float(match.group(1))
            elif current_start is not None and (match := silence_end_pattern.search(line)):
                silence_end = float(match.group(1))
                silence_ranges.append((current_start, silence_end))
                current_start = None

        return silence_ranges
    finally:
        audio_path.unlink(missing_ok=True)


def extract_first_n_seconds_no_silence(
    video_path: Path, silence_ranges: list[tuple[float, float]], target_seconds: float = 30.0
) -> Path:
    """Extract first N seconds of non-silent audio from video."""
    video_duration = get_video_duration(video_path)

    # Calculate speech segments (inverse of silence)
    speech_segments: list[tuple[float, float]] = []
    last_end = 0.0

    for silence_start, silence_end in silence_ranges:
        if silence_start > last_end:
            speech_segments.append((last_end, silence_start))
        last_end = silence_end

    # Add final segment if there's content after last silence
    if last_end < video_duration:
        speech_segments.append((last_end, video_duration))

    # Collect segments until we have target_seconds of audio
    collected_duration = 0.0
    segments_to_use: list[tuple[float, float]] = []

    for start, end in speech_segments:
        remaining = target_seconds - collected_duration
        if remaining <= 0:
            break

        segment_duration = end - start
        if segment_duration <= remaining:
            segments_to_use.append((start, end))
            collected_duration += segment_duration
        else:
            # Partial segment
            segments_to_use.append((start, start + remaining))
            collected_duration += remaining
            break

    # Extract audio from these segments
    output_path = Path(tempfile.mktemp(suffix=".wav"))

    if not segments_to_use:
        # Fallback: just extract first 30 seconds
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-t",
                str(target_seconds),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
    else:
        # Build filter for segment selection
        select_expr = "+".join(
            [f"between(t,{start:.3f},{end:.3f})" for start, end in segments_to_use]
        )

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-af",
                f"aselect='{select_expr}',asetpts=N/SR/TB",
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )

    return output_path


async def transcribe_audio(audio_path: Path) -> str:
    """Transcribe audio using Deepgram."""
    from deepgram import DeepgramClient

    API_KEY = "37e776c73c0de03eeacfaa9635e26ce6787bcf74"

    client = DeepgramClient(api_key=API_KEY)
    with audio_path.open("rb") as f:
        audio_bytes = f.read()

    response = client.listen.v1.media.transcribe_file(
        request=audio_bytes,
        model="nova-2",
        smart_format=True,
        punctuate=True,
    )

    if not response or not response.results:
        return ""

    return response.results.channels[0].alternatives[0].transcript or ""


async def generate_titles(transcript: str, count: int = 3) -> list[str]:
    """Generate title suggestions using LLM."""
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    llm = ChatOpenAI(
        model="google/gemini-3-flash-preview",
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        temperature=0.7,
    ).with_structured_output(TitleOptions)

    response = await llm.ainvoke(
        [
            SystemMessage(
                content=f"You are a social media expert. Generate {count} catchy, engaging title options for this video. "
                "Titles should be attention-grabbing, clear, and optimized for social media engagement."
            ),
            HumanMessage(content=f"Generate {count} title options for this video transcript:\n\n{transcript[:2000]}"),
        ]
    )

    return response.titles[:count]


async def get_title_suggestions(video_path: Path, count: int = 3, verbose: bool = True) -> list[str]:
    """
    Get title suggestions for a video.

    Args:
        video_path: Path to the video file
        count: Number of title suggestions to generate (default: 3)
        verbose: Print progress messages

    Returns:
        List of title suggestions
    """
    if verbose:
        print(f"Analyzing: {video_path.name}")

    # Step 1: Detect silence (~0.3s)
    if verbose:
        print("  Detecting silence...")
    silence_ranges = detect_silence_ranges(video_path)
    if verbose:
        print(f"  Found {len(silence_ranges)} silence ranges")

    # Step 2: Extract first 30s of non-silent audio
    if verbose:
        print("  Extracting first 30s of speech...")
    audio_path = extract_first_n_seconds_no_silence(video_path, silence_ranges, target_seconds=30.0)

    try:
        # Step 3: Transcribe
        if verbose:
            print("  Transcribing...")
        transcript = await transcribe_audio(audio_path)
        if verbose:
            print(f"  Got {len(transcript.split())} words")

        # Step 4: Generate titles
        if verbose:
            print(f"  Generating {count} titles...")
        titles = await generate_titles(transcript, count=count)

        return titles
    finally:
        audio_path.unlink(missing_ok=True)


async def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate title suggestions for a video")
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument(
        "-k", "--count", type=int, default=3, help="Number of title suggestions to generate (default: 3)"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON (no user selection)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")
    args = parser.parse_args()

    video_path = args.video.resolve()
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return

    titles = await get_title_suggestions(video_path, count=args.count, verbose=not args.quiet)

    if args.json:
        print(json.dumps(titles, indent=2))
    else:
        print("\nTitle Suggestions:")
        for i, title in enumerate(titles, 1):
            print(f"  {i}. {title}")

        # Prompt user to select
        print()
        while True:
            try:
                choice = input(f"Select a title (1-{len(titles)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(titles):
                    selected_title = titles[idx]
                    print(f"\n✅ Selected: {selected_title}")
                    return
                print(f"Invalid choice. Please enter a number between 1 and {len(titles)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nCancelled.")
                return


if __name__ == "__main__":
    asyncio.run(main())
