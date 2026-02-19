#!/usr/bin/env python3
"""Apply speedup to a video."""

import asyncio
import subprocess
import sys
from pathlib import Path

from .encode_common import H264_SOCIAL_COLOR_ARGS
from .settings_loader import load_settings


def calculate_wpm(video_duration_seconds: float, word_count: int) -> float:
    """Calculate words per minute from video duration and word count."""
    if video_duration_seconds <= 0:
        return 0.0
    duration_minutes = video_duration_seconds / 60.0
    return word_count / duration_minutes if duration_minutes > 0 else 0.0


def calculate_speedup_from_wpm(
    video_duration_seconds: float, word_count: int, target_wpm: float
) -> float:
    """Calculate speedup needed to reach target WPM.

    Args:
        video_duration_seconds: Current video duration
        word_count: Number of words in transcript
        target_wpm: Target WPM to reach

    Returns:
        Speedup multiplier needed to reach target_wpm
    """
    current_wpm = calculate_wpm(video_duration_seconds, word_count)
    if current_wpm <= 0:
        return 1.0  # Can't calculate, don't speedup

    # Calculate speedup: target_wpm / current_wpm
    speedup = target_wpm / current_wpm
    return speedup


def apply_speedup(input_video: Path, output_video: Path, speedup: float) -> None:
    """Apply speedup to video and audio."""
    if speedup <= 1.0:
        print(f"Speedup {speedup} <= 1.0, copying video as-is")
        subprocess.run(["cp", str(input_video), str(output_video)], check=True)
        return

    print(f"Applying {speedup}x speedup to video...")

    # Video filter: setpts=PTS/speedup
    video_filter = f"setpts=PTS/{speedup}"

    # Audio filter: atempo (supports 0.5-2.0 range)
    # If speedup > 2.0, chain multiple atempo filters
    audio_filters = []
    remaining_speedup = speedup

    while remaining_speedup > 2.0:
        audio_filters.append("atempo=2.0")
        remaining_speedup /= 2.0

    if remaining_speedup > 1.0:
        audio_filters.append(f"atempo={remaining_speedup:.3f}")
    elif remaining_speedup < 1.0:
        audio_filters.append("atempo=0.5")

    audio_filter = ",".join(audio_filters) if audio_filters else "anull"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-vf",
            video_filter,
            "-af",
            audio_filter,
            "-c:v",
            "h264_videotoolbox",
            *H264_SOCIAL_COLOR_ARGS,
            "-c:a",
            "aac",
            "-b:v",
            "5M",
            "-b:a",
            "192k",
            str(output_video),
        ],
        check=True,
    )

    print(f"✅ Speedup applied: {speedup}x")
    print(f"✅ Output: {output_video}")


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Apply speedup to a video")
    parser.add_argument("input_video", type=Path, help="Input video file")
    parser.add_argument("output_video", type=Path, help="Output video file")
    parser.add_argument(
        "--speedup",
        type=float,
        default=None,
        help="Speedup multiplier (e.g., 1.2 = 20%% faster). If not provided and --wpm is set, will calculate from WPM.",
    )
    parser.add_argument(
        "--wpm",
        type=float,
        default=None,
        help="Target WPM. If provided, will calculate speedup automatically based on current WPM.",
    )

    args = parser.parse_args()

    input_video = args.input_video
    output_video = args.output_video

    if not input_video.exists():
        print(f"Error: Input video not found: {input_video}")
        sys.exit(1)

    # Default --wpm from project/session settings if not passed on CLI
    if args.wpm is None and "--wpm" not in sys.argv:
        settings = load_settings(input_video)
        args.wpm = settings.get("speedup", {}).get("target_wpm")

    speedup = args.speedup

    # If WPM is provided, calculate speedup from it
    if args.wpm is not None:
        print(f"🎯 Target WPM: {args.wpm}")
        print("📝 Transcribing video to calculate current WPM...")

        from .get_transcript import get_transcript

        result = asyncio.run(get_transcript(input_video))
        word_count = len(result.get("words", []))
        duration = get_video_duration(input_video)

        calculated_speedup = calculate_speedup_from_wpm(duration, word_count, args.wpm)
        current_wpm = calculate_wpm(duration, word_count)

        print(f"📊 Current WPM: {current_wpm:.1f}")
        print(f"⚡ Calculated speedup: {calculated_speedup:.3f}x")

        if calculated_speedup <= 1.0:
            print(
                f"✅ Current WPM ({current_wpm:.1f}) >= target ({args.wpm}), no speedup needed"
            )
            speedup = 1.0
        else:
            speedup = calculated_speedup

    if speedup is None:
        print("Error: Must provide either --speedup or --wpm")
        sys.exit(1)

    apply_speedup(input_video, output_video, speedup)


if __name__ == "__main__":
    main()
