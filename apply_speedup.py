#!/usr/bin/env python3
"""Apply speedup to a video."""

import subprocess
import sys
from pathlib import Path


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


def main():
    if len(sys.argv) < 3:
        print("Usage: python apply_speedup.py <input_video> <output_video> <speedup>")
        print("  Example: python apply_speedup.py video.mp4 output.mp4 1.2")
        sys.exit(1)

    input_video = Path(sys.argv[1])
    output_video = Path(sys.argv[2])
    speedup = float(sys.argv[3])

    if not input_video.exists():
        print(f"Error: Input video not found: {input_video}")
        sys.exit(1)

    apply_speedup(input_video, output_video, speedup)


if __name__ == "__main__":
    main()
