#!/usr/bin/env python3
"""
Enhance voice in a video using ffmpeg's dialoguenhance filter.

The dialoguenhance filter improves speech clarity and intelligibility.
"""

import subprocess
import sys
from pathlib import Path


def enhance_voice(video_path: Path, output_path: Path) -> None:
    """Enhance voice in video using dialoguenhance filter.

    Args:
        video_path: Input video file
        output_path: Output video file
    """
    print("Enhancing voice with dialoguenhance filter...")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-af",
            "dialoguenhance",  # Enhance dialogue/speech
            "-ac",
            "2",  # Force stereo output (dialoguenhance can change channel layout)
            "-c:v",
            "h264_videotoolbox",  # Hardware-accelerated video encoding
            "-c:a",
            "aac",  # Encode audio to AAC
            str(output_path),
        ],
        check=True,
    )


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python enhance_voice.py <video_file> [output_file]")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    # Parse output path
    output_path = video_path.parent / f"{video_path.stem}-enhanced.mp4"
    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])

    enhance_voice(video_path, output_path)
    print(f"✅ Output video: {output_path}")


if __name__ == "__main__":
    main()
