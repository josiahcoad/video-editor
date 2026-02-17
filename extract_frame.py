#!/usr/bin/env python3
"""
Extract a single frame from a video at a given timestamp.

Usage:
  extract_frame.py <video> [--time <seconds>] [--output <path>]

Examples:
  extract_frame.py segment.mp4                        # frame at t=0, saves to segment_0.0s.jpg
  extract_frame.py segment.mp4 --time 3.5             # frame at t=3.5s
  extract_frame.py segment.mp4 --time 3.5 --output /tmp/check.jpg
"""

import argparse
import subprocess
import sys
from pathlib import Path


def extract_frame(video: Path, time: float, output: Path) -> Path:
    """Extract a single frame from video at the given timestamp."""
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(time),
        "-i",
        str(video),
        "-vframes",
        "1",
        "-q:v",
        "2",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: ffmpeg failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a single frame from a video",
    )
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument(
        "--time",
        "-t",
        type=float,
        default=0.0,
        help="Timestamp in seconds (default: 0)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output image path (default: <video_stem>_<time>s.jpg)",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        args.output = args.video.parent / f"{args.video.stem}_{args.time}s.jpg"

    extract_frame(args.video, args.time, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
