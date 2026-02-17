#!/usr/bin/env python3
"""
Normalize video rotation by detecting rotation metadata and applying it physically.

This prevents rotation issues when converting MOV to MP4.
"""

import subprocess
import sys
from pathlib import Path


def get_rotation_from_displaymatrix(video_path: Path) -> int:
    """Get rotation from displaymatrix by parsing ffprobe output."""
    result = subprocess.run(
        ["ffprobe", str(video_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    import re

    # Look for "displaymatrix: rotation of X.XX degrees" in either stdout or stderr
    output = result.stdout + result.stderr
    match = re.search(
        r"displaymatrix.*?rotation of (-?\d+\.?\d*)\s*degrees",
        output,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        try:
            rotation = float(match.group(1))
            # Normalize -90 to 270
            normalized = int(rotation) % 360
            if normalized < 0:
                normalized = 360 + normalized
            return normalized
        except (ValueError, TypeError):
            pass
    return 0


def get_rotation_metadata(video_path: Path) -> int:
    """Get rotation metadata from video file.

    Checks both side_data rotation and displaymatrix rotation.

    Returns:
        Rotation angle in degrees (0, 90, 180, 270) or 0 if not found.
        Negative values are normalized to positive (e.g., -90 -> 270).
    """
    # First try side_data rotation
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "side_data=rotation",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0 and result.stdout.strip():
        try:
            rotation = int(float(result.stdout.strip()))
            return rotation % 360
        except (ValueError, TypeError):
            pass

    # Check displaymatrix rotation (common in iPhone videos)
    displaymatrix_rotation = get_rotation_from_displaymatrix(video_path)
    if displaymatrix_rotation != 0:
        return displaymatrix_rotation

    return 0


def normalize_rotation(
    video_path: Path, output_path: Path, *, metadata_only: bool = False
) -> None:
    """Normalize video rotation.

    Args:
        video_path: Input video file
        output_path: Output video file
        metadata_only: If True, only remove rotation metadata flag (fast, no re-encoding).
                      If False, apply rotation physically to frames (slow, requires re-encoding).
                      Use metadata_only=True if frames are already correctly oriented.
    """
    rotation = get_rotation_metadata(video_path)

    if rotation == 0:
        print(f"No rotation metadata found. Copying video as-is...")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-c",
                "copy",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
        return

    print(f"Detected rotation: {rotation}°")

    if metadata_only:
        print("Removing rotation metadata flag only (fast, no re-encoding)...")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-c",
                "copy",
                "-metadata:s:v:0",
                "rotate=0",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
        print(f"✅ Rotation metadata removed. Saved to: {output_path}")
        return

    print("Applying rotation physically to frames (requires re-encoding)...")

    # Map rotation to transpose filter
    # 90° clockwise = transpose=1
    # 180° = transpose=1,transpose=1 (or hflip,vflip)
    # 270° clockwise (90° counterclockwise) = transpose=2
    if rotation == 90:
        filter_complex = "transpose=1"
    elif rotation == 180:
        filter_complex = "transpose=1,transpose=1"
    elif rotation == 270:
        filter_complex = "transpose=2"
    else:
        print(f"Warning: Unsupported rotation angle {rotation}°. Copying as-is.")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-c",
                "copy",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
        return

    print(f"Applying rotation filter: {filter_complex}")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hwaccel",
            "videotoolbox",
            "-i",
            str(video_path),
            "-vf",
            filter_complex,
            "-map_metadata",
            "-1",  # Remove all metadata
            "-c:v",
            "h264_videotoolbox",
            "-b:v",
            "12M",  # Match user's bitrate setting
            "-c:a",
            "copy",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    print(f"✅ Normalized video saved to: {output_path}")


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print(
            "Usage: python normalize_rotation.py <video_file> [output_file] [--metadata-only]"
        )
        print(
            "  --metadata-only: Only remove rotation metadata flag (fast, no re-encoding)"
        )
        print("                   Use this if frames are already correctly oriented")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    # Parse arguments
    metadata_only = "--metadata-only" in sys.argv
    output_path = None
    for arg in sys.argv[2:]:
        if arg != "--metadata-only" and not arg.startswith("--"):
            output_path = Path(arg)
            break

    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}-normalized.mp4"

    normalize_rotation(video_path, output_path, metadata_only=metadata_only)


if __name__ == "__main__":
    main()
