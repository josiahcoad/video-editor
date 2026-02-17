#!/usr/bin/env python3
"""
Enhance voice in a video using ffmpeg audio filters.

Applies a podcast-style audio chain: bass warmth, harshness reduction,
dynamic compression, and loudness normalization. Makes iPhone/webcam
recordings sound closer to a professional microphone.

Usage:
  python enhance_voice.py <video_file> [output_file]
"""

import subprocess
import sys
from pathlib import Path

# Podcast-style voice enhancement filter chain:
#   1. highpass=60       — cut rumble below 60Hz
#   2. lowpass=10000     — cut hiss/noise above 10kHz
#   3. EQ +5dB @ 120Hz  — bass warmth (chest resonance)
#   4. EQ +2dB @ 220Hz  — low-mid body (fullness)
#   5. EQ -2dB @ 3kHz   — reduce nasal/harsh midrange
#   6. EQ -4dB @ 6.5kHz — de-ess / reduce tinny sibilance
#   7. compand           — dynamic compression (broadcast feel)
#   8. loudnorm -16 LUFS — loudness normalization (podcast standard)
VOICE_ENHANCE_FILTER = (
    "highpass=f=60,"
    "lowpass=f=10000,"
    "equalizer=f=120:t=q:w=0.8:g=5,"
    "equalizer=f=220:t=q:w=1:g=2,"
    "equalizer=f=3000:t=q:w=1.5:g=-2,"
    "equalizer=f=6500:t=q:w=1:g=-4,"
    "compand=attacks=0.02:decays=0.3:points=-80/-80|-45/-30|-20/-15|0/-10:gain=3,"
    "loudnorm=i=-16:lra=7:tp=-1.5"
)


def enhance_voice(video_path: Path, output_path: Path) -> None:
    """Enhance voice in video using podcast-style audio filter chain.

    Adds bass warmth, reduces harshness/sibilance, compresses dynamics,
    and normalizes loudness. Video stream is copied without re-encoding.

    Args:
        video_path: Input video file
        output_path: Output video file
    """
    print("Enhancing voice (warmth + compression + normalization)...")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-af",
            VOICE_ENHANCE_FILTER,
            "-c:v",
            "copy",  # No video re-encode needed
            "-c:a",
            "aac",
            "-b:a",
            "192k",
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
