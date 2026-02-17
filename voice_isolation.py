#!/usr/bin/env python3
"""
Remove background noise from a video's audio using ElevenLabs Audio Isolation API.

Extracts audio, sends to ElevenLabs for voice isolation, then remuxes with original video.

Usage:
  python voice_isolation.py input.mp4 [output.mp4]
  python voice_isolation.py input.mp4 --output output.mp4

Requires ELEVENLABS_API_KEY environment variable.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx


def isolate_voice(audio_path: Path, output_audio_path: Path, api_key: str) -> None:
    """Send audio to ElevenLabs Audio Isolation API and save the result."""
    url = "https://api.elevenlabs.io/v1/audio-isolation"

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    audio_size_mb = len(audio_bytes) / (1024 * 1024)
    print(f"  Uploading {audio_size_mb:.1f}MB to ElevenLabs...")

    with httpx.Client(timeout=300.0) as client:
        response = client.post(
            url,
            headers={"xi-api-key": api_key},
            files={"audio": (audio_path.name, audio_bytes, "audio/mpeg")},
        )

    if response.status_code != 200:
        print(f"Error: ElevenLabs API returned {response.status_code}")
        print(f"  Response: {response.text[:500]}")
        sys.exit(1)

    with open(output_audio_path, "wb") as f:
        f.write(response.content)

    output_size_mb = len(response.content) / (1024 * 1024)
    print(f"  Received {output_size_mb:.1f}MB isolated audio")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove background noise from video audio using ElevenLabs"
    )
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Output video file (default: <input>-isolated.mp4)",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: ELEVENLABS_API_KEY environment variable must be set")
        sys.exit(1)

    output_path = args.output or args.video.parent / f"{args.video.stem}-isolated.mp4"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        extracted_audio = tmp / "audio.mp3"
        isolated_audio = tmp / "isolated.mp3"

        # Step 1: Extract audio from video
        print("1. Extracting audio from video...")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(args.video),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "44100",
                "-b:a",
                "128k",
                str(extracted_audio),
            ],
            check=True,
            capture_output=True,
        )
        audio_size = extracted_audio.stat().st_size / (1024 * 1024)
        print(f"   Extracted {audio_size:.1f}MB audio")

        # Step 2: Send to ElevenLabs for voice isolation
        print("2. Running voice isolation via ElevenLabs...")
        isolate_voice(extracted_audio, isolated_audio, api_key)

        # Step 3: Remux isolated audio with original video
        print("3. Remuxing isolated audio with video...")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(args.video),
                "-i",
                str(isolated_audio),
                "-c:v",
                "copy",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-shortest",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )

    print(f"✅ Output: {output_path}")


if __name__ == "__main__":
    main()
