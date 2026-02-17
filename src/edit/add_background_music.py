#!/usr/bin/env python3
"""
Add background music to a talking-head video using a single static volume.

Assumptions:
- Voice is present for essentially the entire video
- No music-only sections
- No long silences

Policy:
- Voice normalized to -16 LUFS
- Music normalized to -30 LUFS
"""

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

VOICE_TARGET_LUFS = -16.0
MUSIC_TARGET_LUFS = -35.0


def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def measure_lufs(path: Path) -> float:
    """Return gated integrated LUFS using ffmpeg loudnorm."""
    p = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(path),
            "-af",
            "loudnorm=print_format=json",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    # Extract JSON from stderr - find the JSON object
    stderr = p.stderr
    start_idx = stderr.find("{")
    end_idx = stderr.rfind("}") + 1
    if start_idx == -1 or end_idx == 0:
        raise ValueError("Could not find JSON in ffmpeg output")

    json_str = stderr[start_idx:end_idx]
    stats = json.loads(json_str)
    return float(stats["input_i"])


def normalize_audio(src: Path, dst: Path, target_lufs: float):
    """Two-pass loudness normalization."""
    p = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(src),
            "-af",
            f"loudnorm=I={target_lufs}:print_format=json",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    # Extract JSON from stderr
    stderr = p.stderr
    start_idx = stderr.find("{")
    end_idx = stderr.rfind("}") + 1
    if start_idx == -1 or end_idx == 0:
        raise ValueError("Could not find JSON in ffmpeg output")

    json_str = stderr[start_idx:end_idx]
    stats = json.loads(json_str)

    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-af",
            (
                f"loudnorm=I={target_lufs}:"
                f"measured_I={stats['input_i']}:"
                f"measured_TP={stats['input_tp']}:"
                f"measured_LRA={stats['input_lra']}:"
                f"measured_thresh={stats['input_thresh']}:"
                f"linear=true"
            ),
            str(dst),
        ]
    )


def _get_duration(path: Path) -> float:
    """Return the duration in seconds of a media file."""
    p = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(p.stdout.strip())


def add_music(video: Path, music: Path, output: Path, music_start_offset: float = 5.0):
    """Add background music to video.

    Args:
        video: Input video file
        music: Music file to add
        output: Output video file
        music_start_offset: Seconds into the music track to start (default: 5.0)

    Notes:
        If the music track (after applying the start offset) is shorter than
        the video, the music is looped so it covers the full video duration.
        The start offset is automatically clamped so it never exceeds the
        music length.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        voice_raw = tmp / "voice_raw.wav"
        voice_norm = tmp / "voice_norm.wav"
        music_raw = tmp / "music_raw.wav"
        music_adj = tmp / "music_adj.wav"

        video_dur = _get_duration(video)
        music_dur = _get_duration(music)

        # Clamp offset so we don't seek past the end of the track
        if music_start_offset >= music_dur:
            music_start_offset = 0.0

        remaining_music = music_dur - music_start_offset

        # Extract audio from video
        run(["ffmpeg", "-y", "-i", str(video), "-ac", "1", str(voice_raw)])

        # Extract music — loop if the track is shorter than the video.
        # -stream_loop -1 loops the input infinitely; -t caps to video length.
        music_extract_cmd = ["ffmpeg", "-y"]
        if remaining_music < video_dur:
            music_extract_cmd.extend(["-stream_loop", "-1"])
        music_extract_cmd.extend(
            [
                "-ss",
                str(music_start_offset),
                "-i",
                str(music),
                "-t",
                str(video_dur),
                "-ac",
                "1",
                str(music_raw),
            ]
        )
        run(music_extract_cmd)

        # Normalize voice
        normalize_audio(voice_raw, voice_norm, VOICE_TARGET_LUFS)

        # Normalize music
        normalize_audio(music_raw, music_adj, MUSIC_TARGET_LUFS)

        # Mix — use -shortest so output matches the video length
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video),
                "-i",
                str(voice_norm),
                "-i",
                str(music_adj),
                "-filter_complex",
                "[1:a][2:a]amix=inputs=2[a]",
                "-map",
                "0:v",
                "-map",
                "[a]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(output),
            ]
        )


async def search_music(tags: str, count: int = 3) -> list[dict]:
    """Search for music on Openverse.

    Args:
        tags: Genre/tags to search for (e.g., "hip-hop", "upbeat", "energetic")
        count: Number of results to return (default: 3)

    Returns:
        List of track dictionaries with id, title, creator, url, license
    """
    OPENVERSE_API_BASE = "https://api.openverse.org/v1"

    params = {
        "category": "music",
        "tags": tags,
        "page": 1,
        "page_size": count,
    }

    print(f"Searching Openverse for music with tags: {tags}...")

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{OPENVERSE_API_BASE}/audio/",
            params=params,
        )
        response.raise_for_status()
        data = response.json()

    results = data.get("results", [])
    if not results:
        raise RuntimeError(f"No music found for tags: {tags}")

    tracks = []
    for track in results[:count]:
        tracks.append(
            {
                "id": track.get("id", ""),
                "title": track.get("title", "Unknown"),
                "creator": track.get("creator", "Unknown"),
                "url": track.get("url", ""),
                "license": track.get("license", "Unknown"),
            }
        )

    return tracks


async def download_music(music_url: str, output_path: Path) -> Path:
    """Download music file from URL.

    Args:
        music_url: URL to music file
        output_path: Path where music file should be saved

    Returns:
        Path to downloaded music file
    """
    print(f"Downloading from: {music_url[:80]}...")

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(music_url, follow_redirects=True)
        response.raise_for_status()
        output_path.write_bytes(response.content)

    print(f"Downloaded music to: {output_path}")
    return output_path


def pick_track_from_dir(music_dir: Path, segment_number: int) -> Path:
    """Pick a track by round-robin from a directory of .mp3 files.

    Tracks are sorted by filename; segment_number is 1-based.
    Returns: music_dir / tracks[(segment_number - 1) % len(tracks)]
    """
    tracks = sorted(music_dir.glob("*.mp3"))
    if not tracks:
        raise FileNotFoundError(f"No .mp3 files in {music_dir}")
    idx = (segment_number - 1) % len(tracks)
    return tracks[idx]


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: add_background_music.py <video> [music_file_or_dir] [output] [--tags <genre>] [--segment <N>] [--dry-run]"
        )
        print("  If music_file is not provided, will search Openverse using --tags")
        print("  If music path is a directory: use --segment N to pick track by round-robin (1-based).")
        print("  --dry-run: Show top 3 music options without processing video")
        print("  Example: add_background_music.py video.mp4 ./music/ 07_music.mp4 --segment 3")
        sys.exit(1)

    video = Path(sys.argv[1])
    if not video.exists():
        print(f"Error: Video file not found: {video}")
        sys.exit(1)

    # Parse --segment (for round-robin when music is a directory)
    segment_number = 1
    if "--segment" in sys.argv:
        idx = sys.argv.index("--segment")
        if idx + 1 < len(sys.argv):
            try:
                segment_number = int(sys.argv[idx + 1])
            except ValueError:
                print("Error: --segment must be an integer")
                sys.exit(1)

    # Parse tags flag
    tags = None
    if "--tags" in sys.argv:
        idx = sys.argv.index("--tags")
        if idx + 1 < len(sys.argv):
            tags = sys.argv[idx + 1]

    # Parse music file or directory (optional if tags provided)
    music = None
    music_arg_idx = 2
    if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
        music = Path(sys.argv[2])
        music_arg_idx = 3

    # Parse output path
    output = video.with_stem(video.stem + "-with-music")
    if len(sys.argv) > music_arg_idx and not sys.argv[music_arg_idx].startswith("--"):
        output = Path(sys.argv[music_arg_idx])

    # Check for dry-run flag
    dry_run = "--dry-run" in sys.argv

    # If music is a directory, pick track by round-robin
    if music and music.is_dir():
        music = pick_track_from_dir(music, segment_number)
        print(f"Round-robin: segment {segment_number} → {music.name}")

    # If no music file provided, search
    if not music and tags:
        tracks = asyncio.run(search_music(tags, count=3))

        if dry_run:
            print("\n🔍 DRY RUN: Top 3 music options:")
            for i, track in enumerate(tracks, 1):
                print(f"   {i}. '{track['title']}' by {track['creator']}")
                print(f"      License: {track['license']}")
                print(f"      URL: {track['url']}")
                print()
            return

        # Use first result
        track = tracks[0]
        print(f"Selected: '{track['title']}' by {track['creator']}")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            music_path = Path(tmp.name)
        try:
            asyncio.run(download_music(track["url"], music_path))
            music = music_path
        except Exception as e:
            print(f"Error downloading music: {e}")
            sys.exit(1)
    elif not music:
        print("Error: Either provide a music file or use --tags to search for music")
        print(
            "Usage: add_background_music.py <video> [music_file] [output] [--tags <genre>] [--dry-run]"
        )
        sys.exit(1)

    if not music.exists():
        print(f"Error: Music file not found: {music}")
        sys.exit(1)

    add_music(video, music, output)
    print(f"✅ Output written to {output}")

    # Clean up downloaded music if it was temporary
    if music.name.startswith("tmp") or str(music).startswith("/tmp"):
        music.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
