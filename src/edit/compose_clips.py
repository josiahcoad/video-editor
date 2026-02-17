#!/usr/bin/env python3
"""
Compose multiple video clips into a single video.

Handles clips from different source videos that may have different
resolutions, framerates, or codecs by re-encoding through ffmpeg
filter_complex concat.

Usage:
  python compose_clips.py intro.mp4 body.mp4 outro.mp4 -o final.mp4
  python compose_clips.py clip1.mp4 clip2.mp4 clip3.mp4 -o output.mp4 --resolution 1080x1920
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def probe_video(path: Path) -> dict:
    """Get video stream info using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    video_stream = next(
        (s for s in data.get("streams", []) if s["codec_type"] == "video"), None
    )
    audio_stream = next(
        (s for s in data.get("streams", []) if s["codec_type"] == "audio"), None
    )
    duration = float(data.get("format", {}).get("duration", 0))
    return {
        "video": video_stream,
        "audio": audio_stream,
        "duration": duration,
        "width": int(video_stream["width"]) if video_stream else 0,
        "height": int(video_stream["height"]) if video_stream else 0,
    }


def compose_clips(
    clips: list[Path],
    output: Path,
    resolution: tuple[int, int] | None = None,
    fps: int = 30,
) -> None:
    """Concatenate multiple video clips into a single output video.

    Args:
        clips: List of input video file paths (in order).
        output: Output video file path.
        resolution: Target (width, height). If None, uses the first clip's resolution.
        fps: Target framerate (default 30).
    """
    if not clips:
        raise ValueError("No clips provided")

    # Probe all clips
    probes = []
    for clip in clips:
        if not clip.exists():
            raise FileNotFoundError(f"Clip not found: {clip}")
        probes.append(probe_video(clip))

    # Determine target resolution
    if resolution is None:
        target_w, target_h = probes[0]["width"], probes[0]["height"]
    else:
        target_w, target_h = resolution

    # Check if all clips already match (can use fast concat demuxer)
    all_match = all(p["width"] == target_w and p["height"] == target_h for p in probes)

    if all_match:
        print(f"All clips match {target_w}x{target_h} — using fast concat")
        _concat_demuxer(clips, output)
    else:
        print(f"Clips have different dimensions — re-encoding to {target_w}x{target_h}")
        _concat_filter(clips, output, target_w, target_h, fps)


def _concat_demuxer(clips: list[Path], output: Path) -> None:
    """Fast concat using demuxer (stream copy, no re-encoding).

    Only works when all clips have identical codecs, resolution, and framerate.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for clip in clips:
            f.write(f"file '{clip.absolute()}'\n")
        concat_file = Path(f.name)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                str(output),
            ],
            check=True,
            capture_output=True,
        )
    finally:
        concat_file.unlink(missing_ok=True)


def _concat_filter(
    clips: list[Path],
    output: Path,
    width: int,
    height: int,
    fps: int,
) -> None:
    """Concat using filter_complex (re-encodes, handles different sources)."""
    n = len(clips)

    # Build input args
    input_args = []
    for clip in clips:
        input_args.extend(["-i", str(clip)])

    # Build filter: scale each input to target resolution, then concat
    filters = []
    concat_inputs = ""
    for i in range(n):
        # Scale + pad to target resolution (handles aspect ratio differences)
        filters.append(
            f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}[v{i}]"
        )
        # Normalize audio to consistent format
        filters.append(
            f"[{i}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[a{i}]"
        )
        concat_inputs += f"[v{i}][a{i}]"

    filters.append(f"{concat_inputs}concat=n={n}:v=1:a=1[vout][aout]")
    filter_complex = ";".join(filters)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            *input_args,
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-c:v",
            "h264_videotoolbox",
            "-b:v",
            "8M",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output),
        ],
        check=True,
        capture_output=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose multiple video clips into a single video"
    )
    parser.add_argument(
        "clips",
        nargs="+",
        type=Path,
        help="Input video clips in order",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output video file",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help="Target resolution WIDTHxHEIGHT (e.g., 1080x1920). Default: use first clip's resolution.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target framerate (default: 30)",
    )
    args = parser.parse_args()

    # Parse resolution
    resolution = None
    if args.resolution:
        try:
            w, h = args.resolution.lower().split("x")
            resolution = (int(w), int(h))
        except ValueError:
            print(
                f"Error: Invalid resolution format '{args.resolution}'. Use WIDTHxHEIGHT (e.g., 1080x1920)"
            )
            sys.exit(1)

    # Validate inputs
    for clip in args.clips:
        if not clip.exists():
            print(f"Error: Clip not found: {clip}")
            sys.exit(1)

    total_clips = len(args.clips)
    print(f"Composing {total_clips} clips:")
    for i, clip in enumerate(args.clips, 1):
        probe = probe_video(clip)
        print(
            f"  {i}. {clip.name} ({probe['duration']:.1f}s, {probe['width']}x{probe['height']})"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    compose_clips(args.clips, args.output, resolution=resolution, fps=args.fps)

    # Report output
    out_probe = probe_video(args.output)
    print(
        f"✅ Output: {args.output} ({out_probe['duration']:.1f}s, {out_probe['width']}x{out_probe['height']})"
    )


if __name__ == "__main__":
    main()
