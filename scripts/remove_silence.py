#!/usr/bin/env python3
"""Remove silences >= min_duration from a video.

python scripts/remove_silence.py input.mp4 output.mp4
python scripts/remove_silence.py input.mp4 output.mp4 --min 0.3
"""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path


def get_video_duration(video_path: Path) -> float:
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


def detect_silence_ranges(
    video_path: Path, min_duration: float = 0.2
) -> list[tuple[float, float]]:
    """Detect silence ranges >= min_duration. Returns list of (start, end) in seconds."""
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

        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(audio_path),
                "-af",
                f"silencedetect=noise=-30dB:d={min_duration}",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
        )
    finally:
        audio_path.unlink(missing_ok=True)

    silence_ranges: list[tuple[float, float]] = []
    start_pat = re.compile(r"silence_start: ([\d.]+)")
    end_pat = re.compile(r"silence_end: ([\d.]+)")
    current_start: float | None = None
    for line in result.stderr.split("\n"):
        if m := start_pat.search(line):
            current_start = float(m.group(1))
        elif current_start is not None and (m := end_pat.search(line)):
            end = float(m.group(1))
            if end - current_start >= min_duration:
                silence_ranges.append((current_start, end))
            current_start = None
    return silence_ranges


def silence_to_keep_segments(
    silence_ranges: list[tuple[float, float]], duration: float
) -> list[dict]:
    """Convert silence ranges to segments to keep (inverse)."""
    segments = []
    last_end = 0.0
    for s_start, s_end in sorted(silence_ranges, key=lambda x: x[0]):
        if s_start > last_end:
            segments.append(
                {"start": last_end, "end": s_start, "duration": s_start - last_end}
            )
        last_end = max(last_end, s_end)
    if last_end < duration:
        segments.append(
            {"start": last_end, "end": duration, "duration": duration - last_end}
        )
    return segments


def trim_video_segments(
    video_path: Path, output_path: Path, segments: list[dict]
) -> None:
    """Trim video to keep only specified segments and concatenate."""
    if not segments:
        raise ValueError("No segments to keep")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        files = []
        for i, seg in enumerate(segments):
            f = tmp / f"seg_{i}.mp4"
            dur = seg["end"] - seg["start"]
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(seg["start"]),
                    "-i",
                    str(video_path),
                    "-t",
                    str(dur),
                    "-c:v",
                    "h264_videotoolbox",
                    "-b:v",
                    "8M",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    str(f),
                ],
                check=True,
                capture_output=True,
            )
            files.append(f)
        concat = tmp / "concat.txt"
        concat.write_text("\n".join(f"file '{p.absolute()}'" for p in files))
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat),
                "-c",
                "copy",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove silences from video")
    parser.add_argument("input", type=Path, help="Input video")
    parser.add_argument("output", type=Path, help="Output video")
    parser.add_argument(
        "--min",
        type=float,
        default=0.2,
        help="Minimum silence duration to remove (default: 0.2)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    duration = get_video_duration(args.input)
    silences = detect_silence_ranges(args.input, min_duration=args.min)
    total_removed = sum(e - s for s, e in silences)

    print(f"Duration: {duration:.1f}s, found {len(silences)} silences >= {args.min}s")
    print(f"Removing {total_removed:.1f}s of silence")
    for i, (s, e) in enumerate(silences[:10]):
        print(f"  {i+1}. {s:.2f}s - {e:.2f}s ({e-s:.2f}s)")
    if len(silences) > 10:
        print(f"  ... and {len(silences) - 10} more")

    segments = silence_to_keep_segments(silences, duration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trim_video_segments(args.input, args.output, segments)
    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
