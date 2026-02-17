#!/usr/bin/env python3
"""
Apply manual cuts to a video using timestamp ranges or an EDL file.

Usage:
  python apply_cuts.py <video> <output> --cuts 0.8:7.06,27.54:43.81
  python apply_cuts.py <video> <output> --cuts 0.8:7.06 27.54:43.81
  python apply_cuts.py <video> <output> --edl segment_01.edl
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

END_MARGIN_SECONDS = 0.2  # Extra padding at end of each segment so last word isn't clipped


def trim_video_segments(
    video_path: Path, output_path: Path, segments: list[dict]
) -> None:
    """Trim video to keep only specified word range segments and concatenate them."""
    if not segments:
        raise ValueError("No segments provided")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        segment_files = []
        for i, seg in enumerate(segments):
            segment_file = tmp / f"segment_{i}.mp4"
            duration = seg["end"] - seg["start"] + END_MARGIN_SECONDS
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(seg["start"]),
                    "-i",
                    str(video_path),
                    "-t",
                    str(duration),
                    "-c:v",
                    "h264_videotoolbox",
                    "-b:v",
                    "8M",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    str(segment_file),
                ],
                check=True,
                capture_output=True,
            )
            segment_files.append(segment_file)
        concat_file = tmp / "concat.txt"
        concat_content = "\n".join([f"file '{f.absolute()}'" for f in segment_files])
        concat_file.write_text(concat_content)
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
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )


# ── EDL parsing ───────────────────────────────────────────────────────────────


def timecode_to_seconds(tc: str, fps: int = 30) -> float:
    """Convert SMPTE timecode HH:MM:SS:FF to seconds."""
    hh, mm, ss, ff = (int(x) for x in tc.split(":"))
    return hh * 3600 + mm * 60 + ss + ff / fps


def parse_edl(edl_path: Path, fps: int = 30) -> list[dict]:
    """Parse a CMX 3600 EDL file into segment dicts for trim_video_segments.

    Extracts source IN/OUT timecodes from each edit event.
    """
    segments = []
    for line in edl_path.read_text().splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        src_in = timecode_to_seconds(parts[4], fps)
        src_out = timecode_to_seconds(parts[5], fps)
        segments.append(
            {
                "text": "",
                "start": src_in,
                "end": src_out,
                "duration": round(src_out - src_in, 3),
            }
        )
    segments.sort(key=lambda x: x["start"])
    return segments


# ── Cut string parsing ────────────────────────────────────────────────────────


def parse_cuts(cut_strings: list[str]) -> list[dict]:
    """Parse cut strings like '0.8:7.06' or '0.8:7.06,27.54:43.81' into segment dicts."""
    segments = []
    for cut_str in cut_strings:
        # Support comma-separated within a single string
        for part in cut_str.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                start_str, end_str = part.split(":")
                start = float(start_str)
                end = float(end_str)
                segments.append(
                    {
                        "text": "",
                        "start": start,
                        "end": end,
                        "duration": end - start,
                    }
                )
            except ValueError:
                print(
                    f"Error: Invalid cut format '{part}'. Expected 'start:end' (e.g., '14.6:19.1')"
                )
                sys.exit(1)
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply manual cuts to a video")
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument("output", type=Path, help="Output video file")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--cuts",
        nargs="+",
        help="Timestamp ranges to keep (e.g., '0.8:7.06,27.54:43.81' or '0.8:7.06 27.54:43.81')",
    )
    source.add_argument(
        "--edl",
        type=Path,
        help="EDL file with cuts to apply",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    if args.edl:
        if not args.edl.exists():
            print(f"Error: EDL file not found: {args.edl}")
            sys.exit(1)
        segments = parse_edl(args.edl)
    else:
        segments = parse_cuts(args.cuts)

    if not segments:
        print("Error: No valid cuts provided")
        sys.exit(1)

    total_duration = sum(s["duration"] for s in segments)
    print(f"Applying {len(segments)} cuts (total: {total_duration:.1f}s):")
    for i, seg in enumerate(segments, 1):
        print(
            f"  {i}. {seg['start']:.2f}s - {seg['end']:.2f}s ({seg['duration']:.1f}s)"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    trim_video_segments(args.video, args.output, segments)

    # Write cut-boundary sidecar: timestamps in the OUTPUT timeline where
    # consecutive source segments are joined, plus the source ranges used.
    # Downstream tools (e.g. apply_jump_cuts) use boundaries to avoid
    # placing zoom transitions on top of existing visual discontinuities,
    # and source_ranges to filter/remap the original transcript.
    boundaries: list[float] = []
    source_ranges: list[dict] = []
    pos = 0.0
    for i, seg in enumerate(segments):
        source_ranges.append(
            {"start": round(seg["start"], 4), "end": round(seg["end"], 4)}
        )
        pos += seg["duration"]
        if i < len(segments) - 1:  # no boundary after the last segment
            boundaries.append(round(pos, 4))

    sidecar = {
        "boundaries": boundaries,
        "source_ranges": source_ranges,
    }
    boundaries_path = args.output.with_suffix(".boundaries.json")
    boundaries_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"Done: {args.output}")
    print(f"  Boundaries: {boundaries_path} ({len(boundaries)} join points)")


if __name__ == "__main__":
    main()
