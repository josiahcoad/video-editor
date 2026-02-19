#!/usr/bin/env python3
"""
Apply cuts from a cuts JSON to one or many segment outputs.

Usage:
  python apply_cuts.py --video path/to/source.mp4 --json path/to/shorts_cuts.json --outputs-dir path/to/outputs
  python apply_cuts.py --session-dir projects/HunterZier/editing/videos/260214-hunter-session1
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


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
            duration = seg["end"] - seg["start"]
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


def merge_overlaps(segments: list[dict]) -> list[dict]:
    """Merge consecutive overlapping segments into one contiguous range.

    Segments are often in narrative order (hook, body, CTA), not source-time order.
    Only merge when segment[i+1] starts after segment[i] and overlaps (starts before
    segment[i] ends), e.g. 4.88:10.47 and 10.32:15.43 → 4.88:15.43.
    """
    if len(segments) < 2:
        return segments
    out = []
    for seg in segments:
        s = {
            "text": seg.get("text", ""),
            "start": seg["start"],
            "end": seg["end"],
            "duration": seg["end"] - seg["start"],
        }
        if out and out[-1]["start"] < s["start"] and s["start"] < out[-1]["end"]:
            out[-1]["end"] = max(out[-1]["end"], s["end"])
            out[-1]["duration"] = out[-1]["end"] - out[-1]["start"]
        else:
            out.append(s)
    return out


def write_cut_output(video: Path, output: Path, segments: list[dict]) -> None:
    """Write a cut output file and boundaries sidecar."""
    segments = merge_overlaps(segments)
    total_duration = sum(s["duration"] for s in segments)
    print(f"Applying {len(segments)} cuts (total: {total_duration:.1f}s):")
    for i, seg in enumerate(segments, 1):
        print(
            f"  {i}. {seg['start']:.2f}s - {seg['end']:.2f}s ({seg['duration']:.1f}s)"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    trim_video_segments(video, output, segments)

    # Write cut-boundary sidecar: timestamps in the OUTPUT timeline where
    # consecutive source segments are joined, plus the source ranges used.
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
    boundaries_path = output.with_suffix(".boundaries.json")
    boundaries_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"Done: {output}")
    print(f"  Boundaries: {boundaries_path} ({len(boundaries)} join points)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply cuts from cuts JSON to segment outputs"
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Input video file (required with --json)",
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        help="Session dir containing inputs/ and outputs/ (source video in inputs/, cuts.json in outputs/)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Path to cuts JSON; each segment writes outputs/segment_NN/01_cut.mp4",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        help="Outputs directory (required with --json)",
    )
    args = parser.parse_args()

    if not args.session_dir and not args.json:
        print(
            "Error: one of --session-dir or --json is required",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.session_dir:
        session = args.session_dir.resolve()
        inputs_dir = session / "inputs"
        outputs_dir = session / "outputs"
        cuts_path = outputs_dir / "cuts.json"
        video = None
        for ext in ("*.mp4", "*.MP4", "*.mov", "*.MOV"):
            for f in inputs_dir.glob(ext):
                video = f
                break
            if video is not None:
                break
        if video is None:
            print(
                f"No source video found in {inputs_dir}. Add a .mp4 or .mov file there and re-run.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        if args.video is None:
            print("Error: --video is required with --json", file=sys.stderr)
            sys.exit(1)
        if args.outputs_dir is None:
            print("Error: --outputs-dir is required with --json", file=sys.stderr)
            sys.exit(1)
        video = args.video.resolve()
        outputs_dir = args.outputs_dir.resolve()
        cuts_path = args.json.resolve()

    if not video.exists():
        print(f"Error: Video file not found: {video}", file=sys.stderr)
        sys.exit(1)
    if not cuts_path.exists():
        print(f"Error: cuts JSON not found: {cuts_path}", file=sys.stderr)
        sys.exit(1)
    payload = json.loads(cuts_path.read_text())
    if not isinstance(payload, list) or not payload:
        print(
            "Error: cuts JSON must be a non-empty array of segments",
            file=sys.stderr,
        )
        sys.exit(1)
    for seg in payload:
        num = seg.get("segment")
        cuts_str = seg.get("cuts")
        if num is None or not cuts_str:
            print(
                f"Error: invalid segment entry (requires 'segment' and 'cuts'): {seg}",
                file=sys.stderr,
            )
            sys.exit(1)
        out = outputs_dir / f"segment_{int(num):02d}" / "01_cut.mp4"
        print(f"Segment {int(num)}: {out.relative_to(outputs_dir)}")
        write_cut_output(video, out, parse_cuts([cuts_str]))
    print(f"Done: {len(payload)} segments written under {outputs_dir}")


if __name__ == "__main__":
    main()
