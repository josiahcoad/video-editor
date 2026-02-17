#!/usr/bin/env python3
"""
Apply segment cuts from a cuts.json file (e.g. from propose_cuts) to a source video.
Writes outputs/segment_NN/01_cut.mp4 and 01_cut.boundaries.json for each segment.

Usage:
  # From repo root, with source video in session inputs/:
  dotenvx run -f .env -- uv run python scripts/apply_cuts_from_json.py \\
    --session-dir projects/HunterZier/editing/videos/260214-hunter-session1

  # Or with explicit paths:
  uv run python scripts/apply_cuts_from_json.py \\
    --video path/to/source.mp4 \\
    --cuts-json path/to/outputs/cuts.json \\
    --outputs-dir path/to/outputs
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply cuts from cuts.json to source video"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--session-dir",
        type=Path,
        help="Session dir containing inputs/ and outputs/ (source video in inputs/, cuts.json in outputs/)",
    )
    group.add_argument(
        "--video",
        type=Path,
        help="Source video (use with --cuts-json and --outputs-dir)",
    )
    parser.add_argument(
        "--cuts-json",
        type=Path,
        help="Path to cuts.json (required if using --video)",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        help="Outputs directory; segment_NN/01_cut.mp4 will be written here (required if using --video)",
    )
    args = parser.parse_args()

    if args.session_dir:
        session = args.session_dir.resolve()
        inputs_dir = session / "inputs"
        outputs_dir = session / "outputs"
        cuts_path = outputs_dir / "cuts.json"
        # Source video: VIDEO env, or first .mp4/.mov in inputs/
        video = None
        if os.environ.get("VIDEO"):
            v = Path(os.environ["VIDEO"]).expanduser().resolve()
            if v.exists():
                video = v
        if video is None:
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
        video = args.video.resolve()
        cuts_path = args.cuts_json.resolve()
        outputs_dir = args.outputs_dir.resolve()
        if not video.exists():
            print(f"Error: Video not found: {video}", file=sys.stderr)
            sys.exit(1)

    if not cuts_path.exists():
        print(f"Error: cuts.json not found: {cuts_path}", file=sys.stderr)
        sys.exit(1)

    segments = json.loads(cuts_path.read_text())
    if not segments:
        print("No segments in cuts.json", file=sys.stderr)
        sys.exit(1)

    for seg in segments:
        num = seg["segment"]
        cuts_str = seg["cuts"]
        out_dir = outputs_dir / f"segment_{num:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "01_cut.mp4"
        cmd = [
            sys.executable,
            "-m",
            "src.edit.apply_cuts",
            str(video),
            str(out_file),
            "--cuts",
            cuts_str,
        ]
        print(f"Segment {num}: {out_file.relative_to(outputs_dir)}")
        subprocess.run(cmd, check=True)

    print(f"Done: {len(segments)} segments written under {outputs_dir}")


if __name__ == "__main__":
    main()
