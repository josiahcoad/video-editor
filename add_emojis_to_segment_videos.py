#!/usr/bin/env python3
"""
Add emoji overlays to each of the 7 segment videos (or to videos in a folder).

Discovers the "last" video per segment in outputs_dir, or uses all .mp4 files
in --input-dir if given. Runs add_emojis (transcribe + LLM pick + overlay)
on each and writes to output_dir.

Usage:
  # Add emojis to the last video in each segment (segment_01..07)
  add_emojis_to_segment_videos.py --outputs-dir projects/.../outputs

  # Add emojis to pre-composed videos (e.g. with_hooks/*.mp4)
  add_emojis_to_segment_videos.py --input-dir projects/.../outputs/with_hooks --output-dir projects/.../outputs/with_hooks_emojis

  # Dry-run: show which videos would be processed
  add_emojis_to_segment_videos.py --outputs-dir projects/.../outputs --dry-run
"""

import re
import subprocess
import sys
from pathlib import Path


def step_number(name: str) -> int:
    m = re.match(r"^(\d+)_", name)
    return int(m.group(1)) if m else 0


def last_video_in_dir(segment_dir: Path) -> Path | None:
    mp4s = list(segment_dir.glob("*.mp4"))
    if not mp4s:
        return None
    return max(mp4s, key=lambda p: step_number(p.name))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Add emoji overlays to each segment video",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=None,
        help="Outputs dir containing segment_01..07 (used to find last video per segment)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="If set, use all .mp4 in this dir instead of segment last-videos",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <outputs-dir>/with_emojis or <input-dir>_emojis)",
    )
    parser.add_argument(
        "--max-emojis",
        type=int,
        default=4,
        help="Max emoji overlays per video (default: 4)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.input_dir is not None:
        input_dir = args.input_dir.resolve()
        if not input_dir.is_dir():
            print(f"Error: not a directory: {input_dir}")
            sys.exit(1)
        videos = sorted(input_dir.glob("*.mp4"))
        if not videos:
            print(f"No .mp4 files in {input_dir}")
            sys.exit(1)
        output_dir = args.output_dir or (input_dir.parent / f"{input_dir.name}_emojis")
    elif args.outputs_dir is not None:
        outputs_dir = args.outputs_dir.resolve()
        if not outputs_dir.is_dir():
            print(f"Error: not a directory: {outputs_dir}")
            sys.exit(1)
        segment_dirs = sorted(
            d
            for d in outputs_dir.iterdir()
            if d.is_dir() and d.name.startswith("segment_")
        )[:7]
        videos = []
        for d in segment_dirs:
            v = last_video_in_dir(d)
            if v is None:
                print(f"Warning: no .mp4 in {d.name}, skipping")
                continue
            videos.append(v)
        if len(videos) < 7:
            print(f"Warning: only {len(videos)} segment videos found")
        output_dir = args.output_dir or (outputs_dir / "with_emojis")
    else:
        print("Error: pass either --outputs-dir or --input-dir")
        sys.exit(1)

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent

    print(f"Adding emojis to {len(videos)} video(s) -> {output_dir}")
    for i, v in enumerate(videos, 1):
        # Unique names: segment_01.mp4 .. segment_07.mp4 (avoids 03_jumpcut.mp4 x4 overwrite)
        out_name = f"segment_{i:02d}.mp4"
        out_path = output_dir / out_name
        print(f"  {v.parent.name}/{v.name} -> {out_path.name}")
        if args.dry_run:
            continue
        rc = subprocess.call(
            [
                sys.executable,
                "-m",
                "src.edit.add_emojis",
                str(v),
                str(out_path),
                "--max-emojis",
                str(args.max_emojis),
            ],
            cwd=script_dir,
        )
        if rc != 0:
            print(f"  add_emojis failed for {v.name} (exit {rc})")
            sys.exit(rc)

    if args.dry_run:
        print(
            "\n[--dry-run] Run without --dry-run to transcribe, pick emojis, and render."
        )
    else:
        print(f"\nDone. {len(videos)} videos with emojis in {output_dir}")


if __name__ == "__main__":
    main()
