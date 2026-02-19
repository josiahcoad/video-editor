#!/usr/bin/env python3
"""
Run a single-command template once per segment in a session (outputs/segment_01, ...).

Use this so you don't need a separate "batch" version of each script. The command
string is format()-ted for each segment with the placeholders below.

Session layout:
  <session_dir>/
    inputs/          (e.g. source video)
    outputs/
      segment_01/    (pipeline outputs for short 1)
      segment_02/
      ...

Usage:
  # Run voice_isolation on 06_enhanced → 07_isolated in each segment
  python -m src.edit.apply_batch --session-dir projects/X/editing/videos/session1 -- \
    'python -m src.edit.voice_isolation "{segment_dir}/06_enhanced.mp4" "{segment_dir}/07_isolated.mp4"'

  # Schedule the last video in each segment
  python -m src.edit.apply_batch --session-dir projects/X/editing/videos/session1 -- \
    'python -m src.edit.schedule_post --video "{last_mp4}" --caption "Series — Part {segment_num}"'

  # Only segments 1 and 3, dry run
  python -m src.edit.apply_batch --session-dir ... --segments 1 3 --dry-run -- 'echo {segment_dir}'

Placeholders in the command (each segment gets its own values):
  {session_dir}   Session root (has inputs/, outputs/)
  {inputs_dir}    session_dir/inputs
  {outputs_dir}   session_dir/outputs
  {segment}       Segment dir name, e.g. segment_01
  {segment_num}   Segment number (1, 2, 3, ...)
  {segment_dir}   Full path to outputs/segment_NN
  {last_mp4}      Full path to the .mp4 with highest step number in segment_dir (e.g. 07_music.mp4).
                  Empty string if the segment dir has no .mp4.
  {last_mp4_name} Basename of that file (e.g. 07_music.mp4), or empty.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def step_number(name: str) -> int:
    """Extract leading step number from filename, e.g. 03_jumpcut.mp4 -> 3."""
    m = re.match(r"^(\d+)_", name)
    return int(m.group(1)) if m else 0


def last_mp4_in_dir(segment_dir: Path) -> Path | None:
    """Path to the .mp4 with the highest step number in segment_dir."""
    mp4s = list(segment_dir.glob("*.mp4"))
    if not mp4s:
        return None
    return max(mp4s, key=lambda p: step_number(p.name))


def discover_segments(outputs_dir: Path) -> list[tuple[str, int, Path]]:
    """List (segment_name, segment_num, segment_dir) sorted by segment number."""
    out = []
    for d in sorted(outputs_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("segment_"):
            continue
        try:
            num = int(d.name.replace("segment_", ""))
            out.append((d.name, num, d))
        except ValueError:
            continue
    return sorted(out, key=lambda x: x[1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a command template once per segment in a session",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        required=True,
        help="Session root (contains inputs/ and outputs/)",
    )
    parser.add_argument(
        "--segments",
        type=int,
        nargs="*",
        metavar="N",
        help="Only run on these segment numbers (e.g. 1 3 5). Default: all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command for each segment, do not run",
    )
    parser.add_argument(
        "command",
        nargs=1,
        metavar="COMMAND",
        help="Command string with placeholders {segment_dir}, {segment_num}, {last_mp4}, etc.",
    )
    args = parser.parse_args()

    session_dir = args.session_dir.resolve()
    inputs_dir = session_dir / "inputs"
    outputs_dir = session_dir / "outputs"

    if not session_dir.is_dir():
        print(f"Error: not a directory: {session_dir}", file=sys.stderr)
        sys.exit(1)
    if not outputs_dir.is_dir():
        print(f"Error: outputs/ not found under {session_dir}", file=sys.stderr)
        sys.exit(1)

    segments = discover_segments(outputs_dir)
    if not segments:
        print(f"No segment_* dirs in {outputs_dir}", file=sys.stderr)
        sys.exit(1)

    if args.segments is not None:
        want = set(args.segments)
        segments = [(name, num, path) for name, num, path in segments if num in want]
        if not segments:
            print(f"No matching segments for {sorted(want)}", file=sys.stderr)
            sys.exit(1)

    template = args.command[0]
    failed = 0
    # Run from repo root so "python -m src.edit.X" resolves
    repo_root = Path(__file__).resolve().parents[2]

    for segment_name, segment_num, segment_dir in segments:
        last_mp4 = last_mp4_in_dir(segment_dir)
        try:
            cmd = template.format(
                session_dir=session_dir,
                inputs_dir=inputs_dir,
                outputs_dir=outputs_dir,
                segment=segment_name,
                segment_num=segment_num,
                segment_dir=segment_dir,
                last_mp4=str(last_mp4) if last_mp4 is not None else "",
                last_mp4_name=last_mp4.name if last_mp4 is not None else "",
            )
        except KeyError as e:
            print(f"Error: unknown placeholder {e} in command", file=sys.stderr)
            sys.exit(1)

        print(f"\n--- {segment_name} ---")
        if args.dry_run:
            print(cmd)
            continue

        rc = subprocess.run(cmd, shell=True, cwd=repo_root)
        if rc.returncode != 0:
            print(f"  Command failed with exit code {rc.returncode}", file=sys.stderr)
            failed += 1

    if failed:
        sys.exit(1)
    print(f"\nDone: {len(segments)} segment(s).")


if __name__ == "__main__":
    main()
