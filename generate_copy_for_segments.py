#!/usr/bin/env python3
"""
Generate title + caption for each segment using write_copy (LLM from transcript).

Finds *-words.json (or *-utterances.json) per segment in outputs_dir, runs
write_copy for each, and writes with_copy/segment_01.json .. segment_07.json
with {"title": "...", "caption": "..."}. Segments without a transcript are
skipped or get a fallback title/caption.

Usage:
  generate_copy_for_segments.py --outputs-dir projects/.../outputs --voice path/to/voice.md
  generate_copy_for_segments.py --outputs-dir projects/.../outputs --voice voice.md --caption-prefix "Textbook Interview"

If --caption-prefix is set, segments with no transcript get title/caption
"{prefix} — Part N". Otherwise they are skipped.
Requires OPENROUTER_API_KEY. Voice file is required for write_copy.
"""

import json
import subprocess
import sys
from pathlib import Path


def find_transcript(segment_dir: Path) -> Path | None:
    """Prefer 06_captioned-words.json, then 05_titled-words.json, then any *-words.json."""
    for name in ("06_captioned-words.json", "05_titled-words.json"):
        p = segment_dir / name
        if p.exists():
            return p
    for p in segment_dir.glob("*-words.json"):
        return p
    for p in segment_dir.glob("*-utterances.json"):
        return p
    return None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate title+caption per segment via write_copy",
    )
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument(
        "--voice",
        type=Path,
        required=True,
        help="Voice guidelines file for write_copy",
    )
    parser.add_argument(
        "--platform",
        default="short",
        help="Platform for write_copy (default: short)",
    )
    parser.add_argument(
        "--caption-prefix",
        default=None,
        help="Fallback title/caption for segments with no transcript: '{prefix} — Part N'",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    outputs_dir = args.outputs_dir.resolve()
    if not outputs_dir.is_dir():
        print(f"Error: not a directory: {outputs_dir}")
        sys.exit(1)
    if not args.voice.exists():
        print(f"Error: voice file not found: {args.voice}")
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    segment_dirs = sorted(
        d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("segment_")
    )[:7]
    out_dir = outputs_dir / "with_copy"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, seg_dir in enumerate(segment_dirs, 1):
        transcript_path = find_transcript(seg_dir)
        label = f"segment_{i:02d}"

        if transcript_path is None:
            if args.caption_prefix:
                title = caption = f"{args.caption_prefix} — Part {i}"
                out = out_dir / f"{label}.json"
                out.write_text(
                    json.dumps({"title": title, "caption": caption}, indent=2)
                )
                print(f"  {label}: no transcript, fallback title/caption")
            else:
                print(f"  {label}: no *-words.json, skipping")
            continue

        if args.dry_run:
            print(f"  {label}: would run write_copy on {transcript_path.name}")
            continue

        cmd = [
            sys.executable,
            "-m",
            "src.edit.write_copy",
            "--transcript",
            str(transcript_path),
            "--voice",
            str(args.voice),
            "--platform",
            args.platform,
        ]
        result = subprocess.run(cmd, cwd=script_dir, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  {label}: write_copy failed: {result.stderr}")
            sys.exit(1)
        try:
            data = json.loads(result.stdout)
            out = out_dir / f"{label}.json"
            out.write_text(json.dumps(data, indent=2))
            print(f"  {label}: title={data.get('title', '')[:50]!r}...")
        except json.JSONDecodeError:
            print(f"  {label}: write_copy output not JSON: {result.stdout[:200]}")
            sys.exit(1)

    print(f"\nCopy written to {out_dir}")


if __name__ == "__main__":
    main()
