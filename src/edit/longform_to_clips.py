#!/usr/bin/env python3
"""
Run propose_interview_cuts + apply_cuts in one go: long-form video → N clip files.

Usage:
  python -m src.edit.longform_to_clips --input_video path/to/source.mp4 --clips_count 3

Outputs:
  <output_dir>/words.json, <output_dir>/utterances.json
  <output_dir>/cuts.json
  <output_dir>/segment_01/01_cut.mp4
  <output_dir>/segment_02/01_cut.mp4
  ...

Defaults: output_dir = <input_video>.parent / "<stem>_clips". All other
propose_interview_cuts/apply_cuts options use their defaults.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .apply_cuts import apply_cuts
from .get_transcript import get_transcript
from .propose_interview_cuts import propose_interview_cuts


async def run(
    input_video: Path,
    clips_count: int,
    output_dir: Path | None = None,
) -> None:
    """Transcribe → propose interview cuts → apply cuts; write segment clips to output_dir."""
    input_video = input_video.resolve()
    if not input_video.exists():
        raise FileNotFoundError(f"Video not found: {input_video}")

    if output_dir is None:
        output_dir = input_video.parent / f"{input_video.stem}_clips"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    words_path = output_dir / "words.json"
    utterances_path = output_dir / "utterances.json"
    cuts_path = output_dir / "cuts.json"

    # 1. Transcribe
    print(f"Transcribing {input_video.name}...")
    result = await get_transcript(input_video)
    words_path.write_text(json.dumps(result["words"], indent=2))
    utterances_path.write_text(json.dumps(result["utterances"], indent=2))
    print(f"  {len(result['words'])} words, {len(result['utterances'])} utterances")

    # 2. Propose interview cuts (default duration/tolerance)
    print(f"Proposing {clips_count} segment(s)...")
    segments = await propose_interview_cuts(
        words_path,
        utterances_path,
        num_shorts=clips_count,
    )
    print(f"  {len(segments)} segment(s)")

    # 3. Write cuts JSON
    cuts_path.write_text(json.dumps(segments, indent=2) + "\n")
    print(f"  Wrote {cuts_path}")

    # 4. Apply cuts → segment_01/01_cut.mp4, ...
    print("Applying cuts...")
    apply_cuts(input_video, cuts_path, output_dir)
    print(f"Done. Clips in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Long-form video → N clips in one go (propose_interview_cuts + apply_cuts)"
    )
    parser.add_argument(
        "--input_video",
        type=Path,
        required=True,
        help="Input long-form video file",
    )
    parser.add_argument(
        "--clips_count",
        type=int,
        required=True,
        help="Number of segments/clips to produce",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (default: <input_video>.parent / '<stem>_clips')",
    )
    args = parser.parse_args()

    try:
        asyncio.run(
            run(
                args.input_video,
                args.clips_count,
                args.output_dir,
            )
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
