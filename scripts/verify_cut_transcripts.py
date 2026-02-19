#!/usr/bin/env python3
"""Apply cuts, transcribe results, and verify they match intended scripts.

Usage:
  python scripts/verify_cut_transcripts.py --session-dir projects/.../260214-hunter-session1
  python scripts/verify_cut_transcripts.py --video X.mp4 --json shorts_cuts.json --outputs-dir outputs/
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from rapidfuzz import fuzz

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.edit.get_transcript import get_transcript


def _normalize(text: str) -> str:
    """Normalize for comparison: lowercase, collapse whitespace."""
    t = (text or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _compare_transcripts(intended: str, actual: str) -> tuple[bool, float, str]:
    """Compare intended vs actual transcript. Returns (ok, score, note)."""
    i = _normalize(intended)
    a = _normalize(actual)
    if not i:
        return True, 1.0, "No intended text"
    if not a:
        return False, 0.0, "Actual transcript is empty"
    # partial_ratio: how well intended appears in actual (allows extra in actual)
    score = fuzz.partial_ratio(i, a) / 100.0
    # token_set_ratio: word overlap regardless of order (catches reordering)
    ts_score = fuzz.token_set_ratio(i, a) / 100.0
    best = max(score, ts_score)
    ok = best >= 0.85
    note = f"partial={score:.2f} token_set={ts_score:.2f}"
    return ok, best, note


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply cuts, transcribe, and verify cut transcripts match intended scripts"
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        help="Session dir (inputs/, outputs/, shorts_cuts.json in outputs)",
    )
    parser.add_argument("--video", type=Path, help="Source video (with --json)")
    parser.add_argument("--json", type=Path, help="shorts_cuts.json path")
    parser.add_argument("--outputs-dir", type=Path, help="Outputs dir (with --json)")
    parser.add_argument(
        "--skip-apply",
        action="store_true",
        help="Skip apply_cuts; use existing 01_cut.mp4 files",
    )
    args = parser.parse_args()

    if args.session_dir:
        session = args.session_dir.resolve()
        inputs_dir = session / "inputs"
        outputs_dir = session / "outputs"
        cuts_path = outputs_dir / "shorts_cuts.json"
        video = None
        for ext in ("*.mp4", "*.MP4", "*.mov", "*.MOV"):
            for f in inputs_dir.glob(ext):
                video = f
                break
            if video:
                break
        if not video:
            print(f"No source video in {inputs_dir}", file=sys.stderr)
            return 1
    else:
        if not args.video or not args.json or not args.outputs_dir:
            print("Need --session-dir OR (--video + --json + --outputs-dir)")
            return 1
        video = args.video.resolve()
        cuts_path = args.json.resolve()
        outputs_dir = args.outputs_dir.resolve()

    if not cuts_path.exists():
        print(f"Cuts JSON not found: {cuts_path}", file=sys.stderr)
        return 1

    segments = json.loads(cuts_path.read_text())
    if not isinstance(segments, list):
        print("Cuts JSON must be a list of segments")
        return 1

    if not args.skip_apply:
        import subprocess

        _run = subprocess.run
        result = _run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "src.edit.apply_cuts",
                "--video",
                str(video),
                "--json",
                str(cuts_path),
                "--outputs-dir",
                str(outputs_dir),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            return 1
        print(result.stdout)

    print("\nTranscribing cut videos...")
    all_ok = True
    for seg in segments:
        num = seg.get("segment")
        intended = seg.get("script", "")
        cut_mp4 = outputs_dir / f"segment_{int(num):02d}" / "01_cut.mp4"
        if not cut_mp4.exists():
            print(f"  Segment {num}: 01_cut.mp4 not found")
            all_ok = False
            continue
        result = await get_transcript(cut_mp4, filler_words=False)
        actual = result.get("transcript", "")
        ok, score, note = _compare_transcripts(intended, actual)
        status = "✓" if ok else "✗"
        print(f"\n  Segment {num} {status} (score={score:.2f}, {note})")
        print(f"    Intended: {intended[:120]}...")
        print(f"    Actual:   {actual[:120]}...")
        if not ok:
            all_ok = False

    print("\n" + ("All segments aligned ✓" if all_ok else "Some segments misaligned ✗"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
