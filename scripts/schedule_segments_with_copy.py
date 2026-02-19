#!/usr/bin/env python3
"""
Schedule each segment's 06_music_whiteboard.mp4 using title and caption from with_copy.

Assumes with_copy/segment_01.json ... segment_10.json exist (from generate_copy_for_segments).
Uses src.edit.schedule_post so business_id is read from project editing/settings.json
when video is under projects/<Client>/editing/videos/....

Usage:
  uv run python scripts/schedule_segments_with_copy.py --outputs-dir projects/HunterZier/editing/videos/260214-hunter-session1/outputs
  uv run python scripts/schedule_segments_with_copy.py --outputs-dir ... --dry-run

Requires: SUPABASE_URL, SUPABASE_API_KEY; business_id from project settings or MARKY_BUSINESS_ID.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Match UUID from "Post created: 44a2af8e-808a-4aaa-8bbd-9196e4275223"
POST_ID_RE = re.compile(
    r"Post created:\s+([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.I,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Schedule segment videos using with_copy title/caption",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        required=True,
        help="Outputs dir containing segment_01..NN and with_copy/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be scheduled, do not call schedule_post",
    )
    args = parser.parse_args()
    outputs_dir = args.outputs_dir.resolve()

    with_copy_dir = outputs_dir / "with_copy"
    if not with_copy_dir.is_dir():
        print(f"Error: {with_copy_dir} not found")
        sys.exit(1)

    segments = sorted(
        f.stem
        for f in with_copy_dir.glob("segment_*.json")
        if f.stem.replace("segment_", "").isdigit()
    )
    if not segments:
        print(f"Error: no segment_*.json in {with_copy_dir}")
        sys.exit(1)

    for seg in segments:
        copy_path = with_copy_dir / f"{seg}.json"
        video_path = outputs_dir / seg / "06_music_whiteboard.mp4"
        if not copy_path.is_file():
            print(f"  Skip {seg}: no {copy_path.name}")
            continue
        if not video_path.exists():
            print(f"  Skip {seg}: no {video_path.name}")
            continue
        data = json.loads(copy_path.read_text())
        title = data.get("title", "").strip() or seg.replace("_", " ").title()
        caption = data.get("caption", "").strip()
        if not caption:
            print(f"  Skip {seg}: empty caption")
            continue

        if args.dry_run:
            print(
                f"  Would schedule {seg}: {video_path.name} | title={title[:50]!r}..."
            )
            continue

        cmd = [
            sys.executable,
            "-m",
            "src.edit.schedule_post",
            "--video",
            str(video_path),
            "--title",
            title,
            "--caption",
            caption,
        ]
        print(f"\n🎬 Scheduling {seg}...")
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout.rstrip())
        if result.returncode != 0:
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            print(f"  schedule_post failed with exit code {result.returncode}")
            sys.exit(result.returncode)
        # Write post ID into segment folder for reference in Marky
        match = POST_ID_RE.search(result.stdout)
        if match:
            segment_dir = outputs_dir / seg
            (segment_dir / "post_id.txt").write_text(
                match.group(1).strip(), encoding="utf-8"
            )
            print(f"  📄 Wrote {seg}/post_id.txt")
        else:
            print(f"  ⚠ Could not parse post ID from schedule_post output")

    if not args.dry_run:
        print(f"\n🎉 Scheduled {len(segments)} posts.")


if __name__ == "__main__":
    main()
