#!/usr/bin/env python3
"""
Schedule the last video in each segment, or a folder of finished videos (e.g. with_emojis).

By default: finds segment_01..07 under outputs_dir and uses the "last" .mp4 in each
(highest pipeline step number in filename).

With --videos-dir: schedules all .mp4 in that folder in sorted order (e.g. segment_01.mp4 .. segment_07.mp4).

Usage:
  schedule_segment_videos.py <outputs_dir> [--caption-prefix "Text"] [--dry-run]
  schedule_segment_videos.py <outputs_dir> --videos-dir <path/to/with_emojis> [--caption-prefix "Text"]

Example:
  schedule_segment_videos.py projects/.../outputs --caption-prefix "Textbook Interview"
  schedule_segment_videos.py projects/.../outputs --videos-dir projects/.../outputs/with_emojis --caption-prefix "Textbook Interview"

Uses MARKY_BUSINESS_ID and runs schedule_post.py for each video.
"""

import re
import subprocess
import sys
from pathlib import Path


def step_number(name: str) -> int:
    """Extract leading step number from filename, e.g. 03_jumpcut.mp4 -> 3."""
    m = re.match(r"^(\d+)_", name)
    return int(m.group(1)) if m else 0


def last_video_in_dir(segment_dir: Path) -> Path | None:
    """Return the path to the .mp4 with the highest step number in segment_dir."""
    mp4s = list(segment_dir.glob("*.mp4"))
    if not mp4s:
        return None
    best = max(mp4s, key=lambda p: step_number(p.name))
    return best


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Schedule the last video in each segment",
    )
    parser.add_argument(
        "outputs_dir",
        type=Path,
        help="Path to outputs dir (base for discovery, or when using --videos-dir just for reference)",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=None,
        help="If set, schedule all .mp4 in this dir (e.g. with_emojis/) in sorted order instead of segment subdirs",
    )
    parser.add_argument(
        "--caption-prefix",
        default="Video",
        help="Caption prefix; each post gets '{prefix} — Part N' (default: Video)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which videos would be scheduled, do not call schedule_post",
    )
    args = parser.parse_args()

    outputs_dir = args.outputs_dir.resolve()
    if not outputs_dir.is_dir():
        print(f"Error: not a directory: {outputs_dir}")
        sys.exit(1)

    to_schedule: list[tuple[str, Path]] = []

    if args.videos_dir is not None:
        videos_dir = args.videos_dir.resolve()
        if not videos_dir.is_dir():
            print(f"Error: not a directory: {videos_dir}")
            sys.exit(1)
        videos = sorted(videos_dir.glob("*.mp4"))
        if not videos:
            print(f"No .mp4 files in {videos_dir}")
            sys.exit(1)
        for i, v in enumerate(videos, 1):
            to_schedule.append((f"segment_{i:02d}", v))
    else:
        segment_dirs = sorted(
            d
            for d in outputs_dir.iterdir()
            if d.is_dir() and d.name.startswith("segment_")
        )
        if not segment_dirs:
            print(f"No segment_* dirs in {outputs_dir}")
            sys.exit(1)
        for d in segment_dirs:
            video = last_video_in_dir(d)
            if video is None:
                print(f"  ⚠ No .mp4 in {d.name}")
                continue
            to_schedule.append((d.name, video))

    if not to_schedule:
        print("No videos to schedule.")
        sys.exit(0)

    print(f"Last video per segment ({len(to_schedule)} segments):")
    for label, path in to_schedule:
        print(f"  {label}: {path.name}")

    if args.dry_run:
        print(
            "\n[--dry-run] Would schedule the above. Run without --dry-run to upload and create posts."
        )
        return

    # Run schedule_post.py for each (same env as current process)
    script_dir = Path(__file__).resolve().parent
    schedule_post = script_dir / "schedule_post.py"
    if not schedule_post.exists():
        print(f"Error: schedule_post.py not found at {schedule_post}")
        sys.exit(1)

    for label, video_path in to_schedule:
        part = label.replace("segment_", "")
        caption = f"{args.caption_prefix} — Part {part}"
        cmd = [
            sys.executable,
            str(schedule_post),
            "--video",
            str(video_path),
            "--caption",
            caption,
        ]
        print(f"\n🎬 Scheduling {label}: {video_path.name}")
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"  schedule_post failed with exit code {rc}")
            sys.exit(rc)

    print(f"\n🎉 Scheduled {len(to_schedule)} posts.")


if __name__ == "__main__":
    main()
