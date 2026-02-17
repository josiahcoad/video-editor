#!/usr/bin/env python3
"""
Split a single hook-recording video into 7 clips using transcript timestamps,
then prepend each clip to the corresponding segment's last video.

Usage:
  prepend_hooks.py --hooks-video /path/to/IMG_3335.mov --utterances /path/to/IMG_3335-utterances.json --outputs-dir projects/.../outputs

Reads utterances JSON, derives 7 time ranges (one per segment), extracts 7 hook
clips, then composes hook_N + segment_N_last_video -> with_hooks/segment_N_final.mp4.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

# Segment order and utterance indices / time ranges derived from transcript.
# (start_utt_idx, end_utt_idx) or (start_time, end_time) per segment.
# From IMG_3335: 0=trust gap, 1-6=AI should we retakes, 7-9=500 UGC, 9=think producer,
# 10-18=human in loop+soul, 19-22=JTBD, 23=headline, 24-28=inspect.
HOOK_RANGES = [
    (0, 1),  # 1: trust gap
    (
        1,
        9,
    ),  # 2: AI should we + 500 UGC (utterances 1-8, stop before "think like a producer")
    (9, 10),  # 3: think like a producer (utterance 9 only)
    (10, 19),  # 4: human in loop + soul
    (19, 23),  # 5: JTBD shift
    (23, 24),  # 6: headline
    (24, 29),  # 7: inspect behavior (indices 24-28 inclusive)
]


def step_number(name: str) -> int:
    m = re.match(r"^(\d+)_", name)
    return int(m.group(1)) if m else 0


def last_video_in_dir(segment_dir: Path) -> Path | None:
    mp4s = list(segment_dir.glob("*.mp4"))
    if not mp4s:
        return None
    return max(mp4s, key=lambda p: step_number(p.name))


def extract_clip(source: Path, start_s: float, end_s: float, output: Path) -> None:
    """Extract [start_s, end_s] from source to output. -ss before -i for fast seek."""
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = end_s - start_s
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_s),
            "-i",
            str(source),
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-c:a",
            "aac",
            str(output),
        ],
        check=True,
        capture_output=True,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prepend hook clips to segment videos")
    parser.add_argument("--hooks-video", type=Path, required=True)
    parser.add_argument("--utterances", type=Path, required=True)
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    utterances_path = args.utterances
    if not utterances_path.exists():
        print(f"Error: utterances file not found: {utterances_path}")
        sys.exit(1)
    utterances = json.loads(utterances_path.read_text())
    if not utterances:
        print("Error: no utterances in JSON")
        sys.exit(1)

    # Build 7 (start, end) times from utterance ranges
    ranges = []
    for start_idx, end_idx in HOOK_RANGES:
        start_idx = max(0, min(start_idx, len(utterances) - 1))
        end_idx = min(end_idx, len(utterances))
        start_t = utterances[start_idx]["start"]
        end_t = (
            utterances[end_idx - 1]["end"]
            if end_idx > start_idx
            else utterances[start_idx]["end"]
        )
        ranges.append((start_t, end_t))

    outputs_dir = args.outputs_dir.resolve()
    hooks_video = args.hooks_video.resolve()
    if not hooks_video.exists():
        print(f"Error: hooks video not found: {hooks_video}")
        sys.exit(1)

    # Segment dirs and last-video paths
    segment_dirs = sorted(
        d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("segment_")
    )
    if len(segment_dirs) < 7:
        print(f"Error: expected at least 7 segment dirs, found {len(segment_dirs)}")
        sys.exit(1)

    segment_videos = []
    for d in segment_dirs[:7]:
        v = last_video_in_dir(d)
        if v is None:
            print(f"Error: no .mp4 in {d.name}")
            sys.exit(1)
        segment_videos.append(v)

    clips_dir = outputs_dir / "hooks" / "clips"
    out_dir = outputs_dir / "with_hooks"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("Dry run: would extract and compose the following")
        for i, (s, e) in enumerate(ranges, 1):
            print(f"  Hook {i}: {s:.1f}s - {e:.1f}s -> {segment_videos[i-1].name}")
        return

    # Extract 7 hook clips
    hook_clips = []
    for i, (start_t, end_t) in enumerate(ranges, 1):
        clip_path = clips_dir / f"hook_{i:02d}.mp4"
        print(f"Extracting hook {i}: {start_t:.1f}s - {end_t:.1f}s -> {clip_path.name}")
        extract_clip(hooks_video, start_t, end_t, clip_path)
        hook_clips.append(clip_path)

    # Compose hook_N + segment_N -> with_hooks/segment_N_final.mp4
    compose_script = (
        Path(__file__).resolve().parent / "src" / "edit" / "compose_clips.py"
    )
    if not compose_script.exists():
        print(f"Error: compose_clips.py not found at {compose_script}")
        sys.exit(1)

    for i in range(7):
        out_file = out_dir / f"segment_{i+1:02d}_final.mp4"
        print(
            f"Composing segment {i+1}: hook + {segment_videos[i].name} -> {out_file.name}"
        )
        subprocess.run(
            [
                sys.executable,
                str(compose_script),
                str(hook_clips[i]),
                str(segment_videos[i]),
                "-o",
                str(out_file),
            ],
            check=True,
        )

    print(f"\nDone. 7 videos in {out_dir}")


if __name__ == "__main__":
    main()
