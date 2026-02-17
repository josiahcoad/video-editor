#!/usr/bin/env python3
"""
Add Pexels b-roll clips to a talking-head video.

Uses an LLM to analyze the transcript and suggest 3 b-roll search queries and
3 insertion timestamps. Fetches videos from Pexels API, extracts 3 seconds
from each, and inserts them into the main video (main audio continues during
b-roll). Output video is 9 seconds longer (3 × 3s inserts).

Requires: PEXELS_API_KEY, OPENROUTER_API_KEY (for LLM).

Usage:
  python -m src.edit.add_broll <video.mp4> <output.mp4> --transcript <words.json>
  python -m src.edit.add_broll <video.mp4> <output.mp4> --transcript <words.json> --broll-duration 3 --num-clips 3
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .encode_common import H264_SOCIAL_COLOR_ARGS
from .get_transcript import get_transcript, read_word_transcript_file


# ── Pydantic plan ────────────────────────────────────────────────────────────


class BrollPlan(BaseModel):
    """LLM output: 3 search queries and 3 insertion times (seconds)."""

    search_queries: list[str] = Field(
        description="Exactly 3 Pexels video search queries (e.g. 'first time home buyer', 'house keys', 'mortgage documents'). Each will be used to fetch one b-roll clip."
    )
    insert_times_seconds: list[float] = Field(
        description="Exactly 3 timestamps in seconds where b-roll should be inserted. Must be in ascending order. Avoid first 5s (title) and last 5s (CTA)."
    )
    rationale: list[str] = Field(
        description="One sentence per clip explaining why this query and this timestamp fit the content."
    )


# ── Pexels API ──────────────────────────────────────────────────────────────


def pexels_search_videos(
    query: str,
    api_key: str,
    orientation: str = "portrait",
    per_page: int = 5,
) -> list[dict]:
    """Search Pexels for videos. Returns list of video objects with video_files."""
    url = "https://api.pexels.com/videos/search"
    params = {"query": query, "orientation": orientation, "per_page": per_page}
    resp = httpx.get(
        url,
        params=params,
        headers={"Authorization": api_key},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("videos", [])


def pick_best_video_file(video: dict, min_duration: float = 3.0) -> str | None:
    """From a Pexels video object, pick the best video_file link (portrait-friendly, >= min_duration)."""
    if video.get("duration", 0) < min_duration:
        return None
    files = video.get("video_files", [])
    # Prefer HD portrait (height > width) or square; avoid landscape-only
    for f in files:
        if f.get("quality") == "hd" and f.get("width") and f.get("height"):
            if f["height"] >= f["width"]:  # portrait or square
                return f.get("link")
        if f.get("quality") == "sd" and f.get("width") and f.get("height"):
            if f["height"] >= f["width"]:
                return f.get("link")
    for f in files:
        if f.get("link") and "video/mp4" in f.get("file_type", ""):
            return f.get("link")
    return None


def download_video(url: str, path: Path, timeout: float = 60.0) -> None:
    """Download a video from URL to path."""
    with httpx.stream("GET", url, timeout=timeout) as r:
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)


# ── LLM plan ─────────────────────────────────────────────────────────────────


def plan_broll(
    words: list[dict],
    model: str = "google/gemini-3-flash-preview",
) -> BrollPlan:
    """Use LLM to suggest 3 b-roll search queries and 3 insertion timestamps."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")

    video_duration = words[-1]["end"] if words else 0
    if video_duration < 15:
        raise ValueError("Video too short for 3 b-roll inserts (need at least ~15s)")

    transcript_lines = [f"[{w['start']:.2f}s] {w['word']}" for w in words]
    transcript_text = " ".join(transcript_lines)

    system_prompt = (
        "You are a short-form video editor adding b-roll to a talking-head clip. "
        "Suggest exactly 3 Pexels video search queries and 3 insertion timestamps.\n\n"
        "## RULES\n"
        "- search_queries: 3 short, concrete Pexels-style queries (e.g. 'house keys hand', 'family first time home', 'mortgage documents'). "
        "Queries should be generic enough that Pexels has stock footage; they should match the TOPIC of the moment.\n"
        "- insert_times_seconds: 3 times in seconds, in STRICT ascending order. "
        "Space them out (e.g. ~20s, ~45s, ~70s for an 80s video). "
        "Do NOT put in first 5 seconds (title card) or last 5 seconds (CTA). "
        "Each insert will add 3 seconds of b-roll, so keep gaps between inserts.\n"
        f"- Video duration: {video_duration:.1f}s\n"
        "- rationale: one sentence per clip (why this query and this time).\n"
    )

    llm = ChatOpenAI(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=120.0,
    ).with_structured_output(BrollPlan)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Transcript ({len(words)} words, {video_duration:.1f}s):\n\n"
                f"{transcript_text}\n\n"
                "Output 3 b-roll search queries and 3 insertion timestamps (seconds)."
            )
        ),
    ]

    print(
        f"Planning b-roll (3 clips, 3s each) from transcript ({len(words)} words, {video_duration:.0f}s)...",
        file=sys.stderr,
    )
    plan = llm.invoke(messages)
    return plan


# ── FFmpeg: extract 3s, scale to 1080x1920, 30fps ────────────────────────────


W, H = 1080, 1920
FPS = 30
BROLL_DURATION = 3.0


def _get_duration(path: Path) -> float:
    p = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(p.stdout.strip())


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def prepare_broll_clip(source_path: Path, output_path: Path, duration: float = BROLL_DURATION) -> None:
    """Extract first `duration` seconds from source, scale to W×H, 30fps, no audio."""
    _run([
        "ffmpeg", "-y",
        "-i", str(source_path),
        "-t", str(duration),
        "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={FPS}",
        "-an",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        *H264_SOCIAL_COLOR_ARGS,
        str(output_path),
    ])


def segment_with_main_audio(
    main_path: Path,
    broll_video_path: Path,
    main_start: float,
    main_duration: float,
    output_path: Path,
) -> None:
    """Create a segment that is b-roll video (3s) with main video's audio (main_start to main_start+main_duration)."""
    _run([
        "ffmpeg", "-y",
        "-i", str(broll_video_path),
        "-ss", str(main_start),
        "-t", str(main_duration),
        "-i", str(main_path),
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output_path),
    ])


def trim_main_segment(main_path: Path, start: float, end: float, output_path: Path) -> None:
    """Trim main video to [start, end], ensure 1080x1920 30fps for concat."""
    _run([
        "ffmpeg", "-y",
        "-i", str(main_path),
        "-ss", str(start),
        "-t", str(end - start),
        "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={FPS}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        *H264_SOCIAL_COLOR_ARGS,
        str(output_path),
    ])


def concat_segments(segment_paths: list[Path], output_path: Path) -> None:
    """Concat all segments in order (concat demuxer)."""
    list_file = output_path.with_suffix(".concat_list.txt")
    with open(list_file, "w") as f:
        for p in segment_paths:
            f.write(f"file '{p.absolute()}'\n")
    _run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_path),
    ])
    list_file.unlink(missing_ok=True)


# ── Main ────────────────────────────────────────────────────────────────────


def add_broll(
    video_path: Path,
    output_path: Path,
    transcript_path: Path | None = None,
    broll_duration: float = BROLL_DURATION,
    num_clips: int = 3,
    api_key: str | None = None,
    dry_run: bool = False,
) -> None:
    """Plan b-roll with LLM, fetch from Pexels, insert into video."""
    api_key = api_key or os.getenv("PEXELS_API_KEY")
    if not api_key:
        raise ValueError("PEXELS_API_KEY environment variable must be set")

    if transcript_path:
        _, words = read_word_transcript_file(transcript_path)
        print(f"Loaded {len(words)} words from {transcript_path}", file=sys.stderr)
    else:
        print("Transcribing video...", file=sys.stderr)
        result = asyncio.run(get_transcript(video_path))
        words = result["words"]

    video_dur = _get_duration(video_path)
    plan = plan_broll(words)

    if dry_run:
        print("DRY RUN: plan and Pexels checks only (no download/encode)", file=sys.stderr)
        print(f"  Queries: {plan.search_queries}", file=sys.stderr)
        print(f"  Insert times (s): {plan.insert_times_seconds}", file=sys.stderr)
        for i, r in enumerate(plan.rationale or []):
            print(f"  Rationale {i+1}: {r}", file=sys.stderr)
        for i, query in enumerate(plan.search_queries[:num_clips]):
            videos = pexels_search_videos(query, api_key, orientation="portrait", per_page=5)
            link = None
            for v in videos:
                link = pick_best_video_file(v, min_duration=broll_duration)
                if link:
                    break
            print(f"  Pexels '{query}': {'OK' if link else 'NO MATCH'}", file=sys.stderr)
        return

    # Validate and sort insert times
    t1, t2, t3 = sorted(plan.insert_times_seconds)
    if t1 < 5 or t3 + broll_duration > video_dur - 5:
        print("Warning: LLM insert times near start/end; clamping.", file=sys.stderr)
    t1 = max(5, min(t1, video_dur - 9))
    t2 = max(t1 + broll_duration + 1, min(t2, video_dur - 6))
    t3 = max(t2 + broll_duration + 1, min(t3, video_dur - 3 - 0.01))

    with tempfile.TemporaryDirectory(prefix="broll_") as tmpdir:
        tmp = Path(tmpdir)
        broll_raw = [tmp / f"broll_raw_{i}.mp4" for i in range(num_clips)]
        broll_prepared = [tmp / f"broll_prep_{i}.mp4" for i in range(num_clips)]

        # Download and prepare 3 b-roll clips
        for i, query in enumerate(plan.search_queries[:num_clips]):
            print(f"  Pexels: '{query}' -> clip {i+1}", file=sys.stderr)
            videos = pexels_search_videos(query, api_key, orientation="portrait", per_page=5)
            if not videos:
                raise RuntimeError(f"No Pexels results for query: {query}")
            link = None
            for v in videos:
                link = pick_best_video_file(v, min_duration=broll_duration)
                if link:
                    break
            if not link:
                raise RuntimeError(f"No suitable video file for query: {query}")
            download_video(link, broll_raw[i])
            prepare_broll_clip(broll_raw[i], broll_prepared[i], duration=broll_duration)

        # Build 7 segments
        segs: list[Path] = []
        # 1: main [0, t1]
        s1 = tmp / "seg1.mp4"
        trim_main_segment(video_path, 0, t1, s1)
        segs.append(s1)
        # 2: b-roll 1 + main audio [t1, t1+3]
        s2 = tmp / "seg2.mp4"
        segment_with_main_audio(video_path, broll_prepared[0], t1, broll_duration, s2)
        segs.append(s2)
        # 3: main [t1+3, t2]
        s3 = tmp / "seg3.mp4"
        trim_main_segment(video_path, t1 + broll_duration, t2, s3)
        segs.append(s3)
        # 4: b-roll 2 + main audio [t2, t2+3]
        s4 = tmp / "seg4.mp4"
        segment_with_main_audio(video_path, broll_prepared[1], t2, broll_duration, s4)
        segs.append(s4)
        # 5: main [t2+3, t3]
        s5 = tmp / "seg5.mp4"
        trim_main_segment(video_path, t2 + broll_duration, t3, s5)
        segs.append(s5)
        # 6: b-roll 3 + main audio [t3, t3+3]
        s6 = tmp / "seg6.mp4"
        segment_with_main_audio(video_path, broll_prepared[2], t3, broll_duration, s6)
        segs.append(s6)
        # 7: main [t3+3, end]
        s7 = tmp / "seg7.mp4"
        trim_main_segment(video_path, t3 + broll_duration, video_dur, s7)
        segs.append(s7)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        concat_segments(segs, output_path)

    print(f"Output: {output_path}", file=sys.stderr)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Add Pexels b-roll to video (3 clips, 3s each)")
    parser.add_argument("video", type=Path, help="Input video")
    parser.add_argument("output", type=Path, help="Output video")
    parser.add_argument("--transcript", type=Path, default=None, help="Word-level transcript JSON")
    parser.add_argument("--broll-duration", type=float, default=3.0, help="Seconds per b-roll clip")
    parser.add_argument("--num-clips", type=int, default=3, help="Number of b-roll clips")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and Pexels check only, no encode")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    add_broll(
        args.video,
        args.output,
        transcript_path=args.transcript,
        broll_duration=args.broll_duration,
        num_clips=args.num_clips,
        dry_run=args.dry_run,
    )
    print(f"✅ Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
