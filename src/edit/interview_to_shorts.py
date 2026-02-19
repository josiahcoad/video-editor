#!/usr/bin/env python3
"""End-to-end pipeline: interview video → finished short-form videos.

Takes a raw interview video and produces N polished shorts, each structured
as [hook] + [body] + [CTA].

Pipeline:
  Phase 1 — Analysis (LLM-powered)
    1. Transcribe video (word-level + utterance-level via Deepgram)
    2. Analyse transcript to identify hooks, body segments, and CTA
    3. Propose N shorts (each: hook + body + cta timestamp ranges)

  Phase 2 — Enrichment
    4. Generate search-optimised titles (AnswerThePublic + LLM)
    5. Search & select background music (Openverse + LLM)

  Phase 3 — Rendering (per short)
    6. apply_cuts   → stitch hook + body + CTA
    7. portrait_crop → 9:16 portrait crop
    8. jump_cuts    → remove dead air + filler words + zoom effects
    9. add_music    → mix in background music (round-robin)
   10. add_subtitles → auto-transcribe + burn captions
   11. add_title    → overlay title card

Usage (from video-editor repo root):
  dotenvx run -f .env -- uv run python -m src.edit.interview_to_shorts \\
      path/to/interview.mp4 \\
      path/to/project_dir/ \\
      --num-shorts 10 \\
      --target-duration 35 \\
      --tolerance 15 \\
      --music-vibe "piano, corporate"

  # Resume from a specific phase:
  dotenvx run -f .env -- uv run python -m src.edit.interview_to_shorts \\
      path/to/interview.mp4 \\
      path/to/project_dir/ \\
      --from-phase 3

Requires: OPENROUTER_API_KEY, DEEPGRAM_API_KEY, RAPID_API_KEY
"""

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Ensure repo root is importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.edit.add_background_music import _get_duration as get_music_duration
from src.edit.add_background_music import add_music
from src.edit.portrait_crop import get_portrait_crop_params, portrait_crop_per_cut
from src.edit.get_transcript import get_transcript
from src.edit.music_search import search_and_select_music
from src.edit.propose_interview_cuts import (
    DEFAULT_NUM_SHORTS,
    DEFAULT_TARGET_DURATION,
    DEFAULT_TOLERANCE,
    propose_interview_cuts,
)
from suggest_video_title import (
    extract_seed_keywords,
    get_atp_queries,
    load_transcript,
    pick_titles,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TITLE_DURATION = 4.0  # seconds to show title overlay
DEFAULT_MUSIC_VIBE = "piano, corporate"
DEFAULT_MUSIC_TRACKS_COUNT = 4

# Filler words passed to jump-cut removal
FILLER_WORDS = {"uh", "um", "mhmm", "mm-mm", "uh-uh", "uh-huh", "nuh-uh"}


# ---------------------------------------------------------------------------
# Phase 1: Transcribe + Analyse + Propose
# ---------------------------------------------------------------------------


async def transcribe_video(video_path: Path, project_dir: Path) -> tuple[Path, Path]:
    """Transcribe video with filler word detection.

    Returns (words_json_path, utterances_json_path).
    """
    words_path = video_path.with_name(video_path.stem + "-words.json")
    utterances_path = video_path.with_name(video_path.stem + "-utterances.json")

    if words_path.exists() and utterances_path.exists():
        print(f"✅ Transcripts already exist, skipping transcription")
        print(f"   Words: {words_path}")
        print(f"   Utterances: {utterances_path}")
        return words_path, utterances_path

    print(f"🎙️  Transcribing {video_path.name} (with filler word detection)...")
    result = await get_transcript(video_path, filler_words=True)

    words_path.write_text(json.dumps(result["words"], indent=2))
    utterances_path.write_text(json.dumps(result["utterances"], indent=2))

    word_count = len(result["words"])
    print(f"✅ Transcription complete: {word_count} words")
    return words_path, utterances_path


# ---------------------------------------------------------------------------
# Phase 2: Titles + Music
# ---------------------------------------------------------------------------


async def generate_titles(
    segments: list[dict[str, Any]], words_json: Path
) -> list[dict[str, Any]]:
    """Generate search-optimised titles for each short using ATP demand data."""
    full_transcript = load_transcript(words_json)
    all_words: list[dict[str, Any]] = json.loads(words_json.read_text())

    print("\n🔑 Extracting seed keywords from full transcript...")
    keywords = await extract_seed_keywords(full_transcript)
    print(f"   Seeds: {keywords}")

    print("🌐 Querying AnswerThePublic...")
    atp_queries = await get_atp_queries(keywords)
    print(f"   Got {len(atp_queries)} queries with volume")

    if not atp_queries:
        print("   ⚠️  No ATP results — titles will be generated from transcript only")
        # Fallback: generate titles from transcript alone
        for seg in segments:
            if not seg.get("title"):
                seg["title"] = seg.get("description", f"Short {seg['segment']}")
        return segments

    for seg in segments:
        num = seg["segment"]
        body_ranges_str = seg.get("body", "")
        if not body_ranges_str:
            continue

        # Body can be single "start:end" or comma-separated "start:end,start:end,..."
        body_text_parts: list[str] = []
        for part in body_ranges_str.split(","):
            part = part.strip()
            if not part or ":" not in part:
                continue
            start_str, end_str = part.split(":", 1)
            try:
                start_t = float(start_str)
                end_t = float(end_str)
            except ValueError:
                continue
            body_text_parts.append(
                " ".join(
                    w["word"]
                    for w in all_words
                    if w["start"] >= start_t - 0.05 and w["end"] <= end_t + 0.05
                )
            )
        body_text = " ".join(body_text_parts)
        if not body_text.strip():
            print(f"   Short {num}: no body text found, using description")
            seg["title"] = seg.get("description", f"Short {num}")
            continue

        print(f"\n📝 Short {num}: generating title...")
        try:
            suggestions = await pick_titles(body_text, atp_queries, count=1)
            if suggestions.titles:
                seg["title"] = suggestions.titles[0].title
                print(f"   → '{seg['title']}'")
                print(f"   Inspired by: {suggestions.titles[0].inspired_by}")
            else:
                seg["title"] = seg.get("description", f"Short {num}")
        except Exception as e:
            print(f"   ⚠️  Title gen failed: {e}")
            seg["title"] = seg.get("description", f"Short {num}")

    return segments


# ---------------------------------------------------------------------------
# Phase 3: Render pipeline (per short)
# ---------------------------------------------------------------------------


def add_buffer_to_cuts(
    cuts_str: str, lead_s: float = 0.18, trail_s: float = 0.40
) -> str:
    """Pad each cut range to avoid clipping word onsets and tails."""
    parts = [p.strip() for p in cuts_str.split(",") if p.strip()]
    if not parts:
        return cuts_str

    buffered = []
    for i, part in enumerate(parts):
        start_str, end_str = part.split(":", 1)
        start = float(start_str)
        end = float(end_str)
        if i == 0:
            start = max(0.0, start - lead_s)
        end += trail_s
        buffered.append(f"{start}:{end}")

    return ",".join(buffered)


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess command from the repo root."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, cwd=str(_REPO_ROOT))


def render_short(
    seg: dict[str, Any],
    video_path: Path,
    words_json: Path,
    output_dir: Path,
    music_dir: Path,
    music_tracks: list[str],
    title_duration: float = TITLE_DURATION,
) -> Path | None:
    """Render a single short through the full pipeline.

    Returns the final output path, or None on failure.
    """
    num = seg["segment"]
    label = f"short_{num:02d}"
    seg_dir = output_dir / label
    seg_dir.mkdir(parents=True, exist_ok=True)

    cuts_buffered = add_buffer_to_cuts(seg["cuts"])
    cut_mp4 = seg_dir / "01_cut.mp4"
    portrait_mp4 = seg_dir / "02_portrait.mp4"
    jump_mp4 = seg_dir / "03_jumpcut.mp4"
    final_mp4 = seg_dir / "04_final.mp4"
    captioned_mp4 = seg_dir / "05_captioned.mp4"
    titled_mp4 = seg_dir / "06_titled.mp4"

    title = seg.get("title", f"Short {num}")

    print(f"\n{'='*60}")
    print(f"  Short {num}: {title}")
    print(f"{'='*60}")

    try:
        # ── Step 1: apply_cuts → stitch hook + body + CTA ──
        print(f"\n--- {label} Step 1: apply_cuts ---")
        _run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "src.edit.apply_cuts",
                str(video_path),
                str(cut_mp4),
                "--cuts",
                cuts_buffered,
            ],
            check=True,
        )

        # ── Step 2: 9:16 portrait crop ──
        boundaries_json = cut_mp4.with_suffix(".boundaries.json")
        print(f"\n--- {label} Step 2: 9:16 portrait crop ---")
        try:
            boundaries_data = json.loads(boundaries_json.read_text())
            boundaries = boundaries_data.get("boundaries", [])
            portrait_crop_per_cut(cut_mp4.resolve(), portrait_mp4, boundaries)
        except Exception as e:
            print(f"  Per-cut portrait crop failed, falling back: {e}")
            try:
                crop_x, crop_w, crop_h = get_portrait_crop_params(cut_mp4.resolve())
                crop_vf = f"crop={crop_w}:{crop_h}:{crop_x}:0"
            except Exception:
                crop_vf = (
                    "crop=trunc(ih*9/16/2)*2:ih:" "trunc((iw-trunc(ih*9/16/2)*2)/2)*2:0"
                )
            _run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(cut_mp4),
                    "-vf",
                    crop_vf,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "23",
                    "-c:a",
                    "copy",
                    str(portrait_mp4),
                ],
                check=True,
            )

        # ── Step 3: jump cuts (dead air + filler word removal + zoom) ──
        print(f"\n--- {label} Step 3: jump cuts ---")
        jump_cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "src.edit.add_jump_cuts",
            str(portrait_mp4),
            str(jump_mp4),
            "--gap-threshold",
            "0.4",
            "--max-cut-duration",
            "7.0",
            "--zoom-factor",
            "1.2",
            "--remove-fillers",
        ]
        if words_json.exists():
            jump_cmd.extend(["--transcript", str(words_json)])
        _run(jump_cmd, check=True)

        # ── Step 4: add background music (round-robin) ──
        if music_tracks:
            track_name = music_tracks[(num - 1) % len(music_tracks)]
            track_file = music_dir / track_name
            if track_file.exists():
                print(f"\n--- {label} Step 4: add music ({track_name}) ---")
                track_dur = get_music_duration(track_file)
                offset = 0.0 if track_dur < 60.0 else 5.0
                add_music(jump_mp4, track_file, final_mp4, music_start_offset=offset)
            else:
                print(f"  ⚠️  Track not found: {track_file}")
                final_mp4 = jump_mp4
        else:
            print(f"\n--- {label} Step 4: no music (skipping) ---")
            final_mp4 = jump_mp4

        # ── Step 5: add captions ──
        music_src = final_mp4 if final_mp4.exists() else jump_mp4
        print(f"\n--- {label} Step 5: add captions ---")
        _run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "src.edit.add_subtitles",
                str(music_src),
                "--output",
                str(captioned_mp4),
                "--style",
                "outline",
                "--delay",
                str(title_duration),
            ],
            check=True,
        )

        # ── Step 6: add title overlay ──
        print(f"\n--- {label} Step 6: add title ('{title}') ---")
        _run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "src.edit.add_title",
                str(captioned_mp4),
                title,
                str(titled_mp4),
                "--duration",
                str(title_duration),
                "--anchor",
                "top",
                "--height",
                "95",
            ],
            check=True,
        )

        # Report
        out_file = titled_mp4 if titled_mp4.exists() else captioned_mp4
        try:
            dur_result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    str(out_file),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            dur = float(dur_result.stdout.strip())
            print(f"  ✅ Short {num} final: {dur:.1f}s — {out_file}")
        except Exception:
            print(f"  ✅ Short {num} done — {out_file}")

        return out_file

    except Exception as exc:
        print(f"\n  ❌ Short {num} FAILED: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


async def run_pipeline(
    video_path: Path,
    project_dir: Path,
    num_shorts: int | None = None,
    cuts_per_question: int | None = None,
    target_duration: float = DEFAULT_TARGET_DURATION,
    tolerance: float = DEFAULT_TOLERANCE,
    music_vibe: str = DEFAULT_MUSIC_VIBE,
    from_phase: int = 1,
    from_short: int = 1,
) -> int:
    """Run the full interview-to-shorts pipeline.

    Returns 0 on success, 1 on failure.
    """
    # Set up directory structure
    editing_dir = project_dir / "editing"
    output_dir = editing_dir / "outputs"
    music_dir = editing_dir / "music"
    cuts_json = output_dir / "shorts_cuts.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ======================================================================
    # PHASE 1: Transcribe + Analyse + Propose
    # ======================================================================

    if from_phase <= 1:
        print(f"\n{'='*60}")
        print("  PHASE 1: Transcribe + Analyse + Propose")
        print(f"{'='*60}")

        # Step 1: Transcribe
        words_json, utterances_json = await transcribe_video(video_path, project_dir)

        # Step 2-3: Propose shorts
        if cuts_json.exists():
            print(f"\n✅ {cuts_json} already exists. Using existing cuts.")
            print(f"   (Delete it to re-propose cuts)")
            segments = json.loads(cuts_json.read_text())
        else:
            segments = await propose_interview_cuts(
                words_json,
                utterances_json,
                num_shorts=num_shorts,
                cuts_per_question=cuts_per_question,
                duration=target_duration,
                tolerance=tolerance,
            )
            cuts_json.write_text(json.dumps(segments, indent=4))
            print(f"\n✅ Cuts written to {cuts_json}")
            print(
                f"   ⏸️  Review the cuts and edit if needed, then re-run with --from-phase 2"
            )
    else:
        # Load existing data
        words_json = video_path.with_name(video_path.stem + "-words.json")
        utterances_json = video_path.with_name(video_path.stem + "-utterances.json")
        if not cuts_json.exists():
            print(f"❌ {cuts_json} not found. Run from phase 1 first.")
            return 1
        segments = json.loads(cuts_json.read_text())

    # ======================================================================
    # PHASE 2: Titles + Music
    # ======================================================================

    if from_phase <= 2:
        print(f"\n{'='*60}")
        print("  PHASE 2: Generate Titles + Select Music")
        print(f"{'='*60}")

        words_json = video_path.with_name(video_path.stem + "-words.json")

        # Generate titles
        has_titles = all(seg.get("title") for seg in segments)
        if has_titles:
            print(f"\n✅ All shorts already have titles. Skipping title generation.")
            print(f"   (Clear titles in {cuts_json} to regenerate)")
        else:
            segments = await generate_titles(segments, words_json)
            cuts_json.write_text(json.dumps(segments, indent=4))
            print(f"\n✅ Titles written to {cuts_json}")

        # Search and select music
        music_tracks = await search_and_select_music(
            music_dir, music_vibe, DEFAULT_MUSIC_TRACKS_COUNT
        )
    else:
        # Load existing music
        existing_tracks = sorted(music_dir.glob("*.mp3"))
        music_tracks = [f.name for f in existing_tracks]

    # ======================================================================
    # PHASE 3: Render all shorts
    # ======================================================================

    if from_phase <= 3:
        print(f"\n{'='*60}")
        print("  PHASE 3: Render Shorts")
        print(f"{'='*60}")

        words_json = video_path.with_name(video_path.stem + "-words.json")
        failed: list[int] = []

        for seg in segments:
            num = seg["segment"]
            if num < from_short:
                continue

            result = render_short(
                seg,
                video_path,
                words_json,
                output_dir,
                music_dir,
                music_tracks,
            )
            if result is None:
                failed.append(num)

        # Print summary
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        for seg in segments:
            num = seg["segment"]
            title = seg.get("title", "")
            final = output_dir / f"short_{num:02d}" / "06_titled.mp4"
            if not final.exists():
                final = output_dir / f"short_{num:02d}" / "04_final.mp4"
            if final.exists():
                try:
                    dur_result = subprocess.run(
                        [
                            "ffprobe",
                            "-v",
                            "quiet",
                            "-show_entries",
                            "format=duration",
                            "-of",
                            "csv=p=0",
                            str(final),
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    dur = float(dur_result.stdout.strip())
                    status = f"{dur:.1f}s"
                except Exception:
                    status = "done"
            else:
                status = "MISSING"
            print(f"  Short {num:2d}: [{status:>6s}] {title}")

        if failed:
            print(f"\n⚠️  Failed shorts: {failed}")
            print(f"   Re-run with: --from-phase 3 --from-short <N>")

        print(f"\nFinal files: {output_dir}/short_XX/06_titled.mp4")
        return 1 if failed else 0

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: interview video → finished shorts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Full pipeline from scratch:\n"
            "  dotenvx run -f .env -- uv run python -m src.edit.interview_to_shorts \\\n"
            "      interview.mp4 projects/ClientName/ --num-shorts 10\n\n"
            "  # Resume from rendering phase:\n"
            "  dotenvx run -f .env -- uv run python -m src.edit.interview_to_shorts \\\n"
            "      interview.mp4 projects/ClientName/ --from-phase 3\n\n"
            "  # Re-render from a specific short:\n"
            "  dotenvx run -f .env -- uv run python -m src.edit.interview_to_shorts \\\n"
            "      interview.mp4 projects/ClientName/ --from-phase 3 --from-short 5\n"
        ),
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Path to the interview video file",
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        help="Project directory (e.g. projects/ClientName/)",
    )
    parser.add_argument(
        "--num-shorts",
        type=int,
        default=None,
        help=f"Number of shorts (default: {DEFAULT_NUM_SHORTS}). Ignored if --cuts-per-question is set.",
    )
    parser.add_argument(
        "--cuts-per-question",
        type=int,
        default=None,
        help="Propose this many shorts per identified question/topic (LLM identifies number_questions)",
    )
    parser.add_argument(
        "--target-duration",
        type=float,
        default=DEFAULT_TARGET_DURATION,
        help=f"Target duration per short in seconds (default: {DEFAULT_TARGET_DURATION})",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Duration tolerance ± seconds, e.g. 35±15 (default: {DEFAULT_TOLERANCE})",
    )
    parser.add_argument(
        "--music-vibe",
        type=str,
        default=DEFAULT_MUSIC_VIBE,
        help=f"Music vibe/genre, comma-separated (default: '{DEFAULT_MUSIC_VIBE}')",
    )
    parser.add_argument(
        "--from-phase",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Resume from a specific phase (1=transcribe, 2=titles+music, 3=render)",
    )
    parser.add_argument(
        "--from-short",
        type=int,
        default=1,
        help="Resume rendering from a specific short number (use with --from-phase 3)",
    )

    args = parser.parse_args()

    if not args.video.exists():
        print(f"❌ Video not found: {args.video}")
        return 1

    return asyncio.run(
        run_pipeline(
            video_path=args.video.resolve(),
            project_dir=args.project_dir.resolve(),
            num_shorts=args.num_shorts
            if args.num_shorts is not None
            else DEFAULT_NUM_SHORTS,
            cuts_per_question=args.cuts_per_question,
            target_duration=args.target_duration,
            tolerance=args.tolerance,
            music_vibe=args.music_vibe,
            from_phase=args.from_phase,
            from_short=args.from_short,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
