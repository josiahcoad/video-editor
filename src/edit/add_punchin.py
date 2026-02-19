#!/usr/bin/env python3
"""Apply a face-aware punch-in zoom effect to video.

Detects the face at the start of the punch-in, then zooms in over the specified
duration while keeping the face in the same relative position. Uses a fast
ease-in, gentle ease-out motion curve.

By default, hold_time is auto-detected: transcribes first 10s and finds the
first natural pause (utterance boundary or punctuation-like gap) between 3–8s.
Use --hold-time to override.

Usage:
  dotenvx run -f .env -f .env.dev -- uv run python -m src.edit.add_punchin \\
      input.mp4 output.mp4 --duration 0.2 --zoom 1.15
"""

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

from src.edit.portrait_crop import get_face_center_at_time


def _get_video_dimensions(video_path: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            "-select_streams",
            "v:0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    parts = result.stdout.strip().rstrip(",").split(",")
    return int(parts[0]), int(parts[1])


def _get_video_fps(video_path: Path) -> str:
    """Return fps as 'num/den' (e.g. '30000/1001') for ffmpeg -r."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "csv=p=0",
            "-select_streams",
            "v:0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip() or "30/1"


def _ease_out_quad(t: float) -> float:
    """Quadratic ease-out: fast start, gentle end."""
    return 1.0 - (1.0 - t) * (1.0 - t)


# Punctuation that typically ends a phrase/clause (for natural break detection)
_PHRASE_END_PUNCT = {".", ",", "?", "!", ";", ":"}


def _print_transcript_with_release(
    words: list[dict],
    utterances: list[dict],
    break_time: float,
    max_sec: float = 10.0,
) -> None:
    """Print first ~10s of transcript with [RELEASE] at the chosen break point."""
    print("\n--- Transcript (first ~10s) ---")
    has_break = break_time < float("inf")
    if utterances:
        for u in utterances:
            start = float(u.get("start", 0))
            end = float(u.get("end", 0))
            if start >= max_sec:
                break
            text = (u.get("text") or "").strip()
            if not text:
                continue
            if (
                has_break and end <= break_time + 0.05
            ):  # this utterance ends at or before break
                print(f"  {text} [RELEASE]")
            else:
                print(f"  {text}")
    else:
        # Fallback: build from words
        line_parts: list[str] = []
        for w in words:
            end = float(w.get("end", 0))
            if end > max_sec:
                break
            word_text = (w.get("word") or "").strip()
            if not word_text:
                continue
            line_parts.append(word_text)
            if has_break and end <= break_time + 0.05:
                print(f"  {' '.join(line_parts)} [RELEASE]")
                line_parts = []
        if line_parts:
            print(f"  {' '.join(line_parts)}")
    print("---\n")


def _find_natural_hold_time(
    video_path: Path,
    at: float,
    duration: float,
    transcript_path: Path | None,
    min_hold: float = 3.0,
    max_hold: float = 8.0,
    default_hold: float = 5.0,
) -> float:
    """Find hold_time by detecting first natural pause between min_hold and max_hold seconds.

    Natural breaks: utterance boundaries, or word boundaries after punctuation/gaps.
    Only considers the first 10 seconds of content. Prints transcript with [RELEASE] marker.

    Returns:
        hold_time in seconds (duration to hold at zoom before zooming out).
    """
    words: list[dict]
    utterances: list[dict] = []

    if transcript_path and transcript_path.exists():
        with transcript_path.open() as f:
            words = json.load(f)
        # Sibling utterances file (e.g. 05_titled-utterances.json from 05_titled-words.json)
        utt_path = transcript_path.with_stem(
            transcript_path.stem.replace("-words", "-utterances")
        )
        if utt_path.exists():
            with utt_path.open() as f:
                utterances = json.load(f)
    else:
        from src.edit.get_transcript import get_transcript

        result = asyncio.run(get_transcript(video_path, max_duration=10.0))
        words = result["words"]
        utterances = result.get("utterances", [])

    # Target window: break should occur between min_hold and max_hold seconds
    break_min = min_hold
    break_max = max_hold
    candidates: list[float] = []

    # 1. Utterance boundaries (phrase/sentence ends from Deepgram)
    for u in utterances:
        end = float(u.get("end", 0))
        if break_min <= end <= break_max:
            candidates.append(end)

    # 2. Word boundaries: gap > 0.35s (natural pause) or word ending with punctuation
    for i, w in enumerate(words):
        end = float(w.get("end", 0))
        if end > break_max:
            break
        if end < break_min:
            continue
        word_text = (w.get("word") or "").strip()
        # Trailing punctuation suggests phrase end
        if word_text and word_text[-1] in _PHRASE_END_PUNCT:
            candidates.append(end)
        # Long gap after this word
        if i + 1 < len(words):
            gap = float(words[i + 1].get("start", end)) - end
            if gap >= 0.35:
                candidates.append(end)

    if not candidates:
        # No natural break found; show transcript and note default
        _print_transcript_with_release(words, utterances, float("inf"), max_sec=10.0)
        print("  [RELEASE] (no natural break in 3–8s, using default 5s)\n")
        return default_hold

    # Use earliest natural break in the window
    break_time = min(candidates)
    hold_time = break_time - (at + duration)
    hold_time = max(0.1, hold_time)
    _print_transcript_with_release(words, utterances, break_time, max_sec=10.0)
    return hold_time


def apply_punchin(
    input_video: Path,
    output_video: Path,
    *,
    duration: float = 1.0,
    zoom: float = 1.15,
    at: float = 0.0,
    hold_time: float | None = None,
    transcript_path: Path | None = None,
) -> Path:
    """Apply a face-centered punch-in zoom effect.

    Zooms in uniformly around the detected face, holds, then zooms back out.

    Args:
        input_video: Input video file.
        output_video: Output video file.
        duration: Seconds over which zoom animates in/out (default 1.0).
        zoom: Target zoom factor (default 1.15).
        at: Start time of the punch-in in seconds (default 0).
        hold_time: Seconds to hold at zoomed level. If None, auto-detect from transcript
            (first natural pause between 3–8s).
        transcript_path: Optional path to *-words.json. When hold_time is None, used for
            break detection; if not set, transcribes first 10s of video.

    Returns:
        Path to output video.
    """
    if hold_time is None:
        hold_time = _find_natural_hold_time(
            input_video,
            at=at,
            duration=duration,
            transcript_path=transcript_path,
        )
        print(f"Natural break → hold {hold_time:.1f}s")
    iw, ih = _get_video_dimensions(input_video)
    fps = _get_video_fps(input_video)
    cx, cy = get_face_center_at_time(input_video, at, iw, ih)

    # Ensure output dimensions are even (required by h264)
    out_w = iw - (iw % 2)
    out_h = ih - (ih % 2)

    # Zoom expression: 3 phases. Zoom in (ease-out curve), hold, then stay zoomed.
    # "Ease-out" = fast start, gentle landing on the zoom-IN (not zooming back out).
    if "/" in fps:
        num, den = fps.split("/", 1)
        fps_val = float(num) / float(den)
    else:
        fps_val = float(fps)
    at_frame = at * fps_val
    zoom_in_end_frame = (at + duration) * fps_val
    zoom_in_frames = max(1, int(duration * fps_val))
    t_in = f"min(1,(in-{at_frame})/{zoom_in_frames})"
    # Linear: equal motion every frame so zoom feels immediate from frame 1
    zoom_in = f"1+({zoom}-1)*{t_in}"
    zoom_expr = (
        f"if(lt(in,{at_frame}),1," f"if(lt(in,{zoom_in_end_frame}),{zoom_in},{zoom}))"
    )

    # Pan: keep (cx, cy) at output center. In zoomed space the point is at (cx*zoom, cy*zoom);
    # crop center must match that, so x = cx*zoom - iw/2. Clamp to valid crop bounds.
    x_expr = f"max(0,min({cx}*zoom-iw/2,iw*zoom-iw))"
    y_expr = f"max(0,min({cy}*zoom-ih/2,ih*zoom-ih))"

    # zoompan: z=zoom, x,y=pan, d=1 (1 output frame per input frame), s=output size
    # zoompan defaults to 25fps output; setpts corrects timestamps to input fps for A/V sync
    num, den = fps.split("/", 1) if "/" in fps else (fps, "1")
    setpts_expr = f"PTS*25*{den}/{num}"
    vf = (
        f"zoompan=z='{zoom_expr}':d=1:x='{x_expr}':y='{y_expr}':s={out_w}x{out_h},"
        f"setpts={setpts_expr}"
    )

    print(f"Face center at t={at}s: ({cx}, {cy})")
    print(f"Punch-in: {duration}s in, hold {hold_time}s, zoom {zoom}x")

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-c:a",
            "copy",
            str(output_video),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stderr or result.stdout or "ffmpeg failed", file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, result.args)

    return output_video


# Module entry point (same as apply_punchin).
add_punchin = apply_punchin


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply face-aware punch-in zoom effect"
    )
    parser.add_argument("input", type=Path, help="Input video file")
    parser.add_argument("output", type=Path, help="Output video file")
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Punch-in animation duration in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.15,
        help="Target zoom factor (default: 1.15)",
    )
    parser.add_argument(
        "--at",
        type=float,
        default=0.0,
        help="Start time of punch-in in seconds (default: 0)",
    )
    parser.add_argument(
        "--hold-time",
        type=float,
        default=None,
        dest="hold_time",
        help="Seconds to hold at zoomed level. Default: auto-detect from transcript (3–8s)",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        dest="transcript_path",
        help="Path to *-words.json for break detection (avoids re-transcribing)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input video not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    apply_punchin(
        args.input,
        args.output,
        duration=args.duration,
        zoom=args.zoom,
        at=args.at,
        hold_time=args.hold_time,
        transcript_path=args.transcript_path,
    )
    print(f"✅ Wrote {args.output}")


if __name__ == "__main__":
    main()
