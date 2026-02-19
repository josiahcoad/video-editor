#!/usr/bin/env python3
"""Apply aggressive dead space removal + artificial zoom-based jump cuts.

Combines two operations in a single ffmpeg pass:
1. Removes gaps > gap_threshold (default 0.4s) between spoken words
   → Dead space removal is INVISIBLE: no zoom change, just silence cut.
2. Splits continuous segments > max_cut_duration at word boundaries
3. Alternates zoom ONLY at long-segment split boundaries (not dead space cuts)

V1 (current): Uniform zoom with top-half bias (assumes face is in upper portion).

FUTURE v2 — Smart face-aware zoom:
- Use OpenCV face detection with adaptive sampling:
  Start checking every 10s. If face position moves > threshold between
  samples, subdivide that interval (5s → 2.5s → 1s → 0.5s → 0.1s)
  to track movement precisely.
- Center crop on detected face position per segment
- Apply rule-of-thirds positioning for more cinematic framing
- Smooth pan between face positions across segments (Ken Burns effect)
"""

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

from typing import Any

# Filler words that Deepgram detects when filler_words=True is enabled.
# These are filtered out of the word list before building speaking segments
# so that the gap around each filler causes a cut that removes it.
FILLER_WORDS = {"uh", "um", "mhmm", "mm-mm", "uh-uh", "uh-huh", "nuh-uh"}


def _get_video_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


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


def _remap_words_to_cut_timeline(
    words: list[dict],
    source_ranges: list[dict],
    tolerance: float = 0.05,
) -> list[dict]:
    """Filter and remap words from a full transcript to a cut video's timeline.

    When apply_cuts.py concatenates timestamp ranges [454.4:471.5, 422.6:454.0]
    into a single video, the output timeline is:
      0s-17.1s  = original 454.4-471.5
      17.1s-48.5s = original 422.6-454.0

    This function takes the full original transcript and returns only the words
    that fall within the source ranges, with their timestamps remapped to the
    cut video's timeline.

    Args:
        words: Full word-level transcript (original timestamps)
        source_ranges: List of {"start": float, "end": float} from apply_cuts sidecar
        tolerance: Matching tolerance in seconds

    Returns:
        List of word dicts with remapped start/end timestamps
    """
    remapped: list[dict] = []
    output_offset = 0.0

    for src_range in source_ranges:
        range_start = src_range["start"]
        range_end = src_range["end"]
        range_duration = range_end - range_start

        # Find words within this source range
        range_words = [
            w
            for w in words
            if w["start"] >= range_start - tolerance
            and w["end"] <= range_end + tolerance
        ]

        for w in range_words:
            remapped_word = dict(w)
            # Remap: offset within the source range + cumulative output offset
            remapped_word["start"] = output_offset + (w["start"] - range_start)
            remapped_word["end"] = output_offset + (w["end"] - range_start)
            remapped.append(remapped_word)

        output_offset += range_duration

    return remapped


def _find_speaking_segments(
    words: list[dict], gap_threshold: float = 0.4
) -> list[tuple[float, float]]:
    """Group words into continuous speaking segments (gaps < threshold)."""
    if not words:
        return []

    segments: list[tuple[float, float]] = []
    seg_start = words[0]["start"]
    seg_end = words[0]["end"]

    for i in range(1, len(words)):
        gap = words[i]["start"] - seg_end
        if gap > gap_threshold:
            segments.append((seg_start, seg_end))
            seg_start = words[i]["start"]
        seg_end = words[i]["end"]

    segments.append((seg_start, seg_end))
    return segments


def _find_best_split_point(
    seg_words: list[dict],
    current_start: float,
    max_duration: float,
) -> int | None:
    """Find the best word index to split at, preferring natural boundaries.

    Scores each candidate split point using a weighted combination of:
    1. Gap duration (longer pause = more likely a sentence/clause boundary)
    2. Position balance (prefer splits near the middle of the window so
       sub-segments are roughly equal length — avoids 1s + 4s splits)

    Returns the index of the word BEFORE the split (i.e. the segment ends
    after this word), or None if no valid split found.
    """
    MIN_SUB_SEG = 2.0  # minimum sub-segment length in seconds
    # Find words in the window [current_start, current_start + max_duration]
    raw_candidates: list[tuple[float, float, int]] = []  # (gap, split_time, word_index)

    ideal_split = current_start + max_duration / 2.0  # midpoint of window

    for i, w in enumerate(seg_words):
        elapsed = w["end"] - current_start
        if elapsed < MIN_SUB_SEG:
            # Too short — don't create tiny sub-segments
            continue
        if elapsed > max_duration:
            # Past the limit — if we have candidates, use the best one;
            # otherwise force-split at this word
            if not raw_candidates:
                return i
            break

        if i > 0:
            gap = w["start"] - seg_words[i - 1]["end"]
            split_time = seg_words[i - 1]["end"]
            raw_candidates.append((gap, split_time, i - 1))

    if not raw_candidates:
        return None

    # Relative gap scoring: normalize against the max gap in this window.
    # This ensures the longest pause always scores 1.0, regardless of whether
    # the speaker pauses for 0.3s or 0.08s between sentences.
    max_gap = max(c[0] for c in raw_candidates)
    if max_gap <= 0:
        max_gap = 0.01  # avoid division by zero

    scored: list[tuple[float, int]] = []
    for gap, split_time, word_idx in raw_candidates:
        gap_score = gap / max_gap  # 0.0 to 1.0, relative to this window

        # Balance score: prefer splits near the middle of the window
        dist_from_ideal = abs(split_time - ideal_split)
        max_dist = max_duration / 2.0
        balance_score = max(0.0, 1.0 - (dist_from_ideal / max_dist))

        # Gap-dominant scoring (80/20): gap is the primary signal,
        # balance only breaks ties between equally-paused candidates.
        score = 0.8 * gap_score + 0.2 * balance_score
        scored.append((score, word_idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _split_long_segments(
    segments: list[tuple[float, float]],
    words: list[dict],
    max_duration: float = 5.0,
) -> list[tuple[float, float, bool]]:
    """Split segments > max_duration at natural word boundaries.

    Prefers splitting at the longest pause between words (sentence/clause
    boundaries) rather than at an arbitrary duration threshold.

    Returns list of (start, end, is_continuation) tuples. is_continuation=True
    means this sub-segment was split from a longer speaking segment (and should
    trigger a zoom change). is_continuation=False means it's the first part of
    a speaking segment (zoom stays the same as the current level).
    """
    result: list[tuple[float, float, bool]] = []

    for seg_start, seg_end in segments:
        if seg_end - seg_start <= max_duration:
            result.append((seg_start, seg_end, False))
            continue

        # Get words in this segment
        seg_words = [
            w
            for w in words
            if w["start"] >= seg_start - 0.01 and w["end"] <= seg_end + 0.01
        ]

        if len(seg_words) <= 1:
            result.append((seg_start, seg_end, False))
            continue

        # Iteratively find best split points
        remaining_words = seg_words
        current_start = seg_start
        is_first_in_group = True

        while remaining_words:
            remaining_duration = remaining_words[-1]["end"] - current_start
            if remaining_duration <= max_duration:
                # What's left fits in one segment — preserve the padded
                # seg_end so trailing EDGE_PAD / TRAIL_PAD isn't lost.
                last_word_end = remaining_words[-1]["end"]
                end_ts = max(last_word_end, seg_end)
                result.append(
                    (
                        current_start,
                        end_ts,
                        not is_first_in_group,
                    )
                )
                break

            split_idx = _find_best_split_point(
                remaining_words, current_start, max_duration
            )

            if split_idx is None:
                # Can't find a split — keep the rest as one segment.
                # Preserve padded seg_end.
                last_word_end = remaining_words[-1]["end"]
                end_ts = max(last_word_end, seg_end)
                result.append(
                    (
                        current_start,
                        end_ts,
                        not is_first_in_group,
                    )
                )
                break

            # End this sub-segment after the split word
            split_word = remaining_words[split_idx]
            result.append(
                (
                    current_start,
                    split_word["end"],
                    not is_first_in_group,
                )
            )

            # Next sub-segment starts at next word
            remaining_words = remaining_words[split_idx + 1 :]
            if remaining_words:
                current_start = remaining_words[0]["start"]
            is_first_in_group = False

    return result


def apply_jump_cuts(
    input_video: Path,
    output_video: Path,
    words: list[dict],
    max_cut_duration: float = 5.0,
    zoom_factor: float = 1.2,
    gap_threshold: float = 0.4,
    speedup: float = 1.0,
    cut_boundaries: list[float] | None = None,
    remove_fillers: bool = False,
) -> Path:
    """Remove dead space and apply alternating zoom cuts.

    Args:
        input_video: Input video file
        output_video: Output video file
        words: Word-level transcript (list of dicts with 'start' and 'end')
        max_cut_duration: Max seconds before introducing a zoom cut (default: 5.0)
        zoom_factor: Zoom multiplier for 'tight' shots (default: 1.2)
        gap_threshold: Max gap between words before cutting (default: 0.4)
        speedup: Speedup multiplier (default: 1.0 = no speedup). When > 1.0,
                 the transcript should be from the ORIGINAL (pre-speedup) video.
                 Gap analysis uses the original gaps (which have real signal),
                 then timestamps are scaled by 1/speedup for the output.
        cut_boundaries: Timestamps (in the OUTPUT video timeline) where content
                 cuts from apply_cuts already create visual discontinuities.
                 Zoom toggles are suppressed within ±1s of these points to
                 avoid "double jumps" (content jump + zoom change).
        remove_fillers: If True, filter out filler words (uh, um, etc.) from
                 the word list before building speaking segments. This creates
                 gaps where fillers were, which then get cut.

    Returns:
        Path to output video
    """
    width, height = _get_video_dimensions(input_video)
    input_duration = _get_video_duration(input_video)

    # When speedup > 1.0, the transcript is from the ORIGINAL video.
    # We do gap analysis on original timestamps (which have real pauses),
    # then scale everything by 1/speedup for the actual ffmpeg trim/atempo.
    # This avoids the problem where speedup compresses micro-pauses below
    # Deepgram's resolution, making all gaps appear as 0.000s.

    # Step 0: Optionally remove filler words before gap analysis.
    # Removing fillers from the word list creates larger gaps where fillers
    # were, so _find_speaking_segments naturally excludes them.
    if remove_fillers:
        original_count = len(words)
        words = [
            w for w in words if w.get("word", "").strip().lower() not in FILLER_WORDS
        ]
        removed = original_count - len(words)
        if removed:
            print(f"🗑️  Removed {removed} filler word(s) (uh/um/mhmm)")

    # Step 1: Find continuous speaking segments (using original-pace gaps)
    segments = _find_speaking_segments(words, gap_threshold)

    # Pad EVERY segment boundary to preserve consonant onsets and word tails.
    # Deepgram timestamps are often tight to the vowel, so the last ~150-200ms
    # of each word (trailing consonant / vocal decay) gets cut without padding.
    LEAD_PAD = 0.12  # seconds before each segment's first word
    TRAIL_PAD = 0.30  # seconds after each segment's last word
    EDGE_PAD = 0.40  # extra generous pad for the very first/last segments
    MIN_GAP = 0.02  # minimum gap to preserve between padded segments

    max_ts = input_duration * speedup if speedup > 1.0 else input_duration
    if segments:
        padded_segs: list[tuple[float, float]] = []
        for i, (s, e) in enumerate(segments):
            # Leading pad — extra generous for the very first segment
            lead = EDGE_PAD if i == 0 else LEAD_PAD
            new_s = max(0.0, s - lead)

            # Trailing pad — extra generous for the very last segment
            trail = EDGE_PAD if i == len(segments) - 1 else TRAIL_PAD
            new_e = min(max_ts, e + trail)

            # Clamp so we don't overlap with the raw boundary of neighbours
            if i > 0:
                prev_end = segments[i - 1][1]
                new_s = max(new_s, prev_end + MIN_GAP)
            if i < len(segments) - 1:
                next_start = segments[i + 1][0]
                new_e = min(new_e, next_start - MIN_GAP)

            padded_segs.append((new_s, new_e))
        segments = padded_segs

    total_speech = sum(e - s for s, e in segments)
    dead_in_original = sum(e - s for s, e in segments)
    print(
        f"📊 Input: {input_duration:.1f}s"
        + (
            f" (original: {input_duration * speedup:.1f}s at {speedup:.2f}x)"
            if speedup > 1.0
            else ""
        )
        + f", speaking segments: {len(segments)}"
    )

    # Step 2: Split long segments at word boundaries
    # Use max_cut_duration scaled to original pace for split decisions
    orig_max_duration = (
        max_cut_duration * speedup if speedup > 1.0 else max_cut_duration
    )
    sub_segments = _split_long_segments(segments, words, orig_max_duration)
    long_count = sum(1 for s, e in segments if e - s > orig_max_duration)
    zoom_split_count = sum(1 for _, _, is_cont in sub_segments if is_cont)
    print(
        f"✂️  {len(segments)} speaking segments → {len(sub_segments)} sub-segments "
        f"({zoom_split_count} zoom splits, {len(sub_segments) - len(segments)} dead cuts)"
    )

    if not sub_segments:
        print("No segments found, copying video as-is")
        subprocess.run(["cp", str(input_video), str(output_video)], check=True)
        return output_video

    # Scale timestamps to the actual video timeline
    if speedup > 1.0:
        sub_segments = [
            (s / speedup, e / speedup, is_cont) for s, e, is_cont in sub_segments
        ]

    # Step 2b: Suppress zoom toggles near content-cut boundaries.
    # When apply_cuts stitches segments from different source timestamps,
    # it creates visual discontinuities. If a zoom split lands near one
    # of these, the viewer sees a "double jump" (content change + zoom
    # change in quick succession). Suppress the zoom toggle so the content
    # cut is the only visual break.
    BOUNDARY_TOLERANCE = 1.0  # seconds
    if cut_boundaries:
        suppressed = 0
        for i, (start, end, is_cont) in enumerate(sub_segments):
            if not is_cont:
                continue
            for boundary in cut_boundaries:
                if abs(start - boundary) < BOUNDARY_TOLERANCE:
                    sub_segments[i] = (start, end, False)
                    suppressed += 1
                    break
        if suppressed:
            print(
                f"🔇 Suppressed {suppressed} zoom toggle(s) near content-cut boundaries"
            )

    # Step 2c: Internal boundary padding — don't cut exactly at word boundaries.
    # Extend each segment end/start by a few ms into the gap so we don't clip
    # word tails or onsets (Deepgram timestamps are often tight to the vowel).
    #
    # Use smaller padding at zoom-split boundaries (within continuous speech,
    # where the gap is tiny) and larger padding at dead-space boundaries.
    ZOOM_SPLIT_PAD = 0.04  # seconds — zoom splits within continuous speech
    DEAD_SPACE_PAD = 0.12  # seconds — dead-space removal boundaries
    padded: list[tuple[float, float, bool]] = []
    for i, (s, e, is_cont) in enumerate(sub_segments):
        start, end = s, e
        if i > 0:
            # Choose pad based on whether THIS segment is a zoom continuation
            # (small gap) or a new speaking segment (dead-space boundary)
            pad_size = ZOOM_SPLIT_PAD if is_cont else DEAD_SPACE_PAD
            gap = s - sub_segments[i - 1][1]
            pad = min(pad_size, gap / 2.0 - 0.005) if gap > 0.01 else 0.0
            start = s - pad
        if i < len(sub_segments) - 1:
            next_is_cont = sub_segments[i + 1][2]
            pad_size = ZOOM_SPLIT_PAD if next_is_cont else DEAD_SPACE_PAD
            gap = sub_segments[i + 1][0] - e
            pad = min(pad_size, gap / 2.0 - 0.005) if gap > 0.01 else 0.0
            end = e + pad
        padded.append((start, end, is_cont))
    sub_segments = padded

    # Step 3: Calculate zoom crop params
    # V1: uniform crop biased toward top of frame (where face usually is)
    crop_w = int(width / zoom_factor)
    crop_h = int(height / zoom_factor)
    # Ensure even dimensions (required by h264)
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2
    x_offset = (width - crop_w) // 2  # Centered horizontally
    # Top-half bias: 25% from top of the available margin
    y_offset = int((height - crop_h) * 0.25)

    print(
        f"🔍 Zoom: {zoom_factor}x → crop {crop_w}x{crop_h} "
        f"at ({x_offset},{y_offset}) from {width}x{height}"
    )

    # Step 4: Build ffmpeg filter complex
    # Zoom only toggles on "continuation" splits (long-segment splits),
    # NOT on dead space removal boundaries. Dead space cuts should be
    # invisible — no zoom change, just silence removed.
    video_parts = []
    audio_parts = []
    zoom_count = 0
    is_zoomed = False  # Start with wide shot

    for i, (start, end, is_continuation) in enumerate(sub_segments):
        if is_continuation:
            is_zoomed = not is_zoomed  # Toggle zoom only at split boundaries

        # Use fixed-point notation to avoid scientific notation (e.g. 2e-05)
        # which ffmpeg's trim filter cannot parse.
        s = f"{start:.6f}"
        e = f"{end:.6f}"

        if is_zoomed:
            # Zoomed: crop + scale back to original resolution (scale gives SAR 1:1)
            video_parts.append(
                f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS,"
                f"crop={crop_w}:{crop_h}:{x_offset}:{y_offset},"
                f"scale={width}:{height}:flags=lanczos,setsar=1:1[v{i}]"
            )
            zoom_count += 1
        else:
            # Wide: original frame; normalize SAR so concat inputs match (some sources use 399:400)
            video_parts.append(
                f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS,setsar=1:1[v{i}]"
            )

        audio_parts.append(f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[a{i}]")

    # Concat all segments
    n = len(sub_segments)
    v_concat = "".join(f"[v{i}]" for i in range(n))
    a_concat = "".join(f"[a{i}]" for i in range(n))
    concat_filter = (
        f"{v_concat}concat=n={n}:v=1:a=0[vout];" f"{a_concat}concat=n={n}:v=0:a=1[aout]"
    )

    filter_complex = ";".join(video_parts + audio_parts) + ";" + concat_filter

    def run_ffmpeg(use_videotoolbox: bool) -> subprocess.CompletedProcess:
        if use_videotoolbox:
            cmd = [
                "ffmpeg",
                "-y",
                "-hwaccel",
                "videotoolbox",
                "-i",
                str(input_video),
                "-filter_complex",
                filter_complex,
                "-map",
                "[vout]",
                "-map",
                "[aout]",
                "-c:v",
                "h264_videotoolbox",
                "-b:v",
                "8M",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(output_video),
            ]
        else:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(input_video),
                "-filter_complex",
                filter_complex,
                "-map",
                "[vout]",
                "-map",
                "[aout]",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(output_video),
            ]
        return subprocess.run(cmd, capture_output=True, text=True)

    print(
        f"🎬 Encoding {n} segments ({zoom_count} zoomed, " f"{n - zoom_count} wide)..."
    )
    result = run_ffmpeg(use_videotoolbox=True)
    if result.returncode != 0:
        print("  (videotoolbox failed, retrying with libx264...)", file=sys.stderr)
        result = run_ffmpeg(use_videotoolbox=False)
    if result.returncode != 0:
        print(result.stderr or result.stdout or "ffmpeg failed", file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, result.args)

    output_duration = _get_video_duration(output_video)
    removed = input_duration - output_duration
    print(
        f"✅ Output: {output_video.name} ({output_duration:.1f}s"
        + (f", removed {removed:.1f}s" if removed > 0.1 else "")
        + ")"
    )

    return output_video


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove dead space + apply zoom-based jump cuts"
    )
    parser.add_argument("input", type=Path, help="Input video file")
    parser.add_argument("output", type=Path, help="Output video file")
    parser.add_argument(
        "--transcript",
        type=Path,
        help="Word-level transcript JSON (if not provided, will transcribe)",
    )
    parser.add_argument(
        "--max-cut-duration",
        type=float,
        default=5.0,
        help="Max seconds before introducing a zoom cut (default: 5.0)",
    )
    parser.add_argument(
        "--zoom-factor",
        type=float,
        default=1.2,
        help="Zoom multiplier for tight shots (default: 1.2, i.e. 20%% zoom)",
    )
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=0.4,
        help="Max gap between words in seconds (default: 0.4)",
    )
    parser.add_argument(
        "--speedup",
        type=float,
        default=1.0,
        help="Speedup already applied to the input video (default: 1.0). "
        "When > 1.0, --transcript MUST be from the ORIGINAL (pre-speedup) "
        "video so gap analysis uses real pauses. Timestamps are auto-scaled.",
    )
    parser.add_argument(
        "--boundaries-from",
        type=Path,
        default=None,
        help="Path to .boundaries.json file written by apply_cuts.py. "
        "Contains join timestamps in the CUT video timeline. These are "
        "auto-scaled by 1/speedup to match the sped-up input. Zoom toggles "
        "are suppressed near these points to avoid double jumps.",
    )
    parser.add_argument(
        "--cut-boundaries",
        type=str,
        default=None,
        help="(Manual override) Comma-separated timestamps in the INPUT video "
        "timeline where content cuts create visual discontinuities. "
        "Prefer --boundaries-from for automatic handling.",
    )
    parser.add_argument(
        "--remove-fillers",
        action="store_true",
        default=False,
        help="Remove filler words (uh, um, mhmm, etc.) from the word list "
        "before building speaking segments. Requires a transcript generated "
        "with Deepgram's filler_words=True.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input video not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.speedup > 1.0 and not args.transcript:
        print(
            "Error: When --speedup is set, --transcript must be provided "
            "(from the ORIGINAL pre-speedup video).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load or generate transcript
    transcript_is_external = bool(args.transcript)
    if args.transcript:
        with open(args.transcript) as f:
            words = json.load(f)
        print(f"📝 Loaded {len(words)} words from {args.transcript}")
    else:
        print("📝 Transcribing video...")
        from .get_transcript import get_transcript

        result = asyncio.run(get_transcript(args.input))
        words = result["words"]
        print(f"📝 Transcribed {len(words)} words")

    # Load cut boundaries: --boundaries-from (explicit) takes priority,
    # --cut-boundaries (manual) is next, auto-discovery is the fallback.
    # Auto-discovery: look for *.boundaries.json in the input video's directory.
    boundaries = None
    source_ranges: list[dict] | None = None

    if not args.boundaries_from and not args.cut_boundaries:
        # Auto-discover boundaries file in the same directory
        boundary_files = list(args.input.parent.glob("*.boundaries.json"))
        if len(boundary_files) == 1:
            args.boundaries_from = boundary_files[0]
        elif len(boundary_files) > 1:
            print(
                f"⚠️  Found {len(boundary_files)} boundary files — pass "
                f"--boundaries-from explicitly to disambiguate",
                file=sys.stderr,
            )

    if args.boundaries_from:
        if not args.boundaries_from.exists():
            print(
                f"Warning: Boundaries file not found: {args.boundaries_from}",
                file=sys.stderr,
            )
        else:
            with open(args.boundaries_from) as f:
                sidecar_data: Any = json.load(f)

            # Support both old format (plain list) and new format (dict with
            # boundaries + source_ranges).
            if isinstance(sidecar_data, list):
                raw_boundaries = sidecar_data
            else:
                raw_boundaries = sidecar_data.get("boundaries", [])
                source_ranges = sidecar_data.get("source_ranges")

            # Boundaries are in the CUT video timeline (pre-speedup).
            # Scale to the sped-up input timeline.
            if args.speedup > 1.0:
                boundaries = [round(b / args.speedup, 4) for b in raw_boundaries]
                print(
                    f"📍 {len(boundaries)} boundaries loaded from {args.boundaries_from.name} "
                    f"(scaled by 1/{args.speedup:.4f})"
                )
            else:
                boundaries = raw_boundaries
                print(
                    f"📍 {len(boundaries)} boundaries loaded from {args.boundaries_from.name}"
                )
            if source_ranges:
                if transcript_is_external:
                    print(
                        f"📍 {len(source_ranges)} source ranges found — "
                        f"will remap external transcript to cut timeline"
                    )
                else:
                    print(
                        f"📍 {len(source_ranges)} source ranges found (ignored — "
                        f"using auto-transcribed timestamps)"
                    )
    elif args.cut_boundaries:
        boundaries = [float(t) for t in args.cut_boundaries.split(",")]
        print(f"📍 {len(boundaries)} content-cut boundaries provided (manual)")

    # If an external transcript was provided AND source_ranges are available,
    # remap the words to the cut video's timeline. This is needed when the
    # transcript covers the entire source video (e.g. a 37-min interview) but
    # the input video is a short cut from specific ranges within it.
    # Skip remapping for auto-transcribed audio — those timestamps already
    # match the input video.
    if source_ranges and transcript_is_external:
        original_count = len(words)
        words = _remap_words_to_cut_timeline(words, source_ranges)
        print(
            f"📝 Remapped transcript: {original_count} → {len(words)} words "
            f"(filtered to {len(source_ranges)} source ranges)"
        )

    apply_jump_cuts(
        args.input,
        args.output,
        words,
        max_cut_duration=args.max_cut_duration,
        zoom_factor=args.zoom_factor,
        gap_threshold=args.gap_threshold,
        speedup=args.speedup,
        cut_boundaries=boundaries,
        remove_fillers=args.remove_fillers,
    )


if __name__ == "__main__":
    main()
