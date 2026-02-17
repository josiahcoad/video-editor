#!/usr/bin/env python3
"""
Propose cuts for repurposing a long-form video into short-form segments.

Takes a word-level transcript and uses an LLM to identify coherent segments
with precise timestamp ranges. Outputs a JSON plan that can be fed to apply_cuts.py.

Usage:
  propose_cuts.py --transcript <words.json>
  propose_cuts.py --video <video.mp4>
  propose_cuts.py --transcript <words.json> --duration 45 --count 3
  propose_cuts.py --transcript <words.json> --model anthropic/claude-4-opus

Output (stdout JSON):
  [
    {
      "segment": 1,
      "summary": "...",
      "cuts": "0.8:7.06,27.54:43.81",
      "duration": 40.8
    },
    ...
  ]
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .get_transcript import get_transcript, read_word_transcript_file
from .settings_loader import load_settings


# ── EDL utilities ─────────────────────────────────────────────────────────────


def seconds_to_timecode(seconds: float, fps: int = 30) -> str:
    """Convert seconds to SMPTE timecode HH:MM:SS:FF."""
    total_frames = round(seconds * fps)
    ff = total_frames % fps
    total_secs = total_frames // fps
    ss = total_secs % 60
    mm = (total_secs // 60) % 60
    hh = total_secs // 3600
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def write_edl(
    path: Path,
    title: str,
    sections: list[dict],
    source_name: str = "AX",
    fps: int = 30,
) -> None:
    """Write a CMX 3600 EDL file from a list of time sections.

    Args:
        path: Output .edl file path
        title: EDL title (shown in NLE on import)
        sections: List of dicts with 'start' and 'end' keys (seconds)
        source_name: Reel / clip name for the source media
        fps: Frame rate for timecode conversion
    """
    lines = [
        f"TITLE: {title}",
        "FCM: NON DROP FRAME",
        "",
    ]
    record_offset = 0.0
    for i, section in enumerate(sections, 1):
        src_in = seconds_to_timecode(section["start"], fps)
        src_out = seconds_to_timecode(section["end"], fps)
        dur = section["end"] - section["start"]
        rec_in = seconds_to_timecode(record_offset, fps)
        rec_out = seconds_to_timecode(record_offset + dur, fps)
        record_offset += dur
        lines.append(
            f"{i:03d}  {'AX':<8s} V     C        "
            f"{src_in}  {src_out}  {rec_in}  {rec_out}"
        )
        if source_name != "AX":
            lines.append(f"* FROM CLIP NAME: {source_name}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


# ── Transcript formatting ─────────────────────────────────────────────────────


def _slim_words(words: list[dict]) -> list[dict]:
    """Strip word dicts to only the fields the LLM needs."""
    return [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in words]


def format_transcript_for_llm(words: list[dict]) -> str:
    """Format a word-level transcript as timestamped running text.

    Instead of dumping raw JSON (huge, noisy, unnatural for an LLM to read),
    we produce a human-readable format with inline timestamp markers every ~10
    words. This plays to the LLM's strength — reading natural language — while
    still providing precise timestamps for cut decisions.

    Example output:
        @59.39 there's a gap between people knowing you exist and actually
        wanting to buy from you here's how to close it @64.61
        --- gap 18.7s ---
        @83.27 just because we can do a lot of cool stuff with ai i think
        we should be doing asking ourselves that question @90.22
    """
    if not words:
        return ""

    lines: list[str] = []
    current_line: list[str] = []
    last_end: float = 0.0
    words_since_ts = 0

    for i, w in enumerate(words):
        # Detect silence gaps > 2s — these are meaningful boundaries
        if i > 0 and w["start"] - last_end > 2.0:
            # Flush current line
            if current_line:
                lines.append(" ".join(current_line) + f" @{last_end:.2f}")
                current_line = []
                words_since_ts = 0
            gap = w["start"] - last_end
            lines.append(f"--- gap {gap:.1f}s ---")

        # Insert timestamp marker at start and every ~10 words
        if words_since_ts == 0:
            current_line.append(f'@{w["start"]:.2f}')

        current_line.append(w["word"])
        words_since_ts += 1
        last_end = w["end"]

        # Newline + fresh timestamp every 10 words
        if words_since_ts >= 10:
            lines.append(" ".join(current_line) + f" @{w['end']:.2f}")
            current_line = []
            words_since_ts = 0

    # Flush remaining
    if current_line:
        lines.append(" ".join(current_line) + f" @{last_end:.2f}")

    return "\n".join(lines)


def verify_quoted_words(segments: list[dict], words: list[dict]) -> list[str]:
    """Verify that quoted first_words/last_words match the actual transcript.

    Returns a list of warnings for mismatches.
    """
    warnings: list[str] = []
    for seg in segments:
        seg_num = seg.get("segment", "?")
        cuts_str = seg.get("cuts", "")
        first_words_list = seg.get("_first_words", [])
        last_words_list = seg.get("_last_words", [])

        for idx, part in enumerate(cuts_str.split(",")):
            if not part.strip():
                continue
            s_str, e_str = part.split(":")
            start, end = float(s_str), float(e_str)
            section_words = _words_in_range(words, start, end)
            if not section_words:
                warnings.append(
                    f"Seg {seg_num} section {idx + 1}: NO WORDS found "
                    f"between {start:.2f}s and {end:.2f}s"
                )
                continue

            actual_text = " ".join(w["word"] for w in section_words)
            actual_first_5 = " ".join(w["word"] for w in section_words[:5]).lower()
            actual_last_5 = " ".join(w["word"] for w in section_words[-5:]).lower()

            # Check first_words
            if idx < len(first_words_list) and first_words_list[idx]:
                quoted = first_words_list[idx].lower().strip()
                if quoted and quoted not in actual_first_5:
                    warnings.append(
                        f"Seg {seg_num} section {idx + 1}: "
                        f'first_words mismatch — LLM quoted "{first_words_list[idx]}" '
                        f'but actual start is "{actual_first_5}"'
                    )

            # Check last_words
            if idx < len(last_words_list) and last_words_list[idx]:
                quoted = last_words_list[idx].lower().strip()
                if quoted and quoted not in actual_last_5:
                    warnings.append(
                        f"Seg {seg_num} section {idx + 1}: "
                        f'last_words mismatch — LLM quoted "{last_words_list[idx]}" '
                        f'but actual end is "{actual_last_5}"'
                    )

    return warnings


# ── Pydantic models for structured LLM output ────────────────────────────────


class ContentAnalysis(BaseModel):
    """Pre-cut analysis of the source content — forces the LLM to think before cutting."""

    content_type: str = Field(
        description=(
            "One of: procedural (step-by-step tutorial/guide), "
            "multi_topic (distinct ideas/opinions), interview (Q&A format), "
            "listicle (numbered tips/items), narrative (story with arc), "
            "opinion (single thesis argued throughout), "
            "pre_recorded_batch (multiple standalone clips recorded back-to-back)"
        )
    )
    editing_strategy: str = Field(
        description=(
            "Choose the editing approach based on the transcript structure:\n"
            "- 'highlight_reel': Single continuous recording (interview, talk, "
            "tutorial, vlog). Content flows as one piece. Extract the best moments, "
            "cold opens welcome, non-contiguous sections OK.\n"
            "- 'contiguous_batch': Multiple pre-recorded, self-contained clips "
            "recorded back-to-back in one session. Key signals: (1) the same topic "
            "is repeated 2-5 times with slight variations (RETAKES / false starts), "
            "(2) clear topic changes between groups, (3) each topic group has its "
            "own standalone hook/intro, (4) large silence gaps between groups, "
            "(5) topics are independent — they don't reference each other. "
            "For this strategy: pick the BEST TAKE per topic, NO cold opens, "
            "each segment = one standalone clip."
        ),
    )
    single_or_multi: str = Field(
        description=(
            "Either 'single' or 'multi'. Explain your reasoning in 1-2 sentences. "
            "Choose 'single' for: linear procedures, source under 5 min, "
            "topics too intertwined to separate. "
            "Choose 'multi' ONLY when 2+ topics can each pass the cold viewer test."
        )
    )
    cold_open_candidate: str | None = Field(
        default=None,
        description=(
            "If a standout soundbite from the middle/end of the video would hook "
            "viewers as a cold open, quote the exact words and provide the start/end "
            "timestamps. null if no compelling candidate exists or if the content is "
            "procedural/tutorial (cold opens don't suit how-to content). "
            "Always null for contiguous_batch strategy."
        ),
    )
    keepable_content_seconds: float = Field(
        description=(
            "Estimated total duration of content worth keeping (excluding filler, "
            "tangents, verbose explanations). This helps calibrate how many segments "
            "to create."
        )
    )


class WordRange(BaseModel):
    """A contiguous section of the video to keep."""

    start: float = Field(description="Start timestamp in seconds")
    end: float = Field(description="End timestamp in seconds")
    first_words: str = Field(
        description=(
            "The first 3-5 words of this section, copied EXACTLY from the "
            "transcript. This is your verification that you're cutting at the "
            "right start point."
        )
    )
    last_words: str = Field(
        description=(
            "The last 3-5 words of this section, copied EXACTLY from the "
            "transcript. This is your verification that you're cutting at the "
            "right end point — check that the phrase is COMPLETE."
        )
    )


class SegmentPlan(BaseModel):
    """A single short-form segment plan."""

    segment_number: int = Field(description="Segment number (1, 2, 3, etc.)")
    cold_open_sections: list[WordRange] = Field(
        default_factory=list,
        description=(
            "If a cold open was identified in the analysis, provide the 1-2 sections "
            "(3-8 seconds total) from later in the video that will be placed FIRST "
            "as a teaser before the main chronological flow. "
            "Leave empty ([]) if no cold open. These sections are from LATER "
            "timestamps and are placed before the intro."
        ),
    )
    sections: list[WordRange] = Field(
        description=(
            "Main body sections in CHRONOLOGICAL order. This is the intro → body → "
            "payoff flow. Do NOT include the cold open here — it goes in "
            "cold_open_sections above."
        ),
    )
    summary: str = Field(description="Brief summary of what this segment covers")
    hook_description: str = Field(
        description=(
            "What makes the opening 1–3 seconds grab attention? Prefer describing "
            "as: stakes/tension (e.g. 'fatal mistake' style) or named concept / "
            "curiosity (e.g. 'adjust the access, not the affection')."
        )
    )


class RepurposePlan(BaseModel):
    """Full plan for repurposing a video into segments."""

    analysis: ContentAnalysis = Field(
        description="Pre-cut analysis — complete this BEFORE proposing segments"
    )
    segments: list[SegmentPlan] = Field(description="List of segment plans")
    overall_summary: str = Field(
        description="Overall summary of how the video was repurposed"
    )


# ── Critique / Refine models ─────────────────────────────────────────────────


class SegmentCritique(BaseModel):
    """Critique of a single proposed segment."""

    segment_number: int = Field(description="Which segment this critique is about")
    verdict: str = Field(
        description=(
            "One of: 'keep' (no issues), 'fix' (fixable issues), "
            "'drop' (should be removed/merged), 'split' (should be split)"
        )
    )
    issues: list[str] = Field(
        default_factory=list,
        description=(
            "Specific issues found. Be concrete: quote the problematic words, "
            "give timestamps, name the alternative take."
        ),
    )
    suggested_fixes: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete fix instructions. E.g. 'Replace section 2 (374.96:380.34) "
            "with extended range 374.96:387.38 to include the word call at 386.88s'"
        ),
    )


class CritiquePlan(BaseModel):
    """Full critique of a proposed cut plan."""

    missing_topics: list[str] = Field(
        default_factory=list,
        description=(
            "Topics/content blocks present in the transcript but NOT covered "
            "by any proposed segment. Quote the first few words and give the "
            "approximate timestamp range."
        ),
    )
    segments_to_merge: list[str] = Field(
        default_factory=list,
        description=(
            "Pairs of segment numbers that cover the same topic and should be "
            "merged. E.g. 'Merge segments 5 and 6 — both about automation limits'"
        ),
    )
    segment_critiques: list[SegmentCritique] = Field(
        description="Per-segment critique, one entry per proposed segment"
    )
    overall_assessment: str = Field(
        description=(
            "One of: 'pass' (minor or no issues — proceed to production), "
            "'needs_revision' (has fixable issues — run refinement), "
            "'major_rework' (fundamental problems — re-run from scratch)"
        )
    )
    assessment_summary: str = Field(
        description="2-3 sentence summary of the overall quality and key issues"
    )


class RefinedPlan(BaseModel):
    """Refined plan after applying critique feedback."""

    segments: list[SegmentPlan] = Field(description="Revised list of segment plans")
    changes_made: list[str] = Field(
        description=(
            "List of changes made in response to the critique. Be specific: "
            "'Added missing trust gap segment at 59-65s', "
            "'Extended section 2 to include dropped word call at 386.88s', "
            "'Merged segments 5 and 6 into one automation segment'"
        )
    )
    unaddressed_critiques: list[str] = Field(
        default_factory=list,
        description="Any critique items that could NOT be addressed, with explanation",
    )


# ── Cut boundary verification ─────────────────────────────────────────────────

# Words that should never be the first word of a segment (they continue a thought)
_CONTINUATION_WORDS = {
    "and",
    "but",
    "or",
    "so",
    "because",
    "then",
    "also",
    "plus",
    "yet",
    "nor",
    "which",
    "that",
    "who",
    "whom",
    "whose",
    "where",
    "when",
    "while",
    "although",
    "though",
    "however",
    "therefore",
    "furthermore",
    "moreover",
    "additionally",
    "meanwhile",
    "consequently",
    "hence",
    "thus",
    "nonetheless",
    "nevertheless",
}

# Words that should never be the last word of a section (they start the next thought)
_TRAILING_CONJUNCTIONS = {"and", "but", "or", "so", "because", "then", "like"}

# Minimum word count for duplicate phrase detection (same idea said twice in one segment)
_DUPLICATE_PHRASE_MIN_WORDS = 6


def _find_duplicate_phrases_in_segment(
    section_texts: list[str], min_words: int = _DUPLICATE_PHRASE_MIN_WORDS
) -> list[tuple[str, int, int]]:
    """If the same substantial phrase appears in two different sections, return it.

    Returns list of (phrase, section_i, section_j) where phrase appears in both
    section_i and section_j (i < j). Used to catch e.g. hook and body both
    saying 'everyone thinks you need a perfect credit score to buy a house'.
    """
    # phrase -> set of section indices where it appears
    phrase_to_sections: dict[str, set[int]] = {}
    for sec_idx, text in enumerate(section_texts):
        normalized = re.sub(r"[^\w\s]", "", text.lower())
        words = normalized.split()
        for n in range(min_words, min(12, len(words)) + 1):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i : i + n])
                phrase_to_sections.setdefault(phrase, set()).add(sec_idx)
    duplicates: list[tuple[str, int, int]] = []
    for phrase, sec_set in phrase_to_sections.items():
        if len(sec_set) >= 2:
            sec_list = sorted(sec_set)
            duplicates.append((phrase, sec_list[0], sec_list[1]))
    return duplicates


def _words_in_range(
    words: list[dict], start: float, end: float, tolerance: float = 0.05
) -> list[dict]:
    """Get all words whose start falls within [start - tolerance, end + tolerance]."""
    return [
        w
        for w in words
        if w["start"] >= start - tolerance and w["start"] <= end + tolerance
    ]


def _find_sentence_start_before(
    words: list[dict], timestamp: float, max_lookback: float = 5.0
) -> float | None:
    """Look backward from timestamp to find the start of the current sentence.

    Heuristic: walk backward until we find a word that ends a sentence
    (punctuation: . ? ! ;) or a gap > 1.5s, then return the next word's start.
    """
    # Get words in the lookback window
    candidates = [
        w
        for w in words
        if w["start"] >= timestamp - max_lookback and w["start"] < timestamp
    ]
    if not candidates:
        return None

    # Walk backward from timestamp
    for w in reversed(candidates):
        word_text = w.get("punctuated_word", w["word"])
        # Check if this word ends a sentence
        if word_text and word_text[-1] in ".?!;":
            # The sentence starts AFTER this word — find the next word
            next_words = [nw for nw in words if nw["start"] > w["end"]]
            if next_words:
                return next_words[0]["start"]
            return None
        # Check for a large gap (likely a topic/sentence break)
        prev_words = [pw for pw in candidates if pw["end"] <= w["start"]]
        if prev_words:
            gap = w["start"] - prev_words[-1]["end"]
            if gap > 1.5:
                return w["start"]

    return None


def _find_sentence_end_after(
    words: list[dict], timestamp: float, max_lookahead: float = 3.0
) -> float | None:
    """Look forward from timestamp to find the end of the current sentence."""
    candidates = [
        w
        for w in words
        if w["start"] > timestamp and w["start"] <= timestamp + max_lookahead
    ]
    for w in candidates:
        word_text = w.get("punctuated_word", w["word"])
        if word_text and word_text[-1] in ".?!;":
            return w["end"]
    return None


def verify_cut_boundaries(
    segments: list[dict], words: list[dict]
) -> tuple[list[dict], list[str]]:
    """Verify and auto-fix cut boundaries for sentence alignment.

    For each segment, checks:
    1. First section doesn't start mid-sentence
    2. Sections don't end on trailing conjunctions
    3. All section text is logged for human review

    Returns:
        (fixed_segments, warnings): The corrected segments and a list of warnings.
    """
    warnings: list[str] = []
    fixed_segments = []

    for seg in segments:
        cuts_str = seg.get("cuts", "")
        if not cuts_str:
            fixed_segments.append(seg)
            continue

        sections = []
        for part in cuts_str.split(","):
            s_str, e_str = part.split(":")
            sections.append({"start": float(s_str), "end": float(e_str)})

        fixed_sections = []
        seg_num = seg.get("segment", "?")

        for idx, section in enumerate(sections):
            start, end = section["start"], section["end"]
            section_words = _words_in_range(words, start, end)
            text = " ".join(w["word"] for w in section_words)

            # ── Check 1: NO section (including body after hook) may start with a continuation word ──
            if section_words:
                first_word = section_words[0]["word"].lower().strip()
                if first_word in _CONTINUATION_WORDS:
                    if idx == 0:
                        # First section: try to extend backward to sentence start
                        new_start = _find_sentence_start_before(words, start)
                        if new_start is not None and new_start < start:
                            old_text_preview = " ".join(
                                w["word"] for w in section_words[:5]
                            )
                            new_words = _words_in_range(words, new_start, end)
                            new_text_preview = " ".join(
                                w["word"] for w in new_words[:8]
                            )
                            warnings.append(
                                f"Seg {seg_num} section {idx + 1}: starts with "
                                f"continuation word '{first_word}' "
                                f'("{old_text_preview}..."). '
                                f"Extended back to {new_start:.2f}s "
                                f'("{new_text_preview}...")'
                            )
                            start = new_start
                        else:
                            warnings.append(
                                f"Seg {seg_num} section {idx + 1}: starts with "
                                f"continuation word '{first_word}' "
                                f"(\"{' '.join(w['word'] for w in section_words[:5])}...\") "
                                f"but could not find sentence start — MANUAL FIX NEEDED"
                            )
                    else:
                        # Later section (e.g. body after hook): advance start past continuation word(s)
                        advance_to = None
                        for i, w in enumerate(section_words):
                            if w["word"].lower().strip() not in _CONTINUATION_WORDS:
                                advance_to = w["start"]
                                break
                        if advance_to is not None and advance_to > start:
                            old_preview = " ".join(
                                w["word"] for w in section_words[:5]
                            )
                            new_section_words = _words_in_range(
                                words, advance_to, end
                            )
                            new_preview = " ".join(
                                w["word"] for w in new_section_words[:5]
                            )
                            warnings.append(
                                f"Seg {seg_num} section {idx + 1}: starts with "
                                f"continuation word '{first_word}' "
                                f'("{old_preview}..."). '
                                f"Advanced start to {advance_to:.2f}s "
                                f'("{new_preview}...")'
                            )
                            start = advance_to
                        else:
                            warnings.append(
                                f"Seg {seg_num} section {idx + 1}: starts with "
                                f"continuation word '{first_word}' "
                                f"(\"{' '.join(w['word'] for w in section_words[:5])}...\") "
                                f"— no non-continuation word in section; MANUAL FIX NEEDED"
                            )

            # ── Check 2: Section shouldn't end on a trailing conjunction ──
            section_words_now = _words_in_range(words, start, end)
            if section_words_now:
                last_word = section_words_now[-1]["word"].lower().strip()
                if last_word in _TRAILING_CONJUNCTIONS:
                    # Trim the end to before this word
                    new_end = section_words_now[-1]["start"] - 0.01
                    if new_end > start:
                        warnings.append(
                            f"Seg {seg_num} section {idx + 1}: ends with "
                            f"trailing conjunction '{last_word}'. "
                            f"Trimmed end from {end:.2f}s to {new_end:.2f}s"
                        )
                        end = new_end

            fixed_sections.append({"start": round(start, 2), "end": round(end, 2)})

        # ── Check 3: No duplicate substantial phrase across sections (e.g. hook + body same line) ──
        section_texts = []
        for s in fixed_sections:
            ws = _words_in_range(words, s["start"], s["end"])
            section_texts.append(" ".join(w["word"] for w in ws))
        dupes = _find_duplicate_phrases_in_segment(section_texts)
        for phrase, sec_a, sec_b in dupes:
            warnings.append(
                f"Seg {seg_num}: DUPLICATE PHRASE in section {sec_a + 1} and section {sec_b + 1}: "
                f'"{phrase[:60]}{"..." if len(phrase) > 60 else ""}". '
                f"Trim the later section start so the same line is not said twice — MANUAL FIX NEEDED"
            )

        # Rebuild cuts string
        new_cuts = ",".join(f"{s['start']:.2f}:{s['end']:.2f}" for s in fixed_sections)
        new_dur = sum(s["end"] - s["start"] for s in fixed_sections)

        fixed_seg = dict(seg)
        fixed_seg["cuts"] = new_cuts
        fixed_seg["duration"] = round(new_dur, 1)
        fixed_segments.append(fixed_seg)

    return fixed_segments, warnings


def print_cut_preview(segments: list[dict], words: list[dict]) -> None:
    """Print human-readable text preview of all proposed cuts to stderr.

    This is the programmatic equivalent of Phase 2b in repurpose.md —
    ensures the operator can see exactly what text each cut contains.
    """
    print("\n" + "─" * 60, file=sys.stderr)
    print("CUT TEXT PREVIEW (verify before proceeding)", file=sys.stderr)
    print("─" * 60, file=sys.stderr)

    for seg in segments:
        cuts_str = seg.get("cuts", "")
        if not cuts_str:
            continue

        seg_num = seg.get("segment", "?")
        print(f"\n  Segment {seg_num}: {seg.get('summary', '')}", file=sys.stderr)

        for i, part in enumerate(cuts_str.split(",")):
            s_str, e_str = part.split(":")
            start, end = float(s_str), float(e_str)
            section_words = _words_in_range(words, start, end)
            text = " ".join(w["word"] for w in section_words)
            dur = end - start

            is_cold = seg.get("has_cold_open") and i == 0
            label = "COLD OPEN" if is_cold else f"Section {i + 1}"

            print(
                f"    [{label}] {start:.2f}s → {end:.2f}s ({dur:.1f}s)",
                file=sys.stderr,
            )
            # Truncate very long sections for readability
            if len(text) > 200:
                print(
                    f'    "{text[:100]} ... {text[-80:]}"',
                    file=sys.stderr,
                )
            else:
                print(f'    "{text}"', file=sys.stderr)

    print("\n" + "─" * 60, file=sys.stderr)


# ── Core logic ────────────────────────────────────────────────────────────────


DEFAULT_MODEL = "google/gemini-3-flash-preview"


# ── System prompt ─────────────────────────────────────────────────────────────


def _build_system_prompt(
    target_duration: float,
    min_duration: float,
    max_duration: float,
    total_duration: float,
    max_segments: int | None,
    custom_prompt: str | None,
) -> str:
    """Build the system prompt for the repurposing LLM."""

    if max_segments:
        segment_instruction = (
            f"Extract exactly {max_segments} segment(s). "
            "(For contiguous_batch: this is a hint — if you detect a different "
            "number of distinct topics, use the actual count instead.)"
        )
    else:
        segment_instruction = (
            "Determine the right number of segments based on your content analysis. "
            "For highlight_reel: often that number is 1. Prioritize QUALITY over QUANTITY. "
            "For contiguous_batch: one segment per distinct topic detected."
        )

    custom_section = ""
    if custom_prompt:
        custom_section = f"\n\nADDITIONAL EDIT DIRECTION:\n{custom_prompt}\n"

    return (
        "You are a professional short-form video editor who creates viral TikTok / "
        "Reels / Shorts from long-form video transcripts.\n\n"
        #
        # ── STEP 1: ANALYZE ──────────────────────────────────────────────
        "## STEP 1 — ANALYZE THE CONTENT (do this BEFORE proposing cuts)\n\n"
        "Fill in the `analysis` fields first. This forces you to think about "
        "WHAT you're working with before deciding HOW to cut it.\n\n"
        "### Content type classification\n"
        "- **procedural**: Step-by-step tutorial, setup guide, recipe, walkthrough. "
        "Value is in the linear sequence.\n"
        "- **multi_topic**: Video covers 2+ distinct ideas/opinions that can stand alone.\n"
        "- **interview**: Q&A or conversation with distinct question-answer pairs.\n"
        "- **listicle**: Numbered tips/items ('5 things I learned').\n"
        "- **narrative**: Story with a beginning, middle, and end.\n"
        "- **opinion**: Single thesis argued throughout.\n"
        "- **pre_recorded_batch**: Multiple standalone short-form clips recorded back-to-back "
        "in a single session. NOT a single continuous piece — it's a batch of separate clips "
        "that happen to be in one file.\n\n"
        "### Editing strategy detection\n"
        "Examine the transcript structure to determine the correct approach:\n\n"
        "**highlight_reel** — Choose when the video is a SINGLE continuous recording "
        "(interview, panel, tutorial, vlog, talk). The content flows as one piece. "
        "You will extract the best moments, use cold opens, create tight highlight "
        "segments with multiple sections each.\n\n"
        "**contiguous_batch** — Choose when the video contains MULTIPLE pre-recorded, "
        "self-contained short-form clips recorded back-to-back. Key signals:\n"
        "- The same topic/hook is repeated 2-5 times with slight variations "
        "(these are RETAKES / false starts)\n"
        "- Clear topic changes between groups (new hook, new subject, new intro)\n"
        "- Each topic group has its own standalone hook/intro\n"
        "- Large silence gaps (5-30+ seconds) between topic groups\n"
        "- Topics do NOT reference or depend on each other\n"
        "- The speaker often says 'okay' or pauses between takes\n"
        "- CTA/outros may be recorded separately after each topic\n\n"
        "**This detection is CRITICAL.** If you misclassify a batch as highlight_reel, "
        "you will produce Frankenstein edits that combine unrelated clips. If you "
        "misclassify a single piece as batch, you will miss opportunities for "
        "creative editing.\n\n"
        "### Single vs. multi-segment decision\n"
        "Default to SINGLE unless you have a strong reason for multi. Specifically:\n"
        "- **SINGLE** if: content is procedural, source is under 5 minutes, "
        "topics are intertwined, or splitting would create fragments that fail "
        "the cold viewer test.\n"
        "- **MULTI** only if: there are 2+ genuinely distinct, self-contained topics "
        "where each segment has its own hook, body, and payoff — AND a cold viewer "
        "would understand each segment with zero prior context.\n\n"
        f"Source duration: {total_duration:.0f}s. A {total_duration:.0f}s source "
        f"rarely supports more than {max(1, int(total_duration / 90))} segment(s).\n\n"
        "### Cold open candidate\n"
        "Look for a standout soundbite (surprise, strong opinion, emotional beat) "
        "from the middle or end that would hook viewers if placed first. "
        "NOT appropriate for procedural/tutorial content. Set to null if none exists.\n\n"
        "**If you identify a cold open candidate, you MUST use it** — see "
        "COLD OPEN PLACEMENT in Step 4 for how.\n\n"
        #
        # ── STEP 2: COLD VIEWER TEST ─────────────────────────────────────
        "## STEP 2 — THE COLD VIEWER TEST\n\n"
        "Every segment MUST pass this test:\n"
        "> Would someone who has NEVER seen this video, scrolling TikTok, "
        "understand and want to keep watching THIS segment from second 1?\n\n"
        "If a segment requires context from another segment to make sense, it is a "
        "FRAGMENT, not a short. Merge it or restructure.\n\n"
        "Common fragment signals:\n"
        "- Opens mid-sentence ('paid ads, social media ads, and organic' — "
        "viewer has no idea what about them)\n"
        "- Opens mid-procedure ('Then you need to...' with no setup)\n"
        "- References something explained earlier ('the thing we just set up')\n"
        "- Only makes sense as part 2/3 of a sequence\n"
        "- First words are a continuation of a thought, not the start of one\n\n"
        #
        # ── STEP 3: CUTTING PHILOSOPHY ───────────────────────────────────
        "## STEP 3 — CUTTING PHILOSOPHY\n\n"
        "### If editing_strategy is 'contiguous_batch':\n\n"
        "You are NOT creating a highlight reel. You are identifying and isolating "
        "individual pre-recorded clips from a batch recording session.\n\n"
        "**Procedural rules for contiguous_batch (follow these steps IN ORDER):**\n\n"
        "**Step A — SCAN.** Walk through the ENTIRE transcript chronologically, "
        "first word to last word. Identify every CONTENT BLOCK — a group of "
        "utterances on the same subject. Note the approximate timestamp range of "
        "each block. DO NOT stop scanning early — cover the full transcript.\n\n"
        "**Step B — GROUP into topics.** Cluster content blocks that cover the "
        "same subject:\n"
        "- Retakes: the same hook/idea repeated with slight wording variations → "
        "  these are takes of ONE topic\n"
        "- Body + example + CTA all about the same subject → ONE topic, even if "
        "  there are 5-15 second pauses between them\n"
        "- A hook + a supporting story/argument recorded right after → ONE topic\n"
        "- A completely different subject = a NEW topic\n"
        "- Speakers often pause 5-15 seconds between parts of the SAME recording "
        "  (thinking, checking notes). These pauses do NOT indicate a new topic.\n"
        "- A new topic is signaled by: a new intro/hook phrase, a change of subject, "
        "  or a very long gap (20+ seconds) followed by different content.\n\n"
        "**Step C — SELECT best take.** For each topic, pick the best version:\n"
        "  1. REJECT takes with verbal stumbles ('doing asking', 'started sitting "
        "     around', 'I believe that the'). These are clearly failed takes.\n"
        "  2. Prefer the most COMPLETE version (full hook + body + payoff).\n"
        "  3. Prefer the cleanest delivery (natural flow, confident landing).\n"
        "  4. If equally good, prefer the LAST take (speakers improve).\n"
        "  5. For body/example content that only has ONE version (no retake), "
        "     always include it — it's part of the topic.\n"
        "  6. Include the best CTA take if one was recorded within ~30s of the topic.\n\n"
        "**Step D — VERIFY completeness.** Before outputting, check:\n"
        "  - Have you accounted for content at every part of the transcript?\n"
        "  - Are there any content blocks you skipped? If so, include them.\n"
        "  - Does each topic have its full body, not just the hook?\n\n"
        "**Step E — OUTPUT.** One segment per topic, in chronological order.\n\n"
        "**CRITICAL CONSTRAINTS:**\n"
        "- `cold_open_sections` must be empty ([]) for EVERY segment.\n"
        "- **PROXIMITY RULE:** All sections within a single segment must be from "
        "timestamps within ~60s of each other. NEVER pull content from a distant "
        "part of the video.\n"
        "- **DO NOT SKIP TOPICS.** Even a 5-second clip is a valid segment. "
        "Include ALL detected topics.\n"
        "- **DO NOT over-split topics.** If a hook, body, example, and CTA are all "
        "about the same subject and recorded close together, they are ONE topic/segment.\n"
        "- A batch of 6-12 clips from a 10-15 minute merged recording is normal.\n\n"
        "Skip to STEP 4 (Section Rules) now — Steps 3b-3c below are for "
        "highlight_reel only.\n\n"
        "### If editing_strategy is 'highlight_reel':\n\n"
        "You are creating a HIGHLIGHT REEL, not a summary. Think like a trailer editor:\n\n"
        "**Pace matters more than completeness.** A tight 45s short that moves fast "
        "beats a 60s short that drags.\n\n"
        "**Cut aggressively.** Remove:\n"
        "- Verbose explanations and clarifications\n"
        "- Conditional branches ('if you purchased X...', 'you may or may not...')\n"
        "- Setup/transitions between steps ('now let's move on to...')\n"
        "- Pleasantries, self-corrections, 'um/uh' padding\n"
        "- Repetition (speaker says the same thing twice differently)\n"
        "- Long wind-down outros (15-20s 'thanks for watching, subscribe...')\n"
        "  BUT: Keep punchy ~3s CTAs ('follow me for more', 'follow me at [handle]'). "
        "They're short enough that viewers don't swipe and they drive engagement.\n\n"
        "**Setup-payoff consistency.** If the speaker promises to cover N things "
        "('there are two reasons', '3 tips'), you MUST either:\n"
        "- Deliver on ALL of them (even briefly), OR\n"
        "- Cut the promise itself so the viewer never hears '2 reasons' if you only "
        "cover one. A broken promise is worse than no promise.\n"
        "This also applies to IMPLICIT references: if a section says 'the second "
        "reason', a cold viewer infers they missed a first. Either include the "
        "enumeration setup ('there are two reasons') or don't reference ordinals.\n\n"
        "**Keep the action.** Keep:\n"
        "- The core 'do this' moments (for tutorials)\n"
        "- Key results and payoffs ('setup completed!', 'and now we're streaming')\n"
        "- The hook (opening statement that grabs)\n"
        "- Surprising or insightful moments\n"
        "- Concrete examples and numbers\n"
        "- The climax of the argument — the moment that makes the point land. "
        "If you identified a cold open, the content AROUND it (the build-up "
        "and the cold open's natural context) is almost always the strongest "
        "part of the video. Do NOT skip it.\n\n"
        "**Use MANY short sections for fast pacing.** Don't limit yourself to 2-4 "
        "sections per segment. Use 5-15 shorter sections to create a dynamic, "
        "jump-cut rhythm. This is expected in short-form video — rapid cuts between "
        "key moments keep viewers watching.\n\n"
        #
        # ── STEP 4: SECTION RULES ────────────────────────────────────────
        "## STEP 4 — SECTION RULES\n\n"
        "- **COLD OPEN (MANDATORY if cold_open_candidate is not null):**\n"
        "  Put the cold open soundbite in `cold_open_sections` (separate field). "
        "It should be 3-8 seconds — a complete, self-contained phrase that "
        "creates curiosity or surprise. The code will automatically prepend it "
        "before the main `sections`.\n"
        "- `sections` (main body) MUST be in chronological order\n"
        "- Use the EXACT timestamps from the transcript — do NOT round or adjust\n"
        "- Never cut mid-word. Each section should start at a word's `start` "
        "timestamp and end at a word's `end` timestamp\n"
        "- **CRITICAL — SEGMENT OPENER RULE:** The FIRST section of each segment "
        "(or the first section after a cold open) "
        "MUST begin at a complete, self-contained statement. A cold viewer's very "
        "first words must make grammatical sense without prior context. "
        "NEVER start a segment mid-sentence (e.g. 'paid ads, social media ads, "
        "and organic' when the full sentence was 'I wanted to talk about the "
        "difference between paid ads...'). If the best content starts mid-sentence, "
        "either include the full sentence lead-in OR find a different starting point "
        "that begins a new thought.\n"
        "- **EVERY section (including body after a hook):** No section may start "
        "with a continuation word (Also, So, And, But, Or, Then, However, etc.). "
        "If the body of a segment would start with 'Also, there are...', move the "
        "section start to the next word so it begins with 'There are...'. Same for "
        "any conjunction or transition that continues a prior sentence.\n"
        "- **NO DUPLICATE PHRASES:** The same sentence or near-identical line must "
        "NOT appear in two sections of the same segment (e.g. hook says 'Everybody "
        "thinks you need a perfect credit score to buy a house' and body then says "
        "'So a lot of people think you need a perfect credit score to buy a house'). "
        "If the body repeats the hook's idea, start the body AFTER the repeated "
        "phrase (e.g. at 'And even though that would be awesome...'). One clear "
        "statement of the idea per segment.\n"
        "- **OPENER QUALITY (first 1–3 seconds):** The opener must do two things: "
        "(1) give an idea what the clip is about, and (2) create unresolved tension "
        "or curiosity so the viewer wants to keep watching. Prefer starting a "
        "segment on a line that fits one of these patterns:\n"
        "  • **Stakes / tension:** e.g. 'Most leaders, they make a fatal mistake' — "
        "clear topic + unresolved tension (what mistake?).\n"
        "  • **Named concept / curiosity:** e.g. 'You got to adjust the access, "
        "not the affection' — a concept that begs explanation and pulls the viewer in.\n"
        "  If the natural start of a topic is weak (vague intro, filler), look for a "
        "stronger line within the same topic to start the segment, or flag that the "
        "opener could be stronger.\n"
        "- **GAP-SPANNING SENTENCES:** In natural speech, speakers often pause "
        "mid-sentence (e.g., 1-2s gap between 'When' and 'it comes to'). "
        "If a sentence has internal pauses, STILL include the full sentence from "
        "its first word. The downstream jump-cut pipeline removes gaps automatically "
        "— you do NOT need to avoid gaps.\n"
        "- **PHRASE COMPLETENESS:** Always include complete idiomatic phrases. "
        "Do NOT cut 'two jobs in one' to 'two jobs', or 'what I learned from' "
        "without the object. If a phrase has a natural completion, include it.\n"
        "- **SECTION CONSOLIDATION:** Do NOT create micro-sections under 1.5 seconds "
        "for consecutive words that are part of the same thought. If two sections "
        "are separated by a gap of less than 1 second, merge them into one section "
        "spanning the full range. The jump-cut pipeline removes gaps automatically.\n"
        "- Mid-segment sections (non-openers) do NOT need to be complete sentences — "
        "jump cuts between phrase fragments are fine in short-form\n"
        "- Sections can be as short as 1-3 seconds (a single phrase) but prefer "
        "3-10 second sections for cleaner editing\n\n"
        #
        # ── STEP 5: NARRATIVE COHERENCE ────────────────────────────────────
        "## STEP 5 — NARRATIVE COHERENCE CHECK\n\n"
        "Before finalizing, mentally play back the cold open + all sections in "
        "order. For each transition between sections, ask:\n"
        "- Does the next section follow logically from the previous one?\n"
        "- Would a cold viewer understand the jump, or is there missing context?\n"
        "- If the speaker builds to a climax (contrast, 'but imagine...', "
        "'here's the thing'), is the full build-up + payoff included?\n\n"
        "The build-up to a key insight is often as important as the insight "
        "itself. If the video says 'you have to do X... but imagine instead Y', "
        "keep BOTH the setup (X) and the payoff (Y). Cutting the setup robs "
        "the payoff of its impact.\n\n"
        #
        # ── STEP 5b: BOUNDARY VERIFICATION VIA first_words / last_words ───
        "## STEP 5b — BOUNDARY VERIFICATION (MANDATORY)\n\n"
        "For EVERY section you output, you MUST fill in `first_words` and "
        "`last_words`. These fields force you to verify your cuts:\n\n"
        "1. **`first_words`**: Copy the first 3-5 words EXACTLY from the "
        "transcript at your `start` timestamp. Read them aloud: do they "
        "form the beginning of a complete thought? If the first word is a "
        "conjunction or transition (and, but, so, also, because, then, "
        "which, that, however), you're cutting mid-sentence — for the "
        "first section back up the start; for later sections (e.g. body "
        "after a hook) advance the start to the next word so the section "
        "does not open with that word.\n\n"
        "2. **`last_words`**: Copy the last 3-5 words EXACTLY from the "
        "transcript at your `end` timestamp. Read them aloud: is the "
        "phrase COMPLETE? If the last word is a dangling conjunction or "
        "an incomplete phrase ('every sales' without 'call'), extend "
        "the end timestamp to capture the full phrase.\n\n"
        "**These fields are programmatically verified against the actual "
        "transcript. If they don't match, the section is flagged as broken. "
        "Be precise.**\n\n"
        #
        # ── STEP 6: DURATION ─────────────────────────────────────────────
        "## STEP 6 — DURATION RULES\n\n"
        "### For highlight_reel (STRICT):\n"
        f"- Target: {target_duration:.0f} seconds per segment\n"
        f"- Minimum: {min_duration:.0f} seconds — segments shorter than this are REJECTED\n"
        f"- Maximum: {max_duration:.0f} seconds\n"
        "- Before finalizing, add up ALL section durations (end - start for each). "
        "If the total is below the minimum, add more relevant sections.\n"
        "- If the total exceeds the maximum, cut more aggressively.\n\n"
        "### For contiguous_batch (RELAXED):\n"
        "- Duration constraints are RELAXED. Each segment is the natural length of "
        "the speaker's best take for that topic.\n"
        "- Do NOT pad to hit target duration. Do NOT aggressively trim to fit.\n"
        "- Report the actual duration. Very short segments (under 5s) should be "
        "noted — they may be standalone hooks.\n"
        "- Very long segments (over 90s including CTA) are fine if that's the "
        "natural length of the topic.\n\n"
        #
        # ── TASK ─────────────────────────────────────────────────────────
        f"## YOUR TASK\n\n"
        f"{segment_instruction}\n"
        f"{custom_section}"
    )


# ── Critique / Refine helpers ─────────────────────────────────────────────────


def _reconstruct_segment_text(segment: dict, words: list[dict]) -> str:
    """Reconstruct the would-be spoken text of a segment from its cuts."""
    cuts_str = segment.get("cuts", "")
    if not cuts_str:
        return ""
    lines = []
    for i, part in enumerate(cuts_str.split(",")):
        s_str, e_str = part.split(":")
        start, end = float(s_str), float(e_str)
        section_words = _words_in_range(words, start, end)
        text = " ".join(w["word"] for w in section_words)
        dur = end - start
        label = f"Section {i + 1}"
        if segment.get("has_cold_open") and i == 0:
            label = "COLD OPEN"
        lines.append(f'  [{label}] {start:.1f}s → {end:.1f}s ({dur:.1f}s): "{text}"')
    return "\n".join(lines)


def _format_proposed_cuts_for_critique(segments: list[dict], words: list[dict]) -> str:
    """Format proposed cuts + reconstructed text for the critique LLM."""
    lines = []
    for seg in segments:
        seg_num = seg.get("segment", "?")
        summary = seg.get("summary", "")
        dur = seg.get("duration", 0)
        lines.append(f"### Segment {seg_num}: {summary}")
        lines.append(f"Duration: {dur:.1f}s | Cuts: {seg.get('cuts', '')}")
        if seg.get("has_cold_open"):
            lines.append("Has cold open: yes")
        lines.append("")
        lines.append(_reconstruct_segment_text(seg, words))
        lines.append("")
    return "\n".join(lines)


CRITIQUE_SYSTEM_PROMPT = """\
You are a senior short-form video editor reviewing a junior editor's proposed cuts.

Your job is to find SPECIFIC, ACTIONABLE issues. Do NOT give vague feedback like \
"could be better." Every issue must quote exact words, give timestamps, and suggest \
a concrete fix.

## What to check for EACH segment:

1. **VERBAL STUMBLES** — Does the selected text contain broken phrases like \
"doing asking ourselves", "started sitting around", "I believe that the" (abandoned \
sentence)? If so, check the transcript for a cleaner alternative take at nearby \
timestamps. Quote the stumble AND the cleaner alternative.

2. **CUT-OFF SENTENCES** — Does any section end with an incomplete phrase? \
E.g. "every sales" when the speaker clearly said "every sales call" (the word \
"call" is at a nearby timestamp but was excluded). Check the last 3-4 words of \
each section.

3. **ORPHANED WORDS** — Are there words that got separated from their phrase \
by a section boundary? If "call" at 386.88s belongs to the phrase "every sales \
call" but the section ends at 380.34s, that's an orphaned word.

4. **HOOK QUALITY** — Do the first 1–3 seconds (first 5–10 words) form a strong \
opener? They should: (a) signal what the clip is about, and (b) create unresolved \
tension or curiosity. Strong patterns: stakes/tension (e.g. "most leaders make a \
fatal mistake") or a named concept that begs explanation (e.g. "adjust the access, \
not the affection"). Flag segments that open with a weak or generic line.

5. **MISSING BODY** — Is the hook present but the supporting body/example/payoff \
was cut? A hook without a body is a tease without satisfaction.

6. **OVER-SPLITTING** — Are two adjacent segments actually the same topic that \
should be merged? (E.g. "automation limits" and "marketing has a soul" are both \
about human-in-the-loop.)

7. **CROSS-TOPIC PULLING** — Does any segment contain sections from distant \
timestamps (>60s apart) that don't belong together?

## What to check GLOBALLY:

1. **MISSING TOPICS** — Scan the full transcript for content blocks that aren't \
covered by any segment. Even a 5-second standalone hook is a valid topic.

2. **TAKE SELECTION** — For topics with multiple takes (retakes), was the best \
take selected? Compare the text quality of the selected take vs alternatives.

3. **STRATEGY MATCH** — If the editing strategy is contiguous_batch, verify: \
no cold opens, no cross-topic sections, one segment per topic.

Be BRUTALLY honest. The goal is to catch issues BEFORE the video goes to production."""


async def critique_cuts(
    words: list[dict],
    segments: list[dict],
    model_id: str,
    openrouter_api_key: str,
) -> CritiquePlan:
    """Run critique pass on proposed cuts.

    Takes the original transcript + proposed cuts + reconstructed text,
    returns a structured critique identifying issues per segment and globally.
    """
    llm = ChatOpenAI(
        model=model_id,
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        temperature=0,
        timeout=300.0,
    )
    structured_llm = llm.with_structured_output(CritiquePlan)

    total_duration = words[-1]["end"] if words else 0
    proposed_text = _format_proposed_cuts_for_critique(segments, words)

    # Use the same formatted transcript for consistency
    formatted_transcript = format_transcript_for_llm(words)

    messages = [
        SystemMessage(content=CRITIQUE_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"## Source Transcript ({len(words)} words, {total_duration:.0f}s)\n\n"
                f"{formatted_transcript}\n\n"
                "---\n\n"
                f"## Proposed Cuts ({len(segments)} segments)\n\n"
                f"{proposed_text}\n\n"
                "---\n\n"
                "Review each segment and the overall plan. "
                "Be specific: quote words, give timestamps, name alternatives.\n\n"
                "If the plan is clean, say so — don't invent issues that aren't there."
            )
        ),
    ]

    print("\nCritiquing proposed cuts...", file=sys.stderr)
    t0 = time.time()
    critique = await structured_llm.ainvoke(messages)
    elapsed = time.time() - t0
    print(
        f"Critique done in {elapsed:.1f}s — {critique.overall_assessment}",
        file=sys.stderr,
    )

    # Log critique details
    if critique.missing_topics:
        print(f"\n  Missing topics:", file=sys.stderr)
        for t in critique.missing_topics:
            print(f"    • {t}", file=sys.stderr)
    if critique.segments_to_merge:
        print(f"  Merge suggestions:", file=sys.stderr)
        for m in critique.segments_to_merge:
            print(f"    • {m}", file=sys.stderr)
    for sc in critique.segment_critiques:
        if sc.verdict != "keep":
            print(f"  Seg {sc.segment_number} ({sc.verdict}):", file=sys.stderr)
            for issue in sc.issues:
                print(f"    ⚠ {issue}", file=sys.stderr)
            for fix in sc.suggested_fixes:
                print(f"    → {fix}", file=sys.stderr)
    print(f"\n  Assessment: {critique.assessment_summary}", file=sys.stderr)

    return critique


REFINE_SYSTEM_PROMPT = """\
You are a senior short-form video editor applying specific feedback to improve \
a set of proposed cuts.

You will receive:
1. The original word-level transcript
2. The proposed cuts (from pass 1)
3. A structured critique (from pass 2) with specific issues and suggested fixes

Your job is to produce a REVISED set of segments that addresses every critique item. \
Follow the suggested fixes closely — the critique was produced by a careful review.

## Rules:
- Apply every suggested fix unless it would break something else
- If adding a missing topic, create a new segment for it
- If merging segments, combine their sections under one segment
- If fixing a cut-off sentence, extend the section to include the missing words
- If replacing a bad take, find the cleaner alternative at the timestamps mentioned
- Maintain the same output schema (SegmentPlan with sections and cold_open_sections)
- Keep segments in chronological order by their first section timestamp
- DO NOT introduce new issues while fixing old ones
- If a critique item cannot be addressed, explain why in unaddressed_critiques"""


async def refine_cuts(
    words: list[dict],
    segments: list[dict],
    critique: CritiquePlan,
    model_id: str,
    openrouter_api_key: str,
    target_duration: float,
    tolerance: float,
) -> list[dict]:
    """Run refinement pass: apply critique feedback to produce revised cuts.

    Returns revised segments in the same format as propose_cuts output.
    """
    llm = ChatOpenAI(
        model=model_id,
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        temperature=0,
        timeout=300.0,
    )
    structured_llm = llm.with_structured_output(RefinedPlan)

    total_duration = words[-1]["end"] if words else 0
    proposed_text = _format_proposed_cuts_for_critique(segments, words)

    # Use formatted transcript + compact lookup for the refine step
    formatted_transcript = format_transcript_for_llm(words)
    slim = _slim_words(words)
    word_lookup = "\n".join(
        f'{{"w":"{w["word"]}","s":{w["start"]:.2f},"e":{w["end"]:.2f}}}' for w in slim
    )

    # Format critique as text
    critique_lines = []
    if critique.missing_topics:
        critique_lines.append("## Missing Topics")
        for t in critique.missing_topics:
            critique_lines.append(f"- {t}")
    if critique.segments_to_merge:
        critique_lines.append("\n## Segments to Merge")
        for m in critique.segments_to_merge:
            critique_lines.append(f"- {m}")
    critique_lines.append("\n## Per-Segment Critiques")
    for sc in critique.segment_critiques:
        critique_lines.append(f"\n### Segment {sc.segment_number} — {sc.verdict}")
        for issue in sc.issues:
            critique_lines.append(f"- ISSUE: {issue}")
        for fix in sc.suggested_fixes:
            critique_lines.append(f"- FIX: {fix}")
    critique_lines.append(f"\n## Overall: {critique.assessment_summary}")
    critique_text = "\n".join(critique_lines)

    messages = [
        SystemMessage(content=REFINE_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"## Source Transcript ({len(words)} words, {total_duration:.0f}s)\n\n"
                f"{formatted_transcript}\n\n"
                "## Word Lookup Table\n\n"
                f"{word_lookup}\n\n"
                "---\n\n"
                f"## Proposed Cuts (Pass 1)\n\n{proposed_text}\n\n"
                "---\n\n"
                f"## Critique (Pass 2)\n\n{critique_text}\n\n"
                "---\n\n"
                "Apply the critique fixes and produce revised segments. "
                "Use the EXACT timestamps from the word lookup table. "
                "For EVERY section, fill in `first_words` and `last_words` by "
                "looking up the actual words at your proposed timestamps.\n\n"
                f"Target duration: {target_duration:.0f}s (±{tolerance:.0f}s) — "
                "relaxed for contiguous_batch.\n\n"
                "List every change you made in changes_made."
            )
        ),
    ]

    print("\nRefining cuts based on critique...", file=sys.stderr)
    t0 = time.time()
    refined = await structured_llm.ainvoke(messages)
    elapsed = time.time() - t0
    print(
        f"Refinement done in {elapsed:.1f}s — "
        f"{len(refined.segments)} segments, {len(refined.changes_made)} changes",
        file=sys.stderr,
    )

    # Log changes
    print("\n  Changes applied:", file=sys.stderr)
    for change in refined.changes_made:
        print(f"    ✓ {change}", file=sys.stderr)
    if refined.unaddressed_critiques:
        print("  Unaddressed:", file=sys.stderr)
        for ua in refined.unaddressed_critiques:
            print(f"    ✗ {ua}", file=sys.stderr)

    # Convert RefinedPlan → same output format as propose_cuts
    results = []
    for seg in refined.segments:
        cold_open = [{"start": s.start, "end": s.end} for s in seg.cold_open_sections]
        body = sorted(
            [{"start": s.start, "end": s.end} for s in seg.sections],
            key=lambda x: x["start"],
        )
        all_sections = cold_open + body
        cuts_str = ",".join(f"{s['start']:.2f}:{s['end']:.2f}" for s in all_sections)
        total_dur = sum(s["end"] - s["start"] for s in all_sections)

        entry: dict = {
            "segment": seg.segment_number,
            "summary": seg.summary,
            "hook": seg.hook_description,
            "cuts": cuts_str,
            "duration": round(total_dur, 1),
            "section_count": len(all_sections),
            "has_cold_open": len(cold_open) > 0,
        }
        results.append(entry)

    return results


# ── Core logic ────────────────────────────────────────────────────────────────


async def propose_cuts(
    words: list[dict],
    target_duration: float = 50.0,
    tolerance: float = 30.0,
    custom_prompt: str | None = None,
    max_segments: int | None = None,
    model: str | None = None,
    critique: bool = False,
) -> list[dict]:
    """Use LLM to identify segments to extract from a word-level transcript.

    Args:
        words: List of word dicts with 'word', 'start', 'end'
        target_duration: Target duration per segment in seconds
        tolerance: Acceptable margin in seconds (±)
        custom_prompt: Optional edit prompt override
        max_segments: Max segments to extract (None = as many as possible)
        model: OpenRouter model ID (default: google/gemini-3-flash-preview)
        critique: If True, run critique + refine passes after initial proposal

    Returns:
        List of segment dicts with 'segment', 'summary', 'cuts', 'duration',
        plus 'analysis' on the first segment.
    """
    model_id = model or DEFAULT_MODEL

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")

    # No explicit reasoning/thinking — the prompt already chains thought
    # via the analysis step, and removing it cuts latency dramatically.
    llm = ChatOpenAI(
        model=model_id,
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        temperature=0,  # precision over variety — we need deterministic cuts
        timeout=300.0,  # 5 min — should be plenty without extended thinking
    )
    structured_llm = llm.with_structured_output(RepurposePlan)

    total_duration = words[-1]["end"] if words else 0
    min_duration = max(0, target_duration - tolerance)
    max_duration = target_duration + tolerance

    system_prompt = _build_system_prompt(
        target_duration=target_duration,
        min_duration=min_duration,
        max_duration=max_duration,
        total_duration=total_duration,
        max_segments=max_segments,
        custom_prompt=custom_prompt,
    )

    # Format transcript as timestamped running text (much cheaper + more
    # natural for the LLM to read than raw JSON).
    formatted_transcript = format_transcript_for_llm(words)

    # Also send a compact word lookup table — just word/start/end as JSON
    # lines so the LLM can do precise timestamp lookups.
    slim = _slim_words(words)
    # Compact JSON: one word per line, no indent
    word_lookup = "\n".join(
        f'{{"w":"{w["word"]}","s":{w["start"]:.2f},"e":{w["end"]:.2f}}}' for w in slim
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Video duration: {total_duration:.1f}s\n\n"
                "## Transcript (timestamped running text)\n\n"
                "Read this like a script — `@` markers show timestamps in seconds. "
                "Gaps indicate silences.\n\n"
                f"{formatted_transcript}\n\n"
                "## Word Lookup Table\n\n"
                "Compact word-level data for precise timestamp lookups "
                "(w=word, s=start, e=end):\n\n"
                f"{word_lookup}\n\n"
                "---\n\n"
                "First fill in the analysis (content type, editing strategy, single vs "
                "multi, cold open, keepable seconds). The editing_strategy field is "
                "CRITICAL — examine the transcript for retakes, topic changes, and "
                "per-topic hooks before deciding.\n\n"
                "Then propose segments according to your detected strategy:\n"
                "- highlight_reel: extract best moments, cold opens OK, multiple "
                "sections per segment\n"
                "- contiguous_batch: one segment per topic, best take only, NO cold "
                "opens, no cross-topic sections\n\n"
                "For EVERY section, fill in `first_words` and `last_words` by "
                "looking up the actual words at your proposed timestamps. These are "
                "verified programmatically — they must match.\n\n"
                "If you identified a cold open candidate (highlight_reel only), put "
                "those sections in `cold_open_sections` (NOT in `sections`).\n\n"
                f"Target duration per segment: {target_duration:.0f}s (±{tolerance:.0f}s). "
                f"(Relaxed for contiguous_batch — use natural take length.)"
            )
        ),
    ]

    print(
        f"Analyzing transcript ({len(words)} words, {total_duration:.0f}s) with {model_id}...",
        file=sys.stderr,
    )
    start = time.time()
    response = await structured_llm.ainvoke(messages)
    elapsed = time.time() - start
    print(
        f"Done in {elapsed:.1f}s — found {len(response.segments)} segments",
        file=sys.stderr,
    )

    # Log analysis to stderr for visibility
    analysis = response.analysis
    print(
        f"\n  Content type: {analysis.content_type}\n"
        f"  Editing strategy: {analysis.editing_strategy}\n"
        f"  Single/multi: {analysis.single_or_multi}\n"
        f"  Cold open: {analysis.cold_open_candidate or 'none'}\n"
        f"  Keepable content: ~{analysis.keepable_content_seconds:.0f}s "
        f"(of {total_duration:.0f}s total)\n",
        file=sys.stderr,
    )

    # Convert to clean output format
    results = []
    for i, seg in enumerate(response.segments):
        # Collect all WordRange objects for first_words/last_words verification
        all_ranges: list[WordRange] = list(seg.cold_open_sections) + list(seg.sections)

        # Cold open sections go first (unsorted — they come from later timestamps)
        cold_open = [{"start": s.start, "end": s.end} for s in seg.cold_open_sections]
        # Main body sections in chronological order
        body = sorted(
            [{"start": s.start, "end": s.end} for s in seg.sections],
            key=lambda x: x["start"],
        )
        all_sections = cold_open + body
        cuts_str = ",".join(f"{s['start']:.2f}:{s['end']:.2f}" for s in all_sections)
        total_dur = sum(s["end"] - s["start"] for s in all_sections)

        entry: dict = {
            "segment": seg.segment_number,
            "summary": seg.summary,
            "hook": seg.hook_description,
            "cuts": cuts_str,
            "duration": round(total_dur, 1),
            "section_count": len(all_sections),
            "has_cold_open": len(cold_open) > 0,
            # Store quoted words for programmatic verification (not in final output)
            "_first_words": [r.first_words for r in all_ranges],
            "_last_words": [r.last_words for r in all_ranges],
        }
        if cold_open:
            cold_dur = sum(s["end"] - s["start"] for s in cold_open)
            print(
                f"  Cold open: {len(cold_open)} section(s), {cold_dur:.1f}s",
                file=sys.stderr,
            )
        # Attach analysis to the first segment for downstream consumption
        if i == 0:
            entry["analysis"] = {
                "content_type": analysis.content_type,
                "editing_strategy": analysis.editing_strategy,
                "single_or_multi": analysis.single_or_multi,
                "cold_open_candidate": analysis.cold_open_candidate,
                "keepable_content_seconds": analysis.keepable_content_seconds,
            }
        results.append(entry)

    # ── Post-processing: verify quoted words match actual transcript ───
    quote_warnings = verify_quoted_words(results, words)
    if quote_warnings:
        print("\n⚠️  QUOTED WORD MISMATCHES (LLM boundary errors):", file=sys.stderr)
        for warn in quote_warnings:
            print(f"    • {warn}", file=sys.stderr)

    # ── Post-processing: verify and fix cut boundaries ────────────────
    results, boundary_warnings = verify_cut_boundaries(results, words)
    if boundary_warnings:
        print("\n⚠️  CUT BOUNDARY ISSUES (auto-fixed where possible):", file=sys.stderr)
        for warn in boundary_warnings:
            print(f"    • {warn}", file=sys.stderr)

    # Always print the text preview so operators can verify
    print_cut_preview(results, words)

    # ── Critique + Refine (optional) ──────────────────────────────────
    if critique:
        print(
            f"\n{'═' * 60}\nPASS 2: CRITIQUE\n{'═' * 60}",
            file=sys.stderr,
        )
        critique_result = await critique_cuts(
            words=words,
            segments=results,
            model_id=model_id,
            openrouter_api_key=openrouter_api_key,
        )

        if critique_result.overall_assessment in ("needs_revision", "major_rework"):
            print(
                f"\n{'═' * 60}\nPASS 3: REFINE\n{'═' * 60}",
                file=sys.stderr,
            )
            # Preserve the analysis from pass 1
            analysis_entry = results[0].get("analysis") if results else None

            refined_results = await refine_cuts(
                words=words,
                segments=results,
                critique=critique_result,
                model_id=model_id,
                openrouter_api_key=openrouter_api_key,
                target_duration=target_duration,
                tolerance=tolerance,
            )

            # Re-attach analysis to first segment
            if analysis_entry and refined_results:
                refined_results[0]["analysis"] = analysis_entry

            # Run boundary verification on refined results too
            refined_results, refined_warnings = verify_cut_boundaries(
                refined_results, words
            )
            if refined_warnings:
                print("\n⚠️  REFINED CUT BOUNDARY ISSUES:", file=sys.stderr)
                for warn in refined_warnings:
                    print(f"    • {warn}", file=sys.stderr)

            print_cut_preview(refined_results, words)
            results = refined_results
        else:
            print(
                "\n  ✓ Critique passed — no refinement needed.",
                file=sys.stderr,
            )

    # Strip internal verification fields before returning
    for r in results:
        r.pop("_first_words", None)
        r.pop("_last_words", None)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propose cuts for repurposing a long-form video into shorts",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--video",
        type=Path,
        help="Video file (will transcribe first)",
    )
    group.add_argument(
        "--transcript",
        type=Path,
        help="Word-level transcript JSON file",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help=(
            "Target duration per segment in seconds. "
            "Default: 50, or settings propose_cuts.duration, or this flag."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help=(
            "Duration tolerance ± seconds. "
            "Default: 30, or settings propose_cuts.tolerance, or this flag."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of segments to extract (default: as many as possible)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom edit prompt",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "OpenRouter model ID (default: google/gemini-3-flash-preview). "
            "For higher quality cuts: anthropic/claude-opus-4"
        ),
    )
    parser.add_argument(
        "--no-critique",
        action="store_false",
        dest="critique",
        default=True,
        help=(
            "Skip critique + refine passes. By default, the LLM runs critique "
            "and refinement to fix stumbles, cut-offs, and missing topics (~3x "
            "LLM cost but much higher quality)."
        ),
    )
    parser.add_argument(
        "--edl",
        type=Path,
        default=None,
        help="Output directory for EDL files (one per segment)",
    )
    args = parser.parse_args()

    # Resolve duration/tolerance: script default 50/30 → settings.json → CLI
    context_path = args.transcript or args.video
    settings = load_settings(context_path) if context_path else {}
    propose_opts = settings.get("propose_cuts") or {}
    target_duration = (
        args.duration if args.duration is not None else propose_opts.get("duration", 50)
    )
    tolerance = (
        args.tolerance
        if args.tolerance is not None
        else propose_opts.get("tolerance", 30.0)
    )
    target_duration = int(target_duration)
    tolerance = float(tolerance)

    # Load or generate transcript
    if args.transcript:
        if not args.transcript.exists():
            print(f"Error: Transcript not found: {args.transcript}", file=sys.stderr)
            sys.exit(1)
        _, words = read_word_transcript_file(args.transcript)
        print(f"Loaded {len(words)} words from {args.transcript}", file=sys.stderr)
    else:
        if not args.video.exists():
            print(f"Error: Video not found: {args.video}", file=sys.stderr)
            sys.exit(1)
        print(f"Transcribing {args.video.name}...", file=sys.stderr)
        result = await get_transcript(args.video)
        words = result["words"]
        print(f"Transcribed: {len(words)} words", file=sys.stderr)

    # Propose cuts
    segments = await propose_cuts(
        words,
        target_duration=target_duration,
        tolerance=tolerance,
        custom_prompt=args.prompt,
        max_segments=args.count,
        model=args.model,
        critique=args.critique,
    )

    # Output clean JSON to stdout
    print(json.dumps(segments, indent=2))

    # Write EDL files if requested
    if args.edl:
        args.edl.mkdir(parents=True, exist_ok=True)
        source_name = args.video.name if args.video else "AX"
        for seg in segments:
            sections = []
            for part in seg["cuts"].split(","):
                start_str, end_str = part.split(":")
                sections.append({"start": float(start_str), "end": float(end_str)})
            edl_path = args.edl / f"segment_{seg['segment']:02d}.edl"
            write_edl(
                edl_path,
                title=f"Segment {seg['segment']} - {seg['summary']}",
                sections=sections,
                source_name=source_name,
            )
            print(f"  EDL: {edl_path}", file=sys.stderr)
        print(f"Wrote {len(segments)} EDL files to {args.edl}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
