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
import sys
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from get_transcript import get_transcript, read_word_transcript_file


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


# ── Pydantic models for structured LLM output ────────────────────────────────


class ContentAnalysis(BaseModel):
    """Pre-cut analysis of the source content — forces the LLM to think before cutting."""

    content_type: str = Field(
        description=(
            "One of: procedural (step-by-step tutorial/guide), "
            "multi_topic (distinct ideas/opinions), interview (Q&A format), "
            "listicle (numbered tips/items), narrative (story with arc), "
            "opinion (single thesis argued throughout)"
        )
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
            "procedural/tutorial (cold opens don't suit how-to content)."
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

    words: list[str] = Field(description="The words in this section")
    start: float = Field(description="Start timestamp in seconds")
    end: float = Field(description="End timestamp in seconds")


class SegmentPlan(BaseModel):
    """A single short-form segment plan."""

    segment_number: int = Field(description="Segment number (1, 2, 3, etc.)")
    sections: list[WordRange] = Field(
        description="List of sections to keep (will be concatenated in order)"
    )
    summary: str = Field(description="Brief summary of what this segment covers")
    hook_description: str = Field(
        description="What makes the opening 3-5 seconds grab attention?"
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
        segment_instruction = f"Extract exactly {max_segments} segment(s)."
    else:
        segment_instruction = (
            "Determine the right number of segments based on your content analysis. "
            "Often that number is 1. Prioritize QUALITY over QUANTITY."
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
        "- **opinion**: Single thesis argued throughout.\n\n"
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
        #
        # ── STEP 2: COLD VIEWER TEST ─────────────────────────────────────
        "## STEP 2 — THE COLD VIEWER TEST\n\n"
        "Every segment MUST pass this test:\n"
        "> Would someone who has NEVER seen this video, scrolling TikTok, "
        "understand and want to keep watching THIS segment from second 1?\n\n"
        "If a segment requires context from another segment to make sense, it is a "
        "FRAGMENT, not a short. Merge it or restructure.\n\n"
        "Common fragment signals:\n"
        "- Opens mid-procedure ('Then you need to...' with no setup)\n"
        "- References something explained earlier ('the thing we just set up')\n"
        "- Only makes sense as part 2/3 of a sequence\n\n"
        #
        # ── STEP 3: CUTTING PHILOSOPHY ───────────────────────────────────
        "## STEP 3 — CUTTING PHILOSOPHY\n\n"
        "You are creating a HIGHLIGHT REEL, not a summary. Think like a trailer editor:\n\n"
        "**Pace matters more than completeness.** A tight 45s short that moves fast "
        "beats a 60s short that drags.\n\n"
        "**Cut aggressively.** Remove:\n"
        "- Verbose explanations and clarifications\n"
        "- Conditional branches ('if you purchased X...', 'you may or may not...')\n"
        "- Setup/transitions between steps ('now let's move on to...')\n"
        "- Pleasantries, self-corrections, 'um/uh' padding\n"
        "- Repetition (speaker says the same thing twice differently)\n"
        "- Subscribe/follow CTAs at the end\n\n"
        "**Keep the action.** Keep:\n"
        "- The core 'do this' moments (for tutorials)\n"
        "- Key results and payoffs ('setup completed!', 'and now we're streaming')\n"
        "- The hook (opening statement that grabs)\n"
        "- Surprising or insightful moments\n"
        "- Concrete examples and numbers\n\n"
        "**Use MANY short sections for fast pacing.** Don't limit yourself to 2-4 "
        "sections per segment. Use 5-15 shorter sections to create a dynamic, "
        "jump-cut rhythm. This is expected in short-form video — rapid cuts between "
        "key moments keep viewers watching.\n\n"
        #
        # ── STEP 4: SECTION RULES ────────────────────────────────────────
        "## STEP 4 — SECTION RULES\n\n"
        "- Sections within a segment MUST be in chronological order "
        "(except cold opens, which come from later and are placed FIRST)\n"
        "- Use the EXACT timestamps from the transcript — do NOT round or adjust\n"
        "- Never cut mid-word. Each section should start at a word's `start` "
        "timestamp and end at a word's `end` timestamp\n"
        "- Include complete phrases, but individual sections do NOT need to be "
        "complete sentences — jump cuts between phrase fragments are fine in short-form\n"
        "- Sections can be as short as 1-3 seconds (a single phrase)\n\n"
        #
        # ── STEP 5: DURATION ─────────────────────────────────────────────
        "## STEP 5 — DURATION RULES (STRICT)\n\n"
        f"- Target: {target_duration:.0f} seconds per segment\n"
        f"- Minimum: {min_duration:.0f} seconds — segments shorter than this are REJECTED\n"
        f"- Maximum: {max_duration:.0f} seconds\n"
        "- Before finalizing, add up ALL section durations (end - start for each). "
        "If the total is below the minimum, add more relevant sections.\n"
        "- If the total exceeds the maximum, cut more aggressively.\n\n"
        #
        # ── TASK ─────────────────────────────────────────────────────────
        f"## YOUR TASK\n\n"
        f"{segment_instruction}\n"
        f"{custom_section}"
    )


# ── Core logic ────────────────────────────────────────────────────────────────


async def propose_cuts(
    words: list[dict],
    target_duration: float = 60.0,
    tolerance: float = 20.0,
    custom_prompt: str | None = None,
    max_segments: int | None = None,
    model: str | None = None,
) -> list[dict]:
    """Use LLM to identify segments to extract from a word-level transcript.

    Args:
        words: List of word dicts with 'word', 'start', 'end'
        target_duration: Target duration per segment in seconds
        tolerance: Acceptable margin in seconds (±)
        custom_prompt: Optional edit prompt override
        max_segments: Max segments to extract (None = as many as possible)
        model: OpenRouter model ID (default: google/gemini-3-flash-preview)

    Returns:
        List of segment dicts with 'segment', 'summary', 'cuts', 'duration',
        plus 'analysis' on the first segment.
    """
    model_id = model or DEFAULT_MODEL

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")

    # Configure reasoning tokens based on model capability
    extra_body: dict = {}
    if "gemini" in model_id or "deepseek" in model_id:
        extra_body["reasoning"] = {"max_tokens": 4000, "enabled": True}

    llm = ChatOpenAI(
        model=model_id,
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        timeout=300.0,  # 5 min — Opus can be slower
        **({"extra_body": extra_body} if extra_body else {}),
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

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Video duration: {total_duration:.1f}s\n\n"
                f"Word-level transcript ({len(words)} words):\n"
                f"{json.dumps(words, indent=2)}\n\n"
                "First fill in the analysis (content type, single vs multi, cold open, "
                "keepable seconds). Then propose segments accordingly.\n\n"
                f"Target duration per segment: {target_duration:.0f}s (±{tolerance:.0f}s). "
                f"REJECT any segment under {min_duration:.0f}s."
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
        f"  Strategy: {analysis.single_or_multi}\n"
        f"  Cold open: {analysis.cold_open_candidate or 'none'}\n"
        f"  Keepable content: ~{analysis.keepable_content_seconds:.0f}s "
        f"(of {total_duration:.0f}s total)\n",
        file=sys.stderr,
    )

    # Convert to clean output format
    results = []
    for i, seg in enumerate(response.segments):
        sections = sorted(
            [{"start": s.start, "end": s.end} for s in seg.sections],
            key=lambda x: x["start"],
        )
        cuts_str = ",".join(f"{s['start']:.2f}:{s['end']:.2f}" for s in sections)
        total_dur = sum(s["end"] - s["start"] for s in sections)

        entry: dict = {
            "segment": seg.segment_number,
            "summary": seg.summary,
            "hook": seg.hook_description,
            "cuts": cuts_str,
            "duration": round(total_dur, 1),
            "section_count": len(sections),
        }
        # Attach analysis to the first segment for downstream consumption
        if i == 0:
            entry["analysis"] = {
                "content_type": analysis.content_type,
                "single_or_multi": analysis.single_or_multi,
                "cold_open_candidate": analysis.cold_open_candidate,
                "keepable_content_seconds": analysis.keepable_content_seconds,
            }
        results.append(entry)

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
        default=60,
        help="Target duration per segment in seconds (default: 60)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=20.0,
        help="Duration tolerance ± seconds (default: 20)",
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
        "--edl",
        type=Path,
        default=None,
        help="Output directory for EDL files (one per segment)",
    )
    args = parser.parse_args()

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
        target_duration=args.duration,
        tolerance=args.tolerance,
        custom_prompt=args.prompt,
        max_segments=args.count,
        model=args.model,
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
