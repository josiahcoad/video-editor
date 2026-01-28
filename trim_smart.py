#!/usr/bin/env python3
"""
Intelligently trim video to a target duration using LLM to identify best word ranges.

Uses word-level timestamps to allow precise cuts at any word boundary.
"""

import argparse
import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from get_transcript import get_transcript, read_word_transcript_file
from pydantic import BaseModel, Field

# Try to load from .env file if available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class WordRanges(BaseModel):
    """A section of the video to keep."""

    words: list[str] = Field(description="The words in this section")
    start: float = Field(description="Start timestamp in seconds")
    end: float = Field(description="End timestamp in seconds")


class CutPlan(BaseModel):
    """Response model for sections to keep."""

    sections: list[WordRanges] = Field(
        description="List of sections to keep in the final video"
    )
    edit_summary: str = Field(
        description="One sentence executive summary of the edits made and why (e.g., 'Removed ramblings, 3 retakes, and verbose explanations to create a focused 60-second video')"
    )


class CritiqueResponse(BaseModel):
    """Response model for critique."""

    issue_count: int = Field(
        description="Number of issues found. 0 means the cuts are good."
    )
    critique: str = Field(
        description="Detailed critique of the cuts, or 'OK' if no issues found"
    )


async def get_transcript_with_words(
    video_path: Path, transcript_file: Path | None = None
) -> tuple[str, list[dict]]:
    """Get transcript and word-level timestamps.

    Args:
        video_path: Path to video file
        transcript_file: Optional path to word-level transcript JSON file (from get_transcript.py)

    Returns:
        Tuple of (transcript_text, words_list)
    """
    # If transcript file provided, read from it
    if transcript_file:
        return read_word_transcript_file(transcript_file)

    # Otherwise, transcribe the video
    result = await get_transcript(video_path)
    return result["transcript"], result["words"]


async def find_sections_to_keep(
    words: list[dict],
    target_duration: float | None = 60.0,
    tolerance: float = 20.0,
    brand_brief: str | None = None,
) -> tuple[list[dict], str]:
    """Use LLM to identify which sections to keep from the word-level transcript.

    Args:
        words: List of word dicts with 'word', 'start', 'end'
        target_duration: Target duration in seconds. If None, just "tighten up" without duration constraint.
        tolerance: Acceptable margin in seconds (e.g., 20 means ±20s, so 60s target = 40-80s range). Ignored if target_duration is None.
        brand_brief: Optional brand brief to guide cut planning (e.g., brand voice, audience, content preferences)

    Returns:
        Tuple of (list of segment dicts with 'text', 'start', 'end', 'duration', edit_summary string)
    """
    import json

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")

    llm = ChatOpenAI(
        model="google/gemini-3-flash-preview",
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        extra_body={"reasoning": {"max_tokens": 2000, "enabled": True}},
    )

    structured_llm = llm.with_structured_output(CutPlan)

    # Calculate total video duration
    total_duration = words[-1]["end"] if words else 0

    # Calculate acceptable range
    if target_duration is None:
        # "Tighten up" mode: remove mistakes/fumbles/ramblings but no duration constraint
        min_duration = 0
        max_duration = total_duration
        duration_instruction = "Remove mistakes, fumbles, retakes, ramblings, and asides to create a clean, tight video. There is NO duration target - keep everything that's good content, just remove the fluff."
    else:
        # Normal mode with duration target
        # If target is longer than video, use most/all of the video (allow cutting up to min(10, tolerance) seconds)
        if target_duration > total_duration:
            min_duration = max(0, total_duration - min(10, tolerance))
            max_duration = total_duration
        else:
            min_duration = max(0, target_duration - tolerance)
            max_duration = target_duration + tolerance
        duration_instruction = f"DURATION REQUIREMENT: Your cut must be between {min_duration:.0f} and {max_duration:.0f} seconds.\n"
        duration_instruction += (
            f"  - The target of {target_duration:.0f} seconds is just a guideline\n"
        )
        duration_instruction += f"  - Completeness and coherence are MORE important than hitting the exact target\n"
        duration_instruction += f"  - If you need {max_duration:.0f} seconds to include all key elements, use them\n"
        duration_instruction += f"  - If you can be complete and coherent in {min_duration:.0f} seconds, that's fine too"

    messages = [
        SystemMessage(
            content="You are a professional video editor. Your job is to select sections from a word-level transcript to create a coherent, well-flowing video.\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "1. Sections MUST be in chronological order (start to end of video)\n"
            "2. Each section must be a complete thought or phrase - never cut mid-sentence\n"
            "3. Sections must flow logically from one to the next - the final transcript should read as a coherent narrative\n"
            "4. Maintain the story arc: opening hook, main content, conclusion\n"
            "5. COMPLETENESS: If the video introduces a list, framework, or set of concepts, ensure ALL elements are represented in your selection\n"
            "6. PROMISE FULFILLMENT: If the opening makes a promise or sets up an expectation, ensure that promise is fulfilled in your selection\n"
            "7. LOGICAL FLOW: Ensure transitions between segments make sense - avoid contradictory statements or missing context\n"
            "8. DURATION: Total duration must be within the acceptable range provided. Completeness and coherence are MORE important than hitting the exact target - if you need to use the full range to include all key elements, do so.\n"
            "9. Use the EXACT words and timestamps from the transcript - do not paraphrase or modify words\n\n"
            "WHAT TO CUT:\n"
            "- Ramblings/asides/tangents (off-topic digressions)\n"
            "- Retakes/fumbles (obvious mistakes, false starts, repeated attempts)\n"
            "- Verbose explanations when brief mentions suffice\n\n"
            "WHAT NOT TO CUT:\n"
            "- Core narrative (the main story/argument)\n"
            "- Anything that would make the result not make coherent logical sense\n"
            "- Key structural elements (all steps in a list, all concepts in a framework)\n"
            "- Transitions that provide necessary context\n"
            "- Promises or expectations set up in the opening\n\n"
            "PRIORITIZATION (when time-constrained):\n"
            "- BREADTH over DEPTH: It's better to mention all key concepts briefly than to elaborate on some while omitting others\n"
            "- Introductions/definitions of concepts are higher priority than detailed elaborations\n"
            "- If you must cut, trim verbose explanations first - keep core definitions and all structural elements intact\n"
            "- Every element in a list/framework should at least be named, even if not fully explained\n\n"
            "VALIDATION: Before finalizing, verify:\n"
            "- All key concepts introduced in the opening are covered (even if briefly)\n"
            "- No structural elements are completely missing\n"
            "- The narrative structure is complete and logical"
            + (
                f"\n\nBRAND BRIEF:\n{brand_brief}\n\nUse this brand brief to guide your cuts. Ensure the final video aligns with the brand voice, audience, and content preferences specified."
                if brand_brief
                else ""
            )
        ),
        HumanMessage(
            content=f"{duration_instruction}\n"
            f"Total video duration: {total_duration:.1f} seconds\n\n"
            f"Word-level transcript (JSON):\n{json.dumps(words, indent=2)}\n\n"
            f"Select sections to keep that:\n"
            f"- Are in chronological order (earliest to latest)\n"
            f"- Form a coherent, flowing narrative when read together\n"
            f"- Include ALL key concepts/elements introduced in the video (breadth over depth)\n"
            f"- Prefer brief mentions of all elements over detailed explanations of some\n"
            f"- Fulfill any promises or expectations set up in the opening\n"
            f"- Have logical transitions between segments\n"
            + (
                f"- Total duration MUST be between {min_duration:.0f} and {max_duration:.0f} seconds (use the full range if needed for completeness)\n"
                if target_duration is not None
                else ""
            )
            + f"- Use the EXACT words from the transcript with their original timestamps\n\n"
            f"Return the sections with words, start timestamp, and end timestamp.\n\n"
            f"Also provide a one-sentence executive summary (edit_summary) describing what was edited and why. "
            f"Be specific about what was removed (e.g., 'ramblings', 'retakes', 'verbose explanations', 'tangents') "
            + (
                f"and the goal (e.g., 'to create a focused {target_duration:.0f}-second video', 'to improve flow', 'to remove redundancy')."
                if target_duration is not None
                else "and the goal (e.g., 'to tighten up the video', 'to remove fluff', 'to create a clean short-form video')."
            )
        ),
    ]

    response = await structured_llm.ainvoke(messages)

    print("\n" + "=" * 60)
    print("LLM RESPONSE:")
    print("=" * 60)
    print(f"Keep sections: {len(response.sections)} sections")
    for i, section in enumerate(response.sections, 1):
        print(
            f"  Section {i}: {section.start:.2f}s - {section.end:.2f}s ({section.end - section.start:.1f}s)"
        )
        print(f"    Words: {' '.join(section.words)}")
    print(f"\nEdit Summary: {response.edit_summary}")
    print("=" * 60 + "\n")

    # Convert KeepSection objects to segment dicts and sort by start time
    segments = []
    for section in response.sections:
        segments.append(
            {
                "text": " ".join(section.words),
                "start": section.start,
                "end": section.end,
                "duration": section.end - section.start,
            }
        )

    # Sort segments by start time to ensure chronological order
    segments.sort(key=lambda x: x["start"])

    # Warn if segments were out of order
    if len(response.sections) > 1:
        original_order = [s.start for s in response.sections]
        sorted_order = [s["start"] for s in segments]
        if original_order != sorted_order:
            print(
                "⚠️  WARNING: LLM returned sections out of chronological order. Auto-sorted."
            )

    return segments, response.edit_summary


async def critique_cuts(
    original_transcript: str,
    segments: list[dict],
    target_duration: float,
    brand_brief: str | None = None,
) -> CritiqueResponse:
    """Use LLM to critique the quality of the cuts.

    Args:
        original_transcript: Full original transcript
        segments: List of segment dicts with 'text', 'start', 'end', 'duration'
        target_duration: Target duration in seconds (the constraint we're working within)
        brand_brief: Optional brand brief to guide critique (e.g., brand voice, audience, content preferences)

    Returns:
        CritiqueResponse with issue_count and critique text
    """
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")

    llm = ChatOpenAI(
        model="google/gemini-3-flash-preview",
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        extra_body={"reasoning": {"max_tokens": 2000, "enabled": True}},
    )

    structured_llm = llm.with_structured_output(CritiqueResponse)

    # Format segments info for critique
    segments_info = []
    for i, seg in enumerate(segments, 1):
        segments_info.append(
            f"Segment {i}: {seg['start']:.2f}s - {seg['end']:.2f}s ({seg['duration']:.1f}s)\n"
            f"  Text: {seg['text']}"
        )

    total_duration = sum(seg["duration"] for seg in segments)

    messages = [
        SystemMessage(
            content="You are a professional video editor reviewing a cut. Identify ONLY structural problems with how the content was cut.\n\n"
            "WHAT TO CRITIQUE (real problems):\n"
            "- Missing key structural elements (e.g., if 4 steps are promised, are all 4 mentioned?)\n"
            "- Jarring transitions where context is lost between segments\n"
            "- Mid-sentence or mid-thought cuts that break coherence\n"
            "- Logical gaps where the viewer would be confused\n"
            "- Broken promises (opening sets up something that's never delivered)\n\n"
            "WHAT NOT TO CRITIQUE:\n"
            "- Duration being slightly over/under target (that's acceptable)\n"
            "- Content that could be 'shorter' but flows naturally\n"
            "- Natural conversational phrases or transitions\n"
            "- The closing/CTA unless it's actually broken\n"
            "- Suggesting cuts just to hit a duration target\n\n"
            "Be conservative - only flag issues that would genuinely confuse or frustrate a viewer.\n"
            "Return issue_count=0 and 'OK' if the cuts are coherent and all key elements are present."
            + (
                f"\n\nBRAND BRIEF:\n{brand_brief}\n\nAlso evaluate if the cuts align with the brand voice, audience, and content preferences specified in the brand brief."
                if brand_brief
                else ""
            )
        ),
        HumanMessage(
            content=f"Target duration: ~{target_duration} seconds\n"
            f"Actual cut duration: {total_duration:.1f} seconds\n\n"
            f"ORIGINAL FULL TRANSCRIPT:\n{original_transcript}\n\n"
            f"CUT PLAN:\n{chr(10).join(segments_info)}\n\n"
            f"Are there any REAL structural problems with these cuts? Only flag issues that would genuinely hurt the viewing experience."
            + (
                f"\n\nAlso check if the cuts align with the brand brief provided."
                if brand_brief
                else ""
            )
        ),
    ]

    response = await structured_llm.ainvoke(messages)

    print("\n" + "=" * 60)
    print("CRITIQUE:")
    print("=" * 60)
    print(f"Issue count: {response.issue_count}")
    print(f"\n{response.critique}")
    print("=" * 60 + "\n")

    return response


async def apply_critique(
    words: list[dict],
    original_transcript: str,
    initial_segments: list[dict],
    critique: CritiqueResponse,
    target_duration: float,
    tolerance: float,
    brand_brief: str | None = None,
) -> tuple[list[dict], str]:
    """Apply critique feedback to improve the cut plan.

    Args:
        words: List of word dicts with 'word', 'start', 'end'
        original_transcript: Full original transcript
        initial_segments: Initial cut plan segments
        critique: CritiqueResponse with feedback
        target_duration: Target duration in seconds
        tolerance: Acceptable margin in seconds
        brand_brief: Optional brand brief to guide cut planning (e.g., brand voice, audience, content preferences)

    Returns:
        Tuple of (improved list of segment dicts, edit_summary string)
    """
    import json

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")

    llm = ChatOpenAI(
        model="google/gemini-3-flash-preview",
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        extra_body={"reasoning": {"max_tokens": 2000, "enabled": True}},
    )

    structured_llm = llm.with_structured_output(CutPlan)

    # Calculate total video duration
    total_duration = words[-1]["end"] if words else 0

    # Calculate acceptable range
    # If target is longer than video, use most/all of the video (allow cutting up to min(10, tolerance) seconds)
    if target_duration > total_duration:
        min_duration = max(0, total_duration - min(10, tolerance))
        max_duration = total_duration
    else:
        min_duration = max(0, target_duration - tolerance)
        max_duration = target_duration + tolerance

    # Format initial segments for context
    initial_segments_info = []
    for i, seg in enumerate(initial_segments, 1):
        initial_segments_info.append(
            f"Segment {i}: {seg['start']:.2f}s - {seg['end']:.2f}s ({seg['duration']:.1f}s)\n"
            f"  Text: {seg['text']}"
        )
    initial_total = sum(seg["duration"] for seg in initial_segments)

    messages = [
        SystemMessage(
            content="You are a professional video editor. Your previous cut plan was critiqued, and you need to improve it based on the feedback.\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "1. Address ALL issues raised in the critique\n"
            "2. Sections MUST be in chronological order (start to end of video)\n"
            "3. Each section must be a complete thought or phrase - never cut mid-sentence\n"
            "4. Sections must flow logically from one to the next\n"
            "5. Maintain the story arc: opening hook, main content, conclusion\n"
            "6. COMPLETENESS: Ensure ALL key concepts/elements are represented\n"
            "7. PROMISE FULFILLMENT: Fulfill any promises made in the opening\n"
            "8. LOGICAL FLOW: Ensure smooth transitions between segments\n"
            "9. Use the EXACT words and timestamps from the transcript\n"
            "10. DURATION: Total duration must be within the acceptable range. Fixing critique issues and completeness are MORE important than hitting the exact target - use the full range if needed.\n\n"
            "WHAT TO CUT:\n"
            "- Ramblings/asides/tangents (off-topic digressions)\n"
            "- Retakes/fumbles (obvious mistakes, false starts, repeated attempts)\n"
            "- Verbose explanations when brief mentions suffice\n\n"
            "WHAT NOT TO CUT:\n"
            "- Core narrative (the main story/argument)\n"
            "- Anything that would make the result not make coherent logical sense\n"
            "- Key structural elements (all steps in a list, all concepts in a framework)\n"
            "- Transitions that provide necessary context\n"
            "- Promises or expectations set up in the opening\n\n"
            "IMPROVEMENT PRIORITY:\n"
            "- Fix the specific issues identified in the critique\n"
            "- If critique says something is missing, include it\n"
            "- If critique says transitions are jarring, select better segments\n"
            "- If critique says promises aren't fulfilled, ensure they are\n"
            "- Keep what worked well from the initial cut if it wasn't critiqued"
            + (
                f"\n\nBRAND BRIEF:\n{brand_brief}\n\nUse this brand brief to guide your improved cuts. Ensure the final video aligns with the brand voice, audience, and content preferences specified."
                if brand_brief
                else ""
            )
        ),
        HumanMessage(
            content=f"DURATION REQUIREMENT: Your improved cut must be between {min_duration:.0f} and {max_duration:.0f} seconds.\n"
            f"  - Completeness and fixing the critique issues are MORE important than hitting {target_duration:.0f}s exactly\n"
            f"  - If you need {max_duration:.0f} seconds to include all missing elements, use them\n"
            f"  - Don't cut important content just to stay close to {target_duration:.0f}s\n"
            f"Total video duration: {total_duration:.1f} seconds\n\n"
            f"ORIGINAL FULL TRANSCRIPT:\n{original_transcript}\n\n"
            f"INITIAL CUT PLAN (had {critique.issue_count} issues):\n{chr(10).join(initial_segments_info)}\n"
            f"Initial cut duration: {initial_total:.1f} seconds\n\n"
            f"CRITIQUE FEEDBACK:\n{critique.critique}\n\n"
            f"Word-level transcript (JSON):\n{json.dumps(words, indent=2)}\n\n"
            f"Based on the critique, create an IMPROVED cut plan that:\n"
            f"- Addresses ALL issues mentioned in the critique (this is the top priority)\n"
            f"- Is in chronological order\n"
            f"- Forms a coherent, flowing narrative\n"
            f"- Includes ALL key concepts/elements (if critique says something is missing, include it)\n"
            f"- Fulfills all promises made in the opening\n"
            f"- Has logical transitions between segments\n"
            f"- Total duration MUST be between {min_duration:.0f} and {max_duration:.0f} seconds (use the full range if needed)\n"
            f"- Uses the EXACT words from the transcript with their original timestamps"
            + (f"\n- Aligns with the brand brief provided" if brand_brief else "")
            + f"\n\nReturn the improved sections with words, start timestamp, and end timestamp.\n\n"
            f"Also provide a one-sentence executive summary (edit_summary) describing what was edited and why. "
            f"Be specific about what was removed (e.g., 'ramblings', 'retakes', 'verbose explanations', 'tangents') "
            f"and the goal (e.g., 'to create a focused 60-second video', 'to improve flow', 'to remove redundancy')."
        ),
    ]

    response = await structured_llm.ainvoke(messages)

    print("\n" + "=" * 60)
    print("IMPROVED CUT PLAN (after applying critique):")
    print("=" * 60)
    print(f"Keep sections: {len(response.sections)} sections")
    for i, section in enumerate(response.sections, 1):
        print(
            f"  Section {i}: {section.start:.2f}s - {section.end:.2f}s ({section.end - section.start:.1f}s)"
        )
        print(f"    Words: {' '.join(section.words)}")
    print(f"\nEdit Summary: {response.edit_summary}")
    print("=" * 60 + "\n")

    # Convert KeepSection objects to segment dicts and sort by start time
    segments = []
    for section in response.sections:
        segments.append(
            {
                "text": " ".join(section.words),
                "start": section.start,
                "end": section.end,
                "duration": section.end - section.start,
            }
        )

    # Sort segments by start time to ensure chronological order
    segments.sort(key=lambda x: x["start"])

    return segments, response.edit_summary


END_MARGIN_SECONDS = 0.1  # Extra padding at the end of each segment


def trim_video_segments(
    video_path: Path, output_path: Path, segments: list[dict]
) -> None:
    """Trim video to keep only specified word range segments and concatenate them."""
    if not segments:
        raise ValueError("No segments provided")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Extract each segment
        segment_files = []
        for i, seg in enumerate(segments):
            segment_file = tmp / f"segment_{i}.mp4"
            duration = seg["end"] - seg["start"] + END_MARGIN_SECONDS
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(video_path),
                    "-ss",
                    str(seg["start"]),
                    "-t",
                    str(duration),
                    "-c",
                    "copy",
                    str(segment_file),
                ],
                check=True,
                capture_output=True,
            )
            segment_files.append(segment_file)

        # Create concat file
        concat_file = tmp / "concat.txt"
        concat_content = "\n".join([f"file '{f.absolute()}'" for f in segment_files])
        concat_file.write_text(concat_content)

        # Concatenate segments
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Intelligently trim video to target duration using LLM"
    )
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument(
        "output", type=Path, nargs="?", help="Output video file (optional)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Target duration in seconds. If not provided, just 'tighten up' without duration constraint (default: None)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=20.0,
        help="Acceptable margin in seconds (e.g., 20 means ±20s, so 60s target = 40-80s range) (default: 20)",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        help="Path to word-level transcript JSON file (optional)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show cut plan without editing video",
    )

    args = parser.parse_args()

    video_path = args.video
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    target_duration = args.duration
    transcript_file = args.transcript
    dry_run = args.dry_run

    if args.output:
        output_path = args.output
    elif not dry_run:
        if target_duration:
            output_path = (
                video_path.parent / f"{video_path.stem}-{target_duration}s-smart.mp4"
            )
        else:
            output_path = video_path.parent / f"{video_path.stem}-tightened.mp4"
    else:
        output_path = None

    if transcript_file:
        print(f"Loading word-level transcript from: {transcript_file}")
        transcript, words = read_word_transcript_file(transcript_file)
    else:
        print("Analyzing video transcript...")
        transcript, words = await get_transcript_with_words(video_path)

    print(f"Found {len(words)} words")
    tolerance = args.tolerance

    # Step 1: Generate initial cut plan
    if target_duration:
        print(
            f"Finding sections to keep ({target_duration} seconds ±{tolerance}s) using LLM..."
        )
    else:
        print(
            "Tightening up video (removing mistakes, fumbles, retakes, ramblings, asides)..."
        )
    segments, edit_summary = await find_sections_to_keep(
        words, target_duration, tolerance
    )

    total_duration = sum(seg["duration"] for seg in segments)
    print(f"\n{'='*60}")
    print(
        f"INITIAL CUT PLAN: {len(segments)} segments totaling {total_duration:.1f} seconds"
    )
    print(f"{'='*60}")
    for i, seg in enumerate(segments, 1):
        text_preview = seg["text"]
        print(
            f"  Segment {i}: {seg['start']:.1f}s - {seg['end']:.1f}s ({seg['duration']:.1f}s)"
        )
        print(f"    Text: {text_preview}")
    print(f"{'='*60}\n")

    # Step 2: Critique the cuts (skip if no target duration - just tightening up)
    if target_duration is not None:
        print("Critiquing initial cut plan...")
        critique = await critique_cuts(transcript, segments, target_duration)
    else:
        # For tighten-up mode, skip critique (we're not targeting a duration)
        # Create a dummy critique with no issues
        critique = type("CritiqueResponse", (), {"issue_count": 0, "issues": []})()

    # Step 3: Apply critique to improve the cut plan (skip if no target duration)
    if target_duration is not None and critique.issue_count > 0:
        print(
            f"\n🔧 Applying critique to improve cut plan ({critique.issue_count} issues found)..."
        )
        segments, edit_summary = await apply_critique(
            words, transcript, segments, critique, target_duration, tolerance
        )

        total_duration = sum(seg["duration"] for seg in segments)
        print(f"\n{'='*60}")
        print(
            f"IMPROVED CUT PLAN: {len(segments)} segments totaling {total_duration:.1f} seconds"
        )
        print(f"{'='*60}")
        for i, seg in enumerate(segments, 1):
            text_preview = seg["text"]
            print(
                f"  Segment {i}: {seg['start']:.1f}s - {seg['end']:.1f}s ({seg['duration']:.1f}s)"
            )
            print(f"    Text: {text_preview}")
        print(f"{'='*60}\n")

        # Optional: Critique the improved version (but don't iterate again)
        if target_duration is not None:
            print("Critiquing improved cut plan...")
            final_critique = await critique_cuts(transcript, segments, target_duration)
            if final_critique.issue_count > 0:
                print(
                    f"⚠️  Improved cut still has {final_critique.issue_count} issues, but proceeding with best available cut."
                )
        else:
            print("✅ Tightened cut plan ready.")
    else:
        if target_duration is not None:
            print("✅ Initial cut plan has no issues, using it as-is.")
        else:
            print("✅ Tightened cut plan ready.")

    if dry_run:
        print("🔍 DRY RUN: No video edited. Remove --dry-run to apply cuts.")
        return

    print("Trimming and concatenating video segments...")
    trim_video_segments(video_path, output_path, segments)

    print(f"✅ Output video: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
