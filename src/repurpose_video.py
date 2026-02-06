#!/usr/bin/env python3
"""
Repurpose a long-form video (e.g., Zoom meeting) into multiple ~60 second segments.

Extracts clean, focused segments by:
1. Getting transcript with word-level timestamps
2. Using LLM to identify multiple ~60 second segments (removing ramblings, fillers, etc.)
3. Applying cuts to create multiple output videos
"""

import argparse
import asyncio
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from get_transcript import get_transcript, read_word_transcript_file
from trim_smart import DEFAULT_EDIT_PROMPT, trim_video_segments

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


class SegmentPlan(BaseModel):
    """A single ~60 second segment plan."""

    segment_number: int = Field(description="Segment number (1, 2, 3, etc.)")
    sections: list[WordRanges] = Field(
        description="List of sections to keep in this segment (will be concatenated)"
    )
    summary: str = Field(
        description="Brief summary of what this segment covers (e.g., 'Introduction to topic X and key concept Y')"
    )


class RepurposePlan(BaseModel):
    """Response model for multiple segments to extract."""

    segments: list[SegmentPlan] = Field(
        description="List of segment plans, each targeting ~60 seconds"
    )
    overall_summary: str = Field(
        description="Overall summary of how the video was repurposed into segments"
    )


async def find_segments_to_extract(
    words: list[dict],
    target_duration: float = 60.0,
    tolerance: float = 20.0,
    custom_prompt: str | None = None,
    max_segments: int | None = None,
) -> list[dict]:
    """Use LLM to identify multiple segments to extract from the word-level transcript.

    Args:
        words: List of word dicts with 'word', 'start', 'end'
        target_duration: Target duration per segment in seconds (default: 60.0)
        tolerance: Acceptable margin in seconds per segment (e.g., 20 means ±20s, so 60s target = 40-80s range)
        custom_prompt: Optional custom edit prompt. If None, uses DEFAULT_EDIT_PROMPT.
        max_segments: Maximum number of segments to extract. If None, extracts as many as possible.

    Returns:
        List of segment dicts, each with 'segment_number', 'sections' (list of segment dicts), 'summary'
    """
    import json

    print(f"🔧 DEBUG: Starting find_segments_to_extract")
    print(f"🔧 DEBUG: Input words count: {len(words)}")
    print(f"🔧 DEBUG: Target duration: {target_duration}s, tolerance: {tolerance}s")
    print(f"🔧 DEBUG: Max segments: {max_segments}")

    # Use custom prompt or default
    edit_prompt = custom_prompt if custom_prompt else DEFAULT_EDIT_PROMPT
    print(f"🔧 DEBUG: Using edit prompt: {edit_prompt[:100]}...")

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")
    print("🔧 DEBUG: OpenRouter API key found")

    print("🔧 DEBUG: Initializing LLM client...")
    llm_start = time.time()
    llm = ChatOpenAI(
        model="google/gemini-3-flash-preview",
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        timeout=180.0,  # Increased to 3 minutes for reasoning
        extra_body={
            "reasoning": {"max_tokens": 1000, "enabled": True}
        },  # Reduced from 2000
    )
    print(f"🔧 DEBUG: LLM client initialized ({time.time() - llm_start:.2f}s)")

    print("🔧 DEBUG: Creating structured LLM wrapper...")
    structured_llm = llm.with_structured_output(RepurposePlan)
    print("🔧 DEBUG: Structured LLM wrapper created")

    # Calculate total video duration
    total_duration = words[-1]["end"] if words else 0
    print(f"🔧 DEBUG: Total video duration: {total_duration:.1f}s")

    # Calculate acceptable range per segment
    min_duration = max(0, target_duration - tolerance)
    max_duration = target_duration + tolerance
    print(
        f"🔧 DEBUG: Segment duration range: {min_duration:.0f}s - {max_duration:.0f}s"
    )

    # Estimate how many segments we can extract
    if max_segments:
        estimated_segments = max_segments
        segment_instruction = f"Extract exactly {max_segments} clean, focused segments."
    else:
        estimated_segments = max(1, int(total_duration / target_duration))
        segment_instruction = (
            f"Extract {estimated_segments} or more clean, focused segments."
        )
    print(f"🔧 DEBUG: Estimated segments: {estimated_segments}")

    print("🔧 DEBUG: Building prompt messages...")
    prompt_start = time.time()
    messages = [
        SystemMessage(
            content="You are a professional video editor repurposing a long-form video into multiple short-form segments.\n\n"
            "YOUR TASK:\n"
            f"{segment_instruction}\n"
            f"Each segment should be approximately {target_duration} seconds (±{tolerance}s tolerance).\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "1. Each segment must be self-contained and coherent on its own\n"
            "2. Segments should cover different topics/themes from the video\n"
            "3. Sections within each segment MUST be in chronological order\n"
            "4. Each section must be a complete thought or phrase - never cut mid-sentence\n"
            "5. Sections must flow logically from one to the next\n"
            "6. Use the EXACT words and timestamps from the transcript\n"
            f"7. Each segment's total duration must be between {min_duration:.0f} and {max_duration:.0f} seconds\n\n"
            "WHAT TO CUT:\n"
            "- Ramblings/asides/tangents (off-topic digressions)\n"
            "- Retakes/fumbles (obvious mistakes, false starts, repeated attempts)\n"
            "- Filler words (uh, um, etc.)\n"
            "- Verbose explanations when brief mentions suffice\n"
            "- Repetitive content\n\n"
            "WHAT NOT TO CUT:\n"
            "- Core narrative (the main story/argument)\n"
            "- Key concepts and definitions\n"
            "- Transitions that provide necessary context\n"
            "- Anything that would make a segment not make coherent logical sense\n\n"
            "SEGMENT ORGANIZATION:\n"
            "- Each segment should focus on a distinct topic or theme\n"
            "- Segments can overlap slightly in content if needed for coherence\n"
            "- Try to extract as many segments as possible while maintaining quality\n"
            "- Each segment should be watchable independently\n"
        ),
        HumanMessage(
            content=(
                f"Apply these edits to the video: {edit_prompt}\n\n"
                f"Total video duration: {total_duration:.1f} seconds\n"
                f"Target: Extract multiple segments, each approximately {target_duration} seconds (±{tolerance}s tolerance).\n\n"
                f"Word-level transcript (JSON):\n{json.dumps(words, indent=2)}\n\n"
                f"Identify multiple segments to extract. Each segment should:\n"
                f"- Be self-contained and coherent on its own\n"
                f"- Cover a distinct topic or theme\n"
                f"- Have sections in chronological order\n"
                f"- Form a coherent, flowing narrative when read together\n"
                f"- Total duration between {min_duration:.0f} and {max_duration:.0f} seconds\n"
                f"- Use the EXACT words from the transcript with their original timestamps\n\n"
                + (
                    f"Return exactly {max_segments} segments.\n"
                    if max_segments
                    else "Return multiple segments (as many as you can extract while maintaining quality).\n"
                )
                + f"For each segment, provide:\n"
                + f"- Segment number\n"
                + f"- List of sections (with words, start timestamp, end timestamp)\n"
                + f"- Brief summary of what the segment covers\n\n"
                + f"Also provide an overall summary of how the video was repurposed."
            )
        ),
    ]
    print(f"🔧 DEBUG: Prompt messages built ({time.time() - prompt_start:.2f}s)")
    print(f"🔧 DEBUG: Prompt size: {len(str(messages))} characters")

    print("🔧 DEBUG: Calling LLM (this may take 1-3 minutes with reasoning enabled)...")
    llm_call_start = time.time()
    try:
        response = await structured_llm.ainvoke(messages)
        llm_call_duration = time.time() - llm_call_start
        print(f"🔧 DEBUG: LLM call completed successfully ({llm_call_duration:.1f}s)")
    except Exception as e:
        llm_call_duration = time.time() - llm_call_start
        print(f"❌ ERROR: LLM call failed after {llm_call_duration:.1f}s")
        print(f"❌ ERROR: Exception type: {type(e).__name__}")
        print(f"❌ ERROR: Exception message: {str(e)}")
        import traceback

        print(f"❌ ERROR: Traceback:\n{traceback.format_exc()}")
        raise

    print("\n" + "=" * 60)
    print("REPURPOSE PLAN:")
    print("=" * 60)
    print(f"Overall summary: {response.overall_summary}")
    print(f"\nExtracted {len(response.segments)} segments:")
    for seg_plan in response.segments:
        total_dur = sum(section.end - section.start for section in seg_plan.sections)
        print(f"\n  Segment {seg_plan.segment_number}: {total_dur:.1f}s")
        print(f"    Summary: {seg_plan.summary}")
        for i, section in enumerate(seg_plan.sections, 1):
            print(
                f"      Section {i}: {section.start:.2f}s - {section.end:.2f}s ({section.end - section.start:.1f}s)"
            )
            print(f"        Words: {' '.join(section.words[:20])}...")
    print("=" * 60 + "\n")

    # Convert to list of segment dicts
    segments_list = []
    for seg_plan in response.segments:
        # Convert sections to segment dicts
        segment_sections = []
        for section in seg_plan.sections:
            segment_sections.append(
                {
                    "text": " ".join(section.words),
                    "start": section.start,
                    "end": section.end,
                    "duration": section.end - section.start,
                }
            )
        # Sort sections by start time
        segment_sections.sort(key=lambda x: x["start"])

        segments_list.append(
            {
                "segment_number": seg_plan.segment_number,
                "sections": segment_sections,
                "summary": seg_plan.summary,
            }
        )

    return segments_list


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Repurpose a long-form video into multiple ~60 second segments"
    )
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory where segment videos will be saved",
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
        help="Acceptable margin in seconds per segment (default: 20)",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        help="Path to word-level transcript JSON file (optional)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom edit prompt. If not provided, uses default prompt.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of segments to extract (default: extract as many as possible)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show segment plan without creating videos",
    )

    args = parser.parse_args()

    video_path = args.video
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    target_duration = args.duration
    transcript_file = args.transcript
    dry_run = args.dry_run
    custom_prompt = args.prompt
    max_segments = args.count

    # Start timing
    start_time = time.time()

    if transcript_file:
        print(f"📝 Loading word-level transcript from: {transcript_file}")
        transcript_load_start = time.time()
        transcript, words = read_word_transcript_file(transcript_file)
        transcript_load_duration = time.time() - transcript_load_start
        print(
            f"✅ Transcribed: {len(words)} words (loaded in {transcript_load_duration:.2f}s)"
        )
    else:
        print("📝 Transcribing video...")
        transcript_start = time.time()
        from trim_smart import get_transcript_with_words

        try:
            transcript, words = await get_transcript_with_words(video_path)
            transcript_duration = time.time() - transcript_start
            print(
                f"✅ Transcribed: {len(words)} words (took {transcript_duration:.2f}s)"
            )
        except Exception as e:
            transcript_duration = time.time() - transcript_start
            print(f"❌ ERROR: Transcription failed after {transcript_duration:.2f}s")
            print(f"❌ ERROR: Exception type: {type(e).__name__}")
            print(f"❌ ERROR: Exception message: {str(e)}")
            import traceback

            print(f"❌ ERROR: Traceback:\n{traceback.format_exc()}")
            raise
    tolerance = args.tolerance

    if custom_prompt:
        print(f"Using custom edit prompt: {custom_prompt}")
    if max_segments:
        print(f"Extracting {max_segments} segments")

    # Find segments to extract
    print(
        f"\n🔍 Finding segments to extract ({target_duration}s ±{tolerance}s per segment)..."
    )
    find_segments_start = time.time()
    try:
        segments_list = await find_segments_to_extract(
            words, target_duration, tolerance, custom_prompt, max_segments
        )
        find_segments_duration = time.time() - find_segments_start
        print(
            f"✅ Found {len(segments_list)} segments (took {find_segments_duration:.1f}s)"
        )
    except Exception as e:
        find_segments_duration = time.time() - find_segments_start
        print(f"❌ ERROR: Finding segments failed after {find_segments_duration:.1f}s")
        print(f"❌ ERROR: Exception type: {type(e).__name__}")
        print(f"❌ ERROR: Exception message: {str(e)}")
        import traceback

        print(f"❌ ERROR: Traceback:\n{traceback.format_exc()}")
        raise

    # Collect metadata for markdown (do this whether dry-run or not)
    print(f"\n📝 Generating titles and metadata for {len(segments_list)} segments...")
    metadata_start = time.time()
    segment_metadata = []

    for i, seg_data in enumerate(segments_list, 1):
        print(f"  Generating title for segment {i}/{len(segments_list)}...", end="\r")
        segment_num = seg_data["segment_number"]
        sections = seg_data["sections"]
        summary = seg_data["summary"]

        # Generate output filename
        output_filename = f"segment_{segment_num:02d}.mp4"
        output_path = output_dir / output_filename

        # Get full transcript text for this segment
        segment_text = " ".join([sec["text"] for sec in sections])

        # Generate title from segment text
        from add_title import generate_title_from_transcript

        title = await generate_title_from_transcript(segment_text, count=1)

        # Generate short caption (first 100 chars of summary or transcript)
        caption = summary[:100] if len(summary) <= 100 else summary[:97] + "..."

        # Shortened transcript (first 300 words)
        words_list = segment_text.split()
        shortened_transcript = " ".join(words_list[:300])
        if len(words_list) > 300:
            shortened_transcript += "..."

        total_dur = sum(sec["duration"] for sec in sections)

        segment_metadata.append(
            {
                "segment_number": segment_num,
                "title": title,
                "summary": summary,
                "caption": caption,
                "shortened_transcript": shortened_transcript,
                "video_path": str(output_path),
                "duration": total_dur,
            }
        )
    metadata_duration = time.time() - metadata_start
    print(
        f"✅ Generated metadata for {len(segments_list)} segments (took {metadata_duration:.1f}s)"
    )

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

    # Generate markdown document (always, even in dry-run)
    markdown_path = output_dir / "segments.md"
    with open(markdown_path, "w") as f:
        f.write("# Repurposed Video Segments\n\n")
        f.write(f"Generated from: `{video_path.name}`\n")
        f.write(f"Total segments: {len(segments_list)}\n")
        f.write(f"Time: {time_str}\n")
        if dry_run:
            f.write("**DRY RUN - No videos created**\n")
        f.write("\n---\n\n")

        for meta in segment_metadata:
            f.write(f"## Segment {meta['segment_number']}: {meta['title']}\n\n")
            f.write(f"**Summary:** {meta['summary']}\n\n")
            f.write(f"**Caption:** {meta['caption']}\n\n")
            f.write(f"**Duration:** {meta['duration']:.1f}s\n\n")
            f.write(f"**Video:** `{meta['video_path']}`\n\n")
            f.write(f"**Transcript:**\n{meta['shortened_transcript']}\n\n")
            f.write("---\n\n")

    print(f"📄 Markdown document: {markdown_path}")

    if dry_run:
        print(
            "\n🔍 DRY RUN: No videos created. Remove --dry-run to create segment videos."
        )
        return

    # Create output videos for each segment
    print(f"\nCreating {len(segments_list)} segment videos...")
    for seg_data in segments_list:
        segment_num = seg_data["segment_number"]
        sections = seg_data["sections"]
        summary = seg_data["summary"]

        # Generate output filename
        output_filename = f"segment_{segment_num:02d}.mp4"
        output_path = output_dir / output_filename

        print(f"\nCreating segment {segment_num}: {output_filename}")
        print(f"  Summary: {summary}")
        total_dur = sum(sec["duration"] for sec in sections)
        print(f"  Duration: {total_dur:.1f}s")

        # Apply cuts to create this segment
        trim_video_segments(video_path, output_path, sections)
        print(f"  ✅ Created: {output_path}")

    print(
        f"\n✅ Repurposing complete! Created {len(segments_list)} segments in {output_dir}"
    )


if __name__ == "__main__":
    asyncio.run(main())
