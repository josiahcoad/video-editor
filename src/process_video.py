#!/usr/bin/env python3
"""
Process a video through the complete editing pipeline using Prefect.

Features:
- Human-in-the-loop: Select title and music BEFORE processing
- Caching: Transcripts are cached to avoid re-processing
- State tracking: Each step shows progress in Prefect UI
- Assets: Video/transcript outputs tracked
- Artifacts: Cut plans, thumbnails, progress reports
- Retries: Automatic retries with backoff
- UI: Visual workflow tracking

Pipeline steps:
1. Quick transcript (for title generation)
2. Generate title options → USER SELECTS
3. Search music options → USER SELECTS
4. Remove silence
5. Fix rotation (if MOV)
6. Remove filler words
7. Smart trim
8. Enhance voice
9. Add subtitles
10. Add music (using selected track)
11. Add title (using selected title)
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from prefect import flow, task
from prefect.artifacts import (
    create_markdown_artifact,
    create_table_artifact,
    create_link_artifact,
)
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.logging import get_run_logger

from add_background_music import add_music, download_music, search_music
from add_subtitles import add_srt as generate_srt_content, add_srt_from_utterances
from add_title import add_title
from duration_tracker import estimate_duration, record_run
from enhance_voice import enhance_voice
from get_transcript import get_transcript, read_word_transcript_file
from remove_filler_words import (
    find_filler_word_segments,
    load_words_from_transcript,
    remove_segments_from_video,
)
from review_video import review_video
from trim_smart import (
    apply_critique,
    critique_cuts,
    find_sections_to_keep,
    trim_video_segments,
)


# =============================================================================
# Helper Functions
# =============================================================================


def run_make_command(cmd: str, input_path: Path, output_path: Path) -> None:
    """Run a makefile command."""
    subprocess.run(
        ["make", cmd, f"INPUT={input_path}", f"OUTPUT={output_path}"],
        cwd=Path(__file__).parent,
        check=True,
    )


def _get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _calculate_wpm(video_duration_seconds: float, word_count: int) -> float:
    """Calculate words per minute from video duration and word count."""
    if video_duration_seconds <= 0:
        return 0.0
    duration_minutes = video_duration_seconds / 60.0
    return word_count / duration_minutes if duration_minutes > 0 else 0.0


def _calculate_smart_speedup(
    video_duration_seconds: float,
    word_count: int,
    target_wpm: float = 180.0,
    min_wpm: float = 170.0,
) -> float:
    """Calculate speedup to target WPM, but skip if already above min_wpm.

    Args:
        video_duration_seconds: Current video duration
        word_count: Number of words in transcript
        target_wpm: Target WPM (default: 180)
        min_wpm: Minimum WPM threshold - if current WPM >= this, skip speedup (default: 170)

    Returns:
        Speedup multiplier (1.0 if skipping, otherwise calculated to reach target_wpm)
    """
    current_wpm = _calculate_wpm(video_duration_seconds, word_count)

    if current_wpm >= min_wpm:
        # Already fast enough, skip speedup
        return 1.0

    # Calculate speedup needed to reach target_wpm
    # target_wpm = word_count / (duration_minutes / speedup)
    # target_wpm = word_count * speedup / duration_minutes
    # speedup = target_wpm * duration_minutes / word_count
    # speedup = target_wpm / current_wpm
    speedup = target_wpm / current_wpm

    return speedup


def _extract_first_frame(video_path: Path, output_path: Path) -> Path:
    """Extract first frame as thumbnail."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    return output_path


# =============================================================================
# Auto-positioning: Determine optimal caption/title positions
# =============================================================================


@task(
    retries=1,
    timeout_seconds=30,
    log_prints=True,
)
async def determine_optimal_positions(video_path: Path) -> tuple[int, int]:
    """Extract first frame and use Gemini 3 to determine optimal caption and title positions.

    Returns:
        Tuple of (caption_height, title_height) where both are 0-100 (0 = bottom, 100 = top)
    """
    logger = get_run_logger()
    logger.info("Determining optimal caption and title positions...")

    # Extract first frame efficiently using ffmpeg
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        frame_path = Path(tmp_file.name)

    try:
        # Extract first frame - this is very fast (just one frame, no encoding)
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(video_path),
                "-vf",
                "select=eq(n\\,0)",
                "-frames:v",
                "1",
                "-q:v",
                "2",  # High quality JPEG
                "-y",
                str(frame_path),
            ],
            check=True,
            capture_output=True,
        )

        # Read frame as base64
        import base64

        with open(frame_path, "rb") as f:
            frame_data = base64.b64encode(f.read()).decode("utf-8")

        # Use Gemini 3 to analyze frame
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        from pydantic import BaseModel, Field as PydanticField

        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        class PositionResponse(BaseModel):
            caption_height: int = PydanticField(
                description="Optimal caption position (0-100, where 0 is bottom, 100 is top)",
                ge=0,
                le=100,
            )
            title_height: int = PydanticField(
                description="Optimal title position (0-100, where 0 is bottom, 100 is top)",
                ge=0,
                le=100,
            )

        llm = ChatOpenAI(
            model="google/gemini-3-flash-preview",
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            temperature=0.3,
        ).with_structured_output(PositionResponse)

        response = await llm.ainvoke(
            [
                SystemMessage(
                    content="You are a video editor expert. Analyze the first frame of a video and determine optimal positions for captions and titles.\n\n"
                    "GUIDELINES:\n"
                    "- Captions should be placed where they won't obscure important visual content (faces, text, key objects)\n"
                    "- Titles should be placed at the top or center, avoiding important visual elements\n"
                    "- Use 0-100 scale where 0 = bottom of screen, 100 = top of screen\n"
                    "- IMPORTANT: The number represents distance FROM BOTTOM. So 15 means 15% from bottom (near bottom), 85 means 85% from bottom (near top)\n"
                    "- For captions: typically 10-20 (near bottom, 10-20% from bottom) or 80-90 (near top, 80-90% from bottom) depending on where faces/action are\n"
                    "- For titles: typically 80-90 (near top, 80-90% from bottom) or 40-60 (center, 40-60% from bottom) depending on composition\n"
                    "- Avoid placing text over faces, important text, or key visual elements\n"
                    "- If faces are in the center/middle, place captions at bottom (10-20) or top (80-90)"
                ),
                HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{frame_data}",
                            },
                        },
                        {
                            "type": "text",
                            "text": "Analyze this first frame and determine the optimal positions for captions and title. "
                            "Return caption_height and title_height as integers from 0-100, where the number represents the percentage FROM THE BOTTOM of the screen. "
                            "For example: 15 = 15% from bottom (near bottom), 85 = 85% from bottom (near top).",
                        },
                    ]
                ),
            ]
        )

        logger.info(
            f"Optimal positions determined: captions at {response.caption_height}%, title at {response.title_height}%"
        )

        return response.caption_height, response.title_height

    finally:
        # Clean up temp file
        if frame_path.exists():
            frame_path.unlink()


# =============================================================================
# Phase 1: User Input Tasks (run first, require human selection)
# =============================================================================


@task(
    retries=1,
    timeout_seconds=120,
    log_prints=True,
)
async def quick_transcript_task(video_path: Path) -> str:
    """Get a quick transcript for title generation."""
    logger = get_run_logger()
    logger.info("Getting quick transcript for title generation...")

    result = await get_transcript(video_path)
    transcript = " ".join([w["word"] for w in result["words"]])

    logger.info(f"Transcript ready ({len(result['words'])} words)")
    return transcript


@task(
    retries=2,
    timeout_seconds=60,
    log_prints=True,
)
async def generate_title_options_task(
    transcript: str, brand_brief: str | None = None
) -> list[str]:
    """Generate 3 title options from transcript.

    Args:
        transcript: Video transcript
        brand_brief: Optional brand brief to guide title generation (e.g., brand voice, audience, words to avoid)
    """
    logger = get_run_logger()
    logger.info("Generating title options...")

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import BaseModel, Field as PydanticField

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    class TitleOptions(BaseModel):
        titles: list[str] = PydanticField(description="3 title options for the video")

    llm = ChatOpenAI(
        model="google/gemini-3-flash-preview",
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        temperature=0.7,
    ).with_structured_output(TitleOptions)

    system_content = """You are a social media expert specializing in viral, engaging titles.

Generate 3 title options that are:
- **Hook-driven**: Start with curiosity gaps, questions, or specific problems
- **Clear and specific**: Avoid vague words like "better", "improve", "transform"
- **Relatable**: Use conversational language ("feels", "seems", "actually")
- **Promise-driven**: Hint at a solution or insight ("And how to fix it", "The secret to...")
- **Pattern-aware**: Consider formats like:
  * Question format: "Why [problem] (And how to fix it)"
  * Problem/solution: "[Problem] - Here's what actually works"
  * Specific insight: "The [number] [thing] that [outcome]"

Examples of strong titles:
- "Why your social media feels random (And how to fix it)"
- "The 4-step framework that doubled my engagement"
- "Social media consistency isn't what you think"

Examples of weak titles to avoid:
- "Boost your social media success" (too vague)
- "Transform your social media strategy" (generic, overused words)
- "How to improve social media" (lacks specificity)"""

    if brand_brief:
        system_content += f"\n\nBRAND BRIEF:\n{brand_brief}\n\nFollow the brand brief strictly. If it specifies words to avoid (like 'success', 'transform', 'delve'), do NOT use those words in any title option."

    human_content = (
        f"Generate 3 title options for this video transcript:\n\n{transcript[:2000]}"
    )
    if brand_brief:
        human_content += f"\n\nRemember: Follow the brand brief and avoid any words specified as disliked. Focus on hooks, specificity, and relatability."

    response = await llm.ainvoke(
        [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content),
        ]
    )

    titles = response.titles[:3]
    logger.info(f"Generated {len(titles)} title options")

    # Create artifact showing options
    await create_table_artifact(
        key="title-options",
        table=[{"Option": i + 1, "Title": t} for i, t in enumerate(titles)],
        description="Title options for user selection",
    )

    return titles


@task(
    retries=2,
    timeout_seconds=60,
    log_prints=True,
)
async def search_music_options_task(tags: str) -> list[dict]:
    """Search for 3 music options."""
    logger = get_run_logger()
    logger.info(f"Searching for music: {tags}")

    tracks = await search_music(tags, count=3)

    logger.info(f"Found {len(tracks)} music options")

    # Create artifact showing options (include URL for CLI selection)
    await create_table_artifact(
        key="music-options",
        table=[
            {
                "Option": i + 1,
                "Title": t["title"],
                "Artist": t["creator"],
                "License": t["license"],
                "URL": t["url"],
            }
            for i, t in enumerate(tracks)
        ],
        description="Music options for user selection",
    )

    return tracks


# =============================================================================
# Phase 2: Processing Tasks (run after user selection)
# =============================================================================


def _find_speech_segments_from_words(
    words: list[dict], gap_threshold: float = 0.5, margin: float = 0.15
) -> list[tuple[float, float]]:
    """Find speech segments from word-level transcript.

    Groups consecutive words into segments, splitting where gaps exceed threshold.

    Args:
        words: List of word dicts with 'start' and 'end' keys
        gap_threshold: Minimum gap between words to consider a "silence" (default: 0.5s)
        margin: Buffer to add before/after each segment (default: 0.15s)

    Returns:
        List of (start, end) tuples representing speech segments
    """
    if not words:
        return []

    segments: list[tuple[float, float]] = []
    segment_start = max(0.0, words[0]["start"] - margin)
    segment_end = words[0]["end"]

    for i in range(1, len(words)):
        prev_end = words[i - 1]["end"]
        curr_start = words[i]["start"]
        gap = curr_start - prev_end

        if gap > gap_threshold:
            # End current segment, start new one
            segments.append((segment_start, segment_end + margin))
            segment_start = curr_start - margin

        segment_end = words[i]["end"]

    # Add final segment
    segments.append((segment_start, segment_end + margin))

    return segments


@task(
    retries=2,
    retry_delay_seconds=10,
    timeout_seconds=600,  # Re-encoding with hardware accel
    log_prints=True,
)
async def remove_silence_task(
    input_video: Path,
    output_video: Path,
    margin: float = 0.15,
    words: list[dict] | None = None,
    gap_threshold: float = 0.5,
) -> Path:
    """Remove silence from video using transcript word boundaries.

    Uses word-level transcript to find natural pause points, then stream copy
    for fast cutting. Results in fewer, larger segments than audio-based detection.

    Args:
        input_video: Input video file
        output_video: Output video file
        margin: Buffer in seconds to keep before/after speech (default: 0.15)
        words: Word-level transcript (list of dicts with start/end). If None, transcribes.
        gap_threshold: Minimum gap between words to cut (default: 0.5s)
    """
    import tempfile

    logger = get_run_logger()
    logger.info(
        f"Removing silence from {input_video.name} (gap_threshold={gap_threshold}s)"
    )

    # Step 1: Get transcript if not provided
    if words is None:
        logger.info("Transcribing video to find speech boundaries...")
        from get_transcript import get_transcript

        result = await get_transcript(input_video)
        words = result["words"]
        logger.info(f"Transcribed {len(words)} words")

    if not words:
        logger.info("No speech detected, copying video as-is")
        subprocess.run(["cp", str(input_video), str(output_video)], check=True)
        return output_video

    # Step 2: Find speech segments (groups of words with gaps < threshold)
    segments_to_keep = _find_speech_segments_from_words(words, gap_threshold, margin)

    if not segments_to_keep:
        logger.info("No speech segments found, copying video as-is")
        subprocess.run(["cp", str(input_video), str(output_video)], check=True)
        return output_video

    logger.info(f"Found {len(segments_to_keep)} speech segments to keep")

    # Log segment info
    total_speech = sum(end - start for start, end in segments_to_keep)
    video_duration = _get_video_duration(input_video)
    logger.info(
        f"Speech: {total_speech:.1f}s of {video_duration:.1f}s ({total_speech/video_duration*100:.1f}%)"
    )

    # Step 3: Use single-pass trim/concat filter for perfect A/V sync
    # This re-encodes but with hardware acceleration and fewer segments it's fast

    # Build video and audio filter chains
    video_parts = []
    audio_parts = []
    for i, (start, end) in enumerate(segments_to_keep):
        video_parts.append(
            f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}]"
        )
        audio_parts.append(
            f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}]"
        )

    # Concat all segments
    n = len(segments_to_keep)
    video_concat_inputs = "".join(f"[v{i}]" for i in range(n))
    audio_concat_inputs = "".join(f"[a{i}]" for i in range(n))
    concat_filter = f"{video_concat_inputs}concat=n={n}:v=1:a=0[vout];{audio_concat_inputs}concat=n={n}:v=0:a=1[aout]"

    filter_complex = ";".join(video_parts + audio_parts) + ";" + concat_filter

    logger.info(
        f"Processing with trim/concat filter ({n} segments, hardware encoding)..."
    )

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

    subprocess.run(cmd, check=True, capture_output=True)

    output_duration = _get_video_duration(output_video)
    reduction = (1 - output_duration / video_duration) * 100

    logger.info(
        f"Duration: {video_duration:.1f}s → {output_duration:.1f}s ({reduction:.1f}% reduced)"
    )

    return output_video


@task(
    retries=1,
    retry_delay_seconds=5,
    timeout_seconds=600,
    cache_policy=INPUTS + TASK_SOURCE,
    log_prints=True,
)
async def get_transcript_task(video_path: Path, output_words: Path) -> Path:
    """Get word-level transcript from video (cached).

    Also saves utterance-level transcript for punctuation-aware subtitles.
    """
    logger = get_run_logger()
    logger.info(f"Transcribing {video_path.name}")

    result = await get_transcript(video_path)

    # Ensure output directory exists
    output_words.parent.mkdir(parents=True, exist_ok=True)
    output_words.write_text(json.dumps(result["words"], indent=2))

    # Also save utterance file if available (for punctuation-aware subtitles)
    if result.get("utterances"):
        output_utterances = (
            output_words.parent
            / f"{output_words.stem.replace('-words', '')}-utterances.json"
        )
        output_utterances.write_text(json.dumps(result["utterances"], indent=2))
        logger.info(f"Saved {len(result['utterances'])} utterances")

    logger.info(f"Transcribed {len(result['words'])} words")
    return output_words


@task(
    retries=1,
    timeout_seconds=600,
    log_prints=True,
)
def fix_rotation_task(input_video: Path, output_video: Path, is_mov: bool) -> Path:
    """Fix video rotation if it's a MOV file."""
    logger = get_run_logger()

    # Rotation fix disabled for now - just copy the file
    logger.info("Rotation fix disabled, copying video as-is")
    subprocess.run(["cp", str(input_video), str(output_video)], check=True)

    return output_video


@task(
    retries=1,
    timeout_seconds=600,
    log_prints=True,
)
def crop_video_task(
    input_video: Path,
    output_video: Path,
    crop_top: int,
    crop_bottom: int,
    crop_left: int,
    crop_right: int,
) -> Path:
    """Crop video by percentage from each edge.

    Args:
        crop_top: Percentage to crop from top (0-50)
        crop_bottom: Percentage to crop from bottom (0-50)
        crop_left: Percentage to crop from left (0-50)
        crop_right: Percentage to crop from right (0-50)
    """
    logger = get_run_logger()

    # If no cropping needed, just copy
    if crop_top == 0 and crop_bottom == 0 and crop_left == 0 and crop_right == 0:
        logger.info("No crop specified, copying video as-is")
        subprocess.run(["cp", str(input_video), str(output_video)], check=True)
        return output_video

    logger.info(
        f"Cropping video: top={crop_top}%, bottom={crop_bottom}%, left={crop_left}%, right={crop_right}%"
    )

    # Get video dimensions
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(input_video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    width, height = map(int, result.stdout.strip().split(","))

    # Calculate crop dimensions
    crop_x = crop_left * width // 100
    crop_y = crop_top * height // 100
    crop_w = width - (crop_left * width // 100) - (crop_right * width // 100)
    crop_h = height - (crop_top * height // 100) - (crop_bottom * height // 100)

    # Ensure even dimensions (required by many codecs)
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)

    logger.info(
        f"Crop filter: {width}x{height} -> {crop_w}x{crop_h} (x={crop_x}, y={crop_y})"
    )

    # Apply crop with hardware encoding
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hwaccel",
            "videotoolbox",
            "-i",
            str(input_video),
            "-vf",
            f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}",
            "-c:v",
            "h264_videotoolbox",
            "-b:v",
            "8M",
            "-c:a",
            "copy",
            str(output_video),
        ],
        check=True,
        capture_output=True,
    )

    logger.info(f"Cropped video saved to {output_video}")
    return output_video


@task(
    retries=1,
    timeout_seconds=300,
    log_prints=True,
)
def speedup_video_task(input_video: Path, output_video: Path, speedup: float) -> Path:
    """Apply speedup to video and audio."""
    logger = get_run_logger()

    if speedup <= 1.0:
        logger.info(f"Speedup {speedup} <= 1.0, copying video as-is")
        subprocess.run(["cp", str(input_video), str(output_video)], check=True)
        return output_video

    logger.info(f"Applying {speedup}x speedup to video")

    # Video filter: setpts=PTS/speedup
    video_filter = f"setpts=PTS/{speedup}"

    # Audio filter: atempo (supports 0.5-2.0 range)
    # If speedup > 2.0, chain multiple atempo filters
    audio_filters = []
    remaining_speedup = speedup

    while remaining_speedup > 2.0:
        audio_filters.append("atempo=2.0")
        remaining_speedup /= 2.0

    if remaining_speedup > 1.0:
        audio_filters.append(f"atempo={remaining_speedup:.3f}")
    elif remaining_speedup < 1.0:
        # If we somehow end up with < 1.0, use minimum
        audio_filters.append("atempo=0.5")

    audio_filter = ",".join(audio_filters) if audio_filters else "anull"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-vf",
            video_filter,
            "-af",
            audio_filter,
            "-c:v",
            "h264_videotoolbox",
            "-c:a",
            "aac",
            "-b:v",
            "5M",
            "-b:a",
            "192k",
            str(output_video),
        ],
        check=True,
        capture_output=True,
    )

    logger.info(f"Speedup applied: {speedup}x")
    return output_video


@task(
    retries=1,
    timeout_seconds=300,
    log_prints=True,
)
async def remove_filler_words_task(
    input_video: Path, output_video: Path, transcript_path: Path
) -> Path:
    """Remove filler words from video."""
    logger = get_run_logger()

    words = load_words_from_transcript(transcript_path)
    segments_to_remove = find_filler_word_segments(words)

    if segments_to_remove:
        logger.info(f"Removing {len(segments_to_remove)} filler segments")
    else:
        logger.info("No filler words detected")

    remove_segments_from_video(input_video, output_video, segments_to_remove)
    return output_video


@task(
    retries=2,
    timeout_seconds=300,
    log_prints=True,
)
async def smart_trim_task(
    input_video: Path,
    output_video: Path,
    transcript_path: Path,
    target_duration: int,
    tolerance: float = 20.0,
    brand_brief: str | None = None,
) -> tuple[Path, str]:
    """Smart trim video to target duration using LLM with iterative improvement.

    Returns:
        Tuple of (output_video_path, edit_summary)
    """
    logger = get_run_logger()
    logger.info(f"Smart trimming to {target_duration}s (±{tolerance}s tolerance)")

    transcript_text, words_data = read_word_transcript_file(transcript_path)

    # Step 1: Generate initial cut plan
    logger.info("Generating initial cut plan...")
    segments_to_keep, edit_summary = await find_sections_to_keep(
        words_data,
        target_duration=float(target_duration),
        tolerance=tolerance,
        brand_brief=brand_brief,
    )

    # Step 2: Critique the cuts
    logger.info("Critiquing initial cut plan...")
    critique = await critique_cuts(
        transcript_text,
        segments_to_keep,
        float(target_duration),
        brand_brief=brand_brief,
    )

    # Step 3: Apply critique to improve if there are issues
    if critique.issue_count > 0:
        logger.info(
            f"Applying critique to improve cut plan ({critique.issue_count} issues found)..."
        )
        segments_to_keep, edit_summary = await apply_critique(
            words_data,
            transcript_text,
            segments_to_keep,
            critique,
            float(target_duration),
            tolerance,
            brand_brief=brand_brief,
        )

        # Optional: Final critique check
        final_critique = await critique_cuts(
            transcript_text,
            segments_to_keep,
            float(target_duration),
            brand_brief=brand_brief,
        )
        if final_critique.issue_count > 0:
            logger.warning(
                f"Improved cut still has {final_critique.issue_count} issues, but proceeding with best available cut."
            )
    else:
        logger.info("Initial cut plan has no issues, using it as-is.")

    # Create cut plan artifact
    cut_plan_data = []
    for i, seg in enumerate(segments_to_keep, 1):
        text = (
            " ".join(seg.get("words", []))
            if isinstance(seg.get("words"), list)
            else str(seg.get("text", ""))
        )
        cut_plan_data.append(
            {
                "Segment": i,
                "Start": f"{seg['start']:.2f}s",
                "End": f"{seg['end']:.2f}s",
                "Text": text[:50] + "..." if len(text) > 50 else text,
            }
        )

    await create_table_artifact(
        key="cut-plan",
        table=cut_plan_data,
        description=f"Cut plan: {len(segments_to_keep)} segments",
    )

    await asyncio.to_thread(
        trim_video_segments, input_video, output_video, segments_to_keep
    )

    total_duration = sum(seg["end"] - seg["start"] for seg in segments_to_keep)
    logger.info(f"Trimmed to {total_duration:.1f}s")
    logger.info(f"Edit summary: {edit_summary}")
    return output_video, edit_summary


@task(
    retries=1,
    timeout_seconds=300,
    log_prints=True,
)
def enhance_voice_task(input_video: Path, output_video: Path) -> Path:
    """Enhance voice in video."""
    logger = get_run_logger()
    logger.info("Enhancing voice")

    enhance_voice(input_video, output_video)
    return output_video


@task(
    retries=1,
    timeout_seconds=300,
    log_prints=True,
)
def add_subtitles_task(
    input_video: Path,
    output_video: Path,
    transcript_path: Path,
    word_count: int = 3,
    font_size: int = 14,
    caption_height: int = 10,
) -> Path:
    """Add subtitles to video.

    Args:
        caption_height: Vertical position (0-100, where 0 is bottom, 100 is top, default: 10)
    """
    logger = get_run_logger()
    logger.info(
        f"Adding subtitles ({word_count} words/line, height: {caption_height}%)"
    )

    _, words_data = read_word_transcript_file(transcript_path)

    # Check if utterance file exists (for punctuation-aware subtitles)
    # Transcript files are named like "05_trimmed-words.json", so utterance file would be "05_trimmed-utterances.json"
    base_name = transcript_path.stem.replace("-words", "")
    utterance_path = transcript_path.parent / f"{base_name}-utterances.json"

    # Use utterance-based subtitles if available (punctuation-aware)
    if utterance_path.exists():
        logger.info("Using utterance data for punctuation-aware subtitles...")
        utterances_data = json.loads(utterance_path.read_text())
        srt_content = add_srt_from_utterances(
            utterances_data, words_data, word_count=word_count, size=font_size
        )
    else:
        logger.info("Using word-level data for subtitles (utterance data not found)...")
        srt_content = generate_srt_content(
            words_data, word_count=word_count, size=font_size
        )

    srt_file = output_video.parent / f"{output_video.stem}.srt"
    srt_file.write_text(srt_content)

    # Get video height to calculate proper MarginV
    # MarginV in ASS format is pixels from bottom (higher = higher on screen)
    # caption_height: 0 = bottom, 100 = top (so 15 = 15% from bottom)
    # 0% (bottom) = MarginV = 20, 100% (top) = MarginV = video_height - 20
    # Formula: margin_v = 20 + caption_height * (video_height - 40) / 100
    video_height_result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=height",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(input_video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    video_height = int(video_height_result.stdout.strip())

    # ASS MarginV = pixels from bottom (caption_height: 0=bottom, 100=top → % from bottom)
    if caption_height is not None:
        margin_v = int(caption_height * video_height / 100)
    else:
        margin_v = int(15 * video_height / 100)  # default 15% from bottom
    margin_v = max(20, min(video_height - 40, margin_v))

    roboto_font_dir = "/Users/apple/Downloads/Roboto/static"
    vf_filter = f"subtitles={srt_file}:fontsdir='{roboto_font_dir}':force_style='Alignment=2,FontName=Roboto-Bold,FontSize={max(font_size, 14)},PrimaryColour=&Hffffff,OutlineColour=&H000000,Outline=3,Shadow=2,BorderStyle=1,MarginV={margin_v},MarginL=80,MarginR=80'"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-vf",
            vf_filter,
            "-c:v",
            "h264_videotoolbox",
            "-c:a",
            "copy",
            str(output_video),
        ],
        check=True,
        capture_output=True,
    )

    return output_video


@task(
    retries=2,
    timeout_seconds=600,
    log_prints=True,
)
async def add_background_music_task(
    input_video: Path, output_video: Path, music_url: str, music_title: str
) -> Path:
    """Add background music to video using pre-selected track."""
    logger = get_run_logger()
    logger.info(f"Adding music: {music_title}")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        music_path = Path(tmp.name)
    try:
        await download_music(music_url, music_path)
        add_music(input_video, music_path, output_video)
    finally:
        music_path.unlink(missing_ok=True)

    await create_link_artifact(
        key="selected-music",
        link=music_url,
        link_text=music_title,
        description="Selected background music",
    )

    return output_video


@task(
    retries=1,
    timeout_seconds=300,
    log_prints=True,
)
async def add_title_task(
    input_video: Path, output_video: Path, title_text: str, title_height: int = 10
) -> Path:
    """Add pre-selected title to video.

    Args:
        title_height: Vertical position (0-100, where 0 is bottom, 100 is top, default: 10)
    """
    logger = get_run_logger()
    logger.info(f"Adding title: {title_text} (height: {title_height}%)")

    add_title(
        input_video,
        title_text,
        output_video,
        duration=20.0,
        height_percent=title_height,
    )

    await create_markdown_artifact(
        key="final-title",
        markdown=f"## Video Title\n\n**{title_text}**",
        description="Selected video title",
    )

    return output_video


@task(retries=1, timeout_seconds=300, log_prints=True)
async def review_video_task(
    video_path: Path, transcript_path: Path, brand_brief: str | None = None
) -> dict[str, Any]:
    """Review video and generate vitality score, critiques, and caption."""
    import json

    logger = get_run_logger()
    logger.info("Reviewing video and generating caption...")

    # Load transcript
    transcript_text, _ = read_word_transcript_file(transcript_path)

    # Review video
    result = await review_video(video_path, transcript_text, brand_brief)

    logger.info(f"Vitality score: {result['vitality_score']}/10")

    # Create artifact for review
    await create_markdown_artifact(
        key="video-review",
        markdown=f"""## Video Review

**Vitality Score:** {result['vitality_score']}/10

**Caption:**
{result['caption']}

**Critiques:**
{json.dumps(result['critiques'], indent=2)}
""",
        description="Video review with vitality score and critiques",
    )

    return result


@task(log_prints=True)
async def create_thumbnail_task(video_path: Path, output_dir: Path, title: str) -> Path:
    """Generate YouTube-style thumbnail using Gemini 3 Pro Image.

    Uses the first frame as reference to create an engaging thumbnail with
    graphics, arrows, styled text, and other YouTube-style elements.
    """
    logger = get_run_logger()
    logger.info("Generating YouTube-style thumbnail...")

    # Extract first frame to temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        frame_path = Path(tmp_file.name)

    try:
        _extract_first_frame(video_path, frame_path)

        # Use Gemini 3 Pro Image to generate thumbnail
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.warning(
                "GEMINI_API_KEY not set, falling back to simple frame extraction"
            )
            thumbnail_path = output_dir / f"{video_path.stem}-thumbnail.jpg"
            frame_path.rename(thumbnail_path)
            return thumbnail_path

        from google import genai
        from google.genai import types
        from PIL import Image
        from io import BytesIO

        client = genai.Client(api_key=gemini_api_key)

        # Load the first frame
        reference_image = Image.open(frame_path)

        # Determine aspect ratio from video frame
        video_width, video_height = reference_image.size
        video_ratio = video_width / video_height

        # Map to closest supported aspect ratio
        aspect_ratios = {
            "16:9": 16 / 9,
            "9:16": 9 / 16,
            "4:3": 4 / 3,
            "3:4": 3 / 4,
            "1:1": 1 / 1,
            "21:9": 21 / 9,
        }

        # Find closest aspect ratio
        closest_ratio = min(
            aspect_ratios.items(), key=lambda x: abs(x[1] - video_ratio)
        )
        aspect_ratio = closest_ratio[0]

        logger.info(
            f"Video frame: {video_width}x{video_height} (ratio: {video_ratio:.2f}), using aspect_ratio: {aspect_ratio}"
        )

        # Create prompt for YouTube-style thumbnail
        # Prefix with "Generate an image:" to make intent explicit (prevents text-only responses)
        prompt = f"""Generate an image: Create a highly engaging YouTube-style thumbnail using this video frame as reference.

REQUIREMENTS:
- Use the provided frame as the base/reference image
- MODIFY the image by adding bold, eye-catching text overlay with the title: "{title}"
- ADD graphic elements: arrows, circles, highlights, borders, doodles
- ADD icons or illustrations that enhance the message
- Use high contrast colors for text (white/yellow text on dark backgrounds, or dark text on light)
- Make the title text large, bold, and easy to read
- Add visual interest with shadows, glows, or outlines on text
- MUST include graphic elements like:
  * Arrows pointing to key elements
  * Numbered lists or steps (if applicable)
  * Icons or emojis that relate to the content
  * Borders or frames around important elements
  * Highlights or callout boxes
  * Doodles or hand-drawn style elements

STYLE:
- YouTube thumbnail style: bold, attention-grabbing, professional
- High contrast and vibrant colors
- Text should be readable even at small sizes
- Composition should guide the eye to the title and key visual elements

IMPORTANT: You must actually generate/modify the image with these elements, not just describe them. The output should be a complete thumbnail image with all the requested graphic elements and text overlays."""

        # Generate image using Gemini 3 Pro Image (wrap in thread to avoid blocking)
        def _generate_image():
            return client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=[
                    reference_image,
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                    )
                ),
            )

        response = await asyncio.to_thread(_generate_image)

        # Extract image from response
        generated_image = None
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                generated_image = Image.open(BytesIO(part.inline_data.data))
                break

        if not generated_image:
            raise ValueError("No image generated by Gemini")

        # Save thumbnail
        thumbnail_path = output_dir / f"{video_path.stem}-thumbnail.jpg"
        generated_image.save(thumbnail_path, "JPEG", quality=95)

        logger.info(f"Thumbnail generated: {thumbnail_path}")
        return thumbnail_path

    finally:
        # Clean up temp frame file
        if frame_path.exists():
            frame_path.unlink()


@task(log_prints=True)
async def create_summary_artifact(
    input_video: Path,
    final_video: Path,
    title: str,
    music_title: str,
    input_duration: float,
    step1_duration: float,
    speedup: float,
    current_wpm: float | None,
    filler_count: int,
    step5_duration: float,
    edit_summary: str,
    caption_height: int,
    title_height: int,
    review_result: dict[str, Any],
) -> None:
    """Create final summary artifact."""

    output_duration = _get_video_duration(final_video)
    silence_removed = input_duration - step1_duration
    silence_reduction_pct = (
        (silence_removed / input_duration * 100) if input_duration > 0 else 0
    )

    # Format silence removed with before/after
    silence_str = f"{input_duration:.1f}s → {step1_duration:.1f}s ({silence_reduction_pct:.1f}% reduction)"

    # Format speedup info
    speedup_str = ""
    if speedup > 1.0:
        speedup_str = f"Applied {speedup}x speedup"
    elif current_wpm is not None:
        speedup_str = f"Skipped (WPM {current_wpm:.1f}, already ≥ 170)"
    else:
        speedup_str = "Skipped"

    # Format filler words
    filler_str = ""
    if filler_count > 0:
        filler_str = f"{filler_count} removed"
    else:
        filler_str = "None detected"

    # Format critiques as markdown
    critiques = review_result.get("critiques", {})
    critiques_str = ""
    if critiques:
        critiques_list = []
        for key, value in critiques.items():
            if value:  # Only include non-empty critiques
                # Format value as markdown-friendly text
                if isinstance(value, dict):
                    # Nested dict - format as bullet points
                    formatted_value = "\n  " + "\n  ".join(
                        [f"- {k}: {v}" for k, v in value.items() if v]
                    )
                elif isinstance(value, list):
                    # List - format as bullet points
                    formatted_value = "\n  " + "\n  ".join(
                        [f"- {item}" for item in value if item]
                    )
                elif isinstance(value, str):
                    # String - use as-is (might contain markdown)
                    formatted_value = value
                else:
                    # Other types - convert to string
                    formatted_value = str(value)

                critiques_list.append(
                    f"- **{key.replace('_', ' ').title()}**: {formatted_value}"
                )
        if critiques_list:
            critiques_str = "\n\n**Do This Next Time:**\n" + "\n".join(critiques_list)

    vitality_score = review_result.get("vitality_score", "N/A")
    caption = review_result.get("caption", "")

    summary_md = f"""## Video Processing Complete ✅

### Selections
- **Title**: {title} ({title_height}% from bottom)
- **Music**: {music_title}

### Processing Details
- **Removed silence**: {silence_str}
- **Speedup**: {speedup_str}
- **Filler words**: {filler_str}
- **Smart trim**: Trimmed to {step5_duration:.1f}s
  - {edit_summary}
- **Voice enhancement**: Applied
- **Subtitles**: Added ({caption_height}% from bottom)

### Stats
- **Input**: `{input_video.name}` ({input_duration:.1f}s)
- **Output**: `{final_video.name}` ({output_duration:.1f}s)
- **Total reduction**: {(1 - output_duration / input_duration) * 100:.1f}%

### Review Report
**Vitality Score:** {vitality_score}/10
{critiques_str}

### Social Media Caption
{caption}

### Output Location
`{final_video}`
"""

    await create_markdown_artifact(
        key="processing-summary",
        markdown=summary_md,
        description="Video processing summary",
    )


# =============================================================================
# Main Flow
# =============================================================================


@flow(
    name="process-video",
    log_prints=True,
    description="Process a video through the complete editing pipeline",
)
async def process_video_flow(
    input_video: Path,
    output_dir: Path,
    target_duration: int = 60,
    title: str | None = None,
    music_url: str | None = None,
    music_tags: str = "hip hop",
    word_count: int = 3,
    font_size: int = 14,
    silence_margin: float = 0.2,
    trim_tolerance: float = 20.0,
    speedup: float = 1.0,
    caption_height: int | None = None,
    title_height: int | None = None,
    brand_brief: str | None = None,
    skip_steps: set[str] | None = None,
    crop_top: int = 0,
    crop_bottom: int = 0,
    crop_left: int = 0,
    crop_right: int = 0,
) -> Path:
    """Process video through the complete editing pipeline.

    Title and music can be provided upfront. If not provided, they are auto-generated/selected.
    Use title_suggestions.py to get title suggestions before running this workflow.

    Args:
        input_video: Input video file
        output_dir: Directory for all output files
        target_duration: Target duration for smart trim (seconds)
        title: Video title (if not provided, auto-generates from transcript)
        music_url: Background music URL (if not provided, searches and uses first result)
        music_tags: Tags for background music search (only used if music_url not provided)
        word_count: Words per subtitle line
        font_size: Subtitle font size
        silence_margin: Margin in seconds to keep before/after silence (default: 0.2)
        trim_tolerance: Acceptable margin for trim duration in seconds (e.g., 20 means ±20s) (default: 20.0)
        speedup: Speed multiplier to apply after silence removal (e.g., 1.2 = 20% faster) (default: 1.0)
        caption_height: Vertical position for captions (0-100, where 0 is bottom, 100 is top). If None, auto-determined from first frame.
        title_height: Vertical position for title (0-100, where 0 is bottom, 100 is top). If None, auto-determined from first frame.
        brand_brief: Optional brand brief to guide title generation, cut planning, and review (e.g., brand voice, audience, guardrails)
        skip_steps: Set of step names to skip (e.g., {'smart_trim', 'add_music', 'review_video'})

    Returns:
        Path to final processed video
    """
    logger = get_run_logger()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get input video duration for progress estimation
    input_duration = _get_video_duration(input_video)
    estimated_time = estimate_duration(input_duration, speedup, target_duration)

    logger.info(f"🎬 Starting video processing: {input_video.name}")
    logger.info(f"   Input duration: {input_duration:.1f}s")
    logger.info(
        f"   Estimated processing time: {estimated_time:.0f}s ({estimated_time/60:.1f} min)"
    )

    start_time = time.time()

    # =========================================================================
    # Resolve title and music (if not provided)
    # =========================================================================

    selected_title = title
    selected_music_url = music_url
    selected_music_title = ""

    # If title not provided, generate from transcript
    if not selected_title:
        logger.info("📝 No title provided, generating from transcript...")
        transcript = await quick_transcript_task(input_video)
        title_options = await generate_title_options_task(transcript, brand_brief)
        selected_title = title_options[0]
        logger.info(f"   Auto-selected title: {selected_title}")

    # If music not provided, search and use first result
    if not selected_music_url:
        logger.info(f"🎵 No music URL provided, searching for '{music_tags}'...")
        music_options = await search_music_options_task(music_tags)
        selected_music_url = music_options[0]["url"]
        selected_music_title = (
            f"{music_options[0]['title']} by {music_options[0]['creator']}"
        )
        logger.info(f"   Auto-selected music: {selected_music_title}")
    else:
        selected_music_title = "User-provided music"

    logger.info(f"✅ Using:")
    logger.info(f"   Title: {selected_title}")
    logger.info(f"   Music: {selected_music_title}")

    # =========================================================================
    # Determine optimal positions (if not provided)
    # =========================================================================

    if caption_height is None or title_height is None:
        logger.info("🎯 Determining optimal caption and title positions...")
        auto_caption_height, auto_title_height = await determine_optimal_positions(
            input_video
        )
        if caption_height is None:
            caption_height = auto_caption_height
        if title_height is None:
            title_height = auto_title_height
        logger.info(f"   Caption position: {caption_height}%")
        logger.info(f"   Title position: {title_height}%")

    # =========================================================================
    # PHASE 2: Process video (no more user input needed)
    # =========================================================================

    logger.info("🎥 Phase 2: Processing video...")

    # Initialize skip_steps
    if skip_steps is None:
        skip_steps = set()
    else:
        skip_steps = {step.lower().strip() for step in skip_steps}
        logger.info(f"⏭️  Skipping steps: {', '.join(sorted(skip_steps))}")

    # Step 0: Crop video (if crop params specified)
    has_crop = crop_top > 0 or crop_bottom > 0 or crop_left > 0 or crop_right > 0
    if has_crop:
        step0_video = output_dir / "00_cropped.mp4"
        step0_video = crop_video_task(
            input_video, step0_video, crop_top, crop_bottom, crop_left, crop_right
        )
        video_for_processing = step0_video
    else:
        video_for_processing = input_video

    # Step 1: Remove silence
    step1_video = output_dir / "01_no-silence.mp4"
    step1_video = await remove_silence_task(
        video_for_processing, step1_video, margin=silence_margin
    )
    step1_duration = _get_video_duration(step1_video)
    silence_removed = input_duration - step1_duration

    # Step 2: Apply speedup (smart if speedup == 1.0, otherwise use provided value)
    step2_video = output_dir / "02_speedup.mp4"

    # Track WPM for summary (initialize to None)
    current_wpm: float | None = None

    # If speedup is 1.0 (default), calculate smart speedup based on WPM
    if speedup == 1.0:
        # Get transcript to calculate WPM
        step1_words_temp = output_dir / "01_no-silence-words-temp.json"
        step1_words_temp = await get_transcript_task(step1_video, step1_words_temp)
        from get_transcript import read_word_transcript_file

        _, words_data = read_word_transcript_file(step1_words_temp)
        word_count = len(words_data)

        # Calculate smart speedup
        calculated_speedup = _calculate_smart_speedup(step1_duration, word_count)
        current_wpm = _calculate_wpm(step1_duration, word_count)

        if calculated_speedup > 1.0:
            logger.info(f"📊 Current WPM: {current_wpm:.1f} (target: 180, min: 170)")
            logger.info(f"⚡ Smart speedup: {calculated_speedup:.3f}x to reach 180 WPM")
            speedup = calculated_speedup
        else:
            logger.info(
                f"📊 Current WPM: {current_wpm:.1f} (already >= 170, skipping speedup)"
            )
            speedup = 1.0

    step2_video = speedup_video_task(step1_video, step2_video, speedup)

    # Get transcript after speedup (timestamps must match the sped-up video)
    step2_words = output_dir / "02_speedup-words.json"
    step2_words = await get_transcript_task(step2_video, step2_words)

    # Step 3: Remove filler words (using transcript from sped-up video)
    step3_video = output_dir / "03_no-fillers.mp4"
    step3_video = await remove_filler_words_task(step2_video, step3_video, step2_words)

    # Step 4: Fix rotation (if MOV) - rotation doesn't affect audio timestamps
    step4_video = output_dir / "04_fixed.mp4"
    is_mov = input_video.suffix.lower() == ".mov"
    step4_video = fix_rotation_task(step3_video, step4_video, is_mov)

    # Count filler words removed
    from remove_filler_words import (
        find_filler_word_segments,
        load_words_from_transcript,
    )

    filler_words = load_words_from_transcript(step2_words)
    filler_segments = find_filler_word_segments(filler_words)
    filler_count = len(filler_segments) if filler_segments else 0

    # Get transcript after rotation (rotation doesn't change audio, but get transcript from step4_video for clarity)
    step4_words = output_dir / "04_fixed-words.json"
    step4_words = await get_transcript_task(step4_video, step4_words)

    # Step 5: Smart trim
    if "smart_trim" in skip_steps:
        logger.info("⏭️  Skipping smart trim")
        step5_video = step4_video
        edit_summary = "Smart trim skipped"
        step5_duration = _get_video_duration(step5_video)
        # Use transcript from step4 since we didn't trim
        step5_words = step4_words
    else:
        step5_video = output_dir / "05_trimmed.mp4"
        step5_video, edit_summary = await smart_trim_task(
            step4_video,
            step5_video,
            step4_words,
            target_duration,
            trim_tolerance,
            brand_brief=brand_brief,
        )
        step5_duration = _get_video_duration(step5_video)

        # Get new transcript after trimming
        step5_words = output_dir / "05_trimmed-words.json"
        step5_words = await get_transcript_task(step5_video, step5_words)

    # Step 6: Enhance voice
    step6_video = output_dir / "06_enhanced.mp4"
    step6_video = enhance_voice_task(step5_video, step6_video)

    # Step 7: Add subtitles
    step7_video = output_dir / "07_subtitled.mp4"
    step7_video = add_subtitles_task(
        step6_video, step7_video, step5_words, word_count, font_size, caption_height
    )

    # Step 8: Add music (using selected track)
    if "add_music" in skip_steps:
        logger.info("⏭️  Skipping background music")
        step8_video = step7_video
    else:
        step8_video = output_dir / "08_with-music.mp4"
        step8_video = await add_background_music_task(
            step7_video, step8_video, selected_music_url, selected_music_title
        )

    # Step 9: Add title (using selected title)
    final_video = output_dir / "09_final.mp4"
    final_video = await add_title_task(
        step8_video, final_video, selected_title, title_height
    )

    # Step 10: Review video and generate caption
    if "review_video" in skip_steps:
        logger.info("⏭️  Skipping video review")
        review_result = {
            "vitality_score": "N/A",
            "caption": "",
            "critiques": {},
        }
    else:
        review_result = await review_video_task(final_video, step5_words, brand_brief)

    # Create thumbnail
    await create_thumbnail_task(final_video, output_dir, selected_title)
    await create_summary_artifact(
        input_video,
        final_video,
        selected_title,
        selected_music_title,
        input_duration,
        step1_duration,
        speedup,
        current_wpm,
        filler_count,
        step5_duration,
        edit_summary,
        caption_height,
        title_height,
        review_result,
    )

    # Record run duration for future estimates
    processing_duration = time.time() - start_time
    record_run(
        input_duration=input_duration,
        processing_duration=processing_duration,
        speedup=speedup,
        target_duration=target_duration,
    )

    logger.info(f"✅ Pipeline complete! Final video: {final_video}")
    logger.info(
        f"   Processing took {processing_duration:.1f}s (estimated: {estimated_time:.0f}s)"
    )
    return final_video


# =============================================================================
# CLI Entry Point
# =============================================================================


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process a video through the complete editing pipeline"
    )
    parser.add_argument(
        "--input-video",
        type=Path,
        required=True,
        help="Input video file",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output directory for all processed files",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Target duration for smart trim in seconds (default: 60)",
    )
    parser.add_argument(
        "--music-tags",
        type=str,
        default="hip hop",
        help="Tags for background music search (default: 'hip hop')",
    )
    parser.add_argument(
        "--word-count",
        type=int,
        default=3,
        help="Words per subtitle line (default: 3)",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=14,
        help="Subtitle font size (default: 14)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Video title (if not provided, auto-generates from transcript)",
    )
    parser.add_argument(
        "--music-url",
        type=str,
        default=None,
        help="Background music URL (if not provided, searches and uses first result)",
    )
    parser.add_argument(
        "--silence-margin",
        type=float,
        default=0.2,
        help="Margin in seconds to keep before/after silence (default: 0.2)",
    )
    parser.add_argument(
        "--trim-tolerance",
        type=float,
        default=20.0,
        help="Acceptable margin for trim duration in seconds (e.g., 20 means ±20s, so 60s target = 40-80s range) (default: 20)",
    )
    parser.add_argument(
        "--speedup",
        type=float,
        default=1.0,
        help="Speed multiplier to apply after silence removal (e.g., 1.2 = 20%% faster) (default: 1.0)",
    )
    parser.add_argument(
        "--caption-height",
        type=int,
        default=None,
        help="Vertical position for captions (0-100, where 0 is bottom, 100 is top). If not provided, auto-determined from first frame.",
    )
    parser.add_argument(
        "--title-height",
        type=int,
        default=None,
        help="Vertical position for title (0-100, where 0 is bottom, 100 is top). If not provided, auto-determined from first frame.",
    )
    parser.add_argument(
        "--brand-brief",
        type=str,
        default=None,
        help="Brand brief to guide title generation, cut planning, and review (e.g., brand voice, audience, guardrails)",
    )
    args = parser.parse_args()

    if not args.input_video.exists():
        print(f"Error: Input video not found: {args.input_video}")
        sys.exit(1)

    print("=" * 60)
    print("🎬 Video Processing Pipeline")
    print("=" * 60)
    print(f"Input: {args.input_video}")
    print(f"Output: {args.output_path}")
    print(f"Duration: {args.duration}s (±{args.trim_tolerance}s)")
    print(f"Speedup: {args.speedup}x")
    if args.title:
        print(f"Title: {args.title}")
    else:
        print("Title: (auto-generate from transcript)")
    if args.music_url:
        print(f"Music: {args.music_url}")
    else:
        print(f"Music: (auto-select first '{args.music_tags}' result)")
    print("=" * 60)

    final_video = await process_video_flow(
        args.input_video,
        args.output_path,
        args.duration,
        args.title,
        args.music_url,
        args.music_tags,
        args.word_count,
        args.font_size,
        args.silence_margin,
        args.trim_tolerance,
        args.speedup,
        args.caption_height,
        args.title_height,
        args.brand_brief,
    )
    print(f"\n✅ Pipeline complete! Final video: {final_video}")


if __name__ == "__main__":
    asyncio.run(main())
