#!/usr/bin/env python3
"""Add subtitles to a video using Deepgram transcription.

Generates SRT subtitles with 3 words per line and burns them into the video.
Also generates word-level and utterance-level transcript files.

Usage:
    python add_subtitles.py <video_file> [--output <output_file>] [--title <title_text>]
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from get_transcript import get_transcript


def format_time_srt(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def add_srt(words: list[dict[str, Any]], word_count: int = 3, size: int = 14) -> str:
    """Generate SRT subtitle file with specified words per subtitle.

    Args:
        words: List of word dicts with 'word', 'start', 'end' keys
        word_count: Number of words per subtitle (default: 3)
        size: Font size for subtitles (default: 14)

    Returns:
        SRT file content as string
    """
    srt_lines: list[str] = []
    subtitle_id = 1

    # Group words into chunks of word_count
    for i in range(0, len(words), word_count):
        # Get words for this subtitle
        subtitle_words = words[i : i + word_count]

        if not subtitle_words:
            continue

        start_time = subtitle_words[0].get("start", 0.0)
        end_time = subtitle_words[-1].get("end", 0.0)

        # Join words with spaces
        text = " ".join([w.get("word", "").strip() for w in subtitle_words])

        # Convert to uppercase
        text = text.upper()

        # Format SRT entry
        srt_lines.append(str(subtitle_id))
        srt_lines.append(
            f"{format_time_srt(start_time)} --> {format_time_srt(end_time)}"
        )
        srt_lines.append(text)
        srt_lines.append("")  # Blank line between entries

        subtitle_id += 1

    return "\n".join(srt_lines)


def add_srt_from_utterances(
    utterances: list[dict[str, Any]],
    words: list[dict[str, Any]],
    word_count: int = 3,
    size: int = 14,
) -> str:
    """Generate SRT subtitle file from utterances, splitting at punctuation.

    Uses utterance data (which includes punctuation) to create more natural subtitle breaks.
    Splits immediately at punctuation marks (. ! ? , ;) and respects word_count limit.

    Args:
        utterances: List of utterance dicts with 'text', 'start', 'end' keys
        words: List of word dicts with 'word', 'start', 'end' keys (for precise timing)
        word_count: Maximum words per subtitle (default: 3)
        size: Font size for subtitles (default: 14, unused but kept for compatibility)

    Returns:
        SRT file content as string
    """
    srt_lines: list[str] = []
    subtitle_id = 1

    # Process each utterance
    for utterance in utterances:
        utterance_text = utterance.get("text", "").strip()
        utterance_start = utterance.get("start", 0.0)
        utterance_end = utterance.get("end", 0.0)

        if not utterance_text:
            continue

        # Split utterance into words (preserving punctuation)
        utterance_words = utterance_text.split()

        # Find words within this utterance's time range for precise timing
        utterance_words_list = [
            w
            for w in words
            if w["start"] >= utterance_start and w["end"] <= utterance_end
        ]

        # Group words intelligently: split at punctuation, respect word_count
        current_group = []
        current_group_words = 0
        word_idx = 0

        for word_text in utterance_words:
            current_group.append(word_text)
            current_group_words += 1

            # Check if word ends with punctuation
            ends_with_punct = any(
                word_text.rstrip().endswith(p) for p in [".", "!", "?", ",", ";"]
            )

            # Split if: punctuation OR reached word_count limit
            if ends_with_punct or current_group_words >= word_count:
                # Create subtitle for this group
                text = " ".join(current_group).upper()

                # Calculate timing: use word timestamps if available, otherwise use utterance timing proportionally
                if word_idx < len(
                    utterance_words_list
                ) and word_idx + current_group_words <= len(utterance_words_list):
                    # Use precise word timing
                    group_words = utterance_words_list[
                        word_idx : word_idx + current_group_words
                    ]
                    start_time = group_words[0]["start"]
                    end_time = group_words[-1]["end"]
                else:
                    # Fallback: use utterance timing proportionally
                    utterance_duration = utterance_end - utterance_start
                    words_before = word_idx
                    words_in_group = current_group_words
                    total_words = len(utterance_words)

                    if total_words > 0:
                        start_ratio = words_before / total_words
                        end_ratio = (words_before + words_in_group) / total_words
                        start_time = utterance_start + (
                            utterance_duration * start_ratio
                        )
                        end_time = utterance_start + (utterance_duration * end_ratio)
                    else:
                        start_time = utterance_start
                        end_time = utterance_end

                srt_lines.append(str(subtitle_id))
                srt_lines.append(
                    f"{format_time_srt(start_time)} --> {format_time_srt(end_time)}"
                )
                srt_lines.append(text)
                srt_lines.append("")  # Blank line between entries

                subtitle_id += 1
                word_idx += current_group_words
                current_group = []
                current_group_words = 0

        # Handle remaining words in current_group
        if current_group:
            text = " ".join(current_group).upper()

            # Calculate timing for remaining words
            if word_idx < len(utterance_words_list):
                group_words = utterance_words_list[word_idx:]
                start_time = group_words[0]["start"]
                end_time = group_words[-1]["end"]
            else:
                # Fallback to utterance end
                utterance_duration = utterance_end - utterance_start
                words_before = word_idx
                total_words = len(utterance_words)

                if total_words > 0:
                    start_ratio = words_before / total_words
                    start_time = utterance_start + (utterance_duration * start_ratio)
                else:
                    start_time = utterance_start
                end_time = utterance_end

            srt_lines.append(str(subtitle_id))
            srt_lines.append(
                f"{format_time_srt(start_time)} --> {format_time_srt(end_time)}"
            )
            srt_lines.append(text)
            srt_lines.append("")  # Blank line between entries

            subtitle_id += 1

    return "\n".join(srt_lines)


# TODO: Add word-by-word highlighting using ASS format
# This would require:
# - generate_ass_subtitle() function to create ASS format subtitles
# - _write_sentence_dialogues() helper to generate word-by-word highlight entries
# - format_time_ass() function for ASS time format
# - Option to use ASS instead of SRT when --word-highlight flag is provided


async def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print(
            "Usage: python add_subtitles.py <video_file> [--output <output_file>] [--title <title_text>] [--height <0-100>]"
        )
        print("  --height: Vertical position (0=bottom, 100=top, default=bottom)")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    # Parse title from command line
    title_text = None
    if "--title" in sys.argv:
        idx = sys.argv.index("--title")
        if idx + 1 < len(sys.argv):
            title_text = sys.argv[idx + 1]

    # Parse height (0-100, where 0 is bottom, 100 is top)
    caption_height = None  # None = default (bottom)
    if "--height" in sys.argv:
        idx = sys.argv.index("--height")
        if idx + 1 < len(sys.argv):
            caption_height = int(sys.argv[idx + 1])

    # Get transcript (check if transcript file provided, otherwise transcribe)
    transcript_file = None
    if "--transcript" in sys.argv:
        idx = sys.argv.index("--transcript")
        if idx + 1 < len(sys.argv):
            transcript_file = Path(sys.argv[idx + 1])

    if transcript_file:
        print(f"Loading word-level transcript from: {transcript_file}")
        words_data = json.loads(transcript_file.read_text())
        # Reconstruct result dict from words
        result = {
            "words": words_data,
            "transcript": " ".join([w["word"] for w in words_data]),
            "utterances": [],  # Not needed for add_subtitles
        }
    else:
        print("Transcribing with Deepgram...")
        result = await get_transcript(video_path)

    print(f"Found {len(result['words'])} words")
    if result.get("utterances"):
        print(f"Found {len(result['utterances'])} utterances")

    # Determine output paths
    output_video = video_path.parent / f"{video_path.stem}-subtitled.mp4"
    srt_file = video_path.parent / f"{video_path.stem}.srt"
    word_transcript_file = video_path.parent / f"{video_path.stem}-words.json"
    utterance_transcript_file = video_path.parent / f"{video_path.stem}-utterances.json"

    # Check for --output flag
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_video = Path(sys.argv[idx + 1])
            # Update transcript file names to match output video base name
            base_name = output_video.stem
            word_transcript_file = output_video.parent / f"{base_name}-words.json"
            utterance_transcript_file = (
                output_video.parent / f"{base_name}-utterances.json"
            )
            srt_file = output_video.parent / f"{base_name}.srt"

    # Write word-level transcript (JSON) - only if we transcribed (not from file)
    if not transcript_file:
        word_transcript_file.write_text(json.dumps(result["words"], indent=2))
        print(f"Word-level transcript written: {word_transcript_file}")

        # Write utterance-level transcript (JSON)
        if result.get("utterances"):
            utterance_transcript_file.write_text(
                json.dumps(result["utterances"], indent=2)
            )
            print(f"Utterance-level transcript written: {utterance_transcript_file}")

    # Parse word_count and font_size from command line (defaults: 3, 14)
    word_count = 3
    font_size = 14

    if "--word-count" in sys.argv:
        idx = sys.argv.index("--word-count")
        if idx + 1 < len(sys.argv):
            word_count = int(sys.argv[idx + 1])

    if "--font-size" in sys.argv:
        idx = sys.argv.index("--font-size")
        if idx + 1 < len(sys.argv):
            font_size = int(sys.argv[idx + 1])

    # Use smart utterance-based captioning if utterances are available
    if result.get("utterances"):
        print(
            f"Generating smart SRT subtitle file from utterances (max {word_count} words per line, splits at punctuation, font size {font_size})..."
        )
        srt_content = add_srt_from_utterances(
            result["utterances"], result["words"], word_count=word_count, size=font_size
        )
    else:
        print(
            f"Generating SRT subtitle file ({word_count} words per line, font size {font_size})..."
        )
        print(
            "⚠️  Note: No utterance data available - using simple word-based captioning. For smart captioning with punctuation, ensure Deepgram returns utterances."
        )
        srt_content = add_srt(result["words"], word_count=word_count, size=font_size)

    # Write SRT file
    srt_file.write_text(srt_content)
    print(f"SRT file written: {srt_file}")

    # Build video filter chain
    filter_parts = []

    # Add title overlay if provided
    if title_text:
        # Split long titles into multiple lines (max ~30 chars per line)
        words = title_text.upper().split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            word_len = len(word)
            # If adding this word would exceed ~30 chars, start new line
            if current_length + word_len + 1 > 30 and current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = word_len
            else:
                current_line.append(word)
                current_length += word_len + (1 if current_line else 0)

        if current_line:
            lines.append(" ".join(current_line))

        # Use multiple drawtext filters (one per line) since \n doesn't work reliably
        # Escape special characters for each line
        title_font_size = 50
        line_spacing = 60  # Spacing between lines
        num_lines = len(lines)
        # Total height of all lines
        total_height = num_lines * line_spacing

        for i, line in enumerate(lines):
            # Escape special characters for ffmpeg drawtext
            escaped_line = (
                line.replace("'", "\\'").replace(":", "\\:").replace("\\", "\\\\")
            )

            # Calculate y position for this line
            # Center the block: h/2 - total_height/2 + (line_index * line_spacing)
            y_offset = i * line_spacing
            y_pos = f"h/2-{total_height}/2+{y_offset}"

            # Create drawtext filter for this line
            title_filter = (
                f"drawtext=text='{escaped_line}':"
                f"font=Helvetica-Bold:"
                f"fontsize={title_font_size}:"
                f"fontcolor=black:"
                f"box=1:"
                f"boxcolor=white@1.0:"
                f"boxborderw=15:"
                f"text_align=center:"
                f"x=(w-text_w)/2:"  # Center horizontally
                f"y={y_pos}:"  # Position vertically
                f"enable='between(t,0,2)'"  # Show for first 2 seconds
            )
            filter_parts.append(title_filter)

    # Add subtitle filter (SRT) - configurable font size, all caps, position
    # If caption_height is set, position subtitles at that height (0-100, where 0=bottom, 100=top)
    style_parts = [
        "FontName=Arial",
        f"FontSize={font_size}",
        "PrimaryColour=&Hffffff",
        "OutlineColour=&H000000",
        "Outline=2",
    ]

    if caption_height is not None:
        # Convert height percentage to ASS MarginV (margin from bottom in pixels)
        # For a 1920px tall video: height=100 (top) → MarginV=1700, height=0 (bottom) → MarginV=100
        # We'll use a formula: MarginV = 100 + (100 - height) * 16
        # This gives: height=100 → MarginV=100, height=0 → MarginV=1700
        # Actually, let's use a simpler approach: height=100 means top, so MarginV should be large
        # For top positioning: MarginV ≈ video_height * 0.9 (90% from bottom = 10% from top)
        # We'll use a fixed large value that works for most videos
        margin_v = int(
            100 + (100 - caption_height) * 16
        )  # height=100 → 100, height=0 → 1700
        style_parts.append(f"MarginV={margin_v}")

    style_str = ",".join(style_parts)
    filter_parts.append(f"subtitles={srt_file}:force_style='{style_str}'")

    # Combine filters
    vf_filter = ",".join(filter_parts)

    # Burn subtitles (and title if provided) into video
    print("Burning subtitles into video...")
    if title_text:
        print(f"Adding title overlay: '{title_text.upper()}' (first 2 seconds)")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            vf_filter,
            "-c:v",
            "h264_videotoolbox",
            "-c:a",
            "copy",
            str(output_video),
        ],
        check=True,
    )

    print(f"✅ Output video: {output_video}")
    print(f"✅ SRT file: {srt_file}")
    print(f"✅ Word-level transcript: {word_transcript_file}")
    print(f"✅ Utterance-level transcript: {utterance_transcript_file}")


if __name__ == "__main__":
    asyncio.run(main())
