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

from .encode_common import H264_SOCIAL_COLOR_ARGS
from .get_transcript import get_transcript
from .settings_loader import deep_merge, load_settings


def parse_replacements(replace_str: str) -> dict[str, str]:
    """Parse replacement string into a dictionary.

    Format: "original1:replacement1,original2:replacement2"
    Example: "Marquee:Marky,Josiah:Josia"

    Returns:
        Dict mapping original words to replacements (case-insensitive keys)
    """
    replacements = {}
    if not replace_str:
        return replacements

    for pair in replace_str.split(","):
        pair = pair.strip()
        if ":" in pair:
            original, replacement = pair.split(":", 1)
            # Store lowercase key for case-insensitive matching
            replacements[original.strip().lower()] = replacement.strip()

    return replacements


def apply_replacements(
    words: list[dict[str, Any]], replacements: dict[str, str]
) -> list[dict[str, Any]]:
    """Apply word replacements to the words list.

    Args:
        words: List of word dicts with 'word', 'start', 'end' keys
        replacements: Dict mapping original words (lowercase) to replacements

    Returns:
        New words list with replacements applied
    """
    if not replacements:
        return words

    result = []
    for word_dict in words:
        new_dict = word_dict.copy()
        original_word = word_dict.get("word", "")
        # Check if this word (case-insensitive) should be replaced
        lookup_key = original_word.lower().strip(".,!?;:'\"")
        if lookup_key in replacements:
            new_dict["word"] = replacements[lookup_key]
        result.append(new_dict)

    return result


def format_time_srt(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def add_srt(
    words: list[dict[str, Any]],
    word_count: int = 3,
    size: int = 14,
    caps: bool = True,
    delay: float = 0.0,
) -> str:
    """Generate SRT subtitle file with specified words per subtitle.

    Args:
        words: List of word dicts with 'word', 'start', 'end' keys
        word_count: Number of words per subtitle (default: 3)
        size: Font size for subtitles (default: 14)
        caps: If True, convert text to uppercase (default: True)
        delay: Suppress captions before this timestamp (seconds). Useful to
               avoid overlapping with a title overlay.

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

        # Skip entries that end before the delay threshold
        if delay > 0 and end_time <= delay:
            continue
        # Clamp start to delay if the entry spans across it
        if delay > 0 and start_time < delay:
            start_time = delay

        # Join words with spaces
        text = " ".join([w.get("word", "").strip() for w in subtitle_words])

        if caps:
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
    replacements: dict[str, str] | None = None,
    caps: bool = True,
    delay: float = 0.0,
) -> str:
    """Generate SRT subtitle file from utterances, splitting at punctuation.

    Uses utterance data (which includes punctuation) to create more natural subtitle breaks.
    Splits immediately at punctuation marks (. ! ? , ;) and respects word_count limit.

    Args:
        utterances: List of utterance dicts with 'text', 'start', 'end' keys
        words: List of word dicts with 'word', 'start', 'end' keys (for precise timing)
        word_count: Maximum words per subtitle (default: 3)
        size: Font size for subtitles (default: 14, unused but kept for compatibility)
        caps: If True, convert text to uppercase (default: True)
        delay: Suppress captions before this timestamp (seconds).

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

        # Apply word replacements to utterance text (preserving trailing punctuation)
        if replacements:
            replaced = []
            for w in utterance_words:
                stripped = w.lower().rstrip(".,!?;:'\"")
                if stripped in replacements:
                    # Preserve any trailing punctuation from the original word
                    suffix = w[len(stripped) :]
                    replaced.append(replacements[stripped] + suffix)
                else:
                    replaced.append(w)
            utterance_words = replaced

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
                text = " ".join(current_group)
                if caps:
                    text = text.upper()

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

                # Apply delay: skip entries before the threshold
                if delay > 0 and end_time <= delay:
                    pass  # suppress this entry entirely
                else:
                    if delay > 0 and start_time < delay:
                        start_time = delay
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
            text = " ".join(current_group)
            if caps:
                text = text.upper()

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

            # Apply delay: skip entries before the threshold
            if not (delay > 0 and end_time <= delay):
                if delay > 0 and start_time < delay:
                    start_time = delay
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
            "Usage: python add_subtitles.py <video_file> [--output <output_file>] [--title <title_text>] [--height <0-100>] [--replace <replacements>] [--settings <path>] [--style ...] [--caps] [--delay <seconds>]"
        )
        print("  --height:     Vertical position (0=bottom, 100=top, default=12)")
        print("  --replace:    Word replacements (e.g., 'Marquee:Marky,Josiah:Josia')")
        print(
            '  --settings:   Path to settings.json; uses settings.replacements ({"Markey": "Marky", ...}) if present. Merged with --replace.'
        )
        print(
            "  --style:      Caption preset — 'default' (white on dark box), 'classic' (black on white box), 'outline' (white text, black outline + drop shadow, no box)"
        )
        print("  --caps:       Force ALL CAPS captions")
        print("  --no-caps:    Force sentence case captions")
        print(
            "  --delay:      Suppress captions for the first N seconds (e.g. while title is showing)"
        )
        print("  --font:       Font family name (default=Roboto)")
        print("  --font-size:  ASS font size (default=20)")
        print("  --word-count: Max words per subtitle line (default=3)")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    # Load project/session settings; optional --settings <path> overrides with explicit file
    settings = load_settings(video_path)
    if "--settings" in sys.argv:
        idx = sys.argv.index("--settings")
        if idx + 1 < len(sys.argv):
            settings_path = Path(sys.argv[idx + 1])
            if settings_path.exists():
                settings = deep_merge(settings, json.loads(settings_path.read_text()))

    # Defaults from settings.captions (CLI flags override)
    captions_cfg = settings.get("captions") or {}
    caption_height = int(captions_cfg.get("height_percent", 12))
    word_count = int(captions_cfg.get("word_count", 3))
    font_size = int(captions_cfg.get("font_size", 20))
    caption_style = (captions_cfg.get("style") or "default").lower()
    if caption_style not in ("default", "classic", "outline"):
        caption_style = "default"
    caps = bool(captions_cfg.get("caps", True))
    font_name = str(captions_cfg.get("font") or "Roboto")

    # Replacements: from settings.replacements, then CLI --replace
    replacements = {}
    for wrong, right in settings.get("replacements", {}).items():
        if isinstance(wrong, str) and isinstance(right, str):
            replacements[wrong.strip().lower()] = right.strip()

    # Parse title from command line
    title_text = None
    if "--title" in sys.argv:
        idx = sys.argv.index("--title")
        if idx + 1 < len(sys.argv):
            title_text = sys.argv[idx + 1]

    # Override with CLI --height
    if "--height" in sys.argv:
        idx = sys.argv.index("--height")
        if idx + 1 < len(sys.argv):
            try:
                caption_height = int(sys.argv[idx + 1])
            except ValueError:
                pass

    # Get transcript (check if transcript file provided, otherwise transcribe)
    transcript_file = None
    if "--transcript" in sys.argv:
        idx = sys.argv.index("--transcript")
        if idx + 1 < len(sys.argv):
            transcript_file = Path(sys.argv[idx + 1])

    if transcript_file:
        print(f"Loading word-level transcript from: {transcript_file}")
        words_data = json.loads(transcript_file.read_text())

        # Auto-discover sibling utterances file for smart captioning.
        # If transcript is "foo-words.json", look for "foo-utterances.json".
        utterances_data: list[dict] = []
        stem = transcript_file.stem  # e.g. "02_fast-words"
        if stem.endswith("-words"):
            utterances_path = (
                transcript_file.parent / f"{stem.replace('-words', '-utterances')}.json"
            )
            if utterances_path.exists():
                utterances_data = json.loads(utterances_path.read_text())
                print(
                    f"Loaded {len(utterances_data)} utterances from {utterances_path.name}"
                )
            else:
                print(
                    f"No utterances file found at {utterances_path.name} — falling back to word-based captioning"
                )

        result = {
            "words": words_data,
            "transcript": " ".join([w["word"] for w in words_data]),
            "utterances": utterances_data,
        }
    else:
        print("Transcribing with Deepgram...")
        result = await get_transcript(video_path)

    print(f"Found {len(result['words'])} words")
    if result.get("utterances"):
        print(f"Found {len(result['utterances'])} utterances")

    # Override with CLI --replace (merge into replacements from settings)
    if "--replace" in sys.argv:
        idx = sys.argv.index("--replace")
        if idx + 1 < len(sys.argv):
            for k, v in parse_replacements(sys.argv[idx + 1]).items():
                replacements[k] = v
    if replacements:
        print(f"Word replacements: {replacements}")
        result["words"] = apply_replacements(result["words"], replacements)
        result["transcript"] = " ".join([w["word"] for w in result["words"]])

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

    # Override with CLI --word-count, --font-size, --style, --caps
    if "--word-count" in sys.argv:
        idx = sys.argv.index("--word-count")
        if idx + 1 < len(sys.argv):
            word_count = int(sys.argv[idx + 1])

    if "--font-size" in sys.argv:
        idx = sys.argv.index("--font-size")
        if idx + 1 < len(sys.argv):
            font_size = int(sys.argv[idx + 1])

    if "--style" in sys.argv:
        idx = sys.argv.index("--style")
        if idx + 1 < len(sys.argv):
            caption_style = sys.argv[idx + 1].lower()
            if caption_style not in ("default", "classic", "outline"):
                print(f"⚠️  Unknown style '{caption_style}', falling back to 'default'")
                caption_style = "default"

    if "--caps" in sys.argv:
        caps = True
    if "--no-caps" in sys.argv:
        caps = False

    # Override with CLI --delay
    caption_delay = 0.0
    if "--delay" in sys.argv:
        idx = sys.argv.index("--delay")
        if idx + 1 < len(sys.argv):
            caption_delay = float(sys.argv[idx + 1])
    if caption_delay > 0:
        print(
            f"Caption delay: {caption_delay}s (captions suppressed while title is showing)"
        )

    # Use smart utterance-based captioning if utterances are available
    if result.get("utterances"):
        print(
            f"Generating smart SRT subtitle file from utterances (max {word_count} words per line, splits at punctuation, font size {font_size}, caps={caps})..."
        )
        srt_content = add_srt_from_utterances(
            result["utterances"],
            result["words"],
            word_count=word_count,
            size=font_size,
            replacements=replacements,
            caps=caps,
            delay=caption_delay,
        )
    else:
        print(
            f"Generating SRT subtitle file ({word_count} words per line, font size {font_size}, caps={caps})..."
        )
        print(
            "⚠️  Note: No utterance data available - using simple word-based captioning. For smart captioning with punctuation, ensure Deepgram returns utterances."
        )
        srt_content = add_srt(
            result["words"],
            word_count=word_count,
            size=font_size,
            caps=caps,
            delay=caption_delay,
        )

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
            roboto_bold_path = "/Users/apple/Downloads/Roboto/static/Roboto-Bold.ttf"
            title_filter = (
                f"drawtext=text='{escaped_line}':"
                f"fontfile='{roboto_bold_path}':"  # Custom Roboto Bold font
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

    # Add subtitle filter (SRT)
    # caption_height: 0-100 = % from bottom. ASS MarginV = pixels from bottom (only for bottom-aligned subs).
    # Per ASS spec: MarginV is ignored for midtitles (vertically centered). Set Alignment=2 (bottom center)
    # so MarginV is always respected: https://stackoverflow.com/questions/57869367/ffmpeg-subtitles-alignment-and-position
    #
    # ASS PlayRes defaults (set by ffmpeg SRT→ASS conversion):
    #   PlayResX=384, PlayResY=288
    # All margin/size values below are in these units and scale to actual video resolution.
    print(f"Caption style: {caption_style}")

    # Override with CLI --font
    if "--font" in sys.argv:
        idx = sys.argv.index("--font")
        if idx + 1 < len(sys.argv):
            font_name = sys.argv[idx + 1]
    effective_font_size = max(font_size, 20)

    if caption_style == "classic":
        style_parts = [
            "Alignment=2",
            f"FontName={font_name}",
            "Bold=1",
            f"FontSize={effective_font_size}",
            "PrimaryColour=&H00000000",  # Black text (AABBGGRR)
            "OutlineColour=&H00FFFFFF",  # White outline (matches box)
            "BackColour=&H00FFFFFF",  # Solid white background box
            "Outline=0",  # No outline — clean box edge
            "Shadow=0",
            "BorderStyle=4",  # Background box behind each line
            "MarginL=15",
            "MarginR=15",
        ]
    elif caption_style == "outline":
        style_parts = [
            "Alignment=2",
            f"FontName={font_name}",
            "Bold=1",
            f"FontSize={effective_font_size}",
            "PrimaryColour=&H00FFFFFF",  # White text (AABBGGRR)
            "OutlineColour=&H00000000",  # Black outline
            "BackColour=&H80000000",  # Shadow colour (semi-transparent black)
            "Outline=3",  # Thick outline for readability without a box
            "Shadow=2",  # Drop shadow offset (px in ASS units)
            "BorderStyle=1",  # Outline + shadow only, NO background box
            "MarginL=15",
            "MarginR=15",
        ]
    else:
        style_parts = [
            "Alignment=2",
            f"FontName={font_name}",
            "Bold=1",
            f"FontSize={effective_font_size}",
            "PrimaryColour=&H00FFFFFF",  # White text (AABBGGRR)
            "OutlineColour=&H00000000",  # Black outline
            "BackColour=&H80000000",  # Semi-transparent black background
            "Outline=2",  # Moderate outline for readability
            "Shadow=0",
            "BorderStyle=4",  # Background box behind each line + outline
            "MarginL=15",
            "MarginR=15",
        ]

    # MarginV is interpreted relative to ASS PlayResY, NOT the actual video height.
    # ffmpeg's SRT→ASS conversion uses PlayResY=288 (verified from ffmpeg output header).
    # So we scale caption_height (0-100%) against PlayResY to get correct placement.
    ASS_PLAY_RES_Y = 288
    margin_v = int(caption_height * ASS_PLAY_RES_Y / 100)
    margin_v = max(10, min(ASS_PLAY_RES_Y - 20, margin_v))
    style_parts.append(f"MarginV={margin_v}")
    print(
        f"Caption position: {caption_height}% from bottom → MarginV={margin_v} (ASS PlayResY={ASS_PLAY_RES_Y})"
    )

    style_str = ",".join(style_parts)
    # Use fontsdir to point to Roboto font directory for ASS subtitles
    roboto_font_dir = "/Users/apple/Downloads/Roboto/static"
    filter_parts.append(
        f"subtitles={srt_file}:fontsdir='{roboto_font_dir}':force_style='{style_str}'"
    )

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
            *H264_SOCIAL_COLOR_ARGS,
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
