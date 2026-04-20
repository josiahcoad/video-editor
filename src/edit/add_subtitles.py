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
    caps: bool = False,
    delay: float = 0.0,
) -> str:
    """Generate SRT subtitle file with specified words per subtitle.

    Args:
        words: List of word dicts with 'word', 'start', 'end' keys
        word_count: Number of words per subtitle (default: 3)
        size: Font size for subtitles (default: 14)
        caps: If True, convert text to uppercase (default: False)
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
    caps: bool = False,
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
        caps: If True, convert text to uppercase (default: False)
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


async def add_subtitle(
    video_path: Path,
    output_path: Path | None = None,
    transcript_path: Path | None = None,
    settings_overrides: dict[str, Any] | None = None,
) -> Path:
    """Add subtitles to video (module entry point).

    Uses load_settings(video_path) for caption style, position, etc.
    Pass settings_overrides to override (e.g. captions.style, captions.height_percent).
    Returns path to output video.
    """
    settings = load_settings(video_path)
    if settings_overrides:
        settings = deep_merge(settings, settings_overrides)
    return await _add_subtitle_impl(
        video_path,
        output_path or (video_path.parent / f"{video_path.stem}-subtitled.mp4"),
        transcript_path,
        settings,
        cli_overrides=None,
    )


async def _add_subtitle_impl(
    video_path: Path,
    output_video: Path,
    transcript_file: Path | None,
    settings: dict[str, Any],
    cli_overrides: dict[str, Any] | None,
) -> Path:
    """Shared implementation for add_subtitle and CLI. cli_overrides from argv when run as script."""
    o = cli_overrides or {}
    captions_cfg = settings.get("captions") or {}
    # Default captions sit between 1/3 and 1/2 of the way up the frame so
    # the text lands above chin/shoulder level without floating into the
    # subject's eyeline. 40% → ~42% from bottom once libass renders it
    # (PlayResY=288, Alignment=2, MarginV=115). Verified on 1080x1920.
    caption_height = int(captions_cfg.get("height_percent", 40))
    if o and "height_percent" in o:
        caption_height = int(o["height_percent"])
    word_count = int(captions_cfg.get("word_count", 3))
    if o and "word_count" in o:
        word_count = int(o["word_count"])
    font_size = int(captions_cfg.get("font_size", 11))
    if o and "font_size" in o:
        font_size = int(o["font_size"])
    caption_style = (captions_cfg.get("style") or "huffines").lower()
    if o and "style" in o:
        caption_style = (o["style"] or "huffines").lower()
    # "default" is a legacy alias for "huffines" (white text, black outline,
    # semi-transparent black box, Roboto Bold, sentence case, ~40% from bottom).
    if caption_style == "default":
        caption_style = "huffines"
    if caption_style not in ("huffines", "classic", "outline"):
        caption_style = "huffines"
    caps = bool(captions_cfg.get("caps", False))
    if o and "caps" in o:
        caps = bool(o["caps"])
    font_name = str(captions_cfg.get("font") or "Roboto")
    if o and "font_name" in o:
        font_name = str(o["font_name"])
    replacements = {}
    for wrong, right in settings.get("replacements", {}).items():
        if isinstance(wrong, str) and isinstance(right, str):
            replacements[wrong.strip().lower()] = right.strip()
    if o and o.get("replace_str"):
        for k, v in parse_replacements(o["replace_str"]).items():
            replacements[k] = v
    title_text = o.get("title_text") if o else None
    caption_delay = float(o.get("delay", 0.0)) if o and "delay" in o else 0.0

    if transcript_file:
        print(f"Loading word-level transcript from: {transcript_file}")
        words_data = json.loads(transcript_file.read_text())
        utterances_data: list[dict] = []
        stem = transcript_file.stem
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
        result: dict[str, Any] = {
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
    if replacements:
        print(f"Word replacements: {replacements}")
        result["words"] = apply_replacements(result["words"], replacements)
        result["transcript"] = " ".join([w["word"] for w in result["words"]])

    srt_file = output_video.parent / f"{output_video.stem}.srt"
    word_transcript_file = output_video.parent / f"{output_video.stem}-words.json"
    utterance_transcript_file = (
        output_video.parent / f"{output_video.stem}-utterances.json"
    )

    if not transcript_file:
        word_transcript_file.write_text(json.dumps(result["words"], indent=2))
        print(f"Word-level transcript written: {word_transcript_file}")
        if result.get("utterances"):
            utterance_transcript_file.write_text(
                json.dumps(result["utterances"], indent=2)
            )
            print(f"Utterance-level transcript written: {utterance_transcript_file}")

    if caption_delay > 0:
        print(
            f"Caption delay: {caption_delay}s (captions suppressed while title is showing)"
        )

    if result.get("utterances"):
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
        srt_content = add_srt(
            result["words"],
            word_count=word_count,
            size=font_size,
            caps=caps,
            delay=caption_delay,
        )

    srt_file.write_text(srt_content)
    print(f"SRT file written: {srt_file}")

    filter_parts = []
    if title_text:
        words_list = title_text.upper().split()
        lines, current_line, current_length = [], [], 0
        for word in words_list:
            word_len = len(word)
            if current_length + word_len + 1 > 30 and current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = word_len
            else:
                current_line.append(word)
                current_length += word_len + (1 if current_line else 0)
        if current_line:
            lines.append(" ".join(current_line))
        title_font_size, line_spacing = 50, 60
        total_height = len(lines) * line_spacing
        roboto_bold_path = "/Users/apple/Downloads/Roboto/static/Roboto-Bold.ttf"
        for i, line in enumerate(lines):
            escaped = line.replace("'", "\\'").replace(":", "\\:").replace("\\", "\\\\")
            y_pos = f"h/2-{total_height}/2+{i * line_spacing}"
            filter_parts.append(
                f"drawtext=text='{escaped}':fontfile='{roboto_bold_path}':fontsize={title_font_size}:"
                f"fontcolor=black:box=1:boxcolor=white@1.0:boxborderw=15:text_align=center:x=(w-text_w)/2:y={y_pos}:enable='between(t,0,2)'"
            )

    effective_font_size = max(font_size, 10)
    if caption_style == "classic":
        style_parts = [
            "Alignment=2",
            f"FontName={font_name}",
            "Bold=1",
            f"FontSize={effective_font_size}",
            "PrimaryColour=&H00000000",
            "OutlineColour=&H00FFFFFF",
            "BackColour=&H00FFFFFF",
            "Outline=0",
            "Shadow=0",
            "BorderStyle=4",
            "MarginL=15",
            "MarginR=15",
        ]
    elif caption_style == "outline":
        style_parts = [
            "Alignment=2",
            f"FontName={font_name}",
            "Bold=1",
            f"FontSize={effective_font_size}",
            "PrimaryColour=&H00FFFFFF",
            "OutlineColour=&H00000000",
            "BackColour=&H80000000",
            "Outline=3",
            "Shadow=2",
            "BorderStyle=1",
            "MarginL=15",
            "MarginR=15",
        ]
    else:
        # "huffines" — named after the test video this style was dialed in on.
        # White Roboto Bold text, black outline, semi-transparent black box,
        # sentence case, ~40% from bottom (chest/neck area, 1/3–1/2 up).
        style_parts = [
            "Alignment=2",
            f"FontName={font_name}",
            "Bold=1",
            f"FontSize={effective_font_size}",
            "PrimaryColour=&H00FFFFFF",
            "OutlineColour=&H00000000",
            "BackColour=&H80000000",
            "Outline=2",
            "Shadow=0",
            "BorderStyle=1",
            "MarginL=15",
            "MarginR=15",
        ]
    ASS_PLAY_RES_Y = 288
    margin_v = max(
        10, min(ASS_PLAY_RES_Y - 20, int(caption_height * ASS_PLAY_RES_Y / 100))
    )
    style_parts.append(f"MarginV={margin_v}")
    roboto_font_dir = "/Users/apple/Downloads/Roboto/static"
    filter_parts.append(
        f"subtitles={srt_file}:fontsdir='{roboto_font_dir}':force_style='{','.join(style_parts)}'"
    )

    vf_filter = ",".join(filter_parts)
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
    return output_video


async def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print(
            "Usage: python add_subtitles.py <video_file> [--output <output_file>] [--title <title_text>] [--height <0-100>] [--replace <replacements>] [--settings <path>] [--style ...] [--caps] [--delay <seconds>]"
        )
        print("  --height:     Vertical position (0=bottom, 100=top, default=40)")
        print("  --replace:    Word replacements (e.g., 'Marquee:Marky,Josiah:Josia')")
        print(
            '  --settings:   Path to settings.json; uses settings.replacements ({"Markey": "Marky", ...}) if present. Merged with --replace.'
        )
        print(
            "  --style:      Caption preset — 'huffines' (white on dark box, default), 'classic' (black on white box), 'outline' (white text, black outline + drop shadow, no box)"
        )
        print("  --caps:       Force ALL CAPS captions")
        print("  --no-caps:    Force sentence case captions")
        print(
            "  --delay:      Suppress captions for the first N seconds (e.g. while title is showing)"
        )
        print("  --font:       Font family name (default=Roboto)")
        print("  --font-size:  ASS font size (default=11)")
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

    output_video = video_path.parent / f"{video_path.stem}-subtitled.mp4"
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_video = Path(sys.argv[idx + 1])

    transcript_file = None
    if "--transcript" in sys.argv:
        idx = sys.argv.index("--transcript")
        if idx + 1 < len(sys.argv):
            transcript_file = Path(sys.argv[idx + 1])

    def _arg(key: str, fn=(lambda x: x)):
        if key not in sys.argv:
            return None
        idx = sys.argv.index(key)
        if idx + 1 < len(sys.argv):
            try:
                return fn(sys.argv[idx + 1])
            except (ValueError, TypeError):
                pass
        return None

    cli_overrides: dict[str, Any] = {}
    if _arg("--title") is not None:
        cli_overrides["title_text"] = _arg("--title")
    if _arg("--height", int) is not None:
        cli_overrides["height_percent"] = _arg("--height", int)
    if _arg("--word-count", int) is not None:
        cli_overrides["word_count"] = _arg("--word-count", int)
    if _arg("--font-size", int) is not None:
        cli_overrides["font_size"] = _arg("--font-size", int)
    if _arg("--style") is not None:
        cli_overrides["style"] = _arg("--style")
    if "--caps" in sys.argv:
        cli_overrides["caps"] = True
    if "--no-caps" in sys.argv:
        cli_overrides["caps"] = False
    if _arg("--font") is not None:
        cli_overrides["font_name"] = _arg("--font")
    if _arg("--delay", float) is not None:
        cli_overrides["delay"] = _arg("--delay", float)
    if _arg("--replace") is not None:
        cli_overrides["replace_str"] = _arg("--replace")

    await _add_subtitle_impl(
        video_path, output_video, transcript_file, settings, cli_overrides
    )


if __name__ == "__main__":
    asyncio.run(main())
