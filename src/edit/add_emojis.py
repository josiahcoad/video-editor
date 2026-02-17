#!/usr/bin/env python3
"""
Add emoji overlays at key moments in a video.

Uses an LLM to analyze the transcript and pick 3-5 moments where a visual
emoji reinforces the spoken content. Generates emoji PNGs with a
semi-transparent white circular backing, then overlays them via ffmpeg.

Usage:
  add_emojis.py <video> <output> --transcript <words.json>
  add_emojis.py <video> <output> --transcript <words.json> --max-emojis 5
  add_emojis.py <video> <output> --transcript <words.json> --dry-run

Dry-run mode prints the LLM's emoji picks (timestamp, emoji, rationale)
without rendering, so you can review before committing to a full encode.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from .encode_common import H264_SOCIAL_COLOR_ARGS
from .get_transcript import get_transcript, read_word_transcript_file
from .settings_loader import load_settings


# ── Pydantic models for structured LLM output ────────────────────────────────


class EmojiPick(BaseModel):
    """A single emoji overlay placement."""

    emoji: str = Field(
        description=(
            "A single emoji character that visually reinforces the spoken content "
            "at this moment. Pick universally recognizable emojis."
        )
    )
    start: float = Field(description="Start time in seconds (when the emoji appears)")
    end: float = Field(description="End time in seconds (when the emoji disappears)")
    rationale: str = Field(
        description="Brief explanation of why this emoji fits this moment"
    )
    position: str = Field(
        description=(
            "Placement: 'top-left' or 'top-right'. Alternate sides for variety."
        )
    )


class EmojiPlan(BaseModel):
    """Full set of emoji overlays for a video."""

    picks: list[EmojiPick] = Field(description="List of emoji placements")


# ── Position mapping ─────────────────────────────────────────────────────────

POSITIONS = {
    "top-left": (40, 70),
    "top-right": (800, 70),
}

# ── Emoji PNG generation ─────────────────────────────────────────────────────

EMOJI_SIZE = 240
EMOJI_FONT_PATH = "/System/Library/Fonts/Apple Color Emoji.ttc"
# Apple Color Emoji only supports specific bitmap sizes; 160 is the largest
APPLE_EMOJI_RENDER_SIZE = 160


def generate_emoji_png(emoji_char: str, output_path: Path) -> Path:
    """Render an emoji character to a PNG with a white circular backing."""
    canvas_size = APPLE_EMOJI_RENDER_SIZE + 40  # padding for rendering
    img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype(EMOJI_FONT_PATH, APPLE_EMOJI_RENDER_SIZE)
    bbox = draw.textbbox((0, 0), emoji_char, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (canvas_size - tw) // 2 - bbox[0]
    y = (canvas_size - th) // 2 - bbox[1]
    draw.text((x, y), emoji_char, font=font, embedded_color=True)

    # Crop to content
    content_bbox = img.getbbox()
    if content_bbox:
        img = img.crop(content_bbox)

    # Create final image with white circular backing
    final = Image.new("RGBA", (EMOJI_SIZE, EMOJI_SIZE), (0, 0, 0, 0))
    final_draw = ImageDraw.Draw(final)

    # Semi-transparent white circle
    circle_margin = 10
    final_draw.ellipse(
        [
            circle_margin,
            circle_margin,
            EMOJI_SIZE - circle_margin,
            EMOJI_SIZE - circle_margin,
        ],
        fill=(255, 255, 255, 180),
    )

    # Paste emoji centered on the circle
    emoji_resized = img.resize((EMOJI_SIZE - 60, EMOJI_SIZE - 60), Image.LANCZOS)
    paste_x = (EMOJI_SIZE - emoji_resized.width) // 2
    paste_y = (EMOJI_SIZE - emoji_resized.height) // 2
    final.paste(emoji_resized, (paste_x, paste_y), emoji_resized)

    final.save(output_path)
    return output_path


# ── LLM emoji selection ──────────────────────────────────────────────────────


def pick_emojis(
    words: list[dict],
    max_emojis: int = 4,
    model: str = "google/gemini-3-flash-preview",
) -> list[EmojiPick]:
    """Use an LLM to select emoji overlay placements from a word-level transcript."""
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")

    video_duration = words[-1]["end"] if words else 0

    # Build a readable transcript with timestamps
    transcript_lines = []
    for w in words:
        transcript_lines.append(f"[{w['start']:.2f}s] {w['word']}")
    transcript_text = " ".join(transcript_lines)

    system_prompt = (
        "You are a short-form video editor adding emoji overlays to reinforce "
        "key spoken moments. Your job is to pick the BEST moments for emoji "
        "emphasis — moments where a visual icon adds meaning, humor, or energy.\n\n"
        "## RULES\n\n"
        f"- Pick exactly {max_emojis} emoji placements (or fewer if the content doesn't warrant it)\n"
        "- Each emoji should appear for 1.0–2.0 seconds\n"
        "- Use the EXACT timestamps from the transcript — the emoji should appear "
        "as the relevant word is spoken (start slightly before the word, end ~1.5s later)\n"
        "- Pick universally recognizable emojis (💰 🎯 📈 🔥 🤝 🏠 🛒 📢 etc.)\n"
        "- Do NOT pick emojis for filler words, transitions, or generic moments\n"
        "- DO pick emojis for: concrete nouns, numbers/money, strong verbs, "
        "emotional beats, key concepts\n"
        "- Alternate positions between 'top-left' and 'top-right' for visual variety\n"
        "- Space emojis at least 5 seconds apart — don't cluster them\n"
        "- Avoid the first 2 seconds (title card may be showing) and last 3 seconds (outro)\n"
        f"- Video duration: {video_duration:.1f}s\n"
    )

    llm = ChatOpenAI(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        timeout=120.0,
    ).with_structured_output(EmojiPlan)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Here is the word-level transcript ({len(words)} words, "
                f"{video_duration:.1f}s):\n\n{transcript_text}\n\n"
                f"Pick up to {max_emojis} emoji placements."
            )
        ),
    ]

    print(
        f"Analyzing transcript for emoji moments ({len(words)} words, "
        f"{video_duration:.0f}s)...",
        file=sys.stderr,
    )
    response = llm.invoke(messages)
    picks = response.picks

    # Sort by start time
    picks.sort(key=lambda p: p.start)

    return picks


# ── FFmpeg overlay ────────────────────────────────────────────────────────────


def overlay_emojis(
    video_path: Path,
    output_path: Path,
    picks: list[EmojiPick],
    emoji_pngs: dict[int, Path],
    video_duration: float,
) -> None:
    """Overlay emoji PNGs on the video using a single ffmpeg pass."""
    if not picks:
        # No emojis — just copy
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-c", "copy", str(output_path)],
            check=True,
        )
        return

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y", "-i", str(video_path)]

    # Add each emoji PNG as an input
    for i in range(len(picks)):
        cmd.extend(
            ["-loop", "1", "-t", str(int(video_duration) + 1), "-i", str(emoji_pngs[i])]
        )

    # Build filter_complex
    filters = []
    for i in range(len(picks)):
        filters.append(f"[{i + 1}:v]fps=30,format=yuva420p[e{i}]")

    # Chain overlays
    prev = "0:v"
    for i, pick in enumerate(picks):
        pos = POSITIONS.get(pick.position, POSITIONS["top-left"])
        is_last = i == len(picks) - 1
        out_label = "[v]" if is_last else f"[tmp{i}]"
        shortest = ":shortest=1" if is_last else ""
        filters.append(
            f"[{prev}][e{i}]overlay={pos[0]}:{pos[1]}:"
            f"enable='between(t,{pick.start:.2f},{pick.end:.2f})'{shortest}{out_label}"
        )
        prev = f"tmp{i}" if not is_last else "v"

    filter_complex = ";\n".join(filters)
    cmd.extend(["-filter_complex", filter_complex])
    cmd.extend(["-map", "[v]", "-map", "0:a"])
    cmd.extend(
        [
            "-c:v",
            "h264_videotoolbox",
            *H264_SOCIAL_COLOR_ARGS,
            "-b:v",
            "8M",
            "-c:a",
            "copy",
        ]
    )
    cmd.append(str(output_path))

    print(f"Overlaying {len(picks)} emojis...", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg error:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Output: {output_path}", file=sys.stderr)


# ── CLI ───────────────────────────────────────────────────────────────────────


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add emoji overlays at key moments in a video",
    )
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument("output", type=Path, help="Output video file")
    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="Word-level transcript JSON (if omitted, will transcribe the video)",
    )
    parser.add_argument(
        "--max-emojis",
        type=int,
        default=4,
        help="Maximum number of emoji overlays (default: 4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemini-3-flash-preview",
        help="LLM model for emoji selection (default: gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print emoji picks without rendering (for review)",
    )
    args = parser.parse_args()

    # Default --max-emojis from project/session settings if not passed on CLI
    if "--max-emojis" not in sys.argv:
        settings = load_settings(args.video)
        max_from_settings = settings.get("emojis", {}).get("max_emojis")
        if max_from_settings is not None:
            args.max_emojis = int(max_from_settings)

    if not args.video.exists():
        print(f"Error: Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    # Load or generate transcript
    if args.transcript:
        _, words = read_word_transcript_file(args.transcript)
        print(f"Loaded {len(words)} words from {args.transcript}", file=sys.stderr)
    else:
        print(f"Transcribing {args.video.name}...", file=sys.stderr)
        result = await get_transcript(args.video)
        words = result["words"]
        print(f"Transcribed: {len(words)} words", file=sys.stderr)

    video_duration = words[-1]["end"] if words else 0

    # Step 1: LLM picks emoji moments
    picks = pick_emojis(words, max_emojis=args.max_emojis, model=args.model)

    # Display picks
    print(f"\n{'─' * 60}", file=sys.stderr)
    print(f"Emoji plan ({len(picks)} overlays):", file=sys.stderr)
    for i, p in enumerate(picks, 1):
        print(
            f"  {i}. {p.emoji}  {p.start:.1f}–{p.end:.1f}s  ({p.position})  {p.rationale}",
            file=sys.stderr,
        )
    print(f"{'─' * 60}\n", file=sys.stderr)

    if args.dry_run:
        # Output as JSON for piping
        json.dump(
            [
                {
                    "emoji": p.emoji,
                    "start": p.start,
                    "end": p.end,
                    "position": p.position,
                    "rationale": p.rationale,
                }
                for p in picks
            ],
            sys.stdout,
            indent=2,
        )
        print()
        return

    # Step 2: Generate emoji PNGs
    tmp_dir = Path(tempfile.mkdtemp(prefix="emojis_"))
    emoji_pngs: dict[int, Path] = {}
    for i, pick in enumerate(picks):
        png_path = tmp_dir / f"emoji_{i}.png"
        generate_emoji_png(pick.emoji, png_path)
        emoji_pngs[i] = png_path
        print(f"  Generated: {pick.emoji} → {png_path.name}", file=sys.stderr)

    # Step 3: Overlay onto video
    overlay_emojis(args.video, args.output, picks, emoji_pngs, video_duration)


if __name__ == "__main__":
    asyncio.run(main())
