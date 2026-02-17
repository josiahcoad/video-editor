#!/usr/bin/env python3
"""
Add an infographic overlay at a key moment in a video.

Uses an LLM to analyze the transcript, identify the best concept to
visualize, pick the right timestamp, and generate a structured layout.
Renders the infographic programmatically with Pillow (for pixel-perfect
text control), then overlays it via ffmpeg.

Usage:
  # Dry-run — review the concept, layout, and timing before rendering:
  python -m src.edit.add_infographic <video> <output> \\
    --transcript <words.json> --dry-run

  # Full render:
  python -m src.edit.add_infographic <video> <output> \\
    --transcript <words.json> --duration 7

  # With custom palette (overrides defaults):
  python -m src.edit.add_infographic <video> <output> \\
    --transcript <words.json> \\
    --palette '{"bg": "#FAF5EF", "accent": "#D4856B", "secondary": "#C4AA8A", "text": "#3B2F2F", "muted": "#8B7D6B"}'

Templates:
  funnel    — vertical flow: A → B → C (e.g. trust gap, sales pipeline)
  compare   — side-by-side: X vs Y
  list      — 3-5 key points with icons
  bigstat   — one big number/word with context

The LLM picks the template. Override with --template if needed.
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
from PIL import Image
from pydantic import BaseModel, Field

from .encode_common import H264_SOCIAL_COLOR_ARGS
from .generate_image import generate_image as gemini_generate_image
from .get_transcript import get_transcript, read_word_transcript_file
from .settings_loader import load_settings


# ── Default warm/approachable palette ─────────────────────────────────────────

DEFAULT_PALETTE = {
    "bg": "#FAF5EF",  # warm cream
    "accent": "#D4856B",  # warm coral / terracotta
    "secondary": "#C4AA8A",  # tan
    "text": "#3B2F2F",  # dark brown (not black)
    "muted": "#8B7D6B",  # muted brown for subtitles
    "white": "#FFFFFF",
}

# ── Video dimensions (9:16 portrait) ─────────────────────────────────────────

W, H = 1080, 1920


# ── Pydantic models for structured LLM output ────────────────────────────────


class FunnelItem(BaseModel):
    """A single step in a funnel infographic."""

    label: str = Field(description="Short label for this step (1-3 words)")
    sublabel: str = Field(description="Brief description (3-8 words)")


class CompareItem(BaseModel):
    """One side of a comparison."""

    title: str = Field(description="Column header (1-2 words)")
    points: list[str] = Field(description="2-4 short bullet points")


class ListItem(BaseModel):
    """A single item in a list infographic."""

    text: str = Field(description="The point (5-10 words)")
    icon: str = Field(description="A single emoji that represents this point")


class InfographicPlan(BaseModel):
    """Full infographic plan from the LLM."""

    concept: str = Field(description="The key concept being visualized (2-5 words)")
    title: str = Field(
        description="Bold headline for the infographic (2-6 words, ALL CAPS)"
    )
    template: str = Field(
        description="Template type: 'funnel', 'compare', 'list', or 'bigstat'"
    )
    start_time: float = Field(
        description="When to show the infographic (seconds into the video)"
    )
    rationale: str = Field(
        description="Why this concept benefits from a visual + why this timestamp"
    )

    # Template-specific fields (LLM fills only the relevant one)
    funnel_items: list[FunnelItem] | None = Field(
        default=None,
        description="For 'funnel' template: 2-4 steps in the flow (top to bottom)",
    )
    compare_left: CompareItem | None = Field(
        default=None, description="For 'compare' template: left column"
    )
    compare_right: CompareItem | None = Field(
        default=None, description="For 'compare' template: right column"
    )
    list_items: list[ListItem] | None = Field(
        default=None, description="For 'list' template: 3-5 items"
    )
    bigstat_value: str | None = Field(
        default=None, description="For 'bigstat' template: the big number or keyword"
    )
    bigstat_context: str | None = Field(
        default=None,
        description="For 'bigstat' template: context line below the big value",
    )
    tagline: str | None = Field(
        default=None, description="Optional footer tagline (5-10 words)"
    )


# ── Gemini image generation ───────────────────────────────────────────────────


def _build_infographic_prompt(plan: InfographicPlan, palette: dict) -> str:
    """Build a rich Gemini image-generation prompt from the LLM plan."""

    palette_desc = (
        f"Color palette: warm cream background ({palette['bg']}), "
        f"coral/terracotta accent ({palette['accent']}), "
        f"tan secondary ({palette['secondary']}), "
        f"dark brown text ({palette['text']}). "
    )

    style = (
        "Create a visually striking, modern infographic for a 9:16 portrait phone screen. "
        "This should look like a premium Instagram infographic — NOT a PowerPoint slide. "
        "Use bold typography as the primary design element. "
        "Include subtle geometric shapes, gradients, or abstract design elements for visual interest. "
        "Keep generous whitespace. Make it feel warm, approachable, and professional. "
        "The design should be instantly readable at phone size in 3 seconds. "
        f"{palette_desc}"
        "No stock photos, no clip art, no realistic people. Typography and shapes only. "
    )

    if plan.template == "funnel" and plan.funnel_items:
        items = plan.funnel_items
        steps_text = " → ".join(f"{item.label} ({item.sublabel})" for item in items)
        content = (
            f"Layout: A vertical flow/funnel diagram with {len(items)} steps. "
            f'Headline at top: "{plan.title}". '
            f"Steps flowing downward with arrows or connecting lines between them: {steps_text}. "
            "Each step should be in a distinct rounded shape or card. "
            "Make the flow visually obvious with arrows or gradient progression. "
        )
    elif plan.template == "compare" and plan.compare_left and plan.compare_right:
        left = plan.compare_left
        right = plan.compare_right
        left_pts = ", ".join(left.points)
        right_pts = ", ".join(right.points)
        content = (
            f"Layout: A side-by-side comparison with a clear visual divide. "
            f'Headline at top: "{plan.title}". '
            f'LEFT column titled "{left.title}" with points: {left_pts}. '
            f'RIGHT column titled "{right.title}" with points: {right_pts}. '
            "Use contrasting colors or shapes for each side. "
            "Add a bold 'VS' or dividing element between columns. "
            "Make the contrast visually dramatic — the viewer should feel the difference. "
        )
    elif plan.template == "list" and plan.list_items:
        items_text = "; ".join(f"{item.icon} {item.text}" for item in plan.list_items)
        content = (
            f"Layout: A numbered/icon list with {len(plan.list_items)} items. "
            f'Headline at top: "{plan.title}". '
            f"Items: {items_text}. "
            "Each item in its own row with the emoji icon on the left and text on the right. "
            "Use subtle card backgrounds or alternating accent colors for each row. "
        )
    elif plan.template == "bigstat":
        content = (
            f"Layout: One massive, bold statistic or keyword dominating the center. "
            f'Headline at top: "{plan.title}". '
            f'The BIG value: "{plan.bigstat_value}" — make this HUGE and in the accent color. '
            f'Context below: "{plan.bigstat_context}". '
            "Add subtle geometric accents (circles, lines) around the big number for emphasis. "
        )
    else:
        content = (
            f'Layout: A clean infographic visualizing the concept "{plan.concept}". '
            f'Headline: "{plan.title}". '
            "Use bold typography and geometric shapes to convey the idea visually. "
        )

    tagline = ""
    if plan.tagline:
        tagline = f'Footer tagline at the bottom: "{plan.tagline}". '

    text_warning = (
        "CRITICAL: Spell every word EXACTLY as provided. "
        "Do NOT add extra words, change spelling, or hallucinate text. "
        "Double-check every letter. "
    )

    return f"{style}{content}{tagline}{text_warning}"


async def render_infographic_gemini(
    plan: InfographicPlan, palette: dict
) -> Image.Image:
    """Render an infographic using Gemini image generation."""
    prompt = _build_infographic_prompt(plan, palette)
    print("Generating infographic with Gemini 3 Pro...", file=sys.stderr)

    image, text = await gemini_generate_image(
        prompt=prompt,
        aspect_ratio="9:16",
        model="gemini-3-pro-image-preview",
    )

    # Ensure it's the right size for overlay (1080x1920)
    if image.size != (W, H):
        image = image.resize((W, H), Image.Resampling.LANCZOS)

    return image


# ── LLM concept selection ────────────────────────────────────────────────────


def plan_infographic(
    words: list[dict],
    model: str = "google/gemini-3-flash-preview",
) -> InfographicPlan:
    """Use an LLM to plan the infographic concept, layout, and timing."""
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")

    video_duration = words[-1]["end"] if words else 0

    # Build readable transcript with timestamps
    transcript_lines = []
    for w in words:
        transcript_lines.append(f"[{w['start']:.2f}s] {w['word']}")
    transcript_text = " ".join(transcript_lines)

    system_prompt = (
        "You are a short-form video editor planning an infographic overlay. "
        "Your job is to find the ONE concept in this video that most benefits "
        "from a visual explanation — something that's hard to grasp from audio alone.\n\n"
        "## RULES\n\n"
        "- Pick exactly ONE concept to visualize\n"
        "- The infographic should be **instantly understandable in 3 seconds** on a phone\n"
        "- Maximum 15 words of text on the entire graphic (fewer is better)\n"
        "- The title should be 2-6 words, ALL CAPS\n"
        "- Pick a template that fits the concept:\n"
        "  - 'funnel': for processes, pipelines, cause→effect flows (2-4 steps)\n"
        "  - 'compare': for X vs Y, side-by-side contrasts\n"
        "  - 'list': for key points, tips, features (3-5 items)\n"
        "  - 'bigstat': for a dramatic number, percentage, or single keyword\n"
        "- start_time should be when the speaker BEGINS discussing this concept "
        "(the infographic reinforces the spoken explanation)\n"
        f"- Video duration: {video_duration:.1f}s\n"
        "- Avoid placing in the first 3 seconds (title card) or last 3 seconds (outro)\n"
        "- The infographic will display for 5-8 seconds, so pick a moment where the "
        "speaker discusses the concept for at least that long\n\n"
        "## CONTENT STYLE\n\n"
        "- Warm, approachable tone — think Instagram infographic from a business coach\n"
        "- Labels should be plain English (no jargon)\n"
        "- Sublabels/context should add meaning, not restate the label\n"
        "- If using 'compare', the points should be concrete differences, not vague\n"
        "- If using 'funnel', each step should be a clear progression\n"
    )

    llm = ChatOpenAI(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        timeout=120.0,
    ).with_structured_output(InfographicPlan)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Here is the word-level transcript ({len(words)} words, "
                f"{video_duration:.1f}s):\n\n{transcript_text}\n\n"
                "Plan one infographic overlay."
            )
        ),
    ]

    print(
        f"Analyzing transcript for infographic concept ({len(words)} words, "
        f"{video_duration:.0f}s)...",
        file=sys.stderr,
    )
    plan = llm.invoke(messages)
    return plan


# ── FFmpeg overlay ────────────────────────────────────────────────────────────


def _get_video_dimensions(video_path: Path) -> tuple[int, int]:
    """Return (width, height) of the first video stream."""
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
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    w, h = result.stdout.strip().split(",")
    return int(w), int(h)


def overlay_infographic(
    video_path: Path,
    output_path: Path,
    infographic_path: Path,
    start_time: float,
    duration: float,
    video_duration: float,
) -> None:
    """Overlay the infographic on the video at a specific timestamp.

    The infographic image is scaled to match the video dimensions so it is never
    cropped (video may be 608x1080 etc.; infographic is generated at 1080x1920).
    """
    end_time = start_time + duration
    vw, vh = _get_video_dimensions(video_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-loop",
        "1",
        "-t",
        str(int(video_duration) + 1),
        "-i",
        str(infographic_path),
        "-filter_complex",
        (
            f"[1:v]scale={vw}:{vh}:force_original_aspect_ratio=decrease,"
            f"pad={vw}:{vh}:(ow-iw)/2:(oh-ih)/2,fps=30,format=yuva420p[ig];"
            f"[0:v][ig]overlay=0:0:"
            f"enable='between(t,{start_time:.2f},{end_time:.2f})':shortest=1[v]"
        ),
        "-map",
        "[v]",
        "-map",
        "0:a",
        "-c:v",
        "h264_videotoolbox",
        *H264_SOCIAL_COLOR_ARGS,
        "-b:v",
        "8M",
        "-c:a",
        "copy",
        str(output_path),
    ]

    print(
        f"Overlaying infographic at {start_time:.1f}–{end_time:.1f}s...",
        file=sys.stderr,
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg error:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"Output: {output_path}", file=sys.stderr)


# ── CLI ───────────────────────────────────────────────────────────────────────


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add an infographic overlay at a key moment in a video",
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
        "--duration",
        type=float,
        default=7.0,
        help="How long to show the infographic in seconds (default: 7)",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        choices=["funnel", "compare", "list", "bigstat"],
        help="Force a specific template (default: LLM chooses)",
    )
    parser.add_argument(
        "--palette",
        type=str,
        default=None,
        help='JSON object with color overrides: {"bg": "#FAF5EF", "accent": "#D4856B", ...}',
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemini-3-flash-preview",
        help="LLM model for concept selection (default: gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the infographic plan + save the image to /tmp without overlaying",
    )
    args = parser.parse_args()

    # Defaults from project/session settings if not passed on CLI
    if "--duration" not in sys.argv or "--palette" not in sys.argv:
        settings = load_settings(args.video)
        infographic_cfg = settings.get("infographic") or {}
        if "--duration" not in sys.argv:
            d = infographic_cfg.get("duration")
            if d is not None:
                args.duration = float(d)
        if "--palette" not in sys.argv:
            pal = infographic_cfg.get("palette")
            if pal is not None:
                args.palette = json.dumps(pal) if isinstance(pal, dict) else str(pal)

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

    # Build palette
    palette = dict(DEFAULT_PALETTE)
    if args.palette:
        overrides = json.loads(args.palette)
        palette.update(overrides)

    # Step 1: LLM plans the infographic
    plan = plan_infographic(words, model=args.model)

    # Override template if specified
    if args.template:
        plan.template = args.template

    # Display plan
    print(f"\n{'─' * 60}", file=sys.stderr)
    print(f"Infographic plan:", file=sys.stderr)
    print(f"  Concept:  {plan.concept}", file=sys.stderr)
    print(f"  Title:    {plan.title}", file=sys.stderr)
    print(f"  Template: {plan.template}", file=sys.stderr)
    print(
        f"  Timing:   {plan.start_time:.1f}–{plan.start_time + args.duration:.1f}s",
        file=sys.stderr,
    )
    print(f"  Rationale: {plan.rationale}", file=sys.stderr)

    if plan.template == "funnel" and plan.funnel_items:
        print(f"  Steps:", file=sys.stderr)
        for i, item in enumerate(plan.funnel_items, 1):
            print(f"    {i}. {item.label} — {item.sublabel}", file=sys.stderr)
    elif plan.template == "compare":
        if plan.compare_left:
            print(
                f"  Left:  {plan.compare_left.title}: {', '.join(plan.compare_left.points)}",
                file=sys.stderr,
            )
        if plan.compare_right:
            print(
                f"  Right: {plan.compare_right.title}: {', '.join(plan.compare_right.points)}",
                file=sys.stderr,
            )
    elif plan.template == "list" and plan.list_items:
        print(f"  Items:", file=sys.stderr)
        for item in plan.list_items:
            print(f"    {item.icon} {item.text}", file=sys.stderr)
    elif plan.template == "bigstat":
        print(f"  Value:   {plan.bigstat_value}", file=sys.stderr)
        print(f"  Context: {plan.bigstat_context}", file=sys.stderr)

    if plan.tagline:
        print(f"  Tagline: {plan.tagline}", file=sys.stderr)
    print(f"{'─' * 60}\n", file=sys.stderr)

    # Step 2: Render the infographic via Gemini image gen
    infographic = await render_infographic_gemini(plan, palette)

    if args.dry_run:
        # Save preview image and output plan as JSON
        preview_path = Path(tempfile.gettempdir()) / "infographic_preview.png"
        infographic.save(str(preview_path), quality=95)
        print(f"Preview saved: {preview_path}", file=sys.stderr)

        json.dump(
            {
                "concept": plan.concept,
                "title": plan.title,
                "template": plan.template,
                "start_time": plan.start_time,
                "duration": args.duration,
                "rationale": plan.rationale,
                "preview_image": str(preview_path),
            },
            sys.stdout,
            indent=2,
        )
        print()
        return

    # Step 3: Save infographic and overlay
    tmp_dir = Path(tempfile.mkdtemp(prefix="infographic_"))
    infographic_path = tmp_dir / "infographic.png"
    infographic.save(str(infographic_path), quality=95)
    print(
        f"Rendered: {infographic_path} ({infographic.width}x{infographic.height})",
        file=sys.stderr,
    )

    overlay_infographic(
        args.video,
        args.output,
        infographic_path,
        plan.start_time,
        args.duration,
        video_duration,
    )

    # Write plan sidecar so downstream steps can require including the infographic range
    plan_path = args.output.with_suffix(".plan.json")
    plan_path.write_text(
        json.dumps(
            {
                "start_time": plan.start_time,
                "duration": args.duration,
                "end_time": plan.start_time + args.duration,
                "concept": plan.concept,
                "title": plan.title,
            },
            indent=2,
        )
    )
    print(f"Plan written: {plan_path}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
