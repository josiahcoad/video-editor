#!/usr/bin/env python3
"""
Propose strong hooks for a video based on its transcript.

Analyzes the transcript and generates hook options in two categories:
1. Cold opens — soundbites pulled directly from the video (with timestamps)
2. Written hooks — original opening lines that could be delivered to camera
   or overlaid as text

Usage:
  propose_hooks.py --transcript <words.json>
  propose_hooks.py --transcript <words.json> --voice <voice.md>
  propose_hooks.py --transcript <words.json> --voice <voice.md> --count 5

Requires:
  OPENROUTER_API_KEY env var
"""

import argparse
import json
import os
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class ColdOpen(BaseModel):
    """A soundbite pulled from the video to use as a cold open."""

    text: str = Field(description="The exact quote from the transcript")
    start: float = Field(description="Start timestamp in seconds")
    end: float = Field(description="End timestamp in seconds")
    why: str = Field(description="Why this works as a hook (1 sentence)")
    score: int = Field(
        description="Hook strength 1-10 (10 = impossible to scroll past)"
    )


class WrittenHook(BaseModel):
    """An original hook line to deliver on camera or overlay as text."""

    text: str = Field(description="The hook text (1-2 sentences max)")
    angle: str = Field(
        description="The angle/strategy this hook uses (e.g. 'contrarian', "
        "'concrete number', 'story lead-in', 'framework naming')"
    )
    why: str = Field(description="Why this works as a hook (1 sentence)")
    score: int = Field(
        description="Hook strength 1-10 (10 = impossible to scroll past)"
    )


class HookProposals(BaseModel):
    """All hook proposals for a video."""

    cold_opens: list[ColdOpen] = Field(
        description="Soundbites from the video to use as cold opens"
    )
    written_hooks: list[WrittenHook] = Field(
        description="Original hook lines (not from transcript)"
    )


def load_transcript(transcript_path: Path) -> tuple[str, list[dict]]:
    """Load transcript, returning (full_text, words_list).

    Supports both word-level JSON and plain text files.
    """
    content = transcript_path.read_text()

    if transcript_path.suffix == ".json":
        words = json.loads(content)
        if isinstance(words, dict) and "words" in words:
            words = words["words"]
        full_text = " ".join(w.get("word", w.get("text", "")) for w in words)
        return full_text, words
    else:
        return content, []


def build_timestamped_transcript(words: list[dict]) -> str:
    """Build a transcript string with timestamps for the LLM to reference."""
    if not words:
        return ""

    lines = []
    current_line: list[str] = []
    line_start = words[0].get("start", 0)

    for i, w in enumerate(words):
        word_text = w.get("word", w.get("text", ""))
        current_line.append(word_text)

        # Break into lines of ~10 words for readability
        if len(current_line) >= 10 or i == len(words) - 1:
            end_time = w.get("end", w.get("start", 0))
            text = " ".join(current_line)
            lines.append(f"[{line_start:.1f}s - {end_time:.1f}s] {text}")
            current_line = []
            if i + 1 < len(words):
                line_start = words[i + 1].get("start", end_time)

    return "\n".join(lines)


def propose_hooks(
    transcript_text: str,
    words: list[dict],
    voice_guidelines: str | None = None,
    count: int = 5,
    model: str = "google/gemini-2.5-flash",
) -> HookProposals:
    """Generate hook proposals using an LLM."""
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        print("Error: OPENROUTER_API_KEY env var is required", file=sys.stderr)
        sys.exit(1)

    llm = ChatOpenAI(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        temperature=0.8,
    ).with_structured_output(HookProposals)

    # Build timestamped transcript if we have words
    timestamped = build_timestamped_transcript(words)

    voice_section = ""
    if voice_guidelines:
        voice_section = (
            f"\n\nVOICE GUIDELINES (hooks must match this style):\n"
            f"{voice_guidelines}\n"
        )

    system_prompt = (
        "You are an expert short-form video editor and hook strategist. "
        "Your job is to analyze a video transcript and propose the strongest "
        "possible hooks — both cold opens (pulled from the video) and original "
        "written hooks.\n\n"
        "WHAT MAKES A GREAT HOOK:\n"
        "1. SPECIFIC over generic — a dollar amount, a job title, a named "
        "framework beats any abstract statement\n"
        "2. Creates an OPEN LOOP — the viewer needs to keep watching to "
        "resolve a question or tension\n"
        "3. PATTERN INTERRUPT — says something the viewer doesn't expect\n"
        "4. EMOTIONAL — makes the viewer feel curiosity, surprise, or 'wait, "
        "what?'\n"
        "5. FAST — gets to the punch in under 3 seconds / 10 words\n\n"
        "COLD OPEN RULES:\n"
        "- Must be an EXACT quote from the transcript (don't paraphrase)\n"
        "- 3-8 seconds long — a complete thought but leaves context missing\n"
        "- The best cold opens are moments where the speaker says something "
        "surprising, names a big number, tells a mini-story, or makes a bold "
        "claim\n"
        "- Provide accurate start/end timestamps\n"
        "- DO NOT pick the opening of the video — the whole point is to pull "
        "something from LATER\n\n"
        "WRITTEN HOOK RULES:\n"
        "- Original text, NOT from the transcript\n"
        "- Could be delivered on camera as a re-recorded intro or shown as "
        "text overlay on the first frame\n"
        "- Each hook must use a DIFFERENT angle/strategy\n"
        "- 1-2 sentences max\n"
        "- Must sound like something a real person would say on camera, not "
        "a copywriter's headline\n\n"
        "SCORING:\n"
        "- 1-3: Weak. Generic, could apply to any video\n"
        "- 4-6: Decent. Has a specific detail but missing emotional punch\n"
        "- 7-8: Strong. Specific + creates curiosity + matches voice\n"
        "- 9-10: Exceptional. Would stop someone mid-scroll. Reserve these.\n"
        "- Be HONEST with scores. Most hooks are 5-7. A 9+ is rare.\n"
        f"{voice_section}"
    )

    human_content = f"Propose {count} cold opens and {count} written hooks "
    if timestamped:
        human_content += (
            f"for this video.\n\n" f"TIMESTAMPED TRANSCRIPT:\n{timestamped}"
        )
    else:
        human_content += f"for this video.\n\n" f"TRANSCRIPT:\n{transcript_text}"

    result = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content),
        ]
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Propose strong hooks for a video")
    parser.add_argument(
        "--transcript",
        type=Path,
        required=True,
        help="Word-level transcript JSON or plain text file",
    )
    parser.add_argument(
        "--voice",
        type=Path,
        default=None,
        help="Voice guidelines markdown file",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of hooks to propose per category (default: 5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemini-2.5-flash",
        help="OpenRouter model ID (default: google/gemini-2.5-flash)",
    )

    args = parser.parse_args()

    if not args.transcript.exists():
        print(f"Error: Transcript not found: {args.transcript}", file=sys.stderr)
        sys.exit(1)

    # Load transcript
    transcript_text, words = load_transcript(args.transcript)
    print(
        f"📝 Loaded transcript: {len(words)} words"
        if words
        else f"📝 Loaded transcript: {len(transcript_text)} chars"
    )

    # Load voice guidelines
    voice_guidelines = None
    if args.voice:
        if not args.voice.exists():
            print(f"Error: Voice file not found: {args.voice}", file=sys.stderr)
            sys.exit(1)
        voice_guidelines = args.voice.read_text()
        print(f"🎙️  Voice: {args.voice.name}")

    # Generate hooks
    print(f"🔍 Analyzing transcript for hooks (model: {args.model})...")
    proposals = propose_hooks(
        transcript_text, words, voice_guidelines, args.count, args.model
    )

    # Display results
    print("\n" + "=" * 60)
    print("🎬 COLD OPENS (soundbites from the video)")
    print("=" * 60)
    for i, hook in enumerate(proposals.cold_opens, 1):
        print(f"\n{i}. [{hook.start:.1f}s - {hook.end:.1f}s] (score: {hook.score}/10)")
        print(f'   "{hook.text}"')
        print(f"   → {hook.why}")

    print("\n" + "=" * 60)
    print("✍️  WRITTEN HOOKS (original lines)")
    print("=" * 60)
    for i, hook in enumerate(proposals.written_hooks, 1):
        print(f"\n{i}. [{hook.angle}] (score: {hook.score}/10)")
        print(f'   "{hook.text}"')
        print(f"   → {hook.why}")

    # Also output as JSON for programmatic use
    print("\n" + "=" * 60)
    print("📋 JSON OUTPUT")
    print("=" * 60)
    print(json.dumps(proposals.model_dump(), indent=2))


if __name__ == "__main__":
    main()
