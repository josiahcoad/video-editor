#!/usr/bin/env python3
"""
Generate a title and caption for a video segment.

Reads a transcript and voice guidelines, then produces copy tailored
to a specific platform.

Usage:
  # Single platform:
  write_copy.py --transcript <words.json> --voice <voice.md>
  write_copy.py --transcript <words.json> --voice <voice.md> --platform linkedin
  write_copy.py --transcript <words.json> --voice <voice.md> --platform twitter --count 5

  # Multiple platforms with per-platform voice files (batch mode):
  write_copy.py --transcript <words.json> \
    --platforms short:<client>/editing/voices/marky_video.md twitter:<client>/editing/voices/marky_twitter.md linkedin:<client>/editing/voices/marky_linkedin.md

  Batch mode outputs a keyed JSON object: {"short": {...}, "twitter": {...}, "linkedin": {...}}

Requires:
  OPENROUTER_API_KEY env var
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# Phrases that indicate generic AI/corporate writing.
# Checked case-insensitively after generation; triggers a retry.
BANNED_PHRASES = [
    "crystal clear",
    "crystal-clear",
    "game-changer",
    "game changer",
    "full potential",
    "with purpose and power",
    "powerful execution",
    "across the board",
    "right from the start",
    "it's amazing how",
    "imagine the collective",
    "working smarter not harder",
    "what's your take",
    "what do you think",
    "here's the secret",
    "let's dive in",
    "let me break it down",
    "here's the thing",
    "ever wonder why",
    "pretty smart right",
    "rowing in the same direction",
    "non-negotiable",
    "stifling potential",
]

# Verb-stem patterns that catch all conjugations (unlock, unlocks, unlocked, unlocking)
_BANNED_VERB_PATTERNS = [
    r"\bunlock\w*\b",
    r"\bunleash\w*\b",
    r"\bempower\w*\b",
    r"\btransform\w*\b",
]

# Compile into a single regex for fast checking
_phrase_patterns = [re.escape(p) for p in BANNED_PHRASES]
_all_patterns = _phrase_patterns + _BANNED_VERB_PATTERNS
_BANNED_RE = re.compile(
    "|".join(_all_patterns),
    re.IGNORECASE,
)


class CopyOutput(BaseModel):
    """A single title + caption pair."""

    title: str = Field(description="Short, punchy title (2-8 words, no quotes)")
    caption: str = Field(description="Post caption/body text")


class CopyBatch(BaseModel):
    """Multiple copy options."""

    options: list[CopyOutput] = Field(description="List of title+caption pairs")


PLATFORM_GUIDANCE = {
    "short": (
        "This is for a short-form video (TikTok, Instagram Reel, YouTube Short).\n"
        "- Title: 3-8 words. Make a CLAIM or provoke CURIOSITY. "
        "Good: 'Alignment: The Missing Piece'. Bad: 'Leadership Tips'.\n"
        "- Caption: 4-8 lines. Hook in line 1 — the line that stops the scroll.\n"
        "- Short sentences. Line breaks between ideas. One idea per line.\n"
        "- End with a sharp, specific question or a crisp takeaway — NOT 'What do you think?'\n"
        "- No hashtags unless genuinely relevant (max 3)."
    ),
    "linkedin": (
        "This is for a LinkedIn thought leadership post.\n"
        "- Title: not needed (set to empty string).\n"
        "- Caption: 150-300 words. Written for credibility and engagement.\n"
        "- Open with a strong hook (1-2 lines that stop the scroll). "
        "The hook should reframe the topic — not summarize it.\n"
        "- Establish credibility early (years of experience, specific role, what you've seen).\n"
        "- Share a genuine insight, framework, or pattern — not just the transcript's argument.\n"
        "- Give the reader a mental model or 'number' to hold onto "
        "(e.g., 'three things', 'one pattern I keep seeing').\n"
        "- Use short paragraphs (1-3 sentences) with line breaks.\n"
        "- End with a specific question that invites real responses "
        "(not 'What do you think?' — ask something they'd actually answer).\n"
        "- No hashtags. No emojis. No 'I'm excited to announce'."
    ),
    "linkedin_reel": (
        "This is for a LinkedIn Reel (short video post with caption).\n"
        "- Title: 3-8 words. This is the text overlay on the video — "
        "make a bold claim or pose a tension. "
        "Good: 'Why Your Leadership Team Isn't Aligned'. "
        "Bad: 'Thoughts on Leadership'.\n"
        "- Caption: 40-100 words. Professional but human.\n"
        "- Open with a hook that reframes the topic — not a summary of the video.\n"
        "- Add 1-2 lines of context or a specific insight that complements (not repeats) "
        "what's said in the video.\n"
        "- End with a specific question or call to action.\n"
        "- No hashtags. No emojis."
    ),
    "twitter": (
        "This is for a Twitter/X post (no video).\n"
        "- Title: not needed (set to empty string).\n"
        "- Caption: under 280 characters. One sharp idea.\n"
        "- Distill the single most interesting insight from the transcript.\n"
        "- Punchy, direct, slightly provocative if appropriate.\n"
        "- No hashtags. No threads. Just one standalone thought."
    ),
    "facebook": (
        "This is for a Facebook post.\n"
        "- Title: not needed (set to empty string).\n"
        "- Caption: 100-200 words. Conversational, slightly more casual than LinkedIn.\n"
        "- Tell a mini-story or share a practical takeaway.\n"
        "- End with a question or call to comment.\n"
        "- Light on hashtags (0-2 max)."
    ),
    "carousel": (
        "This is for an Instagram carousel caption (the images are separate).\n"
        "- Title: the carousel hook (what makes someone stop and swipe).\n"
        "- Caption: 50-150 words. Summarize the carousel's content.\n"
        "- End with 'Swipe through' or 'Save this for later'.\n"
        "- 3-5 relevant hashtags at the end."
    ),
}


def load_transcript(path: Path) -> str:
    """Load transcript from a words JSON, utterances JSON, or plain text file."""
    text = path.read_text()

    if path.suffix == ".json":
        data = json.loads(text)
        # Word-level JSON: list of {"word": ..., "start": ..., "end": ...}
        if isinstance(data, list) and len(data) > 0:
            if "word" in data[0]:
                return " ".join(w["word"] for w in data)
            # Utterance-level: list of {"transcript": ...}
            if "transcript" in data[0]:
                return " ".join(u["transcript"] for u in data)
        return text
    else:
        return text


def _normalize_quotes(text: str) -> str:
    """Normalize curly/smart quotes to straight quotes for matching."""
    return (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def _find_banned(text: str) -> list[str]:
    """Return list of banned phrases found in text."""
    normalized = _normalize_quotes(text)
    return list(set(m.group() for m in _BANNED_RE.finditer(normalized)))


def generate_copy(
    transcript: str,
    voice_guidelines: str,
    platform: str,
    count: int,
    model: str = "anthropic/claude-opus-4.6",
    max_retries: int = 3,
    search_context: list[str] | None = None,
    video_title: str | None = None,
) -> list[CopyOutput]:
    """Generate title+caption pairs using an LLM.

    Args:
        search_context: Optional list of related search queries (from ATP)
            that people actually type. Woven into captions for discoverability.
        video_title: Optional pre-determined video title. When set, the LLM
            uses it as the post title (or a light adaptation) instead of
            generating one from scratch.

    Retries up to max_retries times if banned phrases are detected in output.
    """
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        print("Error: OPENROUTER_API_KEY env var is required", file=sys.stderr)
        sys.exit(1)

    platform_guide = PLATFORM_GUIDANCE.get(platform, PLATFORM_GUIDANCE["short"])

    llm = ChatOpenAI(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        temperature=0.85,
    ).with_structured_output(CopyBatch)

    system_prompt = (
        "You are an expert social media copywriter. Your job is to take spoken "
        "content (a transcript) and turn it into compelling written posts.\n\n"
        "CRITICAL: You are REFRAMING ideas, not quoting the transcript. "
        "A good post is INSPIRED BY what the speaker said — it extracts the core "
        "insight and expresses it the way a skilled writer would compose a post "
        "from scratch. A bad post reads like a transcript summary.\n\n"
        f"VOICE GUIDELINES (match this person's tone and style):\n{voice_guidelines}\n\n"
        f"PLATFORM:\n{platform_guide}\n\n"
        "WRITING RULES:\n"
        "1. REFRAME, don't regurgitate. Extract the speaker's IDEAS, FRAMEWORKS, "
        "and INSIGHTS — then express them as fresh, compelling copy. If you find "
        "yourself copying full sentences from the transcript, you're doing it wrong.\n"
        "2. Keep the speaker's KEY TERMS — specific names, numbers, and distinctive "
        "phrases that are part of their brand (e.g., 'FORWARD Executive Performance "
        "Sprints'). But rewrite the surrounding sentences.\n"
        "3. Be SPECIFIC and CONCRETE. Preserve actual numbers, steps, and details "
        "from the transcript. Never generalize what the speaker made specific.\n"
        "4. The HOOK is everything. The first 1-2 lines must stop the scroll. "
        "Reframe the topic with a surprising angle, a pattern, or a contrarian take.\n"
        "5. Give the reader STRUCTURE to hold onto: 'three things', 'one pattern', "
        "'the real reason'. Frameworks are shareable.\n"
        "6. VOICE CHECK: After drafting, scan every sentence for 'corporate polish'. "
        "If a phrase sounds like it belongs in a company all-hands deck or a "
        "motivational poster, cut it. Real founders don't say 'unlock potential', "
        "'crystal clear', 'transform your team', or 'with purpose and power'. "
        "They say specific, concrete things. Replace ANY corporate-motivational "
        "filler with a specific detail, a number, or silence (just delete it).\n"
        "7. End with a SPECIFIC question or CTA from the voice guidelines. "
        "NOT 'Where do you see gaps?' or 'Which area could bring impact?' — "
        "those are generic. Write a question ONLY the reader of this specific "
        "post could answer. Or use a conditional CTA: 'If you're [specific "
        "situation], [specific action].'\n"
        "8. Each option MUST take a genuinely DIFFERENT angle on the content — "
        "different hook, different framing, different structure. Not just rephrases.\n"
        "9. ANTI-AI TEST: Read each post and ask — could you tell this was "
        "AI-generated? If yes, rewrite. Signs of AI writing: every paragraph "
        "is the same length, overly balanced sentence structure, hedging "
        "('it's not just about X, it's about Y'), listing three parallel items "
        "with the same structure. Real people write messily — some paragraphs "
        "are one sentence, some are three. Mix it up.\n"
        "10. Follow the VOICE GUIDELINES closely. Use the voice file's patterns, "
        "sentence structures, and calls to action. If the voice uses dashes, "
        "gratitude openers, or conditional CTAs — use those.\n\n"
        "BAD EXAMPLE (transcript regurgitation — about a DIFFERENT topic):\n"
        "Transcript: 'I've been doing meal prep for 10 years and I found that most "
        "people fail because they try to cook 15 different recipes. Just pick 3 proteins "
        "and 2 carbs and rotate them.'\n"
        "Bad post: 'I've been meal prepping for 10 years and found that most people fail "
        "because they try too many recipes. Pick 3 proteins and 2 carbs and rotate.'\n"
        "^ This just paraphrases the transcript sentence by sentence.\n\n"
        "GOOD EXAMPLE (reframed — same topic):\n"
        "Good post: 'The meal prep advice nobody wants to hear: eat the same 5 ingredients "
        "on repeat.\n\nI know — boring. But after a decade of Sunday cooking sessions, "
        "every person I've seen quit meal prep quit because they made it too complicated.\n\n"
        "3 proteins. 2 carbs. That's it. Rotate weekly.\n\n"
        "Variety is for restaurants. Consistency is for results.'\n"
        "^ Same content, completely rewritten. Has a hook, a framework, personality, "
        "and a punchy close. NEVER copy phrases from the example — use it as a "
        "structural guide only."
    )

    search_block = ""
    if search_context:
        terms = "\n".join(f"- {q}" for q in search_context[:30])
        search_block = (
            f"\n\nSEARCH CONTEXT (real queries people type into Google/ChatGPT — "
            "naturally weave 1-3 of these into the caption where relevant, "
            "don't force them):\n" + terms
        )

    title_block = ""
    if video_title:
        title_block = (
            f"\n\nVIDEO TITLE (use this as the post title, or a light adaptation "
            f"of it for this platform): {video_title}"
        )

    human_prompt = (
        f"Generate {count} title+caption option(s) from the following content.\n\n"
        f"TRANSCRIPT:\n{transcript[:6000]}"
        f"{search_block}{title_block}\n\n"
        f"Remember: {count} distinct options, each taking a DIFFERENT angle.\n"
        "DO NOT copy full sentences from the transcript. Extract the ideas and "
        "rewrite them as compelling copy that sounds like this person actually "
        "sat down and wrote a post."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]

    for attempt in range(1, max_retries + 2):
        response = llm.invoke(messages)
        options = response.options[:count]

        # Check all options for banned phrases
        all_text = " ".join(f"{o.title} {o.caption}" for o in options)
        found = _find_banned(all_text)

        if not found:
            return options

        if attempt <= max_retries:
            print(
                f"  [retry {attempt}/{max_retries}] Banned phrases detected: "
                f"{found} — regenerating...",
                file=sys.stderr,
            )
            # Add feedback as a follow-up message for the retry
            messages.append(
                HumanMessage(
                    content=(
                        f"Your output contained banned corporate phrases: {found}. "
                        "Rewrite ALL options. Replace each flagged phrase with a specific "
                        "detail, a concrete number, or just delete it. Do NOT use any "
                        "variation of these phrases."
                    )
                )
            )
        else:
            print(
                f"  [warning] Banned phrases still present after {max_retries} "
                f"retries: {found}. Returning best effort.",
                file=sys.stderr,
            )

    return options


def _parse_platform_voice(spec: str) -> tuple[str, Path]:
    """Parse a 'platform:voice_file' spec. Returns (platform, voice_path)."""
    if ":" not in spec:
        print(
            f"Error: --platforms entries must be 'platform:voice_file', got '{spec}'",
            file=sys.stderr,
        )
        sys.exit(1)
    platform, voice_path_str = spec.split(":", 1)
    if platform not in PLATFORM_GUIDANCE:
        print(
            f"Error: Unknown platform '{platform}'. "
            f"Valid: {', '.join(PLATFORM_GUIDANCE.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)
    voice_path = Path(voice_path_str)
    if not voice_path.exists():
        print(f"Error: Voice file not found: {voice_path}", file=sys.stderr)
        sys.exit(1)
    return platform, voice_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate title + caption copy from a transcript and voice guidelines",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        required=True,
        help="Path to transcript file (words JSON, utterances JSON, or plain text)",
    )
    parser.add_argument(
        "--voice",
        type=Path,
        default=Path(__file__).parent / "voice.md",
        help="Path to voice guidelines (default: voice.md in same directory)",
    )
    parser.add_argument(
        "--platform",
        choices=list(PLATFORM_GUIDANCE.keys()),
        default="short",
        help="Target platform (default: short). Ignored when --platforms is used.",
    )
    parser.add_argument(
        "--platforms",
        nargs="+",
        metavar="PLATFORM:VOICE_FILE",
        help=(
            "Batch mode: generate copy for multiple platforms with per-platform "
            "voice files. Each entry is 'platform:voice_file.md'. "
            "Outputs a keyed JSON object. Overrides --platform and --voice."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of options to generate per platform (default: 1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-opus-4.6",
        help="LLM model via OpenRouter (default: anthropic/claude-opus-4.6)",
    )
    parser.add_argument(
        "--search-context",
        type=Path,
        default=None,
        help="JSON file: list of search phrases, or output of suggest_video_title.py --save (ATP-backed)",
    )
    parser.add_argument(
        "--video-title",
        type=str,
        default=None,
        help="Pre-determined video title to use as the post title",
    )
    args = parser.parse_args()

    # Validate transcript
    if not args.transcript.exists():
        print(f"Error: Transcript not found: {args.transcript}", file=sys.stderr)
        sys.exit(1)

    transcript = load_transcript(args.transcript)
    if not transcript.strip():
        print("Error: Transcript is empty", file=sys.stderr)
        sys.exit(1)

    # Load optional search context (from ATP script or manual list)
    search_ctx: list[str] | None = None
    if args.search_context and args.search_context.exists():
        raw = json.loads(args.search_context.read_text())
        # suggest_video_title.py --save outputs { "titles": ..., "search_context": [...] }
        if isinstance(raw, dict) and "search_context" in raw:
            raw = raw["search_context"]
        if isinstance(raw, list):
            search_ctx = []
            for item in raw:
                if isinstance(item, str):
                    search_ctx.append(item)
                elif isinstance(item, dict) and "keyword" in item:
                    search_ctx.append(item["keyword"])
                elif isinstance(item, dict) and "inspired_by" in item:
                    search_ctx.extend(item["inspired_by"])

    video_title = args.video_title

    # ── Batch mode (--platforms) ─────────────────────────────────────────
    if args.platforms:
        platform_voice_pairs = [_parse_platform_voice(spec) for spec in args.platforms]

        batch_output: dict[str, list[dict] | dict] = {}
        for platform, voice_path in platform_voice_pairs:
            voice_guidelines = voice_path.read_text()
            print(
                f"Generating {platform} copy (voice: {voice_path.name})...",
                file=sys.stderr,
            )
            options = generate_copy(
                transcript=transcript,
                voice_guidelines=voice_guidelines,
                platform=platform,
                count=args.count,
                model=args.model,
                search_context=search_ctx,
                video_title=video_title,
            )
            items = [{"title": o.title, "caption": o.caption} for o in options]
            batch_output[platform] = items[0] if len(items) == 1 else items

        json.dump(batch_output, sys.stdout, indent=2)
        print()
        return

    # ── Single-platform mode (--platform + --voice) ──────────────────────
    if not args.voice.exists():
        print(f"Error: Voice file not found: {args.voice}", file=sys.stderr)
        sys.exit(1)

    voice_guidelines = args.voice.read_text()

    options = generate_copy(
        transcript=transcript,
        voice_guidelines=voice_guidelines,
        platform=args.platform,
        count=args.count,
        model=args.model,
        search_context=search_ctx,
        video_title=video_title,
    )

    output = [{"title": o.title, "caption": o.caption} for o in options]
    if len(output) == 1:
        json.dump(output[0], sys.stdout, indent=2)
    else:
        json.dump(output, sys.stdout, indent=2)
    print()  # trailing newline


if __name__ == "__main__":
    main()
