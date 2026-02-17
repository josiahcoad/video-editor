#!/usr/bin/env python3
"""
Suggest a search-optimised video title using AnswerThePublic demand data.

Flow:
  1. Read transcript (words.json, utterances.json, or plain text)
  2. LLM extracts 2-3 short seed keywords (1-2 words) from the transcript
  3. Query AnswerThePublic for each keyword → real search queries with volume
  4. LLM selects the best-matching query and rewrites it as a punchy video title

The goal: every title is something (or close to something) a real person
would type into a search bar or AI chatbot — so our video answers a specific
question people are actually asking.

Usage:
  suggest_video_title.py --transcript path/to/words.json
  suggest_video_title.py --transcript path/to/utterances.json --count 3
  suggest_video_title.py --video path/to/video.mp4          # transcribes first

Requires: OPENROUTER_API_KEY, RAPID_API_KEY
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.ideate.ideate import query_answer_the_public


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------


class SeedKeywords(BaseModel):
    """2-3 short keywords extracted from the video transcript."""

    keywords: list[str] = Field(
        description=(
            "2-3 seed keywords (1-2 words each) that capture the core topics "
            "of the video. These will be used to query AnswerThePublic, so "
            "they should be broad enough to return results but specific enough "
            "to be relevant. Think: what would someone Google before finding this video?"
        ),
    )


class TitleSuggestion(BaseModel):
    """A single title suggestion grounded in real search demand."""

    title: str = Field(
        description=(
            "The video title, written as a natural-language question or statement "
            "that a real person would type into ChatGPT or Google. "
            "5-10 words. Conversational and specific — NOT keyword-stuffed. "
            "Good: 'how to build trust with niche clients'. "
            "Bad: 'Build Trust: Understanding the Business Trust Gap'."
        ),
    )
    inspired_by: list[str] = Field(
        description=(
            "1-3 ATP search queries that informed this title (exact phrases from the list). "
            "The title is a synthesis — it doesn't need to match any of these exactly."
        ),
    )
    rationale: str = Field(
        description=(
            "Why this title works: (1) what demand signal from ATP backs it, "
            "(2) how the video specifically answers this question. 1-2 sentences."
        ),
    )


class TitleSuggestions(BaseModel):
    """Ranked title options for a video."""

    titles: list[TitleSuggestion] = Field(
        description="Title suggestions ranked best-first.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_volume(vol_str: str) -> int:
    """Parse ATP volume strings like '170', '1.6k', '2.3k' into ints."""
    vol_str = vol_str.strip().lower().replace(",", "")
    if vol_str.endswith("k"):
        return int(float(vol_str[:-1]) * 1000)
    if vol_str.endswith("m"):
        return int(float(vol_str[:-1]) * 1_000_000)
    try:
        return int(vol_str)
    except ValueError:
        return 0


def _get_llm(model: str = "google/gemini-2.5-flash") -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")
    return ChatOpenAI(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def load_transcript(path: Path) -> str:
    """Load transcript from words.json, utterances.json, .srt, or plain text."""
    text = path.read_text()
    if path.suffix == ".json":
        data = json.loads(text)
        if isinstance(data, list) and data:
            if "text" in data[0]:
                # utterances format
                return " ".join(item["text"] for item in data)
            elif "word" in data[0]:
                # words format
                return " ".join(item["word"] for item in data)
        return text
    return text


async def transcribe_video(video_path: Path) -> str:
    """Quick transcription using Deepgram (via our existing get_transcript)."""
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.edit.get_transcript",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Transcription failed: {result.stderr}")

    # get_transcript writes *-words.json next to the input
    words_path = video_path.with_name(video_path.stem + "-words.json")
    if words_path.exists():
        return load_transcript(words_path)
    # fallback: stdout
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Step 1: Extract seed keywords from transcript
# ---------------------------------------------------------------------------


async def extract_seed_keywords(transcript: str) -> list[str]:
    """Use LLM to extract 2-3 short seed keywords from the transcript."""
    llm = _get_llm()
    structured = llm.with_structured_output(SeedKeywords)

    response = await structured.ainvoke(
        [
            SystemMessage(
                content=(
                    "You are a keyword researcher. Given a video transcript, extract "
                    "2-3 seed keywords (1-2 words each) that capture the main topics.\n\n"
                    "Rules:\n"
                    "- Each keyword MUST be 1-2 words (AnswerThePublic needs short seeds)\n"
                    "- Focus on what the viewer would search for, not jargon from the video\n"
                    "- Think: what broad topic would someone Google *before* finding this video?\n"
                    "- Avoid brand names or overly specific terms\n"
                    "- Include at least one keyword about the PROBLEM the video solves\n"
                    "- Include at least one keyword about the DOMAIN/INDUSTRY"
                )
            ),
            HumanMessage(content=f"Video transcript:\n\n{transcript[:3000]}"),
        ]
    )
    return response.keywords  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Step 2: Query ATP and collect search queries
# ---------------------------------------------------------------------------


async def get_atp_queries(keywords: list[str]) -> list[dict[str, Any]]:
    """Query ATP for each keyword, return flat list of {keyword, volume} dicts."""
    all_queries: list[dict[str, Any]] = []

    for kw in keywords:
        try:
            print(f"  Querying ATP for: {kw!r}", file=sys.stderr)
            result = await query_answer_the_public(kw)
            ak = result.get("all_keywords", {})
            raw_list = ak.get("all_keywords", [])
            for item in raw_list:
                vol = _parse_volume(item.get("searche volume", "0"))
                if vol > 0:
                    all_queries.append(
                        {
                            "keyword": item["keyword"],
                            "volume": vol,
                            "seed": kw,
                        }
                    )
        except Exception as e:
            print(f"  ATP query failed for {kw!r}: {e}", file=sys.stderr)

    # Deduplicate by keyword, keep highest volume
    seen: dict[str, dict[str, Any]] = {}
    for q in all_queries:
        k = q["keyword"].lower().strip()
        if k not in seen or q["volume"] > seen[k]["volume"]:
            seen[k] = q

    # Sort by volume descending, cap at 80 to stay within token limits
    return sorted(seen.values(), key=lambda x: x["volume"], reverse=True)[:80]


# ---------------------------------------------------------------------------
# Step 3: LLM picks the best title from ATP queries
# ---------------------------------------------------------------------------


async def pick_titles(
    transcript: str,
    atp_queries: list[dict[str, Any]],
    count: int = 3,
) -> TitleSuggestions:
    """LLM picks the best ATP queries and turns them into video titles."""
    llm = _get_llm()
    structured = llm.with_structured_output(TitleSuggestions)

    queries_text = "\n".join(f"- [{q['volume']}] {q['keyword']}" for q in atp_queries)

    response = await structured.ainvoke(
        [
            SystemMessage(
                content=(
                    "You write video titles that match real search demand.\n\n"
                    "You will receive:\n"
                    "1. A video transcript (what the video actually covers)\n"
                    "2. A list of real search queries people type into Google/ChatGPT, "
                    "with monthly search volume\n\n"
                    "Your job: write titles that sit at the INTERSECTION of what the video "
                    "covers and what people are already searching for.\n\n"
                    "HOW TO WRITE A GREAT TITLE:\n"
                    "- The ATP queries are DEMAND SIGNALS, not templates. Use them to "
                    "understand what people care about, then SYNTHESIZE a title that "
                    "captures the video's specific angle within that demand.\n"
                    "- Write like a real human asking a question to ChatGPT or typing "
                    "into Google. Natural language, conversational, specific.\n"
                    "- The title should feel like something you'd text a friend: "
                    "'how to build trust with niche clients before the sales call' — "
                    "NOT 'Build Trust: Understanding the Business Trust Gap'.\n\n"
                    "EXAMPLES of great titles (for reference):\n"
                    "- 'social media roi for service businesses'\n"
                    "- 'how to build trust with niche clients'\n"
                    "- 'tips for being comfortable on camera'\n"
                    "- 'closing high ticket deals without being salesy'\n"
                    "- 'overcoming objections through content'\n\n"
                    "EXAMPLES of BAD titles:\n"
                    "- 'Marketing and Strategy Tips' (too generic, no angle)\n"
                    "- 'Build Trust: Understanding the Business Trust Gap' (keyword-stuffed, robotic)\n"
                    "- 'Learn Strategic Digital Marketing' (sounds like a course ad)\n\n"
                    f"Write {count} title options, ranked best-first.\n\n"
                    "Title rules:\n"
                    "- 5-10 words, lowercase (like a real search query), no colons\n"
                    "- Must be specific to what THIS video covers — not a generic topic title\n"
                    "- Questions or 'how to' formats work great\n"
                    "- Each title should target a different angle or audience"
                )
            ),
            HumanMessage(
                content=(
                    f"## Video Transcript\n\n{transcript[:3000]}\n\n"
                    f"## Real Search Queries (with monthly volume)\n\n{queries_text}\n\n"
                    f"Write {count} titles."
                )
            ),
        ]
    )
    return response  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


class SuggestTitleResult:
    """Bundle of title suggestions + the ATP queries that informed them."""

    def __init__(
        self, suggestions: TitleSuggestions, atp_queries: list[dict[str, Any]]
    ):
        self.suggestions = suggestions
        self.atp_queries = atp_queries

    @property
    def titles(self) -> list[TitleSuggestion]:
        return self.suggestions.titles

    @property
    def best_title(self) -> str:
        return self.suggestions.titles[0].title if self.suggestions.titles else ""

    @property
    def search_terms(self) -> list[str]:
        """Flat list of ATP query strings for use as search_context."""
        return [q["keyword"] for q in self.atp_queries]


async def suggest_title(
    transcript: str | None = None,
    transcript_path: Path | None = None,
    video_path: Path | None = None,
    count: int = 3,
) -> SuggestTitleResult:
    """Full pipeline: transcript → seed keywords → ATP → title suggestions."""

    # Resolve transcript
    if transcript is None:
        if transcript_path:
            transcript = load_transcript(transcript_path)
        elif video_path:
            print("Transcribing video...", file=sys.stderr)
            transcript = await transcribe_video(video_path)
        else:
            raise ValueError("Provide transcript, transcript_path, or video_path")

    if not transcript.strip():
        raise ValueError("Transcript is empty")

    print(f"Transcript: {len(transcript.split())} words", file=sys.stderr)

    # Step 1: Extract seed keywords
    print("Step 1: Extracting seed keywords...", file=sys.stderr)
    keywords = await extract_seed_keywords(transcript)
    print(f"  Seeds: {keywords}", file=sys.stderr)

    # Step 2: Query ATP
    print("Step 2: Querying AnswerThePublic...", file=sys.stderr)
    atp_queries = await get_atp_queries(keywords)
    print(f"  Got {len(atp_queries)} queries with volume", file=sys.stderr)

    if not atp_queries:
        raise RuntimeError(
            "No ATP results with search volume. Try a broader topic. "
            f"Seeds were: {keywords}"
        )

    # Step 3: Pick titles
    print(f"Step 3: Picking top {count} titles...", file=sys.stderr)
    suggestions = await pick_titles(transcript, atp_queries, count=count)
    return SuggestTitleResult(suggestions=suggestions, atp_queries=atp_queries)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Suggest search-optimised video titles using ATP demand data",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--transcript",
        type=Path,
        help="Path to transcript file (words.json, utterances.json, or plain text)",
    )
    group.add_argument(
        "--video",
        type=Path,
        help="Path to video file (will transcribe first)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of title suggestions (default: 3)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted text",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save full output (titles + ATP search context) to this JSON file",
    )
    args = parser.parse_args()

    result = asyncio.run(
        suggest_title(
            transcript_path=args.transcript,
            video_path=args.video,
            count=args.count,
        )
    )

    full_output = {
        "titles": [
            {
                "title": t.title,
                "inspired_by": t.inspired_by,
                "rationale": t.rationale,
            }
            for t in result.titles
        ],
        "search_context": result.search_terms,
    }

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(full_output, indent=2))
        print(f"Saved to {args.save}", file=sys.stderr)

    if args.json:
        json.dump(full_output, sys.stdout, indent=2)
        print()
    else:
        print("\nTitle Suggestions:\n")
        for i, t in enumerate(result.titles, 1):
            print(f"  {i}. {t.title}")
            print(f"     Inspired by: {', '.join(t.inspired_by)}")
            print(f"     {t.rationale}")
            print()


if __name__ == "__main__":
    main()
