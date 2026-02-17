"""Ideate stage: Generate high-signal topic ideas from demand data + brand profile.

Flow:
1. Load the brand profile (output of Define stage)
2. LLM proposes ~10 search terms based on the profile
3. Query AnswerThePublic API for each term
4. LLM filters & ranks results against what this specific customer would want to talk about

Usage:
    dotenvx run -f .env -- uv run python -m src.ideate.idea_generator \
        --profile projects/my-client/01_definition.md \
        --output projects/my-client/02_ideas.json
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- AnswerThePublic API ---

ATP_BASE_URL = "https://answer-the-public1.p.rapidapi.com/answer-the-public"
ATP_HEADERS = {
    "x-rapidapi-host": "answer-the-public1.p.rapidapi.com",
    "x-rapidapi-key": "",  # Set from env
}


async def query_answer_the_public(
    keyword: str,
    country: str = "us",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Query AnswerThePublic for a keyword. Returns raw API response."""
    headers = {
        **ATP_HEADERS,
        "x-rapidapi-key": api_key or os.environ.get("RAPID_API_KEY", ""),
    }

    if not headers["x-rapidapi-key"]:
        raise ValueError("RAPID_API_KEY not set. Add it to your .env.")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            ATP_BASE_URL,
            params={"keyword": keyword, "country": country},
            headers=headers,
        )
        response.raise_for_status()
        return response.json()


async def batch_query_atp(
    keywords: list[str],
    country: str = "us",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Query ATP for multiple keywords. Returns {keyword: response}."""
    results = {}
    # Run sequentially to avoid rate limits
    for kw in keywords:
        try:
            logger.info("Querying ATP for: %s", kw)
            data = await query_answer_the_public(kw, country=country, api_key=api_key)
            results[kw] = data
            logger.info("Got results for: %s", kw)
        except Exception as e:
            logger.warning("Failed to query '%s': %s", kw, e)
            results[kw] = {"error": str(e)}
    return results


# --- LLM Structured Outputs ---


class SearchTerms(BaseModel):
    """Search terms proposed by the LLM based on the brand profile."""

    terms: list[str] = Field(
        description="10 high-signal search terms to query AnswerThePublic with. "
        "Each MUST be 1-2 words only (ATP returns poor data for 3+ word queries). "
        "Represent topics the customer's ICP is searching for.",
    )
    reasoning: str = Field(
        description="Brief explanation of why these terms were chosen.",
    )


class TopicCandidate(BaseModel):
    """A single topic candidate for content creation."""

    topic: str = Field(description="The topic title — short, punchy, quotable")
    pillar: str = Field(
        description="Which messaging pillar from the brand profile this topic belongs to. "
        "Use the exact pillar name from the profile."
    )
    icp: str = Field(
        description="Which ICP segment(s) this topic targets. "
        "Be specific (e.g., 'aspiring GMs' not just 'managers')."
    )
    hook: str = Field(
        description="The opening line for the video — a ~3 second sound bite that stops the scroll. "
        "ONE simple sentence or fragment only (roughly 8–12 words). No compound sentences: "
        "avoid 'When X, Y' or 'If X, Y' — use a single punchy idea so the viewer doesn't tune out. "
        "Written as the founder would say it."
    )
    prompt: str = Field(
        description="A question the interviewer / producer would ask the founder "
        "to naturally draw out the angle. Written as a direct question to the founder. "
        "e.g., 'What's the one thing you wish someone had told you before you became a GM?'"
    )
    angle: str = Field(
        description="The specific take, story, or argument the founder should make. "
        "2-4 sentences expanding on the hook with concrete details."
    )
    source_keyword: str = Field(
        description="Which of the original seed keywords (from the search terms list) "
        "produced this topic. Must be one of the exact seed terms provided."
    )
    user_search_term: str = Field(
        description="The specific ATP suggestion / long-tail query this topic was derived from. "
        "This is the actual phrase people are searching for."
    )
    relevance_score: int = Field(
        description="1-10 score: how well this aligns with the brand profile",
        ge=1,
        le=10,
    )
    excitement_score: int = Field(
        description="1-10 score: how likely the founder would enjoy talking about this",
        ge=1,
        le=10,
    )
    buyer_intent_score: int = Field(
        description="1-10 score: how close this is to buying intent for their ICP",
        ge=1,
        le=10,
    )
    rationale: str = Field(
        description="Why this topic is a strong fit for this specific founder — "
        "reference their brand profile, authority, or audience."
    )


class FilteredTopics(BaseModel):
    """LLM-filtered and ranked topics."""

    topics: list[TopicCandidate] = Field(
        description="Filtered and ranked topic candidates, best first.",
    )
    summary: str = Field(
        description="Brief summary of the overall topic strategy.",
    )


def _get_llm() -> ChatOpenAI:
    """Get the LLM instance using OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")

    return ChatOpenAI(
        model="google/gemini-3-flash-preview",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


# --- Step 1: Propose Search Terms ---


async def propose_search_terms(brand_profile: dict[str, Any] | str) -> SearchTerms:
    """Use LLM to propose ~10 search terms based on the brand profile."""
    llm = _get_llm()
    structured_llm = llm.with_structured_output(SearchTerms)

    profile_text = (
        brand_profile
        if isinstance(brand_profile, str)
        else json.dumps(brand_profile, indent=2)
    )

    response = await structured_llm.ainvoke(
        [
            SystemMessage(
                content=(
                    "You are a content strategist specializing in demand-driven topic research. "
                    "Your job is to propose search terms that will surface what a founder's "
                    "ideal customers are actually searching for online.\n\n"
                    "Given a brand profile, propose exactly 10 search terms to query "
                    "AnswerThePublic with.\n\n"
                    "## Hard constraints\n"
                    "- Each term MUST be 1-2 words. Never 3+. ATP returns poor data for longer queries.\n\n"
                    "## What makes a GOOD seed term\n"
                    "- Targets **processes, tactics, pain points, and decisions** the ICP faces\n"
                    "  e.g. 'dealer fees', 'car financing', 'trade in', 'car broker'\n"
                    "- Maps to a specific messaging pillar from the brand profile\n"
                    "- Covers different stages of the customer journey (awareness → consideration → decision)\n"
                    "- At least 2 terms the ICP would search BEFORE knowing the founder's solution exists\n\n"
                    "## What makes a BAD seed term\n"
                    "- Specific vehicle makes/models ('ford bronco', 'tesla model')\n"
                    "- Deal-hunting terms ('lease deals', 'car deals') — these return transactional queries "
                    "that don't lead to educational content\n"
                    "- The founder's internal jargon that buyers don't search for\n"
                    "- Overly generic terms ('cars', 'buying')\n"
                )
            ),
            HumanMessage(
                content=f"Here is the brand profile:\n\n{profile_text}\n\n"
                "Propose 10 search terms for AnswerThePublic. "
                "Each term should map to one of the profile's messaging pillars."
            ),
        ]
    )
    return response  # type: ignore[return-value]


# --- Step 2: Extract Topics from ATP Results ---


def extract_topics_from_atp(atp_results: dict[str, Any]) -> list[str]:
    """Extract all unique topic strings from ATP API responses.

    ATP response structure varies but typically includes categories like
    questions, prepositions, comparisons, alphabeticals, and related terms.
    Each category contains items with a 'keyword' or 'query' field.
    """
    topics: set[str] = set()

    for keyword, data in atp_results.items():
        if isinstance(data, dict) and "error" not in data:
            _extract_recursive(data, topics)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    topics.add(item)
                elif isinstance(item, dict):
                    _extract_recursive(item, topics)

    return sorted(topics)


def _extract_recursive(obj: Any, topics: set[str]) -> None:
    """Recursively extract topic strings from nested ATP response data."""
    if isinstance(obj, dict):
        # Common ATP field names for the actual query text
        for key in ("keyword", "query", "text", "suggestion", "term", "phrase"):
            if key in obj and isinstance(obj[key], str) and len(obj[key].strip()) > 3:
                topics.add(obj[key].strip())
        # Recurse into nested structures
        for value in obj.values():
            _extract_recursive(value, topics)
    elif isinstance(obj, list):
        for item in obj:
            _extract_recursive(item, topics)


# --- Step 3: Filter & Rank Topics ---


async def filter_and_rank_topics(
    brand_profile: dict[str, Any] | str,
    raw_topics: list[str],
    search_terms: list[str] | None = None,
    max_results: int = 20,
) -> FilteredTopics:
    """Use LLM to filter ATP results based on brand profile fit."""
    llm = _get_llm()
    structured_llm = llm.with_structured_output(FilteredTopics)

    profile_text = (
        brand_profile
        if isinstance(brand_profile, str)
        else json.dumps(brand_profile, indent=2)
    )
    # Sample representatively: take from each keyword's results via stratified random sample
    # (raw_topics is alphabetically sorted — taking [:500] biases toward early letters)
    max_sample = 500
    if len(raw_topics) <= max_sample:
        sampled_topics = raw_topics
    else:
        random.seed(42)  # Reproducible
        sampled_topics = random.sample(raw_topics, max_sample)
        sampled_topics.sort()

    topics_text = "\n".join(f"- {t}" for t in sampled_topics)

    response = await structured_llm.ainvoke(
        [
            SystemMessage(
                content=(
                    "You are a content strategist creating video topic ideas for a founder.\n\n"
                    "You will receive:\n"
                    "1. A brand profile (who the founder is, what they sell, their messaging pillars)\n"
                    "2. Raw search queries from AnswerThePublic showing what real people are Googling\n\n"
                    "## Your job\n"
                    "Use the search demand data as EVIDENCE of audience pain points, then craft "
                    f"{max_results} original, compelling content ideas that this founder should record.\n\n"
                    "## CRITICAL: Transform, don't restate\n"
                    "The raw topics are search queries ('$99 lease deals near me'). "
                    "DO NOT use these as topic titles. Instead, extract the underlying pain point "
                    "and create an educational, principle-based topic:\n"
                    "- BAD topic: '$99 Lease Deals Near Me' (just a search query)\n"
                    "- BAD topic: 'Tesla Model Y: Lease vs Buy' (too vehicle-specific)\n"
                    "- GOOD topic: 'The $99 Lease Tease' (exposes a tactic)\n"
                    "- GOOD topic: 'The Trade-In Shell Game' (teaches a principle)\n"
                    "- GOOD topic: 'Stop Negotiating Monthly Payments' (universal advice)\n\n"
                    "## Topic quality rules\n"
                    "- Every topic must teach a **transferable principle, expose a tactic, or give a defensive move**\n"
                    "- Topics must be **evergreen** — still relevant in 2 years, not tied to a specific car model or year\n"
                    "- Topics must map to a **messaging pillar** from the brand profile\n"
                    "- Topics should sound like something the founder would passionately rant about on TikTok\n"
                    "- Spread topics across ALL pillars — don't cluster on just one\n\n"
                    "## Filter OUT\n"
                    "- Vehicle-specific topics (specific makes/models/years as the primary subject)\n"
                    "- Pure deal-hunting topics ('best lease deals 2024')\n"
                    "- Topics outside the founder's expertise or brand boundaries\n"
                    "- Topics that are too generic to be a compelling video\n\n"
                    "## Scoring\n"
                    "- **Relevance (1-10)**: Alignment with messaging pillars and brand positioning\n"
                    "- **Excitement (1-10)**: How likely the founder is to enjoy talking about this\n"
                    "- **Buyer Intent (1-10)**: How close the searcher is to needing the founder's services\n\n"
                    "## Output fields\n"
                    "- `topic`: Short, punchy, quotable title (NOT a search query)\n"
                    "- `hook`: ONE simple sentence/fragment, ~8-12 words, ~3 seconds when spoken. "
                    "No compound sentences ('When X, Y' or 'If X, Y'). One punchy idea.\n"
                    "- `prompt`: Question a producer would ask the founder to draw out the angle naturally\n"
                    "- `angle`: The specific take, story, or argument (2-4 sentences with concrete details)\n"
                    "- `pillar`: Exact pillar name from the brand profile\n"
                    "- `icp`: Specific ICP segment this targets\n"
                    "- `rationale`: Why this fits THIS founder — reference their brand, authority, or audience\n"
                    "- `source_keyword`: Which seed keyword produced the demand signal\n"
                    "- `user_search_term`: The actual ATP query that inspired this (the real thing people Google)"
                )
            ),
            HumanMessage(
                content=(
                    f"## Brand Profile\n\n{profile_text}\n\n"
                    + (
                        f"## Seed Keywords Used\n\n"
                        + "\n".join(f"- {t}" for t in search_terms)
                        + "\n\n"
                        if search_terms
                        else ""
                    )
                    + f"## Search Demand Data ({len(raw_topics)} unique queries from AnswerThePublic)\n\n"
                    f"{topics_text}\n\n"
                    f"Create the top {max_results} content ideas for this founder. "
                    f"Spread them across the brand's messaging pillars."
                )
            ),
        ]
    )
    return response  # type: ignore[return-value]


# --- ideas.json → ideas.md ---

BAR_CHAR = "▮"


def _virality_bars(topic: TopicCandidate) -> str:
    """(relevance + excitement + buyer_intent) / 3, rounded to int 1-10, as ▮ bars."""
    raw = (
        topic.relevance_score + topic.excitement_score + topic.buyer_intent_score
    ) / 3
    n = max(1, min(10, round(raw)))
    return BAR_CHAR * n


def _virality_score(topic: TopicCandidate) -> float:
    """Numeric virality for display in idea blocks."""
    return (
        topic.relevance_score + topic.excitement_score + topic.buyer_intent_score
    ) / 3


def write_ideas_markdown(
    result: FilteredTopics,
    output_path: Path,
    atp_raw_topic_count: int,
    search_terms: SearchTerms,
    used_strategy: bool = False,
) -> Path:
    """Write ideas.md from FilteredTopics. Returns path to written file."""
    out_dir = output_path.parent
    client_name = out_dir.name.replace("_", " ")
    ideas_md = out_dir / "ideas.md"

    def _cell(s: str) -> str:
        """Escape pipe so table cells don't break."""
        return s.replace("|", "\\|")

    # Summary checklist and table — sorted by virality (highest first)
    topics_sorted = sorted(
        result.topics,
        key=lambda t: (t.relevance_score + t.excitement_score + t.buyer_intent_score)
        / 3,
        reverse=True,
    )
    checklist_lines = [
        f"- [ ] {i}. {t.user_search_term}" for i, t in enumerate(topics_sorted, 1)
    ]
    # Table: # | Prompt | Virality Score | Hook (3s open)
    table_rows = [
        f"| {i} | {_cell(t.prompt)} | {_virality_score(t):.1f} | {_cell(t.hook)} |"
        for i, t in enumerate(topics_sorted, 1)
    ]

    # Group topics by pillar (preserve order of first occurrence)
    pillars_order: list[str] = []
    by_pillar: dict[str, list[tuple[int, TopicCandidate]]] = {}
    for i, topic in enumerate(result.topics, 1):
        if topic.pillar not in by_pillar:
            pillars_order.append(topic.pillar)
            by_pillar[topic.pillar] = []
        by_pillar[topic.pillar].append((i, topic))

    # Build "Top N Ideas" sections
    idea_sections: list[str] = []
    for pillar in pillars_order:
        idea_sections.append(f'\n### Pillar: "{pillar}"\n')
        for num, topic in by_pillar[pillar]:
            vs = _virality_score(topic)
            idea_sections.append(
                f'**{num}. "{topic.topic}"** — [Virality Score: {vs:.1f}]\n'
                f"- ICP: {topic.icp}\n"
                f"- Pillar: {topic.pillar}\n"
                f'- Hook: "{topic.hook}"\n'
                f'- Prompt: "{topic.prompt}"\n'
                f"- Angle: {topic.angle}\n"
                f"- Rationale: {topic.rationale}\n"
                f'- Source search keyword: "{topic.source_keyword}"\n'
                f'- User search term: "{topic.user_search_term}"\n\n'
            )

    based_on = (
        f"Brand profile"
        + (" + content strategy" if used_strategy else "")
        + f" + AnswerThePublic demand data ({len(search_terms.terms)} seed keywords, {atp_raw_topic_count:,} topics filtered)"
    )
    checklist = "\n".join(checklist_lines)
    summary_table = "\n".join(table_rows)

    md = f"""# Recording Session Ideas — {client_name}

**Based on:** {based_on}
**Format:** Sub-35s direct-to-camera (confirm with client)
**Next step:** Pick 5–7 for a first recording session

---

## Summary — Pick your batch

Both the checklist and table below are sorted by virality (highest first).

{checklist}

| # | Prompt | Virality Score | Hook (3s open) |
|---|--------|----------------|----------------|
{summary_table}

---

## Top {len(result.topics)} Ideas
{"".join(idea_sections)}
---

## Session Planning

**Recommended first session (pick 5–7):**

| # | Topic | Why |
|---|-------|-----|
{"".join(f"| {i} | {_cell(t.topic)} | Strong fit |\n" for i, t in enumerate(topics_sorted[:7], 1))}

---

## Production Reminders

*(From profile: confirm format with client. Add any content rules from the definition file.)*
"""

    ideas_md.write_text(md, encoding="utf-8")
    logger.info("Saved ideas.md to %s", ideas_md)
    return ideas_md


# --- Main Pipeline ---


def _load_profile_and_strategy(
    brand_profile_path: Path,
    strategy_path: Path | None,
) -> tuple[str | dict[str, Any], bool]:
    """Load profile and optional strategy. Returns (context_for_llm, used_strategy)."""
    if not brand_profile_path.exists():
        raise FileNotFoundError(f"Brand profile not found: {brand_profile_path}")

    with open(brand_profile_path) as f:
        if brand_profile_path.suffix == ".json":
            brand_profile: dict[str, Any] | str = json.load(f)
        else:
            brand_profile = f.read()

    used_strategy = False
    if strategy_path and strategy_path.exists():
        with open(strategy_path) as f:
            strategy_content = f.read()
        profile_text = (
            brand_profile
            if isinstance(brand_profile, str)
            else json.dumps(brand_profile, indent=2)
        )
        context = (
            f"## Brand Profile\n\n{profile_text}\n\n"
            f"## Content Strategy (format, platforms, UVP, content plan)\n\n{strategy_content}"
        )
        used_strategy = True
        return context, used_strategy

    return brand_profile, used_strategy


async def generate_ideas(
    brand_profile_path: str | Path,
    output_path: str | Path | None = None,
    strategy_path: str | Path | None = None,
    country: str = "us",
    max_topics: int = 20,
) -> FilteredTopics:
    """Full pipeline: profile (+ optional strategy) → search terms → ATP → filtered topics.

    Args:
        brand_profile_path: Path to the brand profile (.md or .json).
        output_path: Optional path to save results JSON.
        strategy_path: Optional path to content strategy (.md). If present, merged with profile for LLM context.
        country: Country code for ATP queries.
        max_topics: Maximum number of topics to return.

    Returns:
        FilteredTopics with ranked topic candidates.
    """
    profile_path = Path(brand_profile_path)
    strategy_path_resolved = Path(strategy_path) if strategy_path else None
    if not strategy_path_resolved and profile_path.suffix == ".md":
        # Default: strategy.md next to profile.md
        sibling = profile_path.parent / "strategy.md"
        if sibling.exists():
            strategy_path_resolved = sibling

    context_for_llm, used_strategy = _load_profile_and_strategy(
        profile_path, strategy_path_resolved
    )
    logger.info(
        "Loaded brand profile from %s (%s)",
        profile_path,
        "profile + strategy" if used_strategy else profile_path.suffix,
    )

    # Step 1: Propose search terms
    logger.info("Step 1: Proposing search terms...")
    search_terms = await propose_search_terms(context_for_llm)
    logger.info("Proposed %d terms: %s", len(search_terms.terms), search_terms.terms)
    logger.info("Reasoning: %s", search_terms.reasoning)

    # Step 2: Query ATP
    logger.info(
        "Step 2: Querying AnswerThePublic for %d terms...", len(search_terms.terms)
    )
    atp_results = await batch_query_atp(search_terms.terms, country=country)

    # Step 3: Extract raw topics
    raw_topics = extract_topics_from_atp(atp_results)
    logger.info("Extracted %d unique topics from ATP results", len(raw_topics))

    if not raw_topics:
        logger.warning("No topics extracted from ATP. Using search terms as fallback.")
        raw_topics = search_terms.terms

    # Step 4: Filter & rank
    logger.info("Step 3: Filtering and ranking topics...")
    filtered = await filter_and_rank_topics(
        context_for_llm,
        raw_topics,
        search_terms=search_terms.terms,
        max_results=max_topics,
    )
    logger.info(
        "Selected %d topics. Strategy: %s",
        len(filtered.topics),
        filtered.summary,
    )

    # Save output
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(
                {
                    "search_terms": search_terms.model_dump(),
                    "atp_raw_topic_count": len(raw_topics),
                    "filtered_topics": filtered.model_dump(),
                    "used_strategy": used_strategy,
                },
                f,
                indent=2,
            )
        logger.info("Saved results to %s", out)
        write_ideas_markdown(
            result=filtered,
            output_path=out,
            atp_raw_topic_count=len(raw_topics),
            search_terms=search_terms,
            used_strategy=used_strategy,
        )

    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Generate content topic ideas from demand data + brand profile",
    )
    parser.add_argument(
        "--profile",
        "-p",
        help="Path to the brand profile (.md or .json). Required unless --markdown-only.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output path for results JSON (and ideas.md). For --markdown-only, path to existing ideas.json.",
    )
    parser.add_argument(
        "--strategy",
        "-s",
        default=None,
        help="Path to content strategy (.md). Optional; defaults to strategy.md next to profile if present.",
    )
    parser.add_argument(
        "--markdown-only",
        "-m",
        action="store_true",
        help="Regenerate ideas.md from an existing ideas.json (no API calls). Use -o path/to/ideas.json.",
    )
    parser.add_argument(
        "--country",
        "-c",
        default="us",
        help="Country code for AnswerThePublic (default: us)",
    )
    parser.add_argument(
        "--max-topics",
        "-n",
        type=int,
        default=20,
        help="Max number of topics to return (default: 20)",
    )

    args = parser.parse_args()

    if args.markdown_only:
        if not args.output:
            parser.error("--markdown-only requires --output path/to/ideas.json")
        json_path = Path(args.output)
        if not json_path.exists():
            parser.error(f"File not found: {json_path}")
        with open(json_path) as f:
            data = json.load(f)
        search_terms = SearchTerms(**data["search_terms"])
        atp_count = data["atp_raw_topic_count"]
        filtered = FilteredTopics(**data["filtered_topics"])
        used_strategy = data.get("used_strategy", False)
        write_ideas_markdown(
            result=filtered,
            output_path=json_path,
            atp_raw_topic_count=atp_count,
            search_terms=search_terms,
            used_strategy=used_strategy,
        )
        print(f"Regenerated ideas.md next to {json_path}")
        return

    if not args.profile:
        parser.error("--profile is required (unless using --markdown-only)")

    result = asyncio.run(
        generate_ideas(
            brand_profile_path=args.profile,
            output_path=args.output,
            strategy_path=args.strategy,
            country=args.country,
            max_topics=args.max_topics,
        )
    )

    # Print results to stdout
    print("\n" + "=" * 60)
    print(f"TOPIC IDEAS ({len(result.topics)} selected)")
    print("=" * 60)
    print(f"\nStrategy: {result.summary}\n")

    for i, topic in enumerate(result.topics, 1):
        virality = (
            topic.relevance_score + topic.excitement_score + topic.buyer_intent_score
        ) / 3
        print(f"{i:2d}. [Virality Score: {virality:.1f}] {topic.topic}")
        print(f"    Pillar: {topic.pillar}")
        print(f"    ICP: {topic.icp}")
        print(f"    Hook: {topic.hook}")
        print(f"    Prompt: {topic.prompt}")
        print(f"    Angle: {topic.angle}")
        print(f"    Rationale: {topic.rationale}")
        print(
            f"    Scores: relevance={topic.relevance_score} "
            f"excitement={topic.excitement_score} "
            f"intent={topic.buyer_intent_score}"
        )
        print(f"    Search keyword: {topic.source_keyword}")
        print(f"    User search term: {topic.user_search_term}")
        print()


if __name__ == "__main__":
    main()
