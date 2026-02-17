#!/usr/bin/env python3
"""
Eval harness for propose_cuts.

Runs propose_cuts against a dataset of (transcript, ground_truth_cuts) pairs,
scores outputs with a judge LLM, and reports multi-dimensional results.

This prevents the "overfit to one video" trap — every prompt change is measured
against ALL test cases, not just the one you're currently looking at.

Usage:
  dotenvx run -f .env -f .env.dev -- uv run python -m evals.propose_cuts.run_eval
  dotenvx run -f .env -f .env.dev -- uv run python -m evals.propose_cuts.run_eval --model anthropic/claude-4-opus
  dotenvx run -f .env -f .env.dev -- uv run python -m evals.propose_cuts.run_eval --case 260210-textbook-interview
  dotenvx run -f .env -f .env.dev -- uv run python -m evals.propose_cuts.run_eval --runs 3  # variance check
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.edit.get_transcript import read_word_transcript_file
from src.edit.propose_cuts import propose_cuts

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = Path(__file__).parent / "dataset"
RESULTS_DIR = Path(__file__).parent / "results"

DEFAULT_JUDGE_MODEL = "openai/gpt-4o"


# ── Pydantic models for structured judge output ──────────────────────────────


class DimensionScore(BaseModel):
    """Score for a single evaluation dimension."""

    score: int = Field(ge=1, le=10, description="Score from 1-10")
    reasoning: str = Field(description="1-2 sentence explanation for this score")


class JudgeEvaluation(BaseModel):
    """Complete judge evaluation of proposed cuts vs ground truth."""

    cold_viewer_test: DimensionScore = Field(
        description="Would a cold viewer understand each segment from second 1?"
    )
    hook_strength: DimensionScore = Field(
        description="Do the first 3-5 seconds of each segment grab attention?"
    )
    narrative_coherence: DimensionScore = Field(
        description="Do sections within each segment flow logically when assembled?"
    )
    content_selection: DimensionScore = Field(
        description="Did the proposed cuts select the strongest, most engaging content?"
    )
    cut_precision: DimensionScore = Field(
        description=(
            "Do sections start and end at natural sentence/clause boundaries? "
            "Check: (1) first words of each segment form a complete statement, "
            "not a mid-sentence fragment; (2) sections don't end on trailing "
            "conjunctions (and, but, so); (3) no words are clipped mid-phrase. "
            "Score 10 = every cut is on a clean boundary. "
            "Score 1 = multiple sections start/end mid-sentence."
        )
    )
    duration_accuracy: DimensionScore = Field(
        description="Are segments within the target duration range?"
    )
    segment_structure: DimensionScore = Field(
        description=(
            "Is the number/shape of segments appropriate? "
            "Right count, reasonable sections per segment, appropriate cold opens?"
        )
    )
    overall_notes: str = Field(
        description=(
            "Key differences between ground truth and proposed cuts. "
            "What did the proposed cuts do well? What did they miss?"
        )
    )


# ── Utility functions ─────────────────────────────────────────────────────────


def cuts_to_sections(words: list[dict], cuts_str: str) -> list[dict]:
    """Map a cuts string back to the actual words/text for each section.

    Returns a list of dicts: {start, end, text, duration, word_count}
    """
    sections = []
    for part in cuts_str.split(","):
        start_str, end_str = part.split(":")
        start, end = float(start_str), float(end_str)
        section_words = [
            w["word"]
            for w in words
            if w["start"] >= start - 0.05 and w["end"] <= end + 0.05
        ]
        sections.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "text": " ".join(section_words),
                "duration": round(end - start, 2),
                "word_count": len(section_words),
            }
        )
    return sections


def format_segments_for_judge(
    segments: list[dict], words: list[dict], label: str
) -> str:
    """Render a segment list into human-readable text the judge can evaluate."""
    lines = [f"## {label}\n"]

    for seg in segments:
        cuts_str = seg.get("cuts", "")
        if not cuts_str:
            continue
        sections = cuts_to_sections(words, cuts_str)
        total_dur = sum(s["duration"] for s in sections)
        total_words = sum(s["word_count"] for s in sections)

        lines.append(f"### Segment {seg.get('segment', '?')}")
        lines.append(f"Summary: {seg.get('summary', 'N/A')}")
        lines.append(
            f"Duration: {total_dur:.1f}s | {len(sections)} sections | {total_words} words"
        )
        if seg.get("has_cold_open"):
            lines.append("Cold open: yes")
        lines.append("")

        for i, section in enumerate(sections):
            is_cold = seg.get("has_cold_open") and i == 0
            label_tag = "COLD OPEN" if is_cold else f"Section {i + 1}"
            lines.append(
                f"  [{label_tag}] {section['start']:.1f}s → {section['end']:.1f}s "
                f"({section['duration']:.1f}s, {section['word_count']} words)"
            )
            lines.append(f'  "{section["text"]}"')
            lines.append("")

    return "\n".join(lines)


def build_transcript_text(words: list[dict]) -> str:
    """Build plain-text transcript from word-level data."""
    return " ".join(w["word"] for w in words)


# ── Judge prompt ──────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """\
You are an expert short-form video editor evaluating machine-generated video cuts.

You will see:
1. The full transcript of a long-form video
2. GROUND TRUTH: Human-curated segments approved for production
3. PROPOSED: Machine-generated segments being evaluated

Score the PROPOSED cuts on 7 dimensions (1-10 each).

Use GROUND TRUTH as a reference for what "good" looks like, but do NOT penalize
valid alternatives. Different editors make different valid choices — what matters
is whether the PROPOSED cuts would produce good short-form content.

## Scoring Rubric

### 1. COLD VIEWER TEST (1-10)
Would someone scrolling TikTok understand each proposed segment from second 1,
with zero prior context?
- 10: Every segment opens with a clear, self-contained statement
- 7: Mostly standalone, minor context gaps
- 4: Some segments open mid-thought or assume prior knowledge
- 1: Most segments are unintelligible without context

### 2. HOOK STRENGTH (1-10)
Do the first 3-5 seconds of each segment grab attention?
- 10: Compelling hooks — surprise, bold claim, vivid scenario
- 7: Solid but not exceptional
- 4: Flat or generic openings
- 1: Would cause immediate scroll-past

### 3. NARRATIVE COHERENCE (1-10)
When sections within each segment are assembled in sequence, do they flow?
- 10: Seamless, cuts feel natural
- 7: Mostly coherent, minor abrupt transitions
- 4: Noticeable gaps or confusing jumps
- 1: Random-feeling section order

### 4. CONTENT SELECTION (1-10)
Did the proposed cuts pick the strongest, most engaging content from the source?
- 10: Comparable quality to ground truth — nails the best moments
- 7: Good but missed some strong moments or included weaker material
- 4: Mediocre — several missed opportunities
- 1: Picked filler, missed the best parts

### 5. CUT PRECISION (1-10)
Do sections start and end at natural sentence/clause boundaries?
- 10: Every section starts at a sentence beginning and ends at a natural break
- 7: Most cuts are clean, 1-2 minor boundary issues
- 4: Several sections start mid-sentence or end on trailing conjunctions
- 1: Most sections have broken boundaries — mid-sentence starts, trailing "and/but/so"

Check the FIRST 5-8 WORDS of each segment especially carefully. If they read
like a sentence fragment ("a product, I, like, probably" instead of
"when it came to building a product"), that's a critical failure.

### 6. DURATION ACCURACY (1-10)
Are segments within the target duration range?
- 10: All within target ± tolerance
- 7: Mostly within range
- 4: Some noticeably outside
- 1: Multiple segments far outside range

### 7. SEGMENT STRUCTURE (1-10)
Is the number and shape of segments appropriate for the source material?
(Compare to ground truth — right number of segments? Reasonable section count?
Appropriate use of cold opens?)
- 10: Structure matches ground truth quality
- 7: Reasonable, minor improvements possible
- 4: Too many/few segments or poor internal structure
- 1: Fundamentally wrong approach"""


def build_judge_user_prompt(
    transcript_text: str,
    ground_truth_formatted: str,
    proposed_formatted: str,
    params: dict,
) -> str:
    """Build the user message for the judge."""
    # Truncate transcript middle if too long (keep context from both ends)
    max_chars = 15_000
    if len(transcript_text) > max_chars:
        half = max_chars // 2
        transcript_text = (
            transcript_text[:half]
            + "\n\n[... middle of transcript omitted ...]\n\n"
            + transcript_text[-half:]
        )

    target = params.get("target_duration", 60)
    tolerance = params.get("tolerance", 20)

    return (
        f"## Source Transcript\n\n{transcript_text}\n\n"
        f"---\n\n"
        f"## Parameters\n"
        f"Target duration: {target}s (±{tolerance}s)\n"
        f"Acceptable range: {target - tolerance}s – {target + tolerance}s\n\n"
        f"---\n\n"
        f"{ground_truth_formatted}\n"
        f"---\n\n"
        f"{proposed_formatted}\n"
        f"---\n\n"
        f"Score the PROPOSED cuts on all 7 dimensions. "
        f"Use GROUND TRUTH as a quality reference, but don't penalize "
        f"equally valid alternative editorial choices."
    )


# ── Core eval logic ───────────────────────────────────────────────────────────


async def judge_output(
    words: list[dict],
    ground_truth: list[dict],
    proposed: list[dict],
    params: dict,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> JudgeEvaluation:
    """Score proposed cuts against ground truth using a judge LLM."""
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")

    llm = ChatOpenAI(
        model=judge_model,
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        timeout=120.0,
    )
    structured_llm = llm.with_structured_output(JudgeEvaluation)

    transcript_text = build_transcript_text(words)
    gt_formatted = format_segments_for_judge(
        ground_truth, words, "GROUND TRUTH (human-curated, production-approved)"
    )
    proposed_formatted = format_segments_for_judge(
        proposed, words, "PROPOSED (machine-generated, being evaluated)"
    )

    messages = [
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(
            content=build_judge_user_prompt(
                transcript_text, gt_formatted, proposed_formatted, params
            )
        ),
    ]

    return await structured_llm.ainvoke(messages)


DIMENSIONS = [
    "cold_viewer_test",
    "hook_strength",
    "narrative_coherence",
    "content_selection",
    "cut_precision",
    "duration_accuracy",
    "segment_structure",
]


async def run_single_case(
    case: dict,
    model: str | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> dict:
    """Run eval on one test case: propose_cuts → judge → scores."""
    case_id = case["id"]
    transcript_path = PROJECT_ROOT / case["transcript_path"]
    params = case["params"]
    ground_truth = case["ground_truth"]

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"CASE: {case_id}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    # ── Load transcript ──────────────────────────────────────────────────
    _, words = read_word_transcript_file(transcript_path)
    print(f"Loaded {len(words)} words from {transcript_path.name}", file=sys.stderr)

    # ── Run propose_cuts ─────────────────────────────────────────────────
    model_display = model or "default"
    print(f"\nRunning propose_cuts (model: {model_display})...", file=sys.stderr)
    t0 = time.time()
    proposed = await propose_cuts(
        words,
        target_duration=params.get("target_duration", 60),
        tolerance=params.get("tolerance", 20),
        custom_prompt=params.get("custom_prompt"),
        max_segments=params.get("max_segments"),
        model=model,
    )
    propose_elapsed = time.time() - t0
    print(
        f"propose_cuts: {len(proposed)} segment(s) in {propose_elapsed:.1f}s",
        file=sys.stderr,
    )

    # ── Judge ────────────────────────────────────────────────────────────
    print(f"\nJudging with {judge_model}...", file=sys.stderr)
    t0 = time.time()
    evaluation = await judge_output(words, ground_truth, proposed, params, judge_model)
    judge_elapsed = time.time() - t0
    print(f"Judge: {judge_elapsed:.1f}s", file=sys.stderr)

    # ── Collect scores ───────────────────────────────────────────────────
    scores = {}
    for dim in DIMENSIONS:
        dim_score: DimensionScore = getattr(evaluation, dim)
        scores[dim] = {"score": dim_score.score, "reasoning": dim_score.reasoning}

    mean_score = sum(s["score"] for s in scores.values()) / len(scores)

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n  Scores:", file=sys.stderr)
    for dim, data in scores.items():
        bar = "█" * data["score"] + "░" * (10 - data["score"])
        print(
            f"    {dim:<24s} {bar} {data['score']:>2d}/10  {data['reasoning']}",
            file=sys.stderr,
        )
    print(f"    {'MEAN':<24s} {'':>10s} {mean_score:.1f}/10", file=sys.stderr)
    print(f"\n  Notes: {evaluation.overall_notes}", file=sys.stderr)

    return {
        "case_id": case_id,
        "gt_segment_count": len(ground_truth),
        "proposed_segment_count": len(proposed),
        "proposed_segments": proposed,
        "scores": scores,
        "mean_score": round(mean_score, 2),
        "overall_notes": evaluation.overall_notes,
        "elapsed": {
            "propose_s": round(propose_elapsed, 1),
            "judge_s": round(judge_elapsed, 1),
        },
    }


# ── Aggregation ───────────────────────────────────────────────────────────────


def aggregate_results(results: list[dict]) -> dict:
    """Compute aggregate statistics across all results."""
    agg = {}
    for dim in DIMENSIONS:
        dim_scores = [r["scores"][dim]["score"] for r in results]
        agg[dim] = {
            "mean": round(sum(dim_scores) / len(dim_scores), 2),
            "min": min(dim_scores),
            "max": max(dim_scores),
        }

    all_means = [r["mean_score"] for r in results]
    return {
        "overall_mean": round(sum(all_means) / len(all_means), 2),
        "overall_min": min(all_means),
        "overall_max": max(all_means),
        "per_dimension": agg,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval harness for propose_cuts — measure before you tweak",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenRouter model ID to test (default: propose_cuts default)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model ID (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Run only a specific test case by ID",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Run each case N times to measure variance (default: 1)",
    )
    args = parser.parse_args()

    # ── Load test cases ──────────────────────────────────────────────────
    if not DATASET_DIR.exists():
        print(f"Error: Dataset directory not found: {DATASET_DIR}", file=sys.stderr)
        sys.exit(1)

    cases = []
    for f in sorted(DATASET_DIR.glob("*.json")):
        case = json.loads(f.read_text())
        if args.case and case["id"] != args.case:
            continue
        transcript_path = PROJECT_ROOT / case["transcript_path"]
        if not transcript_path.exists():
            print(
                f"Skipping {case['id']}: transcript not found at {case['transcript_path']}",
                file=sys.stderr,
            )
            continue
        cases.append(case)

    if not cases:
        print(
            "No matching test cases with existing transcripts found.", file=sys.stderr
        )
        sys.exit(1)

    print(f"Eval: {len(cases)} case(s) × {args.runs} run(s)", file=sys.stderr)
    print(f"Model under test: {args.model or 'default'}", file=sys.stderr)
    print(f"Judge: {args.judge_model}", file=sys.stderr)

    # ── Run ───────────────────────────────────────────────────────────────
    all_results = []
    for run_idx in range(args.runs):
        if args.runs > 1:
            print(
                f"\n{'─' * 40} Run {run_idx + 1}/{args.runs} {'─' * 40}",
                file=sys.stderr,
            )
        for case in cases:
            result = await run_single_case(
                case, model=args.model, judge_model=args.judge_model
            )
            result["run"] = run_idx + 1
            all_results.append(result)

    # ── Aggregate & report ────────────────────────────────────────────────
    agg = aggregate_results(all_results)

    output = {
        "run_id": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": args.model or "default",
        "judge_model": args.judge_model,
        "case_count": len(cases),
        "run_count": args.runs,
        "results": all_results,
        "aggregate": agg,
    }

    # Save to results/
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_slug = (args.model or "default").replace("/", "_")
    results_path = RESULTS_DIR / f"{ts}_{model_slug}.json"
    results_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {results_path}", file=sys.stderr)

    # Summary table
    print(f"\n{'=' * 60}", file=sys.stderr)
    print("EVAL SUMMARY", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"Overall mean: {agg['overall_mean']}/10", file=sys.stderr)
    if args.runs > 1:
        print(
            f"Overall range: {agg['overall_min']} – {agg['overall_max']}",
            file=sys.stderr,
        )
    for dim, data in agg["per_dimension"].items():
        range_str = f" (range: {data['min']}–{data['max']})" if args.runs > 1 else ""
        print(f"  {dim:<24s} {data['mean']:>4.1f}/10{range_str}", file=sys.stderr)
    print(f"{'=' * 60}\n", file=sys.stderr)

    # Stdout JSON for piping
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
