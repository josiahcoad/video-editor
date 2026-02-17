#!/usr/bin/env python3
"""
Compare propose_cuts across multiple models on all eval datasets.

Runs each model against each test case, judges the output, and produces
a side-by-side comparison table.

Usage:
  dotenvx run -f .env -f ../fast-backend/.env -- uv run python -m evals.propose_cuts.compare_models
  dotenvx run -f .env -f ../fast-backend/.env -- uv run python -m evals.propose_cuts.compare_models --case 260210-content-batch
  dotenvx run -f .env -f ../fast-backend/.env -- uv run python -m evals.propose_cuts.compare_models --parallel 3
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from evals.propose_cuts.run_eval import (
    DATASET_DIR,
    DIMENSIONS,
    RESULTS_DIR,
    run_single_case,
)

# ── Models to compare ─────────────────────────────────────────────────────────

MODELS = [
    ("Opus 4.5", "anthropic/claude-opus-4.5"),
    ("Opus 4.6", "anthropic/claude-opus-4.6"),
    ("Gemini 3 Flash", "google/gemini-3-flash-preview"),
    ("Gemini 3 Pro", "google/gemini-3-pro-preview"),
    ("GPT 5.2", "openai/gpt-5.2"),
    ("GPT 5.2 Pro", "openai/gpt-5.2-pro"),
    ("Grok 4.1 Fast", "x-ai/grok-4.1-fast"),
    ("Kimi K2", "moonshotai/kimi-k2-thinking"),
]


async def run_model_on_case(
    model_name: str, model_id: str, case: dict, judge_model: str
) -> dict:
    """Run a single model on a single case, with error handling."""
    try:
        result = await run_single_case(case, model=model_id, judge_model=judge_model)
        result["model_name"] = model_name
        result["model_id"] = model_id
        return result
    except Exception as e:
        print(
            f"\n  ❌ FAILED: {model_name} on {case['id']}: {e}",
            file=sys.stderr,
        )
        return {
            "case_id": case["id"],
            "model_name": model_name,
            "model_id": model_id,
            "error": str(e),
            "mean_score": 0,
            "scores": {dim: {"score": 0, "reasoning": f"Error: {e}"} for dim in DIMENSIONS},
            "proposed_segment_count": 0,
            "gt_segment_count": len(case.get("ground_truth", [])),
            "elapsed": {"propose_s": 0, "judge_s": 0},
        }


async def run_comparison(
    cases: list[dict],
    judge_model: str,
    parallel: int = 2,
) -> list[dict]:
    """Run all models on all cases with controlled parallelism."""
    # Build the full task list: (model, case) pairs
    tasks = []
    for model_name, model_id in MODELS:
        for case in cases:
            tasks.append((model_name, model_id, case))

    all_results = []
    # Process in batches to respect rate limits
    for i in range(0, len(tasks), parallel):
        batch = tasks[i : i + parallel]
        batch_desc = ", ".join(f"{mn} × {c['id']}" for mn, _, c in batch)
        print(
            f"\n{'━' * 60}",
            file=sys.stderr,
        )
        print(
            f"Batch {i // parallel + 1}/{(len(tasks) + parallel - 1) // parallel}: "
            f"{batch_desc}",
            file=sys.stderr,
        )

        coros = [
            run_model_on_case(mn, mid, case, judge_model)
            for mn, mid, case in batch
        ]
        results = await asyncio.gather(*coros)
        all_results.extend(results)

    return all_results


def print_comparison_table(results: list[dict], cases: list[dict]) -> None:
    """Print a formatted comparison table to stderr."""
    # Group results by case
    by_case: dict[str, list[dict]] = {}
    for r in results:
        cid = r["case_id"]
        by_case.setdefault(cid, []).append(r)

    for case_id, case_results in by_case.items():
        print(f"\n{'═' * 80}", file=sys.stderr)
        print(f"  CASE: {case_id}", file=sys.stderr)
        print(f"{'═' * 80}", file=sys.stderr)

        # Sort by mean score descending
        case_results.sort(key=lambda r: r.get("mean_score", 0), reverse=True)

        # Header
        print(
            f"\n  {'Model':<20s} {'Mean':>5s} │ "
            + " ".join(f"{d[:6]:>6s}" for d in DIMENSIONS)
            + " │ {'Segs':>4s} {'Time':>5s}",
            file=sys.stderr,
        )
        print(f"  {'─' * 20} {'─' * 5} ┼ " + " ".join("─" * 6 for _ in DIMENSIONS) + " ┼ " + "─" * 4 + " " + "─" * 5, file=sys.stderr)

        for r in case_results:
            mean = r.get("mean_score", 0)
            scores = r.get("scores", {})
            seg_count = r.get("proposed_segment_count", 0)
            elapsed = r.get("elapsed", {})
            total_time = elapsed.get("propose_s", 0) + elapsed.get("judge_s", 0)

            dim_scores = " ".join(
                f"{scores.get(d, {}).get('score', 0):>6d}" for d in DIMENSIONS
            )

            error = r.get("error")
            if error:
                print(
                    f"  {r['model_name']:<20s} {'ERR':>5s} │ {error[:60]}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"  {r['model_name']:<20s} {mean:>5.1f} │ {dim_scores} │ {seg_count:>4d} {total_time:>5.0f}s",
                    file=sys.stderr,
                )

        # Print notes for top 3
        print(f"\n  Notes:", file=sys.stderr)
        for i, r in enumerate(case_results[:3]):
            if not r.get("error"):
                notes = r.get("overall_notes", "N/A")
                print(f"    #{i + 1} {r['model_name']}: {notes[:120]}", file=sys.stderr)

    # Overall summary across all cases
    if len(by_case) > 1:
        print(f"\n{'═' * 80}", file=sys.stderr)
        print(f"  OVERALL SUMMARY (across {len(by_case)} cases)", file=sys.stderr)
        print(f"{'═' * 80}", file=sys.stderr)

        model_means: dict[str, list[float]] = {}
        for r in results:
            mn = r.get("model_name", "?")
            model_means.setdefault(mn, []).append(r.get("mean_score", 0))

        ranked = sorted(
            model_means.items(),
            key=lambda x: sum(x[1]) / len(x[1]),
            reverse=True,
        )
        for mn, scores in ranked:
            avg = sum(scores) / len(scores)
            print(f"  {mn:<20s} {avg:.1f}/10", file=sys.stderr)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare propose_cuts across models"
    )
    parser.add_argument("--case", type=str, default=None, help="Run specific case only")
    parser.add_argument(
        "--judge-model",
        type=str,
        default="openai/gpt-4o",
        help="Judge model (default: openai/gpt-4o)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=2,
        help="Max parallel model runs (default: 2)",
    )
    args = parser.parse_args()

    # Load cases
    if not DATASET_DIR.exists():
        print(f"Error: {DATASET_DIR} not found", file=sys.stderr)
        sys.exit(1)

    cases = []
    for f in sorted(DATASET_DIR.glob("*.json")):
        case = json.loads(f.read_text())
        if args.case and case["id"] != args.case:
            continue
        cases.append(case)

    if not cases:
        print("No matching cases", file=sys.stderr)
        sys.exit(1)

    print(
        f"Model comparison: {len(MODELS)} models × {len(cases)} case(s)",
        file=sys.stderr,
    )
    print(f"Judge: {args.judge_model}", file=sys.stderr)
    print(f"Parallelism: {args.parallel}", file=sys.stderr)
    print(f"Models: {', '.join(n for n, _ in MODELS)}", file=sys.stderr)

    t0 = time.time()
    results = await run_comparison(cases, args.judge_model, args.parallel)
    total_elapsed = time.time() - t0

    print(f"\nTotal time: {total_elapsed:.0f}s", file=sys.stderr)

    # Print comparison table
    print_comparison_table(results, cases)

    # Save full results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = {
        "run_id": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "type": "model_comparison",
        "judge_model": args.judge_model,
        "models": [{"name": n, "id": mid} for n, mid in MODELS],
        "case_count": len(cases),
        "results": results,
        "total_elapsed_s": round(total_elapsed, 1),
    }
    results_path = RESULTS_DIR / f"{ts}_model_comparison.json"
    results_path.write_text(json.dumps(output, indent=2))
    print(f"\nFull results: {results_path}", file=sys.stderr)

    # Also output JSON to stdout for piping
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
