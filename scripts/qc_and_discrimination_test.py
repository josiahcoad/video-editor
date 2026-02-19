#!/usr/bin/env python3
"""
QC and discrimination test for AI-generated social copy (e.g. write_copy output).

Use this after generating captions with write_copy or apply_batch + write_copy
using a client voice file. It (1) rates how well the copy matches the voice, and
(2) tests whether other LLMs can tell the AI captions from real ones.

When to use
-----------
- You added or updated a voice file (e.g. from extracted IG captions) and want to
  validate that write_copy output matches that voice.
- You want to know if the AI copy is distinguishable from the client's real
  posts (discrimination test across N models).

Usage
-----
  # Default: HunterZier with_copy + voice; runs QC then discrimination test
  uv run python scripts/qc_and_discrimination_test.py

  # Custom paths
  uv run python scripts/qc_and_discrimination_test.py \\
    --with-copy-dir projects/HunterZier/editing/videos/.../outputs/with_copy \\
    --voice projects/HunterZier/voices/default.md

  # Skip QC, only run discrimination test
  uv run python scripts/qc_and_discrimination_test.py --skip-qc

  # Use specific models for discrimination (OpenRouter model IDs)
  uv run python scripts/qc_and_discrimination_test.py --models anthropic/claude-3.5-sonnet openai/gpt-4o

Requires OPENROUTER_API_KEY.

Interpreting results
--------------------
QC (style match)
  One LLM scores the first 5 generated captions 1–5 (5 = could pass as the
  client's post). Read the per-caption "reason" for tells (e.g. "bullet points
  too formal", "slightly more polished"). Use this to refine the voice file or
  prompt if scores are low or reasons repeat.

Discrimination test
  We shuffle 10 captions (5 real from REAL_HUNTER_CAPTIONS, 5 AI from
  with_copy), label them 1–10, and ask each model: "Which 5 are AI-generated?"
  We score how many of the 5 AI indices the model correctly identified.

  Random chance: P(5/5 correct) ≈ 0.4%, P(4/5) ≈ 10%. So 4 or 5 correct suggests
  the model can distinguish AI from real. Mean across models ~2/5 (or below
  random expectation 2.5) means no statistical significance — the AI copy is
  hard to tell apart, which is good for voice match.

  Summary: If most models get 2/5 or 3/5, the voice is doing its job. If several
  get 4+ or 5/5, consider tightening the voice file or reducing polish in
  write_copy (e.g. fewer bullet lists).
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

# Add repo root for src.edit
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_openai import ChatOpenAI

# 5 real Hunter Zier captions (first-person, substantive; from extracted IG alts)
# Using triple-quotes to allow internal double quotes.
REAL_HUNTER_CAPTIONS = [
    """If you've owned a home for a while, you've probably noticed something…
... the refinance mail never really stops.
And at some point, you start wondering if you're supposed to be doing something about it.
A lower rate sounds good. Of course it does.
But that doesn't automatically mean it's the right move for you.
You've got to look at the full picture and ask yourself questions like:
How long are you planning to stay?
What do the costs actually look like?
What are you trying to accomplish financially?
Sometimes refinancing makes a lot of sense.
Sometimes it barely changes anything.
This isn't something you want to guess on.
So if you've been unsure whether it's worth exploring, let's just run the numbers together.
Send us a message anytime. We're happy to take a look. 👋""",
    """For a lot of homebuyers, the biggest stress isn't qualifying for a mortgage…
It's the fear of covering two places at once.
Of draining your savings right at the start. 💸
Of turning what should feel exciting into a financial scramble.
The good news is, with the right planning, those early months don't have to feel that way.
If timing, cash flow, or overlapping payments are part of what's holding you back, let's talk it through.
Sometimes a small adjustment can make a big difference. 😉
Reach out anytime. We're here to help.""",
    """Let's be honest for a second… buying a home right now can feel impossible, right? 😔
And somewhere in the back of your mind, you're probably wondering…
"What if I wait too long?"
Or…
"What if I move too fast and regret it?" 🤔
That hesitation makes sense. This isn't like any everyday purchase... It's a life decision.
But here's the thing: you don't need to have everything figured out today.
If you're clear on where you want to live, and what feels financially comfortable for you, you're doing exactly what you should be doing right now.
The rest? You don't have to overthink it. You don't have to navigate it alone.
That's where we come in. We're here to help you think it through. 😉""",
    """Only need 3 things to buy a home✅
Credit, 580 minimum for FHA loan. Every 20 points you go up you get a better interest rate. We assist you in getting pre approved for free, only using a soft pull, doesn't show up as hard inquiry on credit.
Collateral, 0% down programs available. You can even do a PMI buy out on a 3% down conventional loan. A PMI buy out costs about 2.75% of the loan amount. So you can budget 5% down if you'd like to buy out your PMI without putting 20% down. *Need 700 credit for conventional loan*
Capacity, 2 years of work history, 49% of your income minus is your monthly debt (cars, student loans, credit cards)
Are you hourly or salary?
$30 an hour, 40 hours a week x 4.3 weeks in a month. Is $5160 gross, x 0.49 = $2,528 max monthly payment. Go to a mortgage calculator and find out how much $ you'd be pre approved for✅ In this scenario it would be $325,000-$375,000 depending on your credit score! that's it! And you're ready to buy a home!
Comment or DM me the word "Home" and I'll send you my free home buyer guide and help you calculate what you'd be pre approved for!""",
    """Just wanted to let you guys know @huntergzier and I got our buyers a fully renovated home for $44,000 off price..🤯
Got it under contract for $295,000 using the first time home buyers program, and got the seller to contribute $13,800 to closing costs and an interest rate buy down to 5.3%!! Payment just under $2,100
Just wanted to give you a heads up and good news, it's a strong buyers market. This home was originally listed for $325,000!
And after all seller paid closing costs, our offer was all the way down to $283,000🔥amazing almost $44,000 off price""",
]


def get_openrouter_llm(model: str, temperature: float = 0.3):
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("Error: OPENROUTER_API_KEY required")
    return ChatOpenAI(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        temperature=temperature,
    )


def run_qc(voice_path: Path, with_copy_dir: Path, qc_model: str) -> None:
    """Rate generated captions against Hunter's voice (1-5 + explanation)."""
    voice_text = voice_path.read_text()
    copy_files = sorted(with_copy_dir.glob("segment_*.json"))[:5]
    captions = []
    for f in copy_files:
        data = json.loads(f.read_text())
        captions.append(data.get("caption", "")[:800])
    sample = "\n\n---\n\n".join(f"[Caption {i+1}]\n{c}" for i, c in enumerate(captions))

    prompt = f"""You are a style editor. Hunter Zier is a loan officer (mortgage/home buying content) who posts on Instagram. His voice is direct, reassuring, educational, soft CTAs ("reach out anytime", "let's run the numbers"), moderate emojis, no pushy sales.

Below are 5 captions generated by an AI to match Hunter's voice (from his voice guidelines and transcript).

VOICE GUIDELINES (excerpt):
{voice_text[:2500]}

GENERATED CAPTIONS:
{sample}

For each caption, rate 1-5 how well it matches Hunter's style (5 = could pass as his own post, 1 = clearly generic AI). Then give one sentence explaining the main strength or tell. Output valid JSON only, no markdown:
{{ "ratings": [ {{ "caption": 1, "score": 4, "reason": "..." }}, ... ] }}"""

    llm = get_openrouter_llm(qc_model)
    out = llm.invoke(prompt)
    text = out.content if hasattr(out, "content") else str(out)
    # Strip markdown code block if present
    if "```" in text:
        text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    try:
        data = json.loads(text)
        print("\n--- QC: Style match (1-5) ---")
        for r in data.get("ratings", data if isinstance(data, list) else [])[:5]:
            cap = r.get("caption", r.get("caption_number", "?"))
            score = r.get("score", "?")
            reason = r.get("reason", "")
            print(f"  Caption {cap}: {score}/5 — {reason}")
    except json.JSONDecodeError:
        print("\n--- QC (raw) ---\n" + text[:1500])


def parse_model_answer(text: str) -> set[int]:
    """Extract set of 1-10 from model response (the indices it says are AI)."""
    # Find all numbers 1-10 in the response
    numbers = re.findall(r"\b([1-9]|10)\b", text)
    return set(int(n) for n in numbers)


def run_discrimination_test(
    with_copy_dir: Path,
    models: list[str],
    num_ai: int = 5,
    seed: int = 42,
) -> dict[str, int]:
    """
    Shuffle 5 real + 5 AI captions (labeled 1-10). Ask each model: which 5 are AI?
    Score = how many of the 5 AI indices it correctly identified.
    Random chance of 5/5 = 1/C(10,5) ≈ 0.4%. 4/5 ≈ 9.9%.
    """
    random.seed(seed)
    copy_files = sorted(with_copy_dir.glob("segment_*.json"))
    if len(copy_files) < num_ai:
        raise SystemExit(f"Need at least {num_ai} segment JSONs in {with_copy_dir}")
    ai_captions = []
    for f in copy_files[:num_ai]:
        data = json.loads(f.read_text())
        ai_captions.append(data.get("caption", "").strip())
    real_captions = REAL_HUNTER_CAPTIONS[:num_ai]
    # Build labeled list: (label 1-10, caption, is_ai)
    combined = [(i + 1, c, True) for i, c in enumerate(ai_captions)]
    combined += [(num_ai + i + 1, c, False) for i, c in enumerate(real_captions)]
    random.shuffle(combined)
    # Map label -> is_ai for scoring
    label_to_ai = {label: is_ai for label, _c, is_ai in combined}
    ai_labels = {label for label, is_ai in label_to_ai.items() if is_ai}

    numbered = "\n\n".join(
        f"[{label}]\n{caption[:1200]}" for label, caption, _ in combined
    )

    prompt = f"""Hunter Zier is a loan officer who posts on Instagram (mortgage, first-time homebuyer, Spokane/WA/Idaho). His captions are direct, reassuring, educational, with soft CTAs like "reach out anytime" and "let's run the numbers."

Below are 10 captions. Exactly 5 were written by Hunter. The other 5 were generated by an AI to mimic his voice.

Your task: Which 5 are the AI-generated captions? Look for tells (e.g. overly even structure, generic phrasing, slight mismatch to his voice).

{numbered}

Reply with ONLY the 5 numbers that are AI-generated, comma-separated (e.g. 2, 4, 5, 7, 9). No explanation."""

    results = {}
    for model in models:
        llm = get_openrouter_llm(model)
        try:
            out = llm.invoke(prompt)
            text = out.content if hasattr(out, "content") else str(out)
            guessed = parse_model_answer(text)
            if len(guessed) > 5:
                guessed = set(list(guessed)[:5])
            correct = len(guessed & ai_labels)
            results[model] = correct
            print(
                f"  {model}: guessed {sorted(guessed)} -> {correct}/5 correct (AI were {sorted(ai_labels)})"
            )
        except Exception as e:
            print(f"  {model}: error — {e}")
            results[model] = -1
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QC AI copy vs Hunter voice + discrimination test across N models",
    )
    parser.add_argument(
        "--with-copy-dir",
        type=Path,
        default=REPO_ROOT
        / "projects/HunterZier/editing/videos/260214-hunter-session1/outputs/with_copy",
        help="Directory with segment_01.json ... (caption field used as AI copy)",
    )
    parser.add_argument(
        "--voice",
        type=Path,
        default=REPO_ROOT / "projects/HunterZier/voices/default.md",
        help="Hunter voice file for QC",
    )
    parser.add_argument(
        "--skip-qc",
        action="store_true",
        help="Skip the style-match QC, only run discrimination test",
    )
    parser.add_argument(
        "--qc-model",
        default="anthropic/claude-3.5-sonnet",
        help="Model for QC rating",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "google/gemini-2.0-flash-001",
            "meta-llama/llama-3.1-70b-instruct",
            "anthropic/claude-opus-4",
        ],
        help="OpenRouter model IDs for discrimination test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffle (reproducible)",
    )
    args = parser.parse_args()

    args.with_copy_dir = args.with_copy_dir.resolve()
    args.voice = args.voice.resolve()
    if not args.with_copy_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.with_copy_dir}")
    if not args.voice.is_file():
        raise SystemExit(f"Voice file not found: {args.voice}")

    if not args.skip_qc:
        run_qc(args.voice, args.with_copy_dir, args.qc_model)

    print("\n--- Discrimination test (which 5 are AI?) ---")
    print(
        "Random chance: 5/5 ≈ 0.4%%, 4/5 ≈ 10%%. So 4+ suggests the model can distinguish.\n"
    )
    results = run_discrimination_test(args.with_copy_dir, args.models, seed=args.seed)

    print("\n--- Summary ---")
    for model, correct in results.items():
        if correct >= 0:
            sig = " (suggestive)" if correct >= 4 else ""
            print(f"  {model}: {correct}/5{sig}")
    n = len([r for r in results.values() if r >= 0])
    if n:
        avg = sum(r for r in results.values() if r >= 0) / n
        print(f"\n  Mean correct across models: {avg:.1f}/5")


if __name__ == "__main__":
    main()
