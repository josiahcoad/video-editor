"""One-off script to test the terse coaching prompt with a sample transcript.
Run from sales-coach: dotenvx run -f .env -f .env.dev -- uv run python test_coaching_prompt.py
"""

import asyncio
import os

from app import get_coaching_advice_only

# Sample transcript: rep uncovering volume problem, prospect engaged but no commitment yet
SAMPLE_TRANSCRIPT = [
    "Rep: Hi, this is Alex from Acme. Do you have a few minutes to talk about how we're helping companies like yours reduce support tickets?",
    "Prospect: Yeah, I guess. We're pretty busy though.",
    "Rep: I hear you. What's the biggest challenge you're facing with support right now?",
    "Prospect: Honestly it's just volume. We're getting a lot of tickets and the team is overwhelmed.",
    "Rep: How many tickets are we talking about per week?",
    "Prospect: Around 500. It's been growing.",
    "Rep: What have you tried so far to handle the load?",
    "Prospect: We hired two more people but it's still not enough. And we're not sure we can keep adding headcount.",
]

# Second scenario: prospect stalls
STALL_TRANSCRIPT = [
    "Rep: Based on what you've shared, it sounds like Acme could help. What would you need to see to move forward?",
    "Prospect: I don't know, let me think about it.",
    "Rep: Sure. Is there anything specific you want to run by your team?",
    "Prospect: Maybe. I'll need to check with my manager and get back to you.",
]


async def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "Set OPENROUTER_API_KEY (e.g. dotenvx run -f .env -f .env.dev -- uv run python test_coaching_prompt.py)"
        )
        return

    for name, transcript in [
        ("Discovery (volume)", SAMPLE_TRANSCRIPT),
        ("Stall", STALL_TRANSCRIPT),
    ]:
        print("=" * 60)
        print("Scenario:", name)
        print("Transcript (last 2 turns):")
        for i, t in enumerate(transcript[-2:], start=len(transcript) - 1):
            print(f"  {i}: {t[:75]}...")
        print()

        result = await get_coaching_advice_only(
            transcript,
            custom_question=None,
            contact_notes="",
            call_prep="",
        )

        advice = result.get("advice", "")
        print("Advice (slip of paper):", repr(advice))
        print("Length (chars):", len(advice))
        if len(advice) > 80:
            print("⚠️  Advice is long — consider tightening.")
        else:
            print("✓  Terse.")
        print()


if __name__ == "__main__":
    asyncio.run(main())
