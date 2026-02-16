"""One-shot script to import Billy's conversation from text files into the DB."""

import asyncio
import json
import pathlib
import re

from db import init_db, get_db

CONV_DIR = pathlib.Path(__file__).parent / "conversations" / "billy"


def parse_conversation(text: str) -> list[dict]:
    turns = []
    blocks = re.split(r"\n(?=TURN \d+)", text.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if not lines or not lines[0].startswith("TURN"):
            continue
        turn_num = int(re.search(r"\d+", lines[0]).group())
        confidence = lines[1].strip() if len(lines) > 1 else ""
        body = "\n".join(lines[2:]).strip()
        turns.append(
            {
                "turn": turn_num,
                "confidence": confidence,
                "text": body,
            }
        )
    return turns


def parse_coaching(text: str) -> list[dict]:
    tips = []
    blocks = re.split(r"\n(?=#\d+)", text.strip())
    for block in blocks:
        block = block.strip()
        m = re.match(r"#(\d+)\s*\n\s*turn\s+(\d+)\s*\n(.*)", block, re.DOTALL)
        if not m:
            continue
        tips.append(
            {
                "number": int(m.group(1)),
                "turn": int(m.group(2)),
                "advice": m.group(3).strip().strip('"'),
                "close_score": -1,
                "steps_to_close": -1,
                "is_custom": False,
            }
        )
    return tips


async def main():
    await init_db()
    db = await get_db()
    try:
        # Create contact
        cursor = await db.execute(
            "INSERT INTO contacts (name, company, status, notes) VALUES (?, ?, ?, ?)",
            (
                "Billy",
                None,
                "reviewing_contract",
                "Warranty business owner. Wants to blow up on social media. Concerned about video quality and DM management.",
            ),
        )
        contact_id = cursor.lastrowid

        # Parse files
        conv_text = (CONV_DIR / "conversation.txt").read_text()
        coach_text = (CONV_DIR / "coach.txt").read_text()
        transcript = parse_conversation(conv_text)
        coaching = parse_coaching(coach_text)

        # Create conversation
        cursor = await db.execute(
            """INSERT INTO conversations
               (contact_id, context, transcript_json, coaching_json,
                started_at, ended_at, final_close_score, final_steps_to_close,
                mode, next_step)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                contact_id,
                "Kickoff/sales call. Billy interested in social media management package ($4k/mo). "
                "Wants video creation + DM/comment management.",
                json.dumps(transcript),
                json.dumps(coaching),
                "2025-02-10T10:00:00",
                "2025-02-10T10:45:00",
                75,
                2,
                "live",
                "Sign contract on Friday kickoff call at 10 AM PST",
            ),
        )
        conv_id = cursor.lastrowid

        # Insert hesitations
        hesitations = [
            ("Wants to see sample work / video quality proof", "open", 16, None),
            ("$48k/year is significant — needs to justify ROI", "open", 17, None),
            ("Waiting on lawyer callback before signing", "open", 34, None),
            (
                "Overlap with existing web guy — doesn't want to pay twice",
                "open",
                17,
                None,
            ),
            (
                "Worried about spending 4k/mo and still doing a lot of social media work himself",
                "open",
                26,
                None,
            ),
        ]
        for text, status, surfaced, resolved in hesitations:
            await db.execute(
                """INSERT INTO hesitations
                   (conversation_id, contact_id, text, status, surfaced_at_turn, resolved_at_turn)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (conv_id, contact_id, text, status, surfaced, resolved),
            )

        await db.commit()
        print(f"Imported: contact_id={contact_id}, conversation_id={conv_id}")
        print(
            f"  {len(transcript)} turns, {len(coaching)} tips, {len(hesitations)} hesitations"
        )
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
