"""
SQLite persistence for Sales Coach.

Tables:
  contacts       — people you sell to (with pipeline status)
  conversations  — call sessions (transcript + coaching as JSON, next_step)
  hesitations    — objections/concerns tracked across conversations
"""

import json
import logging
import pathlib
from datetime import datetime, timezone
from typing import Any

import aiosqlite

logger = logging.getLogger("sales-coach")

DB_PATH = pathlib.Path(__file__).parent / "sales_coach.db"

CONTACT_STATUSES = [
    "prospect",
    "interested",
    "demoing",
    "reviewing_contract",
    "closed_won",
    "closed_lost",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS contacts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    company    TEXT,
    notes      TEXT,
    status     TEXT DEFAULT 'prospect',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS conversations (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    contact_id            INTEGER REFERENCES contacts(id),
    context               TEXT,
    transcript_json       TEXT,
    coaching_json         TEXT,
    started_at            TEXT DEFAULT (datetime('now')),
    ended_at              TEXT,
    final_close_score     INTEGER,
    final_steps_to_close  INTEGER,
    mode                  TEXT DEFAULT 'live',
    next_step             TEXT,
    review_json           TEXT,
    review_generated_at   TEXT
);

CREATE TABLE IF NOT EXISTS hesitations (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id       INTEGER NOT NULL REFERENCES conversations(id),
    contact_id            INTEGER REFERENCES contacts(id),
    text                  TEXT NOT NULL,
    status                TEXT DEFAULT 'open' CHECK(status IN ('open', 'resolved', 'carried_forward')),
    rank                  TEXT CHECK(rank IN ('S', 'M', 'L')),
    resolution_suggestion TEXT,
    surfaced_at_turn     INTEGER,
    resolved_at_turn      INTEGER,
    created_at            TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_hesitations_conversation ON hesitations(conversation_id);
CREATE INDEX IF NOT EXISTS idx_hesitations_contact      ON hesitations(contact_id);
CREATE INDEX IF NOT EXISTS idx_hesitations_status        ON hesitations(status);
CREATE INDEX IF NOT EXISTS idx_conversations_contact     ON conversations(contact_id);
"""

# Migrations for existing DBs (idempotent ALTER TABLE statements)
MIGRATIONS = [
    "ALTER TABLE contacts ADD COLUMN status TEXT DEFAULT 'prospect'",
    "ALTER TABLE conversations ADD COLUMN next_step TEXT",
    "ALTER TABLE hesitations ADD COLUMN resolution_suggestion TEXT",
    "ALTER TABLE hesitations ADD COLUMN rank TEXT",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db() -> None:
    db = await get_db()
    try:
        await db.executescript(SCHEMA)
        await db.commit()
        # Run migrations (ignore "duplicate column" errors)
        for sql in MIGRATIONS:
            try:
                await db.execute(sql)
                await db.commit()
            except Exception:
                pass
    finally:
        await db.close()


# ── Contacts ─────────────────────────────────────────────────────


async def create_contact(
    name: str,
    company: str | None = None,
    notes: str | None = None,
    status: str = "prospect",
) -> int:
    db = await get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO contacts (name, company, notes, status) VALUES (?, ?, ?, ?)",
            (name, company, notes, status),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def list_contacts() -> list[dict]:
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT * FROM contacts ORDER BY updated_at DESC"
        )
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_contact(contact_id: int) -> dict | None:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM contacts WHERE id = ?", (contact_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def update_contact(contact_id: int, **fields) -> None:
    allowed = {"name", "company", "notes", "status"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return
    db = await get_db()
    try:
        sets = ", ".join(f"{k} = ?" for k in fields)
        vals = list(fields.values()) + [contact_id]
        await db.execute(
            f"UPDATE contacts SET {sets}, updated_at = datetime('now') WHERE id = ?",
            vals,
        )
        await db.commit()
    finally:
        await db.close()


async def get_contact_detail(contact_id: int) -> dict | None:
    """Full contact with latest conversation's next_step, conversations list, and open hesitations."""
    contact = await get_contact(contact_id)
    if not contact:
        return None

    db = await get_db()
    try:
        # Conversations for this contact (most recent first)
        rows = await db.execute_fetchall(
            """SELECT id, context, started_at, ended_at,
                      final_close_score, final_steps_to_close, mode,
                      next_step, review_generated_at
               FROM conversations
              WHERE contact_id = ?
              ORDER BY started_at DESC""",
            (contact_id,),
        )
        conversations = [dict(r) for r in rows]

        # Derive next_step from latest conversation that has one
        next_step = None
        for conv in conversations:
            if conv.get("next_step"):
                next_step = conv["next_step"]
                break

        # Open hesitations across all conversations
        hes_rows = await db.execute_fetchall(
            """SELECT h.*, c.started_at AS conversation_date
               FROM hesitations h
               LEFT JOIN conversations c ON h.conversation_id = c.id
              WHERE h.contact_id = ? AND h.status = 'open'
              ORDER BY h.created_at DESC""",
            (contact_id,),
        )
        open_hesitations = [dict(r) for r in hes_rows]

        # Conversation count
        contact["conversations"] = conversations
        contact["open_hesitations"] = open_hesitations
        contact["next_step"] = next_step
        contact["conversation_count"] = len(conversations)
        return contact
    finally:
        await db.close()


async def list_contacts_with_summary() -> list[dict]:
    """List all contacts with their latest next_step and open hesitation count."""
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            """SELECT
                 ct.*,
                 (SELECT next_step FROM conversations
                   WHERE contact_id = ct.id AND next_step IS NOT NULL
                   ORDER BY started_at DESC LIMIT 1) AS next_step,
                 (SELECT COUNT(*) FROM conversations
                   WHERE contact_id = ct.id) AS conversation_count,
                 (SELECT COUNT(*) FROM hesitations
                   WHERE contact_id = ct.id AND status = 'open') AS open_hesitation_count,
                 (SELECT final_close_score FROM conversations
                   WHERE contact_id = ct.id AND final_close_score IS NOT NULL
                   ORDER BY started_at DESC LIMIT 1) AS latest_close_score
               FROM contacts ct
              ORDER BY ct.updated_at DESC"""
        )
        return [dict(r) for r in rows]
    finally:
        await db.close()


# ── Conversations ────────────────────────────────────────────────


async def create_conversation(
    contact_id: int | None = None,
    context: str = "",
    mode: str = "live",
) -> int:
    db = await get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO conversations (contact_id, context, mode, started_at) VALUES (?, ?, ?, ?)",
            (contact_id, context, mode, _now()),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def end_conversation(
    conversation_id: int,
    transcript: list[dict],
    coaching: list[dict],
    final_close_score: int | None = None,
    final_steps_to_close: int | None = None,
) -> None:
    db = await get_db()
    try:
        await db.execute(
            """UPDATE conversations
               SET transcript_json = ?, coaching_json = ?,
                   final_close_score = ?, final_steps_to_close = ?,
                   ended_at = ?
             WHERE id = ?""",
            (
                json.dumps(transcript),
                json.dumps(coaching),
                final_close_score,
                final_steps_to_close,
                _now(),
                conversation_id,
            ),
        )
        await db.commit()
    finally:
        await db.close()


async def save_review(conversation_id: int, review: dict) -> None:
    """Save review and extract next_step into its own column."""
    next_step = review.get("next_step", "")
    db = await get_db()
    try:
        await db.execute(
            """UPDATE conversations
               SET review_json = ?, review_generated_at = ?, next_step = ?
             WHERE id = ?""",
            (json.dumps(review), _now(), next_step, conversation_id),
        )
        # If the review suggests a status change, update the contact
        suggested_status = review.get("suggested_status")
        if suggested_status and suggested_status in CONTACT_STATUSES:
            cursor = await db.execute(
                "SELECT contact_id FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            row = await cursor.fetchone()
            if row and row["contact_id"]:
                await db.execute(
                    "UPDATE contacts SET status = ?, updated_at = datetime('now') WHERE id = ?",
                    (suggested_status, row["contact_id"]),
                )
        await db.commit()
    finally:
        await db.close()


async def get_conversation(conversation_id: int) -> dict | None:
    db = await get_db()
    try:
        cursor = await db.execute(
            """SELECT c.*, ct.name AS contact_name, ct.company AS contact_company
             FROM conversations c
             LEFT JOIN contacts ct ON c.contact_id = ct.id
            WHERE c.id = ?""",
            (conversation_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        d["transcript"] = json.loads(d.pop("transcript_json") or "[]")
        d["coaching"] = json.loads(d.pop("coaching_json") or "[]")
        d["review"] = json.loads(d.pop("review_json") or "null")
        return d
    finally:
        await db.close()


async def list_conversations(
    limit: int = 50, contact_id: int | None = None
) -> list[dict]:
    db = await get_db()
    try:
        if contact_id:
            rows = await db.execute_fetchall(
                """SELECT c.id, c.contact_id, c.context, c.started_at, c.ended_at,
                          c.final_close_score, c.final_steps_to_close, c.mode,
                          c.next_step, c.review_generated_at,
                          ct.name AS contact_name, ct.company AS contact_company
                   FROM conversations c
                   LEFT JOIN contacts ct ON c.contact_id = ct.id
                  WHERE c.contact_id = ?
                  ORDER BY c.started_at DESC LIMIT ?""",
                (contact_id, limit),
            )
        else:
            rows = await db.execute_fetchall(
                """SELECT c.id, c.contact_id, c.context, c.started_at, c.ended_at,
                          c.final_close_score, c.final_steps_to_close, c.mode,
                          c.next_step, c.review_generated_at,
                          ct.name AS contact_name, ct.company AS contact_company
                   FROM conversations c
                   LEFT JOIN contacts ct ON c.contact_id = ct.id
                  ORDER BY c.started_at DESC LIMIT ?""",
                (limit,),
            )
        return [dict(r) for r in rows]
    finally:
        await db.close()


# ── Hesitations ──────────────────────────────────────────────────


async def save_hesitations(
    conversation_id: int,
    hesitations: list[dict],
    contact_id: int | None = None,
) -> None:
    """Replace all hesitations for a conversation with the latest set."""
    db = await get_db()
    try:
        await db.execute(
            "DELETE FROM hesitations WHERE conversation_id = ?", (conversation_id,)
        )
        for h in hesitations:
            rank = h.get("rank")
            if rank not in ("S", "M", "L"):
                rank = None
            await db.execute(
                """INSERT INTO hesitations
                   (conversation_id, contact_id, text, status, rank, resolution_suggestion, surfaced_at_turn, resolved_at_turn)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    conversation_id,
                    contact_id,
                    h.get("text", ""),
                    h.get("status", "open"),
                    rank,
                    h.get("resolution_suggestion"),
                    h.get("surfaced_at_turn"),
                    h.get("resolved_at_turn"),
                ),
            )
        await db.commit()
    finally:
        await db.close()


async def get_hesitations(
    conversation_id: int | None = None,
    contact_id: int | None = None,
    status: str | None = None,
) -> list[dict]:
    db = await get_db()
    try:
        clauses = []
        params: list[Any] = []
        if conversation_id:
            clauses.append("h.conversation_id = ?")
            params.append(conversation_id)
        if contact_id:
            clauses.append("h.contact_id = ?")
            params.append(contact_id)
        if status:
            clauses.append("h.status = ?")
            params.append(status)
        where = " AND ".join(clauses) if clauses else "1=1"
        rows = await db.execute_fetchall(
            f"""SELECT h.*, c.started_at AS conversation_date
              FROM hesitations h
              LEFT JOIN conversations c ON h.conversation_id = c.id
             WHERE {where}
             ORDER BY h.created_at DESC""",
            params,
        )
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_contact_hesitations_summary(contact_id: int) -> dict:
    """Get a summary of all hesitations for a contact across all conversations."""
    all_h = await get_hesitations(contact_id=contact_id)
    open_h = [h for h in all_h if h["status"] == "open"]
    resolved_h = [h for h in all_h if h["status"] == "resolved"]
    carried = [h for h in all_h if h["status"] == "carried_forward"]
    return {
        "total": len(all_h),
        "open": open_h,
        "resolved": resolved_h,
        "carried_forward": carried,
    }
