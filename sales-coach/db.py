"""
SQLite persistence for Sales Coach.

Tables:
  sales_reps     — salespeople using the tool
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
CREATE TABLE IF NOT EXISTS sales_reps (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name            TEXT NOT NULL,
    last_name             TEXT NOT NULL,
    performance_review_json TEXT,
    review_generated_at   TEXT,
    created_at            TEXT DEFAULT (datetime('now')),
    updated_at            TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS contacts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    company    TEXT,
    phone      TEXT,
    notes      TEXT,
    status     TEXT DEFAULT 'prospect',
    sales_rep_id INTEGER REFERENCES sales_reps(id),
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
    "ALTER TABLE contacts ADD COLUMN sales_rep_id INTEGER REFERENCES sales_reps(id)",
    "ALTER TABLE contacts ADD COLUMN phone TEXT",
    "ALTER TABLE contacts ADD COLUMN research TEXT",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_phone_line_from_notes(notes: str | None) -> str:
    """Remove a leading line like 'Phone: ...' or 'Phone: ... Email: ...' from notes."""
    if not notes or not notes.strip():
        return notes or ""
    text = notes.strip()
    if not text.upper().startswith("PHONE:"):
        return notes
    idx = text.find("\n")
    if idx >= 0:
        return text[idx + 1 :].strip()
    return ""


async def _clean_notes_phone_lines(db: aiosqlite.Connection) -> None:
    """Remove leading 'Phone: ...' line from notes for contacts that have phone set."""
    cursor = await db.execute(
        "SELECT id, phone, notes FROM contacts WHERE phone IS NOT NULL AND notes IS NOT NULL AND trim(notes) != ''"
    )
    rows = await cursor.fetchall()
    for row in rows:
        notes = row["notes"] or ""
        if notes.strip().upper().startswith("PHONE:"):
            stripped = _strip_phone_line_from_notes(notes)
            await db.execute(
                "UPDATE contacts SET notes = ?, updated_at = datetime('now') WHERE id = ?",
                (stripped, row["id"]),
            )
    await db.commit()


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
        # One-time: remove redundant "Phone: ..." line from notes when contact has phone
        await _clean_notes_phone_lines(db)

        # Seed default sales rep if none exists
        cursor = await db.execute("SELECT COUNT(*) AS cnt FROM sales_reps")
        row = await cursor.fetchone()
        if row["cnt"] == 0:
            await db.execute(
                "INSERT INTO sales_reps (first_name, last_name) VALUES (?, ?)",
                ("Josiah", "Coad"),
            )
            await db.commit()
            # Assign orphan contacts to the first rep
            await db.execute(
                "UPDATE contacts SET sales_rep_id = 1 WHERE sales_rep_id IS NULL"
            )
            await db.commit()
    finally:
        await db.close()


# ── Contacts ─────────────────────────────────────────────────────


async def create_contact(
    name: str,
    company: str | None = None,
    phone: str | None = None,
    notes: str | None = None,
    status: str = "prospect",
    sales_rep_id: int | None = None,
) -> int:
    db = await get_db()
    try:
        # Default to the first sales rep if none specified
        if sales_rep_id is None:
            cur = await db.execute("SELECT id FROM sales_reps ORDER BY id LIMIT 1")
            row = await cur.fetchone()
            if row:
                sales_rep_id = row["id"]
        cursor = await db.execute(
            "INSERT INTO contacts (name, company, phone, notes, status, sales_rep_id) VALUES (?, ?, ?, ?, ?, ?)",
            (name, company, phone, notes, status, sales_rep_id),
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
    allowed = {"name", "company", "phone", "notes", "research", "status", "sales_rep_id"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return
    if "notes" in fields:
        contact = await get_contact(contact_id)
        if contact and contact.get("phone"):
            fields["notes"] = _strip_phone_line_from_notes(fields.get("notes") or "")
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


async def delete_contact(contact_id: int) -> None:
    """Delete contact and all their conversations and hesitations."""
    db = await get_db()
    try:
        await db.execute("DELETE FROM hesitations WHERE contact_id = ?", (contact_id,))
        await db.execute(
            "DELETE FROM conversations WHERE contact_id = ?", (contact_id,)
        )
        await db.execute("DELETE FROM contacts WHERE id = ?", (contact_id,))
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
            """SELECT id, started_at, ended_at,
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
                   ORDER BY started_at DESC LIMIT 1) AS latest_close_score,
                 (SELECT started_at FROM conversations
                   WHERE contact_id = ct.id
                   ORDER BY started_at DESC LIMIT 1) AS last_conversation_at
               FROM contacts ct
              ORDER BY ct.updated_at DESC"""
        )
        return [dict(r) for r in rows]
    finally:
        await db.close()


# ── Conversations ────────────────────────────────────────────────


async def create_conversation(
    contact_id: int | None = None,
    mode: str = "live",
) -> int:
    db = await get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO conversations (contact_id, mode, started_at) VALUES (?, ?, ?)",
            (contact_id, mode, _now()),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def delete_conversation(conversation_id: int) -> None:
    """Delete a conversation and its hesitations."""
    db = await get_db()
    try:
        await db.execute(
            "DELETE FROM hesitations WHERE conversation_id = ?", (conversation_id,)
        )
        await db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        await db.commit()
    finally:
        await db.close()


async def update_conversation_progress(
    conversation_id: int,
    transcript: list[dict],
    coaching: list[dict],
    hesitations: list[dict],
    contact_id: int | None = None,
) -> None:
    """Save transcript and coaching without ending the conversation (for pause/resume)."""
    db = await get_db()
    try:
        await db.execute(
            """UPDATE conversations
               SET transcript_json = ?, coaching_json = ?
             WHERE id = ? AND ended_at IS NULL""",
            (json.dumps(transcript), json.dumps(coaching), conversation_id),
        )
        await db.commit()
        if hesitations:
            await save_hesitations(conversation_id, hesitations, contact_id)
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
        d.pop("context", None)
        d["transcript"] = json.loads(d.pop("transcript_json") or "[]")
        d["coaching"] = json.loads(d.pop("coaching_json") or "[]")
        d["review"] = json.loads(d.pop("review_json") or "null")
        return d
    finally:
        await db.close()


async def list_conversation_ids_with_reviews() -> list[int]:
    """Return conversation ids that have a non-null review (for migration)."""
    db = await get_db()
    try:
        cursor = await db.execute(
            """SELECT id FROM conversations
               WHERE review_json IS NOT NULL
                 AND trim(review_json) != ''
                 AND review_json != 'null'"""
        )
        rows = await cursor.fetchall()
        return [r["id"] for r in rows]
    finally:
        await db.close()


async def list_conversations(
    limit: int = 50, contact_id: int | None = None
) -> list[dict]:
    db = await get_db()
    try:
        if contact_id:
            rows = await db.execute_fetchall(
                """SELECT c.id, c.contact_id, c.started_at, c.ended_at,
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
                """SELECT c.id, c.contact_id, c.started_at, c.ended_at,
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


async def update_hesitation(
    hesitation_id: int,
    *,
    rank: str | None = None,
    resolution_suggestion: str | None = None,
) -> None:
    """Update rank and/or resolution_suggestion for a hesitation."""
    if rank is not None and rank not in ("S", "M", "L"):
        rank = None
    db = await get_db()
    try:
        if rank is not None and resolution_suggestion is not None:
            await db.execute(
                "UPDATE hesitations SET rank = ?, resolution_suggestion = ? WHERE id = ?",
                (rank, resolution_suggestion, hesitation_id),
            )
        elif rank is not None:
            await db.execute(
                "UPDATE hesitations SET rank = ? WHERE id = ?", (rank, hesitation_id)
            )
        elif resolution_suggestion is not None:
            await db.execute(
                "UPDATE hesitations SET resolution_suggestion = ? WHERE id = ?",
                (resolution_suggestion, hesitation_id),
            )
        await db.commit()
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


# ── Sales Reps ───────────────────────────────────────────────────


async def get_sales_rep(rep_id: int) -> dict | None:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM sales_reps WHERE id = ?", (rep_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def list_sales_reps() -> list[dict]:
    db = await get_db()
    try:
        rows = await db.execute_fetchall("SELECT * FROM sales_reps ORDER BY id")
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_pipeline_breakdown(sales_rep_id: int | None = None) -> list[dict]:
    """Count contacts by status. Optionally filter to a specific rep."""
    db = await get_db()
    try:
        if sales_rep_id:
            rows = await db.execute_fetchall(
                """SELECT status, COUNT(*) AS count
                     FROM contacts
                    WHERE sales_rep_id = ?
                    GROUP BY status
                    ORDER BY status""",
                (sales_rep_id,),
            )
        else:
            rows = await db.execute_fetchall(
                """SELECT status, COUNT(*) AS count
                     FROM contacts
                    GROUP BY status
                    ORDER BY status"""
            )
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_next_steps_for_rep(sales_rep_id: int) -> list[dict]:
    """Contacts with their latest next_step and notes for a sales rep (only those with a next_step)."""
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            """SELECT ct.name, ct.company, ct.notes,
                      (SELECT c.next_step FROM conversations c
                       WHERE c.contact_id = ct.id AND c.next_step IS NOT NULL
                       ORDER BY c.started_at DESC LIMIT 1) AS next_step
                 FROM contacts ct
                WHERE ct.sales_rep_id = ?
                  AND EXISTS (
                    SELECT 1 FROM conversations c
                    WHERE c.contact_id = ct.id AND c.next_step IS NOT NULL
                  )
                ORDER BY ct.updated_at DESC""",
            (sales_rep_id,),
        )
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_all_conversations_for_rep(sales_rep_id: int) -> list[dict]:
    """Fetch all conversations (with transcript + coaching) for a sales rep's contacts."""
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            """SELECT conv.id, conv.transcript_json, conv.coaching_json,
                      conv.started_at, conv.ended_at,
                      conv.final_close_score, conv.review_json,
                      ct.name AS contact_name, ct.company AS contact_company
                 FROM conversations conv
                 JOIN contacts ct ON conv.contact_id = ct.id
                WHERE ct.sales_rep_id = ?
                ORDER BY conv.started_at ASC""",
            (sales_rep_id,),
        )
        result = []
        for r in rows:
            d = dict(r)
            d["transcript"] = json.loads(d.pop("transcript_json") or "[]")
            d["coaching"] = json.loads(d.pop("coaching_json") or "[]")
            d["review"] = json.loads(d.pop("review_json") or "null")
            result.append(d)
        return result
    finally:
        await db.close()


async def save_performance_review(sales_rep_id: int, review_json: str) -> None:
    db = await get_db()
    try:
        await db.execute(
            """UPDATE sales_reps
               SET performance_review_json = ?, review_generated_at = ?, updated_at = datetime('now')
             WHERE id = ?""",
            (review_json, _now(), sales_rep_id),
        )
        await db.commit()
    finally:
        await db.close()


async def get_conversation_start_times(sales_rep_id: int, days: int = 30) -> list[str]:
    """Return started_at (ISO) for each conversation in the last `days` days (UTC window).
    Frontend should group by local date so the chart respects browser timezone."""
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            """SELECT conv.started_at
                 FROM conversations conv
                 JOIN contacts ct ON conv.contact_id = ct.id
                WHERE ct.sales_rep_id = ? AND conv.started_at >= datetime('now', ?)
                ORDER BY conv.started_at""",
            (sales_rep_id, f"-{days} days"),
        )
        return [row["started_at"] or "" for row in rows if row["started_at"]]
    finally:
        await db.close()


async def get_home_data(sales_rep_id: int) -> dict:
    """Aggregate data for the home tab: rep info, pipeline, latest performance review."""
    rep = await get_sales_rep(sales_rep_id)
    if not rep:
        return {"error": "Sales rep not found"}

    pipeline = await get_pipeline_breakdown(sales_rep_id)
    conversation_start_times = await get_conversation_start_times(sales_rep_id, days=30)

    # Total conversation count
    db = await get_db()
    try:
        cursor = await db.execute(
            """SELECT COUNT(*) AS cnt FROM conversations conv
                JOIN contacts ct ON conv.contact_id = ct.id
               WHERE ct.sales_rep_id = ?""",
            (sales_rep_id,),
        )
        row = await cursor.fetchone()
        total_conversations = row["cnt"] if row else 0
    finally:
        await db.close()

    review = json.loads(rep.get("performance_review_json") or "null")

    return {
        "sales_rep": {
            "id": rep["id"],
            "first_name": rep["first_name"],
            "last_name": rep["last_name"],
        },
        "pipeline": pipeline,
        "total_contacts": sum(s["count"] for s in pipeline),
        "total_conversations": total_conversations,
        "conversation_start_times": conversation_start_times,
        "performance_review": review,
        "review_generated_at": rep.get("review_generated_at"),
    }
