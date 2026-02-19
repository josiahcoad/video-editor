"""
SQLite persistence for Sales Coach.

Tables:
  sellers        — people using the tool (sellers)
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
CREATE TABLE IF NOT EXISTS sellers (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name            TEXT NOT NULL,
    last_name             TEXT NOT NULL,
    company_info          TEXT,
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
    email      TEXT,
    notes      TEXT,
    status     TEXT DEFAULT 'prospect',
    seller_id   INTEGER REFERENCES sellers(id),
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
    review_score          INTEGER,
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

CREATE TABLE IF NOT EXISTS todos (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    seller_id   INTEGER NOT NULL REFERENCES sellers(id),
    type        TEXT NOT NULL CHECK(type IN ('email', 'call', 'other')),
    title       TEXT NOT NULL,
    contact_id  INTEGER REFERENCES contacts(id),
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_hesitations_conversation ON hesitations(conversation_id);
CREATE INDEX IF NOT EXISTS idx_hesitations_contact      ON hesitations(contact_id);
CREATE INDEX IF NOT EXISTS idx_hesitations_status        ON hesitations(status);
CREATE INDEX IF NOT EXISTS idx_conversations_contact     ON conversations(contact_id);
CREATE INDEX IF NOT EXISTS idx_todos_seller              ON todos(seller_id);
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
    "ALTER TABLE conversations ADD COLUMN prep_notes TEXT",
    "ALTER TABLE sales_reps RENAME TO sellers",
    "ALTER TABLE contacts RENAME COLUMN sales_rep_id TO seller_id",
    """CREATE TABLE IF NOT EXISTS todos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        seller_id INTEGER NOT NULL REFERENCES sellers(id),
        type TEXT NOT NULL CHECK(type IN ('email', 'call', 'other')),
        title TEXT NOT NULL,
        contact_id INTEGER REFERENCES contacts(id),
        created_at TEXT DEFAULT (datetime('now'))
    )""",
    "CREATE INDEX IF NOT EXISTS idx_todos_seller ON todos(seller_id)",
    "ALTER TABLE contacts ADD COLUMN email TEXT",
    "ALTER TABLE conversations ADD COLUMN review_score INTEGER",
    "ALTER TABLE conversations ADD COLUMN next_step_todo_id INTEGER REFERENCES todos(id)",
    "ALTER TABLE sellers ADD COLUMN company_info TEXT",
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


async def _extract_email_from_notes(db: aiosqlite.Connection) -> None:
    """One-time: move 'Email: ...' from notes into the email column."""
    import re

    cursor = await db.execute(
        """SELECT id, email, notes FROM contacts
            WHERE notes IS NOT NULL AND trim(notes) != ''
              AND (email IS NULL OR trim(email) = '')"""
    )
    rows = await cursor.fetchall()
    email_re = re.compile(r"(?i)^email:\s*(\S+@\S+)", re.MULTILINE)
    for row in rows:
        notes = row["notes"] or ""
        m = email_re.search(notes)
        if m:
            email = m.group(1).strip().rstrip(",;")
            # Remove the "Email: ..." line from notes
            cleaned = email_re.sub("", notes, count=1).strip()
            await db.execute(
                "UPDATE contacts SET email = ?, notes = ?, updated_at = datetime('now') WHERE id = ?",
                (email, cleaned or None, row["id"]),
            )
    await db.commit()


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
        # One-time: extract "Email: ..." from notes into the email column
        await _extract_email_from_notes(db)

        # Seed default seller if none exists
        cursor = await db.execute("SELECT COUNT(*) AS cnt FROM sellers")
        row = await cursor.fetchone()
        if row["cnt"] == 0:
            await db.execute(
                "INSERT INTO sellers (first_name, last_name) VALUES (?, ?)",
                ("Josiah", "Coad"),
            )
            await db.commit()
            # Assign orphan contacts to the first seller
            await db.execute(
                "UPDATE contacts SET seller_id = 1 WHERE seller_id IS NULL"
            )
            await db.commit()
        # One-time: set company_info for the first seller if still null
        await _backfill_first_seller_company_info(db)
    finally:
        await db.close()


DEFAULT_COMPANY_INFO = """Marky is a social media concierge service that specializes in thought leadership content to establish trust. We handle premier end-to-end social media management (recording, editing, publishing and community engagement). We are new. We don't have any clients yet but we're hungry and we're going to totally kill it. We will beat any price out there. We spent three years building out proprietary AI technology to help you establish yourself as an expert in the field and optimize your efforts."""


async def _backfill_first_seller_company_info(db: aiosqlite.Connection) -> None:
    """Set company_info for the first seller if column exists and value is null."""
    await db.execute(
        """UPDATE sellers SET company_info = ?
           WHERE id = (SELECT id FROM sellers ORDER BY id LIMIT 1)
             AND (company_info IS NULL OR trim(company_info) = '')""",
        (DEFAULT_COMPANY_INFO,),
    )
    await db.commit()


# ── Contacts ─────────────────────────────────────────────────────


async def create_contact(
    name: str,
    company: str | None = None,
    phone: str | None = None,
    email: str | None = None,
    notes: str | None = None,
    status: str = "prospect",
    seller_id: int | None = None,
) -> int:
    db = await get_db()
    try:
        # Default to the first seller if none specified
        if seller_id is None:
            cur = await db.execute("SELECT id FROM sellers ORDER BY id LIMIT 1")
            row = await cur.fetchone()
            if row:
                seller_id = row["id"]
        cursor = await db.execute(
            "INSERT INTO contacts (name, company, phone, email, notes, status, seller_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, company, phone, email, notes, status, seller_id),
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
    allowed = {
        "name",
        "company",
        "phone",
        "email",
        "notes",
        "research",
        "status",
        "seller_id",
    }
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
    prep_notes: str | None = None,
) -> int:
    db = await get_db()
    try:
        cursor = await db.execute(
            """INSERT INTO conversations (contact_id, mode, started_at, prep_notes)
               VALUES (?, ?, ?, ?)""",
            (contact_id, mode, _now(), prep_notes),
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


async def update_conversation_prep_notes(
    conversation_id: int, prep_notes: str | None
) -> None:
    """Update prep_notes for an existing conversation."""
    db = await get_db()
    try:
        await db.execute(
            "UPDATE conversations SET prep_notes = ? WHERE id = ?",
            (prep_notes or None, conversation_id),
        )
        await db.commit()
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
    """Save review, extract next_step into its own column, and link to a todo when present."""
    next_step = (review.get("next_step") or "").strip()
    review_score = review.get("score")
    if isinstance(review_score, str):
        try:
            review_score = int(review_score)
        except (TypeError, ValueError):
            review_score = None
    elif not isinstance(review_score, int):
        review_score = None

    conv = await get_conversation_row(conversation_id)
    next_step_todo_id = conv.get("next_step_todo_id") if conv else None
    if next_step:
        if next_step_todo_id:
            await update_todo_title(next_step_todo_id, next_step)
        elif conv and conv.get("contact_id"):
            contact = await get_contact(conv["contact_id"])
            if contact and contact.get("seller_id"):
                next_step_todo_id = await create_todo(
                    contact["seller_id"],
                    "other",
                    next_step,
                    contact_id=conv["contact_id"],
                )

    db = await get_db()
    try:
        await db.execute(
            """UPDATE conversations
               SET review_json = ?, review_generated_at = ?, next_step = ?,
                   next_step_todo_id = ?, review_score = ?
             WHERE id = ?""",
            (
                json.dumps(review),
                _now(),
                next_step or None,
                next_step_todo_id,
                review_score,
                conversation_id,
            ),
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


async def get_conversation_row(conversation_id: int) -> dict | None:
    """Lightweight fetch: id, contact_id, next_step_todo_id (for save_review)."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, contact_id, next_step_todo_id FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
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
                          c.next_step, c.prep_notes, c.review_generated_at,
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
                          c.next_step, c.prep_notes, c.review_generated_at,
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


# ── Sellers ───────────────────────────────────────────────────────


async def get_seller(seller_id: int) -> dict | None:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM sellers WHERE id = ?", (seller_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def list_sellers() -> list[dict]:
    db = await get_db()
    try:
        rows = await db.execute_fetchall("SELECT * FROM sellers ORDER BY id")
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_pipeline_breakdown(seller_id: int | None = None) -> list[dict]:
    """Count contacts by status. Optionally filter to a specific seller."""
    db = await get_db()
    try:
        if seller_id:
            rows = await db.execute_fetchall(
                """SELECT status, COUNT(*) AS count
                     FROM contacts
                    WHERE seller_id = ?
                    GROUP BY status
                    ORDER BY status""",
                (seller_id,),
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


async def get_next_steps_for_seller(seller_id: int) -> list[dict]:
    """Contacts with their latest next_step and notes for a seller (only those with a next_step)."""
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            """SELECT ct.id, ct.name, ct.company, ct.notes,
                      (SELECT c.next_step FROM conversations c
                       WHERE c.contact_id = ct.id AND c.next_step IS NOT NULL
                       ORDER BY c.started_at DESC LIMIT 1) AS next_step
                 FROM contacts ct
                WHERE ct.seller_id = ?
                  AND EXISTS (
                    SELECT 1 FROM conversations c
                    WHERE c.contact_id = ct.id AND c.next_step IS NOT NULL
                  )
                ORDER BY ct.updated_at DESC""",
            (seller_id,),
        )
        return [dict(r) for r in rows]
    finally:
        await db.close()


# ── Todos ─────────────────────────────────────────────────────────


async def list_todos(seller_id: int) -> list[dict]:
    """List todos for a seller, newest first. Includes contact_name if linked."""
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            """SELECT t.id, t.seller_id, t.type, t.title, t.contact_id, t.created_at,
                      ct.name AS contact_name
               FROM todos t
               LEFT JOIN contacts ct ON t.contact_id = ct.id
              WHERE t.seller_id = ?
              ORDER BY t.created_at DESC""",
            (seller_id,),
        )
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def create_todo(
    seller_id: int,
    type: str,
    title: str,
    contact_id: int | None = None,
) -> int:
    """Create a todo. type must be 'email', 'call', or 'other'. Returns new id."""
    if type not in ("email", "call", "other"):
        raise ValueError("type must be 'email', 'call', or 'other'")
    db = await get_db()
    try:
        cursor = await db.execute(
            """INSERT INTO todos (seller_id, type, title, contact_id)
               VALUES (?, ?, ?, ?)""",
            (seller_id, type, title.strip(), contact_id),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def update_todo_title(todo_id: int, title: str) -> bool:
    """Update a todo's title. Returns True if updated."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "UPDATE todos SET title = ? WHERE id = ?", (title.strip(), todo_id)
        )
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()


async def delete_todo(todo_id: int) -> bool:
    """Delete a todo. Returns True if deleted."""
    db = await get_db()
    try:
        cursor = await db.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()


async def get_all_conversations_for_seller(seller_id: int) -> list[dict]:
    """Fetch all conversations (with transcript + coaching) for a seller's contacts."""
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            """SELECT conv.id, conv.transcript_json, conv.coaching_json,
                      conv.started_at, conv.ended_at,
                      conv.final_close_score, conv.review_json,
                      ct.name AS contact_name, ct.company AS contact_company
                 FROM conversations conv
                 JOIN contacts ct ON conv.contact_id = ct.id
                WHERE ct.seller_id = ?
                ORDER BY conv.started_at ASC""",
            (seller_id,),
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


async def save_performance_review(seller_id: int, review_json: str) -> None:
    db = await get_db()
    try:
        await db.execute(
            """UPDATE sellers
               SET performance_review_json = ?, review_generated_at = ?, updated_at = datetime('now')
             WHERE id = ?""",
            (review_json, _now(), seller_id),
        )
        await db.commit()
    finally:
        await db.close()


async def get_conversation_start_times(seller_id: int, days: int = 30) -> list[str]:
    """Return started_at (ISO) for each conversation in the last `days` days (UTC window).
    Frontend should group by local date so the chart respects browser timezone."""
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            """SELECT conv.started_at
                 FROM conversations conv
                 JOIN contacts ct ON conv.contact_id = ct.id
                WHERE ct.seller_id = ? AND conv.started_at >= datetime('now', ?)
                ORDER BY conv.started_at""",
            (seller_id, f"-{days} days"),
        )
        return [row["started_at"] or "" for row in rows if row["started_at"]]
    finally:
        await db.close()


async def get_home_data(seller_id: int) -> dict:
    """Aggregate data for the home tab: seller info, pipeline, latest performance review."""
    seller = await get_seller(seller_id)
    if not seller:
        return {"error": "Seller not found"}

    pipeline = await get_pipeline_breakdown(seller_id)
    conversation_start_times = await get_conversation_start_times(seller_id, days=30)

    # Total conversation count
    db = await get_db()
    try:
        cursor = await db.execute(
            """SELECT COUNT(*) AS cnt FROM conversations conv
                JOIN contacts ct ON conv.contact_id = ct.id
               WHERE ct.seller_id = ?""",
            (seller_id,),
        )
        row = await cursor.fetchone()
        total_conversations = row["cnt"] if row else 0
    finally:
        await db.close()

    review = json.loads(seller.get("performance_review_json") or "null")

    return {
        "seller": {
            "id": seller["id"],
            "first_name": seller["first_name"],
            "last_name": seller["last_name"],
            "company_info": seller.get("company_info"),
        },
        "pipeline": pipeline,
        "total_contacts": sum(s["count"] for s in pipeline),
        "total_conversations": total_conversations,
        "conversation_start_times": conversation_start_times,
        "performance_review": review,
        "review_generated_at": seller.get("review_generated_at"),
    }
