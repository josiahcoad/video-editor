"""
Sales Coach — FastAPI backend for real-time sales coaching.

Architecture:
  Browser (mic via getUserMedia)
    → WebSocket (binary PCM audio)
    → FastAPI
    → Deepgram Flux (transcription + end-of-turn)
    → OpenRouter / Claude Haiku 4.5 (coaching)
    → WebSocket (JSON messages)
    → Browser UI

  Persistence:
    SQLite via aiosqlite (contacts, conversations, hesitations)

  Test mode:
    Upload a conversation.txt file → replay turns, coaching on demand

Usage:
    dotenvx run -f .env -- uv run python sales-coach/app.py
    Open http://localhost:5050
"""

import asyncio
import json
import logging
import os
import pathlib
import re
from contextlib import asynccontextmanager
from typing import Any

import openai
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType

from db import (
    CONTACT_STATUSES,
    init_db,
    create_contact,
    list_contacts,
    list_contacts_with_summary,
    get_contact,
    get_contact_detail,
    update_contact,
    create_conversation,
    end_conversation,
    save_review,
    get_conversation,
    list_conversations,
    save_hesitations,
    get_hesitations,
    get_contact_hesitations_summary,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sales-coach")

BASE_DIR = pathlib.Path(__file__).parent
SAMPLE_RATE = 16000


@asynccontextmanager
async def lifespan(application: FastAPI):
    await init_db()
    logger.info("Database initialized")
    yield


app = FastAPI(title="Sales Coach", lifespan=lifespan)

SYSTEM_PROMPT = """\
You are a sharp, experienced sales coach listening to a live call in real time.
Your job is to help the sales rep move toward closing — not by being pushy, but by \
uncovering and resolving objections, building urgency, and advancing commitment.

RESPOND IN VALID JSON with this exact structure:
{
  "advice": "1-2 sentences. Specific to what was just said. What should the rep say next.",
  "close_score": 0-100,
  "steps_to_close": 1-10,
  "objections": [
    {"text": "short description of objection", "status": "open", "rank": "S|M|L", "resolution_suggestion": "one short sentence: what rep could say or do to resolve this"},
    {"text": "previously identified objection now addressed", "status": "resolved", "rank": "", "resolution_suggestion": ""}
  ]
}

RANK for each objection: S = minor, not deal-breaking | M = currently blocking but overcomeable | L = hard stop, can't or very unlikely to move forward.

RULES FOR ADVICE:
- MAX 2 sentences. They need to read this instantly while on the call.
- Reference actual words from the conversation — never be generic.
- Always think: "What is standing between us and a signed deal right now?"
- If the prospect is in rapport/storytelling mode, don't interrupt — coach silence or a deepening question.
- If an objection surfaces (stated or implied), flag it and suggest how to probe it.
- If a stall appears ("let me think about it", "I need to check with X", "let's revisit Friday"), \
treat it as an unresolved objection. Coach the rep to probe: what specifically needs to happen?
- If buying signals appear, suggest a trial close or commitment step.
- If all objections are resolved and the prospect is warm, suggest asking for the close.

RULES FOR SCORES:
- close_score: 0 = ice cold, 100 = ready to sign right now. Base this on: \
verbal commitment level, objections resolved vs open, urgency, and buying signals.
- steps_to_close: estimated remaining steps (1 = just needs signature, 10 = long way to go). \
Steps include: uncover needs, present solution, handle objections, get verbal commitment, \
discuss terms, sign contract.

RULES FOR OBJECTIONS:
- Track ALL objections surfaced during the call (explicit or implied).
- For each objection set "rank": S (minor, not deal-breaking), M (currently blocking but overcomeable), or L (hard stop, can't or very unlikely to move forward).
- For each open objection include "resolution_suggestion": one short, specific suggestion for \
what the rep could say or do to resolve it (e.g. "Ask: What would need to be true for you to move forward?").
- Include previously identified objections with updated status. Resolved ones can have empty resolution_suggestion.
- Mark "resolved" only when the prospect has explicitly acknowledged the concern is addressed.
- Common hidden objections: price sensitivity, risk aversion, need for approval from others, \
timing ("not right now"), trust/proof ("show me examples"), scope uncertainty.
"""

# Actor: suggest multiple candidate next steps (for critique mode)
ACTOR_CANDIDATES_PROMPT = """\
You are a sharp sales coach. Given the conversation so far, suggest 3–5 DISTINCT potential \
next steps the rep could take. Include variety: e.g. one that pushes toward close, one that \
probes a specific hesitation, one that builds rapport or deepens the relationship. The path \
to close is not monotonic — sometimes stepping back or addressing trust matters more than \
pushing for commitment.

RESPOND IN VALID JSON only:
{
  "candidates": [
    { "action": "1-2 sentences of what to say or do next.", "one_liner": "Short label e.g. Probe price concern" },
    ...
  ],
  "objections": [
    {"text": "short description", "status": "open", "rank": "S|M|L", "resolution_suggestion": "one short sentence: what rep could say/do to resolve"},
    {"text": "resolved one", "status": "resolved", "rank": "", "resolution_suggestion": ""}
  ]
}
Rank: S = minor | M = blocking but overcomeable | L = hard stop.
"""

# Critic: score each candidate on probability of eventually closing
CRITIC_PROMPT = """\
You are a sales coach evaluating potential next steps. Given the current state of the call \
and a list of candidate actions, score each action on how likely it is to lead to a close \
(eventually). Consider: trajectory (if rep does this, how might the prospect respond? what \
happens next?), relationship risk (e.g. pushing too hard can lose the client), and whether \
this step clears a blocking hesitation or advances commitment.

The path to close is NOT always "push harder now" — sometimes probing an objection, \
building trust, or locking a small commitment is the best move. Score 0–100.

RESPOND IN VALID JSON only:
{
  "evaluations": [
    { "index": 0, "close_probability": 0-100, "trajectory_notes": "1-2 sentences: what likely happens if rep does this" },
    ...
  ]
}
"""

REVIEW_PROMPT = """\
You are a senior sales coach reviewing a completed sales call. Analyze the full \
conversation and coaching that was provided, then give an honest, specific post-call review.

RESPOND IN VALID JSON:
{
  "score": 1-10,
  "went_well": ["specific thing 1", "specific thing 2", ...],
  "improve": ["specific actionable improvement 1", ...],
  "hesitations_summary": {
    "uncovered": ["hesitation that was identified"],
    "handled": ["hesitation that was resolved during the call"],
    "remaining": ["hesitation that is still open / unresolved"]
  },
  "follow_ups": ["specific follow-up action 1", "specific follow-up action 2", ...],
  "key_moment": "The single most important moment in the call and what the rep should learn from it.",
  "next_call_prep": "If there's a follow-up call, what should the rep prepare for?",
  "next_step": "The single most important next action the rep should take. Short, specific, with a deadline if possible. e.g. 'Send contract by Wednesday' or 'Call back after lawyer review on Friday'",
  "suggested_status": "prospect|interested|demoing|reviewing_contract|closed_won|closed_lost"
}

RULES:
- Be brutally honest but constructive.
- Reference specific turns and quotes from the conversation.
- "went_well" should cite real moments, not generic praise.
- "improve" should be actionable — what to say differently, when to push harder, when to listen.
- For hesitations: distinguish between ones the rep surfaced and handled vs ones that \
slipped through. Stalls ("let me check with my lawyer", "let's revisit Friday") count as \
unresolved hesitations unless the rep probed the underlying concern.
- follow_ups should be concrete next actions with suggested timelines.
- next_step should be a SHORT one-liner (max 15 words) — the single most critical action item.
- suggested_status should reflect where this contact is in the pipeline based on the call outcome.
"""


def _get_llm() -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )


def parse_coaching_response(raw: str) -> dict:
    """Extract JSON from the LLM response, handling markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "advice": raw.strip(),
            "close_score": -1,
            "steps_to_close": -1,
            "objections": [],
        }


def _strip_json_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


def parse_conversation_file(content: str) -> list[dict]:
    """Parse a conversation.txt file into a list of turns."""
    turns = []
    blocks = re.split(r"\n(?=TURN \d+)", content.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        header = lines[0]
        match = re.match(r"TURN (\d+)", header)
        if not match:
            continue
        turn_num = int(match.group(1))
        confidence_str = lines[1].strip().rstrip("%")
        try:
            confidence = int(confidence_str) / 100
        except ValueError:
            confidence = 0.0
        text = "\n".join(lines[2:]).strip()
        turns.append({"turn": turn_num, "confidence": confidence, "text": text})
    return turns


# ── Startup ──────────────────────────────────────────────────────

# ── Pages ────────────────────────────────────────────────────────


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


# ── REST API: Contacts ───────────────────────────────────────────


@app.get("/api/contacts")
async def api_list_contacts(summary: bool = False):
    if summary:
        return await list_contacts_with_summary()
    return await list_contacts()


@app.get("/api/contacts/statuses")
async def api_contact_statuses():
    return CONTACT_STATUSES


@app.get("/api/contacts/{contact_id}/detail")
async def api_contact_detail(contact_id: int):
    detail = await get_contact_detail(contact_id)
    if not detail:
        return {"error": "not found"}
    return detail


@app.post("/api/contacts")
async def api_create_contact(body: dict):
    cid = await create_contact(
        name=body["name"],
        company=body.get("company"),
        notes=body.get("notes"),
        status=body.get("status", "prospect"),
    )
    return {"id": cid}


@app.put("/api/contacts/{contact_id}")
async def api_update_contact(contact_id: int, body: dict):
    await update_contact(contact_id, **body)
    return {"ok": True}


# ── REST API: Conversations ──────────────────────────────────────


@app.get("/api/conversations")
async def api_list_conversations(contact_id: int | None = None, limit: int = 50):
    return await list_conversations(limit=limit, contact_id=contact_id)


@app.get("/api/conversations/{conversation_id}")
async def api_get_conversation(conversation_id: int):
    conv = await get_conversation(conversation_id)
    if not conv:
        return {"error": "not found"}
    conv["hesitations"] = await get_hesitations(conversation_id=conversation_id)
    return conv


@app.post("/api/conversations/{conversation_id}/review")
async def api_generate_review(conversation_id: int):
    """Generate (or regenerate) a post-call review."""
    conv = await get_conversation(conversation_id)
    if not conv:
        return {"error": "not found"}
    review = await generate_review(
        transcript=conv["transcript"],
        coaching=conv["coaching"],
        context=conv.get("context", ""),
    )
    await save_review(conversation_id, review)
    return review


# ── REST API: Hesitations ────────────────────────────────────────


@app.get("/api/hesitations")
async def api_list_hesitations(
    conversation_id: int | None = None,
    contact_id: int | None = None,
    status: str | None = None,
):
    return await get_hesitations(
        conversation_id=conversation_id,
        contact_id=contact_id,
        status=status,
    )


@app.get("/api/contacts/{contact_id}/hesitations")
async def api_contact_hesitations(contact_id: int):
    return await get_contact_hesitations_summary(contact_id)


# ── Coaching LLM call ────────────────────────────────────────────


async def get_coaching(
    llm: openai.AsyncOpenAI,
    conversation: list[str],
    objections: list[dict],
    call_context: str,
    turn_num: int,
    custom_question: str | None = None,
) -> dict:
    """Call the LLM for coaching advice. Returns parsed coaching dict."""
    recent = conversation[-15:]
    conv_text = "\n".join(f"Turn {i + 1}: {t}" for i, t in enumerate(recent))

    user_content = ""
    if call_context:
        user_content += f"PRE-CALL CONTEXT:\n{call_context}\n\n"

    if objections:
        obj_summary = "\n".join(
            f"- [{o['status'].upper()}] {o['text']}" for o in objections
        )
        user_content += f"HESITATIONS TRACKED SO FAR:\n{obj_summary}\n\n"

    user_content += f"TRANSCRIPT (most recent turns):\n{conv_text}\n\n"

    if custom_question:
        user_content += (
            f"THE SALES REP IS ASKING YOU SPECIFICALLY:\n{custom_question}\n\n"
            f"Answer their question with context from the conversation, "
            f"and still provide your standard coaching JSON."
        )
    else:
        user_content += "What should the sales rep say next?"

    response = await llm.chat.completions.create(
        model="anthropic/claude-haiku-4-5",
        max_tokens=300,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    raw = response.choices[0].message.content.strip()
    return parse_coaching_response(raw)


def _coaching_user_content(
    conversation: list[str],
    objections: list[dict],
    call_context: str,
    custom_question: str | None,
) -> str:
    """Build user message for both single and actor/critic coaching."""
    recent = conversation[-15:]
    conv_text = "\n".join(f"Turn {i + 1}: {t}" for i, t in enumerate(recent))
    out = ""
    if call_context:
        out += f"PRE-CALL CONTEXT:\n{call_context}\n\n"
    if objections:
        obj_summary = "\n".join(
            f"- [{o['status'].upper()}] {o['text']}" for o in objections
        )
        out += f"HESITATIONS TRACKED SO FAR:\n{obj_summary}\n\n"
    out += f"TRANSCRIPT (most recent turns):\n{conv_text}\n\n"
    if custom_question:
        out += f"THE SALES REP IS ASKING:\n{custom_question}\n\n"
    return out


async def get_coaching_with_critique(
    llm: openai.AsyncOpenAI,
    conversation: list[str],
    objections: list[dict],
    call_context: str,
    turn_num: int,
    custom_question: str | None = None,
) -> dict:
    """Actor: suggest 3–5 candidate next steps. Critic: score each for P(close). Return best + full list."""
    user_content = _coaching_user_content(
        conversation, objections, call_context, custom_question
    )
    user_content += "Suggest 3–5 distinct candidate next steps (JSON with candidates and objections)."

    # Actor: get candidates
    response = await llm.chat.completions.create(
        model="anthropic/claude-haiku-4-5",
        max_tokens=500,
        messages=[
            {"role": "system", "content": ACTOR_CANDIDATES_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    raw_actor = response.choices[0].message.content.strip()
    try:
        actor_data = json.loads(_strip_json_fences(raw_actor))
    except json.JSONDecodeError:
        return {
            "advice": raw_actor[:200],
            "close_score": -1,
            "steps_to_close": -1,
            "objections": [],
        }
    candidates = actor_data.get("candidates") or []
    new_objections = actor_data.get("objections") or objections
    if not candidates:
        return {
            "advice": "No candidates generated.",
            "close_score": -1,
            "steps_to_close": -1,
            "objections": new_objections,
        }

    # Critic: score each candidate
    state_summary = user_content[:2000]
    list_actions = "\n".join(
        f"{i}. {c.get('action', c.get('one_liner', ''))}"
        for i, c in enumerate(candidates)
    )
    critic_user = f"CURRENT STATE:\n{state_summary}\n\nCANDIDATE ACTIONS:\n{list_actions}\n\nScore each action 0–100 for probability of eventually closing (JSON evaluations)."
    response = await llm.chat.completions.create(
        model="anthropic/claude-haiku-4-5",
        max_tokens=600,
        messages=[
            {"role": "system", "content": CRITIC_PROMPT},
            {"role": "user", "content": critic_user},
        ],
    )
    raw_critic = response.choices[0].message.content.strip()
    try:
        critic_data = json.loads(_strip_json_fences(raw_critic))
    except json.JSONDecodeError:
        best = candidates[0]
        return {
            "advice": best.get("action", ""),
            "close_score": -1,
            "steps_to_close": -1,
            "objections": new_objections,
            "candidates": [
                {
                    "action": c.get("action", ""),
                    "close_probability": -1,
                    "trajectory_notes": "",
                    "one_liner": c.get("one_liner", ""),
                }
                for c in candidates
            ],
        }
    evaluations = critic_data.get("evaluations") or []
    by_index = {
        e["index"]: e for e in evaluations if "index" in e and "close_probability" in e
    }
    scored = []
    for i, c in enumerate(candidates):
        e = by_index.get(i, {})
        prob = e.get("close_probability", -1)
        if not isinstance(prob, (int, float)):
            prob = -1
        scored.append(
            {
                "action": c.get("action", ""),
                "one_liner": c.get("one_liner", ""),
                "close_probability": int(prob) if prob >= 0 else -1,
                "trajectory_notes": e.get("trajectory_notes", "") or "",
            }
        )
    scored.sort(key=lambda x: x["close_probability"], reverse=True)
    best = (
        scored[0]
        if scored
        else {"action": candidates[0].get("action", ""), "close_probability": -1}
    )
    return {
        "advice": best["action"],
        "close_score": best["close_probability"]
        if best["close_probability"] >= 0
        else -1,
        "steps_to_close": -1,
        "objections": new_objections,
        "candidates": scored,
    }


# ── Post-call review generation ──────────────────────────────────


async def generate_review(
    transcript: list[dict],
    coaching: list[dict],
    context: str = "",
) -> dict:
    llm = _get_llm()

    conv_text = "\n".join(
        f"Turn {t.get('turn', i+1)}: {t.get('text', '')}"
        for i, t in enumerate(transcript)
    )
    coach_text = "\n".join(
        f"Coaching #{c.get('number', i+1)} (turn {c.get('turn', '?')}): {c.get('advice', '')}"
        for i, c in enumerate(coaching)
    )

    user_content = ""
    if context:
        user_content += f"PRE-CALL CONTEXT:\n{context}\n\n"
    user_content += f"FULL TRANSCRIPT:\n{conv_text}\n\n"
    if coach_text:
        user_content += f"COACHING PROVIDED DURING CALL:\n{coach_text}\n\n"
    user_content += "Provide your post-call review."

    response = await llm.chat.completions.create(
        model="anthropic/claude-haiku-4-5",
        max_tokens=800,
        messages=[
            {"role": "system", "content": REVIEW_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    raw = response.choices[0].message.content.strip()
    return parse_coaching_response(raw)


# ── Session data collector ───────────────────────────────────────


class SessionData:
    """Accumulates data during a WebSocket session for DB persistence."""

    def __init__(self):
        self.conversation_id: int | None = None
        self.contact_id: int | None = None
        self.turns: list[dict] = []
        self.coaching_log: list[dict] = []
        self.objections: list[dict] = []
        self.last_close_score: int | None = None
        self.last_steps_to_close: int | None = None

    def add_turn(self, turn_num: int, text: str, confidence: float):
        self.turns.append({"turn": turn_num, "text": text, "confidence": confidence})

    def add_coaching(
        self,
        number: int,
        turn: int,
        advice: str,
        close_score: int,
        steps_to_close: int,
        is_custom: bool,
    ):
        self.coaching_log.append(
            {
                "number": number,
                "turn": turn,
                "advice": advice,
                "close_score": close_score,
                "steps_to_close": steps_to_close,
                "is_custom": is_custom,
            }
        )
        if close_score >= 0:
            self.last_close_score = close_score
        if steps_to_close >= 0:
            self.last_steps_to_close = steps_to_close

    async def save(self) -> int | None:
        if not self.conversation_id or not self.turns:
            return None
        await end_conversation(
            conversation_id=self.conversation_id,
            transcript=self.turns,
            coaching=self.coaching_log,
            final_close_score=self.last_close_score,
            final_steps_to_close=self.last_steps_to_close,
        )
        if self.objections:
            await save_hesitations(
                self.conversation_id, self.objections, self.contact_id
            )
        return self.conversation_id


# ── Live coaching WebSocket ──────────────────────────────────────


@app.websocket("/ws/coach")
async def coaching_session(ws: WebSocket):
    await ws.accept()

    llm = _get_llm()
    session = SessionData()

    # First message: config
    try:
        init_msg = await ws.receive_json()
        call_context = init_msg.get("context", "").strip()
        auto_coach = init_msg.get("auto_coach", True)
        critique_mode = init_msg.get("critique_mode", False)
        contact_id = init_msg.get("contact_id")
        if contact_id:
            session.contact_id = int(contact_id)
    except Exception:
        call_context = ""
        auto_coach = True
        critique_mode = False

    if call_context:
        logger.info("Call context: %s", call_context[:120])

    # Create conversation record
    session.conversation_id = await create_conversation(
        contact_id=session.contact_id,
        context=call_context,
        mode="live",
    )
    await ws.send_json({"type": "session_id", "id": session.conversation_id})

    conversation: list[str] = []
    objections: list[dict] = []
    coaching_count = 0
    msg_queue: asyncio.Queue[Any] = asyncio.Queue()
    cmd_queue: asyncio.Queue[dict] = asyncio.Queue()
    running = True

    def on_dg_message(message: Any) -> None:
        if running:
            msg_queue.put_nowait(message)

    async def do_coaching(turn_num: int, custom_question: str | None = None) -> None:
        nonlocal coaching_count, objections
        try:
            if critique_mode:
                result = await get_coaching_with_critique(
                    llm,
                    conversation,
                    objections,
                    call_context,
                    turn_num,
                    custom_question,
                )
            else:
                result = await get_coaching(
                    llm,
                    conversation,
                    objections,
                    call_context,
                    turn_num,
                    custom_question,
                )
            new_objs = result.get("objections", [])
            if new_objs:
                objections = new_objs
                session.objections = new_objs

            coaching_count += 1
            advice = result.get("advice", "")
            cs = result.get("close_score", -1)
            sts = result.get("steps_to_close", -1)

            session.add_coaching(
                coaching_count, turn_num, advice, cs, sts, custom_question is not None
            )

            payload = {
                "type": "coaching",
                "advice": advice,
                "close_score": cs,
                "steps_to_close": sts,
                "objections": objections,
                "number": coaching_count,
                "turn": turn_num,
                "is_custom": custom_question is not None,
            }
            if result.get("candidates"):
                payload["candidates"] = result["candidates"]
            await ws.send_json(payload)
        except Exception as e:
            await ws.send_json({"type": "error", "message": str(e)})

    async def process_commands() -> None:
        while running:
            try:
                cmd = await asyncio.wait_for(cmd_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            cmd_type = cmd.get("type")
            if cmd_type == "coach_me":
                turn_num = len(conversation)
                asyncio.create_task(do_coaching(turn_num, cmd.get("question")))
            elif cmd_type == "set_auto_coach":
                nonlocal auto_coach
                auto_coach = cmd.get("value", True)
            elif cmd_type == "set_critique_mode":
                nonlocal critique_mode
                critique_mode = cmd.get("value", False)

    async def process_messages() -> None:
        while running:
            try:
                message = await asyncio.wait_for(msg_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            msg_type = getattr(message, "type", None)
            event = getattr(message, "event", None)
            transcript = getattr(message, "transcript", None)

            try:
                if msg_type == "TurnInfo":
                    if event == "Update" and transcript:
                        await ws.send_json(
                            {"type": "transcript_interim", "text": transcript}
                        )
                    elif event == "EndOfTurn" and transcript and transcript.strip():
                        text = transcript.strip()
                        conversation.append(text)
                        turn_num = len(conversation)
                        confidence = getattr(message, "end_of_turn_confidence", 0)
                        session.add_turn(turn_num, text, round(confidence, 2))
                        await ws.send_json(
                            {
                                "type": "turn_end",
                                "text": text,
                                "turn": turn_num,
                                "confidence": round(confidence, 2),
                            }
                        )
                        if auto_coach:
                            asyncio.create_task(do_coaching(turn_num))

                elif msg_type == "Connected":
                    await ws.send_json(
                        {"type": "connected", "message": "Deepgram Flux ready"}
                    )
            except Exception:
                break

    # Main session
    dg_client = AsyncDeepgramClient()

    try:
        async with dg_client.listen.v2.connect(
            model="flux-general-en",
            encoding="linear16",
            sample_rate=str(SAMPLE_RATE),
            eot_timeout_ms="3000",
        ) as dg_conn:
            dg_conn.on(EventType.MESSAGE, on_dg_message)
            dg_conn.on(
                EventType.ERROR,
                lambda e: msg_queue.put_nowait(("error", str(e))),
            )

            listen_task = asyncio.create_task(dg_conn.start_listening())
            process_task = asyncio.create_task(process_messages())
            cmd_task = asyncio.create_task(process_commands())

            await ws.send_json({"type": "status", "message": "Ready — start talking!"})

            try:
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.receive":
                        if "bytes" in msg and msg["bytes"]:
                            await dg_conn.send_media(msg["bytes"])
                        elif "text" in msg and msg["text"]:
                            try:
                                cmd = json.loads(msg["text"])
                                cmd_queue.put_nowait(cmd)
                            except json.JSONDecodeError:
                                pass
            except WebSocketDisconnect:
                pass
            finally:
                running = False
                listen_task.cancel()
                process_task.cancel()
                cmd_task.cancel()

    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass

    # ── Session end: persist + generate review ──
    cid = await session.save()
    if cid and session.turns:
        logger.info("Conversation %d saved (%d turns)", cid, len(session.turns))
        try:
            review = await generate_review(
                transcript=session.turns,
                coaching=session.coaching_log,
                context=call_context,
            )
            await save_review(cid, review)
            logger.info("Review generated for conversation %d", cid)
            # Try to send review to browser (may already be disconnected)
            try:
                await ws.send_json(
                    {"type": "review", "conversation_id": cid, "review": review}
                )
            except Exception:
                pass
        except Exception as e:
            logger.error("Failed to generate review: %s", e)


# ── Test mode WebSocket ──────────────────────────────────────────


@app.websocket("/ws/test")
async def test_session(ws: WebSocket):
    """Replay a conversation file turn-by-turn for testing coaching prompts."""
    await ws.accept()

    llm = _get_llm()
    session = SessionData()

    # First message: config
    try:
        init_msg = await ws.receive_json()
        call_context = init_msg.get("context", "").strip()
        conversation_text = init_msg.get("conversation", "")
        contact_id = init_msg.get("contact_id")
        critique_mode = init_msg.get("critique_mode", False)
        if contact_id:
            session.contact_id = int(contact_id)
    except Exception:
        await ws.send_json(
            {"type": "error", "message": "Expected init message with conversation"}
        )
        return

    turns_data = parse_conversation_file(conversation_text)
    if not turns_data:
        await ws.send_json(
            {"type": "error", "message": "Could not parse conversation file"}
        )
        return

    # Create conversation record
    session.conversation_id = await create_conversation(
        contact_id=session.contact_id,
        context=call_context,
        mode="test",
    )
    for td in turns_data:
        session.add_turn(td["turn"], td["text"], td["confidence"])

    await ws.send_json(
        {"type": "status", "message": f"Test mode: {len(turns_data)} turns loaded"}
    )
    await ws.send_json({"type": "session_id", "id": session.conversation_id})

    conversation: list[str] = []
    objections: list[dict] = []
    coaching_count = 0

    for td in turns_data:
        await ws.send_json(
            {
                "type": "turn_end",
                "text": td["text"],
                "turn": td["turn"],
                "confidence": td["confidence"],
                "pending": True,
            }
        )

    try:
        while True:
            cmd = await ws.receive_json()
            cmd_type = cmd.get("type")

            if cmd_type == "coach_me":
                up_to = cmd.get("up_to_turn", len(turns_data))
                custom_question = cmd.get("question")
                conversation = [t["text"] for t in turns_data[:up_to]]
                try:
                    if critique_mode:
                        result = await get_coaching_with_critique(
                            llm,
                            conversation,
                            objections,
                            call_context,
                            up_to,
                            custom_question,
                        )
                    else:
                        result = await get_coaching(
                            llm,
                            conversation,
                            objections,
                            call_context,
                            up_to,
                            custom_question,
                        )
                    new_objs = result.get("objections", [])
                    if new_objs:
                        objections = new_objs
                        session.objections = new_objs
                    coaching_count += 1
                    advice = result.get("advice", "")
                    cs = result.get("close_score", -1)
                    sts = result.get("steps_to_close", -1)
                    session.add_coaching(
                        coaching_count,
                        up_to,
                        advice,
                        cs,
                        sts,
                        custom_question is not None,
                    )
                    payload = {
                        "type": "coaching",
                        "advice": advice,
                        "close_score": cs,
                        "steps_to_close": sts,
                        "objections": objections,
                        "number": coaching_count,
                        "turn": up_to,
                        "is_custom": custom_question is not None,
                    }
                    if result.get("candidates"):
                        payload["candidates"] = result["candidates"]
                    await ws.send_json(payload)
                except Exception as e:
                    await ws.send_json({"type": "error", "message": str(e)})

            elif cmd_type == "set_critique_mode":
                critique_mode = cmd.get("value", False)

            elif cmd_type == "generate_review":
                try:
                    conversation = [t["text"] for t in turns_data]
                    review = await generate_review(
                        transcript=session.turns,
                        coaching=session.coaching_log,
                        context=call_context,
                    )
                    await save_review(session.conversation_id, review)
                    await ws.send_json(
                        {
                            "type": "review",
                            "conversation_id": session.conversation_id,
                            "review": review,
                        }
                    )
                except Exception as e:
                    await ws.send_json({"type": "error", "message": str(e)})

            elif cmd_type == "reset":
                objections = []
                coaching_count = 0
                session.coaching_log = []
                session.objections = []
                await ws.send_json({"type": "status", "message": "Reset — cleared"})

    except WebSocketDisconnect:
        pass

    # Persist on disconnect
    await session.save()
    if session.objections:
        await save_hesitations(
            session.conversation_id, session.objections, session.contact_id
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
