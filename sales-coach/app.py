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
import time
from contextlib import asynccontextmanager
from typing import Any

import openai
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType

from db import (
    CONTACT_STATUSES,
    init_db,
    create_contact,
    delete_contact,
    list_contacts,
    list_contacts_with_summary,
    get_contact,
    get_contact_detail,
    update_contact,
    create_conversation,
    delete_conversation,
    end_conversation,
    update_conversation_progress,
    save_review,
    get_conversation,
    list_conversation_ids_with_reviews,
    list_conversations,
    save_hesitations,
    get_hesitations,
    get_contact_hesitations_summary,
    get_home_data,
    get_all_conversations_for_rep,
    get_next_steps_for_rep,
    get_sales_rep,
    save_performance_review,
    list_sales_reps,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sales-coach")

BASE_DIR = pathlib.Path(__file__).parent
SAMPLE_RATE = 16000


@asynccontextmanager
async def lifespan(application: FastAPI):
    await init_db()
    logger.info("Database initialized")
    await _migrate_reviews_to_canonical()
    yield


app = FastAPI(title="Sales Coach", lifespan=lifespan)

SYSTEM_PROMPT = """\
You are a sharp, experienced sales coach listening to a live call in real time.
Your job is to help the sales rep move toward closing — not by being pushy, but by \
uncovering and resolving objections, building urgency, and advancing commitment.

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


class CoachingObjection(BaseModel):
    """Single objection / hesitation."""

    text: str = Field(description="Short description of the objection")
    status: str = Field(description="open or resolved")
    rank: str | None = Field(default=None, description="S, M, or L")
    resolution_suggestion: str | None = Field(default=None)


class CoachingResponse(BaseModel):
    """Structured coaching reply — use with LLM structured output."""

    advice: str = Field(
        description="1-2 sentences. What should the rep say or do next. No JSON, no markdown."
    )
    close_score: int = Field(description="0=ice cold, 100=ready to sign")
    steps_to_close: int = Field(description="Estimated remaining steps to close (1-10)")
    objections: list[CoachingObjection] = Field(default_factory=list)


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
- next_call_prep: if you list multiple items (e.g. (1) ... (2) ...), put each item on its own line so the UI displays them clearly.
- suggested_status should reflect where this contact is in the pipeline based on the call outcome.
"""


class HesitationsSummary(BaseModel):
    """Hesitations uncovered, handled, and remaining from the call."""

    uncovered: list[str] = Field(default_factory=list)
    handled: list[str] = Field(default_factory=list)
    remaining: list[str] = Field(default_factory=list)


class PostCallReview(BaseModel):
    """Structured post-call review — use with LLM structured output."""

    score: int = Field(description="Overall effectiveness 1-10")
    went_well: list[str] = Field(default_factory=list)
    improve: list[str] = Field(default_factory=list)
    hesitations_summary: HesitationsSummary = Field(default_factory=HesitationsSummary)
    follow_ups: list[str] = Field(default_factory=list)
    key_moment: str | None = Field(default=None)
    next_call_prep: str | None = Field(
        default=None,
        description="What the rep should prepare before the next call. Use newlines between each numbered or bullet point so the UI can display them on separate lines.",
    )
    next_step: str | None = Field(default=None)
    suggested_status: str | None = Field(default=None)


class PerformanceReview(BaseModel):
    """Structured performance review across all conversations."""

    overall_assessment: str = Field(
        description="2-3 sentence summary of the rep's performance"
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="1-3 items only: what they do well consistently",
    )
    areas_for_improvement: list[str] = Field(
        default_factory=list,
        description="1-3 items only: recurring weaknesses or patterns",
    )
    common_mistakes: list[str] = Field(
        default_factory=list,
        description="1-3 items only: specific bad habits across conversations",
    )
    coaching_recommendations: list[str] = Field(
        default_factory=list,
        description="1-3 items only: actionable advice for their next calls",
    )
    score: int = Field(description="1-10 overall effectiveness rating")


class QuickActions(BaseModel):
    """Exactly 3 suggested next steps for the sales rep."""

    actions: list[str] = Field(
        min_length=3,
        max_length=3,
        description="Exactly 3 concrete, actionable next steps (strings).",
    )


def _get_llm() -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )


def _strip_markdown_json(text: str) -> str:
    """Remove markdown code fences (e.g. ```json ... ```) from a string.

    Handles trailing content after the closing fence (e.g. **Context:** …).
    """
    if not text or not isinstance(text, str):
        return text
    t = text.strip()
    # Match opening ``` … closing ``` even when there is trailing content
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?\s*```", t, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: strip opening fence only
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _extract_first_json_object(text: str) -> str | None:
    """Return the first complete JSON object or None.

    Uses json.loads as the oracle instead of fragile brace-counting
    (which breaks on braces inside string values).
    """
    if not text:
        return None
    s = text.strip()
    start = s.find("{")
    if start < 0:
        return None
    end = s.rfind("}")
    while end >= start:
        try:
            json.loads(s[start : end + 1])
            return s[start : end + 1]
        except json.JSONDecodeError:
            end = s.rfind("}", start, end)
    return None


def _sanitize_advice(advice: str) -> str:
    """Ensure advice is plain text, not raw JSON or markdown (including JSON + trailing **Why:**)."""
    if not advice or not isinstance(advice, str):
        return advice
    out = _strip_markdown_json(advice)
    if out.startswith("{"):
        first_json = _extract_first_json_object(out)
        if first_json:
            try:
                data = json.loads(first_json)
                if isinstance(data, dict) and "advice" in data:
                    return _sanitize_advice(str(data["advice"]))
            except json.JSONDecodeError:
                pass
    return out


def parse_coaching_response(raw: str) -> dict:
    """Extract JSON from the LLM response, handling markdown fences."""
    text = _strip_markdown_json(raw)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "advice" in data:
            data["advice"] = _sanitize_advice(str(data["advice"]))
        return data
    except json.JSONDecodeError:
        return {
            "advice": _sanitize_advice(raw.strip()),
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


def _normalize_confidence(raw: Any) -> float:
    """Coerce stored confidence (int, float, or string like '85' / '85%') to 0..1 float."""
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        try:
            v = float(raw)
            return v if 0 <= v <= 1 else v / 100.0 if v > 1 else 0.0
        except (TypeError, ValueError):
            return 0.0
    s = str(raw).strip().rstrip("%")
    try:
        v = float(s)
        return v if 0 <= v <= 1 else v / 100.0
    except ValueError:
        return 0.0


def _transcript_to_turns_data(transcript: list[Any]) -> list[dict]:
    """Normalize DB transcript (list of dicts or strings) to [{turn, text, confidence}]."""
    out = []
    for i, t in enumerate(transcript):
        if isinstance(t, str):
            out.append({"turn": i + 1, "text": t or "", "confidence": 0.0})
            continue
        if not isinstance(t, dict):
            continue
        turn = t.get("turn", i + 1)
        try:
            turn = int(turn) if turn is not None else i + 1
        except (TypeError, ValueError):
            turn = i + 1
        text = t.get("text") or t.get("content") or ""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        conf = _normalize_confidence(t.get("confidence"))
        out.append({"turn": turn, "text": text, "confidence": conf})
    return out


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
    return JSONResponse({"status": "ok", "service": "sales-coach-api"})


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
        phone=body.get("phone"),
        notes=body.get("notes"),
        status=body.get("status", "prospect"),
    )
    return {"id": cid}


@app.put("/api/contacts/{contact_id}")
async def api_update_contact(contact_id: int, body: dict):
    await update_contact(contact_id, **body)
    return {"ok": True}


@app.delete("/api/contacts/{contact_id}")
async def api_delete_contact(contact_id: int):
    await delete_contact(contact_id)
    return {"ok": True}


# ── REST API: Home / Sales Reps ──────────────────────────────────


@app.get("/api/home")
async def api_home(sales_rep_id: int = 1):
    return await get_home_data(sales_rep_id)


@app.get("/api/sales-reps")
async def api_list_sales_reps():
    return await list_sales_reps()


PERFORMANCE_REVIEW_PROMPT = """\
You are an expert sales manager conducting a performance review for a sales rep.
Below is the complete history of their sales conversations, including transcripts, \
coaching they received during calls, and post-call reviews.

**Tone:** Address the rep directly. Use their first name (e.g. "Josiah") and write \
in second person throughout: "you", "your" (e.g. "You demonstrate strong rapport…" \
"Your outreach needs refinement."). Do NOT use third person (he/his, she/her, \
"[Name] demonstrates", "they need to improve").

Analyze all of this data and produce a thorough performance review covering:

1. **Overall Assessment** (2-3 sentences addressing the rep directly: "You…" / "Your…")
2. **Strengths** (1-3 items only — the most important things they do well; "You…" or short phrases)
3. **Areas for Improvement** (1-3 items only — recurring weaknesses; address as "you/your")
4. **Common Mistakes** (1-3 items only — specific bad habits; address as "you/your")
5. **Coaching Recommendations** (1-3 items only — actionable advice for their next calls; "You should…" etc.)
6. **Score** (1-10 overall effectiveness rating — see rubric below)

For each list (Strengths, Areas for Improvement, Common Mistakes, Coaching Recommendations), include only 1-3 items — pick the most impactful. Be specific and reference actual patterns. Don't be generic.
Be constructive but honest — the goal is to help them improve.

**SCORING RUBRIC (use this consistently so scores are comparable over time):**
- **1–2**: Consistently poor. Frequently loses the prospect, ignores coaching, or undermines the sale. Little to no discovery, rapport, or closing behavior. Repeated critical mistakes.
- **3–4**: Below expectations. Major gaps in discovery, handling objections, or moving the deal forward. Often misses obvious next steps or fails to follow up on coaching. Inconsistent.
- **5–6**: Meets baseline / developing. Does the basics: some discovery, some rapport, follows some coaching. Still makes recurring mistakes or leaves opportunities on the table. Room for clear improvement.
- **7–8**: Solid performer. Good discovery and rapport, generally applies coaching, moves deals forward. Minor, fixable patterns. Reliable and effective most of the time.
- **9–10**: Exceptional. Strong discovery, handles objections well, builds trust, closes effectively. Consistently applies or exceeds coaching. Few or no recurring issues; stands out as a top performer.

Assign the score that best matches the rep's typical performance across the conversations provided. If evidence is mixed, weight the majority of calls and the trend (e.g. improving vs. declining)."""


def _get_performance_review_llm() -> ChatOpenAI:
    """LangChain client for OpenRouter — Gemini 2.5 Flash for large context."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")
    llm = ChatOpenAI(
        model="google/gemini-2.5-flash",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        max_tokens=4096,
        temperature=0.3,
    )
    return llm.with_structured_output(PerformanceReview)


@app.post("/api/sales-reps/{rep_id}/performance-review")
async def api_generate_performance_review(rep_id: int):
    """Aggregate all conversations for the rep and generate a performance review."""
    rep = await get_sales_rep(rep_id)
    if not rep:
        return JSONResponse({"error": "Sales rep not found"}, status_code=404)

    conversations = await get_all_conversations_for_rep(rep_id)
    if not conversations:
        return {"error": "No conversations found for this sales rep"}

    first_name = rep.get("first_name") or "there"

    # Build a big text block with all conversations
    parts: list[str] = []
    for conv in conversations:
        header = (
            f"--- Conversation #{conv['id']} with {conv.get('contact_name', 'Unknown')}"
        )
        if conv.get("contact_company"):
            header += f" @ {conv['contact_company']}"
        header += f" ({conv.get('started_at', '?')})"
        if conv.get("final_close_score") is not None:
            header += f" [close score: {conv['final_close_score']}%]"
        header += " ---"
        parts.append(header)

        transcript = conv.get("transcript") or []
        for t in transcript:
            parts.append(f"  Turn {t.get('turn', '?')}: {t.get('text', '')}")

        coaching = conv.get("coaching") or []
        if coaching:
            parts.append("  Coaching provided during call:")
            for c in coaching:
                parts.append(
                    f"    Coach #{c.get('number', '?')} (turn {c.get('turn', '?')}): {c.get('advice', '')}"
                )

        review = conv.get("review")
        if review and isinstance(review, dict):
            parts.append(f"  Post-call review score: {review.get('score', 'N/A')}/10")
            for item in review.get("went_well", []):
                parts.append(f"    + {item}")
            for item in review.get("improve", []):
                parts.append(f"    - {item}")

        parts.append("")

    all_data = "\n".join(parts)

    try:
        structured_llm = _get_performance_review_llm()
        system_content = (
            PERFORMANCE_REVIEW_PROMPT
            + f"\n\nThe sales rep's first name is **{first_name}**. Address them directly using this name and second person (you/your)."
        )
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=f"SALES CONVERSATION HISTORY:\n\n{all_data}"),
        ]
        result: PerformanceReview = await structured_llm.ainvoke(messages)
        review_data = result.model_dump()
        for key in (
            "strengths",
            "areas_for_improvement",
            "common_mistakes",
            "coaching_recommendations",
        ):
            if isinstance(review_data.get(key), list):
                review_data[key] = review_data[key][:3]
    except Exception as e:
        logger.exception("Performance review generation failed")
        return JSONResponse(
            {"error": f"Failed to generate review: {str(e)}"}, status_code=500
        )

    await save_performance_review(rep_id, json.dumps(review_data))
    return review_data


QUICK_ACTIONS_PROMPT = """\
You are a sales coach. Given today's date and, for each contact, their current "next step" \
(from their latest call) and your notes, suggest exactly 3 quick-action next steps for the rep to do. \
Use both next steps and notes to pick the most impactful and time-sensitive. \
Each action should be one concrete sentence \
(e.g. "Send Kristine 2-3 ROI case studies before Wednesday's call" or "Follow up with Billy on budget timeline"). \
Output exactly 3 strings, no more, no less. Prioritize by urgency and deal momentum."""


def _get_quick_actions_llm() -> ChatOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")
    llm = ChatOpenAI(
        model="google/gemini-2.5-flash",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        max_tokens=512,
        temperature=0.4,
    )
    return llm.with_structured_output(QuickActions)


@app.get("/api/sales-reps/{rep_id}/quick-actions")
async def api_get_quick_actions(rep_id: int):
    """Return 3 suggested next steps based on contacts' next_step and today's date."""
    from datetime import datetime, timezone

    rep = await get_sales_rep(rep_id)
    if not rep:
        return JSONResponse({"error": "Sales rep not found"}, status_code=404)

    next_steps = await get_next_steps_for_rep(rep_id)
    today = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")

    if not next_steps:
        return {"actions": []}

    lines = []
    for s in next_steps:
        line = (
            f"- {s['name']}"
            + (f" (@ {s['company']})" if s.get("company") else "")
            + f": next_step = {s['next_step']}"
        )
        if s.get("notes") and (notes := (s["notes"] or "").strip()):
            line += f"; notes = {notes}"
        lines.append(line)
    user_content = (
        f"Today's date: {today}\n\nContacts (next_step and notes):\n" + "\n".join(lines)
    )

    try:
        structured_llm = _get_quick_actions_llm()
        messages = [
            SystemMessage(content=QUICK_ACTIONS_PROMPT),
            HumanMessage(content=user_content),
        ]
        result: QuickActions = await structured_llm.ainvoke(messages)
        actions = result.actions[:3]
    except Exception as e:
        logger.exception("Quick actions generation failed")
        return JSONResponse(
            {"error": f"Failed to generate quick actions: {str(e)}"}, status_code=500
        )

    return {"actions": actions}


class ColdCallPrepRequest(BaseModel):
    contact_name: str
    company: str | None = None


def _get_cold_call_prep_llm() -> ChatOpenAI:
    """OpenRouter with :online for web search (Exa)."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")
    return ChatOpenAI(
        model="anthropic/claude-haiku-4.5:online",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        max_tokens=1024,
        temperature=0.3,
    )


COLD_CALL_PREP_PROMPT = """\
Tell me just enough information for me to know to pick up a phone and cold call this business and offer social media services. \
Be concise: a few bullet points or short paragraphs. Include what the business does, any relevant recent news or context, and why they might care about social media. \
Cite sources using markdown links if the model provided them."""


@app.post("/api/cold-call-prep")
async def api_cold_call_prep(body: ColdCallPrepRequest):
    """Web-search-backed cold call prep for a contact's business. Uses OpenRouter :online."""
    if not body.contact_name.strip():
        return JSONResponse({"error": "contact_name is required"}, status_code=400)
    user_content = f"Contact name: {body.contact_name.strip()}\nBusiness: {(body.company or '').strip() or '(not specified)'}"
    try:
        llm = _get_cold_call_prep_llm()
        messages = [
            SystemMessage(content=COLD_CALL_PREP_PROMPT),
            HumanMessage(content=user_content),
        ]
        result = await llm.ainvoke(messages)
        content = (
            result.content if hasattr(result, "content") and result.content else ""
        )
        return {"content": content}
    except Exception as e:
        logger.exception("Cold call prep failed")
        return JSONResponse(
            {"error": f"Cold call prep failed: {str(e)}"}, status_code=500
        )


# ── Help me prepare (pre-call brief) ──────────────────────────────

PREPARE_PROMPT = """\
You are a sales coach. Given a contact's notes, outstanding hesitations (objections/concerns), past conversations with this contact, the sales rep's name, and the rep's latest performance review, write a short pre-call preparation brief.

Include:
1. Key context from the contact's notes.
2. Any outstanding hesitations and how to address them.
3. What happened on past calls (next steps, scores, any pattern).
4. 2–3 suggested goals or talking points for this call, tailored to this contact and to the rep's coaching recommendations where relevant.

Be concise and actionable. Write in second person ("You...") addressing the sales rep. Use bullet points or short paragraphs."""


def _get_prepare_llm() -> ChatOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")
    return ChatOpenAI(
        model="anthropic/claude-haiku-4.5",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        max_tokens=1024,
        temperature=0.3,
    )


async def _generate_prepare_brief(contact_id: int) -> tuple[str, str]:
    """Return (contact_notes, call_prep) for a contact. On failure returns (notes, '')."""
    contact = await get_contact_detail(contact_id)
    if not contact:
        return ("", "")
    notes_str = (contact.get("notes") or "").strip()

    rep_id = contact.get("sales_rep_id")
    if not rep_id:
        return (notes_str, "")
    rep = await get_sales_rep(rep_id)
    if not rep:
        return (notes_str, "")

    parts = []
    parts.append(
        f"Sales rep: {rep.get('first_name', '')} {rep.get('last_name', '')}".strip()
    )
    parts.append(
        f"Contact: {contact.get('name', '')} @ {contact.get('company', '') or '(no company)'}".strip()
    )
    parts.append("")
    if notes_str:
        parts.append("CONTACT NOTES:")
        parts.append(notes_str)
        parts.append("")
    else:
        parts.append("CONTACT NOTES: (none)")
        parts.append("")

    research_str = (contact.get("research") or "").strip()
    if research_str:
        parts.append("CONTACT RESEARCH (saved cold-call prep):")
        parts.append(research_str)
        parts.append("")
    else:
        parts.append("CONTACT RESEARCH: (none)")
        parts.append("")

    open_hes = contact.get("open_hesitations") or []
    if open_hes:
        parts.append("OUTSTANDING HESITATIONS:")
        for h in open_hes:
            rank = h.get("rank") or "?"
            text = h.get("text") or ""
            resolution = (h.get("resolution_suggestion") or "").strip()
            line = f"- [{rank}] {text}"
            if resolution:
                line += f" → Suggestion: {resolution}"
            parts.append(line)
        parts.append("")
    else:
        parts.append("OUTSTANDING HESITATIONS: (none)")
        parts.append("")

    convos = contact.get("conversations") or []
    if convos:
        parts.append("PAST CONVERSATIONS (most recent first):")
        for c in convos[:10]:
            line = (
                f"- {c.get('started_at', '?')}: next_step = {c.get('next_step') or '—'}"
            )
            if c.get("final_close_score") is not None:
                line += f", close score = {c.get('final_close_score')}%"
            parts.append(line)
        parts.append("")
    else:
        parts.append("PAST CONVERSATIONS: (none yet)")
        parts.append("")

    review_json = rep.get("performance_review_json")
    if review_json:
        try:
            review = json.loads(review_json)
            parts.append("REP'S LATEST PERFORMANCE REVIEW:")
            if isinstance(review, dict):
                if review.get("overall_assessment"):
                    parts.append(f"Overall: {review['overall_assessment']}")
                for key in (
                    "strengths",
                    "areas_for_improvement",
                    "common_mistakes",
                    "coaching_recommendations",
                ):
                    vals = review.get(key)
                    if vals and isinstance(vals, list):
                        parts.append(
                            f"{key.replace('_', ' ').title()}: "
                            + "; ".join(str(v) for v in vals[:3])
                        )
            else:
                parts.append(str(review)[:1500])
        except (json.JSONDecodeError, TypeError):
            parts.append("(review parse error)")
    else:
        parts.append("REP'S LATEST PERFORMANCE REVIEW: (none)")

    user_content = "\n".join(parts)
    try:
        llm = _get_prepare_llm()
        messages = [
            SystemMessage(content=PREPARE_PROMPT),
            HumanMessage(content=user_content),
        ]
        result = await llm.ainvoke(messages)
        content = (
            result.content if hasattr(result, "content") and result.content else ""
        )
        return (notes_str, content)
    except Exception as e:
        logger.warning("Prepare brief failed for contact %s: %s", contact_id, e)
        return (notes_str, "")


@app.get("/api/contacts/{contact_id}/prepare")
async def api_contact_prepare(contact_id: int):
    """Generate a pre-call preparation brief from contact notes, past conversations, rep, and last performance review."""
    contact = await get_contact_detail(contact_id)
    if not contact:
        return JSONResponse({"error": "Contact not found"}, status_code=404)
    if not contact.get("sales_rep_id"):
        return JSONResponse(
            {"error": "Contact has no assigned sales rep"}, status_code=400
        )
    try:
        _, content = await _generate_prepare_brief(contact_id)
        return {"content": content}
    except Exception as e:
        logger.exception("Prepare brief failed")
        return JSONResponse({"error": f"Prepare failed: {str(e)}"}, status_code=500)


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
    # Normalize legacy coaching: advice may be stored as raw JSON + markdown
    for c in conv.get("coaching") or []:
        if isinstance(c, dict) and c.get("advice"):
            c["advice"] = _sanitize_advice(str(c["advice"]))
    return conv


@app.delete("/api/conversations/{conversation_id}")
async def api_delete_conversation(conversation_id: int):
    await delete_conversation(conversation_id)
    return {"ok": True}


@app.post("/api/conversations/{conversation_id}/review")
async def api_generate_review(conversation_id: int):
    """Generate (or regenerate) a post-call review."""
    conv = await get_conversation(conversation_id)
    if not conv:
        return {"error": "not found"}
    review = await generate_review(
        transcript=conv["transcript"],
        coaching=conv["coaching"],
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
    turn_num: int,
    custom_question: str | None = None,
    contact_notes: str = "",
    call_prep: str = "",
) -> dict:
    """Call the LLM for coaching advice. Returns dict with advice, close_score, steps_to_close, objections."""
    recent = conversation[-15:]
    conv_text = "\n".join(f"Turn {i + 1}: {t}" for i, t in enumerate(recent))

    user_content = ""
    if contact_notes or call_prep:
        if contact_notes:
            user_content += f"CONTACT NOTES:\n{contact_notes}\n\n"
        if call_prep:
            user_content += f"PRE-CALL PREPARATION (use this context when advising):\n{call_prep}\n\n"
    if objections:
        obj_summary = "\n".join(
            f"- [{o['status'].upper()}] {o['text']}" for o in objections
        )
        user_content += f"HESITATIONS TRACKED SO FAR:\n{obj_summary}\n\n"

    user_content += f"TRANSCRIPT (most recent turns):\n{conv_text}\n\n"

    if custom_question:
        user_content += (
            f"THE SALES REP IS ASKING YOU SPECIFICALLY:\n{custom_question}\n\n"
            "Answer their question with context from the conversation, "
            "and still provide your standard coaching (advice, scores, objections)."
        )
    else:
        user_content += "What should the sales rep say next?"

    structured_llm = _get_coaching_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
    response: CoachingResponse = await structured_llm.ainvoke(messages)
    out = response.model_dump()
    return out


def _coaching_user_content(
    conversation: list[str],
    objections: list[dict],
    custom_question: str | None,
    contact_notes: str = "",
    call_prep: str = "",
) -> str:
    """Build user message for both single and actor/critic coaching."""
    recent = conversation[-15:]
    conv_text = "\n".join(f"Turn {i + 1}: {t}" for i, t in enumerate(recent))
    out = ""
    if contact_notes or call_prep:
        if contact_notes:
            out += f"CONTACT NOTES:\n{contact_notes}\n\n"
        if call_prep:
            out += f"PRE-CALL PREPARATION (use this context when advising):\n{call_prep}\n\n"
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
    turn_num: int,
    custom_question: str | None = None,
    contact_notes: str = "",
    call_prep: str = "",
) -> dict:
    """Actor: suggest 3–5 candidate next steps. Critic: score each for P(close). Return best + full list."""
    user_content = _coaching_user_content(
        conversation, objections, custom_question, contact_notes, call_prep
    )
    user_content += "Suggest 3–5 distinct candidate next steps (JSON with candidates and objections)."

    # Actor: get candidates
    response = await llm.chat.completions.create(
        model="anthropic/claude-haiku-4-5",
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


def _get_review_llm() -> ChatOpenAI:
    """LangChain client for OpenRouter used with structured output."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")
    llm = ChatOpenAI(
        model="anthropic/claude-haiku-4-5",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        max_tokens=4096,
    )
    return llm.with_structured_output(PostCallReview)


def _get_coaching_llm() -> ChatOpenAI:
    """LangChain client for OpenRouter used with structured coaching output."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")
    llm = ChatOpenAI(
        model="anthropic/claude-haiku-4-5",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        max_tokens=4096,
    )
    return llm.with_structured_output(CoachingResponse)


async def generate_review(
    transcript: list[dict],
    coaching: list[dict],
) -> dict:
    conv_text = "\n".join(
        f"Turn {t.get('turn', i+1)}: {t.get('text', '')}"
        for i, t in enumerate(transcript)
    )
    coach_text = "\n".join(
        f"Coaching #{c.get('number', i+1)} (turn {c.get('turn', '?')}): {c.get('advice', '')}"
        for i, c in enumerate(coaching)
    )

    user_content = f"FULL TRANSCRIPT:\n{conv_text}\n\n"
    if coach_text:
        user_content += f"COACHING PROVIDED DURING CALL:\n{coach_text}\n\n"
    user_content += "Provide your post-call review."

    structured_llm = _get_review_llm()
    messages = [
        SystemMessage(content=REVIEW_PROMPT),
        HumanMessage(content=user_content),
    ]
    response: PostCallReview = await structured_llm.ainvoke(messages)
    return response.model_dump()


def _legacy_review_to_canonical(review: dict) -> dict:
    """Normalize legacy review JSON to canonical shape (score, went_well, improve, etc.)."""
    if not review or not isinstance(review, dict):
        return review
    # went_well
    for key, alts in (
        ("went_well", ("what_went_well", "whatWentWell", "strengths")),
        ("improve", ("areas_to_improve", "areasToImprove", "improvements")),
        ("follow_ups", ("followUps", "next_steps")),
    ):
        if key not in review or not isinstance(review.get(key), list):
            for alt in alts:
                if alt in review and isinstance(review[alt], list):
                    review[key] = [str(x) for x in review[alt]]
                    break
            else:
                review[key] = review.get(key) or []
    # score
    raw = review.get("score")
    if isinstance(raw, (int, float)):
        review["score"] = max(0, min(10, int(raw)))
    else:
        for alt in ("overall_score", "effectiveness_score", "effectiveness", "rating"):
            v = review.get(alt)
            if v is None:
                continue
            try:
                n = int(v) if isinstance(v, (int, float)) else int(str(v).strip())
                review["score"] = max(0, min(10, n))
                break
            except (TypeError, ValueError):
                continue
        else:
            review["score"] = 0
    for key in ("went_well", "improve", "follow_ups"):
        review[key] = [str(x) for x in (review.get(key) or []) if x is not None]
    # hesitations_summary
    hs = review.get("hesitations_summary")
    if not isinstance(hs, dict):
        hs = {}
    for canonical, alts in (
        ("uncovered", ("uncovered", "identified", "found")),
        ("handled", ("handled", "resolved", "addressed")),
        ("remaining", ("remaining", "open", "unresolved")),
    ):
        if canonical not in hs or not isinstance(hs.get(canonical), list):
            for a in alts:
                if a in hs and isinstance(hs[a], list):
                    hs[canonical] = [str(x) for x in hs[a]]
                    break
            else:
                hs[canonical] = hs.get(canonical) or []
    review["hesitations_summary"] = {
        "uncovered": [str(x) for x in (hs.get("uncovered") or [])],
        "handled": [str(x) for x in (hs.get("handled") or [])],
        "remaining": [str(x) for x in (hs.get("remaining") or [])],
    }
    return review


async def _migrate_reviews_to_canonical() -> None:
    """One-time: normalize all existing review_json to canonical shape."""
    ids = await list_conversation_ids_with_reviews()
    if not ids:
        return
    logger.info("Migrating %d conversation review(s) to canonical shape", len(ids))
    for cid in ids:
        try:
            conv = await get_conversation(cid)
            if (
                not conv
                or not conv.get("review")
                or not isinstance(conv["review"], dict)
            ):
                continue
            normalized = _legacy_review_to_canonical(conv["review"])
            await save_review(cid, normalized)
        except Exception as e:
            logger.warning("Failed to migrate review for conversation %s: %s", cid, e)


# ── Session data collector ───────────────────────────────────────


class SessionData:
    """Accumulates data during a WebSocket session for DB persistence."""

    def __init__(self):
        self.conversation_id: int | None = None
        self.contact_id: int | None = None
        self.contact_notes: str = ""
        self.call_prep: str = ""
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

    # First message: config (or resume with conversation_id)
    try:
        init_msg = await ws.receive_json()
        auto_coach = init_msg.get("auto_coach", True)
        critique_mode = init_msg.get("critique_mode", False)
        contact_id = init_msg.get("contact_id")
        if contact_id:
            session.contact_id = int(contact_id)
        resume_id = init_msg.get("conversation_id")
    except Exception:
        auto_coach = True
        critique_mode = False
        resume_id = None

    conversation: list[str] = []
    objections: list[dict] = []
    coaching_count = 0

    if resume_id:
        # Resume: load existing conversation (must not be ended)
        existing = await get_conversation(int(resume_id))
        if existing and not existing.get("ended_at"):
            session.conversation_id = existing["id"]
            session.contact_id = existing.get("contact_id") or session.contact_id
            for t in existing.get("transcript") or []:
                session.turns.append(
                    {
                        "turn": t.get("turn", len(session.turns) + 1),
                        "text": t.get("text", ""),
                        "confidence": t.get("confidence", 0),
                    }
                )
            conversation = [
                t.get("text", "") for t in (existing.get("transcript") or [])
            ]
            for c in existing.get("coaching") or []:
                session.coaching_log.append(c)
                coaching_count += 1
            objections = await get_hesitations(conversation_id=session.conversation_id)
            session.objections = objections
            if session.coaching_log:
                session.last_close_score = session.coaching_log[-1].get("close_score")
                session.last_steps_to_close = session.coaching_log[-1].get(
                    "steps_to_close"
                )
            logger.info(
                "Resuming conversation %s (%d turns)",
                session.conversation_id,
                len(conversation),
            )
        else:
            resume_id = None
    if not session.conversation_id:
        session.conversation_id = await create_conversation(
            contact_id=session.contact_id,
            mode="live",
        )
    # Load contact notes and generate call prep so the coach can use them
    if session.contact_id:
        try:
            session.contact_notes, session.call_prep = await _generate_prepare_brief(
                session.contact_id
            )
        except Exception as e:
            logger.warning("Could not load contact notes/prep for coach: %s", e)

    await ws.send_json({"type": "session_id", "id": session.conversation_id})

    msg_queue: asyncio.Queue[Any] = asyncio.Queue()
    cmd_queue: asyncio.Queue[dict] = asyncio.Queue()
    running = True
    last_speech_at: float = 0.0
    SILENCE_AUTO_END_SEC = 20.0
    pause_requested = False
    auto_end_requested = False

    def on_dg_message(message: Any) -> None:
        if running:
            msg_queue.put_nowait(message)

    async def silence_checker() -> None:
        nonlocal auto_end_requested
        while running:
            await asyncio.sleep(2.0)
            if not running:
                return
            if (
                last_speech_at > 0
                and (time.monotonic() - last_speech_at) >= SILENCE_AUTO_END_SEC
            ):
                auto_end_requested = True
                try:
                    await ws.send_json({"type": "auto_end", "reason": "silence"})
                except Exception:
                    pass
                return

    async def do_coaching(turn_num: int, custom_question: str | None = None) -> None:
        nonlocal coaching_count, objections
        try:
            await ws.send_json({"type": "coaching_started"})
            if critique_mode:
                result = await get_coaching_with_critique(
                    llm,
                    conversation,
                    objections,
                    turn_num,
                    custom_question,
                    session.contact_notes,
                    session.call_prep,
                )
            else:
                result = await get_coaching(
                    llm,
                    conversation,
                    objections,
                    turn_num,
                    custom_question,
                    session.contact_notes,
                    session.call_prep,
                )
            new_objs = result.get("objections", [])
            if new_objs:
                objections = new_objs
                session.objections = new_objs

            coaching_count += 1
            advice = _sanitize_advice(result.get("advice", "") or "")
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
        nonlocal last_speech_at
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
                    last_speech_at = time.monotonic()
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
            silence_task = asyncio.create_task(silence_checker())

            await ws.send_json({"type": "status", "message": "Ready — start talking!"})

            try:
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
                    except asyncio.TimeoutError:
                        if auto_end_requested:
                            break
                        continue
                    if msg["type"] == "websocket.receive":
                        if "bytes" in msg and msg["bytes"]:
                            await dg_conn.send_media(msg["bytes"])
                        elif "text" in msg and msg["text"]:
                            try:
                                cmd = json.loads(msg["text"])
                                if cmd.get("type") == "pause":
                                    pause_requested = True
                                    cid = session.conversation_id
                                    await ws.send_json(
                                        {"type": "paused", "conversation_id": cid}
                                    )
                                    if cid:
                                        try:
                                            await update_conversation_progress(
                                                cid,
                                                session.turns,
                                                session.coaching_log,
                                                session.objections,
                                                session.contact_id,
                                            )
                                        except Exception as e:
                                            logger.warning(
                                                "Pause: failed to save progress for conversation %s: %s",
                                                cid,
                                                e,
                                            )
                                    break
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
                silence_task.cancel()
                try:
                    await silence_task
                except asyncio.CancelledError:
                    pass

    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass

    # ── Session end: persist + generate review (skip if user paused) ──
    if pause_requested:
        logger.info("Conversation %s paused (not ended)", session.conversation_id)
    else:
        cid = await session.save()
        if cid and session.turns:
            logger.info("Conversation %d saved (%d turns)", cid, len(session.turns))
            try:
                review = await generate_review(
                    transcript=session.turns,
                    coaching=session.coaching_log,
                )
                await save_review(cid, review)
                logger.info("Review generated for conversation %d", cid)
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
    """Replay a conversation (by id from DB or from pasted file) for testing coaching prompts."""
    await ws.accept()

    llm = _get_llm()
    session = SessionData()

    # First message: config — either conversation_id (load from DB) or conversation (raw text)
    try:
        init_msg = await ws.receive_json()
        conversation_text = init_msg.get("conversation", "")
        conversation_id_load = init_msg.get("conversation_id")
        contact_id = init_msg.get("contact_id")
        critique_mode = init_msg.get("critique_mode", False)
        if contact_id:
            session.contact_id = int(contact_id)
    except Exception:
        await ws.send_json(
            {
                "type": "error",
                "message": "Expected init message with conversation_id or conversation",
            }
        )
        return

    turns_data: list[dict] = []
    if conversation_id_load is not None:
        conv = await get_conversation(int(conversation_id_load))
        if not conv or not conv.get("transcript"):
            await ws.send_json(
                {
                    "type": "error",
                    "message": "Conversation not found or has no transcript",
                }
            )
            return
        if conv.get("contact_id") and not session.contact_id:
            session.contact_id = conv["contact_id"]
        try:
            turns_data = _transcript_to_turns_data(conv["transcript"])
        except Exception as e:
            await ws.send_json(
                {
                    "type": "error",
                    "message": f"Could not load conversation transcript: {e!s}",
                }
            )
            return
    else:
        turns_data = parse_conversation_file(conversation_text)

    # Create conversation record (turns_data may be empty for manual-turn mode)
    session.conversation_id = await create_conversation(
        contact_id=session.contact_id,
        mode="test",
    )
    if session.contact_id:
        try:
            session.contact_notes, session.call_prep = await _generate_prepare_brief(
                session.contact_id
            )
        except Exception as e:
            logger.warning("Could not load contact notes/prep for test coach: %s", e)
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
                await ws.send_json({"type": "coaching_started"})
                try:
                    if critique_mode:
                        result = await get_coaching_with_critique(
                            llm,
                            conversation,
                            objections,
                            up_to,
                            custom_question,
                            session.contact_notes,
                            session.call_prep,
                        )
                    else:
                        result = await get_coaching(
                            llm,
                            conversation,
                            objections,
                            up_to,
                            custom_question,
                            session.contact_notes,
                            session.call_prep,
                        )
                    new_objs = result.get("objections", [])
                    if new_objs:
                        objections = new_objs
                        session.objections = new_objs
                    coaching_count += 1
                    advice = _sanitize_advice(result.get("advice", "") or "")
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

            elif cmd_type == "add_turn":
                text = (cmd.get("text") or "").strip()
                if text:
                    turn_num = len(turns_data) + 1
                    td_new = {"turn": turn_num, "text": text, "confidence": 1.0}
                    turns_data.append(td_new)
                    session.add_turn(turn_num, text, 1.0)
                    conversation = [t["text"] for t in turns_data]
                    await ws.send_json(
                        {
                            "type": "turn_end",
                            "text": text,
                            "turn": turn_num,
                            "confidence": 1.0,
                        }
                    )
                    # Auto-coach on the new turn
                    await ws.send_json({"type": "coaching_started"})
                    try:
                        if critique_mode:
                            result = await get_coaching_with_critique(
                                llm,
                                conversation,
                                objections,
                                turn_num,
                                None,
                                session.contact_notes,
                                session.call_prep,
                            )
                        else:
                            result = await get_coaching(
                                llm,
                                conversation,
                                objections,
                                turn_num,
                                None,
                                session.contact_notes,
                                session.call_prep,
                            )
                        new_objs = result.get("objections", [])
                        if new_objs:
                            objections = new_objs
                            session.objections = new_objs
                        coaching_count += 1
                        advice = _sanitize_advice(result.get("advice", "") or "")
                        cs = result.get("close_score", -1)
                        sts = result.get("steps_to_close", -1)
                        session.add_coaching(
                            coaching_count, turn_num, advice, cs, sts, False
                        )
                        payload = {
                            "type": "coaching",
                            "advice": advice,
                            "close_score": cs,
                            "steps_to_close": sts,
                            "objections": objections,
                            "number": coaching_count,
                            "turn": turn_num,
                            "is_custom": False,
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
    uvicorn.run(app, host="0.0.0.0", port=5050, reload=True)
