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
from collections.abc import AsyncGenerator
from typing import Any

import openai
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
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
    update_conversation_prep_notes,
    save_review,
    get_conversation,
    list_conversation_ids_with_reviews,
    list_conversations,
    save_hesitations,
    get_hesitations,
    get_contact_hesitations_summary,
    get_home_data,
    get_all_conversations_for_seller,
    get_next_steps_for_seller,
    get_seller,
    save_performance_review,
    list_sellers,
    list_todos,
    create_todo,
    delete_todo,
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

# Minimal system prompt for advice-only (low latency). No scores, no objections.
ADVICE_ONLY_SYSTEM_PROMPT = """\
You are a sharp sales coach. The seller is on a live call. Give ONE line only: \
a question to ask (e.g. "What's your timeline?") or a 3–7 word hint (e.g. "Probe budget."). \
Like a slip of paper. No sentences, no bullets, no explanation. Be specific to the conversation.
"""

# System prompt for scores + objections only (run on-demand when user clicks refresh).
SCORES_SYSTEM_PROMPT = """\
You are a sales coach. Based on the conversation so far, provide:
1. close_score: 0 = ice cold, 100 = ready to sign. Base on verbal commitment, objections resolved, urgency, buying signals.
2. steps_to_close: 1–10 (1 = just needs signature, 10 = long way to go).
3. objections: List all objections/hesitations surfaced (explicit or implied). For each: text, status (open/resolved), rank (S/M/L), resolution_suggestion for open ones.
Rank: S = minor | M = blocking but overcomeable | L = hard stop. Include previously identified objections with updated status.
"""


class CoachingObjection(BaseModel):
    """Single objection / hesitation."""

    text: str = Field(description="Short description of the objection")
    status: str = Field(description="open or resolved")
    rank: str | None = Field(default=None, description="S, M, or L")
    resolution_suggestion: str | None = Field(default=None)


class CoachingAdviceOnly(BaseModel):
    """Advice only — for low-latency in-call coaching."""

    advice: str = Field(
        description="ONE line only: a short question to ask or 3–7 word hint. Like a slip of paper. No JSON, no markdown."
    )


class CoachingScoresResponse(BaseModel):
    """Scores and objections only — for on-demand refresh."""

    close_score: int = Field(description="0=ice cold, 100=ready to sign")
    steps_to_close: int = Field(description="Estimated remaining steps to close (1-10)")
    objections: list[CoachingObjection] = Field(default_factory=list)


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
        description="2-3 sentence summary of the seller's performance"
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


class SuggestedTodo(BaseModel):
    """A single AI-suggested task."""

    type: str = Field(description="One of: email, call, other")
    title: str = Field(description="Concise actionable task (one sentence)")
    contact_id: int | None = Field(
        default=None,
        description="The contact ID this task relates to (from the list provided), or null if general",
    )


class SuggestedTodos(BaseModel):
    """Up to 5 suggested tasks for the seller."""

    items: list[SuggestedTodo] = Field(
        min_length=1,
        max_length=5,
        description="1-5 concrete, actionable tasks for the seller.",
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
        email=body.get("email"),
        notes=body.get("notes"),
        status=body.get("status", "prospect"),
    )
    return {"id": cid}


@app.put("/api/contacts/{contact_id}")
async def api_update_contact(contact_id: int, body: dict):
    await update_contact(contact_id, **body)
    return {"ok": True}


class AskCoachBody(BaseModel):
    prompt: str = Field(..., min_length=1)


def _get_ask_coach_llm() -> ChatOpenAI:
    """LangChain client for OpenRouter — Gemini 2.5 Flash for ask-coach."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")
    return ChatOpenAI(
        model="google/gemini-2.5-flash",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        max_tokens=4096,
        temperature=0.3,
    )


@app.post("/api/contacts/{contact_id}/ask-coach")
async def api_contact_ask_coach(contact_id: int, body: AskCoachBody):
    """Ask the coach a question with full context: contact, seller, and all conversations."""
    contact_detail = await get_contact_detail(contact_id)
    if not contact_detail:
        return JSONResponse({"error": "Contact not found"}, status_code=404)

    seller_id = contact_detail.get("seller_id")
    seller_block = ""
    if seller_id:
        seller = await get_seller(seller_id)
        if seller:
            seller_block = f"""
SELLER:
- Name: {seller.get('first_name', '')} {seller.get('last_name', '')}
"""
            company_info = (seller.get("company_info") or "").strip()
            if company_info:
                seller_block += f"- Company / what we offer: {company_info}\n"
            review_json = seller.get("performance_review_json")
            if review_json:
                try:
                    review = json.loads(review_json)
                    seller_block += (
                        "- Performance review (summary): "
                        + json.dumps(review, indent=2)
                        + "\n"
                    )
                except Exception:
                    pass

    contact_block = f"""
CONTACT:
- Name: {contact_detail.get('name', '')}
- Company: {contact_detail.get('company') or '(none)'}
- Phone: {contact_detail.get('phone') or '(none)'}
- Status: {contact_detail.get('status', '')}
- Notes: {contact_detail.get('notes') or '(none)'}
- Research: {contact_detail.get('research') or '(none)'}
- Next step (from latest call): {contact_detail.get('next_step') or '(none)'}
"""
    if contact_detail.get("open_hesitations"):
        contact_block += "- Open hesitations:\n"
        for h in contact_detail["open_hesitations"]:
            contact_block += f"  - [{h.get('rank', '')}] {h.get('text', '')}"
            if h.get("resolution_suggestion"):
                contact_block += f" (suggestion: {h['resolution_suggestion']})"
            contact_block += "\n"

    conv_blocks = []
    for conv_summary in contact_detail.get("conversations") or []:
        conv_id = conv_summary.get("id")
        if not conv_id:
            continue
        conv = await get_conversation(conv_id)
        if not conv:
            continue
        parts = [
            f"--- Conversation #{conv_id} ({conv.get('started_at', '?')})",
            f"Close score: {conv.get('final_close_score')}%",
            f"Steps to close: {conv.get('final_steps_to_close')}",
        ]
        transcript = conv.get("transcript") or []
        for t in transcript:
            parts.append(f"  Turn {t.get('turn', '?')}: {t.get('text', '')}")
        coaching = conv.get("coaching") or []
        if coaching:
            parts.append("  Coaching given:")
            for c in coaching:
                parts.append(
                    f"    #{c.get('number')} (turn {c.get('turn')}): {c.get('advice', '')}"
                )
        review = conv.get("review")
        if review and isinstance(review, dict):
            parts.append(f"  Post-call review: {review.get('score')}/10")
            for item in review.get("went_well", []):
                parts.append(f"    + {item}")
            for item in review.get("improve", []):
                parts.append(f"    - {item}")
        conv_blocks.append("\n".join(parts))

    context = (
        "You are a sales coach. Use the following context about the contact, the seller, and their conversations to answer the user's question. Be concise and actionable.\n\n"
        + seller_block
        + contact_block
        + "\nCONVERSATIONS:\n"
        + ("\n\n".join(conv_blocks) if conv_blocks else "\n(No conversations yet.)")
    )
    user_message = context + "\n\n---\n\nUser question: " + body.prompt

    try:
        llm = _get_ask_coach_llm()
        messages = [
            SystemMessage(
                content="You are a sales coach. Answer the user's question using only the context provided. Be specific and actionable; cite the contact or conversation when relevant."
            ),
            HumanMessage(content=user_message),
        ]
        response = await llm.ainvoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
        return {"response": text}
    except Exception as e:
        logger.exception("Ask coach failed")
        return JSONResponse(
            {"error": f"Failed to get coach response: {str(e)}"}, status_code=500
        )


@app.delete("/api/contacts/{contact_id}")
async def api_delete_contact(contact_id: int):
    await delete_contact(contact_id)
    return {"ok": True}


# ── REST API: Home / Sellers ───────────────────────────────────────


@app.get("/api/home")
async def api_home(seller_id: int = 1):
    return await get_home_data(seller_id)


@app.get("/api/sellers")
async def api_list_sellers():
    return await list_sellers()


PERFORMANCE_REVIEW_PROMPT = """\
You are an expert sales manager conducting a performance review for a seller.
Below is the complete history of their sales conversations, including transcripts, \
coaching they received during calls, and post-call reviews.

**Tone:** Address the seller directly. Use their first name (e.g. "Josiah") and write \
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

Assign the score that best matches the seller's typical performance across the conversations provided. If evidence is mixed, weight the majority of calls and the trend (e.g. improving vs. declining)."""


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


@app.post("/api/sellers/{seller_id}/performance-review")
async def api_generate_performance_review(seller_id: int):
    """Aggregate all conversations for the seller and generate a performance review."""
    seller = await get_seller(seller_id)
    if not seller:
        return JSONResponse({"error": "Seller not found"}, status_code=404)

    conversations = await get_all_conversations_for_seller(seller_id)
    if not conversations:
        return {"error": "No conversations found for this seller"}

    first_name = seller.get("first_name") or "there"
    company_info = (seller.get("company_info") or "").strip()

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
            + f"\n\nThe seller's first name is **{first_name}**. Address them directly using this name and second person (you/your)."
        )
        if company_info:
            system_content += f"\n\nContext: The seller represents a company that does the following: {company_info}"
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

    await save_performance_review(seller_id, json.dumps(review_data))
    return review_data


SUGGEST_TODOS_PROMPT = """\
You are a sales coach. Given today's date and, for each contact, their ID, current "next step" \
(from their latest call), and notes, suggest 3-5 actionable tasks for the seller.

For each task, decide the type:
- "email" if the task is to send/reply to an email
- "call" if the task is to make a phone call
- "other" for anything else (research, prep, follow-up, etc.)

Set contact_id to the contact's ID when the task is about a specific contact. \
Leave contact_id null only if the task is general (e.g. "Review your pipeline").

Each title should be one concrete sentence \
(e.g. "Send Kristine 2-3 ROI case studies before Wednesday's call"). \
Prioritize by urgency and deal momentum."""


def _get_suggest_todos_llm() -> ChatOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")
    llm = ChatOpenAI(
        model="google/gemini-2.5-flash",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        max_tokens=1024,
        temperature=0.4,
    )
    return llm.with_structured_output(SuggestedTodos)


@app.post("/api/sellers/{seller_id}/suggest-todos")
async def api_suggest_todos(seller_id: int):
    """Use AI to suggest tasks and save them as todos."""
    from datetime import datetime, timezone

    seller = await get_seller(seller_id)
    if not seller:
        return JSONResponse({"error": "Seller not found"}, status_code=404)

    next_steps = await get_next_steps_for_seller(seller_id)
    today = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")

    if not next_steps:
        return {"created": 0, "todos": []}

    # Build context with contact IDs and optional seller company context
    user_parts = [f"Today's date: {today}"]
    company_info = (seller.get("company_info") or "").strip()
    if company_info:
        user_parts.append("")
        user_parts.append("What the seller's company offers:")
        user_parts.append(company_info)
    user_parts.append("")
    user_parts.append("Contacts (id, next_step, notes):")
    lines = []
    valid_contact_ids = set()
    for s in next_steps:
        valid_contact_ids.add(s["id"])
        line = (
            f"- [id={s['id']}] {s['name']}"
            + (f" (@ {s['company']})" if s.get("company") else "")
            + f": next_step = {s['next_step']}"
        )
        if s.get("notes") and (notes := (s["notes"] or "").strip()):
            line += f"; notes = {notes}"
        lines.append(line)
    user_parts.append("\n".join(lines))
    user_content = "\n".join(user_parts)

    try:
        structured_llm = _get_suggest_todos_llm()
        messages = [
            SystemMessage(content=SUGGEST_TODOS_PROMPT),
            HumanMessage(content=user_content),
        ]
        result: SuggestedTodos = await structured_llm.ainvoke(messages)
    except Exception as e:
        logger.exception("Suggest todos failed")
        return JSONResponse(
            {"error": f"Failed to generate suggestions: {str(e)}"}, status_code=500
        )

    # Save each suggestion as a todo
    created_ids = []
    for item in result.items[:5]:
        todo_type = item.type if item.type in ("email", "call", "other") else "other"
        # Validate contact_id — only allow IDs that were in the context
        cid = item.contact_id if item.contact_id in valid_contact_ids else None
        todo_id = await create_todo(seller_id, todo_type, item.title.strip(), cid)
        created_ids.append(todo_id)

    # Return the refreshed list
    todos = await list_todos(seller_id)
    return {"created": len(created_ids), "todos": todos}


# ── Todos ─────────────────────────────────────────────────────────


class CreateTodoBody(BaseModel):
    type: str  # 'email' | 'call' | 'other'
    title: str
    contact_id: int | None = None


@app.get("/api/sellers/{seller_id}/todos")
async def api_list_todos(seller_id: int):
    """List todos for the seller."""
    seller = await get_seller(seller_id)
    if not seller:
        return JSONResponse({"error": "Seller not found"}, status_code=404)
    todos = await list_todos(seller_id)
    return {"todos": todos}


@app.post("/api/sellers/{seller_id}/todos")
async def api_create_todo(seller_id: int, body: CreateTodoBody):
    """Create a todo. type must be email, call, or other."""
    seller = await get_seller(seller_id)
    if not seller:
        return JSONResponse({"error": "Seller not found"}, status_code=404)
    if body.type not in ("email", "call", "other"):
        return JSONResponse(
            {"error": "type must be 'email', 'call', or 'other'"}, status_code=400
        )
    try:
        todo_id = await create_todo(
            seller_id, body.type, body.title.strip(), body.contact_id
        )
        return {"id": todo_id}
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.delete("/api/todos/{todo_id}")
async def api_delete_todo(todo_id: int):
    """Delete a todo."""
    deleted = await delete_todo(todo_id)
    if not deleted:
        return JSONResponse({"error": "Todo not found"}, status_code=404)
    return {"ok": True}


class ColdCallPrepRequest(BaseModel):
    contact_name: str
    company: str | None = None
    seller_id: int | None = None


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
Tell me just enough information to cold call this business and offer our services. \
Use the seller's company description (what we offer) below to tailor your answer. \
Be concise: a few bullet points or short paragraphs. Include what the business does, any relevant recent news or context, and why they might care about what we offer. \
Cite sources using markdown links if the model provided them."""


@app.post("/api/cold-call-prep")
async def api_cold_call_prep(body: ColdCallPrepRequest):
    """Web-search-backed cold call prep for a contact's business. Uses OpenRouter :online."""
    if not body.contact_name.strip():
        return JSONResponse({"error": "contact_name is required"}, status_code=400)
    user_parts = []
    if body.seller_id:
        seller = await get_seller(body.seller_id)
        if seller and (seller.get("company_info") or "").strip():
            user_parts.append(
                "What we offer (seller's company):\n"
                + (seller["company_info"] or "").strip()
            )
            user_parts.append("")
    user_parts.append(
        f"Contact name: {body.contact_name.strip()}\nBusiness: {(body.company or '').strip() or '(not specified)'}"
    )
    user_content = "\n".join(user_parts)
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
You are a sales coach. Given a contact's notes, outstanding hesitations (objections/concerns), past conversations with this contact, the seller's name, and the seller's latest performance review, write a short pre-call preparation brief.

Include:
1. Key context from the contact's notes.
2. Any outstanding hesitations and how to address them.
3. What happened on past calls (next steps, scores, any pattern).
4. 2–3 suggested goals or talking points for this call, tailored to this contact and to the seller's coaching recommendations where relevant.

Be concise and actionable. Write in second person ("You...") addressing the seller. Use bullet points or short paragraphs."""


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


def _build_prepare_user_content(contact: dict, seller: dict) -> tuple[str, str]:
    """Build the user content and notes_str for prepare. Returns (user_content, notes_str)."""
    notes_str = (contact.get("notes") or "").strip()
    parts = []
    parts.append(
        f"Seller: {seller.get('first_name', '')} {seller.get('last_name', '')}".strip()
    )
    company_info = (seller.get("company_info") or "").strip()
    if company_info:
        parts.append("WHAT WE OFFER (seller's company):")
        parts.append(company_info)
        parts.append("")
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

    review_json = seller.get("performance_review_json")
    if review_json:
        try:
            review = json.loads(review_json)
            parts.append("SELLER'S LATEST PERFORMANCE REVIEW:")
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
        parts.append("SELLER'S LATEST PERFORMANCE REVIEW: (none)")

    return ("\n".join(parts), notes_str)


async def _generate_prepare_brief(contact_id: int) -> tuple[str, str]:
    """Return (contact_notes, call_prep) for a contact. On failure returns (notes, '')."""
    contact = await get_contact_detail(contact_id)
    if not contact:
        return ("", "")
    notes_str = (contact.get("notes") or "").strip()
    seller_id = contact.get("seller_id")
    if not seller_id:
        return (notes_str, "")
    seller = await get_seller(seller_id)
    if not seller:
        return (notes_str, "")
    user_content, notes_str = _build_prepare_user_content(contact, seller)
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


async def _stream_prepare_brief(contact_id: int) -> AsyncGenerator[bytes, None]:
    """Stream LLM tokens for the prepare brief. Yields UTF-8 bytes."""
    contact = await get_contact_detail(contact_id)
    if not contact:
        return
    seller_id = contact.get("seller_id")
    if not seller_id:
        return
    seller = await get_seller(seller_id)
    if not seller:
        return
    user_content, _ = _build_prepare_user_content(contact, seller)
    llm = _get_prepare_llm()
    messages = [
        SystemMessage(content=PREPARE_PROMPT),
        HumanMessage(content=user_content),
    ]
    async for chunk in llm.astream(messages):
        content = getattr(chunk, "content", None)
        if content:
            yield content.encode("utf-8")


@app.get("/api/contacts/{contact_id}/prepare")
async def api_contact_prepare(contact_id: int):
    """Generate a pre-call preparation brief from contact notes, past conversations, rep, and last performance review."""
    contact = await get_contact_detail(contact_id)
    if not contact:
        return JSONResponse({"error": "Contact not found"}, status_code=404)
    if not contact.get("seller_id"):
        return JSONResponse(
            {"error": "Contact has no assigned seller"}, status_code=400
        )
    try:
        _, content = await _generate_prepare_brief(contact_id)
        return {"content": content}
    except Exception as e:
        logger.exception("Prepare brief failed")
        return JSONResponse({"error": f"Prepare failed: {str(e)}"}, status_code=500)


@app.get("/api/contacts/{contact_id}/prepare/stream")
async def api_contact_prepare_stream(contact_id: int):
    """Stream the pre-call preparation brief (LLM output) as plain text."""
    contact = await get_contact_detail(contact_id)
    if not contact:
        return JSONResponse({"error": "Contact not found"}, status_code=404)
    if not contact.get("seller_id"):
        return JSONResponse(
            {"error": "Contact has no assigned seller"}, status_code=400
        )
    return StreamingResponse(
        _stream_prepare_brief(contact_id),
        media_type="text/plain; charset=utf-8",
    )


# ── REST API: Conversations ──────────────────────────────────────


@app.get("/api/conversations")
async def api_list_conversations(contact_id: int | None = None, limit: int = 50):
    return await list_conversations(limit=limit, contact_id=contact_id)


@app.post("/api/conversations")
async def api_create_conversation(body: dict):
    """Create a new conversation (e.g. for live call). Optionally link contact and generate prep."""
    contact_id = body.get("contact_id")
    if contact_id is not None:
        contact_id = int(contact_id)
    prep_notes = None
    if contact_id:
        try:
            _, prep_notes = await _generate_prepare_brief(contact_id)
        except Exception as e:
            logger.warning("Could not generate prep for conversation: %s", e)
    cid = await create_conversation(
        contact_id=contact_id,
        mode="live",
        prep_notes=prep_notes or None,
    )
    return {"id": cid}


@app.get("/api/conversations/{conversation_id}")
async def api_get_conversation(conversation_id: int):
    conv = await get_conversation(conversation_id)
    if not conv:
        return JSONResponse({"error": "not found"}, status_code=404)
    conv["hesitations"] = await get_hesitations(conversation_id=conversation_id)
    # Normalize legacy coaching: advice may be stored as raw JSON + markdown
    for c in conv.get("coaching") or []:
        if isinstance(c, dict) and c.get("advice"):
            c["advice"] = _sanitize_advice(str(c["advice"]))
    return conv


@app.patch("/api/conversations/{conversation_id}")
async def api_patch_conversation(conversation_id: int, body: dict):
    """Update conversation fields (e.g. prep_notes)."""
    prep_notes = body.get("prep_notes")
    if prep_notes is not None:
        await update_conversation_prep_notes(conversation_id, prep_notes)
    return {"ok": True}


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
    company_info = None
    contact_id = conv.get("contact_id")
    if contact_id:
        contact = await get_contact(contact_id)
        if contact and contact.get("seller_id"):
            seller = await get_seller(contact["seller_id"])
            if seller:
                company_info = (seller.get("company_info") or "").strip() or None
    review = await generate_review(
        transcript=conv["transcript"],
        coaching=conv["coaching"],
        company_info=company_info,
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


# ── Coaching LLM calls ────────────────────────────────────────────


def _coaching_advice_user_content(
    conversation: list[str],
    custom_question: str | None,
    contact_notes: str = "",
    call_prep: str = "",
) -> str:
    """Build user message for advice-only prompt."""
    recent = conversation[-15:]
    conv_text = "\n".join(f"Turn {i + 1}: {t}" for i, t in enumerate(recent))
    out = ""
    if contact_notes:
        out += f"CONTACT NOTES:\n{contact_notes}\n\n"
    if call_prep:
        out += f"PRE-CALL PREP:\n{call_prep}\n\n"
    out += f"TRANSCRIPT (recent):\n{conv_text}\n\n"
    if custom_question:
        out += f"REP ASKS:\n{custom_question}\n\n"
    out += "One short line only: question to ask or 3–7 word hint (slip of paper)."
    return out


async def get_coaching_advice_only(
    conversation: list[str],
    custom_question: str | None = None,
    contact_notes: str = "",
    call_prep: str = "",
) -> dict:
    """Low-latency: return only advice (one line). No scores or objections."""
    user_content = _coaching_advice_user_content(
        conversation, custom_question, contact_notes, call_prep
    )
    structured_llm = _get_coaching_advice_llm()
    messages = [
        SystemMessage(content=ADVICE_ONLY_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
    response: CoachingAdviceOnly = await structured_llm.ainvoke(messages)
    return {"advice": response.advice}


def _scores_user_content(
    conversation: list[str],
    objections: list[dict],
    contact_notes: str = "",
    call_prep: str = "",
) -> str:
    """Build user message for scores/objections prompt."""
    recent = conversation[-15:]
    conv_text = "\n".join(f"Turn {i + 1}: {t}" for i, t in enumerate(recent))
    out = ""
    if contact_notes:
        out += f"CONTACT NOTES:\n{contact_notes}\n\n"
    if call_prep:
        out += f"PRE-CALL PREP:\n{call_prep}\n\n"
    if objections:
        obj_summary = "\n".join(
            f"- [{o.get('status', 'open').upper()}] {o.get('text', '')}"
            for o in objections
        )
        out += f"CURRENT OBJECTIONS:\n{obj_summary}\n\n"
    out += f"TRANSCRIPT (recent):\n{conv_text}\n\n"
    out += "Provide close_score (0-100), steps_to_close (1-10), and updated objections list."
    return out


async def get_coaching_scores_and_objections(
    conversation: list[str],
    objections: list[dict],
    contact_notes: str = "",
    call_prep: str = "",
) -> dict:
    """On-demand: return close_score, steps_to_close, objections. No advice."""
    user_content = _scores_user_content(
        conversation, objections, contact_notes, call_prep
    )
    structured_llm = _get_coaching_scores_llm()
    messages = [
        SystemMessage(content=SCORES_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
    response: CoachingScoresResponse = await structured_llm.ainvoke(messages)
    return response.model_dump()


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


def _get_coaching_advice_llm() -> ChatOpenAI:
    """LangChain client for advice-only (low latency)."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")
    llm = ChatOpenAI(
        model="anthropic/claude-haiku-4-5",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        max_tokens=256,
    )
    return llm.with_structured_output(CoachingAdviceOnly)


def _get_coaching_scores_llm() -> ChatOpenAI:
    """LangChain client for scores + objections (on-demand)."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")
    llm = ChatOpenAI(
        model="anthropic/claude-haiku-4-5",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        max_tokens=2048,
    )
    return llm.with_structured_output(CoachingScoresResponse)


async def generate_review(
    transcript: list[dict],
    coaching: list[dict],
    company_info: str | None = None,
) -> dict:
    conv_text = "\n".join(
        f"Turn {t.get('turn', i+1)}: {t.get('text', '')}"
        for i, t in enumerate(transcript)
    )
    coach_text = "\n".join(
        f"Coaching #{c.get('number', i+1)} (turn {c.get('turn', '?')}): {c.get('advice', '')}"
        for i, c in enumerate(coaching)
    )

    user_content = ""
    if (company_info or "").strip():
        user_content += (
            f"CONTEXT (what the seller's company offers):\n{company_info.strip()}\n\n"
        )
    user_content += f"FULL TRANSCRIPT:\n{conv_text}\n\n"
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
        self.last_turn_end_at: float | None = (
            None  # perf_counter() when turn ended (for latency_ms)
        )

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
        *,
        latency_ms: int | None = None,
    ):
        self.coaching_log.append(
            {
                "number": number,
                "turn": turn,
                "advice": advice,
                "close_score": close_score,
                "steps_to_close": steps_to_close,
                "is_custom": is_custom,
                "latency_ms": latency_ms,
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

    session = SessionData()

    # First message: config (or resume with conversation_id)
    try:
        init_msg = await ws.receive_json()
        auto_coach = init_msg.get("auto_coach", True)
        contact_id = init_msg.get("contact_id")
        if contact_id:
            session.contact_id = int(contact_id)
        resume_id = init_msg.get("conversation_id")
    except Exception:
        auto_coach = True
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
        # Generate prepare brief first so we can store it on the conversation
        if session.contact_id:
            try:
                (
                    session.contact_notes,
                    session.call_prep,
                ) = await _generate_prepare_brief(session.contact_id)
            except Exception as e:
                logger.warning("Could not load contact notes/prep for coach: %s", e)
        session.conversation_id = await create_conversation(
            contact_id=session.contact_id,
            mode="live",
            prep_notes=session.call_prep or None,
        )
    elif session.contact_id:
        # Resuming: load contact notes and prep for coach (not stored on conversation)
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
    pause_requested = False

    def on_dg_message(message: Any) -> None:
        if running:
            msg_queue.put_nowait(message)

    async def do_coaching(turn_num: int, custom_question: str | None = None) -> None:
        nonlocal coaching_count
        t0 = (
            session.last_turn_end_at
            if session.last_turn_end_at is not None
            else time.perf_counter()
        )
        try:
            await ws.send_json({"type": "coaching_started"})
            result = await get_coaching_advice_only(
                conversation,
                custom_question,
                session.contact_notes,
                session.call_prep,
            )
            coaching_count += 1
            advice = _sanitize_advice(result.get("advice", "") or "")
            latency_ms = round((time.perf_counter() - t0) * 1000)

            session.add_coaching(
                coaching_count,
                turn_num,
                advice,
                -1,
                -1,
                custom_question is not None,
                latency_ms=latency_ms,
            )

            payload = {
                "type": "coaching",
                "advice": advice,
                "number": coaching_count,
                "turn": turn_num,
                "is_custom": custom_question is not None,
                "latency_ms": latency_ms,
            }
            await ws.send_json(payload)
            logger.info(
                "coaching_latency conversation_id=%s turn=%s latency_ms=%s",
                session.conversation_id,
                turn_num,
                latency_ms,
            )
        except Exception as e:
            await ws.send_json({"type": "error", "message": str(e)})

    async def do_refresh_scores() -> None:
        nonlocal objections
        try:
            result = await get_coaching_scores_and_objections(
                conversation,
                objections,
                session.contact_notes,
                session.call_prep,
            )
            objections = result.get("objections", [])
            if objections:
                session.objections = objections
            await ws.send_json(
                {
                    "type": "scores",
                    "close_score": result.get("close_score", -1),
                    "steps_to_close": result.get("steps_to_close", -1),
                    "objections": objections,
                }
            )
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
            elif cmd_type == "refresh_scores":
                asyncio.create_task(do_refresh_scores())
            elif cmd_type == "set_auto_coach":
                nonlocal auto_coach
                auto_coach = cmd.get("value", True)

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
                        session.last_turn_end_at = time.perf_counter()
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
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
                    except asyncio.TimeoutError:
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
            company_info = None
            if session.contact_id:
                contact = await get_contact(session.contact_id)
                if contact and contact.get("seller_id"):
                    seller = await get_seller(contact["seller_id"])
                    if seller:
                        company_info = (
                            seller.get("company_info") or ""
                        ).strip() or None
            try:
                review = await generate_review(
                    transcript=session.turns,
                    coaching=session.coaching_log,
                    company_info=company_info,
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

    session = SessionData()

    # First message: config — either conversation_id (load from DB) or conversation (raw text)
    try:
        init_msg = await ws.receive_json()
        conversation_text = init_msg.get("conversation", "")
        conversation_id_load = init_msg.get("conversation_id")
        contact_id = init_msg.get("contact_id")
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

    # Generate prepare brief first so we can store it on the conversation
    if session.contact_id:
        try:
            session.contact_notes, session.call_prep = await _generate_prepare_brief(
                session.contact_id
            )
        except Exception as e:
            logger.warning("Could not load contact notes/prep for test coach: %s", e)
    # Create conversation record (turns_data may be empty for manual-turn mode)
    session.conversation_id = await create_conversation(
        contact_id=session.contact_id,
        mode="test",
        prep_notes=session.call_prep if session.contact_id else None,
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
                t0_coach = time.perf_counter()
                await ws.send_json({"type": "coaching_started"})
                try:
                    result = await get_coaching_advice_only(
                        conversation,
                        custom_question,
                        session.contact_notes,
                        session.call_prep,
                    )
                    coaching_count += 1
                    advice = _sanitize_advice(result.get("advice", "") or "")
                    latency_ms = round((time.perf_counter() - t0_coach) * 1000)
                    session.add_coaching(
                        coaching_count,
                        up_to,
                        advice,
                        -1,
                        -1,
                        custom_question is not None,
                        latency_ms=latency_ms,
                    )
                    payload = {
                        "type": "coaching",
                        "advice": advice,
                        "number": coaching_count,
                        "turn": up_to,
                        "is_custom": custom_question is not None,
                        "latency_ms": latency_ms,
                    }
                    await ws.send_json(payload)
                    logger.info(
                        "coaching_latency conversation_id=%s turn=%s latency_ms=%s",
                        session.conversation_id,
                        up_to,
                        latency_ms,
                    )
                except Exception as e:
                    await ws.send_json({"type": "error", "message": str(e)})

            elif cmd_type == "refresh_scores":
                try:
                    conversation = [t["text"] for t in turns_data]
                    result = await get_coaching_scores_and_objections(
                        conversation,
                        objections,
                        session.contact_notes,
                        session.call_prep,
                    )
                    objections = result.get("objections", [])
                    if objections:
                        session.objections = objections
                    await ws.send_json(
                        {
                            "type": "scores",
                            "close_score": result.get("close_score", -1),
                            "steps_to_close": result.get("steps_to_close", -1),
                            "objections": objections,
                        }
                    )
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
                    session.last_turn_end_at = time.perf_counter()
                    await ws.send_json(
                        {
                            "type": "turn_end",
                            "text": text,
                            "turn": turn_num,
                            "confidence": 1.0,
                        }
                    )
                    # Auto-coach on the new turn (advice only)
                    t0_coach = session.last_turn_end_at
                    await ws.send_json({"type": "coaching_started"})
                    try:
                        result = await get_coaching_advice_only(
                            conversation,
                            None,
                            session.contact_notes,
                            session.call_prep,
                        )
                        coaching_count += 1
                        advice = _sanitize_advice(result.get("advice", "") or "")
                        latency_ms = round((time.perf_counter() - t0_coach) * 1000)
                        session.add_coaching(
                            coaching_count,
                            turn_num,
                            advice,
                            -1,
                            -1,
                            False,
                            latency_ms=latency_ms,
                        )
                        payload = {
                            "type": "coaching",
                            "advice": advice,
                            "number": coaching_count,
                            "turn": turn_num,
                            "is_custom": False,
                            "latency_ms": latency_ms,
                        }
                        await ws.send_json(payload)
                        logger.info(
                            "coaching_latency conversation_id=%s turn=%s latency_ms=%s",
                            session.conversation_id,
                            turn_num,
                            latency_ms,
                        )
                    except Exception as e:
                        await ws.send_json({"type": "error", "message": str(e)})

            elif cmd_type == "generate_review":
                try:
                    company_info = None
                    if session.contact_id:
                        contact = await get_contact(session.contact_id)
                        if contact and contact.get("seller_id"):
                            seller = await get_seller(contact["seller_id"])
                            if seller:
                                company_info = (
                                    seller.get("company_info") or ""
                                ).strip() or None
                    review = await generate_review(
                        transcript=session.turns,
                        coaching=session.coaching_log,
                        company_info=company_info,
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
