"""Propose N short-form segments from an interview transcript (hook + body + CTA).

Uses word-level and utterance-level transcripts to suggest timestamp ranges
for each short. Output format matches shorts_cuts.json consumed by
longform_to_clips and apply_cuts.

Duration/tolerance work like propose_cuts: e.g. duration=35, tolerance=15
→ acceptable range 20–50s per short.
"""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, TypeVar

from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

DEFAULT_NUM_SHORTS = 10
DEFAULT_TARGET_DURATION = 35  # seconds
DEFAULT_TOLERANCE = 15  # ± seconds (e.g. 35±15 → 20–50s)
RETRY_TOLERANCE = 5  # only retry if duration is > this many seconds outside target
STREAMING_MODEL = (
    "anthropic/claude-opus-4.6"  # use Opus for stream_thinking (shows reasoning)
)


def _lc_messages_to_openai(messages: list[BaseMessage]) -> list[dict[str, str]]:
    """Convert LangChain messages to OpenAI API format."""
    out: list[dict[str, str]] = []
    for m in messages:
        role = "user"
        if isinstance(m, SystemMessage):
            role = "system"
        elif hasattr(m, "type") and m.type == "ai":
            role = "assistant"
        content = m.content if hasattr(m, "content") else str(m)
        out.append({"role": role, "content": content})
    return out


T = TypeVar("T", bound=BaseModel)


def _stream_structured_invoke(
    messages: list[BaseMessage],
    response_model: type[T],
    model_id: str = STREAMING_MODEL,
) -> T:
    """Stream reasoning + content to stdout, then parse JSON into Pydantic model."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    openai_messages = _lc_messages_to_openai(messages)
    schema = response_model.model_json_schema()
    openai_messages[-1]["content"] += (
        f"\n\nRespond with valid JSON only that matches this schema. "
        f"No markdown, no explanation. Schema: {json.dumps(schema)}"
    )
    stream = client.chat.completions.create(
        model=model_id,
        messages=openai_messages,
        max_tokens=16000,
        extra_body={"reasoning": {"max_tokens": 8000}},
        stream=True,
    )
    content_parts: list[str] = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if hasattr(delta, "reasoning_details") and delta.reasoning_details:
            for rd in delta.reasoning_details:
                if isinstance(rd, dict) and rd.get("type") == "reasoning.text":
                    t = rd.get("text", "")
                    if t:
                        print(t, end="", flush=True)
        elif getattr(delta, "content", None):
            c = delta.content
            content_parts.append(c)
            print(c, end="", flush=True)
    print()  # newline after stream
    raw = "".join(content_parts).strip()
    # strip ```json ... ``` if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw)
    return response_model.model_validate(data)


def _get_llm(model: str | None = None, stream_thinking: bool = False) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")
    model_id = model or os.getenv(
        "PROPOSE_INTERVIEW_MODEL", "google/gemini-3-flash-preview"
    )
    extra_body: dict[str, Any] = {"reasoning": {"effort": "medium"}}
    if stream_thinking:
        extra_body["include_reasoning"] = True
    kwargs: dict[str, Any] = {
        "model": model_id,
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": api_key,
        "extra_body": extra_body,
        "streaming": stream_thinking,
    }
    if stream_thinking:
        kwargs["callbacks"] = [StreamingStdOutCallbackHandler()]
    return ChatOpenAI(**kwargs)


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------


class ShortSegment(BaseModel):
    """A single short video segment (hook + body; CTA comes from proposal)."""

    segment: int = Field(description="Segment number (1-based)")
    hook_text: str = Field(
        description="EXACT verbatim phrase from the transcript for the hook."
    )
    body_phrases: list[str] = Field(
        description=(
            "EXACT verbatim phrase(s) from the transcript for the body. "
            "Each item is one contiguous run of words copied exactly from the transcript. "
            "Use multiple items to skip over tangents or repetitions (non-contiguous cuts)."
        )
    )


class ShortsProposal(BaseModel):
    """Complete proposal for N shorts from an interview."""

    number_questions: int = Field(
        description="Number of distinct questions or topics the interviewee answered"
    )
    cta_text: str = Field(
        description="EXACT verbatim phrase from the transcript for the CTA."
    )
    shorts: list[ShortSegment] = Field(
        description="List of short segments, each with hook and body"
    )


class CritiqueEdit(BaseModel):
    """Critique + applied edit for one short."""

    segment: int = Field(description="Segment number (1-based)")
    hook_text: str = Field(
        description="EXACT verbatim phrase from transcript for the updated hook"
    )
    body_phrases: list[str] = Field(
        description=(
            "EXACT verbatim phrase(s) from transcript for the updated body. "
            "Each item is one contiguous run of words. "
            "Use multiple items for non-contiguous cuts."
        )
    )


class CritiqueRefinement(BaseModel):
    """Critique pass that also applies edits to produce revised cuts."""

    cta_text: str = Field(
        description="EXACT verbatim phrase from transcript for the shared CTA"
    )
    edits: list[CritiqueEdit] = Field(description="Per-short edits with revised text")


# ---------------------------------------------------------------------------
# Deterministic text → timestamp snapping
# ---------------------------------------------------------------------------
_WORD_END_MARGIN = 0  # no padding at end; use exact word boundary


def _normalize_text(text: str) -> str:
    """Normalize text for robust string comparison."""
    t = (text or "").lower()
    t = t.replace("zero", "0")
    t = re.sub(r"[^a-z0-9%']+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _build_norm_tokens(words: list[dict]) -> list[tuple[int, str]]:
    """Pre-compute (word_index, normalized_token) pairs, skipping empties."""
    return [
        (i, n) for i, w in enumerate(words) if (n := _normalize_text(w.get("word", "")))
    ]


def _snap_phrase_to_words(
    phrase: str,
    words: list[dict],
    norm_tokens: list[tuple[int, str]],
) -> tuple[float, float, str, float] | None:
    """Find the best contiguous word span matching *phrase* in the transcript.

    Tries exact token-sequence match first, then falls back to
    rapidfuzz partial_ratio_alignment on token lists.

    Returns ``(start_ts, end_ts, verbatim_text, confidence)`` or ``None``.
    """
    phrase_norm = _normalize_text(phrase)
    if not phrase_norm:
        return None
    phrase_toks = phrase_norm.split()
    n = len(phrase_toks)
    if n == 0:
        return None
    tok_list = [t for _, t in norm_tokens]
    h = len(tok_list)
    if h < n:
        return None

    def _result(start_tok: int, end_tok: int, conf: float):
        ws = norm_tokens[start_tok][0]
        we = norm_tokens[end_tok][0]
        return (
            float(words[ws].get("start", 0)),
            float(words[we].get("end", 0)) + _WORD_END_MARGIN,
            " ".join(words[j].get("word", "") for j in range(ws, we + 1)).strip(),
            conf,
        )

    # --- 1) exact token-sequence match (first occurrence) ---
    for i in range(h - n + 1):
        if tok_list[i : i + n] == phrase_toks:
            return _result(i, i + n - 1, 1.0)

    # --- 2) fuzzy match via rapidfuzz (token-level) ---
    alignment = fuzz.partial_ratio_alignment(phrase_toks, tok_list)
    if alignment is None:
        return None
    score = alignment.score / 100.0
    # dest_start/dest_end are indices into tok_list (end is exclusive)
    start_idx = alignment.dest_start
    end_idx = max(alignment.dest_end - 1, start_idx)
    if score < 1.0:
        print(f"   🔧 fuzzy snap ({score:.2f}): '{phrase[:60]}…'")
    return _result(start_idx, end_idx, score)


def _ranges_duration(ranges_str: str) -> float:
    """Parse 'start:end' or 'start:end,start:end,...' and return total duration."""
    total = 0.0
    for part in ranges_str.split(","):
        pieces = part.strip().split(":")
        if len(pieces) == 2:
            try:
                total += float(pieces[1]) - float(pieces[0])
            except ValueError:
                pass
    return total


def _snap_segment(
    hook_text: str,
    body_phrases: list[str],
    cta_text: str,
    words: list[dict],
    norm_tokens: list[tuple[int, str]],
) -> dict[str, Any]:
    """Snap LLM-produced text to deterministic transcript timestamps."""

    def _from_snap(
        snap: tuple[float, float, str, float] | None,
        raw: str,
    ) -> tuple[str, str, float]:
        if snap:
            return f"{snap[0]:.2f}:{snap[1]:.2f}", snap[2], snap[3]
        return "", raw, 0.0

    hook = _snap_phrase_to_words(hook_text, words, norm_tokens)
    cta = _snap_phrase_to_words(cta_text, words, norm_tokens)
    body_snaps = [_snap_phrase_to_words(bp, words, norm_tokens) for bp in body_phrases]

    h_range, h_text, h_conf = _from_snap(hook, hook_text)
    c_range, c_text, c_conf = _from_snap(cta, cta_text)

    b_ranges: list[str] = []
    b_texts: list[str] = []
    b_confs: list[float] = []
    for i, snap in enumerate(body_snaps):
        raw = body_phrases[i] if i < len(body_phrases) else ""
        r, t, c = _from_snap(snap, raw)
        if r:
            b_ranges.append(r)
        b_texts.append(t)
        b_confs.append(c)

    all_confs = [h_conf, c_conf] + b_confs
    return {
        "hook_range": h_range,
        "hook_text": h_text,
        "body_ranges": ",".join(b_ranges),
        "body_phrases": b_texts,
        "cta_range": c_range,
        "cta_text": c_text,
        "min_confidence": min(all_confs) if all_confs else 0.0,
    }


# ---------------------------------------------------------------------------
# Main proposal pipeline
# ---------------------------------------------------------------------------


async def propose_interview_cuts(
    words_json: Path,
    utterances_json: Path,
    num_shorts: int | None = None,
    cuts_per_question: int | None = None,
    duration: float = DEFAULT_TARGET_DURATION,
    tolerance: float = DEFAULT_TOLERANCE,
    allow_hook_reuse: bool = True,
    stream_thinking: bool = False,
) -> list[dict[str, Any]]:
    """Use LLM to analyse transcript and propose short segments.

    Returns list of segment dicts in shorts_cuts.json format.
    """
    if cuts_per_question is None and num_shorts is None:
        num_shorts = DEFAULT_NUM_SHORTS
    words: list[dict] = json.loads(words_json.read_text())
    utterances: list[dict] = json.loads(utterances_json.read_text())

    transcript_for_llm = ""
    for utt in utterances:
        text = utt.get("text", "") or utt.get("transcript", "")
        transcript_for_llm += f"{text}\n\n"

    if words:
        total_duration = float(words[-1].get("end", 0))
    else:
        total_duration = float(utterances[-1].get("end", 0)) if utterances else 0
    total_words_count = len(words)
    avg_wpm = (total_words_count / total_duration) * 60 if total_duration > 0 else 150

    llm = _get_llm(stream_thinking=stream_thinking)
    min_dur = max(0, duration - tolerance)
    max_dur = duration + tolerance
    retry_min = min_dur - RETRY_TOLERANCE
    retry_max = max_dur + RETRY_TOLERANCE
    min_words = int(min_dur * avg_wpm / 60)
    max_words = int(max_dur * avg_wpm / 60)

    use_cuts_per_question = cuts_per_question is not None
    if use_cuts_per_question:
        num_shorts_instruction = (
            f"First identify how many distinct questions or topics the interviewee answered "
            f"(number_questions). Then propose number_questions × {cuts_per_question} = "
            f"(number_questions * {cuts_per_question}) short videos."
        )
        target_desc = f"{duration:.0f}s each (±{tolerance:.0f}s), {cuts_per_question} cut(s) per question/topic"
    else:
        num_shorts_instruction = f"identify {num_shorts} short videos"
        target_desc = f"{num_shorts} shorts, {duration:.0f}s each (±{tolerance:.0f}s)"

    print(
        f"🧠 Analysing transcript ({len(utterances)} utterances, {total_duration:.0f}s, "
        f"~{avg_wpm:.0f} WPM)..."
    )
    print(f"   Target: {target_desc} → ~{min_words}-{max_words} words per short")
    if stream_thinking:
        print("   Streaming model output to stdout...\n")

    system_content = (
        "You are an expert video editor who creates viral short-form content "
        "from long interviews.\n\n"
        "You'll receive an interview transcript. Your job is to "
        f"{num_shorts_instruction}, each structured as:\n"
        "  [HOOK] — quick attention-grabbing statement\n"
        "  [BODY] — substantive answer/insight\n"
        "  [CTA]  — call-to-action (shared across all shorts)\n\n"
        "TEXT RULES — CRITICAL:\n"
        "- hook_text, body_phrases, cta_text must be EXACT verbatim phrases from the transcript.\n"
        "- We do deterministic string lookup to derive timestamps, so text must match exactly.\n"
        "- Copy-paste the exact words — no paraphrasing, no added/removed punctuation.\n"
        "- body_phrases is a list — each item is one contiguous run of words from the transcript.\n"
        "- Use multiple items in body_phrases to skip over tangents or repetitions.\n"
        "- Don't start a body phrase with filler words like 'But', 'So', 'Um'.\n\n"
        "INTERVIEW STRUCTURE:\n"
        "- The interviewee answered several questions on different topics\n"
        "- They then recorded multiple hook takes (short, punchy statements)\n"
        "- At the end, they recorded one or more CTA takes\n\n"
        "YOUR PROCESS:\n"
        "1. Find the CTA — usually near the end. Pick the best/cleanest take. "
        "   ALL shorts share the same CTA.\n"
        "2. Identify body segments — substantive answers covering ONE clear topic each.\n"
        "3. Match hooks to bodies — attention-grabbing openings from:\n"
        "   - Dedicated hook takes (short, punchy statements)\n"
        "   - Strong opening lines from other answers\n"
        "   - The same answer's first sentence (if compelling)\n"
        + (
            "4. A hook CAN be reused across 2 shorts if it fits both bodies.\n\n"
            if allow_hook_reuse
            else "4. Do NOT reuse the same hook; each short must have a distinct hook.\n\n"
        )
        + f"WORD COUNT GUIDE: The speaker averages ~{avg_wpm:.0f} words per minute. "
        f"Each short should total approximately {min_words}-{max_words} words "
        f"(hook + body + CTA combined) to hit {min_dur:.0f}-{max_dur:.0f}s. "
        "Hooks are typically 5-15 words, CTAs 10-25 words, "
        f"so body should be roughly {max(5, min_words - 30)}-{max_words - 15} words. "
        "If in doubt, err shorter — we can always extend.\n\n"
        "NO REPETITION: Interviewees often restate the same point. Include only ONE "
        "clear statement of each point per short. Use multiple items in body_phrases "
        "to jump over repeated phrases or tangents. Choose the punchiest version.\n\n"
        "QUALITY CRITERIA:\n"
        "- Each body = one complete thought, no mid-sentence cuts; prefer tight over comprehensive\n"
        "- Hooks: curiosity, surprise, bold claims\n"
        "- Avoid overlapping content between shorts; vary topics"
    )

    human_base = (
        f"Here is the interview transcript ({total_words_count} words):\n\n"
        f"{transcript_for_llm}\n\n"
        + (
            f"Identify number_questions and propose number_questions × {cuts_per_question} shorts."
            if use_cuts_per_question
            else f"Please propose {num_shorts} shorts."
        )
        + f" Each short should be approximately {min_words}-{max_words} words total."
    )

    norm_tokens = _build_norm_tokens(words)

    # ------------------------------------------------------------------
    # STEP 1: Propose shorts
    # ------------------------------------------------------------------
    response = None
    snapped_proposal: list[dict[str, Any]] = []
    for attempt in range(8):
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_base),
        ]
        if attempt > 0:
            messages.append(
                HumanMessage(
                    content=(
                        f"RETRY {attempt}: Fix the issues below and re-propose. "
                        "Text fields must be EXACT verbatim phrases from the transcript."
                    )
                )
            )
        if stream_thinking:
            if attempt == 0:
                print("\n--- Proposal (streaming) ---\n")
            response = await asyncio.to_thread(
                _stream_structured_invoke, messages, ShortsProposal
            )
        else:
            structured = llm.with_structured_output(ShortsProposal)
            response = await structured.ainvoke(messages)

        snap_failures: list[str] = []
        out_of_range: list[tuple[int, float]] = []
        snapped_proposal = []
        for s in response.shorts:
            snapped = _snap_segment(
                s.hook_text,
                s.body_phrases,
                response.cta_text,
                words,
                norm_tokens,
            )
            snapped_proposal.append(snapped)
            total = _ranges_duration(
                f"{snapped['hook_range']},{snapped['body_ranges']},{snapped['cta_range']}"
            )
            if total < retry_min or total > retry_max:
                out_of_range.append((s.segment, total))
            if snapped["min_confidence"] < 0.7:
                snap_failures.append(
                    f"Segment {s.segment}: could not locate text in transcript"
                )

        if not out_of_range and not snap_failures:
            break

        feedback: list[str] = []
        if out_of_range:
            dur_detail = ", ".join(f"seg {seg}={dur:.0f}s" for seg, dur in out_of_range)
            feedback.append(
                f"Duration out of {min_dur:.0f}-{max_dur:.0f}s range: {dur_detail}. "
                f"Adjust word count (target {min_words}-{max_words} words)."
            )
        if snap_failures:
            feedback.append(
                "Text not found in transcript (must be EXACT words): "
                + "; ".join(snap_failures[:4])
            )
        print(f"   Attempt {attempt + 1}: {'; '.join(feedback)}. Retrying...")
        human_base += "\n\nFEEDBACK: " + "; ".join(feedback)

    if response is None:
        raise RuntimeError("No valid ShortsProposal response from LLM after retries")

    print(f"\n🔗 Snapped {len(snapped_proposal)} segments to transcript timestamps.")

    # ------------------------------------------------------------------
    # STEP 2: Critique + apply edits
    # ------------------------------------------------------------------
    critique = llm.with_structured_output(CritiqueRefinement)
    proposed_lines = []
    for i, s in enumerate(response.shorts):
        sp = snapped_proposal[i]
        total = _ranges_duration(
            f"{sp['hook_range']},{sp['body_ranges']},{sp['cta_range']}"
        )
        body_word_count = sum(len(p.split()) for p in s.body_phrases)
        word_count = len(s.hook_text.split()) + body_word_count
        body_display = " | ".join(sp["body_phrases"])
        proposed_lines.append(
            f"Segment {s.segment}  ({total:.1f}s, ~{word_count} words)\n"
            f"Hook text: {sp['hook_text']}\n"
            f"Body phrases: {body_display}\n"
            f"CTA text: {sp['cta_text']}"
        )

    critique_system = (
        "You are a senior short-form editor. Your job is to critique AND apply edits.\n\n"
        "Return revised text that is VERBATIM from the transcript (no rewritten words).\n"
        "Your output MUST include revised hook_text/body_phrases for each segment.\n"
        "body_phrases is a list — each item is one contiguous run of transcript words.\n"
        "Text must be EXACT verbatim phrases — we derive timestamps from text.\n\n"
        "Each segment's actual duration (in seconds) is shown. If a segment is "
        f"outside {min_dur:.0f}-{max_dur:.0f}s, adjust the body phrases (add or trim words) "
        f"to bring it within range (~{min_words}-{max_words} words total, "
        f"speaker pace ~{avg_wpm:.0f} WPM).\n\n"
        "Prefer concise, non-repetitive scripts. Remove repeated phrases, tangents, "
        "false starts, and duplicated ideas."
    )
    critique_human_base = (
        "Here are the current proposed shorts with actual durations:\n\n"
        + "\n\n---\n\n".join(proposed_lines)
        + "\n\nPlease critique and apply edits to produce improved cuts. "
        "Keep the same segment count and segment numbering."
    )

    refine: CritiqueRefinement | None = None
    snapped_critique: list[dict[str, Any]] = []
    critique_feedback = ""
    critique = (
        None if stream_thinking else llm.with_structured_output(CritiqueRefinement)
    )
    for attempt in range(4):
        critique_messages = [
            SystemMessage(content=critique_system),
            HumanMessage(content=critique_human_base + critique_feedback),
        ]
        if attempt > 0:
            critique_messages.append(
                HumanMessage(
                    content=(
                        f"RETRY {attempt}: Fix issues below. "
                        "Text must be EXACT verbatim phrases."
                    )
                )
            )
        if stream_thinking:
            if attempt == 0:
                print("\n--- Critique (streaming) ---\n")
            candidate = await asyncio.to_thread(
                _stream_structured_invoke, critique_messages, CritiqueRefinement
            )
        else:
            assert critique is not None
            candidate = await critique.ainvoke(critique_messages)
        if len(candidate.edits) != len(response.shorts):
            continue

        bad: list[tuple[int, float]] = []
        c_snap_failures: list[str] = []
        snapped_critique = []
        for e in candidate.edits:
            snapped = _snap_segment(
                e.hook_text,
                e.body_phrases,
                candidate.cta_text,
                words,
                norm_tokens,
            )
            snapped_critique.append(snapped)
            total = _ranges_duration(
                f"{snapped['hook_range']},{snapped['body_ranges']},{snapped['cta_range']}"
            )
            if total < retry_min or total > retry_max:
                bad.append((e.segment, total))
            if snapped["min_confidence"] < 0.7:
                c_snap_failures.append(
                    f"Segment {e.segment}: could not locate text in transcript"
                )

        if not bad and not c_snap_failures:
            refine = candidate
            break

        c_feedback: list[str] = []
        if bad:
            dur_detail = ", ".join(f"seg {s}={d:.0f}s" for s, d in bad)
            c_feedback.append(
                f"Duration out of range: {dur_detail}. "
                f"Adjust body word count (~{min_words}-{max_words} words total)."
            )
        if c_snap_failures:
            c_feedback.append(
                "Text not found in transcript: " + "; ".join(c_snap_failures[:4])
            )
        critique_feedback = "\n\nFEEDBACK: " + "; ".join(c_feedback)
        print(
            f"   Critique attempt {attempt + 1}: {'; '.join(c_feedback)}. Retrying..."
        )

    # ------------------------------------------------------------------
    # Build final output
    # ------------------------------------------------------------------
    use_critique = refine is not None and snapped_critique
    final_snapped = snapped_critique if use_critique else snapped_proposal

    segments: list[dict[str, Any]] = []
    for i, sp in enumerate(final_snapped):
        seg_src = response.shorts[i]
        body_text_joined = " | ".join(sp["body_phrases"])
        seg = {
            "segment": seg_src.segment,
            "title": "",
            "hook": sp["hook_range"],
            "body": sp["body_ranges"],
            "cta": sp["cta_range"],
            "cuts": f"{sp['hook_range']},{sp['body_ranges']},{sp['cta_range']}",
            "hook_text": sp["hook_text"],
            "body_text": body_text_joined,
            "cta_text": sp["cta_text"],
            "script": f"{sp['hook_text']} | {body_text_joined} | {sp['cta_text']}",
        }
        segments.append(seg)

    min_ok = duration - tolerance
    max_ok = duration + tolerance
    print(
        f"\n📋 Proposed {len(segments)} shorts "
        f"(target {duration:.0f}s ±{tolerance:.0f}s = {min_ok:.0f}-{max_ok:.0f}s):"
    )
    for seg in segments:
        total = _ranges_duration(seg["cuts"])
        hook_dur = _ranges_duration(seg["hook"])
        body_dur = _ranges_duration(seg["body"])
        cta_dur = _ranges_duration(seg["cta"])
        mark = "✓" if min_ok <= total <= max_ok else "⚠"
        print(
            f"   {seg['segment']:2d}. {total:.1f}s {mark} "
            f"(hook={hook_dur:.1f}s body={body_dur:.1f}s cta={cta_dur:.1f}s)"
        )

    print("\n📝 Scripts:")
    for seg in segments:
        print(f"\n  clip {seg['segment']}: {seg['script']}")

    return segments


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Propose interview cuts (hook + body + CTA) from word/utterance transcripts"
    )
    parser.add_argument("--words", type=Path, required=True, help="Path to words JSON")
    parser.add_argument(
        "--utterances", type=Path, required=True, help="Path to utterances JSON"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write shorts_cuts.json",
    )
    parser.add_argument(
        "--num-shorts",
        type=int,
        default=None,
        help=f"Number of shorts (default: {DEFAULT_NUM_SHORTS}). Ignored if --cuts-per-question is set.",
    )
    parser.add_argument(
        "--cuts-per-question",
        type=int,
        default=None,
        help="Propose this many shorts per identified question/topic",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_TARGET_DURATION,
        help=f"Target duration per short in seconds (default: {DEFAULT_TARGET_DURATION})",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Duration tolerance ± seconds (default: {DEFAULT_TOLERANCE})",
    )
    parser.add_argument(
        "--no-allow-hook-reuse",
        action="store_true",
        help="Require each short to have a distinct hook",
    )
    parser.add_argument(
        "--stream-thinking",
        action="store_true",
        help="Stream model reasoning tokens to stdout",
    )
    args = parser.parse_args()

    if not args.words.exists():
        print(f"❌ Words file not found: {args.words}")
        return 1
    if not args.utterances.exists():
        print(f"❌ Utterances file not found: {args.utterances}")
        return 1

    segments = asyncio.run(
        propose_interview_cuts(
            args.words,
            args.utterances,
            num_shorts=args.num_shorts,
            cuts_per_question=args.cuts_per_question,
            duration=args.duration,
            tolerance=args.tolerance,
            allow_hook_reuse=not args.no_allow_hook_reuse,
            stream_thinking=args.stream_thinking,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(segments, indent=4))
    print(f"\n✅ Wrote {len(segments)} segments to {args.output}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
