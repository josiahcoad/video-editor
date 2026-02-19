"""Test harness for streaming + reasoning with OpenRouter/LangChain."""

import asyncio
import os


# ---------------------------------------------------------------------------
# Test 1: Raw OpenAI client (stream=True) — reasoning_details in delta
# ---------------------------------------------------------------------------
def test_raw_openai():
    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    stream = client.chat.completions.create(
        model="anthropic/claude-opus-4.6",
        messages=[{"role": "user", "content": "What's bigger, 9.9 or 9.11?"}],
        max_tokens=10000,
        extra_body={"reasoning": {"max_tokens": 8000}},
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if hasattr(delta, "reasoning_details") and delta.reasoning_details:
            for rd in delta.reasoning_details:
                if isinstance(rd, dict) and rd.get("type") == "reasoning.text":
                    print(rd.get("text", ""), end="", flush=True)
        elif getattr(delta, "content", None):
            print(delta.content, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# Test 2: LangChain ChatOpenAI.astream — check additional_kwargs for reasoning
# ---------------------------------------------------------------------------
async def test_langchain_astream():
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="anthropic/claude-opus-4.6",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        streaming=True,
        extra_body={"reasoning": {"max_tokens": 8000}},
    )
    print("LangChain astream (no structured output):\n")
    async for chunk in llm.astream("What's bigger, 9.9 or 9.11?"):
        if hasattr(chunk, "additional_kwargs") and chunk.additional_kwargs:
            rc = chunk.additional_kwargs.get("reasoning_content")
            if rc:
                print(f"THINKING: {rc}", end="", flush=True)
        if hasattr(chunk, "content") and chunk.content:
            print(chunk.content, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# Test 3: LangChain with structured output + astream (may not stream reasoning)
# ---------------------------------------------------------------------------
async def test_langchain_structured_astream():
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field

    class Answer(BaseModel):
        result: str = Field(description="The answer")
        explanation: str = Field(description="Brief explanation")

    llm = ChatOpenAI(
        model="anthropic/claude-opus-4.6",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        streaming=True,
        extra_body={"reasoning": {"max_tokens": 8000}},
    )
    structured = llm.with_structured_output(Answer)
    print("LangChain astream + with_structured_output:\n")
    async for chunk in structured.astream(
        "What's bigger, 9.9 or 9.11? Reply with result and explanation."
    ):
        if hasattr(chunk, "additional_kwargs") and chunk.additional_kwargs:
            rc = chunk.additional_kwargs.get("reasoning_content")
            if rc:
                print(f"THINKING: {rc}", end="", flush=True)
        if chunk:
            print(f"CHUNK: {chunk}", flush=True)
    print()


if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "raw"
    if which == "raw":
        test_raw_openai()
    elif which == "langchain":
        asyncio.run(test_langchain_astream())
    elif which == "structured":
        asyncio.run(test_langchain_structured_astream())
    else:
        print("Usage: python stream.py [raw|langchain|structured]")
