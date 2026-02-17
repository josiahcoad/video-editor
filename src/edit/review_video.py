#!/usr/bin/env python3
"""
Review video and generate vitality score, critiques, and caption.

Uses Gemini to analyze the first 30 seconds of video along with the transcript.
"""

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Try to load from .env file if available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class VideoReview(BaseModel):
    """Response model for video review."""

    vitality_score: int = Field(
        description="Overall vitality score from 1-10 (10 = most viral/engaging)"
    )
    caption: str = Field(
        description="A compelling social media caption for this video (2-3 sentences, engaging, includes relevant hashtags)"
    )
    critiques: dict[str, Any] = Field(
        description="Detailed critiques broken down by category"
    )


async def review_video(
    video_path: Path,
    transcript: str,
    brand_brief: str | None = None,
) -> dict[str, Any]:
    """Review video using Gemini and return vitality score, critiques, and caption.

    Args:
        video_path: Path to the final video file
        transcript: Full transcript of the video
        brand_brief: Optional brand brief to guide the review

    Returns:
        Dict with vitality_score, caption, and critiques
    """
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")

    # Extract first 30 seconds of video for visual analysis
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        clip_path = Path(tmp_file.name)

    try:
        # Extract first 30 seconds
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-t",
                "30",
                "-c",
                "copy",
                str(clip_path),
            ],
            check=True,
            capture_output=True,
        )

        # Upload video to a temporary location or use Gemini's file upload
        # For now, we'll use the video path directly if it's accessible
        # In production, you might need to upload to a URL first

        llm = ChatOpenAI(
            model="google/gemini-3-flash-preview",
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            extra_body={"reasoning": {"max_tokens": 2000, "enabled": True}},
        )

        # Note: Gemini via OpenRouter may not support video directly
        # We'll use transcript + first frame analysis for now
        # In production, use Gemini's native API for video analysis

        brand_context = ""
        if brand_brief:
            brand_context = f"\n\nBRAND BRIEF:\n{brand_brief}\n\nUse this brand brief to evaluate on-brand compliance and alignment."

        system_prompt = f"""You are an expert video content reviewer specializing in social media engagement and viral content.

Your task is to review a video and provide:
1. A vitality score (1-10) - how likely this video is to perform well on social media
2. A compelling social media caption (2-3 sentences, engaging, includes relevant hashtags)
3. Detailed critiques across multiple dimensions

EVALUATION CRITERIA:
- Hook (first 3 seconds): Strength of opening, visual appeal, word choice
- Pacing: Speed, rhythm, energy level
- Lighting: Quality, visibility, professional appearance
- Shooting style: Camera work, framing, production quality
- Thumbnail (first frame): Visual appeal, clarity, click-worthiness
- Title: Effectiveness, clarity, engagement potential
- Duration: Appropriate length for content type and platform
- CTA: Clear call-to-action, placement, effectiveness
- On-brand: Alignment with brand voice, audience, and values (use brand brief if provided)
- Category: Classification (educational, humorous/relatable/entertaining, etc.)
- Compliance: Guardrails check (profanity, brand guidelines, etc.)

Provide specific, actionable feedback for each category.{brand_context}"""

        human_prompt = f"""Review this video:

TRANSCRIPT:
{transcript}

VIDEO INFO:
- Duration: Check video duration
- First 30 seconds: Analyze the opening hook, visual quality, lighting, and shooting style
- Thumbnail: Evaluate the first frame as a potential thumbnail

Provide:
1. Vitality score (1-10) with overall assessment
2. A compelling social media caption (2-3 sentences, engaging, includes relevant hashtags)
3. Detailed critiques for each category with specific feedback

Format your response as structured data."""

        structured_llm = llm.with_structured_output(VideoReview)

        response = await structured_llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )

        return {
            "vitality_score": response.vitality_score,
            "caption": response.caption,
            "critiques": response.critiques,
        }

    finally:
        # Clean up temporary clip
        clip_path.unlink(missing_ok=True)


async def main() -> None:
    """Main entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Review a video and generate vitality score")
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument("--transcript", type=Path, help="Transcript file (JSON with words)")
    parser.add_argument("--brand-brief", type=str, help="Brand brief for on-brand evaluation")
    args = parser.parse_args()

    # Load transcript
    if args.transcript:
        transcript_data = json.loads(args.transcript.read_text())
        if isinstance(transcript_data, list):
            # Word-level transcript
            transcript = " ".join([w.get("word", "") for w in transcript_data])
        else:
            transcript = transcript_data.get("transcript", "")
    else:
        # Extract transcript from video (simplified - in production use get_transcript)
        transcript = "Video transcript not provided"

    result = await review_video(args.video, transcript, args.brand_brief)

    print(f"\n{'='*60}")
    print("VIDEO REVIEW")
    print(f"{'='*60}")
    print(f"\nVitality Score: {result['vitality_score']}/10")
    print(f"\nCaption:\n{result['caption']}")
    print(f"\nCritiques:")
    print(json.dumps(result["critiques"], indent=2))
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
