#!/usr/bin/env python3
"""One-off script to regenerate whiteboard video with better prompt."""

import asyncio
import os
import time
from pathlib import Path

from google import genai
from google.genai import types


async def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)

    prompt = (
        "A close-up shot of a hand with a black marker drawing on a clean white "
        "whiteboard. The hand draws a simple comparison diagram in portrait orientation. "
        "First, it writes 'DOWN PAYMENT OPTIONS' at the top with an underline. "
        "Then it draws three boxes stacked vertically: "
        "Box 1 labeled 'VA Loan' with '0% DOWN' circled next to it. "
        "Box 2 labeled 'USDA' with '0% DOWN' circled next to it. "
        "Box 3 labeled 'FHA / Conv.' with '3-5% DOWN' written next to it. "
        "The hand draws arrows connecting them and adds a checkmark next to each option. "
        "Clean, professional whiteboard drawing style with neat handwriting. "
        "White background, black marker ink. Smooth drawing animation."
    )

    config = types.GenerateVideosConfig(
        aspect_ratio="9:16",
        negative_prompt="human face close-up, realistic person, photograph, watermark, blurry, messy",
        resolution="1080p",
    )

    print(f"Starting VeO3 generation...", flush=True)
    operation = client.models.generate_videos(
        model="veo-3.1-fast-generate-preview",
        prompt=prompt,
        config=config,
    )
    print(f"Operation: {operation.name}", flush=True)

    start = time.time()
    while True:
        op_obj = types.GenerateVideosOperation(name=operation.name)
        op = client.operations.get(op_obj)
        elapsed = time.time() - start

        if op.done:
            if op.error:
                print(f"Error: {op.error.message}", flush=True)
                return
            if not op.response or not op.response.generated_videos:
                rai = getattr(op.response, "rai_media_filtered_count", None)
                reasons = getattr(op.response, "rai_media_filtered_reasons", None)
                print(f"No video. RAI: {rai}, Reasons: {reasons}", flush=True)
                return
            video = op.response.generated_videos[0]
            video_bytes = client.files.download(file=video.video)
            out = Path(
                "projects/HunterZier/editing/videos/260214-hunter-session1/outputs/segment_01/whiteboard_overlay.mp4"
            )
            out.write_bytes(video_bytes)
            print(f"Done! {out} ({elapsed:.0f}s, {len(video_bytes)} bytes)", flush=True)
            return

        if elapsed > 300:
            print("Timeout after 5 minutes", flush=True)
            return

        print(f"  Polling... ({elapsed:.0f}s)", flush=True)
        await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())
