#!/usr/bin/env python3
"""
Describe or query a local video using Gemini.

Uploads the video to Gemini's File API, waits for processing,
then prompts the model and prints the response.

Usage:
  python describe_video.py <video_file> [--prompt "your question"]
  python describe_video.py <video_file>  # defaults to "describe the video in detail"

Requires GEMINI_API_KEY environment variable.
"""

import argparse
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types


def describe_video(
    video_path: Path,
    prompt: str = "describe the video in detail",
    model: str = "gemini-3-flash-preview",
) -> str:
    """Upload a local video to Gemini and get a text response.

    Args:
        video_path: Path to local video file
        prompt: Question or instruction about the video
        model: Gemini model to use

    Returns:
        Model's text response
    """
    import os

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable must be set")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # Upload video
    print(f"Uploading {video_path.name} to Gemini...")
    uploaded_file = client.files.upload(file=str(video_path))
    print(f"  Uploaded: {uploaded_file.name}")

    # Wait for processing
    print("Waiting for video processing...", end="", flush=True)
    poll_interval = 2
    timeout = 5 * 60
    deadline = time.monotonic() + timeout

    while True:
        file_info = client.files.get(name=uploaded_file.name)
        state = file_info.state.name if hasattr(file_info.state, "name") else str(file_info.state)

        if state == "ACTIVE":
            print(" ready.")
            break
        if state not in {"PROCESSING", "PROCESSING_UPLOAD"}:
            error = getattr(file_info, "error", None)
            raise RuntimeError(f"File processing failed ({state}): {error}")
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for video processing")

        print(".", end="", flush=True)
        time.sleep(poll_interval)

    # Generate response
    print(f"Prompting {model}...")
    response = client.models.generate_content(
        model=model,
        contents=types.Content(
            parts=[
                types.Part(
                    file_data=types.FileData(file_uri=file_info.uri),
                ),
                types.Part(text=prompt),
            ]
        ),
    )

    text = response.text or ""

    # Token usage
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        um = response.usage_metadata
        print(
            f"  Tokens: {um.prompt_token_count} prompt, "
            f"{um.candidates_token_count} completion, "
            f"{um.total_token_count} total"
        )

    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a video with Gemini")
    parser.add_argument("video", type=Path, help="Local video file")
    parser.add_argument(
        "--prompt",
        default="describe the video in detail",
        help="Question or instruction about the video",
    )
    parser.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="Gemini model (default: gemini-3-flash-preview)",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    result = describe_video(args.video, args.prompt, args.model)
    print(f"\n{result}")


if __name__ == "__main__":
    main()
