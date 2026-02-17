"""Generate images using Google Gemini (Nano Banana / Nano Banana Pro).

Adapted from fast-backend/src/vendors/gemini.py for standalone use.

Usage:
    dotenvx run -f .env -f .env.dev -- uv run python -m src.generate_image \
        --prompt "A modern dashboard showing social media analytics" \
        --output slides/01_cover.png \
        --aspect-ratio 16:9 \
        --model gemini-3-pro-image-preview

Models:
    - gemini-2.5-flash-image (Nano Banana): Fast, ~10 seconds
    - gemini-3-pro-image-preview (Nano Banana Pro): Higher quality, ~20 seconds
"""

import argparse
import asyncio
import logging
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Literal

import httpx
from google import genai
from google.genai import types
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ASPECT_RATIOS = [
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
]

MODELS = {
    "flash": "gemini-2.5-flash-image",
    "pro": "gemini-3-pro-image-preview",
}

_HTTP_HEADERS = {"User-Agent": "Marky/1.0 (https://mymarky.ai; support@mymarky.ai)"}


async def download_image(url: str) -> Image.Image:
    """Download an image from a URL."""
    async with httpx.AsyncClient(timeout=30.0, headers=_HTTP_HEADERS) as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))


async def generate_image(
    prompt: str,
    image_urls: list[str] | None = None,
    aspect_ratio: str | None = None,
    model: str = "gemini-3-pro-image-preview",
    api_key: str | None = None,
) -> tuple[Image.Image, str]:
    """Generate an image using Gemini.

    Args:
        prompt: Text prompt describing the image to generate.
        image_urls: Optional reference image URLs to guide generation.
        aspect_ratio: One of the supported aspect ratios (e.g. "16:9").
        model: Gemini model identifier.
        api_key: Google API key. Falls back to GEMINI_API_KEY env var.

    Returns:
        Tuple of (PIL Image, response text).
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        logger.error("GEMINI_API_KEY not set. Add it to your .env or pass --api-key.")
        sys.exit(1)

    client = genai.Client(api_key=key)

    # Download any reference images
    ref_images: list[Image.Image] = []
    if image_urls:
        ref_images = list(
            await asyncio.gather(*(download_image(url) for url in image_urls))
        )

    # Prefix to make image generation intent explicit
    image_prompt = f"Generate an image: {prompt}"

    logger.info("Generating image with %s ...", model)
    response = client.models.generate_content(
        model=model,
        contents=[
            *ref_images,
            image_prompt,
        ],
        config=types.GenerateContentConfig(
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
            )
        ),
    )

    image = None
    text = ""
    for part in response.candidates[0].content.parts:  # type: ignore[union-attr]
        if part.text is not None:
            text = part.text
        elif part.inline_data is not None:
            image = Image.open(BytesIO(part.inline_data.data))  # type: ignore[arg-type]

    if not image:
        raise ValueError(
            f"No image generated. Gemini response: {text[:200] if text else 'No text returned'}"
        )

    return image, text


async def generate_and_save(
    prompt: str,
    output: str | Path,
    image_urls: list[str] | None = None,
    aspect_ratio: str | None = None,
    model: str = "gemini-3-pro-image-preview",
    api_key: str | None = None,
) -> Path:
    """Generate an image and save it to disk.

    Returns the output path.
    """
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image, text = await generate_image(
        prompt=prompt,
        image_urls=image_urls,
        aspect_ratio=aspect_ratio,
        model=model,
        api_key=api_key,
    )

    image.save(str(output_path), quality=95)
    logger.info("Saved to %s (%dx%d)", output_path, image.width, image.height)

    if text:
        logger.info("Model response text: %s", text[:200])

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate images with Gemini (Nano Banana Pro)",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        required=True,
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output.png",
        help="Output file path (default: output.png)",
    )
    parser.add_argument(
        "--aspect-ratio",
        "-a",
        choices=ASPECT_RATIOS,
        default=None,
        help="Aspect ratio (default: 1:1 or matches input image)",
    )
    parser.add_argument(
        "--model",
        "-m",
        choices=list(MODELS.keys()) + list(MODELS.values()),
        default="pro",
        help="Model: 'flash' (fast) or 'pro' (higher quality). Default: pro",
    )
    parser.add_argument(
        "--image-urls",
        "-i",
        nargs="*",
        default=None,
        help="Reference image URLs to guide generation",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Google API key (defaults to GEMINI_API_KEY env var)",
    )

    args = parser.parse_args()

    # Resolve model alias
    model = MODELS.get(args.model, args.model)

    asyncio.run(
        generate_and_save(
            prompt=args.prompt,
            output=args.output,
            image_urls=args.image_urls,
            aspect_ratio=args.aspect_ratio,
            model=model,
            api_key=args.api_key,
        )
    )


if __name__ == "__main__":
    main()
