#!/usr/bin/env python3
"""
Preview social-media chrome overlays on a video frame for QC.

Extracts a frame and draws semi-transparent overlays showing where
platform UI elements (top bar, bottom bar, side icons) will obscure
the content.  Helps verify that important elements — face, title,
captions — are not hidden.

Usage:
  python preview_chrome.py video.mp4
  python preview_chrome.py video.mp4 --time 3.5
  python preview_chrome.py video.mp4 --platform instagram
  python preview_chrome.py video.mp4 --platform all --open
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Chrome-zone definitions — fractions of a 1080×1920 (9:16) reference frame.
#
# Each zone: (x_frac, y_frac, w_frac, h_frac, label)
#   x_frac / y_frac  = top-left corner as fraction of width / height
#   w_frac / h_frac  = size as fraction of width / height
#
# Sources:
#   Instagram Reels — topiasupply safe-zone guide (1080×1920):
#       top 220 px, bottom 450 px, left/right 35 px
#   TikTok — community-measured safe zones (~1080×1920):
#       top ~150 px, bottom ~480 px, right sidebar ~90 px
#   YouTube Shorts — community-measured safe zones (~1080×1920):
#       top ~180 px, bottom ~380 px, right sidebar ~80 px
# ---------------------------------------------------------------------------

PLATFORMS: dict[str, dict] = {
    "instagram": {
        "label": "Instagram Reels",
        "color": (227, 73, 148),  # IG pink
        "zones": [
            (0.0, 0.0, 1.0, 220 / 1920, "Top (username, audio, follow)"),
            (0.0, 1.0 - 450 / 1920, 1.0, 450 / 1920, "Bottom (likes, caption, tabs)"),
        ],
        "safe_margin": {
            "left": 35 / 1080,
            "right": 35 / 1080,
            "top": 220 / 1920,
            "bottom": 450 / 1920,
        },
    },
    "tiktok": {
        "label": "TikTok",
        "color": (0, 242, 234),  # TikTok teal
        "zones": [
            (0.0, 0.0, 1.0, 150 / 1920, "Top (status, back, search)"),
            (
                0.0,
                1.0 - 480 / 1920,
                1.0,
                480 / 1920,
                "Bottom (user, desc, music, tabs)",
            ),
            (1.0 - 90 / 1080, 0.30, 90 / 1080, 0.40, "Side icons"),
        ],
        "safe_margin": {
            "left": 20 / 1080,
            "right": 90 / 1080,
            "top": 150 / 1920,
            "bottom": 480 / 1920,
        },
    },
    "youtube": {
        "label": "YouTube Shorts",
        "color": (255, 0, 0),  # YT red
        "zones": [
            (0.0, 0.0, 1.0, 180 / 1920, "Top (status, search, camera)"),
            (0.0, 1.0 - 380 / 1920, 1.0, 380 / 1920, "Bottom (subscribe, desc, music)"),
            (1.0 - 80 / 1080, 0.35, 80 / 1080, 0.35, "Side icons"),
        ],
        "safe_margin": {
            "left": 20 / 1080,
            "right": 80 / 1080,
            "top": 180 / 1920,
            "bottom": 380 / 1920,
        },
    },
}


def extract_frame(video: Path, time: float, output: Path) -> None:
    """Extract a single frame from *video* at *time* seconds."""
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(time),
        "-i",
        str(video),
        "-vframes",
        "1",
        "-q:v",
        "2",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: ffmpeg failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try system fonts, fall back to Pillow default."""
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_chrome_overlay(
    frame_path: Path,
    output_path: Path,
    platforms: list[str],
) -> None:
    """Composite semi-transparent chrome zones onto *frame_path*."""
    img = Image.open(frame_path).convert("RGBA")
    w, h = img.size

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = _load_font(max(16, h // 50))
    small_font = _load_font(max(12, h // 65))

    zone_alpha = 70

    # --- Draw each platform's no-go zones -----------------------------------
    for platform_key in platforms:
        pf = PLATFORMS[platform_key]
        r, g, b = pf["color"]

        for x_frac, y_frac, w_frac, h_frac, label in pf["zones"]:
            x0 = int(x_frac * w)
            y0 = int(y_frac * h)
            x1 = int((x_frac + w_frac) * w)
            y1 = int((y_frac + h_frac) * h)
            draw.rectangle([x0, y0, x1, y1], fill=(r, g, b, zone_alpha))
            draw.rectangle([x0, y0, x1, y1], outline=(r, g, b, 200), width=2)

            # Zone label (inside the rectangle, near its top-left)
            lx = x0 + 8 if x0 < w // 2 else x0 + 4
            ly = y0 + 4
            draw.text((lx, ly), label, fill=(255, 255, 255, 220), font=small_font)

    # --- Combined safe zone (intersection of all margins) --------------------
    max_l = max(PLATFORMS[k]["safe_margin"]["left"] for k in platforms)
    max_r = max(PLATFORMS[k]["safe_margin"]["right"] for k in platforms)
    max_t = max(PLATFORMS[k]["safe_margin"]["top"] for k in platforms)
    max_b = max(PLATFORMS[k]["safe_margin"]["bottom"] for k in platforms)

    sx0 = int(max_l * w)
    sy0 = int(max_t * h)
    sx1 = int((1.0 - max_r) * w)
    sy1 = int((1.0 - max_b) * h)
    draw.rectangle([sx0, sy0, sx1, sy1], outline=(0, 255, 0, 220), width=3)
    draw.text((sx0 + 8, sy0 + 6), "SAFE ZONE", fill=(0, 255, 0, 240), font=font)

    # --- Legend (top-right inside safe zone) ---------------------------------
    legend_x = sx1 - 200
    legend_y = sy0 + 10
    for platform_key in platforms:
        pf = PLATFORMS[platform_key]
        r, g, b = pf["color"]
        draw.rectangle(
            [legend_x, legend_y, legend_x + 14, legend_y + 14],
            fill=(r, g, b, 220),
        )
        draw.text(
            (legend_x + 20, legend_y - 1),
            pf["label"],
            fill=(255, 255, 255, 240),
            font=small_font,
        )
        legend_y += int(small_font.size * 1.5) if hasattr(small_font, "size") else 20

    # --- Composite and save --------------------------------------------------
    result = Image.alpha_composite(img, overlay)
    result.convert("RGB").save(output_path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview social-media chrome overlays on a video frame",
    )
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument(
        "--time",
        "-t",
        type=float,
        default=0.0,
        help="Timestamp in seconds (default: 0)",
    )
    parser.add_argument(
        "--platform",
        "-p",
        choices=list(PLATFORMS.keys()) + ["all"],
        default="all",
        help="Which platform chrome to show (default: all)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path (default: <video_stem>_chrome.jpg)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the image in the default viewer after saving",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        args.output = args.video.parent / f"{args.video.stem}_chrome.jpg"

    platforms = list(PLATFORMS.keys()) if args.platform == "all" else [args.platform]

    with tempfile.TemporaryDirectory() as tmpdir:
        raw = Path(tmpdir) / "frame.jpg"
        print(f"Extracting frame at {args.time}s …")
        extract_frame(args.video, args.time, raw)
        print(f"Drawing chrome overlays: {', '.join(p for p in platforms)} …")
        draw_chrome_overlay(raw, args.output, platforms)

    print(f"Saved → {args.output}")

    if args.open:
        subprocess.run(["open", str(args.output)])


if __name__ == "__main__":
    main()
