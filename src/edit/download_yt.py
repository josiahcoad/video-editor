#!/usr/bin/env python3
"""
Download a YouTube video using yt-dlp (no dependency on fast-backend).

Output: single merged MP4 (best video ≤1080p + best audio, or best single file).
Uses cookies from Safari by default so YouTube is less likely to block.

Usage:
    uv run python src/edit/download_yt.py "https://www.youtube.com/watch?v=VIDEO_ID"
    uv run python src/edit/download_yt.py "https://www.youtube.com/watch?v=VIDEO_ID" -o projects/MyProject/editing/videos/source/
"""

import argparse
import sys
from pathlib import Path

import yt_dlp


def download(
    url: str,
    output_dir: Path | None = None,
    output_template: str = "%(id)s.%(ext)s",
    cookies_browser: str | None = "safari",
) -> Path:
    """Download a YouTube video. Returns path to the downloaded file."""
    outdir = output_dir or Path.cwd()
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / output_template

    opts = {
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(outpath),
        "merge_output_format": "mp4",
        "quiet": False,
    }
    if cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser,)

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    if not info:
        raise RuntimeError("Failed to extract video info")
    # Final file is outdir / id + extension (merge gives mp4)
    final = outdir / f"{info['id']}.mp4"
    if not final.exists():
        raise FileNotFoundError(f"Expected file not found: {final}")
    return final


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a YouTube video with yt-dlp (no fast-backend).",
    )
    parser.add_argument(
        "url", help="YouTube URL (e.g. https://www.youtube.com/watch?v=...)"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save the video (default: current directory)",
    )
    parser.add_argument(
        "--output-template",
        default="%(id)s.%(ext)s",
        help="yt-dlp output template (default: %%(id)s.%%(ext)s)",
    )
    parser.add_argument(
        "--no-cookies",
        action="store_true",
        help="Don't use browser cookies (may get 403 from YouTube)",
    )
    parser.add_argument(
        "--cookies-from",
        default="safari",
        help="Browser to use for cookies (default: safari). Ignored if --no-cookies.",
    )
    args = parser.parse_args()

    try:
        path = download(
            args.url,
            output_dir=args.output_dir,
            output_template=args.output_template,
            cookies_browser=None if args.no_cookies else args.cookies_from,
        )
        print(f"✅ Downloaded: {path}")
        return 0
    except Exception as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
