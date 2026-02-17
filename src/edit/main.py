#!/usr/bin/env python3
"""
Unified entry point for video editing tools.

Commands:
  search_music      Search for background music
  suggest_titles    Generate title suggestions for a video
  run_autoedit      Run the full video processing pipeline
"""

import argparse
import asyncio
import sys
from pathlib import Path

from .music_search import search_music
from .title_suggestions import get_title_suggestions


async def cmd_search_music(args) -> None:
    """Search for background music."""
    import json

    print(f"Searching for '{args.query}'...\n")
    tracks = await search_music(args.query, args.count)

    if args.json:
        print(json.dumps(tracks, indent=2))
    else:
        print("Music Options:")
        for i, track in enumerate(tracks, 1):
            duration_ms = track.get("duration", 0) or 0
            duration_sec = duration_ms // 1000
            minutes, seconds = divmod(duration_sec, 60)
            duration_str = f"{minutes}:{seconds:02d}" if duration_ms else "?"
            print(f"\n  {i}. {track['title']}")
            print(f"     Artist: {track['creator']}")
            print(f"     Duration: {duration_str}")
            print(f"     License: {track['license']}")
            print(f"     URL: {track['url']}")


async def cmd_suggest_titles(args) -> None:
    """Generate title suggestions for a video."""
    import json

    video_path = args.input_video.resolve()
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    titles = await get_title_suggestions(
        video_path, count=args.count, verbose=not args.quiet
    )

    if args.json:
        print(json.dumps(titles, indent=2))
    else:
        print("\nTitle Suggestions:")
        for i, title in enumerate(titles, 1):
            print(f"  {i}. {title}")

        # Prompt user to select
        print()
        while True:
            try:
                choice = input(f"Select a title (1-{len(titles)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(titles):
                    selected_title = titles[idx]
                    print(f"\n✅ Selected: {selected_title}")
                    return
                print(
                    f"Invalid choice. Please enter a number between 1 and {len(titles)}."
                )
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nCancelled.")
                return


async def cmd_run_autoedit(args) -> None:
    """Run the full video processing pipeline (deprecated)."""
    print(
        "The autoedit pipeline (process_video) has been removed.\n"
        "Use propose_cuts + scripts/apply_cuts_from_json.py for long-form → segments,\n"
        "or interview_to_shorts for interview → N shorts."
    )
    sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Video editing tools - unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands", required=True)

    # =========================================================================
    # search_music command
    # =========================================================================
    search_parser = subparsers.add_parser(
        "search_music", help="Search for background music"
    )
    search_parser.add_argument(
        "--query",
        type=str,
        default="hip hop",
        help="Search query/tags (default: 'hip hop')",
    )
    search_parser.add_argument(
        "-n", "--count", type=int, default=5, help="Number of results (default: 5)"
    )
    search_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # =========================================================================
    # suggest_titles command
    # =========================================================================
    titles_parser = subparsers.add_parser(
        "suggest_titles", help="Generate title suggestions for a video"
    )
    titles_parser.add_argument(
        "--input_video",
        type=Path,
        required=True,
        help="Input video file",
    )
    titles_parser.add_argument(
        "-k",
        "--count",
        type=int,
        default=3,
        help="Number of title suggestions to generate (default: 3)",
    )
    titles_parser.add_argument(
        "--json", action="store_true", help="Output as JSON (no user selection)"
    )
    titles_parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")

    # =========================================================================
    # run_autoedit command
    # =========================================================================
    edit_parser = subparsers.add_parser(
        "run_autoedit", help="Run the full video processing pipeline"
    )
    edit_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input video file",
    )
    edit_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for all processed files",
    )
    edit_parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Video title (if not provided, auto-generates from transcript)",
    )
    edit_parser.add_argument(
        "--music_url",
        type=str,
        default=None,
        help="Background music URL (if not provided, searches and uses first result)",
    )
    edit_parser.add_argument(
        "--music_tags",
        type=str,
        default="hip hop",
        help="Tags for background music search (default: 'hip hop')",
    )
    edit_parser.add_argument(
        "--target_duration",
        type=int,
        default=60,
        help="Target duration for smart trim in seconds (default: 60)",
    )
    edit_parser.add_argument(
        "--speedup",
        type=float,
        default=1.0,
        help="Speed multiplier to apply after silence removal (e.g., 1.2 = 20%% faster) (default: 1.0)",
    )
    edit_parser.add_argument(
        "--caption_word_count",
        type=int,
        default=3,
        help="Words per subtitle line (default: 3)",
    )
    edit_parser.add_argument(
        "--font_size",
        type=int,
        default=14,
        help="Subtitle font size (default: 14)",
    )
    edit_parser.add_argument(
        "--silence_margin",
        type=float,
        default=0.2,
        help="Margin in seconds to keep before/after silence (default: 0.2)",
    )
    edit_parser.add_argument(
        "--trim_tolerance",
        type=float,
        default=20.0,
        help="Acceptable margin for trim duration in seconds (e.g., 20 means ±20s, so 60s target = 40-80s range) (default: 20)",
    )
    edit_parser.add_argument(
        "--caption_height",
        type=int,
        default=None,
        help="Vertical position for captions (0-100, where 0 is bottom, 100 is top). If not provided, auto-determined from first frame.",
    )
    edit_parser.add_argument(
        "--title_height",
        type=int,
        default=None,
        help="Vertical position for title (0-100, where 0 is bottom, 100 is top). If not provided, auto-determined from first frame.",
    )
    edit_parser.add_argument(
        "--ui",
        type=str,
        default="stepper,logs",
        help="UI elements to show: 'logs', 'progress', 'stepper', 'none' (quiet). Comma-separated (default: 'stepper,logs')",
    )
    edit_parser.add_argument(
        "--brand_brief",
        type=str,
        default=None,
        help="Brand brief to guide title generation, cut planning, and review (e.g., brand voice, audience, guardrails)",
    )
    edit_parser.add_argument(
        "--skip",
        type=str,
        default=None,
        help="Comma-separated list of steps to skip (e.g., 'smart_trim,add_music,review_video')",
    )
    edit_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate speedup without processing video (shows WPM and calculated speedup)",
    )
    edit_parser.add_argument(
        "--crop_top",
        type=int,
        default=0,
        help="Percentage to crop from top (0-50, default: 0)",
    )
    edit_parser.add_argument(
        "--crop_bottom",
        type=int,
        default=0,
        help="Percentage to crop from bottom (0-50, default: 0)",
    )
    edit_parser.add_argument(
        "--crop_left",
        type=int,
        default=0,
        help="Percentage to crop from left (0-50, default: 0)",
    )
    edit_parser.add_argument(
        "--crop_right",
        type=int,
        default=0,
        help="Percentage to crop from right (0-50, default: 0)",
    )

    args = parser.parse_args()

    # Route to appropriate command handler
    if args.command == "search_music":
        asyncio.run(cmd_search_music(args))
    elif args.command == "suggest_titles":
        asyncio.run(cmd_suggest_titles(args))
    elif args.command == "run_autoedit":
        asyncio.run(cmd_run_autoedit(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
