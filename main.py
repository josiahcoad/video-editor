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

from music_search import search_music
from process_video import process_video_flow
from title_suggestions import get_title_suggestions


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
    """Run the full video processing pipeline."""
    from duration_tracker import estimate_duration, get_stats
    import subprocess

    # Parse UI mode
    ui_mode_str = args.ui.lower().strip()
    if ui_mode_str in ("none", "quiet"):
        ui_mode = {"none"}
    else:
        ui_mode = set(ui_mode_str.split(",")) if ui_mode_str else {"stepper", "logs"}

    input_video = args.input.resolve()
    if not input_video.exists():
        print(f"Error: Input video not found: {input_video}")
        sys.exit(1)

    # Get input video duration
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(input_video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    input_duration = float(result.stdout.strip())

    # Dry-run mode: calculate speedup without processing
    if args.dry_run:
        print("=" * 60)
        print("🔍 DRY RUN MODE - Speedup Calculation")
        print("=" * 60)
        print(f"Input: {input_video}")
        print(f"Input duration: {input_duration:.1f}s")
        print("-" * 60)

        # Get transcript to calculate WPM
        from get_transcript import get_transcript

        print("📝 Transcribing video to calculate WPM...")
        transcript_result = await get_transcript(input_video)
        word_count = len(transcript_result["words"])

        # Calculate WPM
        from process_video import _calculate_wpm, _calculate_smart_speedup

        current_wpm = _calculate_wpm(input_duration, word_count)

        print(f"📊 Word count: {word_count} words")
        print(f"📊 Current WPM: {current_wpm:.1f}")
        print("-" * 60)

        # Calculate smart speedup if speedup == 1.0
        if args.speedup == 1.0:
            calculated_speedup = _calculate_smart_speedup(input_duration, word_count)
            if calculated_speedup > 1.0:
                print(f"⚡ Smart speedup: {calculated_speedup:.3f}x")
                print(f"   (Target: 180 WPM, Current: {current_wpm:.1f} WPM)")
                print(
                    f"   → Would speed up to ~{current_wpm * calculated_speedup:.1f} WPM"
                )
            else:
                print(f"✅ Current WPM ({current_wpm:.1f}) >= 170, skipping speedup")
                print("   (No speedup needed)")
        else:
            print(f"⚡ Explicit speedup: {args.speedup}x")
            new_wpm = current_wpm * args.speedup
            print(f"   → Would speed up to ~{new_wpm:.1f} WPM")

        print("=" * 60)
        print("✅ Dry run complete - no video processing performed")
        return

    # Get time estimate
    estimated_time = estimate_duration(
        input_duration, args.speedup, args.target_duration
    )
    stats = get_stats()

    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🎬 Video Processing Pipeline")
    print("=" * 60)
    print(f"Input: {input_video}")
    print(f"Output: {output_dir}")
    print(f"Input duration: {input_duration:.1f}s")
    print(f"Duration: {args.target_duration}s (±{args.trim_tolerance}s)")
    print(f"Speedup: {args.speedup}x")
    print(f"Caption word count: {args.caption_word_count}")
    if args.title:
        print(f"Title: {args.title}")
    else:
        print("Title: (auto-generate from transcript)")
    if args.music_url:
        print(f"Music: {args.music_url}")
    else:
        print(f"Music: (auto-select first '{args.music_tags}' result)")
    print("-" * 60)
    print(
        f"⏱️  Estimated processing time: {estimated_time:.0f}s ({estimated_time/60:.1f} min)"
    )
    if stats.get("total_runs", 0) > 0:
        print(f"   (Based on {stats['total_runs']} previous runs)")
    else:
        print("   (No historical data - using conservative estimate)")
    print("=" * 60)

    # Parse skip list
    skip_steps = set()
    if args.skip:
        skip_steps = {step.strip() for step in args.skip.split(",") if step.strip()}

    # Use cli_runner for UI modes, direct call for quiet mode
    if "none" not in ui_mode and "quiet" not in ui_mode:
        from cli_runner import start_and_stream

        await start_and_stream(
            input_video=input_video,
            output_path=output_dir,
            duration=args.target_duration,
            title=args.title,
            music_url=args.music_url,
            music_tags=args.music_tags,
            speedup=args.speedup,
            ui_mode=ui_mode,
            word_count=args.caption_word_count,
            font_size=args.font_size,
            silence_margin=args.silence_margin,
            trim_tolerance=args.trim_tolerance,
            caption_height=args.caption_height,
            title_height=args.title_height,
            brand_brief=args.brand_brief,
            skip_steps=skip_steps,
            crop_top=args.crop_top,
            crop_bottom=args.crop_bottom,
            crop_left=args.crop_left,
            crop_right=args.crop_right,
        )
        # Note: start_and_stream handles completion message
        return

    # Quiet mode: call flow directly
    final_video = await process_video_flow(
        input_video=input_video,
        output_dir=output_dir,
        target_duration=args.target_duration,
        title=args.title,
        music_url=args.music_url,
        music_tags=args.music_tags,
        word_count=args.caption_word_count,
        font_size=args.font_size,
        silence_margin=args.silence_margin,
        trim_tolerance=args.trim_tolerance,
        speedup=args.speedup,
        caption_height=args.caption_height,
        title_height=args.title_height,
        brand_brief=args.brand_brief,
        skip_steps=skip_steps,
        crop_top=args.crop_top,
        crop_bottom=args.crop_bottom,
        crop_left=args.crop_left,
        crop_right=args.crop_right,
    )

    print(f"\n✅ Pipeline complete! Final video: {final_video}")


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
