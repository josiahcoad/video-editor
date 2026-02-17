#!/usr/bin/env python3
"""
Add title card (2s), burned-in captions, and background music to each video in a dir.

Input: dir of segment_01.mp4 .. segment_07.mp4 (e.g. with_emojis/)
Output: with_title_caption_music/segment_01.mp4 .. segment_07.mp4

Pipeline per video: add_title -> add_subtitles -> add_background_music.
Uses one music track for all (searched via --tags on first run).
"""

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Add title, captions, and music to segment videos",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Dir containing segment_01.mp4 .. segment_07.mp4",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default: <input-dir>/../with_title_caption_music)",
    )
    parser.add_argument(
        "--caption-prefix",
        default="Textbook Interview",
        help="Title text prefix: '{prefix} — Part N' (used when --copy-dir has no segment_NN.json)",
    )
    parser.add_argument(
        "--copy-dir",
        type=Path,
        default=None,
        help="Dir with segment_01.json .. segment_07.json from generate_copy_for_segments (title used for title card)",
    )
    parser.add_argument(
        "--music-tags",
        default="corporate upbeat background",
        help="Openverse tags to search for music (one track used for all)",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=None,
        help="Path to settings.json (overrides default title/caption params)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        print(f"Error: not a directory: {input_dir}")
        sys.exit(1)
    output_dir = args.output_dir or (input_dir.parent / "with_title_caption_music")
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(input_dir.glob("*.mp4"))
    if not videos:
        print(f"No .mp4 in {input_dir}")
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    music_path = output_dir / "_music_track.mp3"

    # Load settings: from --settings file if provided, else from project/session (via first video path)
    from src.edit.settings_loader import load_settings

    settings: dict = {}
    if args.settings and args.settings.exists():
        settings = json.loads(args.settings.read_text())
    else:
        settings = load_settings(videos[0])
    title_settings = settings.get("title", {})
    caption_settings = settings.get("captions", {})

    async def ensure_music() -> Path:
        from src.edit.add_background_music import download_music, search_music

        if music_path.exists():
            return music_path
        print(f"Searching for music: {args.music_tags}")
        tracks = await search_music(args.music_tags, count=1)
        await download_music(tracks[0]["url"], music_path)
        return music_path

    if not args.dry_run:
        music_path = asyncio.run(ensure_music())

    copy_dir = None
    if args.copy_dir and args.copy_dir.resolve().is_dir():
        copy_dir = args.copy_dir.resolve()

    for i, video in enumerate(videos, 1):
        part_title = f"{args.caption_prefix} — Part {i}"
        if copy_dir:
            copy_file = copy_dir / f"segment_{i:02d}.json"
            if copy_file.exists():
                data = json.loads(copy_file.read_text())
                # Support both formats:
                #   {title: "..."} (from generate_copy_for_segments)
                #   {titles: [{title: "..."}]} (from suggest_video_title)
                if (
                    "titles" in data
                    and isinstance(data["titles"], list)
                    and data["titles"]
                ):
                    part_title = data["titles"][0].get("title") or part_title
                else:
                    part_title = data.get("title") or part_title
        base = output_dir / f"segment_{i:02d}"
        titled = output_dir / f"segment_{i:02d}_titled.mp4"
        captioned = output_dir / f"segment_{i:02d}_captioned.mp4"
        final_path = output_dir / f"segment_{i:02d}.mp4"

        print(f"\n--- Segment {i}: {video.name} ---")
        if args.dry_run:
            print(f"  Would: title -> captions -> music -> {final_path.name}")
            continue

        # 1. Title card
        title_dur = str(title_settings.get("duration", 2))
        title_cmd = [
            sys.executable,
            "-m",
            "src.edit.add_title",
            str(video),
            part_title,
            str(titled),
            "--duration",
            title_dur,
        ]
        if "height_percent" in title_settings:
            title_cmd += ["--height-percent", str(title_settings["height_percent"])]
        if "anchor" in title_settings:
            title_cmd += ["--anchor", title_settings["anchor"]]
        if "style" in title_settings:
            title_cmd += ["--style", title_settings["style"]]
        if title_settings.get("caps"):
            title_cmd += ["--caps"]
        print(f"  1. Adding title ({title_dur}s): {part_title!r}")
        run_cmd(title_cmd, cwd=script_dir)

        # 2. Burned-in captions (transcribes then burns)
        caption_delay = str(title_settings.get("duration", 2))
        caption_cmd = [
            sys.executable,
            "-m",
            "src.edit.add_subtitles",
            str(titled),
            "--output",
            str(captioned),
            "--delay",
            caption_delay,
        ]
        if "style" in caption_settings:
            caption_cmd += ["--style", caption_settings["style"]]
        if "height_percent" in caption_settings:
            caption_cmd += ["--height-percent", str(caption_settings["height_percent"])]
        if "font" in caption_settings:
            caption_cmd += ["--font", caption_settings["font"]]
        if "font_size" in caption_settings:
            caption_cmd += ["--font-size", str(caption_settings["font_size"])]
        if "word_count" in caption_settings:
            caption_cmd += ["--word-count", str(caption_settings["word_count"])]
        if caption_settings.get("caps"):
            caption_cmd += ["--caps"]
        print("  2. Adding captions")
        run_cmd(caption_cmd, cwd=script_dir)
        titled.unlink(missing_ok=True)

        # 3. Background music (direct call to reuse one downloaded track)
        print("  3. Adding music")
        from src.edit.add_background_music import add_music

        add_music(Path(captioned), music_path, Path(final_path))
        captioned.unlink(missing_ok=True)

        print(f"  ✅ {final_path.name}")

    if args.dry_run:
        print("\n[--dry-run] Run without --dry-run to process.")
    else:
        print(f"\n🎉 Done. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
