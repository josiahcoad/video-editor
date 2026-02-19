#!/usr/bin/env python3
"""
Search for background music using Openverse API.

- search_music(tags, count): raw Openverse search.
- search_and_select_music(music_dir, music_vibe, num_tracks): search, LLM selection, download.
Run the module directly to browse/pick music from the CLI.
"""

import argparse
import asyncio
import json
import os
import subprocess
from pathlib import Path

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


def _get_llm(model: str = "google/gemini-2.5-flash") -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")
    return ChatOpenAI(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


class MusicSelection(BaseModel):
    """LLM-selected music tracks from search results."""

    selected_indices: list[int] = Field(
        description="Indices (0-based) of the best tracks from the search results"
    )
    reasoning: str = Field(description="Why these tracks were selected")


async def search_music(tags: str, count: int = 5) -> list[dict]:
    """Search Openverse for music matching tags."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.openverse.org/v1/audio/",
            params={
                "q": tags,
                "license_type": "all-cc",
                "page_size": count,
            },
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", [])[:count]:
            # The 'url' field from Openverse is typically the direct audio file URL
            # (e.g., from CDN like cdn.freesound.org)
            results.append(
                {
                    "title": item.get("title", "Unknown"),
                    "creator": item.get("creator", "Unknown"),
                    "url": item.get("url", ""),  # Direct audio file URL (works for preview & download)
                    "license": item.get("license", ""),
                    "duration": item.get("duration", 0),
                }
            )

        return results


def _write_music_preferences(
    music_dir: Path,
    vibe: str,
    all_results: list[dict],
    selection: MusicSelection,
    filenames: list[str],
) -> None:
    """Write preferences.md documenting the music selection."""
    prefs_path = music_dir / "preferences.md"
    lines = [
        "# Music Preferences\n",
        f"\n## Vibe\n- {vibe}\n",
        "\n## Approved Tracks\n",
        "\n| File | Title | Artist | License | Duration |",
        "\n|------|-------|--------|---------|----------|",
    ]
    for i, idx in enumerate(selection.selected_indices):
        if idx < len(all_results) and i < len(filenames):
            track = all_results[idx]
            lines.append(
                f"\n| {filenames[i]} | {track.get('title', '?')} | "
                f"{track.get('creator', '?')} | {track.get('license', '?')} | "
                f"{track.get('duration', '?')}s |"
            )
    lines.append(f"\n\n## Selection Reasoning\n{selection.reasoning}\n")
    prefs_path.write_text("".join(lines))


async def search_and_select_music(
    music_dir: Path,
    music_vibe: str,
    num_tracks: int = 4,
) -> list[str]:
    """Search Openverse for music, use LLM to select best tracks, download them.

    Returns list of filenames (saved in music_dir).
    """
    music_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(music_dir.glob("*.mp3"))
    if existing:
        names = [f.name for f in existing]
        print(f"✅ Music already exists ({len(names)} tracks): {names}")
        return names

    vibes = [v.strip() for v in music_vibe.split(",") if v.strip()]
    print(f"\n🎵 Searching for music: {vibes}")
    all_results: list[dict] = []
    for vibe in vibes:
        for term in [vibe, f"{vibe} instrumental", f"{vibe} background"]:
            try:
                results = await search_music(term, count=5)
                all_results.extend(results)
            except Exception as e:
                print(f"   Search failed for '{term}': {e}")

    if not all_results:
        print("   ⚠️  No music found. Pipeline will continue without music.")
        return []

    seen_urls: set[str] = set()
    unique_results: list[dict] = []
    for r in all_results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)

    print(
        f"\n🤖 Selecting best {num_tracks} tracks from {len(unique_results)} results..."
    )
    results_text = ""
    for i, r in enumerate(unique_results):
        results_text += (
            f"[{i}] \"{r.get('title', 'Unknown')}\" by {r.get('creator', 'Unknown')} "
            f"— duration: {r.get('duration', '?')}s, license: {r.get('license', '?')}\n"
        )

    llm = _get_llm()
    structured = llm.with_structured_output(MusicSelection)
    response = await structured.ainvoke(
        [
            SystemMessage(
                content=(
                    "You are selecting background music for short-form videos.\n\n"
                    f"Desired vibe: {music_vibe}\n"
                    "Requirements:\n"
                    "- Instrumental only (no vocals)\n"
                    "- Not too intense or distracting\n"
                    "- Professional/clean feel\n"
                    "- Variety: pick tracks with different feels\n"
                    f"- Select exactly {num_tracks} tracks\n\n"
                    "Prefer tracks that are 30-180 seconds long."
                )
            ),
            HumanMessage(
                content=f"Available tracks:\n\n{results_text}\n\nSelect {num_tracks} tracks."
            ),
        ]
    )

    selected_filenames: list[str] = []
    for idx in response.selected_indices[:num_tracks]:
        if idx < 0 or idx >= len(unique_results):
            continue
        track = unique_results[idx]
        url = track.get("url", "")
        title = track.get("title", "unknown").lower()
        creator = track.get("creator", "unknown").lower()
        safe_title = "".join(c if c.isalnum() or c in "-_ " else "" for c in title)
        safe_title = safe_title.strip().replace(" ", "-")[:30]
        safe_creator = "".join(c if c.isalnum() or c in "-_ " else "" for c in creator)
        safe_creator = safe_creator.strip().replace(" ", "-")[:20]
        num = len(selected_filenames) + 1
        filename = f"{num:02d}_{safe_title}_{safe_creator}.mp3"
        filepath = music_dir / filename
        print(f"   Downloading: {filename}")
        try:
            subprocess.run(
                ["curl", "-sL", "-o", str(filepath), url],
                check=True,
                capture_output=True,
                timeout=60,
            )
            selected_filenames.append(filename)
            print(f"   ✅ Saved: {filepath.name}")
        except Exception as e:
            print(f"   ❌ Failed to download: {e}")

    _write_music_preferences(
        music_dir, music_vibe, unique_results, response, selected_filenames
    )
    print(f"\n✅ {len(selected_filenames)} music tracks ready")
    return selected_filenames


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Search for background music")
    parser.add_argument("tags", type=str, nargs="?", default="hip hop", help="Search tags (default: 'hip hop')")
    parser.add_argument("-n", "--count", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    print(f"Searching for '{args.tags}'...\n")
    tracks = await search_music(args.tags, args.count)

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


if __name__ == "__main__":
    asyncio.run(main())
