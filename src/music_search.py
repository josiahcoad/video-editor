#!/usr/bin/env python3
"""
Search for background music using Openverse API.

Run independently before the main workflow to browse and pick music.
"""

import argparse
import asyncio
import json

import httpx


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
