#!/usr/bin/env python3
"""
Replace the video on existing scheduled posts with new video files.

Uploads each new video to Supabase Storage, creates business_media, then
PATCHes the post's media_urls to the new URL. Use this after adding emojis
(or re-editing) to swap in the updated videos without changing captions or post IDs.

Usage:
  replace_post_videos.py --post-ids "id1,id2,id3,id4,id5,id6,id7" --videos-dir projects/.../outputs/with_emojis

Videos in --videos-dir are used in sorted order (segment_01.mp4 -> first post ID, etc.).
Requires SUPABASE_URL, SUPABASE_API_KEY, MARKY_BUSINESS_ID in env.
"""

import os
import sys
from pathlib import Path

import httpx

# Reuse schedule_post helpers
from schedule_post import (
    SUPABASE_KEY,
    SUPABASE_URL,
    _headers,
    create_business_media,
    upload_video,
)

MARKY_BUSINESS_ID = os.environ.get("MARKY_BUSINESS_ID", "")


def update_post_media_url(post_id: str, media_url: str) -> None:
    """PATCH post.media_urls to [media_url]."""
    url = f"{SUPABASE_URL}/rest/v1/posts?id=eq.{post_id}"
    resp = httpx.patch(
        url,
        json={"media_urls": [media_url]},
        headers={
            **_headers(profile=True),
            "content-type": "application/json",
        },
        timeout=30,
    )
    if resp.status_code not in (200, 204):
        print(f"  Post update failed ({resp.status_code}): {resp.text}")
        sys.exit(1)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Replace videos on existing posts with new files",
    )
    parser.add_argument(
        "--post-ids",
        default=None,
        help="Comma-separated list of post UUIDs in segment order (1st video -> 1st ID)",
    )
    parser.add_argument(
        "--post-ids-file",
        type=Path,
        default=None,
        help="Path to file with one post UUID per line (skip # and blank). Use instead of --post-ids.",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        required=True,
        help="Directory containing segment_01.mp4 .. segment_07.mp4 (or any .mp4 in order)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: SUPABASE_URL and SUPABASE_API_KEY required")
        sys.exit(1)
    if not MARKY_BUSINESS_ID:
        print("Error: MARKY_BUSINESS_ID required")
        sys.exit(1)

    if args.post_ids_file is not None:
        path = args.post_ids_file.resolve()
        if not path.exists():
            print(f"Error: post IDs file not found: {path}")
            sys.exit(1)
        post_ids = [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    elif args.post_ids:
        post_ids = [s.strip() for s in args.post_ids.split(",") if s.strip()]
    else:
        print("Error: pass --post-ids or --post-ids-file")
        sys.exit(1)
    videos_dir = args.videos_dir.resolve()
    if not videos_dir.is_dir():
        print(f"Error: not a directory: {videos_dir}")
        sys.exit(1)

    videos = sorted(videos_dir.glob("*.mp4"))
    if len(videos) != len(post_ids):
        print(f"Warning: {len(videos)} videos vs {len(post_ids)} post IDs; using min.")
    n = min(len(videos), len(post_ids))

    print(f"Replacing video on {n} posts with files from {videos_dir.name}")
    for i in range(n):
        post_id = post_ids[i]
        video_path = videos[i]
        print(f"\n  Post {post_id[:8]}... <- {video_path.name}")
        if args.dry_run:
            continue
        storage_key = upload_video(video_path, MARKY_BUSINESS_ID)
        file_size = video_path.stat().st_size
        create_business_media(MARKY_BUSINESS_ID, storage_key, file_size)
        media_url = f"{SUPABASE_URL}/storage/v1/object/public/media/{storage_key}"
        update_post_media_url(post_id, media_url)
        print(f"  ✅ Post updated")

    if args.dry_run:
        print("\n[--dry-run] Run without --dry-run to upload and update.")
    else:
        print(f"\n🎉 Replaced {n} post videos.")


if __name__ == "__main__":
    main()
