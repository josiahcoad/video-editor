#!/usr/bin/env python3
"""
Upload a video and create a scheduled post in Marky.

Steps:
  1. Upload video to Supabase Storage
  2. Create a business_media record
  3. Create a post (SCHEDULED by default)

Usage:
  schedule_post.py --video <path> --caption "caption text" [--title "title"]
  schedule_post.py --video <path> --caption "caption text" --business_id <uuid>

Requires env vars:
  SUPABASE_URL        – e.g. https://yiescabgrmnfqlxtyqkw.supabase.co
  SUPABASE_API_KEY    – service-role key
  MARKY_BUSINESS_ID   – default business ID (can be overridden with --business_id)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import httpx

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_API_KEY", "")


def _headers(*, profile: bool = False) -> dict[str, str]:
    """Standard headers for Supabase REST / Storage API."""
    h = {
        "apikey": SUPABASE_KEY,
        "authorization": f"Bearer {SUPABASE_KEY}",
    }
    if profile:
        h["accept-profile"] = "marky"
        h["content-profile"] = "marky"
    return h


# ── Step 1: Upload video to storage ──────────────────────────────────────────

def upload_video(video_path: Path, business_id: str) -> str:
    """Upload video file to Supabase Storage.

    Returns the storage key (without the bucket prefix).
    """
    ts = int(time.time() * 1000)
    storage_key = f"workspace/{business_id}/video-{ts}.mp4"
    url = f"{SUPABASE_URL}/storage/v1/object/media/{storage_key}"

    file_size = video_path.stat().st_size
    print(f"Uploading {video_path.name} ({file_size / 1_000_000:.1f} MB)...")

    with open(video_path, "rb") as f:
        resp = httpx.post(
            url,
            content=f,
            headers={**_headers(), "content-type": "video/mp4"},
            timeout=300,
        )

    if resp.status_code not in (200, 201):
        print(f"Upload failed ({resp.status_code}): {resp.text}")
        sys.exit(1)

    data = resp.json()
    print(f"  ✅ Uploaded: {data.get('Key', storage_key)}")
    return storage_key


# ── Step 2: Create business_media record ─────────────────────────────────────

def create_business_media(
    business_id: str,
    storage_key: str,
    file_size: int,
) -> str:
    """Create a business_media row. Returns the new media ID."""
    url = f"{SUPABASE_URL}/rest/v1/business_media?select=id"

    payload = {
        "business_id": business_id,
        "supabase_storage_key": storage_key,
        "expected_size": file_size,
        "type": "video",
        "transcoding_status": "ready",
    }

    resp = httpx.post(
        url,
        json=payload,
        headers={
            **_headers(profile=True),
            "content-type": "application/json",
            "accept": "application/vnd.pgrst.object+json",
            "prefer": "return=representation",
        },
        timeout=30,
    )

    if resp.status_code not in (200, 201):
        print(f"business_media creation failed ({resp.status_code}): {resp.text}")
        sys.exit(1)

    media_id = resp.json()["id"]
    print(f"  ✅ business_media: {media_id}")
    return media_id


# ── Step 3: Create post ──────────────────────────────────────────────────────

def create_post(
    business_id: str,
    storage_key: str,
    caption: str,
    title: str | None = None,
    status: str = "SCHEDULED",
) -> str:
    """Create a post row. Returns the new post ID."""
    media_url = f"{SUPABASE_URL}/storage/v1/object/public/media/{storage_key}"

    payload: dict = {
        "business_id": business_id,
        "caption": caption,
        "media_urls": [media_url],
        "status": status,
    }
    if title:
        payload["title"] = title

    url = f"{SUPABASE_URL}/rest/v1/posts?select=id,status"

    resp = httpx.post(
        url,
        json=payload,
        headers={
            **_headers(profile=True),
            "content-type": "application/json",
            "accept": "application/vnd.pgrst.object+json",
            "prefer": "return=representation",
        },
        timeout=30,
    )

    if resp.status_code not in (200, 201):
        print(f"Post creation failed ({resp.status_code}): {resp.text}")
        sys.exit(1)

    data = resp.json()
    post_id = data["id"]
    print(f"  ✅ Post created: {post_id} (status={data['status']})")
    print(f"     Media URL: {media_url}")
    return post_id


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a video and create a scheduled post in Marky",
    )
    parser.add_argument(
        "--video", type=Path, required=True, help="Path to the video file",
    )
    parser.add_argument(
        "--caption", required=True, help="Post caption text",
    )
    parser.add_argument(
        "--title", default=None, help="Post title (optional)",
    )
    parser.add_argument(
        "--business_id",
        default=os.environ.get("MARKY_BUSINESS_ID"),
        help="Business UUID (defaults to MARKY_BUSINESS_ID env var)",
    )
    parser.add_argument(
        "--status",
        default="SCHEDULED",
        choices=["NEW", "SCHEDULED", "LIKED"],
        help="Post status (default: SCHEDULED)",
    )
    args = parser.parse_args()

    # Validate
    if not SUPABASE_URL:
        print("Error: SUPABASE_URL env var is required")
        sys.exit(1)
    if not SUPABASE_KEY:
        print("Error: SUPABASE_API_KEY env var is required")
        sys.exit(1)
    if not args.business_id:
        print("Error: --business_id or MARKY_BUSINESS_ID env var is required")
        sys.exit(1)
    if not args.video.exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    print(f"🎬 Scheduling post for business {args.business_id}")
    print(f"   Video:   {args.video}")
    print(f"   Caption: {args.caption[:80]}{'...' if len(args.caption) > 80 else ''}")
    if args.title:
        print(f"   Title:   {args.title}")
    print()

    # Step 1
    storage_key = upload_video(args.video, args.business_id)

    # Step 2
    file_size = args.video.stat().st_size
    create_business_media(args.business_id, storage_key, file_size)

    # Step 3
    post_id = create_post(
        args.business_id,
        storage_key,
        args.caption,
        args.title,
        args.status,
    )

    print(f"\n🎉 Done! Post {post_id} is {args.status}")


if __name__ == "__main__":
    main()
