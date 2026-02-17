#!/usr/bin/env python3
"""
Upload a video and create a scheduled post in Marky.

Steps:
  1. Upload video to Supabase Storage
  2. Create a business_media record
  3. Create a post (SCHEDULED by default)
  4. (Optional) Create per-platform caption overrides

Usage:
  schedule_post.py --video <path> --caption "caption text" [--title "title"]
  schedule_post.py --video <path> --caption "caption text" --business_id <uuid>

  # With per-platform overrides:
  schedule_post.py --video <path> --caption "base caption" \
    --override twitter "Short tweet text" \
    --override linkedIn @linkedin_caption.txt

  Caption and override values starting with @ are read from a file (e.g. --caption @caption.txt).
  Valid platforms: instagram, facebook, linkedIn, twitter, tiktok, pinterest, googleBusiness

Requires env vars:
  SUPABASE_URL        – e.g. https://yiescabgrmnfqlxtyqkw.supabase.co
  SUPABASE_API_KEY    – service-role key
  Business ID is resolved in order: --business_id flag, then project settings.json
  (when the video path is under projects/<ClientName>/editing/videos/...), then
  MARKY_BUSINESS_ID env var. See repurpose.md "Per-project settings".
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_API_KEY", "")

VALID_PLATFORMS = {
    "instagram",
    "facebook",
    "linkedIn",
    "twitter",
    "tiktok",
    "pinterest",
    "googleBusiness",
}


def _business_id_from_project_settings(video_path: Path) -> str | None:
    """If video is under projects/<Client>/editing/videos/..., load business_id from that editing/settings.json."""
    try:
        resolved = video_path.resolve()
    except OSError:
        return None
    # Walk up from the video's parent until we find a dir that has both "videos" and "settings.json" (project editing root)
    p = resolved.parent
    while p != p.parent:
        settings_file = p / "settings.json"
        if (p / "videos").is_dir() and settings_file.is_file():
            try:
                data = json.loads(settings_file.read_text())
                return data.get("business_id")
            except (json.JSONDecodeError, OSError):
                return None
        p = p.parent
    return None


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


# ── Step 4: Create per-platform overrides ─────────────────────────────────────


def create_overrides(
    post_id: str,
    overrides: list[tuple[str, str]],
) -> None:
    """Insert post_channel_overrides rows for per-platform captions.

    Args:
        post_id: The post UUID to attach overrides to.
        overrides: List of (platform, caption) tuples.
    """
    if not overrides:
        return

    rows = []
    for platform, caption in overrides:
        # Read from file if caption starts with @
        if caption.startswith("@"):
            file_path = Path(caption[1:])
            if not file_path.exists():
                print(
                    f"  ⚠️  Override file not found: {file_path} — skipping {platform}"
                )
                continue
            caption = file_path.read_text().strip()

        if platform not in VALID_PLATFORMS:
            print(
                f"  ⚠️  Unknown platform '{platform}' — "
                f"valid: {', '.join(sorted(VALID_PLATFORMS))}"
            )
            continue

        rows.append(
            {
                "post_id": post_id,
                "integration_type": platform,
                "caption": caption,
            }
        )

    if not rows:
        return

    url = f"{SUPABASE_URL}/rest/v1/post_channel_overrides"
    resp = httpx.post(
        url,
        json=rows,
        headers={
            **_headers(profile=True),
            "content-type": "application/json",
            "accept": "application/json",
            "prefer": "return=representation",
        },
        timeout=30,
    )

    if resp.status_code not in (200, 201):
        print(f"  ⚠️  Override creation failed ({resp.status_code}): {resp.text}")
        return

    data = resp.json()
    for row in data:
        caption_preview = (
            row["caption"][:60] + "..." if len(row["caption"]) > 60 else row["caption"]
        )
        print(f"  ✅ Override ({row['integration_type']}): {caption_preview}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a video and create a scheduled post in Marky",
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the video file",
    )
    parser.add_argument(
        "--caption",
        required=True,
        help="Post caption text",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Post title (optional)",
    )
    parser.add_argument(
        "--business_id",
        default=None,
        help="Business UUID (default: from project settings.json, else MARKY_BUSINESS_ID)",
    )
    parser.add_argument(
        "--status",
        default="SCHEDULED",
        choices=["NEW", "SCHEDULED", "LIKED"],
        help="Post status (default: SCHEDULED)",
    )
    parser.add_argument(
        "--override",
        nargs=2,
        action="append",
        metavar=("PLATFORM", "CAPTION"),
        default=[],
        help=(
            "Per-platform caption override (repeatable). "
            "PLATFORM is one of: instagram, facebook, linkedIn, twitter, "
            "tiktok, pinterest, googleBusiness. "
            "CAPTION is the text or @filepath to read from a file."
        ),
    )
    args = parser.parse_args()

    # Resolve business_id: flag > project settings.json (when video under editing/videos/...) > env
    business_id = args.business_id
    if not business_id:
        business_id = _business_id_from_project_settings(args.video)
    if not business_id:
        business_id = os.environ.get("MARKY_BUSINESS_ID")
    if not business_id:
        print(
            "Error: business_id required. Set --business_id, add business_id to project editing/settings.json, or set MARKY_BUSINESS_ID env var."
        )
        sys.exit(1)
    args.business_id = business_id

    # Validate
    if not SUPABASE_URL:
        print("Error: SUPABASE_URL env var is required")
        sys.exit(1)
    if not SUPABASE_KEY:
        print("Error: SUPABASE_API_KEY env var is required")
        sys.exit(1)
    if not args.video.exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    overrides: list[tuple[str, str]] = [(p, c) for p, c in args.override]

    # Resolve @filepath for main caption (same as overrides)
    caption = args.caption
    if caption.startswith("@"):
        caption_path = Path(caption[1:].strip())
        if not caption_path.is_absolute():
            caption_path = Path.cwd() / caption_path
        if caption_path.exists():
            caption = caption_path.read_text().strip()
        else:
            print(f"Error: Caption file not found: {caption_path}")
            sys.exit(1)

    print(f"🎬 Scheduling post for business {args.business_id}")
    print(f"   Video:   {args.video}")
    print(f"   Caption: {caption[:80]}{'...' if len(caption) > 80 else ''}")
    if args.title:
        print(f"   Title:   {args.title}")
    if overrides:
        print(f"   Overrides: {', '.join(p for p, _ in overrides)}")
    print()

    # Step 1: Upload video
    storage_key = upload_video(args.video, args.business_id)

    # Step 2: Create business_media
    file_size = args.video.stat().st_size
    create_business_media(args.business_id, storage_key, file_size)

    # Step 3: Create post
    post_id = create_post(
        args.business_id,
        storage_key,
        caption,
        args.title,
        args.status,
    )

    # Step 4: Create per-platform overrides
    if overrides:
        print()
        create_overrides(post_id, overrides)

    print(f"\n🎉 Done! Post {post_id} is {args.status}")
    if overrides:
        print(f"   {len(overrides)} channel override(s) created")


if __name__ == "__main__":
    main()
