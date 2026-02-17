#!/usr/bin/env python3
"""
Set title (and optionally caption) on existing posts by ID.
Supports platform-specific channel overrides (LinkedIn, Twitter, etc.).

Usage:
  update_post_titles.py --post-ids-file path/to/scheduled_post_ids.txt --caption-prefix "Textbook Interview"
  update_post_titles.py --post-ids-file path/to/scheduled_post_ids.txt --titles "Part 1,Part 2,..."
  update_post_titles.py --post-ids-file path/to/scheduled_post_ids.txt --copy-dir outputs/with_copy

If --caption-prefix is set, title becomes "{prefix} — Part N" (N=1..7).
If --titles is set, use those comma-separated titles in order.
If --copy-dir has multi-platform JSON (short/linkedin/twitter), it sets the
default post from "short" and creates post_channel_overrides for linkedin/twitter.
Requires SUPABASE_URL, SUPABASE_API_KEY.
"""

import json
import os
import sys
from pathlib import Path

import httpx

from schedule_post import SUPABASE_KEY, SUPABASE_URL, _headers

# Map write_copy platform names to integration_type in post_channel_overrides
PLATFORM_TO_INTEGRATION = {
    "linkedin": "LINKEDIN",
    "linkedin_reel": "LINKEDIN",
    "twitter": "TWITTER",
    "facebook": "FACEBOOK",
}


def update_post(
    post_id: str, title: str | None = None, caption: str | None = None
) -> None:
    """PATCH post title and/or caption."""
    payload = {}
    if title is not None:
        payload["title"] = title
    if caption is not None:
        payload["caption"] = caption
    if not payload:
        return
    url = f"{SUPABASE_URL}/rest/v1/posts?id=eq.{post_id}"
    resp = httpx.patch(
        url,
        json=payload,
        headers={
            **_headers(profile=True),
            "content-type": "application/json",
        },
        timeout=30,
    )
    if resp.status_code not in (200, 204):
        print(f"  Post update failed ({resp.status_code}): {resp.text}")
        sys.exit(1)


def upsert_channel_override(
    post_id: str,
    integration_type: str,
    title: str | None = None,
    caption: str | None = None,
) -> None:
    """Create or update a post_channel_overrides row."""
    payload: dict = {
        "post_id": post_id,
        "integration_type": integration_type,
    }
    if title is not None:
        payload["title"] = title
    if caption is not None:
        payload["caption"] = caption

    url = (
        f"{SUPABASE_URL}/rest/v1/post_channel_overrides"
        f"?on_conflict=post_id,integration_type"
    )
    resp = httpx.post(
        url,
        json=payload,
        headers={
            **_headers(profile=True),
            "content-type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        },
        timeout=30,
    )
    if resp.status_code not in (200, 201, 204):
        print(
            f"  Channel override upsert failed ({resp.status_code}): {resp.text}",
        )
        # Non-fatal — continue


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Set title/caption on posts")
    parser.add_argument("--post-ids-file", type=Path, required=True)
    parser.add_argument(
        "--caption-prefix",
        default=None,
        help="Set title to '{prefix} — Part N' for each post",
    )
    parser.add_argument(
        "--titles",
        default=None,
        help="Comma-separated titles in order (overrides --caption-prefix)",
    )
    parser.add_argument(
        "--caption",
        action="store_true",
        help="Also set caption to same as title (if using --caption-prefix)",
    )
    parser.add_argument(
        "--copy-dir",
        type=Path,
        default=None,
        help="Dir with segment_01.json .. from generate_copy_for_segments; use title+caption from each (overrides --caption-prefix/--titles)",
    )
    args = parser.parse_args()

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: SUPABASE_URL and SUPABASE_API_KEY required")
        sys.exit(1)

    path = args.post_ids_file.resolve()
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)
    post_ids = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    copy_dir = (
        args.copy_dir.resolve() if args.copy_dir and args.copy_dir.exists() else None
    )
    if copy_dir and copy_dir.is_dir():
        all_copy: list[dict] = []
        for i in range(1, len(post_ids) + 1):
            copy_file = copy_dir / f"segment_{i:02d}.json"
            if copy_file.exists():
                all_copy.append(json.loads(copy_file.read_text()))
            else:
                all_copy = []
                break

        if all_copy and len(all_copy) == len(post_ids):
            # Detect format: multi-platform {short: {...}, linkedin: {...}, ...}
            # or simple {title: ..., caption: ...}
            is_multi = any(k in all_copy[0] for k in ("short", "linkedin", "twitter"))

            for idx, (pid, data) in enumerate(zip(post_ids, all_copy)):
                if is_multi:
                    # Use "short" for the default post
                    default = data.get("short", {})
                    title = default.get("title", "")
                    caption = default.get("caption", "")
                    print(f"  Post {pid[:8]}... title={title[:50]!r}")
                    update_post(pid, title=title, caption=caption)

                    # Create channel overrides for other platforms
                    for platform_key, item in data.items():
                        if platform_key == "short":
                            continue
                        integration = PLATFORM_TO_INTEGRATION.get(platform_key)
                        if not integration:
                            continue
                        ov_title = item.get("title") or title  # fallback to short title
                        ov_caption = item.get("caption", "")
                        print(
                            f"    ↳ {integration} override: " f"title={ov_title[:40]!r}"
                        )
                        upsert_channel_override(
                            pid, integration, title=ov_title, caption=ov_caption
                        )
                else:
                    # Simple format
                    title = data.get("title", "")
                    caption = data.get("caption", "")
                    print(f"  Post {pid[:8]}... title={title[:50]!r}")
                    update_post(pid, title=title, caption=caption)

            print(f"\n✅ Updated {len(post_ids)} posts (from --copy-dir).")
            return
        # else fall through to prefix/titles
    else:
        copy_dir = None

    if args.titles:
        titles = [s.strip() for s in args.titles.split(",") if s.strip()]
    elif args.caption_prefix:
        titles = [
            f"{args.caption_prefix} — Part {i}" for i in range(1, len(post_ids) + 1)
        ]
    else:
        print("Error: pass --caption-prefix, --titles, or --copy-dir")
        sys.exit(1)

    if len(titles) < len(post_ids):
        titles.extend([titles[-1]] * (len(post_ids) - len(titles)))
    n = min(len(post_ids), len(titles))

    for i in range(n):
        title = titles[i]
        caption = title if args.caption else None
        print(f"  Post {post_ids[i][:8]}... title={title!r}")
        update_post(post_ids[i], title=title, caption=caption)

    print(f"\n✅ Updated {n} posts.")


if __name__ == "__main__":
    main()
