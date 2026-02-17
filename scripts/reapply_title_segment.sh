#!/usr/bin/env bash
# Re-apply title from segment_XX/title.txt and re-run pipeline from 05_titled through 09_infographic.
# Use when you want to change the burned-in title for one segment (e.g. segment_06/title.txt).
#
# Usage: bash scripts/reapply_title_segment.sh [segment_number]
# Example: bash scripts/reapply_title_segment.sh 6
# If no argument, runs for any segment that has a title.txt (e.g. segment_06).
#
# Run from video-editor repo root. Uses same OUT and MUSIC_DIR as rerun_06_to_09_all_segments.sh.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
OUT="${OUT:-$ROOT/projects/HunterZier/editing/videos/260214-hunter-session1/outputs}"
MUSIC_DIR="${MUSIC_DIR:-$ROOT/projects/HunterZier/editing/music}"

SEG_NUM="${1:-}"
if [ -z "$SEG_NUM" ]; then
  # Find any segment that has title.txt
  for f in "$OUT"/segment_*/title.txt; do
    [ -f "$f" ] || continue
    seg_dir=$(dirname "$f")
    seg_name=$(basename "$seg_dir")
    n=${seg_name#segment_}
    n=$((10#$n))
    SEG_NUM=$n
    break
  done
  if [ -z "$SEG_NUM" ]; then
    echo "No segment number given and no title.txt found in any segment. Usage: $0 <segment_number>"
    echo "Or create e.g. outputs/segment_06/title.txt with the desired title and run again."
    exit 1
  fi
fi

seg=$(printf "segment_%02d" "$SEG_NUM")
title_file="$OUT/$seg/title.txt"
if [ ! -f "$title_file" ]; then
  echo "Missing $title_file. Create it with the desired title (one line)."
  exit 1
fi

title_text=$(head -1 "$title_file" | tr -d '\r\n')
if [ -z "$title_text" ]; then
  echo "Empty title in $title_file"
  exit 1
fi

pre_title="$OUT/$seg/04_enhanced.mp4"
if [ ! -f "$pre_title" ]; then
  echo "Missing $pre_title. Cannot re-apply title."
  exit 1
fi

echo "========== Re-applying title for $seg =========="
echo "Title: $title_text"

# 1. Re-burn title -> 05_titled.mp4 (shorts: duration 6, height 20, anchor bottom per QC)
uv run python -m src.edit.add_title "$pre_title" "$title_text" "$OUT/$seg/05_titled.mp4" \
  --duration 6 --height 20 --anchor bottom

# 2. Re-run captions (transcript unchanged; use existing 05_titled-words.json)
TRANSCRIPT="$OUT/$seg/05_titled-words.json"
uv run python -m src.edit.add_subtitles "$OUT/$seg/05_titled.mp4" \
  --output "$OUT/$seg/06_captioned.mp4" \
  --transcript "$TRANSCRIPT" \
  --height 12 --style outline --font-size 16 --caps --delay 6

# 3. Enhance, music, emojis, infographic (same as rerun_06_to_09)
uv run python -m src.edit.enhance_voice "$OUT/$seg/06_captioned.mp4" "$OUT/$seg/06_enhanced.mp4"
uv run python -m src.edit.add_background_music "$OUT/$seg/06_enhanced.mp4" "$MUSIC_DIR" "$OUT/$seg/07_music.mp4" --segment "$SEG_NUM"
if [ -f "$OUT/$seg/07_music-words.json" ]; then
  TRANSCRIPT="$OUT/$seg/07_music-words.json"
else
  TRANSCRIPT="$OUT/$seg/06_captioned-words.json"
fi
uv run python -m src.edit.add_emojis "$OUT/$seg/07_music.mp4" "$OUT/$seg/08_emojis.mp4" --transcript "$TRANSCRIPT"
uv run python -m src.edit.add_infographic "$OUT/$seg/08_emojis.mp4" "$OUT/$seg/09_infographic.mp4" --transcript "$TRANSCRIPT"

echo "Done $seg. Title is now: $title_text"
