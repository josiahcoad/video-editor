#!/usr/bin/env bash
# Re-run pipeline from 06_captioned through 09_infographic for all 10 segments.
# Use after updating 06_captioned (e.g. new caption position) so finals get the change.
#
# Run from video-editor:
#   dotenvx run -f .env -- bash scripts/rerun_06_to_09_all_segments.sh
# Log: scripts/rerun_06_to_09.log
#
# Music dir: projects/HunterZier/editing/music (mp3s in that folder).

set -e
# Run from video-editor repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/projects/HunterZier/editing/videos/260214-hunter-session1/outputs"
MUSIC_DIR="$ROOT/projects/HunterZier/editing/music"

for n in 1 2 3 4 5 6 7 8 9 10; do
  seg=$(printf "segment_%02d" "$n")
  echo "========== $seg =========="
  uv run python -m src.edit.enhance_voice "$OUT/$seg/06_captioned.mp4" "$OUT/$seg/06_enhanced.mp4"
  uv run python -m src.edit.add_background_music "$OUT/$seg/06_enhanced.mp4" "$MUSIC_DIR" "$OUT/$seg/07_music.mp4" --segment "$n"
  # Prefer 07_music-words.json if present (same duration); else 06_captioned-words.json
  if [ -f "$OUT/$seg/07_music-words.json" ]; then
    TRANSCRIPT="$OUT/$seg/07_music-words.json"
  else
    TRANSCRIPT="$OUT/$seg/06_captioned-words.json"
  fi
  uv run python -m src.edit.add_emojis "$OUT/$seg/07_music.mp4" "$OUT/$seg/08_emojis.mp4" --transcript "$TRANSCRIPT"
  uv run python -m src.edit.add_infographic "$OUT/$seg/08_emojis.mp4" "$OUT/$seg/09_infographic.mp4" --transcript "$TRANSCRIPT"
  echo "Done $seg"
done
echo "All 10 segments done."
