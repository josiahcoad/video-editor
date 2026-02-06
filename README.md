# Marky Video Editor

Turn long-form video into short-form clips with AI-driven editorial decisions.

## Quick Start (Docker)

```bash
docker build -t marky-video-editor .

# From a video file:
docker run --rm \
  -e OPENROUTER_API_KEY=your_key \
  -e DEEPGRAM_API_KEY=your_key \
  -v $(pwd):/data \
  marky-video-editor --video /data/video.mp4

# From an existing transcript:
docker run --rm \
  -e OPENROUTER_API_KEY=your_key \
  -v $(pwd):/data \
  marky-video-editor --transcript /data/words.json
```

`OPENROUTER_API_KEY` is always required. `DEEPGRAM_API_KEY` is only needed when using `--video` (auto-transcription).

## Local Setup

```bash
uv sync
```

Requires `ffmpeg` installed on the system and an `OPENROUTER_API_KEY` in `.env`.

## CLI

### `propose_cuts` — AI-powered segment planning

Analyzes a video transcript and proposes how to cut it into short-form segments. The LLM classifies the content type (tutorial, interview, opinion, etc.), decides whether to create one or multiple segments, and outputs precise timestamp ranges.

```bash
# From a video file (auto-transcribes via Deepgram):
dotenvx run -f .env -- uv run python src/propose_cuts.py --video <video.mp4>

# From an existing word-level transcript:
dotenvx run -f .env -- uv run python src/propose_cuts.py --transcript <words.json>
```

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--video <file>` | — | Video file to analyze (will transcribe first) |
| `--transcript <file>` | — | Word-level transcript JSON (skip transcription) |
| `--duration <sec>` | 60 | Target duration per segment |
| `--tolerance <sec>` | 20 | Acceptable duration margin (±) |
| `--prompt <text>` | — | Custom edit direction for the LLM |
| `--model <id>` | `google/gemini-3-flash-preview` | OpenRouter model ID |
| `--count <n>` | auto | Force exactly N segments |
| `--edl <dir>` | — | Output CMX 3600 EDL files (one per segment) |

#### Output

JSON array to stdout. Progress and analysis go to stderr.

```json
[
  {
    "segment": 1,
    "summary": "Quick Wyze Cam setup walkthrough covering unboxing through first stream",
    "hook": "Direct promise to set up in under 5 minutes",
    "cuts": "3.44:6.98,7.28:11.22,22.04:25.18,...",
    "duration": 43.6,
    "section_count": 16,
    "analysis": {
      "content_type": "procedural",
      "single_or_multi": "single",
      "cold_open_candidate": null,
      "keepable_content_seconds": 60.0
    }
  }
]
```

The `cuts` field contains comma-separated `start:end` timestamp ranges (in seconds) that can be passed directly to `ffmpeg` or downstream tools.

#### Examples

```bash
# Punchy 40-second shorts:
uv run python src/propose_cuts.py --video talk.mp4 --duration 40 --tolerance 15

# Use a stronger model for better editorial decisions:
uv run python src/propose_cuts.py --transcript talk-words.json --model anthropic/claude-opus-4

# Export EDL files for use in Premiere/Resolve:
uv run python src/propose_cuts.py --video talk.mp4 --edl edl_output/

# Custom edit direction:
uv run python src/propose_cuts.py --video talk.mp4 --prompt "Focus on actionable advice, skip anecdotes"
```
