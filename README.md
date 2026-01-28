# Video Editor Scripts

Video editing scripts for Marky video processing pipeline.

## Setup

Install dependencies:

```bash
uv sync
# or
pip install -e .
```

## Scripts

### `add_title.py`

Adds a title overlay to a video.

- Centered, multiline title support
- White background box with black text
- Configurable duration (default: 2 seconds)
- Automatically wraps long titles to multiple lines
- Can auto-generate title from transcript using LLM (Gemini 3 Flash)
- Supports `--dry-run` to preview top 3 title suggestions

**Usage:**
```bash
python add_title.py <video_file> [title_text] [output_file] [--duration <seconds>] [--transcript <file>] [--dry-run]
python add_title.py video.mp4 --transcript utterances.txt --dry-run  # Preview titles
```

### `enhance_voice.py`

Enhances voice/speech clarity in a video using ffmpeg's dialoguenhance filter.

- Improves speech intelligibility
- Uses hardware-accelerated encoding
- Preserves stereo audio output

**Usage:**
```bash
python enhance_voice.py <video_file> [output_file]
```

### `add_background_music.py`

Adds background music to a video with automatic loudness normalization.

- Voice normalized to -16 LUFS
- Music normalized to -26 LUFS
- Can search Openverse for music by genre tags
- Supports `--dry-run` to preview top 3 music options

**Usage:**
```bash
python add_background_music.py <video_file> [music_file] [output_file] [--tags <genre>] [--dry-run]
python add_background_music.py video.mp4 --tags "hip-hop" --dry-run  # Preview options
python add_background_music.py video.mp4 --tags "hip-hop" output.mp4  # Use first result
```

### `add_subtitles.py`

Adds subtitles to a video using Deepgram transcription.

- Uses Deepgram Nova-2 for transcription
- Generates SRT subtitles with 2 words per line (all caps)
- Burns subtitles into video
- Exports word-level and utterance-level transcripts (JSON and TXT)
- Optional title overlay

**Usage:**
```bash
python add_subtitles.py <video_file> [--output <output_file>] [--title <title_text>]
```

### `remove_filler_words.py`

Removes filler words from video using Deepgram's filler word detection.

- Detects filler words: uh, um, mhmm, mm-mm, uh-uh, uh-huh, nuh-uh
- Silences detected filler word segments
- Can use word-level transcript JSON file (from `add_subtitles.py`) instead of transcribing

**Usage:**
```bash
python remove_filler_words.py <video_file> [output_file] [--transcript <word_transcript.json>]
```

## Dependencies

- `deepgram-sdk`: For transcription and filler word detection
- `ffmpeg`: Required for video/audio processing (system dependency)
