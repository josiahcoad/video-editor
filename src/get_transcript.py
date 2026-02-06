#!/usr/bin/env python3
"""
Extract word-level and utterance-level transcript from a video.

Outputs JSON files with timestamps that can be reused by other scripts.
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from deepgram import DeepgramClient

API_KEY = "37e776c73c0de03eeacfaa9635e26ce6787bcf74"


async def get_transcript(video_path: Path) -> dict:
    """Get word-level and utterance-level transcript from video.

    Returns:
        Dict with keys: 'words', 'utterances', 'transcript'
    """
    import subprocess
    import time

    print(f"🔧 DEBUG [deepgram]: Starting transcription for: {video_path}")

    # Use MP3 with low bitrate for much smaller file size (~7MB vs 100MB for WAV)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio_tmp:
        audio_path = Path(audio_tmp.name)

    try:
        print(f"🔧 DEBUG [deepgram]: Extracting audio to: {audio_path}")
        audio_extract_start = time.time()
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vn",  # No video
                "-ac",
                "1",  # Mono
                "-ar",
                "16000",  # 16kHz sample rate (sufficient for speech)
                "-b:a",
                "16k",  # 16kbps bitrate (very small, sufficient for Deepgram)
                str(audio_path),
            ],
            check=True,
            capture_output=True,
        )
        audio_extract_duration = time.time() - audio_extract_start
        print(
            f"🔧 DEBUG [deepgram]: Audio extraction completed ({audio_extract_duration:.2f}s)"
        )

        print(f"🔧 DEBUG [deepgram]: Initializing Deepgram client...")
        client_init_start = time.time()
        client = DeepgramClient(api_key=API_KEY)
        client_init_duration = time.time() - client_init_start
        print(
            f"🔧 DEBUG [deepgram]: Deepgram client initialized ({client_init_duration:.2f}s)"
        )

        print(f"🔧 DEBUG [deepgram]: Reading audio file...")
        read_start = time.time()
        with audio_path.open("rb") as f:
            audio_bytes = f.read()
        read_duration = time.time() - read_start
        audio_size_mb = len(audio_bytes) / (1024 * 1024)
        print(
            f"🔧 DEBUG [deepgram]: Read {audio_size_mb:.2f}MB audio ({read_duration:.2f}s)"
        )

        print(
            f"🔧 DEBUG [deepgram]: Calling Deepgram API (this may take 5-30 seconds)..."
        )
        api_call_start = time.time()
        try:
            response = client.listen.v1.media.transcribe_file(
                request=audio_bytes,
                model="nova-2",
                smart_format=True,
                punctuate=True,
                utterances=True,
            )
            api_call_duration = time.time() - api_call_start
            print(
                f"🔧 DEBUG [deepgram]: Deepgram API call completed ({api_call_duration:.2f}s)"
            )
        except Exception as e:
            api_call_duration = time.time() - api_call_start
            print(
                f"❌ ERROR [deepgram]: API call failed after {api_call_duration:.2f}s"
            )
            print(f"❌ ERROR [deepgram]: Exception: {type(e).__name__}: {str(e)}")
            raise

        if not response or not response.results:
            raise RuntimeError("Deepgram returned empty response")

        channel = response.results.channels[0]
        alternative = channel.alternatives[0]

        transcript = alternative.transcript or ""

        # Extract words with timestamps
        words = []
        if alternative.words:
            for word in alternative.words:
                words.append(
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                    }
                )

        # Extract utterances
        utterances = []
        if response.results.utterances:
            for utterance in response.results.utterances:
                utterances.append(
                    {
                        "text": utterance.transcript or "",
                        "start": utterance.start,
                        "end": utterance.end,
                        "duration": utterance.end - utterance.start,
                    }
                )

        return {
            "words": words,
            "utterances": utterances,
            "transcript": transcript,
        }
    finally:
        audio_path.unlink(missing_ok=True)


def read_word_transcript_file(transcript_path: Path) -> tuple[str, list[dict]]:
    """Read word-level transcript from a JSON file.

    Expected format: list of dicts with 'word', 'start', 'end' keys.

    Returns:
        Tuple of (transcript_text, words_list)
    """
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    content = transcript_path.read_text()
    words = json.loads(content)

    if not isinstance(words, list) or not all(
        "word" in w and "start" in w and "end" in w for w in words
    ):
        raise ValueError(
            "Invalid word-level transcript JSON format. Expected list of dicts with 'word', 'start', 'end' keys."
        )

    # Reconstruct transcript text from words
    transcript = " ".join([w["word"] for w in words])

    return transcript, words


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print(
            "Usage: python get_transcript.py <video_file> [--output-dir <dir>] [--output-words <file>] [--output-utterances <file>]"
        )
        print(
            "  --output-dir: Directory to save transcript files (default: same as video)"
        )
        print("  --output-words: Custom filename for word-level transcript JSON")
        print(
            "  --output-utterances: Custom filename for utterance-level transcript JSON"
        )
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    # Parse output directory
    output_dir = video_path.parent
    if "--output-dir" in sys.argv:
        idx = sys.argv.index("--output-dir")
        if idx + 1 < len(sys.argv):
            output_dir = Path(sys.argv[idx + 1])
            output_dir.mkdir(parents=True, exist_ok=True)

    # Parse custom output filenames
    word_file = None
    utterance_file = None
    if "--output-words" in sys.argv:
        idx = sys.argv.index("--output-words")
        if idx + 1 < len(sys.argv):
            word_file = Path(sys.argv[idx + 1])
    if "--output-utterances" in sys.argv:
        idx = sys.argv.index("--output-utterances")
        if idx + 1 < len(sys.argv):
            utterance_file = Path(sys.argv[idx + 1])

    print(f"Transcribing {video_path}...")
    result = asyncio.run(get_transcript(video_path))

    # Write word-level transcript (JSON)
    if word_file is None:
        word_file = output_dir / f"{video_path.stem}-words.json"
    word_file.write_text(json.dumps(result["words"], indent=2))
    print(f"✅ Word-level transcript: {word_file}")

    # Write utterance-level transcript (JSON)
    if utterance_file is None:
        utterance_file = output_dir / f"{video_path.stem}-utterances.json"
    utterance_file.write_text(json.dumps(result["utterances"], indent=2))
    print(f"✅ Utterance-level transcript: {utterance_file}")

    # Write full transcript (text)
    transcript_file = output_dir / f"{video_path.stem}-transcript.txt"
    transcript_file.write_text(result["transcript"])
    print(f"✅ Full transcript: {transcript_file}")


if __name__ == "__main__":
    main()
