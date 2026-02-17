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


async def get_transcript(video_path: Path, filler_words: bool = False) -> dict:
    """Get word-level and utterance-level transcript from video.

    Args:
        video_path: Path to the video file.
        filler_words: If True, enable Deepgram filler-word detection so that
            "uh", "um", "mhmm" etc. appear as explicit word entries.

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
            transcribe_kwargs: dict = dict(
                request=audio_bytes,
                model="nova-2",
                smart_format=True,
                punctuate=True,
                utterances=True,
            )
            if filler_words:
                transcribe_kwargs["filler_words"] = True
            response = client.listen.v1.media.transcribe_file(**transcribe_kwargs)
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
    import argparse as _ap

    parser = _ap.ArgumentParser(
        description="Extract word-level and utterance-level transcript from a video.",
    )
    parser.add_argument("video", type=Path, help="Path to the video file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save transcript files (default: same as video)",
    )
    parser.add_argument(
        "--output-words",
        type=Path,
        default=None,
        help="Custom filename for word-level transcript JSON",
    )
    parser.add_argument(
        "--output-utterances",
        type=Path,
        default=None,
        help="Custom filename for utterance-level transcript JSON",
    )
    parser.add_argument(
        "--filler-words",
        action="store_true",
        default=False,
        help="Enable Deepgram filler-word detection (uh, um, mhmm, etc.)",
    )
    args = parser.parse_args()

    video_path = args.video
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    output_dir = args.output_dir or video_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    word_file = args.output_words
    utterance_file = args.output_utterances

    print(f"Transcribing {video_path}...")
    if args.filler_words:
        print("  (filler-word detection enabled)")
    result = asyncio.run(get_transcript(video_path, filler_words=args.filler_words))

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
