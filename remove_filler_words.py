#!/usr/bin/env python3
"""
Remove filler words from a video using Deepgram's filler word detection.

Uses Deepgram's filler_words feature to identify filler words with timestamps,
then removes those segments from the video.
"""

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from deepgram import DeepgramClient

# Filler words that Deepgram can detect
FILLER_WORDS = {"uh", "um", "mhmm", "mm-mm", "uh-uh", "uh-huh", "nuh-uh"}

API_KEY = "37e776c73c0de03eeacfaa9635e26ce6787bcf74"


async def transcribe_with_filler_words(audio_path: Path) -> dict[str, Any]:
    """Transcribe audio with Deepgram and return words including filler words."""
    client = DeepgramClient(api_key=API_KEY)
    
    with audio_path.open("rb") as audio_file:
        audio_bytes = audio_file.read()
    
    response = client.listen.v1.media.transcribe_file(
        request=audio_bytes,
        model="nova-2",
        smart_format=True,
        punctuate=True,
        utterances=True,
        filler_words=True,  # Enable filler word detection
    )
    
    if not response or not response.results:
        raise RuntimeError("Deepgram returned empty response")
    
    channel = response.results.channels[0]
    alternative = channel.alternatives[0]
    
    # Get words array with filler words
    words = []
    if alternative.words:
        for word in alternative.words:
            words.append({
                "word": word.word.strip().lower(),
                "start": word.start,
                "end": word.end,
            })
    
    return {
        "words": words,
        "transcript": alternative.transcript or "",
    }


def find_filler_word_segments(words: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Find all filler word segments that should be removed.
    
    Returns:
        List of (start_time, end_time) tuples for filler words
    """
    filler_segments = []
    
    for word in words:
        word_text = word.get("word", "").strip().lower()
        if word_text in FILLER_WORDS:
            start = word.get("start", 0.0)
            end = word.get("end", 0.0)
            filler_segments.append((start, end))
    
    return filler_segments


def remove_segments_from_video(
    video_path: Path,
    output_path: Path,
    segments_to_remove: list[tuple[float, float]],
) -> None:
    """Remove audio segments from video by silencing them.
    
    Args:
        video_path: Input video file
        output_path: Output video file
        segments_to_remove: List of (start, end) tuples in seconds to silence
    """
    if not segments_to_remove:
        # No filler words, just copy the video
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(video_path),
                "-c:v", "copy",
                "-c:a", "copy",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
        return
    
    # Build volume filter to silence filler word segments
    # Format: volume=enable='between(t,start,end)':volume=0
    volume_filters = []
    
    # First, get video duration to handle edge cases
    duration_cmd = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    video_duration = float(duration_cmd.stdout.strip())
    
    # Build complex filter to silence filler word segments
    # We'll use the volume filter with enable conditions
    for start, end in segments_to_remove:
        # Clamp to video duration
        start = max(0, min(start, video_duration))
        end = max(0, min(end, video_duration))
        
        if start < end:
            # Silence this segment
            volume_filters.append(
                f"volume=enable='between(t,{start},{end})':volume=0"
            )
    
    if volume_filters:
        # Apply all volume filters in sequence
        afilter = ",".join(volume_filters)
        
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(video_path),
                "-af", afilter,
                "-c:v", "copy",
                "-c:a", "aac",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
    else:
        # No filters to apply, just copy
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(video_path),
                "-c:v", "copy",
                "-c:a", "copy",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )


def load_words_from_transcript(transcript_path: Path) -> list[dict[str, Any]]:
    """Load word-level transcript from JSON file.
    
    Args:
        transcript_path: Path to JSON file with word-level transcript
        
    Returns:
        List of word dicts with 'word', 'start', 'end' keys
    """
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
    
    content = transcript_path.read_text()
    words = json.loads(content)
    
    if not isinstance(words, list):
        raise ValueError("Transcript file must contain a JSON array of words")
    
    # Validate structure
    for word in words:
        if not all(key in word for key in ["word", "start", "end"]):
            raise ValueError("Each word must have 'word', 'start', and 'end' keys")
    
    return words


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python remove_filler_words.py <video_file> [output_file] [--transcript <file>]")
        print("  --transcript: Path to word-level transcript JSON file (from add_subtitles.py)")
        print("                If not provided, will transcribe video with Deepgram")
        sys.exit(1)
    
    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Parse transcript file flag
    transcript_file = None
    if "--transcript" in sys.argv:
        idx = sys.argv.index("--transcript")
        if idx + 1 < len(sys.argv):
            transcript_file = Path(sys.argv[idx + 1])
    
    # Parse output path
    output_path = video_path.parent / f"{video_path.stem}-no-fillers.mp4"
    output_arg_idx = 2
    if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
        output_path = Path(sys.argv[2])
        output_arg_idx = 3
    
    # Get words from transcript file or transcribe
    if transcript_file:
        print(f"Loading word-level transcript from: {transcript_file}")
        words = load_words_from_transcript(transcript_file)
        print(f"Loaded {len(words)} words from transcript")
    else:
        # Extract audio
        print("Extracting audio from video...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_tmp:
            audio_path = Path(audio_tmp.name)
        
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i", str(video_path),
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    str(audio_path),
                ],
                check=True,
                capture_output=True,
            )
            
            # Transcribe with filler words
            print("Transcribing with Deepgram (filler words enabled)...")
            result = asyncio.run(transcribe_with_filler_words(audio_path))
            words = result["words"]
            print(f"Found {len(words)} total words")
        finally:
            audio_path.unlink(missing_ok=True)
    
    # Find filler word segments
    filler_segments = find_filler_word_segments(words)
    
    if not filler_segments:
        print("No filler words detected. Video will be copied unchanged.")
    else:
        print(f"Found {len(filler_segments)} filler words:")
        for start, end in filler_segments:
            print(f"  - {start:.2f}s to {end:.2f}s")
    
    # Remove filler word segments
    print(f"\nRemoving filler words from video...")
    remove_segments_from_video(video_path, output_path, filler_segments)
    
    print(f"✅ Output video: {output_path}")


if __name__ == "__main__":
    main()
