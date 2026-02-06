#!/usr/bin/env python3
"""
Apply manual edits to video based on transcript analysis.

Removes:
- Repetitive sections
- Filler words
- Fixes ending issues
"""

import subprocess
import sys
import tempfile
from pathlib import Path


def remove_segment(video_path: Path, output_path: Path, start_time: float, end_time: float) -> None:
    """Remove a segment from video by cutting it out.
    
    Args:
        video_path: Input video
        output_path: Output video
        start_time: Start of segment to remove (seconds)
        end_time: End of segment to remove (seconds)
    """
    # Use ffmpeg to cut out the segment
    # We'll split into two parts and concatenate them
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        
        part1 = tmp / "part1.mp4"
        part2 = tmp / "part2.mp4"
        concat_file = tmp / "concat.txt"
        
        # Extract part before cut
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(video_path),
                "-t", str(start_time),
                "-c", "copy",
                str(part1),
            ],
            check=True,
            capture_output=True,
        )
        
        # Extract part after cut
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(video_path),
                "-ss", str(end_time),
                "-c", "copy",
                str(part2),
            ],
            check=True,
            capture_output=True,
        )
        
        # Create concat file
        concat_file.write_text(f"file '{part1.absolute()}'\nfile '{part2.absolute()}'")
        
        # Concatenate
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )


def remove_ending_text(video_path: Path, output_path: Path, cut_start: float) -> None:
    """Remove text from the end of video.
    
    Args:
        video_path: Input video
        output_path: Output video
        cut_start: Time to cut from (seconds) - everything after this is removed
    """
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-t", str(cut_start),
            "-c", "copy",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python apply_edits.py <video_file> [output_file]")
        sys.exit(1)
    
    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    output_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else video_path.parent / f"{video_path.stem}-edited.mp4"
    )
    
    # Step 1: Remove repetitive section (92.17s to 96.97s)
    print("Step 1: Removing repetitive section (92.17s - 96.97s)...")
    temp1 = video_path.parent / f"{video_path.stem}-temp1.mp4"
    remove_segment(video_path, temp1, 92.17, 96.97)
    
    # Step 2: Remove filler words
    print("Step 2: Removing filler words...")
    temp2 = video_path.parent / f"{video_path.stem}-temp2.mp4"
    # Import and use the remove_filler_words functionality
    from remove_filler_words import transcribe_with_filler_words, find_filler_word_segments, remove_segments_from_video
    import asyncio
    
    # Extract audio for filler word detection
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_tmp:
        audio_path = Path(audio_tmp.name)
    
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(temp1),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(audio_path),
            ],
            check=True,
            capture_output=True,
        )
        
        result = asyncio.run(transcribe_with_filler_words(audio_path))
        filler_segments = find_filler_word_segments(result["words"])
        
        if filler_segments:
            print(f"Found {len(filler_segments)} filler words to remove")
        remove_segments_from_video(temp1, temp2, filler_segments)
    finally:
        audio_path.unlink(missing_ok=True)
        temp1.unlink(missing_ok=True)
    
    # Step 3: Fix ending - remove "in 2020" (cut at 148.68s, before "in")
    print("Step 3: Fixing ending (removing 'in 2020')...")
    remove_ending_text(temp2, output_path, 148.68)
    temp2.unlink(missing_ok=True)
    
    print(f"✅ Output video: {output_path}")


if __name__ == "__main__":
    main()
