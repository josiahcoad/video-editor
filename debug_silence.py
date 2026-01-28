#!/usr/bin/env python3
"""
Debug script to see what silence ranges are being detected.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path


def detect_silence(input_video: Path, margin: float = 0.2):
    """Detect silence ranges and show what would be kept."""
    print(f"Analyzing: {input_video}")
    print(f"Margin: {margin}s")
    print()
    
    # Extract and normalize audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        audio_path = Path(tmp_audio.name)
    
    try:
        # Extract and normalize audio
        extract_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-vn",
            "-af",
            "dynaudnorm=g=5:s=0.95",
            "-ar",
            "8000",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            str(audio_path),
        ]
        subprocess.run(extract_cmd, check=True, capture_output=True)
        
        # Detect silence
        detect_cmd = [
            "ffmpeg",
            "-i",
            str(audio_path),
            "-af",
            "silencedetect=noise=-30dB:d=0.3",
            "-f",
            "null",
            "-",
        ]
        
        result = subprocess.run(
            detect_cmd,
            capture_output=True,
            text=True,
        )
    finally:
        audio_path.unlink(missing_ok=True)
    
    # Parse silence ranges
    silence_ranges = []
    silence_start_pattern = re.compile(r"silence_start: ([\d.]+)")
    silence_end_pattern = re.compile(r"silence_end: ([\d.]+)")
    
    current_start = None
    for line in result.stderr.split("\n"):
        if match := silence_start_pattern.search(line):
            current_start = float(match.group(1))
        elif current_start is not None and (match := silence_end_pattern.search(line)):
            silence_end = float(match.group(1))
            silence_ranges.append((current_start, silence_end))
            current_start = None
    
    # Get video duration
    duration_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_video),
    ]
    duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
    video_duration = float(duration_result.stdout.strip())
    
    print(f"Video duration: {video_duration:.2f}s ({video_duration/60:.1f} min)")
    print(f"Found {len(silence_ranges)} silence ranges")
    print()
    
    # Show first 20 silence ranges
    print("=" * 80)
    print("First 20 silence ranges:")
    print("=" * 80)
    for i, (start, end) in enumerate(silence_ranges[:20], 1):
        duration = end - start
        print(f"{i:3d}. {start:7.2f}s - {end:7.2f}s ({duration:5.2f}s)")
    if len(silence_ranges) > 20:
        print(f"... and {len(silence_ranges) - 20} more")
    print()
    
    # Merge overlapping/adjacent silences
    silence_ranges.sort(key=lambda x: x[0])
    merged_silences = []
    if silence_ranges:
        current_start, current_end = silence_ranges[0]
        for silence_start, silence_end in silence_ranges[1:]:
            if silence_start <= current_end + (2 * margin):
                current_end = max(current_end, silence_end)
            else:
                merged_silences.append((current_start, current_end))
                current_start, current_end = silence_start, silence_end
        merged_silences.append((current_start, current_end))
    
    print(f"Merged to {len(merged_silences)} silence ranges (from {len(silence_ranges)})")
    print()
    
    # Calculate segments to keep (inverse of merged silences)
    segments_to_keep = []
    last_end = 0.0
    
    for silence_start, silence_end in merged_silences:
        keep_start = last_end
        keep_end = max(keep_start, silence_start - margin)
        if keep_end > keep_start:
            segments_to_keep.append((keep_start, keep_end))
        last_end = max(0.0, silence_end + margin)
    
    if last_end < video_duration:
        segments_to_keep.append((last_end, video_duration))
    
    total_kept = sum(end - start for start, end in segments_to_keep)
    total_silence = sum(end - start for start, end in silence_ranges)
    
    print("=" * 80)
    print("Segments that would be kept:")
    print("=" * 80)
    print(f"Total segments: {len(segments_to_keep)}")
    print(f"Total kept duration: {total_kept:.2f}s ({total_kept/60:.1f} min)")
    print(f"Total silence duration: {total_silence:.2f}s ({total_silence/60:.1f} min)")
    print(f"Reduction: {video_duration - total_kept:.2f}s ({(video_duration - total_kept)/video_duration*100:.1f}%)")
    print()
    
    # Show first 20 segments
    print("First 20 segments:")
    for i, (start, end) in enumerate(segments_to_keep[:20], 1):
        duration = end - start
        print(f"{i:3d}. {start:7.2f}s - {end:7.2f}s ({duration:5.2f}s)")
    if len(segments_to_keep) > 20:
        print(f"... and {len(segments_to_keep) - 20} more")
    print()
    
    # Check for potential issues
    print("=" * 80)
    print("Potential Issues:")
    print("=" * 80)
    
    # Check for very short segments
    short_segments = [(s, e) for s, e in segments_to_keep if (e - s) < 0.5]
    if short_segments:
        print(f"⚠️  {len(short_segments)} segments are < 0.5s (might cause choppiness)")
        for start, end in short_segments[:5]:
            print(f"   {start:.2f}s - {end:.2f}s ({end-start:.2f}s)")
    
    # Check for very long silences
    long_silences = [(s, e) for s, e in silence_ranges if (e - s) > 2.0]
    if long_silences:
        print(f"⚠️  {len(long_silences)} silences are > 2s (might be cutting too much)")
        for start, end in long_silences[:5]:
            print(f"   {start:.2f}s - {end:.2f}s ({end-start:.2f}s)")
    
    # Check gaps between segments
    gaps = []
    for i in range(len(segments_to_keep) - 1):
        gap = segments_to_keep[i+1][0] - segments_to_keep[i][1]
        if gap > 0.1:  # More than 100ms gap
            gaps.append((segments_to_keep[i][1], segments_to_keep[i+1][0], gap))
    
    if gaps:
        print(f"⚠️  {len(gaps)} gaps > 0.1s between segments (content might be lost)")
        for prev_end, next_start, gap in gaps[:5]:
            print(f"   Gap: {prev_end:.2f}s - {next_start:.2f}s ({gap:.2f}s)")
    
    if not short_segments and not long_silences and not gaps:
        print("✓ No obvious issues detected")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_silence.py <input_video> [--margin <seconds>]")
        sys.exit(1)
    
    input_video = Path(sys.argv[1])
    if not input_video.exists():
        print(f"Error: Input video not found: {input_video}")
        sys.exit(1)
    
    margin = 0.2
    if "--margin" in sys.argv:
        idx = sys.argv.index("--margin")
        if idx + 1 < len(sys.argv):
            margin = float(sys.argv[idx + 1])
    
    detect_silence(input_video, margin)
