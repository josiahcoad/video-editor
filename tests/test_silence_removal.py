#!/usr/bin/env python3
"""
Standalone script to test silence removal.
"""

import sys
from pathlib import Path

# Import the function from process_video
from process_video import remove_silence_task, _get_video_duration


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_silence_removal.py <input_video> [output_video] [--margin <seconds>]")
        sys.exit(1)
    
    input_video = Path(sys.argv[1])
    if not input_video.exists():
        print(f"Error: Input video not found: {input_video}")
        sys.exit(1)
    
    # Parse output path
    if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
        output_video = Path(sys.argv[2])
    else:
        output_video = input_video.parent / f"{input_video.stem}_no_silence.mp4"
    
    # Parse margin
    margin = 0.2
    if "--margin" in sys.argv:
        idx = sys.argv.index("--margin")
        if idx + 1 < len(sys.argv):
            margin = float(sys.argv[idx + 1])
    
    print(f"Input: {input_video}")
    print(f"Output: {output_video}")
    print(f"Margin: {margin}s")
    print()
    
    # Get input duration
    input_duration = _get_video_duration(input_video)
    print(f"Input duration: {input_duration:.2f}s ({input_duration/60:.1f} min)")
    print()
    
    # Run silence removal
    print("Running silence removal...")
    result = remove_silence_task(input_video, output_video, margin=margin)
    
    # Get output duration
    output_duration = _get_video_duration(result)
    reduction = input_duration - output_duration
    reduction_pct = (reduction / input_duration) * 100
    
    print()
    print("=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Input duration:  {input_duration:.2f}s ({input_duration/60:.1f} min)")
    print(f"Output duration: {output_duration:.2f}s ({output_duration/60:.1f} min)")
    print(f"Silence removed: {reduction:.2f}s ({reduction/60:.1f} min)")
    print(f"Reduction:       {reduction_pct:.1f}%")
    print(f"Output file:     {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
