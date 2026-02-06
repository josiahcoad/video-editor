#!/usr/bin/env python3
"""
Streamlit UI for Talking-Head Short-Form Creator.

Run with: streamlit run ui.py
"""

import asyncio
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from music_search import search_music
from title_suggestions import get_title_suggestions
from get_transcript import get_transcript

# Default output directory (used when Output UI is hidden)
DEFAULT_OUTPUT_DIR = Path("./output").expanduser().resolve()

# Profile file: persists preferences across "Clear All" / cache clears
PROFILE_PATH = Path(__file__).parent / "profile.json"

# Session state keys to persist in profile.json
PROFILE_KEYS = [
    "title_height",
    "caption_height",
    "word_replacements",
    "full_pipeline_duration",
    "full_pipeline_tolerance",
    "full_pipeline_words",
    "custom_trim_prompt",
    "speedup_mode",
    "speedup_value",
    "target_wpm",
    "selected_title",
    "bookmarked_music",
]


def load_profile() -> dict:
    """Load saved preferences from profile.json."""
    if PROFILE_PATH.exists():
        try:
            return json.loads(PROFILE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_profile() -> None:
    """Save current preferences to profile.json."""
    data = {}
    for key in PROFILE_KEYS:
        if key in st.session_state:
            val = st.session_state[key]
            if isinstance(val, (str, int, float, bool, type(None))):
                data[key] = val
            elif isinstance(val, (list, dict)) and key == "bookmarked_music":
                data[key] = val  # Persist bookmarked tracks
            else:
                data[key] = str(val)
    try:
        PROFILE_PATH.write_text(json.dumps(data, indent=2))
    except OSError:
        pass


def extract_first_frame(video_path: Path) -> Image.Image:
    """Extract first frame from video as PIL Image."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        frame_path = Path(tmp.name)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            "select=eq(n\\,0)",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(frame_path),
        ],
        check=True,
        capture_output=True,
    )

    img = Image.open(frame_path)
    frame_path.unlink()
    return img


def draw_preview_overlay(
    img: Image.Image,
    title_text: str,
    title_height: int,
    caption_height: int,
) -> Image.Image:
    """Draw title and caption preview on the frame."""
    img = img.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    width, height = img.size

    # Try to get a decent font, fall back to default
    try:
        title_font_size = max(24, width // 20)
        caption_font_size = max(18, width // 30)
        title_font = ImageFont.truetype(
            "/System/Library/Fonts/Helvetica.ttc", title_font_size
        )
        caption_font = ImageFont.truetype(
            "/System/Library/Fonts/Helvetica.ttc", caption_font_size
        )
    except (OSError, IOError):
        title_font = ImageFont.load_default()
        caption_font = ImageFont.load_default()

    # Calculate Y positions (0 = bottom, 100 = top)
    title_y = height - (title_height * height // 100) - title_font_size
    caption_y = height - (caption_height * height // 100) - caption_font_size

    # Draw title preview (centered)
    if title_text:
        bbox = draw.textbbox((0, 0), title_text, font=title_font)
        text_width = bbox[2] - bbox[0]
        title_x = (width - text_width) // 2

        for dx, dy in [
            (-2, -2),
            (-2, 2),
            (2, -2),
            (2, 2),
            (-2, 0),
            (2, 0),
            (0, -2),
            (0, 2),
        ]:
            draw.text(
                (title_x + dx, title_y + dy), title_text, font=title_font, fill="black"
            )
        draw.text((title_x, title_y), title_text, font=title_font, fill="white")

    # Draw caption preview (centered)
    caption_text = "Sample caption text"
    bbox = draw.textbbox((0, 0), caption_text, font=caption_font)
    text_width = bbox[2] - bbox[0]
    caption_x = (width - text_width) // 2

    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        draw.text(
            (caption_x + dx, caption_y + dy),
            caption_text,
            font=caption_font,
            fill="black",
        )
    draw.text((caption_x, caption_y), caption_text, font=caption_font, fill="yellow")

    # Draw position indicator lines
    draw.line([(0, title_y), (width, title_y)], fill="red", width=1)
    draw.text(
        (10, title_y - 20),
        f"Title: {title_height}%",
        font=caption_font,
        fill="red",
    )
    draw.line([(0, caption_y), (width, caption_y)], fill="yellow", width=1)
    draw.text(
        (10, caption_y + 5),
        f"Subtitle: {caption_height}%",
        font=caption_font,
        fill="yellow",
    )

    return img


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def format_duration(ms: int) -> str:
    """Format duration from milliseconds to mm:ss."""
    seconds = ms // 1000
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}:{secs:02d}"


def get_available_assets(
    output_dir: Path, original_video: Path | None
) -> list[tuple[str, Path]]:
    """Get list of available video assets from output directory.

    Returns:
        List of (display_name, path) tuples, including original video first.
    """
    assets = []

    # Always include original video first
    if original_video and original_video.exists():
        assets.append(("📹 Original Video", original_video))

    # Scan output directory for generated files
    if output_dir and output_dir.exists():
        # Common output file patterns from the pipeline
        patterns = [
            ("01_no-silence.mp4", "🔇 No Silence"),
            ("02_speedup.mp4", "⚡ Speedup"),
            ("03_no-fillers.mp4", "🗑️ No Fillers"),
            ("04_fixed.mp4", "🔧 Fixed"),
            ("05_trimmed.mp4", "✂️ Trimmed"),
            ("06_enhanced.mp4", "🎤 Enhanced"),
            ("07_subtitled.mp4", "📝 With Subtitles"),
            ("08_with-music.mp4", "🎵 With Music"),
            ("09_final.mp4", "✅ Final"),
        ]

        for filename, label in patterns:
            filepath = output_dir / filename
            if filepath.exists():
                assets.append((f"{label} ({filename})", filepath))

    return assets


def get_next_output_number(output_dir: Path) -> int:
    """Get the next available number for output files (e.g., 01_, 02_, etc.).

    Returns:
        Next available number (1-based), or 1 if no numbered files exist.
    """
    if not output_dir or not output_dir.exists():
        return 1

    # Find all numbered files (pattern: NN_filename.mp4)
    numbered_files = []
    for file in output_dir.glob("*.mp4"):
        name = file.name
        # Check if file starts with two digits followed by underscore
        if len(name) >= 4 and name[0:2].isdigit() and name[2] == "_":
            try:
                num = int(name[0:2])
                numbered_files.append(num)
            except ValueError:
                continue

    if not numbered_files:
        return 1

    return max(numbered_files) + 1


def get_latest_video(output_dir: Path) -> Path | None:
    """Get the most recently modified video file from output directory.

    Returns:
        Path to latest video, or None if no videos found.
    """
    if not output_dir or not output_dir.exists():
        return None

    videos = list(output_dir.glob("*.mp4"))
    if not videos:
        return None

    # Sort by modification time, most recent first
    videos.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return videos[0]


def add_event(
    operation: str,
    input_file: str,
    input_args: dict,
    output_file: str,
    video_duration: float | None = None,
    processing_time: float | None = None,
):
    """Add an event to the history.

    Args:
        operation: Name of the operation
        input_file: Input video file path
        input_args: Dictionary of input arguments
        output_file: Output video file path
        video_duration: Duration of the output video in seconds (optional)
        processing_time: Time taken to process in seconds (optional)
    """
    import time

    event = {
        "timestamp": time.time(),
        "operation": operation,
        "input_file": input_file,
        "input_args": input_args,
        "output_file": output_file,
        "duration": video_duration,  # Video duration
        "processing_time": processing_time,  # How long the operation took
    }
    st.session_state.events.append(event)
    # Keep only last 50 events
    if len(st.session_state.events) > 50:
        st.session_state.events = st.session_state.events[-50:]


def main():
    st.set_page_config(
        page_title="Talking-Head Short-Form Creator", page_icon="🎬", layout="wide"
    )
    st.title("🎬 Talking-Head Short-Form Creator")

    # Initialize session state
    if "title_suggestions" not in st.session_state:
        st.session_state.title_suggestions = []
    if "music_results" not in st.session_state:
        st.session_state.music_results = []
    if "bookmarked_music" not in st.session_state:
        st.session_state.bookmarked_music = []
    if "first_frame" not in st.session_state:
        st.session_state.first_frame = None
    if "events" not in st.session_state:
        st.session_state.events = []

    # Load preferences from profile.json (persists across cache clears)
    profile = load_profile()
    for key, value in profile.items():
        if key in PROFILE_KEYS and key not in st.session_state:
            st.session_state[key] = value
    # Ensure bookmarked_music is a list (legacy profile may have missing key)
    if "bookmarked_music" not in st.session_state:
        st.session_state.bookmarked_music = []
    elif not isinstance(st.session_state.bookmarked_music, list):
        st.session_state.bookmarked_music = []

    # Sidebar
    with st.sidebar:
        st.header("📁 Input")

        # Just use path input - no file uploader
        video_path_input = st.text_input(
            "Video file path",
            placeholder="/path/to/video.mp4",
            help="Enter the full path to your video file",
        )

        video_path = None
        if video_path_input:
            video_path = Path(video_path_input).expanduser().resolve()
            if video_path.exists():
                st.success(f"✅ {video_path.name}")
                try:
                    duration = get_video_duration(video_path)
                    st.info(f"Duration: {duration:.1f}s")
                except Exception:
                    duration = None

                # Track when a new video is loaded
                if st.session_state.get("loaded_video_path") != str(video_path):
                    with st.spinner("Loading frame..."):
                        st.session_state.first_frame = extract_first_frame(video_path)
                        st.session_state.loaded_video_path = str(video_path)

                    # Transcribe video
                    filler_words = {
                        "uh",
                        "um",
                        "mhmm",
                        "mm-mm",
                        "uh-uh",
                        "uh-huh",
                        "nuh-uh",
                    }
                    transcript_key = f"transcript_{str(video_path)}"
                    if transcript_key not in st.session_state:
                        with st.spinner("Transcribing..."):
                            try:
                                import asyncio as async_io

                                result = async_io.run(get_transcript(video_path))
                                words = result.get("words", [])
                                word_count = len(words)

                                # Count filler words (case-insensitive)
                                filler_count = 0
                                for word_obj in words:
                                    word_text = word_obj.get("word", "").lower().strip()
                                    if word_text in filler_words:
                                        filler_count += 1

                                # Calculate silence (gaps between words > 0.5s threshold, same as remove_silence_task)
                                silence_duration = 0.0
                                gap_threshold = (
                                    0.5  # Same as remove_silence_task default
                                )
                                if words and duration:
                                    # Calculate gaps between consecutive words
                                    for i in range(len(words) - 1):
                                        gap = words[i + 1]["start"] - words[i]["end"]
                                        if gap > gap_threshold:
                                            silence_duration += (
                                                gap - gap_threshold
                                            )  # Only count excess silence

                                    # Also account for silence at start/end
                                    if words:
                                        # Silence before first word (if any)
                                        if words[0]["start"] > gap_threshold:
                                            silence_duration += (
                                                words[0]["start"] - gap_threshold
                                            )
                                        # Silence after last word
                                        if words[-1]["end"] < duration:
                                            end_silence = duration - words[-1]["end"]
                                            if end_silence > gap_threshold:
                                                silence_duration += (
                                                    end_silence - gap_threshold
                                                )

                                # Calculate WPM as if silence has already been removed
                                wpm = 0.0
                                if words and duration:
                                    duration_without_silence = (
                                        duration - silence_duration
                                    )
                                    if duration_without_silence > 0:
                                        duration_minutes = (
                                            duration_without_silence / 60.0
                                        )
                                        wpm = (
                                            word_count / duration_minutes
                                            if duration_minutes > 0
                                            else 0.0
                                        )

                                st.session_state[transcript_key] = {
                                    "word_count": word_count,
                                    "filler_count": filler_count,
                                    "silence_duration": silence_duration,
                                    "wpm": wpm,
                                }
                            except Exception as e:
                                st.session_state[transcript_key] = {
                                    "word_count": None,
                                    "filler_count": None,
                                    "silence_duration": None,
                                    "wpm": None,
                                    "error": str(e),
                                }

                    # Display transcript banner
                    transcript_data = st.session_state.get(transcript_key)
                    if transcript_data:
                        if transcript_data.get("error"):
                            st.warning(
                                f"⚠️ Transcript error: {transcript_data['error']}"
                            )
                        elif transcript_data.get("word_count") is not None:
                            word_count = transcript_data["word_count"]
                            filler_count = transcript_data["filler_count"]
                            silence_duration = transcript_data.get(
                                "silence_duration", 0.0
                            )
                            wpm = transcript_data.get("wpm", 0.0)

                            # Format silence duration
                            silence_str = ""
                            if silence_duration and silence_duration > 0:
                                if silence_duration < 60:
                                    silence_str = f" • {int(silence_duration)}s silence"
                                else:
                                    minutes = int(silence_duration // 60)
                                    seconds = int(silence_duration % 60)
                                    silence_str = f" • {minutes}m {seconds}s silence"

                            # Format WPM
                            wpm_str = ""
                            if wpm and wpm > 0:
                                wpm_str = f" • {wpm:.0f} WPM"

                            st.info(
                                f"📝 Transcribed: {word_count} words ({filler_count} filler){silence_str}{wpm_str}"
                            )

                    # Record "Chose Initial Video" event
                    add_event(
                        "Chose Initial Video",
                        str(video_path),  # Input is the video itself
                        {"path": str(video_path)},
                        str(video_path),  # Output is also the video (no processing yet)
                        video_duration=duration,
                        processing_time=None,  # No processing time for initial selection
                    )
                else:
                    # Video already loaded, just show transcript if available
                    transcript_key = f"transcript_{str(video_path)}"
                    transcript_data = st.session_state.get(transcript_key)
                    if (
                        transcript_data
                        and transcript_data.get("word_count") is not None
                    ):
                        word_count = transcript_data["word_count"]
                        filler_count = transcript_data["filler_count"]
                        silence_duration = transcript_data.get("silence_duration", 0.0)
                        wpm = transcript_data.get("wpm", 0.0)

                        # Format silence duration
                        silence_str = ""
                        if silence_duration and silence_duration > 0:
                            if silence_duration < 60:
                                silence_str = f" • {int(silence_duration)}s silence"
                            else:
                                minutes = int(silence_duration // 60)
                                seconds = int(silence_duration % 60)
                                silence_str = f" • {minutes}m {seconds}s silence"

                        # Format WPM
                        wpm_str = ""
                        if wpm and wpm > 0:
                            wpm_str = f" • {wpm:.0f} WPM"

                        st.info(
                            f"📝 Transcribed: {word_count} words ({filler_count} filler){silence_str}{wpm_str}"
                        )
            else:
                st.error("❌ File not found")

        st.divider()
        st.header("🔄 Reset")
        if st.button(
            "Clear All & Start Fresh", type="secondary", use_container_width=True
        ):
            # Delete output folder and its contents
            if DEFAULT_OUTPUT_DIR.exists():
                try:
                    shutil.rmtree(DEFAULT_OUTPUT_DIR)
                except OSError:
                    pass
            # Clear session state (profile.json keeps preferences for next load)
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ============================================================================
    # MAIN LAYOUT: [Settings] [Steps] [Video] [Events] - all equal width
    # ============================================================================
    output_dir = DEFAULT_OUTPUT_DIR
    col_settings, col_steps, col_video, col_events = st.columns([1, 1, 1, 1])

    # Column 1: Settings
    with col_settings:
        st.header("⚙️ Settings")

        st.subheader("✏️ Title")

        # Helper to get current video for title generation
        def get_current_video_for_title():
            # Check if we have a loaded video path
            if st.session_state.get("loaded_video_path"):
                loaded_path = Path(st.session_state.loaded_video_path)
                if loaded_path.exists():
                    return loaded_path
            # Otherwise check if video_path is set
            if video_path and video_path.exists():
                return video_path
            return None

        current_video_for_title = get_current_video_for_title()

        if st.button(
            "🔄 Generate Suggestions",
            disabled=current_video_for_title is None,
            help="Generate title suggestions from video transcript"
            if current_video_for_title
            else "Load a video first",
        ):
            if current_video_for_title:
                with st.spinner("Generating..."):
                    try:
                        import asyncio as async_io

                        st.session_state.title_suggestions = async_io.run(
                            get_title_suggestions(
                                current_video_for_title, count=3, verbose=False
                            )
                        )
                    except Exception as e:
                        st.error(str(e))
                        import traceback

                        st.code(traceback.format_exc())

        if st.session_state.title_suggestions:
            st.session_state.selected_title = st.radio(
                "Select:", st.session_state.title_suggestions
            )

        custom = st.text_area(
            "Or custom title:",
            height=100,
            placeholder="Enter title (multiple lines allowed)",
            key="custom_title_input",
        )
        if custom:
            st.session_state.selected_title = custom.strip()

        st.subheader("🎵 Music")

        # Bookmarked tracks (persist in profile)
        bookmarked = st.session_state.get("bookmarked_music") or []
        if bookmarked:
            st.caption("📌 Bookmarked")
            for bi, track in enumerate(bookmarked):
                with st.expander(
                    f"📌 {track.get('title', 'Unknown')} - {track.get('creator', 'Unknown')}",
                    expanded=False,
                ):
                    st.caption(f"License: {track.get('license', 'Unknown')}")
                    duration_str = (
                        format_duration(track.get("duration", 0))
                        if track.get("duration")
                        else "?"
                    )
                    st.caption(f"Duration: {duration_str}")
                    audio_url = track.get("url", "")
                    if audio_url:
                        st.audio(audio_url, format="audio/mpeg")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button(
                            "Select",
                            key=f"select_bookmarked_{bi}",
                            use_container_width=True,
                        ):
                            st.session_state.selected_music = track
                            st.rerun()
                    with col_b:
                        if st.button(
                            "Remove bookmark",
                            key=f"unbookmark_{bi}",
                            use_container_width=True,
                        ):
                            st.session_state.bookmarked_music = [
                                t
                                for t in st.session_state.bookmarked_music
                                if t.get("url") != track.get("url")
                            ]
                            st.rerun()

        music_query = st.text_input("Search", value="hip hop", key="music_query")
        if st.button("🔍 Search"):
            with st.spinner("Searching..."):
                try:
                    import asyncio as async_io

                    st.session_state.music_results = async_io.run(
                        search_music(music_query, 5)
                    )
                except Exception as e:
                    st.error(str(e))
                    import traceback

                    st.code(traceback.format_exc())

        if st.session_state.music_results:
            st.caption("Search results")
            for i, track in enumerate(st.session_state.music_results):
                with st.expander(
                    f"🎵 {track['title']} - {track['creator']}", expanded=(i == 0)
                ):
                    st.caption(f"License: {track.get('license', 'Unknown')}")
                    duration_str = (
                        format_duration(track.get("duration", 0))
                        if track.get("duration")
                        else "?"
                    )
                    st.caption(f"Duration: {duration_str}")
                    audio_url = track.get("url", "")
                    if audio_url:
                        st.audio(audio_url, format="audio/mpeg")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button(
                            "Select this track",
                            key=f"select_music_{i}",
                            use_container_width=True,
                        ):
                            st.session_state.selected_music = track
                            st.rerun()
                    with col_b:
                        already_bookmarked = any(
                            t.get("url") == track.get("url")
                            for t in (st.session_state.get("bookmarked_music") or [])
                        )
                        if st.button(
                            "📌 Bookmark" if not already_bookmarked else "✓ Bookmarked",
                            key=f"bookmark_music_{i}",
                            use_container_width=True,
                            disabled=already_bookmarked,
                        ):
                            if not already_bookmarked:
                                st.session_state.bookmarked_music = (
                                    st.session_state.get("bookmarked_music") or []
                                ) + [track]
                                st.rerun()

            if st.session_state.get("selected_music"):
                st.success(f"✅ Selected: {st.session_state.selected_music['title']}")

        st.subheader("📍 Text Positions")
        title_height = st.slider(
            "Title position (% from bottom)", 0, 100, 85, key="title_height"
        )
        caption_height = st.slider(
            "Subtitle position (% from bottom)", 0, 100, 15, key="caption_height"
        )
        word_replacements = st.text_input(
            "Word replacements",
            key="word_replacements",
            help="Replace words in subtitles. Format: original1:replacement1,original2:replacement2",
            placeholder="e.g., Marquee:Marky,Josiah:Josia",
        )

        st.subheader("🎬 Auto-Editor Settings")
        target_duration = st.number_input(
            "Target duration (s)", 10, 600, 60, key="full_pipeline_duration"
        )
        trim_tolerance = st.number_input(
            "Tolerance (±s)", 5, 60, 20, key="full_pipeline_tolerance"
        )
        word_count = st.number_input(
            "Words/caption", 1, 10, 3, key="full_pipeline_words"
        )
        st.subheader("✂️ Smart Trim Settings")
        custom_trim_prompt = st.text_area(
            "Custom edit prompt (optional)",
            key="custom_trim_prompt",
            help="Custom instructions for what edits to apply. If empty, uses default prompt.",
            placeholder="e.g., Remove all mentions of competitors, focus on benefits only",
        )
        speedup_mode = st.radio(
            "Speedup mode:",
            ["Manual (x)", "Target WPM"],
            key="speedup_mode",
            horizontal=True,
        )
        if speedup_mode == "Manual (x)":
            st.number_input(
                "Speedup (x)",
                1.0,
                3.0,
                1.0,
                0.1,
                key="speedup_value",
                help="1.0 = no speedup, 1.2 = 20% faster",
            )
        else:
            st.number_input(
                "Target WPM",
                100,
                300,
                180,
                10,
                key="target_wpm",
                help="Target words per minute (will calculate speedup automatically)",
            )

    # Column 2: Steps
    with col_steps:
        st.header("🔧 Steps")

        # Helper to get current video
        def get_current_video():
            # First check if a revert was set
            if "current_video" in st.session_state:
                revert_path = Path(st.session_state.current_video)
                if revert_path.exists():
                    return revert_path
            # Otherwise, use latest generated or original
            if output_dir:
                latest = get_latest_video(output_dir)
                if latest:
                    return latest
                elif video_path:
                    return video_path
            elif video_path:
                return video_path
            return None

        current_video = get_current_video()
        has_video = current_video is not None

        # Check transcript data for original video to determine if steps should be disabled
        has_silence = True  # Default to enabled if no transcript data yet
        has_filler_words = True  # Default to enabled if no transcript data yet
        if has_video and video_path:
            # Check transcript for original video
            transcript_key = f"transcript_{str(video_path)}"
            transcript_data = st.session_state.get(transcript_key)
            if transcript_data and transcript_data.get("word_count") is not None:
                silence_duration = transcript_data.get("silence_duration", 0.0)
                filler_count = transcript_data.get("filler_count", 0)
                has_silence = silence_duration and silence_duration > 0
                has_filler_words = filler_count and filler_count > 0

        # Step 1: Remove Silence
        if st.button(
            "1. 🔇 Remove Silence",
            key="btn_trim_silence",
            disabled=not has_video or not has_silence,
            use_container_width=True,
        ):
            silence_input = str(current_video)
            # Auto-generate output path
            if output_dir:
                next_num = get_next_output_number(output_dir)
                output_path = output_dir / f"{next_num:02d}_no-silence.mp4"
            else:
                output_path = (
                    Path(silence_input).parent
                    / f"{Path(silence_input).stem}-no-silence.mp4"
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Use process_video's remove_silence_task
            from process_video import remove_silence_task

            start_time = time.time()
            with st.spinner("Trimming silence..."):
                try:
                    # Ensure asyncio is available
                    import asyncio as async_io

                    result_path = async_io.run(
                        remove_silence_task(
                            Path(silence_input), output_path, margin=0.2
                        )
                    )
                    processing_time = time.time() - start_time
                    st.success(f"✅ Silence trimmed: {result_path}")
                    # Record event
                    try:
                        video_duration = get_video_duration(result_path)
                    except Exception:
                        video_duration = None
                    add_event(
                        "Trim Silence",
                        silence_input,
                        {"margin": "0.2s"},
                        str(result_path),
                        video_duration=video_duration,
                        processing_time=processing_time,
                    )
                    # Clear revert state so latest video becomes current
                    if "current_video" in st.session_state:
                        del st.session_state.current_video
                    st.rerun()  # Refresh to show new asset in dropdown
                except Exception as e:
                    st.error(f"❌ Failed: {e}")
                    import traceback

                    st.code(traceback.format_exc())

        # Step 2: Speedup
        speedup_mode = st.session_state.get("speedup_mode", "Manual (x)")
        speedup_value = st.session_state.get("speedup_value", 1.0)
        target_wpm = st.session_state.get("target_wpm", 180)

        # Determine if button should be disabled
        should_disable = not has_video
        if speedup_mode == "Manual (x)" and speedup_value <= 1.0:
            should_disable = True

        button_help = ""
        if speedup_mode == "Manual (x)":
            button_help = f"Speed up video by {speedup_value}x"
        else:
            button_help = f"Speed up video to {target_wpm} WPM"

        if st.button(
            "2. ⚡ Speedup",
            key="btn_speedup",
            disabled=should_disable,
            use_container_width=True,
            help=button_help,
        ):
            speedup_input = str(current_video)
            # Auto-generate output path
            if output_dir:
                next_num = get_next_output_number(output_dir)
                output_path = output_dir / f"{next_num:02d}_speedup.mp4"
            else:
                output_path = (
                    Path(speedup_input).parent
                    / f"{Path(speedup_input).stem}-speedup.mp4"
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                "python",
                "-m",
                "apply_speedup",
                speedup_input,
                str(output_path),
            ]

            if speedup_mode == "Manual (x)":
                cmd.extend(["--speedup", str(speedup_value)])
            else:
                cmd.extend(["--wpm", str(target_wpm)])
            start_time = time.time()
            with st.spinner(f"Speeding up video by {speedup_value}x..."):
                result = subprocess.run(
                    cmd, cwd=Path(__file__).parent, capture_output=True, text=True
                )
                processing_time = time.time() - start_time
                if result.returncode == 0:
                    st.success(f"✅ Video sped up: {output_path}")
                    # Record event
                    try:
                        video_duration = get_video_duration(output_path)
                    except Exception:
                        video_duration = None
                    add_event(
                        "Speedup",
                        speedup_input,
                        {"speedup": f"{speedup_value}x"},
                        str(output_path),
                        video_duration=video_duration,
                        processing_time=processing_time,
                    )
                    # Clear revert state so latest video becomes current
                    if "current_video" in st.session_state:
                        del st.session_state.current_video
                    st.rerun()  # Refresh to show new asset in dropdown
                else:
                    st.error("❌ Failed")
                    st.code(
                        result.stderr[-1000:]
                        if len(result.stderr) > 1000
                        else result.stderr
                    )

        # Step 3: Remove Filler Words
        if st.button(
            "3. 🗑️ Remove Filler Words",
            key="btn_remove_fillers",
            disabled=not has_video or not has_filler_words,
            use_container_width=True,
        ):
            filler_input = str(current_video)
            # Auto-generate output path
            if output_dir:
                next_num = get_next_output_number(output_dir)
                output_path = output_dir / f"{next_num:02d}_no-fillers.mp4"
            else:
                output_path = (
                    Path(filler_input).parent
                    / f"{Path(filler_input).stem}-no-fillers.mp4"
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                "python",
                "-m",
                "remove_filler_words",
                filler_input,
                str(output_path),
            ]
            start_time = time.time()
            with st.spinner("Removing filler words..."):
                result = subprocess.run(
                    cmd, cwd=Path(__file__).parent, capture_output=True, text=True
                )
                processing_time = time.time() - start_time
                if result.returncode == 0:
                    st.success(f"✅ Filler words removed: {output_path}")
                    # Record event
                    try:
                        video_duration = get_video_duration(output_path)
                    except Exception:
                        video_duration = None
                    add_event(
                        "Remove Filler Words",
                        filler_input,
                        {},
                        str(output_path),
                        video_duration=video_duration,
                        processing_time=processing_time,
                    )
                    # Clear revert state so latest video becomes current
                    if "current_video" in st.session_state:
                        del st.session_state.current_video
                    st.rerun()  # Refresh to show new asset in dropdown
                else:
                    st.error("❌ Failed")
                    st.code(
                        result.stderr[-1000:]
                        if len(result.stderr) > 1000
                        else result.stderr
                    )

        # Step 4: Smart Trim
        if st.button(
            "4. ✂️ Smart Trim",
            key="btn_smart_trim",
            disabled=not has_video,
            use_container_width=True,
            help="Remove mistakes, fumbles, retakes, ramblings, and asides. Tightens up the video without a duration target.",
        ):
            smart_trim_input = str(current_video)
            # Auto-generate output path
            if output_dir:
                next_num = get_next_output_number(output_dir)
                output_path = output_dir / f"{next_num:02d}_smart-trimmed.mp4"
            else:
                output_path = (
                    Path(smart_trim_input).parent
                    / f"{Path(smart_trim_input).stem}-tightened.mp4"
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            start_time = time.time()
            with st.spinner("Smart trimming (this may take a while)..."):
                # No --duration flag = tighten up mode (remove fluff without duration target)
                cmd = [
                    "python",
                    "-m",
                    "trim_smart",
                    smart_trim_input,
                    str(output_path),
                ]
                # Add custom prompt if provided
                custom_prompt = st.session_state.get("custom_trim_prompt", "").strip()
                if custom_prompt:
                    cmd.extend(["--prompt", custom_prompt])
                result = subprocess.run(
                    cmd, cwd=Path(__file__).parent, capture_output=True, text=True
                )
                processing_time = time.time() - start_time
                if result.returncode == 0:
                    st.success(f"✅ Smart trimmed: {output_path}")
                    # Record event
                    try:
                        video_duration = get_video_duration(output_path)
                    except Exception:
                        video_duration = None
                    add_event(
                        "Smart Trim",
                        smart_trim_input,
                        {"mode": "tighten up, no duration target"},
                        str(output_path),
                        video_duration=video_duration,
                        processing_time=processing_time,
                    )
                    # Clear revert state so latest video becomes current
                    if "current_video" in st.session_state:
                        del st.session_state.current_video
                    st.rerun()  # Refresh to show new asset in dropdown
                else:
                    st.error("❌ Failed")
                    st.code(
                        result.stderr[-1000:]
                        if len(result.stderr) > 1000
                        else result.stderr
                    )

        # Step 5: Add Subtitles
        if st.button(
            "5. 📝 Add Subtitles",
            key="btn_add_subtitles",
            disabled=not has_video,
            use_container_width=True,
        ):
            caption_input = str(current_video)
            # Auto-generate output path
            if output_dir:
                next_num = get_next_output_number(output_dir)
                output_path = output_dir / f"{next_num:02d}_with-subtitles.mp4"
            else:
                output_path = (
                    Path(caption_input).parent
                    / f"{Path(caption_input).stem}-subtitles.mp4"
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            caption_height_val = st.session_state.get("caption_height", 15)
            cmd = [
                "python",
                "-m",
                "add_subtitles",
                caption_input,
                "--output",
                str(output_path),
                "--height",
                str(caption_height_val),
            ]
            # Add word replacements if provided
            word_replacements = st.session_state.get("word_replacements", "").strip()
            if word_replacements:
                cmd.extend(["--replace", word_replacements])
            start_time = time.time()
            with st.spinner(f"Adding subtitles at {caption_height_val}%..."):
                result = subprocess.run(
                    cmd, cwd=Path(__file__).parent, capture_output=True, text=True
                )
                processing_time = time.time() - start_time
                if result.returncode == 0:
                    # Record event
                    try:
                        video_duration = get_video_duration(output_path)
                    except Exception:
                        video_duration = None
                    add_event(
                        "Add Captions",
                        caption_input,
                        {"replacements": word_replacements}
                        if word_replacements
                        else {},
                        str(output_path),
                        video_duration=video_duration,
                        processing_time=processing_time,
                    )
                    # Clear revert state so latest video becomes current
                    if "current_video" in st.session_state:
                        del st.session_state.current_video
                    st.success(f"✅ Subtitles added: {output_path}")
                    st.rerun()  # Refresh to show new asset in dropdown
                else:
                    st.error("❌ Failed")
                    st.code(
                        result.stderr[-1000:]
                        if len(result.stderr) > 1000
                        else result.stderr
                    )

        # Step 6: Add Title
        if st.button(
            "6. 📌 Add Title",
            key="btn_add_title",
            disabled=not has_video,
            use_container_width=True,
        ):
            title_input = str(current_video)
            selected_title = st.session_state.get("selected_title")

            # Get title height from settings
            title_height = st.session_state.get("title_height", 85)

            # Auto-generate output path
            if output_dir:
                next_num = get_next_output_number(output_dir)
                output_path = output_dir / f"{next_num:02d}_with-title.mp4"
            else:
                output_path = (
                    Path(title_input).parent / f"{Path(title_input).stem}-title.mp4"
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = ["python", "-m", "add_title", title_input]
            # Only append title if one is selected (if None, script will auto-generate)
            if selected_title:
                cmd.append(selected_title)
            cmd.append(str(output_path))
            cmd.extend(["--height", str(title_height)])
            start_time = time.time()
            with st.spinner("Adding title..."):
                result = subprocess.run(
                    cmd, cwd=Path(__file__).parent, capture_output=True, text=True
                )
                processing_time = time.time() - start_time
                if result.returncode == 0:
                    st.success(f"✅ Title added: {output_path}")
                    # Record event
                    try:
                        video_duration = get_video_duration(output_path)
                    except Exception:
                        video_duration = None
                    title_info = (
                        f"title={selected_title[:50]}"
                        if selected_title
                        else "auto-generated"
                    )
                    title_info = selected_title if selected_title else "auto-generated"
                    add_event(
                        "Add Title",
                        title_input,
                        {"title": title_info},
                        str(output_path),
                        video_duration=video_duration,
                        processing_time=processing_time,
                    )
                    # Clear revert state so latest video becomes current
                    if "current_video" in st.session_state:
                        del st.session_state.current_video
                    st.rerun()  # Refresh to show new asset in dropdown
                else:
                    st.error("❌ Failed")
                    st.code(
                        result.stderr[-1000:]
                        if len(result.stderr) > 1000
                        else result.stderr
                    )

        # Step 7: Add Music
        if st.button(
            "7. 🎵 Add Music",
            key="btn_add_music",
            disabled=not has_video,
            use_container_width=True,
        ):
            music_input = str(current_video)
            selected_music = st.session_state.get("selected_music")
            music_file = None
            # Auto-generate output path
            if output_dir:
                next_num = get_next_output_number(output_dir)
                output_path = output_dir / f"{next_num:02d}_with-music.mp4"
            else:
                output_path = (
                    Path(music_input).parent / f"{Path(music_input).stem}-music.mp4"
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # If music file provided, use it; otherwise use selected music URL
            if music_file:
                # Use local file
                cmd = [
                    "python",
                    "-m",
                    "add_background_music",
                    music_input,
                    music_file,
                    str(output_path),
                ]
            elif selected_music and selected_music.get("url"):
                # Download and use selected music
                import tempfile
                import asyncio
                from add_background_music import download_music

                with st.spinner("Downloading selected music..."):
                    with tempfile.NamedTemporaryFile(
                        suffix=".mp3", delete=False
                    ) as tmp:
                        music_path = Path(tmp.name)
                    try:
                        asyncio.run(download_music(selected_music["url"], music_path))
                        cmd = [
                            "python",
                            "-m",
                            "add_background_music",
                            music_input,
                            str(music_path),
                            str(output_path),
                        ]
                    except Exception as e:
                        st.error(f"Failed to download music: {e}")
                        return
            else:
                st.error("Please select music in the Settings section above.")
                return

            start_time = time.time()
            with st.spinner("Adding music..."):
                result = subprocess.run(
                    cmd, cwd=Path(__file__).parent, capture_output=True, text=True
                )
                processing_time = time.time() - start_time
                if result.returncode == 0:
                    st.success(f"✅ Music added: {output_path}")
                    # Record event
                    try:
                        video_duration = get_video_duration(output_path)
                    except Exception:
                        video_duration = None
                    music_info = (
                        f"file={Path(music_file).name}"
                        if music_file
                        else f"track={selected_music['title']}"
                        if selected_music
                        else "unknown"
                    )
                    add_event(
                        "Add Music",
                        music_input,
                        {"music": music_info},
                        str(output_path),
                        video_duration=video_duration,
                        processing_time=processing_time,
                    )
                    # Clear revert state so latest video becomes current
                    if "current_video" in st.session_state:
                        del st.session_state.current_video
                    # Clean up temp file if we downloaded it
                    if not music_file and selected_music:
                        music_path.unlink(missing_ok=True)
                    st.rerun()  # Refresh to show new asset in dropdown
                else:
                    st.error("❌ Failed")
                    st.code(
                        result.stderr[-1000:]
                        if len(result.stderr) > 1000
                        else result.stderr
                    )

        # Step 8: Enhance Voice
        if st.button(
            "8. 🎤 Enhance Voice",
            key="btn_enhance_voice",
            disabled=not has_video,
            use_container_width=True,
        ):
            enhance_input = str(current_video)
            # Auto-generate output path
            if output_dir:
                next_num = get_next_output_number(output_dir)
                output_path = output_dir / f"{next_num:02d}_enhanced.mp4"
            else:
                output_path = (
                    Path(enhance_input).parent
                    / f"{Path(enhance_input).stem}-enhanced.mp4"
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                "python",
                "-m",
                "enhance_voice",
                enhance_input,
                str(output_path),
            ]
            start_time = time.time()
            with st.spinner("Enhancing voice..."):
                result = subprocess.run(
                    cmd, cwd=Path(__file__).parent, capture_output=True, text=True
                )
                processing_time = time.time() - start_time
                if result.returncode == 0:
                    st.success(f"✅ Voice enhanced: {output_path}")
                    # Record event
                    try:
                        video_duration = get_video_duration(output_path)
                    except Exception:
                        video_duration = None
                    add_event(
                        "Enhance Voice",
                        enhance_input,
                        {},
                        str(output_path),
                        video_duration=video_duration,
                        processing_time=processing_time,
                    )
                    # Clear revert state so latest video becomes current
                    if "current_video" in st.session_state:
                        del st.session_state.current_video
                    st.rerun()  # Refresh to show new asset in dropdown
                else:
                    st.error("❌ Failed")
                    st.code(
                        result.stderr[-1000:]
                        if len(result.stderr) > 1000
                        else result.stderr
                    )

        st.divider()

        # Apply All button
        if st.button(
            "▶️ Apply All",
            type="primary",
            key="btn_apply_all",
            disabled=not has_video,
            use_container_width=True,
        ):
            title = st.session_state.get("selected_title")
            music = st.session_state.get("selected_music")
            if not title:
                st.error("Select a title first (in Settings section)")
            elif not music:
                st.error("Select music first (in Settings section)")
            else:
                output_path = output_dir
                output_path.mkdir(parents=True, exist_ok=True)
                # Get current settings values
                current_title_height = st.session_state.get("title_height", 85)
                current_caption_height = st.session_state.get("caption_height", 15)
                target_duration = st.session_state.get("full_pipeline_duration", 60)
                trim_tolerance = st.session_state.get("full_pipeline_tolerance", 20)
                word_count = st.session_state.get("full_pipeline_words", 3)
                speedup_mode = st.session_state.get("speedup_mode", "Manual (x)")
                speedup_value = st.session_state.get("speedup_value", 1.0)
                target_wpm = st.session_state.get("target_wpm", 180)

                # Use speedup value for Apply All (WPM mode not supported in full pipeline yet)
                speedup_for_pipeline = (
                    speedup_value if speedup_mode == "Manual (x)" else 1.0
                )

                cmd = [
                    "python",
                    "-m",
                    "main",
                    "run_autoedit",
                    "--input",
                    str(video_path),
                    "--output",
                    str(output_path),
                    "--title",
                    title,
                    "--music_url",
                    music["url"],
                    "--target_duration",
                    str(target_duration),
                    "--trim_tolerance",
                    str(trim_tolerance),
                    "--caption_word_count",
                    str(word_count),
                    "--speedup",
                    str(speedup_for_pipeline),
                    "--caption_height",
                    str(current_caption_height),
                    "--title_height",
                    str(current_title_height),
                ]
                st.code(" ".join(cmd))
                with st.spinner("Processing... (may take several minutes)"):
                    result = subprocess.run(
                        cmd, cwd=Path(__file__).parent, capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        st.success("✅ Done!")
                        st.code(
                            result.stdout[-2000:]
                            if len(result.stdout) > 2000
                            else result.stdout
                        )
                    else:
                        st.error("❌ Failed")
                        st.code(
                            result.stderr[-2000:]
                            if len(result.stderr) > 2000
                            else result.stderr
                        )

    # Column 3: Current Video
    with col_video:
        st.header("🎥 Current Video")
        # Add CSS to limit video height
        st.markdown(
            """
            <style>
            div[data-testid="stVideo"] {
                max-height: 600px;
                overflow: auto;
            }
            div[data-testid="stVideo"] video {
                max-height: 600px;
                width: auto;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Determine current video: check revert first, then latest generated or original input
        current_display_video = None
        # First check if a revert was set
        if "current_video" in st.session_state:
            revert_path = Path(st.session_state.current_video)
            if revert_path.exists():
                current_display_video = revert_path

        # If no revert, use latest generated or original
        if not current_display_video:
            if output_dir:
                latest_video = get_latest_video(output_dir)
                if latest_video:
                    current_display_video = latest_video
                elif video_path and video_path.exists():
                    current_display_video = video_path
            elif video_path and video_path.exists():
                current_display_video = video_path

        if current_display_video:
            st.success(f"📹 Current: `{current_display_video.name}`")
            try:
                duration = get_video_duration(current_display_video)
                st.caption(f"Duration: {duration:.1f}s")
            except Exception:
                pass

            # Show video player
            with open(current_display_video, "rb") as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)
        else:
            st.info("👆 Enter video path in sidebar to view it here.")

    # Column 4: Events History
    with col_events:
        st.header("📋 Events History")

        if st.session_state.events:
            # Show events in reverse order (newest first)
            for i, event in enumerate(reversed(st.session_state.events)):
                output_file = event.get("output_file") or event.get("output", "")
                input_file = event.get("input_file") or event.get("input", "")
                with st.expander(
                    f"{event['operation']} - {Path(output_file).name}",
                    expanded=(i == 0),
                ):
                    # Special display for "Chose Initial Video"
                    if event["operation"] == "Chose Initial Video":
                        st.markdown(f"**Path:** `{input_file}`")
                    else:
                        st.markdown(f"**Input:** `{Path(input_file).name}`")
                        args = event.get("input_args") or event.get("args", {})
                        if args:
                            if isinstance(args, dict):
                                args_str = ", ".join(
                                    [f"{k}={v}" for k, v in args.items()]
                                )
                            else:
                                args_str = str(args)
                            st.markdown(f"**Args:** {args_str}")
                        st.markdown(f"**Output:** `{Path(output_file).name}`")
                    if event.get("duration"):
                        st.markdown(f"**Duration:** {event['duration']:.1f}s")
                    if event.get("processing_time"):
                        processing_time = event["processing_time"]
                        if processing_time < 60:
                            time_str = f"{processing_time:.1f}s"
                        else:
                            minutes = int(processing_time // 60)
                            seconds = processing_time % 60
                            time_str = f"{minutes}m {seconds:.1f}s"
                        st.markdown(f"**Time:** {time_str}")
                    if event.get("timestamp"):
                        from datetime import datetime

                        dt = datetime.fromtimestamp(event["timestamp"])
                        st.caption(f"Timestamp: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                    # Don't show revert button for the most recent event (i == 0)
                    if i > 0:
                        if st.button(
                            "Revert to here",
                            key=f"revert_{len(st.session_state.events) - 1 - i}",
                            use_container_width=True,
                        ):
                            # Set this output as the "current" video
                            revert_path = Path(output_file)
                            if revert_path.exists():
                                # Calculate the actual index in the events list
                                # i is the position in reversed list (0 = newest)
                                # Actual index = len(events) - 1 - i
                                revert_index = len(st.session_state.events) - 1 - i

                                # Remove all events after this one
                                # Keep events up to and including revert_index
                                st.session_state.events = st.session_state.events[
                                    : revert_index + 1
                                ]

                                st.session_state.current_video = str(revert_path)
                                st.success(f"✅ Reverted to: {revert_path.name}")
                                st.rerun()
                            else:
                                st.error(f"File not found: {revert_path}")
                    else:
                        st.caption("📍 Current state")
        else:
            st.info("No events yet. Run a step to see history here.")

    # Persist preferences to profile.json (survives Clear All / cache clear)
    save_profile()


if __name__ == "__main__":
    main()
