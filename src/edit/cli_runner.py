#!/usr/bin/env python3
"""
CLI runner for video processing pipeline with real-time streaming.

Streams events, logs, and handles user input for paused flows.

Usage:
    # List recent flow runs
    python cli_runner.py list

    # Stream an existing flow run
    python cli_runner.py stream <flow_run_id>

    # Run a new video processing pipeline
    python cli_runner.py run -i input.MOV -o ./output

    # Run with auto-select (no user input)
    python cli_runner.py run -i input.MOV -o ./output --auto
"""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from prefect import get_client
from prefect.client.schemas.filters import (
    LogFilter,
    LogFilterFlowRunId,
    LogFilterTimestamp,
    TaskRunFilter,
    TaskRunFilterFlowRunId,
)
from prefect.client.schemas.objects import Log
from prefect.events import Event
from prefect.states import StateType

# Task order for stepper display
TASK_ORDER = [
    "remove_silence_task",
    "get_transcript_task",
    "fix_rotation_task",
    "remove_filler_words_task",
    "smart_trim_task",
    "enhance_voice_task",
    "add_subtitles_task",
    "add_background_music_task",
    "add_title_task",
]

# ANSI colors
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"


def clear_screen() -> None:
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")


def move_cursor(row: int, col: int) -> None:
    """Move cursor to position."""
    print(f"\033[{row};{col}H", end="")


def state_icon(state_type: StateType | None) -> str:
    """Get icon for state."""
    if state_type is None:
        return f"{DIM}○{RESET}"
    match state_type:
        case StateType.COMPLETED:
            return f"{GREEN}✓{RESET}"
        case StateType.RUNNING:
            return f"{YELLOW}◐{RESET}"
        case StateType.FAILED:
            return f"{RED}✗{RESET}"
        case StateType.PENDING:
            return f"{DIM}○{RESET}"
        case StateType.PAUSED:
            return f"{MAGENTA}⏸{RESET}"
        case StateType.CANCELLED:
            return f"{RED}⊘{RESET}"
        case _:
            return f"{DIM}?{RESET}"


def format_task_name(name: str) -> str:
    """Format task name for display."""
    # Remove _task suffix and any trailing numbers
    base = name.split("-")[0]
    if base.endswith("_task"):
        base = base[:-5]
    return base.replace("_", " ").title()


async def get_flow_state(client: Any, flow_run_id: UUID) -> dict:
    """Get current flow run state and task states."""
    flow_run = await client.read_flow_run(flow_run_id)

    task_runs = await client.read_task_runs(
        task_run_filter=TaskRunFilter(
            flow_run_id=TaskRunFilterFlowRunId(any_=[flow_run_id])
        )
    )

    # Map task names to states
    task_states = {}
    for tr in task_runs:
        base_name = tr.name.split("-")[0]
        task_states[base_name] = {
            "name": tr.name,
            "state": tr.state,
            "state_type": tr.state.type if tr.state else None,
        }

    return {
        "flow_run": flow_run,
        "task_states": task_states,
        "is_paused": flow_run.state and flow_run.state.type == StateType.PAUSED,
        "is_terminal": flow_run.state
        and flow_run.state.type
        in (StateType.COMPLETED, StateType.FAILED, StateType.CANCELLED),
    }


async def get_new_logs(
    client: Any, flow_run_id: UUID, since: datetime
) -> tuple[list[Any], datetime]:
    """Get logs since timestamp."""
    logs = await client.read_logs(
        log_filter=LogFilter(
            flow_run_id=LogFilterFlowRunId(any_=[flow_run_id]),
            timestamp=LogFilterTimestamp(after_=since),
        ),
        limit=100,
        sort="TIMESTAMP_ASC",
    )

    new_since = since
    if logs:
        new_since = max(log.timestamp for log in logs)

    return logs, new_since


async def get_pending_input(client: Any, flow_run_id: UUID) -> dict | None:
    """Check for pending user input."""
    try:
        inputs = await client.filter_flow_run_input(flow_run_id=flow_run_id)
        if inputs:
            return inputs[0]
    except Exception:
        pass
    return None


async def display_stepper(task_states: dict, show: bool = True) -> int:
    """Display stepper diagram. Returns number of lines printed."""
    if not show:
        return 0
    lines = []
    lines.append(f"\n{BOLD}═══ Pipeline Progress ═══{RESET}\n")

    for task_name in TASK_ORDER:
        state_info = task_states.get(task_name, {})
        state_type = state_info.get("state_type")
        icon = state_icon(state_type)
        display_name = format_task_name(task_name)

        # Add state name if running or failed
        suffix = ""
        if state_type == StateType.RUNNING:
            suffix = f" {DIM}(running...){RESET}"
        elif state_type == StateType.FAILED:
            msg = state_info.get("state", {})
            if hasattr(msg, "message") and msg.message:
                suffix = f" {RED}({msg.message[:30]}...){RESET}"

        lines.append(f"  {icon}  {display_name}{suffix}")

    lines.append("")
    for line in lines:
        print(line)
    return len(lines)


def display_logs(logs: list[Any], max_lines: int = 15, show: bool = True) -> None:
    """Display recent logs."""
    if not show:
        return
    print(f"\n{BOLD}═══ Recent Logs ═══{RESET}\n")

    recent = logs[-max_lines:] if len(logs) > max_lines else logs

    for log in recent:
        level = log.level
        if level >= 40:  # ERROR
            color = RED
            level_name = "ERR"
        elif level >= 30:  # WARNING
            color = YELLOW
            level_name = "WRN"
        elif level >= 20:  # INFO
            color = BLUE
            level_name = "INF"
        else:
            color = DIM
            level_name = "DBG"

        timestamp = log.timestamp.strftime("%H:%M:%S")
        msg = log.message[:80] + "..." if len(log.message) > 80 else log.message
        print(f"  {DIM}{timestamp}{RESET} {color}[{level_name}]{RESET} {msg}")


async def stream_flow_run_websocket(
    flow_run_id: UUID, ui_mode: set[str] | None = None
) -> None:
    """Stream events and logs using WebSocket (real-time).

    Args:
        ui_mode: Set of UI elements to show: 'stepper', 'logs', 'progress'. None = all.
    """
    from prefect.events.subscribers import FlowRunSubscriber

    if ui_mode is None:
        ui_mode = {"stepper", "logs"}

    print(f"\n{BOLD}{CYAN}Streaming flow run (WebSocket): {flow_run_id}{RESET}")
    print(f"{DIM}Press Ctrl+C to exit{RESET}\n")

    all_logs: list[Any] = []
    task_states: dict = {}

    async with get_client() as client:
        # Get initial state
        state = await get_flow_state(client, flow_run_id)
        task_states = state["task_states"]

        try:
            async with FlowRunSubscriber(flow_run_id=flow_run_id) as subscriber:
                async for item in subscriber:
                    if isinstance(item, Log):
                        all_logs.append(item)
                    elif isinstance(item, Event):
                        # Update task states based on event
                        event_name = item.event
                        if "task-run" in event_name:
                            # Refresh task states
                            state = await get_flow_state(client, flow_run_id)
                            task_states = state["task_states"]

                    # Redraw UI
                    clear_screen()
                    await display_stepper(task_states, show="stepper" in ui_mode)
                    display_logs(all_logs, show="logs" in ui_mode)

        except Exception as e:
            if "websocket" in str(e).lower():
                print(
                    f"\n{YELLOW}WebSocket not available, falling back to polling...{RESET}\n"
                )
                await stream_flow_run_polling(flow_run_id)
            else:
                raise

    # Final state check
    state = await get_flow_state(client, flow_run_id)
    if state["is_terminal"]:
        flow_run = state["flow_run"]
        if flow_run.state.type == StateType.COMPLETED:
            print(f"\n{GREEN}{BOLD}✓ Flow completed successfully!{RESET}\n")
        elif flow_run.state.type == StateType.FAILED:
            print(f"\n{RED}{BOLD}✗ Flow failed: {flow_run.state.message}{RESET}\n")
        else:
            print(f"\n{YELLOW}{BOLD}⊘ Flow cancelled{RESET}\n")


async def stream_flow_run_polling(
    flow_run_id: UUID, ui_mode: set[str] | None = None
) -> None:
    """Stream events and logs using polling (fallback).

    Args:
        ui_mode: Set of UI elements to show: 'stepper', 'logs', 'progress'. None = all.
    """
    if ui_mode is None:
        ui_mode = {"stepper", "logs"}

    print(f"\n{BOLD}{CYAN}Streaming flow run (polling): {flow_run_id}{RESET}")
    print(f"{DIM}Press Ctrl+C to exit{RESET}\n")

    all_logs: list[Any] = []
    log_since = datetime.now(timezone.utc) - timedelta(hours=1)

    async with get_client() as client:
        while True:
            try:
                # Get current state
                state = await get_flow_state(client, flow_run_id)

                # Get new logs
                new_logs, log_since = await get_new_logs(client, flow_run_id, log_since)
                all_logs.extend(new_logs)

                # Clear and redraw
                clear_screen()

                # Display stepper
                await display_stepper(state["task_states"], show="stepper" in ui_mode)

                # Display logs
                display_logs(all_logs, show="logs" in ui_mode)

                # Check for terminal state
                if state["is_terminal"]:
                    flow_run = state["flow_run"]
                    if flow_run.state.type == StateType.COMPLETED:
                        print(f"\n{GREEN}{BOLD}✓ Flow completed successfully!{RESET}\n")
                    elif flow_run.state.type == StateType.FAILED:
                        print(
                            f"\n{RED}{BOLD}✗ Flow failed: {flow_run.state.message}{RESET}\n"
                        )
                    else:
                        print(f"\n{YELLOW}{BOLD}⊘ Flow cancelled{RESET}\n")
                    break

                # Wait before next poll
                await asyncio.sleep(1.0)

            except KeyboardInterrupt:
                print(f"\n\n{YELLOW}Interrupted by user{RESET}\n")
                break
            except Exception as e:
                print(f"\n{RED}Error: {e}{RESET}")
                await asyncio.sleep(2.0)


async def stream_flow_run(
    flow_run_id: UUID, use_websocket: bool = True, ui_mode: set[str] | None = None
) -> None:
    """Stream events and logs for a flow run.

    Args:
        ui_mode: Set of UI elements to show: 'stepper', 'logs', 'progress'. None = all.
    """
    if use_websocket:
        try:
            await stream_flow_run_websocket(flow_run_id, ui_mode)
        except Exception as e:
            print(f"{YELLOW}WebSocket failed: {e}, using polling{RESET}")
            await stream_flow_run_polling(flow_run_id, ui_mode)
    else:
        await stream_flow_run_polling(flow_run_id, ui_mode)


async def start_and_stream(
    input_video: Path,
    output_path: Path,
    duration: int = 60,
    title: str | None = None,
    music_url: str | None = None,
    music_tags: str = "hip hop",
    speedup: float = 1.0,
    ui_mode: set[str] | None = None,
    word_count: int = 3,
    font_size: int = 14,
    silence_margin: float = 0.2,
    trim_tolerance: float = 20.0,
    caption_height: int | None = None,
    title_height: int | None = None,
    brand_brief: str | None = None,
    skip_steps: set[str] | None = None,
    crop_top: int = 0,
    crop_bottom: int = 0,
    crop_left: int = 0,
    crop_right: int = 0,
) -> None:
    """Start a new flow run and stream it with our UI.

    Args:
        ui_mode: Set of UI elements to show: 'stepper', 'logs', 'progress'.
                 Empty set or None = all. If 'none' or 'quiet' in set, no streaming.
    """
    import sys

    print(
        f"{RED}{BOLD}The video processing pipeline (process_video) has been removed.{RESET}\n"
        "Use propose_cuts + scripts/apply_cuts_from_json.py or interview_to_shorts instead."
    )
    sys.exit(1)


async def list_recent_runs() -> None:
    """List recent flow runs."""
    async with get_client() as client:
        runs = await client.read_flow_runs(limit=10)

        print(f"\n{BOLD}Recent Flow Runs:{RESET}\n")
        for run in runs:
            state = run.state
            state_type = state.type if state else None
            icon = state_icon(state_type)
            created = run.created.strftime("%Y-%m-%d %H:%M:%S")
            print(f"  {icon} {run.id}")
            print(f"     Name: {run.name}")
            print(f"     Created: {created}")
            print(f"     State: {state.name if state else 'Unknown'}")
            print()


async def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Video processing CLI with real-time streaming"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run video processing pipeline")
    run_parser.add_argument(
        "-i", "--input", type=Path, required=True, help="Input video file"
    )
    run_parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Output directory"
    )
    run_parser.add_argument(
        "-d", "--duration", type=int, default=60, help="Target duration (seconds)"
    )
    run_parser.add_argument(
        "-t",
        "--title",
        type=str,
        default=None,
        help="Video title (auto-generates if not provided)",
    )
    run_parser.add_argument(
        "-u",
        "--music-url",
        type=str,
        default=None,
        help="Music URL (auto-selects if not provided)",
    )
    run_parser.add_argument(
        "-m",
        "--music-tags",
        type=str,
        default="hip hop",
        help="Music search tags (used if no URL)",
    )
    run_parser.add_argument(
        "--speedup",
        type=float,
        default=1.0,
        help="Speed multiplier to apply after silence removal (e.g., 1.2 = 20%% faster) (default: 1.0)",
    )

    # Stream command
    stream_parser = subparsers.add_parser("stream", help="Stream an existing flow run")
    stream_parser.add_argument("flow_run_id", type=str, help="Flow run ID to stream")
    stream_parser.add_argument(
        "--poll", action="store_true", help="Use polling instead of WebSocket"
    )

    # List command
    subparsers.add_parser("list", help="List recent flow runs")

    args = parser.parse_args()

    if args.command == "run":
        await start_and_stream(
            input_video=args.input,
            output_path=args.output,
            duration=args.duration,
            title=args.title,
            music_url=args.music_url,
            music_tags=args.music_tags,
            speedup=args.speedup,
        )
    elif args.command == "stream":
        flow_run_id = UUID(args.flow_run_id)
        await stream_flow_run(flow_run_id, use_websocket=not args.poll)
    elif args.command == "list":
        await list_recent_runs()
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
