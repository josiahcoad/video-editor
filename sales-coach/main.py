#!/usr/bin/env python3
"""
Sales Coach — Real-time AI coaching for live sales calls.

Uses Deepgram Flux for streaming transcription with end-of-turn detection,
and Claude Haiku 4.5 (via OpenRouter) for instant coaching advice.

Usage:
    dotenvx run -f .env -- uv run python sales-coach/main.py
    dotenvx run -f .env -- uv run python sales-coach/main.py --debug
    dotenvx run -f .env -- uv run python sales-coach/main.py --device 2

Requires: DEEPGRAM_API_KEY, OPENROUTER_API_KEY in environment.
"""

import argparse
import asyncio
import os
import queue
import sys
import time
from typing import Any

import numpy as np
import openai
import sounddevice as sd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.listen import ListenV2TurnInfo

# Flux recommends 80ms chunks at 16kHz mono (linear16)
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = 1280  # 80ms * 16kHz = 1280 samples

console = Console()

SYSTEM_PROMPT = """\
You are a real-time sales coach whispering advice during a live call.
After each pause in conversation, give the sales rep ONE specific thing to say next.

Rules:
- MAX 2 sentences. They need to read this instantly while on the call.
- Reference what was actually said — never be generic.
- Push toward closing: handle objections, ask discovery questions, trial-close, create urgency.
- If buying signals appear (pricing questions, timeline, next steps), suggest a closing technique.
- If an objection was raised, suggest a specific rebuttal using the prospect's own words.
- If it's early conversation, suggest an open-ended question to uncover pain points.
- Format: Just the advice. No labels, no preamble, no "You could say..."
"""


class SalesCoach:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.transcript_buffer: list[str] = []
        self.full_conversation: list[str] = []
        self.coaching_count = 0
        self.start_time: float = 0
        self.running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._audio_queue: queue.Queue[bytes] = queue.Queue()
        self._llm = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )

    # ── Audio capture ──────────────────────────────────────────────

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: Any, status: Any
    ) -> None:
        """Sounddevice callback — pushes raw PCM bytes into a thread-safe queue."""
        if status and self.debug:
            console.print(f"[yellow]⚠ Audio: {status}[/yellow]")
        self._audio_queue.put(indata.copy().tobytes())

    async def _pump_audio(self, connection: Any) -> None:
        """Drains the audio queue and sends chunks to Deepgram via send_media."""
        while self.running:
            sent = False
            try:
                while True:
                    chunk = self._audio_queue.get_nowait()
                    await connection.send_media(chunk)
                    sent = True
            except queue.Empty:
                pass
            if not sent:
                await asyncio.sleep(0.01)

    # ── Coaching ───────────────────────────────────────────────────

    async def _get_coaching(self, conversation_text: str) -> None:
        """Sends recent conversation to Claude Haiku 4.5 (via OpenRouter) for coaching."""
        try:
            response = await self._llm.chat.completions.create(
                model="anthropic/claude-haiku-4-5",
                max_tokens=120,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Live call transcript (most recent turns):\n\n"
                            f"{conversation_text}\n\n"
                            f"What should the sales rep say next?"
                        ),
                    },
                ],
            )
            advice = response.choices[0].message.content.strip()
            self.coaching_count += 1

            console.print()
            console.print(
                Panel(
                    f"[bold bright_green]{advice}[/bold bright_green]",
                    title=f"🎯 Coach #{self.coaching_count}",
                    border_style="bright_green",
                    padding=(0, 1),
                )
            )

        except Exception as e:
            console.print(f"[red]Coaching error: {e}[/red]")

    # ── Deepgram message handler ──────────────────────────────────

    def _on_message(self, message: Any) -> None:
        """Handles every Deepgram Flux WebSocket message.

        ListenV2TurnInfo messages have:
          - type: "TurnInfo"
          - event: "Update" | "StartOfTurn" | "EagerEndOfTurn" | "TurnResumed" | "EndOfTurn"
          - transcript: str
          - words: list
          - end_of_turn_confidence: float
        """
        msg_type = getattr(message, "type", None)
        event = getattr(message, "event", None)

        if self.debug:
            console.print(
                f"  [dim magenta][type={msg_type} event={event}][/dim magenta] "
                f"{getattr(message, 'transcript', '')[:80]}"
            )

        # Only process TurnInfo messages
        if msg_type != "TurnInfo":
            if msg_type == "Connected":
                console.print("[green]✓ Deepgram session established[/green]")
            return

        # Show transcript on updates
        transcript = getattr(message, "transcript", None)
        if event == "Update" and transcript and transcript.strip():
            # Overwrite line with latest interim transcript
            console.print(f"  [dim white]{transcript.strip()}[/dim white]")

        # End of turn — commit transcript and trigger coaching
        if event == "EndOfTurn" and transcript and transcript.strip():
            text = transcript.strip()
            self.full_conversation.append(text)

            turn_num = len(self.full_conversation)
            confidence = getattr(message, "end_of_turn_confidence", 0)
            console.print(
                f"\n[bold cyan]── end of turn {turn_num} "
                f"(confidence: {confidence:.2f}) ──[/bold cyan]"
            )
            console.print(f"  [white]{text}[/white]")

            # Build context from last 10 turns
            recent = self.full_conversation[-10:]
            conv_text = "\n".join(f"Turn {i + 1}: {t}" for i, t in enumerate(recent))

            # Non-blocking — don't stall transcription
            if self._loop:
                self._loop.create_task(self._get_coaching(conv_text))

    # ── Entrypoint ────────────────────────────────────────────────

    async def start(self, device: int | None = None) -> None:
        """Starts mic capture, Deepgram Flux, and coaching loop."""
        self._loop = asyncio.get_running_loop()
        self.start_time = time.time()
        self.running = True

        # Resolve audio device
        if device is not None:
            device_info = sd.query_devices(device)
        else:
            device_info = sd.query_devices(kind="input")
        device_name = device_info["name"]

        console.print(
            Panel(
                "[bold]Sales Coach[/bold] — Real-time AI coaching for your sales calls\n\n"
                f"  Mic:   [cyan]{device_name}[/cyan]\n"
                "  STT:   Deepgram Flux (end-of-turn detection)\n"
                "  Coach: Claude Haiku 4.5 (via OpenRouter)\n\n"
                "[yellow]Tip:[/yellow] Put your call on speaker, or use a virtual audio\n"
                "device (BlackHole on macOS) to capture both sides.\n\n"
                "[dim]Ctrl+C to stop.[/dim]",
                title="🎙️  Sales Coach",
                border_style="blue",
            )
        )
        console.print()

        client = AsyncDeepgramClient()

        try:
            async with client.listen.v2.connect(
                model="flux-general-en",
                encoding="linear16",
                sample_rate=str(SAMPLE_RATE),
                eot_timeout_ms="3000",
            ) as connection:
                # Register event handlers
                connection.on(
                    EventType.OPEN,
                    lambda _: console.print(
                        "[green]✓ Connected to Deepgram Flux[/green]"
                    ),
                )
                connection.on(EventType.MESSAGE, self._on_message)
                connection.on(
                    EventType.CLOSE,
                    lambda _: console.print("\n[yellow]Connection closed[/yellow]"),
                )
                connection.on(
                    EventType.ERROR,
                    lambda e: console.print(f"[red]Deepgram error: {e}[/red]"),
                )

                # Start WebSocket listener
                listen_task = asyncio.create_task(connection.start_listening())

                # Start microphone stream
                stream = sd.InputStream(
                    device=device,
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="int16",
                    blocksize=BLOCKSIZE,
                    callback=self._audio_callback,
                )
                stream.start()
                console.print("[green]✓ Microphone active — start your call![/green]\n")

                # Pump audio from mic → Deepgram
                send_task = asyncio.create_task(self._pump_audio(connection))

                try:
                    await asyncio.gather(listen_task, send_task)
                except asyncio.CancelledError:
                    pass
                finally:
                    stream.stop()
                    stream.close()

        except Exception as e:
            console.print(f"[red]Fatal: {e}[/red]")

        finally:
            self.running = False
            elapsed = time.time() - self.start_time
            console.print()
            console.print(
                Panel(
                    f"Duration: {elapsed / 60:.1f} min  •  "
                    f"Coaching moments: {self.coaching_count}  •  "
                    f"Turns: {len(self.full_conversation)}",
                    title="📊 Session Summary",
                    border_style="blue",
                )
            )


def list_devices() -> None:
    """Print available audio input devices."""
    devices = sd.query_devices()
    table = Table(title="Audio Input Devices")
    table.add_column("ID", style="cyan", width=4)
    table.add_column("Name", style="white")
    table.add_column("Channels", style="green", width=10)
    table.add_column("Sample Rate", style="yellow", width=12)

    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            table.add_row(
                str(i),
                d["name"],
                str(d["max_input_channels"]),
                str(int(d["default_samplerate"])),
            )
    console.print(table)


async def run(args: argparse.Namespace) -> None:
    missing = []
    if not os.environ.get("DEEPGRAM_API_KEY"):
        missing.append("DEEPGRAM_API_KEY")
    if not os.environ.get("OPENROUTER_API_KEY"):
        missing.append("OPENROUTER_API_KEY")
    if missing:
        console.print(f"[red]Missing env vars: {', '.join(missing)}[/red]")
        console.print("[dim]Add them to .env or export before running.[/dim]")
        sys.exit(1)

    coach = SalesCoach(debug=args.debug)
    await coach.start(device=args.device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time AI sales coaching powered by Deepgram Flux + Claude"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show all Deepgram message types"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device ID (see --list-devices)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
