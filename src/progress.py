#!/usr/bin/env python3
"""
Progress tracking and display for video processing.

Shows estimated time remaining and progress bar based on task completion.
"""

import time
from typing import Callable

from duration_tracker import estimate_duration, get_stats, record_run


class ProgressTracker:
    """Tracks progress through video processing pipeline."""

    def __init__(
        self,
        input_duration: float,
        speedup: float = 1.0,
        target_duration: int = 60,
        total_tasks: int = 9,
    ):
        """Initialize progress tracker.

        Args:
            input_duration: Input video duration in seconds
            speedup: Speed multiplier being applied
            target_duration: Target duration for trim
            total_tasks: Total number of tasks in pipeline
        """
        self.input_duration = input_duration
        self.speedup = speedup
        self.target_duration = target_duration
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.start_time = time.time()
        self.estimated_total = estimate_duration(input_duration, speedup, target_duration)

    def start_task(self, task_name: str) -> None:
        """Mark a task as starting."""
        self.current_task = task_name
        self.task_start_time = time.time()

    def complete_task(self, task_name: str) -> None:
        """Mark a task as completed."""
        self.completed_tasks += 1
        elapsed = time.time() - self.start_time
        remaining_tasks = self.total_tasks - self.completed_tasks

        # Update estimate based on actual progress
        if self.completed_tasks > 0:
            avg_time_per_task = elapsed / self.completed_tasks
            estimated_remaining = avg_time_per_task * remaining_tasks
        else:
            estimated_remaining = self.estimated_total - elapsed

        self._display_progress(elapsed, estimated_remaining)

    def finish(self) -> None:
        """Mark the entire process as complete."""
        total_time = time.time() - self.start_time

        # Record in database for future estimates
        record_run(
            input_duration=self.input_duration,
            processing_duration=total_time,
            speedup=self.speedup,
            target_duration=self.target_duration,
        )

        print(f"\n✅ Completed in {total_time:.1f}s")

    def _display_progress(self, elapsed: float, estimated_remaining: float) -> None:
        """Display progress bar and time estimates."""
        progress = self.completed_tasks / self.total_tasks
        bar_width = 40
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)

        elapsed_str = f"{elapsed:.0f}s"
        remaining_str = f"{estimated_remaining:.0f}s" if estimated_remaining > 0 else "?"

        print(
            f"\r[{bar}] {self.completed_tasks}/{self.total_tasks} tasks | "
            f"Elapsed: {elapsed_str} | Est. remaining: {remaining_str}",
            end="",
            flush=True,
        )


def get_initial_estimate(input_duration: float, speedup: float, target_duration: int) -> dict:
    """Get initial time estimate before starting."""
    estimated = estimate_duration(input_duration, speedup, target_duration)
    stats = get_stats()

    return {
        "estimated_seconds": estimated,
        "estimated_minutes": estimated / 60,
        "total_runs": stats.get("total_runs", 0),
        "confidence": "high" if stats.get("total_runs", 0) >= 5 else "low",
    }
