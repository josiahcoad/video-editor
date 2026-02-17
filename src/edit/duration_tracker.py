#!/usr/bin/env python3
"""
Track video processing durations and estimate future run times.

Uses SQLite to store historical data: input video duration vs. processing time.
"""

import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

DB_PATH = Path.home() / ".video_editor" / "duration_history.db"


class RunRecord(NamedTuple):
    """A single run record."""

    input_duration: float  # seconds
    processing_duration: float  # seconds
    timestamp: datetime
    speedup: float = 1.0
    target_duration: int = 60


def _init_db() -> None:
    """Initialize the SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_duration REAL NOT NULL,
            processing_duration REAL NOT NULL,
            speedup REAL DEFAULT 1.0,
            target_duration INTEGER DEFAULT 60,
            timestamp TEXT NOT NULL
        )
    """
    )

    conn.commit()
    conn.close()


def record_run(
    input_duration: float,
    processing_duration: float,
    speedup: float = 1.0,
    target_duration: int = 60,
) -> None:
    """Record a completed run."""
    _init_db()

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO runs (input_duration, processing_duration, speedup, target_duration, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """,
        (
            input_duration,
            processing_duration,
            speedup,
            target_duration,
            datetime.now().isoformat(),
        ),
    )

    conn.commit()
    conn.close()


def estimate_duration(
    input_duration: float, speedup: float = 1.0, target_duration: int = 60
) -> float:
    """
    Estimate processing duration based on historical data.

    Uses simple linear regression: processing_time = a * input_duration + b
    Falls back to a ratio-based estimate if not enough data.

    Returns:
        Estimated processing duration in seconds
    """
    _init_db()

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Get recent runs (last 100) with similar parameters
    cursor.execute(
        """
        SELECT input_duration, processing_duration
        FROM runs
        WHERE ABS(speedup - ?) < 0.1 AND ABS(target_duration - ?) < 10
        ORDER BY timestamp DESC
        LIMIT 100
    """,
        (speedup, target_duration),
    )

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        # No historical data - use a conservative estimate
        # Assume ~2x real-time processing (e.g., 60s video takes ~30s to process)
        return input_duration * 0.5

    if len(rows) < 3:
        # Not enough data for regression - use average ratio
        ratios = [proc / inp for inp, proc in rows]
        avg_ratio = sum(ratios) / len(ratios)
        return input_duration * avg_ratio

    # Simple linear regression: y = ax + b
    # where x = input_duration, y = processing_duration
    x_values = [row[0] for row in rows]
    y_values = [row[1] for row in rows]

    n = len(rows)
    sum_x = sum(x_values)
    sum_y = sum(y_values)
    sum_xy = sum(x * y for x, y in zip(x_values, y_values))
    sum_x2 = sum(x * x for x in x_values)

    # Calculate slope (a) and intercept (b)
    denominator = n * sum_x2 - sum_x * sum_x
    if abs(denominator) < 1e-10:
        # Fallback to ratio if denominator is too small
        ratios = [y / x for x, y in zip(x_values, y_values)]
        avg_ratio = sum(ratios) / len(ratios)
        return input_duration * avg_ratio

    a = (n * sum_xy - sum_x * sum_y) / denominator
    b = (sum_y - a * sum_x) / n

    estimated = a * input_duration + b

    # Ensure estimate is reasonable (at least 10% of input, at most 5x input)
    estimated = max(input_duration * 0.1, min(estimated, input_duration * 5.0))

    return estimated


def get_stats() -> dict:
    """Get statistics about historical runs."""
    _init_db()

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM runs")
    total_runs = cursor.fetchone()[0]

    if total_runs == 0:
        conn.close()
        return {"total_runs": 0}

    cursor.execute(
        """
        SELECT 
            AVG(input_duration),
            AVG(processing_duration),
            AVG(processing_duration / input_duration) as avg_ratio
        FROM runs
    """
    )
    stats = cursor.fetchone()

    conn.close()

    return {
        "total_runs": total_runs,
        "avg_input_duration": stats[0],
        "avg_processing_duration": stats[1],
        "avg_ratio": stats[2],
    }
