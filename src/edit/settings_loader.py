"""
Load project/session settings.json for edit scripts.

Resolution order for each option: script default → project editing/settings.json
→ session settings.json (videos/<session>/settings.json) → CLI flags (override).

Scripts pass a context path (usually the video or output path). We find the
project editing dir (contains "videos" and "settings.json") and optionally the
session dir (contains "inputs" and "outputs"), load both, merge (session over
project), and return a single dict. Scripts then use this for defaults and let
CLI flags override.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _find_editing_dir(context_path: Path) -> Path | None:
    """Directory that contains 'videos' and 'settings.json' (project editing root)."""
    try:
        resolved = context_path.resolve()
    except OSError:
        return None
    p = resolved.parent
    while p != p.parent:
        if (p / "videos").is_dir() and (p / "settings.json").is_file():
            return p
        p = p.parent
    return None


def _find_session_dir(context_path: Path) -> Path | None:
    """Directory that contains both 'inputs' and 'outputs' (session root)."""
    try:
        resolved = context_path.resolve()
    except OSError:
        return None
    p = resolved.parent
    while p != p.parent:
        if (p / "inputs").is_dir() and (p / "outputs").is_dir():
            return p
        p = p.parent
    return None


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base. Override values take precedence."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_settings(context_path: Path) -> dict[str, Any]:
    """
    Load merged settings from project and session settings.json.

    Args:
        context_path: Path under the project (e.g. video file or output file).
                      Used to locate projects/<Client>/editing/ and
                      editing/videos/<session>/.

    Returns:
        Merged dict: project editing/settings.json (if found), then session
        settings.json (if found) merged on top. Empty dict if neither exists.
    """
    out: dict[str, Any] = {}
    try:
        context_path = context_path.resolve()
    except OSError:
        return out

    # Project-level: .../editing/settings.json
    editing_dir = _find_editing_dir(context_path)
    if editing_dir:
        settings_file = editing_dir / "settings.json"
        if settings_file.is_file():
            try:
                out = json.loads(settings_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass

    # Session-level: .../videos/<session>/settings.json (overrides project)
    session_dir = _find_session_dir(context_path)
    if session_dir:
        session_settings = session_dir / "settings.json"
        if session_settings.is_file():
            try:
                session_data = json.loads(session_settings.read_text())
                out = deep_merge(out, session_data)
            except (json.JSONDecodeError, OSError):
                pass

    return out
