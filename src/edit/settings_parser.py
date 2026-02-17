"""
Argparse wrapper that uses project/session settings.json as defaults with CLI overrides.

Use when you want: script default → settings (project + session) → CLI flags (win).

  parser = SettingsArgParser(context_path=Path("video.mp4"))
  parser.add_arg("--duration", settings_key="title.duration", type=float, default=20)
  parser.add_arg("--height", settings_key="title.height_percent", type=int, default=10)
  args = parser.parse_args()

If the user does not pass --duration, args.duration comes from settings["title"]["duration"],
then from default. If they pass --duration 5, 5 wins.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .settings_loader import load_settings


def _get_nested(d: dict[str, Any], path: tuple[str, ...]) -> Any:
    """Get d[k0][k1][...] for path=(k0, k1, ...). Returns None if any key missing."""
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


class SettingsArgParser(argparse.ArgumentParser):
    """
    ArgumentParser that fills unspecified options from settings (by settings_key).

    Pass either context_path (to discover and load settings) or settings dict.
    For each add_arg(..., settings_key="section.key"), if that option is not
    present in sys.argv, the value is taken from settings["section"]["key"],
    then from the default you passed to add_arg.
    """

    def __init__(
        self,
        context_path: Path | None = None,
        settings: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if settings is not None:
            self._settings = settings
        elif context_path is not None:
            self._settings = load_settings(Path(context_path))
        else:
            self._settings = {}
        # (option_strings, dest, key_path, default, type)
        self._settings_overrides: list[
            tuple[list[str], str, tuple[str, ...], Any, Any]
        ] = []

    def add_arg(
        self,
        *args: Any,
        settings_key: str | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> argparse.Action:
        """
        Same as add_argument, but if settings_key is set then when the option
        is not in sys.argv, the value is taken from settings (nested key)
        then from default.

        settings_key: dot path or tuple, e.g. "title.duration" or ("title", "duration").
        """
        action = self.add_argument(*args, **kwargs)
        if settings_key is not None:
            key_path = (
                tuple(settings_key.split("."))
                if isinstance(settings_key, str)
                else tuple(settings_key)
            )
            default = kwargs.get("default")
            type_ = kwargs.get("type")
            self._settings_overrides.append(
                (list(action.option_strings), action.dest, key_path, default, type_)
            )
        return action

    def parse_args(
        self,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        result = super().parse_args(args, namespace)
        argv = sys.argv if args is None else [self.prog] + (args or [])
        for option_strings, dest, key_path, default, type_ in self._settings_overrides:
            if not any(opt in argv for opt in option_strings):
                raw = _get_nested(self._settings, key_path)
                if raw is not None:
                    if type_ is not None:
                        try:
                            setattr(result, dest, type_(raw))
                        except (TypeError, ValueError):
                            setattr(result, dest, default)
                    else:
                        setattr(result, dest, raw)
                elif default is not None:
                    setattr(result, dest, default)
        return result
