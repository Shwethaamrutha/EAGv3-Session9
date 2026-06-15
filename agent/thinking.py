"""Streaming thought — live status updates during agent execution.

Provides a callback-based interface that both CLI and WebSocket can hook into.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ThinkingEvent:
    """A single thought/status update."""
    stage: str          # "memory", "perception", "decision", "action", "system"
    message: str        # human-readable status
    detail: str = ""    # optional extra context


# Type for callback functions
ThinkingCallback = Callable[[ThinkingEvent], None]


class ThinkingStream:
    """Manages streaming thought output."""

    def __init__(self, callback: ThinkingCallback | None = None):
        self._callback = callback or self._default_cli_callback
        self._spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self._spinner_idx = 0

    def emit(self, stage: str, message: str, detail: str = ""):
        self._callback(ThinkingEvent(stage=stage, message=message, detail=detail))

    def _default_cli_callback(self, event: ThinkingEvent):
        """CLI inline thinking — overwrites current line on interactive terminals."""
        if not sys.stderr.isatty():
            return  # Don't emit spinners when piped/redirected

        DIM = "\033[2m"
        RESET = "\033[0m"

        spinner = self._spinner_chars[self._spinner_idx % len(self._spinner_chars)]
        self._spinner_idx += 1

        sys.stderr.write(f"\r{DIM}{spinner} [{event.stage}] {event.message}{RESET}\033[K")
        sys.stderr.flush()

    def clear_line(self):
        """Clear the thinking line when done."""
        if not sys.stderr.isatty():
            return
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()


# Global default stream for CLI use
_cli_stream: ThinkingStream | None = None


def get_stream() -> ThinkingStream:
    global _cli_stream
    if _cli_stream is None:
        _cli_stream = ThinkingStream()
    return _cli_stream


def think(stage: str, message: str, detail: str = ""):
    """Emit a thinking event to the active stream."""
    get_stream().emit(stage, message, detail)


def done():
    """Clear the thinking indicator."""
    get_stream().clear_line()
