"""What a person sees on the terminal while a run runs (docs/cli.md#console-output).

``logging.console_level`` (``--verbose``, ``--quiet``, ``--log-level``):

- ``normal`` (default): one line per stage with its result, a live line for the stage that is running,
  warnings and errors. The stage's own messages are not printed; on a terminal the latest one is the live
  line's activity.
- ``info`` (``--verbose``): every message as it is logged, and a line per finished Heretic trial.
- ``debug``: also each tool's own output.
- ``warning`` (``--quiet``): warnings, errors and the run summary only.

Every message reaches ``logs/run.log`` and ``logs/events.jsonl`` at every level; this module only decides
what is printed. On a terminal the live line is redrawn in place once a second (elapsed time, activity);
elsewhere (a file, a pipe) nothing is redrawn: the stage's start, milestones (a finished trial) and its result
are printed as lines. Colour only on a terminal, and never with ``NO_COLOR`` set. Standard library only.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from collections.abc import Callable

LEVELS = {"debug": 10, "info": 20, "normal": 25, "warning": 30, "error": 40}
OUTCOME_WORDS = {"succeeded": "ok", "reused": "reused", "skipped": "skipped", "failed": "FAILED", "blocked": "BLOCKED", "interrupted": "stopped"}
COLORS = {"ok": "32", "reused": "32", "skipped": "2", "FAILED": "31", "BLOCKED": "31", "stopped": "33", "WARN": "33", "ERROR": "31"}


def duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def clock(seconds: float) -> str:
    """Elapsed time as m:ss (h:mm:ss past an hour)."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class Console:
    def __init__(self, stream=None, level: str = "normal", tty: bool | None = None, color: bool | None = None, tick: float = 1.0) -> None:
        self.stream = stream or sys.stderr
        self.level = LEVELS.get(level, LEVELS["normal"])
        if tty is None:
            try:
                tty = self.stream.isatty()
            except (AttributeError, ValueError):
                tty = False
        self.tty = bool(tty)
        self.color = (self.tty and "NO_COLOR" not in os.environ) if color is None else color
        self.tick = tick
        self._lock = threading.RLock()
        self._live: dict | None = None
        self._drawn = False
        self._stop = threading.Event()
        self._ticker: threading.Thread | None = None

    @property
    def normal(self) -> bool:
        return self.level == LEVELS["normal"]

    # ------------------------------------------------------------------ messages
    def message(self, level: str, text: str, lifecycle: bool = False) -> None:
        """A message from the event log. ``lifecycle``: a stage's start/end line, which the normal view draws itself."""
        rank = LEVELS.get(level, LEVELS["info"])
        if lifecycle and self.level >= LEVELS["normal"]:
            return
        if rank < self.level:
            if self.normal and rank == LEVELS["info"]:
                self.activity(text)
            return
        tag = {"warning": "WARN", "error": "ERROR"}.get(level)
        self.line(f"{self.paint(tag)} {text}" if tag else text)

    def line(self, text: str) -> None:
        """A permanent line (the live line is cleared first and redrawn after it)."""
        with self._lock:
            self._clear()
            self.stream.write(text + "\n")
            self._draw()
            self.stream.flush()

    def paint(self, word: str, code: str | None = None) -> str:
        code = code or COLORS.get(word.strip())
        return f"\x1b[{code}m{word}\x1b[0m" if self.color and code else word

    # ------------------------------------------------------------------ stages
    def stage_start(self, label: str) -> None:
        if not self.normal:
            return
        with self._lock:
            self._live = {"label": label, "started": time.monotonic(), "text": ""}
            if self.tty:
                self._draw()
                self.stream.flush()
                self._start_ticker()
            else:
                self.stream.write(f"{label} started\n")
                self.stream.flush()

    def activity(self, text: str | Callable[[], str]) -> None:
        """What the running stage is doing now (a string, or a function re-read at every redraw)."""
        with self._lock:
            if self._live is not None:
                self._live["text"] = text
                self._draw()
                self.stream.flush()

    def milestone(self, text: str) -> None:
        """A progress event worth a line where nothing is redrawn (a file, a pipe) and in the verbose view."""
        if self.level <= LEVELS["info"]:
            self.line(f"  {text}")
        elif self.normal and not self.tty and self._live is not None:
            self.line(f"{self._live['label']} {text}")

    def stage_end(self, label: str, outcome: str, seconds: float, summary: str = "") -> None:
        word = OUTCOME_WORDS.get(outcome, outcome)
        with self._lock:
            self._live = None
            self._stop_ticker()
            if not self.normal:
                return
            timing = "" if outcome in ("reused", "skipped") else duration(seconds)
            self.line(f"{label:<22} {self.paint(f'{word:<8}')} {timing:>7}  {summary}".rstrip())

    def close(self) -> None:
        with self._lock:
            self._live = None
            self._stop_ticker()
            self._clear()
            self.stream.flush()

    # ------------------------------------------------------------------ the live line
    def _render(self) -> str:
        live = self._live
        text = live["text"]() if callable(live["text"]) else live["text"]
        body = f"{live['label']} {clock(time.monotonic() - live['started'])}" + (f" · {text}" if text else "")
        width = shutil.get_terminal_size((100, 20)).columns - 1
        return body if len(body) <= width else body[: max(10, width - 1)] + "…"

    def _draw(self) -> None:
        if not (self.tty and self._live is not None):
            return
        try:
            line = self._render()
        except Exception:  # noqa: BLE001 - a progress text must never break the run
            return
        self.stream.write("\r\x1b[2K" + line)
        self._drawn = True

    def _clear(self) -> None:
        if self.tty and self._drawn:
            self.stream.write("\r\x1b[2K")
            self._drawn = False

    def _start_ticker(self) -> None:
        if self._ticker is not None:
            return
        self._stop.clear()

        def tick() -> None:
            while not self._stop.wait(self.tick):
                with self._lock:
                    self._draw()
                    self.stream.flush()

        self._ticker = threading.Thread(target=tick, name="console-ticker", daemon=True)
        self._ticker.start()

    def _stop_ticker(self) -> None:
        if self._ticker is not None:
            self._stop.set()
            self._ticker = None
        self._clear()
