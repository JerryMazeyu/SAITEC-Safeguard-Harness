from __future__ import annotations

import os
import sys
from typing import TextIO


DISABLE_VALUES = {"0", "false", "no", "off"}


class TerminalProgress:
    def __init__(
        self,
        label: str,
        total: int,
        *,
        stream: TextIO | None = None,
        enabled: bool | None = None,
        width: int = 24,
    ) -> None:
        self.label = label
        self.total = max(total, 0)
        self.stream = stream or sys.stderr
        self.width = max(width, 1)
        self.enabled = _progress_enabled() if enabled is None else enabled
        self._last_line = ""

    def start(self) -> None:
        self.update(0)

    def update(self, processed: int, *, current: str | None = None) -> None:
        if not self.enabled:
            return
        processed = max(0, min(processed, self.total)) if self.total else max(processed, 0)
        percent = 100.0 if self.total == 0 else (processed / self.total) * 100
        filled = self.width if self.total == 0 else round((processed / self.total) * self.width)
        bar = "#" * filled + "-" * (self.width - filled)
        line = f"\r[{self.label}] [{bar}] {processed}/{self.total} {percent:5.1f}%"
        if current:
            line += f" current={_compact(current)}"
        self.stream.write(line)
        self.stream.flush()
        self._last_line = line

    def finish(self, *, current: str | None = "done") -> None:
        if not self.enabled:
            return
        self.update(self.total, current=current)
        self.stream.write("\n")
        self.stream.flush()
        self._last_line = ""

    def fail(self, *, processed: int, error: str) -> None:
        if not self.enabled:
            return
        self.update(processed, current=f"failed: {_compact(error, limit=96)}")
        self.stream.write("\n")
        self.stream.flush()
        self._last_line = ""


def _progress_enabled() -> bool:
    value = os.environ.get("SAFEGUARD_HARNESS_PROGRESS", "1").strip().casefold()
    return value not in DISABLE_VALUES


def _compact(value: str, *, limit: int = 80) -> str:
    compacted = " ".join(str(value).split())
    if len(compacted) <= limit:
        return compacted
    return f"{compacted[: max(limit - 3, 0)]}..."
