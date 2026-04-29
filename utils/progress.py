from __future__ import annotations

import sys
import time


class TerminalProgress:
    def __init__(self, total: int, title: str, width: int = 28) -> None:
        self.total = max(int(total), 1)
        self.title = title
        self.width = width
        self.start_time = time.time()
        self.current = 0
        self._last_render_len = 0
        self._finished = False

    def _format_eta(self, seconds: float) -> str:
        seconds = max(int(seconds), 0)
        minutes, sec = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{sec:02d}"
        return f"{minutes:02d}:{sec:02d}"

    def _render(self, postfix: str = "") -> None:
        ratio = min(max(self.current / self.total, 0.0), 1.0)
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.start_time
        eta = 0.0 if self.current <= 0 else elapsed * (self.total - self.current) / max(self.current, 1)
        line = (
            f"\r[{self.title}] [{bar}] "
            f"{self.current:>5d}/{self.total:<5d} "
            f"{ratio * 100:6.2f}% "
            f"elapsed {self._format_eta(elapsed)} "
            f"eta {self._format_eta(eta)}"
        )
        if postfix:
            line += f" | {postfix}"
        padded = line
        if len(line) < self._last_render_len:
            padded += " " * (self._last_render_len - len(line))
        self._last_render_len = len(line)
        sys.stdout.write(padded)
        sys.stdout.flush()

    def update(self, step: int = 1, postfix: str = "") -> None:
        if self._finished:
            return
        self.current = min(self.current + step, self.total)
        self._render(postfix)

    def finish(self, postfix: str = "") -> None:
        if self._finished:
            return
        self.current = self.total
        self._render(postfix)
        sys.stdout.write("\n")
        sys.stdout.flush()
        self._finished = True
