"""Scratchpad helper for tool demonstrations."""

from typing import List


class Scratchpad:
    def __init__(self, max_lines: int = 8):
        self.max_lines = max_lines
        self._lines: List[str] = []

    def add(self, line: str) -> None:
        if len(self._lines) >= self.max_lines:
            self._lines.pop(0)
        self._lines.append(line.strip())

    def render(self) -> str:
        return "\n".join(self._lines)


__all__ = ["Scratchpad"]
