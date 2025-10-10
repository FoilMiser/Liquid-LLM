"""Structured scratchpad for reasoning traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Scratchpad:
    max_lines: int = 12
    template: List[str] = field(default_factory=lambda: ["Given:", "Compute:", "Conclude:"])

    def __post_init__(self) -> None:
        self.lines: List[str] = list(self.template)

    def append(self, section: str, text: str) -> None:
        entry = f"{section.strip()}: {text.strip()}"
        if len(self.lines) >= max(1, self.max_lines - 1):
            raise ValueError("Scratchpad full")
        self.lines.append(entry)

    def render(self) -> str:
        return "\n".join(self.lines)

    def reset(self) -> None:
        self.lines = list(self.template)


__all__ = ["Scratchpad"]
