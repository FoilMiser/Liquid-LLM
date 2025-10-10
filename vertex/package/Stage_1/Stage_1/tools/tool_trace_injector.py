"""Inject tool call traces for supervised fine-tuning."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .calculator import Calculator
from .scratchpad import Scratchpad


_CALL_RE = re.compile(r'CALL\s+(?P<tool>calculator|scratchpad):"(?P<payload>.*?)"')


@dataclass
class ToolTraceInjector:
    max_calls: int = 4
    calculator_enabled: bool = True
    scratchpad_enabled: bool = True
    degrees: bool = False

    def __post_init__(self) -> None:
        self.calculator = Calculator(degrees=self.degrees)
        self.scratchpad = Scratchpad()

    def inject(self, tokens: List[str]) -> List[str]:
        result: List[str] = []
        call_count = 0
        for token in tokens:
            result.append(token)
            match = _CALL_RE.search(token)
            if not match:
                continue
            if call_count >= self.max_calls:
                continue
            tool = match.group("tool")
            payload = match.group("payload")
            if tool == "calculator" and self.calculator_enabled:
                value = self.calculator.evaluate(payload)
                result.append(f"RESULT:{value}")
            elif tool == "scratchpad" and self.scratchpad_enabled:
                self.scratchpad.append("Model", payload)
                result.append(f"RESULT:{self.scratchpad.render()}")
            call_count += 1
        return result


__all__ = ["ToolTraceInjector"]
