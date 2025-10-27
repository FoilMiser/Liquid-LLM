"""Trace parser for CALL/RESULT tool supervision."""

import re
from typing import Iterable

from .registry import call_tool

CALL_RE = re.compile(r'^CALL\s+(?P<tool>[\w_]+):"(?P<arg>.*)"$')


def maybe_inject_tool_result(text: str) -> str:
    lines = text.splitlines()
    output_lines = []
    for line in lines:
        output_lines.append(line)
        match = CALL_RE.match(line.strip())
        if match:
            tool = match.group("tool")
            arg = match.group("arg")
            result = call_tool(tool, arg)
            output_lines.append(f"RESULT: {result}")
    return "\n".join(output_lines)


def inject_for_lines(lines: Iterable[str]) -> str:
    return maybe_inject_tool_result("\n".join(lines))


__all__ = ["maybe_inject_tool_result", "inject_for_lines"]
