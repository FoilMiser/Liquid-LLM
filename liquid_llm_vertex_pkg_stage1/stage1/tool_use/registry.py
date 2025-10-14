"""Registry of available tools."""

from typing import Callable, Dict

from .calculator import SafeCalculator
from .scratchpad import Scratchpad

_CALCULATOR = SafeCalculator()
_SCRATCHPAD = Scratchpad()

TOOL_REGISTRY: Dict[str, Callable[[str], str]] = {
    "calculator": _CALCULATOR.evaluate,
    "scratchpad_append": lambda text: _register_scratchpad(text),
}


def _register_scratchpad(text: str) -> str:
    _SCRATCHPAD.add(text)
    return _SCRATCHPAD.render()


def call_tool(name: str, arg: str) -> str:
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown tool: {name}")
    return TOOL_REGISTRY[name](arg)


__all__ = ["TOOL_REGISTRY", "call_tool"]
