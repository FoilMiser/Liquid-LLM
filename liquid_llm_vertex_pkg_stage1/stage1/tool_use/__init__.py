"""Tool use helpers for training."""

from .calculator import SafeCalculator
from .scratchpad import Scratchpad
from .registry import TOOL_REGISTRY, call_tool
from .traces import maybe_inject_tool_result

__all__ = [
    "SafeCalculator",
    "Scratchpad",
    "TOOL_REGISTRY",
    "call_tool",
    "maybe_inject_tool_result",
]
