"""Safe scientific calculator for tool supervision."""

import math
from typing import Any, Dict

_ALLOWED_NAMES: Dict[str, Any] = {
    name: getattr(math, name)
    for name in [
        "pi",
        "e",
        "tau",
        "inf",
        "nan",
        "sin",
        "cos",
        "tan",
        "log",
        "log10",
        "exp",
        "sqrt",
        "floor",
        "ceil",
        "fabs",
    ]
}
_ALLOWED_NAMES.update({"abs": abs, "round": round})


class SafeCalculator:
    """Evaluates mathematical expressions with a strict whitelist."""

    def __init__(self):
        self._globals = {"__builtins__": {}}
        self._locals = dict(_ALLOWED_NAMES)

    def evaluate(self, expression: str) -> str:
        expression = expression.strip()
        if not expression:
            raise ValueError("Empty expression")
        result = eval(expression, self._globals, self._locals)  # noqa: S307 - controlled environment
        if isinstance(result, float):
            return f"{result:.8g}"
        return str(result)


__all__ = ["SafeCalculator"]
