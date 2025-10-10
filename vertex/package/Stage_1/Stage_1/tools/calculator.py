"""Deterministic scientific calculator."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Dict


_ALLOWED_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "exp": math.exp,
    "sqrt": math.sqrt,
}


@dataclass
class Calculator:
    degrees: bool = False
    max_nodes: int = 128

    def _convert_angle(self, value: float) -> float:
        return math.radians(value) if self.degrees else value

    def evaluate(self, expression: str) -> float:
        expression = expression.strip()
        if len(expression) > 256:
            raise ValueError("Expression too long")
        tree = ast.parse(expression, mode="eval")
        nodes = list(ast.walk(tree))
        if len(nodes) > self.max_nodes:
            raise ValueError("Expression too complex")
        return float(self._eval(tree.body))

    def _eval(self, node):
        if isinstance(node, ast.BinOp):
            left = self._eval(node.left)
            right = self._eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            if isinstance(node.op, ast.BitXor):
                return left ** right
            raise ValueError("Unsupported binary operator")
        if isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator")
        if isinstance(node, ast.Num):  # type: ignore[deprecated]
            return node.n
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Unsupported constant")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Unsupported function call")
            name = node.func.id
            if name not in _ALLOWED_FUNCS:
                raise ValueError(f"Function {name} not allowed")
            if len(node.args) != 1:
                raise ValueError("Only single-argument functions supported")
            value = self._eval(node.args[0])
            if name in {"sin", "cos", "tan"}:
                value = self._convert_angle(value)
            return _ALLOWED_FUNCS[name](value)
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


__all__ = ["Calculator"]
