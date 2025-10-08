"""Utility helpers for lightweight scalar schedules used by the Vertex trainer."""

from __future__ import annotations

from typing import Callable, Iterable, Tuple
import math


ScheduleFn = Callable[[int], float]


def _build_piecewise_schedule(pairs: Iterable[Tuple[int, float]], default: float) -> ScheduleFn:
    points = sorted(((int(step), float(val)) for step, val in pairs), key=lambda item: item[0])

    def schedule(step: int) -> float:
        value = default
        for trigger, val in points:
            if step >= trigger:
                value = val
            else:
                break
        return value

    return schedule


def build_scalar_schedule(spec: str | None, base_value: float | None) -> ScheduleFn:
    """Parse a scalar schedule specification.

    Parameters
    ----------
    spec:
        Either ``None`` (identity schedule), a typed schedule description such as
        ``"cosine:hold_steps=2000,start=0.2,end=0.05,total_steps=25000"`` or
        ``"linear:start=1.5,end=1.0,total_steps=10000"``, or a legacy comma-separated
        list of ``step:value`` entries.
    base_value:
        Fallback scalar to use when ``spec`` is ``None``. If the schedule omits a
        ``start`` value this fallback is also used.

    Returns
    -------
    Callable[[int], float]
        Function mapping an integer step (>=0) to the scheduled scalar value.
    """

    if spec is None:
        default = 0.0 if base_value is None else float(base_value)

        def identity(step: int) -> float:  # pragma: no cover - trivial branch
            return default

        return identity

    if isinstance(spec, str):
        text = spec.strip()
    else:
        text = str(spec).strip()

    if not text:
        return build_scalar_schedule(None, base_value)

    # Legacy format "step:value,step:value"
    if ":" in text and "=" not in text.partition(":")[2]:
        pairs: list[Tuple[int, float]] = []
        for item in text.split(','):
            item = item.strip()
            if not item:
                continue
            if ':' not in item:
                raise ValueError(f"Invalid schedule entry '{item}' (expected step:value)")
            step_s, val_s = item.split(':', 1)
            pairs.append((int(step_s.strip()), float(val_s.strip())))
        default = float(base_value) if base_value is not None else (pairs[0][1] if pairs else 0.0)
        return _build_piecewise_schedule(pairs, default)

    kind, _, params_text = text.partition(':')
    kind = kind.strip().lower()
    params = {}
    if params_text:
        for item in params_text.split(','):
            item = item.strip()
            if not item:
                continue
            if '=' not in item:
                raise ValueError(f"Invalid schedule parameter '{item}' (expected key=value)")
            key, value = item.split('=', 1)
            params[key.strip().lower()] = value.strip()

    def get_float(name: str, fallback: float | None) -> float:
        if name in params:
            return float(params[name])
        if fallback is None:
            raise ValueError(f"Schedule parameter '{name}' is required for spec '{text}'")
        return float(fallback)

    if kind == 'linear':
        start = get_float('start', base_value)
        end = get_float('end', start)
        total_steps = int(float(params.get('total_steps', 0)))
        if total_steps <= 0:
            raise ValueError("linear schedule requires total_steps > 0")

        def linear(step: int) -> float:
            step = max(0, int(step))
            progress = min(1.0, step / float(total_steps))
            return float(start + (end - start) * progress)

        return linear

    if kind == 'cosine':
        start = get_float('start', base_value)
        end = get_float('end', start)
        total_steps = int(float(params.get('total_steps', 0)))
        hold_steps = int(float(params.get('hold_steps', 0)))
        if total_steps <= 0:
            raise ValueError("cosine schedule requires total_steps > 0")
        if hold_steps < 0:
            raise ValueError("cosine schedule hold_steps must be >= 0")

        active_steps = max(0, total_steps - hold_steps)

        def cosine(step: int) -> float:
            step = max(0, int(step))
            if step < hold_steps:
                return float(start)
            if active_steps <= 0:
                return float(end)
            rel = min(1.0, (step - hold_steps) / float(active_steps))
            weight = 0.5 * (1.0 + math.cos(math.pi * rel))
            return float(end + (start - end) * weight)

        return cosine

    raise ValueError(f"Unknown schedule kind '{kind}' in spec '{text}'")
