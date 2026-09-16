"""Observable, fixed-threshold gradient stability gates (not clipping tricks)."""

from __future__ import annotations

import math
from statistics import median


class StabilityGateError(RuntimeError):
    """Training is finite but fails its registered gradient quality gate."""


def gradient_window_report(
    norms: list[float], *, clip_norm: float = 1.0, max_clip_fraction: float = 0.2
) -> dict[str, float | int | bool]:
    if not norms or clip_norm <= 0 or not 0 <= max_clip_fraction <= 1:
        raise ValueError("invalid stability window or thresholds")
    if not all(math.isfinite(value) and value >= 0 for value in norms):
        raise FloatingPointError("non-finite or negative gradient norm")
    scales = [min(1.0, clip_norm / (value + 1e-6)) for value in norms]
    fraction = sum(value > clip_norm for value in norms) / len(norms)
    # A low frequency alone can hide catastrophic isolated spikes.
    severe_fraction = sum(value < 0.1 for value in scales) / len(scales)
    passed = fraction <= max_clip_fraction and severe_fraction == 0
    return {
        "samples": len(norms),
        "clip_norm": clip_norm,
        "max_clip_fraction": max_clip_fraction,
        "clip_fraction": fraction,
        "severe_clip_fraction": severe_fraction,
        "grad_median": median(norms),
        "grad_max": max(norms),
        "scale_median": median(scales),
        "scale_min": min(scales),
        "passed": passed,
    }


def learning_rate_factor(
    step: int, *, max_steps: int, warmup_steps: int = 0, min_ratio: float = 1.0
) -> float:
    if not 1 <= step <= max_steps or not 0 <= warmup_steps < max_steps:
        raise ValueError("invalid learning rate schedule steps")
    if not 0 < min_ratio <= 1:
        raise ValueError("min_ratio must be in (0, 1]")
    if warmup_steps and step <= warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_ratio + (1 - min_ratio) * (1 + math.cos(math.pi * progress)) / 2
