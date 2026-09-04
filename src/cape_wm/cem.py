from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .types import Array


@dataclass(frozen=True, slots=True)
class CEMConfig:
    samples: int = 256
    elites: int = 32
    iterations: int = 4
    momentum: float = 0.1
    min_std: float = 0.03
    seed: int = 0
    clip: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.elites <= self.samples:
            raise ValueError("elites must lie in [1, samples]")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")


@dataclass(slots=True)
class CEMResult:
    value: Array
    objective: float
    evaluations: int
    mean: Array
    std: Array


def cem_optimize(
    objective: Callable[[Array], Array],
    shape: tuple[int, ...],
    low: float | Array,
    high: float | Array,
    config: CEMConfig,
    initial_mean: Array | None = None,
    initial_std: Array | None = None,
    initial_population: Array | None = None,
) -> CEMResult:
    """Minimize a batched objective with a reproducible truncated-Gaussian CEM."""

    rng = np.random.default_rng(config.seed)
    low_array = np.broadcast_to(np.asarray(low, dtype=np.float32), shape)
    high_array = np.broadcast_to(np.asarray(high, dtype=np.float32), shape)
    mean = (
        np.asarray(initial_mean, dtype=np.float32).copy()
        if initial_mean is not None
        else (low_array + high_array) / 2.0
    )
    std = (
        np.asarray(initial_std, dtype=np.float32).copy()
        if initial_std is not None
        else (high_array - low_array) / 2.0
    )
    best_value = mean.copy()
    best_objective = float("inf")

    if initial_population is not None:
        initial_population = np.asarray(initial_population, dtype=np.float32)
        if initial_population.shape != (config.samples, *shape):
            raise ValueError("initial_population has the wrong shape")

    for iteration in range(config.iterations):
        if iteration == 0 and initial_population is not None:
            population = initial_population.copy()
        else:
            noise = rng.standard_normal((config.samples, *shape), dtype=np.float32)
            population = mean + std * noise
            if config.clip:
                population = np.clip(population, low_array, high_array)
        scores = np.asarray(objective(population), dtype=np.float64).reshape(-1)
        if scores.shape != (config.samples,):
            raise ValueError("objective must return one score per population member")
        if not np.all(np.isfinite(scores)):
            scores = np.nan_to_num(scores, nan=np.inf, posinf=np.inf, neginf=-np.inf)
        elite_indices = np.argpartition(scores, config.elites - 1)[: config.elites]
        elites = population[elite_indices]
        elite_mean = elites.mean(axis=0)
        elite_std = elites.std(axis=0)
        mean = config.momentum * mean + (1.0 - config.momentum) * elite_mean
        std = np.maximum(
            config.momentum * std + (1.0 - config.momentum) * elite_std,
            config.min_std,
        )
        iteration_best = int(np.argmin(scores))
        if scores[iteration_best] < best_objective:
            best_objective = float(scores[iteration_best])
            best_value = population[iteration_best].copy()

    return CEMResult(
        value=best_value,
        objective=best_objective,
        evaluations=config.samples * config.iterations,
        mean=mean,
        std=std,
    )
