from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .types import Array, CandidatePlan


class WorldModelAdapter(ABC):
    """Minimal frozen-backbone interface used by CAPE-WM.

    Implementations may wrap LeWM, DINO-WM, or a small deterministic model.  The
    planner never receives simulator state through this interface.
    """

    @abstractmethod
    def encode(self, observation: Any) -> Array:
        """Encode a single image observation into a flat planner-facing latent."""

    @abstractmethod
    def rollout(self, latent: Array, actions: Array) -> Array:
        """Return predicted latents after every action with shape ``[T, latent_dim]``."""

    @abstractmethod
    def goal_cost(self, latent: Array, goal_latent: Array) -> float:
        """Return the shared, horizon-matched terminal cost."""

    @property
    @abstractmethod
    def action_shape(self) -> tuple[int, ...]:
        """Shape of one continuous action."""

    @property
    def action_low(self) -> Array:
        return np.full(self.action_shape, -1.0, dtype=np.float32)

    @property
    def action_high(self) -> Array:
        return np.full(self.action_shape, 1.0, dtype=np.float32)


class CandidateGenerator(ABC):
    """Generate duration-specific high-level proposals under a matched budget."""

    @abstractmethod
    def propose(
        self,
        current_latent: Array,
        goal_latent: Array,
        durations: tuple[int, ...],
        max_duration: int,
    ) -> list[CandidatePlan]:
        """Return zero or more candidates for every allowed duration."""

    def refine(
        self,
        current_latent: Array,
        candidate: CandidatePlan,
        remaining_duration: int,
    ) -> CandidatePlan:
        """Optionally replan the low-level actions while preserving the subgoal."""

        return candidate

    def fallback(
        self,
        current_latent: Array,
        goal_latent: Array,
        duration: int,
    ) -> CandidatePlan | None:
        """Return the shortest standard low-level MPC plan when hierarchy is unsafe."""

        candidates = self.propose(
            current_latent,
            goal_latent,
            durations=(duration,),
            max_duration=duration,
        )
        return candidates[0] if candidates else None


class RiskEstimator(ABC):
    """Predict and calibrate low-level executability of a proposed subgoal."""

    @abstractmethod
    def assess(
        self, current_latent: Array, subgoal: Array, duration: int
    ) -> tuple[float, float, float]:
        """Return ``(predicted_miss, miss_upper_bound, success_probability)``."""

    @abstractmethod
    def tube_threshold(
        self,
        current_latent: Array,
        subgoal: Array,
        duration: int,
        step: int,
    ) -> float:
        """Return the calibrated simultaneous execution-tube threshold."""


def ensure_flat_latent(value: Any) -> Array:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        raise ValueError("latent must contain at least one feature")
    if array.ndim > 1:
        array = array.reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError("latent contains non-finite values")
    return array
