from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .planner import CAPEPlanner
from .types import Array, PlanDiagnostics


class GoalImagePolicy:
    """Small policy bridge for external environment/evaluation frameworks.

    The bridge deliberately avoids importing stable-worldmodel.  It supports
    both ``policy(observation, goal)`` and dict observations containing image
    and goal-image keys, which keeps the core package insulated from upstream
    evaluator API changes.
    """

    def __init__(
        self,
        planner: CAPEPlanner,
        observation_key: str = "pixels",
        goal_key: str = "goal_pixels",
    ) -> None:
        self.planner = planner
        self.observation_key = observation_key
        self.goal_key = goal_key
        self.last_diagnostics: PlanDiagnostics | None = None

    def reset(self) -> None:
        self.planner.reset()
        self.last_diagnostics = None

    def act(self, observation: Any, goal_image: Any | None = None) -> Array:
        if goal_image is None:
            if not isinstance(observation, dict):
                raise TypeError("goal_image is required when observation is not a mapping")
            goal_image = observation[self.goal_key]
            observation = observation[self.observation_key]
        action, diagnostics = self.planner.plan(observation, goal_image)
        self.last_diagnostics = diagnostics
        return action

    def __call__(self, observation: Any, goal_image: Any | None = None) -> Array:
        return self.act(observation, goal_image)


class StableWorldModelCAPEPolicy:
    """Vectorized ``get_action(info)`` bridge for ``swm.World.set_policy``."""

    def __init__(
        self,
        planner_factory: Callable[[], CAPEPlanner],
        num_envs: int,
        pixels_key: str = "pixels",
        goal_key: str = "goal",
        action_postprocess: Callable[[Array], Array] | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.planners = [planner_factory() for _ in range(num_envs)]
        self.pixels_key = pixels_key
        self.goal_key = goal_key
        self.action_postprocess = action_postprocess
        self.last_diagnostics: list[PlanDiagnostics | None] = []
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.diagnostic_history: list[list[PlanDiagnostics]] = [
            [] for _ in range(num_envs)
        ]

    def set_env(self, env: Any) -> None:
        n_envs = int(getattr(env, "num_envs", 1))
        if n_envs != len(self.planners):
            raise ValueError(f"policy was built for {len(self.planners)} envs, received {n_envs}")
        self.env = env

    def reset(self) -> None:
        for planner in self.planners:
            planner.reset()
        self.last_diagnostics = []
        self.steps.fill(0)
        self.diagnostic_history = [[] for _ in self.planners]

    @staticmethod
    def _last_frame(value: Any, batch: int) -> np.ndarray:
        array = np.asarray(value)
        if array.shape[0] != batch:
            raise ValueError("stable-worldmodel info has an unexpected environment batch")
        # World infos normally use [B,T,H,W,C]. A direct environment wrapper
        # may instead provide [B,H,W,C].
        if array.ndim >= 5:
            return array[:, -1]
        return array

    def get_action(self, info: dict[str, Any]) -> Array:
        batch = len(self.planners)
        observations = self._last_frame(info[self.pixels_key], batch)
        goals = self._last_frame(info[self.goal_key], batch)
        reset_value = info.get(
            "_needs_flush",
            info.get("is_first", info.get("reset", np.zeros(batch, dtype=bool))),
        )
        first = np.asarray(reset_value)
        if first.ndim > 1:
            first = first[:, -1]
        first = first.reshape(-1)
        terminated = np.asarray(
            info.get("terminated", np.zeros(batch, dtype=bool)), dtype=bool
        ).reshape(-1)
        truncated = np.asarray(
            info.get("truncated", np.zeros(batch, dtype=bool)), dtype=bool
        ).reshape(-1)
        dead = terminated | truncated
        actions = []
        diagnostics: list[PlanDiagnostics | None] = []
        for index, planner in enumerate(self.planners):
            if index < len(first) and first[index]:
                planner.reset()
                self.steps[index] = 0
                self.diagnostic_history[index] = []
            if index < len(dead) and dead[index]:
                actions.append(np.zeros(planner.model.action_shape, dtype=np.float32))
                diagnostics.append(None)
                continue
            action, diagnostic = planner.plan(observations[index], goals[index])
            actions.append(action)
            diagnostics.append(diagnostic)
            self.steps[index] += 1
            self.diagnostic_history[index].append(diagnostic)
        self.last_diagnostics = diagnostics
        result = np.stack(actions).astype(np.float32)
        if self.action_postprocess is not None:
            result = np.asarray(self.action_postprocess(result), dtype=np.float32)
        return result
