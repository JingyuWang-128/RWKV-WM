from __future__ import annotations

from typing import Any

import numpy as np

from ..interfaces import WorldModelAdapter
from ..types import Array


class PointImageEnv:
    """Tiny image-only control environment for end-to-end smoke tests."""

    def __init__(self, image_size: int = 32, action_scale: float = 0.08) -> None:
        self.image_size = image_size
        self.action_scale = action_scale
        self.state = np.zeros(2, dtype=np.float32)
        self.goal = np.asarray((0.8, 0.8), dtype=np.float32)
        self.steps = 0

    def render_state(self, state: Array) -> Array:
        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        pixel = np.rint(np.clip(state, 0.0, 1.0) * (self.image_size - 1)).astype(int)
        x, y = int(pixel[0]), int(pixel[1])
        image[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = 1.0
        return image

    def reset(self, *, seed: int = 0, options: dict[str, Any] | None = None):
        rng = np.random.default_rng(seed)
        options = options or {}
        self.state = np.asarray(
            options.get("start", rng.uniform(0.05, 0.25, size=2)), dtype=np.float32
        )
        self.goal = np.asarray(options.get("goal", (0.8, 0.8)), dtype=np.float32)
        self.steps = 0
        return self.render_state(self.state), {
            "state_distance": float(np.linalg.norm(self.state - self.goal))
        }

    def step(self, action: Array):
        self.state = np.clip(
            self.state + self.action_scale * np.clip(action, -1.0, 1.0), 0.0, 1.0
        ).astype(np.float32)
        self.steps += 1
        distance = float(np.linalg.norm(self.state - self.goal))
        success = distance <= 0.045
        return (
            self.render_state(self.state),
            -distance,
            success,
            False,
            {
                "success": success,
                "state_distance": distance,
            },
        )


class PointImageWorldModel(WorldModelAdapter):
    """Exact latent dynamics whose encoder only receives rendered pixels."""

    def __init__(self, action_scale: float = 0.08, image_size: int = 32) -> None:
        self.action_scale = action_scale
        self.image_size = image_size

    def encode(self, observation: Any) -> Array:
        image = np.asarray(observation, dtype=np.float32)
        weights = image.mean(axis=-1) if image.ndim == 3 else image
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("toy observation contains no visible point")
        ys, xs = np.indices(weights.shape)
        return np.asarray(
            [(xs * weights).sum() / total, (ys * weights).sum() / total], dtype=np.float32
        ) / (self.image_size - 1)

    def rollout(self, latent: Array, actions: Array) -> Array:
        state = np.asarray(latent, dtype=np.float32).copy()
        predictions = []
        for action in np.asarray(actions, dtype=np.float32):
            state = np.clip(state + self.action_scale * np.clip(action, -1.0, 1.0), 0.0, 1.0)
            predictions.append(state.copy())
        return np.asarray(predictions, dtype=np.float32)

    def batch_rollout(self, latent: Array, action_population: Array) -> Array:
        actions = np.asarray(action_population, dtype=np.float32)
        state = np.broadcast_to(np.asarray(latent, dtype=np.float32), (len(actions), 2)).copy()
        paths = []
        for step in range(actions.shape[1]):
            state = np.clip(
                state + self.action_scale * np.clip(actions[:, step], -1.0, 1.0), 0.0, 1.0
            )
            paths.append(state.copy())
        return np.stack(paths, axis=1)

    def goal_cost(self, latent: Array, goal_latent: Array) -> float:
        return float(np.linalg.norm(np.asarray(latent) - np.asarray(goal_latent)))

    def batch_goal_cost(self, latents: Array, goal_latent: Array) -> Array:
        return np.linalg.norm(np.asarray(latents) - np.asarray(goal_latent)[None], axis=-1)

    @property
    def action_shape(self) -> tuple[int, ...]:
        return (2,)


class PointMacroDynamics:
    macro_dim = 2

    def __init__(self, step_scale: float = 0.06) -> None:
        self.step_scale = step_scale

    def predict(self, latent: Array, macro_action: Array, duration: int) -> Array:
        displacement = np.tanh(np.asarray(macro_action, dtype=np.float32))
        return np.clip(np.asarray(latent) + self.step_scale * duration * displacement, 0.0, 1.0)

    def batch_predict(self, latent: Array, macro_action: Array, duration: int) -> Array:
        displacement = np.tanh(np.asarray(macro_action, dtype=np.float32))
        return np.clip(np.asarray(latent) + self.step_scale * duration * displacement, 0.0, 1.0)


class PointRiskPredictor:
    def __call__(self, current: Array, subgoal: Array, duration: int) -> tuple[float, float, Array]:
        distance = float(np.linalg.norm(np.asarray(subgoal) - np.asarray(current)))
        reachable_distance = 0.075 * duration
        miss = 0.01 + max(0.0, distance - reachable_distance)
        probability = float(np.exp(-8.0 * miss))
        scale = np.full(duration, 0.012 + 0.0002 * duration, dtype=np.float32)
        return miss, probability, scale
