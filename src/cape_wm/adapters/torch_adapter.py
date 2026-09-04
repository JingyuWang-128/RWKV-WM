from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..interfaces import WorldModelAdapter, ensure_flat_latent
from ..types import Array


class TorchLatentWorldModelAdapter(WorldModelAdapter):
    """Generic adapter for DINO-WM or another PyTorch latent model.

    Model-specific tensor layout remains in the supplied callables, which avoids
    coupling CAPE-WM to unstable upstream private APIs.
    """

    def __init__(
        self,
        encoder: Callable[[Any], Array],
        rollout_fn: Callable[[Array, Array], Array],
        action_shape: tuple[int, ...],
        goal_metric: Callable[[Array, Array], float] | None = None,
        batch_rollout_fn: Callable[[Array, Array], Array] | None = None,
        batch_goal_metric: Callable[[Array, Array], Array] | None = None,
        action_low: float | Array = -1.0,
        action_high: float | Array = 1.0,
    ) -> None:
        self.encoder = encoder
        self.rollout_fn = rollout_fn
        self._action_shape = action_shape
        self.goal_metric = goal_metric
        self.batch_rollout_fn = batch_rollout_fn
        self.batch_goal_metric = batch_goal_metric
        self._action_low = np.broadcast_to(action_low, action_shape).astype(np.float32).copy()
        self._action_high = np.broadcast_to(action_high, action_shape).astype(np.float32).copy()

    def encode(self, observation: Any) -> Array:
        return ensure_flat_latent(self.encoder(observation))

    def rollout(self, latent: Array, actions: Array) -> Array:
        result = np.asarray(self.rollout_fn(latent, actions), dtype=np.float32)
        if result.shape[0] != len(actions):
            raise ValueError("rollout callable must return one latent per action")
        return result

    def batch_rollout(self, latent: Array, action_population: Array) -> Array:
        if self.batch_rollout_fn is not None:
            return np.asarray(self.batch_rollout_fn(latent, action_population), dtype=np.float32)
        return np.stack([self.rollout(latent, actions) for actions in action_population])

    def goal_cost(self, latent: Array, goal_latent: Array) -> float:
        if self.goal_metric is not None:
            return float(self.goal_metric(latent, goal_latent))
        return float(np.mean(np.square(np.asarray(latent) - np.asarray(goal_latent))))

    def batch_goal_cost(self, latents: Array, goal_latent: Array) -> Array:
        if self.batch_goal_metric is not None:
            return np.asarray(self.batch_goal_metric(latents, goal_latent), dtype=np.float32)
        if self.goal_metric is not None:
            return np.asarray(
                [self.goal_metric(latent, goal_latent) for latent in latents],
                dtype=np.float32,
            )
        return np.mean(np.square(np.asarray(latents) - np.asarray(goal_latent)[None]), axis=-1)

    @property
    def action_shape(self) -> tuple[int, ...]:
        return self._action_shape

    @property
    def action_low(self) -> Array:
        return self._action_low

    @property
    def action_high(self) -> Array:
        return self._action_high
