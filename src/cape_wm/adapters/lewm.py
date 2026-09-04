from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from ..device import resolve_device
from ..interfaces import WorldModelAdapter
from ..types import Array


class LeWorldModelAdapter(WorldModelAdapter):
    """Adapter for the official ``jepa.JEPA`` checkpoint.

    Observations must be transformed exactly as in the upstream evaluation
    config before encoding. The adapter accepts a transform callable so ImageNet
    normalization and resize remain reproducible per checkpoint.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        action_shape: tuple[int, ...],
        transform: Callable[[Any], torch.Tensor] | None = None,
        goal_metric: Callable[[Array, Array], float] | None = None,
        device: str | torch.device = "auto",
        history_size: int = 3,
        action_block: int = 1,
        action_low: float | Array = -1.0,
        action_high: float | Array = 1.0,
    ) -> None:
        required = ("encode", "predict", "action_encoder")
        missing = [name for name in required if not hasattr(model, name)]
        if missing:
            raise TypeError(f"checkpoint is not an official LeWM JEPA model; missing {missing}")
        resolved_device = resolve_device(device)
        self.model = model.to(resolved_device).eval().requires_grad_(False)
        self.transform = transform
        self.goal_metric = goal_metric
        self.device = resolved_device
        self.history_size = history_size
        if action_block <= 0:
            raise ValueError("action_block must be positive")
        self.action_block = int(action_block)
        self._action_shape = action_shape
        self._action_low = np.broadcast_to(action_low, action_shape).astype(np.float32).copy()
        self._action_high = np.broadcast_to(action_high, action_shape).astype(np.float32).copy()

    def _pixels(self, observation: Any) -> torch.Tensor:
        tensor = (
            self.transform(observation)
            if self.transform is not None
            else torch.as_tensor(observation)
        )
        tensor = tensor.to(self.device, dtype=torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError("transformed observation must have shape [T,C,H,W] or [C,H,W]")
        return tensor.unsqueeze(0)

    @torch.inference_mode()
    def encode(self, observation: Any) -> Array:
        info = self.model.encode({"pixels": self._pixels(observation)})
        return info["emb"][0, -1].detach().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def batch_encode(self, observations: list[Any] | Array) -> Array:
        tensors = []
        for observation in observations:
            tensor = (
                self.transform(observation)
                if self.transform is not None
                else torch.as_tensor(observation)
            )
            tensors.append(tensor.to(self.device, dtype=torch.float32))
        if not tensors:
            return np.empty((0, 0), dtype=np.float32)
        pixels = torch.stack(tensors)
        if pixels.ndim != 4:
            raise ValueError("batched transformed observations must have shape [B,C,H,W]")
        info = self.model.encode({"pixels": pixels[:, None]})
        return info["emb"][:, -1].detach().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def batch_rollout_tensor(
        self, latent: Array | torch.Tensor, action_population: Array | torch.Tensor
    ) -> torch.Tensor:
        """Torch-native rollout used by GPU planners without host round trips."""

        actions = torch.as_tensor(action_population, dtype=torch.float32, device=self.device)
        batch, horizon = actions.shape[:2]
        flat_action_dim = int(np.prod(actions.shape[2:]))
        actions = actions.reshape(batch, horizon, flat_action_dim)
        padding = (-horizon) % self.action_block
        if padding:
            actions = torch.cat(
                (
                    actions,
                    torch.zeros(
                        batch,
                        padding,
                        flat_action_dim,
                        dtype=actions.dtype,
                        device=actions.device,
                    ),
                ),
                dim=1,
            )
        blocks = actions.reshape(batch, -1, self.action_block * flat_action_dim)
        embeddings = torch.as_tensor(latent, dtype=torch.float32, device=self.device)
        embeddings = embeddings.reshape(1, 1, -1).expand(batch, 1, -1).clone()
        action_history: list[torch.Tensor] = []
        predictions: list[torch.Tensor] = []
        for step in range(blocks.shape[1]):
            action_history.append(blocks[:, step : step + 1])
            action_tensor = torch.cat(action_history, dim=1)[:, -self.history_size :]
            state_tensor = embeddings[:, -self.history_size :]
            encoded_actions = self.model.action_encoder(action_tensor)
            prediction = self.model.predict(state_tensor, encoded_actions)[:, -1:]
            embeddings = torch.cat((embeddings, prediction), dim=1)
            predictions.append(prediction[:, 0])
        # LeWM predicts once per action block.  CAPE's durations and execution
        # monitor are expressed in environment steps, so expose a path at that
        # resolution.  A block prediction is the committed endpoint for each
        # of its primitive actions; only block boundaries carry a newly
        # predicted latent.
        path = torch.stack(predictions, dim=1).repeat_interleave(self.action_block, dim=1)
        return path[:, :horizon]

    @torch.inference_mode()
    def batch_rollout(self, latent: Array, action_population: Array) -> Array:
        path = self.batch_rollout_tensor(latent, action_population)
        return path.detach().cpu().numpy().astype(np.float32)

    def rollout(self, latent: Array, actions: Array) -> Array:
        return self.batch_rollout(latent, np.asarray(actions)[None])[0]

    def goal_cost(self, latent: Array, goal_latent: Array) -> float:
        if self.goal_metric is not None:
            return float(self.goal_metric(latent, goal_latent))
        return float(np.sum(np.square(np.asarray(latent) - np.asarray(goal_latent))))

    def batch_goal_cost(self, latents: Array, goal_latent: Array) -> Array:
        batch_metric = getattr(self.goal_metric, "batch", None)
        if callable(batch_metric):
            return np.asarray(batch_metric(latents, goal_latent), dtype=np.float32)
        if self.goal_metric is not None:
            return np.asarray(
                [self.goal_metric(latent, goal_latent) for latent in latents],
                dtype=np.float32,
            )
        return np.sum(np.square(np.asarray(latents) - np.asarray(goal_latent)[None]), axis=-1)

    @property
    def action_shape(self) -> tuple[int, ...]:
        return self._action_shape

    @property
    def action_low(self) -> Array:
        return self._action_low

    @property
    def action_high(self) -> Array:
        return self._action_high
