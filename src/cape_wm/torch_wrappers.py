from __future__ import annotations

import numpy as np
import torch

from .comparison_models import PairwiseReachabilityMetric
from .device import resolve_device
from .models import (
    DurationConditionedMacroPredictor,
    ExecutabilityRiskHead,
    TrajectoryReachabilityMetric,
)
from .types import Array


class TorchMacroDynamics:
    def __init__(
        self,
        predictor: DurationConditionedMacroPredictor,
        macro_dim: int,
        device: str | torch.device = "auto",
    ) -> None:
        device = resolve_device(device)
        self.predictor = predictor.to(device).eval()
        self._macro_dim = macro_dim
        self.device = torch.device(device)

    @property
    def macro_dim(self) -> int:
        return self._macro_dim

    @torch.inference_mode()
    def predict(self, latent: Array, macro_action: Array, duration: int) -> Array:
        latent_tensor = torch.as_tensor(latent, dtype=torch.float32, device=self.device).reshape(
            1, -1
        )
        macro_tensor = torch.as_tensor(
            macro_action, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        duration_tensor = torch.as_tensor([duration], device=self.device)
        return (
            self.predictor(latent_tensor, macro_tensor, duration_tensor)[0].detach().cpu().numpy()
        )

    @torch.inference_mode()
    def batch_predict(self, latent: Array, macro_action: Array, duration: int) -> Array:
        latent_tensor = torch.as_tensor(latent, dtype=torch.float32, device=self.device)
        macro_tensor = torch.as_tensor(macro_action, dtype=torch.float32, device=self.device)
        durations = torch.full(
            (len(latent_tensor),), duration, dtype=torch.long, device=self.device
        )
        return (
            self.predictor(latent_tensor, macro_tensor, durations)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )


class TorchRiskPredictor:
    def __init__(
        self,
        model: ExecutabilityRiskHead,
        device: str | torch.device = "auto",
        miss_scale: float = 1.0,
    ) -> None:
        device = resolve_device(device)
        if miss_scale <= 0:
            raise ValueError("miss_scale must be positive")
        self.model = model.to(device).eval()
        self.device = device
        self.miss_scale = float(miss_scale)

    @torch.inference_mode()
    def __call__(self, current: Array, subgoal: Array, duration: int) -> tuple[float, float, Array]:
        current_tensor = torch.as_tensor(current, dtype=torch.float32, device=self.device).reshape(
            1, -1
        )
        subgoal_tensor = torch.as_tensor(subgoal, dtype=torch.float32, device=self.device).reshape(
            1, -1
        )
        duration_tensor = torch.as_tensor([duration], device=self.device)
        miss, success, scale = self.model(current_tensor, subgoal_tensor, duration_tensor)
        return (
            float(miss.item()) * self.miss_scale,
            float(success.item()),
            scale[0, :duration].detach().cpu().numpy().astype(np.float32),
        )


class TorchTRMCost:
    def __init__(
        self,
        model: TrajectoryReachabilityMetric,
        horizon: int = 100,
        trm_weight: float = 0.9,
        l2_weight: float = 0.1,
        device: str | torch.device = "auto",
    ) -> None:
        device = resolve_device(device)
        self.model = model.to(device).eval()
        self.horizon = horizon
        self.trm_weight = trm_weight
        self.l2_weight = l2_weight
        self.device = device

    @torch.inference_mode()
    def __call__(self, source: Array, goal: Array) -> float:
        source_tensor = torch.as_tensor(source, dtype=torch.float32, device=self.device).reshape(
            1, -1
        )
        goal_tensor = torch.as_tensor(goal, dtype=torch.float32, device=self.device).reshape(1, -1)
        horizon = torch.as_tensor([self.horizon], device=self.device)
        reachability = self.model(source_tensor, goal_tensor, horizon).item()
        l2 = float(np.mean(np.square(np.asarray(source) - np.asarray(goal))))
        return float(self.trm_weight * reachability + self.l2_weight * l2)

    @torch.inference_mode()
    def batch(self, sources: Array, goal: Array) -> Array:
        source_tensor = torch.as_tensor(sources, dtype=torch.float32, device=self.device)
        goal_tensor = (
            torch.as_tensor(goal, dtype=torch.float32, device=self.device)
            .reshape(1, -1)
            .expand(len(source_tensor), -1)
        )
        horizons = torch.full(
            (len(source_tensor),), self.horizon, dtype=torch.long, device=self.device
        )
        reachability = self.model(source_tensor, goal_tensor, horizons)
        l2 = (source_tensor - goal_tensor).square().mean(dim=-1)
        return (
            (self.trm_weight * reachability + self.l2_weight * l2)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )


class TorchPairwiseHybridCost:
    """Expose the shared Two-Room TRM/L2 mixture to the NumPy CAPE planner."""

    def __init__(
        self,
        model: PairwiseReachabilityMetric,
        trm_weight: float = 0.9,
        l2_weight: float = 0.1,
        device: str | torch.device = "auto",
    ) -> None:
        device = resolve_device(device)
        self.model = model.to(device).eval()
        self.trm_weight = float(trm_weight)
        self.l2_weight = float(l2_weight)
        self.device = device

    @staticmethod
    def _standardize(value: torch.Tensor) -> torch.Tensor:
        return (value - value.mean()) / value.std(unbiased=False).clamp_min(1e-6)

    @torch.inference_mode()
    def __call__(self, source: Array, goal: Array) -> float:
        source_tensor = torch.as_tensor(source, dtype=torch.float32, device=self.device).reshape(
            1, -1
        )
        goal_tensor = torch.as_tensor(goal, dtype=torch.float32, device=self.device).reshape(1, -1)
        trm = self.model(source_tensor, goal_tensor).item()
        l2 = (source_tensor - goal_tensor).square().mean().item()
        return float(self.trm_weight * trm + self.l2_weight * l2)

    @torch.inference_mode()
    def batch(self, sources: Array, goal: Array) -> Array:
        source_tensor = torch.as_tensor(sources, dtype=torch.float32, device=self.device)
        goal_tensor = (
            torch.as_tensor(goal, dtype=torch.float32, device=self.device)
            .reshape(1, -1)
            .expand(len(source_tensor), -1)
        )
        trm = self.model(source_tensor, goal_tensor)
        l2 = (source_tensor - goal_tensor).square().sum(dim=-1)
        hybrid = self.trm_weight * self._standardize(trm) + self.l2_weight * self._standardize(l2)
        return hybrid.detach().cpu().numpy().astype(np.float32)
