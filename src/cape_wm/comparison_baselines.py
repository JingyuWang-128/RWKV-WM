from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .baseline import PlanningCounters
from .comparison_models import PairwiseReachabilityMetric, VariableLengthPredictor
from .models import DurationConditionedMacroPredictor


class HybridTRMObjective(nn.Module):
    """Candidate-batch standardized TRM/L2 goal objective."""

    def __init__(
        self,
        trm: PairwiseReachabilityMetric,
        trm_weight: float = 0.9,
        l2_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.trm = trm
        self.trm_weight = trm_weight
        self.l2_weight = l2_weight

    @staticmethod
    def _standardize(value: Tensor) -> Tensor:
        mean = value.mean(dim=1, keepdim=True)
        scale = value.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (value - mean) / scale

    def components(self, source: Tensor, goal: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if goal.ndim == 2:
            goal = goal[:, None, :]
        goal = goal.expand_as(source)
        trm = self.trm(source, goal)
        l2 = (source - goal).square().sum(dim=-1)
        hybrid = self.trm_weight * self._standardize(trm) + self.l2_weight * self._standardize(l2)
        return hybrid, trm, l2

    def forward(self, info_dict: dict[str, Tensor]) -> Tensor:
        predicted = info_dict["predicted_emb"][:, :, -1]
        goal_emb = info_dict["goal_emb"]
        goal = goal_emb[:, 0, -1] if goal_emb.ndim == 4 else goal_emb[:, -1]
        return self.components(predicted, goal)[0]


class VLWMDynamics(nn.Module):
    """Expose a VLWM action-token predictor through stable-worldmodel's rollout API."""

    def __init__(
        self,
        base_model: nn.Module,
        predictor: VariableLengthPredictor,
        schedule: tuple[int, ...],
    ) -> None:
        super().__init__()
        if not schedule or any(length <= 0 for length in schedule):
            raise ValueError("VLWM schedule must contain positive chunks")
        self.base_model = base_model
        self.predictor = predictor
        self.schedule = schedule

    def encode(self, info: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.base_model.encode(info)

    def rollout(self, info: dict[str, Tensor], action_sequence: Tensor) -> dict[str, Tensor]:
        batch, samples, horizon = action_sequence.shape[:3]
        if sum(self.schedule) != horizon:
            raise ValueError(f"VLWM schedule {self.schedule} does not cover horizon {horizon}")
        if "emb" not in info:
            initial = {"pixels": info["pixels"][:, 0]}
            initial = self.base_model.encode(initial)
            info["emb"] = initial["emb"].detach().unsqueeze(1).expand(batch, samples, -1, -1)
        current = info["emb"][:, :, -1]
        trajectory = [current]
        start = 0
        for length in self.schedule:
            chunk = action_sequence[:, :, start : start + length]
            flat_actions = chunk.reshape(batch * samples, length, -1)
            flat_current = current.reshape(batch * samples, -1)
            lengths = torch.full(
                (batch * samples,), length, dtype=torch.long, device=action_sequence.device
            )
            current = self.predictor(flat_current, flat_actions, lengths).reshape(
                batch, samples, -1
            )
            trajectory.append(current)
            start += length
        info["predicted_emb"] = torch.stack(trajectory, dim=2)
        return info


@dataclass(slots=True)
class HierarchicalCounters:
    high_level_invocations: int = 0
    high_level_candidates: int = 0
    high_level_predicted_transitions: int = 0


class HierarchicalMacroSolver:
    """Fixed-duration HWM/Hi-LeWM-C high-level CEM plus shared low-level MPC."""

    def __init__(
        self,
        *,
        base_model: nn.Module,
        macro_predictor: DurationConditionedMacroPredictor,
        low_solver: Any,
        objective: HybridTRMObjective,
        macro_dim: int,
        duration: int,
        high_horizon: int,
        num_samples: int,
        iterations: int,
        elites: int,
        device: torch.device,
        seed: int,
        mode: str,
        empirical_bank: Tensor | None = None,
        empirical_residual_scale: float = 0.25,
        planning_counters: PlanningCounters | None = None,
    ) -> None:
        if mode not in {"hwm", "hilewm_c"}:
            raise ValueError(mode)
        if mode == "hilewm_c" and empirical_bank is None:
            raise ValueError("Hi-LeWM-C requires an empirical macro bank")
        self.base_model = base_model
        self.macro_predictor = macro_predictor
        self.low_solver = low_solver
        self.objective = objective
        self.macro_dim = macro_dim
        self.duration = duration
        self.high_horizon = high_horizon
        self.num_samples = num_samples
        self.iterations = iterations
        self.elites = elites
        self.device = device
        self.mode = mode
        self.empirical_bank = empirical_bank
        self.empirical_residual_scale = empirical_residual_scale
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.high_counters = HierarchicalCounters()
        self.planning_counters = planning_counters

    def configure(self, *, action_space: Any, n_envs: int, config: Any) -> None:
        self.low_solver.configure(action_space=action_space, n_envs=n_envs, config=config)

    @property
    def n_envs(self) -> int:
        return self.low_solver.n_envs

    @property
    def action_dim(self) -> int:
        return self.low_solver.action_dim

    @property
    def horizon(self) -> int:
        return self.low_solver.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]:
        return self.solve(*args, **kwargs)

    def _encode_current_goal(self, info: dict[str, Any]) -> tuple[Tensor, Tensor]:
        model_info = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in info.items()
        }
        goal = self.base_model.encode({"pixels": model_info["goal"]})["emb"][:, -1]
        current_info = {"pixels": model_info["pixels"]}
        current = self.base_model.encode(current_info)["emb"][:, -1]
        return current, goal

    def _initial_candidates(self, batch: int) -> Tensor:
        shape = (batch, self.num_samples, self.high_horizon, self.macro_dim)
        if self.mode == "hwm":
            return torch.randn(shape, generator=self.generator, device=self.device)
        assert self.empirical_bank is not None
        indices = torch.randint(
            len(self.empirical_bank),
            shape[:-1],
            generator=self.generator,
            device=self.device,
        )
        anchors = self.empirical_bank[indices]
        residual = torch.randn(shape, generator=self.generator, device=self.device)
        return anchors + self.empirical_residual_scale * residual

    def _rollout_macros(self, current: Tensor, macros: Tensor) -> Tensor:
        batch, samples = macros.shape[:2]
        latent = current[:, None].expand(batch, samples, -1)
        predictions = []
        duration = torch.full(
            (batch * samples,), self.duration, dtype=torch.long, device=self.device
        )
        for step in range(self.high_horizon):
            latent = self.macro_predictor(
                latent.reshape(batch * samples, -1),
                macros[:, :, step].reshape(batch * samples, -1),
                duration,
            ).reshape(batch, samples, -1)
            predictions.append(latent)
        return torch.stack(predictions, dim=2)

    @torch.inference_mode()
    def _plan_subgoal(self, current: Tensor, goal: Tensor) -> Tensor:
        batch = len(current)
        candidates = self._initial_candidates(batch)
        mean = candidates.mean(dim=1)
        std = candidates.std(dim=1, unbiased=False).clamp_min(0.05)
        first_predictions = None
        for iteration in range(self.iterations):
            if iteration:
                noise = torch.randn(candidates.shape, generator=self.generator, device=self.device)
                candidates = mean[:, None] + std[:, None] * noise
            predictions = self._rollout_macros(current, candidates)
            costs = self.objective.components(predictions[:, :, -1], goal[:, None])[0]
            elite_ids = torch.topk(costs, self.elites, dim=1, largest=False).indices
            batch_ids = torch.arange(batch, device=self.device)[:, None]
            elite = candidates[batch_ids, elite_ids]
            mean = elite.mean(dim=1)
            std = elite.std(dim=1, unbiased=False).clamp_min(0.02)
            first_predictions = self._rollout_macros(current, mean[:, None])[:, 0, 0]
        self.high_counters.high_level_invocations += 1
        self.high_counters.high_level_candidates += batch * self.num_samples * self.iterations
        transitions = batch * self.num_samples * self.iterations * self.high_horizon
        self.high_counters.high_level_predicted_transitions += transitions
        if self.planning_counters is not None:
            self.planning_counters.high_level_invocations += 1
            self.planning_counters.high_level_candidates += (
                batch * self.num_samples * self.iterations
            )
            self.planning_counters.high_level_predicted_transitions += transitions
        assert first_predictions is not None
        return first_predictions

    @torch.inference_mode()
    def solve(
        self, info_dict: dict[str, Any], init_action: Tensor | None = None
    ) -> dict[str, Tensor]:
        current, goal = self._encode_current_goal(info_dict)
        subgoal = self._plan_subgoal(current, goal)
        low_info = dict(info_dict)
        low_info["goal_emb"] = subgoal[:, None]
        return self.low_solver.solve(low_info, init_action=init_action)
