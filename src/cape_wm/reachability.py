from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .conformal import conformal_quantile
from .interfaces import WorldModelAdapter, ensure_flat_latent
from .models import ContinuousReachabilityDistribution, DirectedReachabilityDistribution
from .types import Array, ExecutionEvent, PlanDiagnostics


class ReachabilityAdvantageCalibrator:
    """Mondrian conformal lower bounds for realized reachability progress.

    Calibration targets the exact quantity optimized by the planner.  It does
    not reject candidates: each duration receives a conservative correction
    and remains comparable in the common progress-per-step objective.
    """

    method = "duration_conditional_reachability_advantage_v1"

    def __init__(self, alpha: float = 0.1, scale_mode: str = "predicted") -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if scale_mode not in {"predicted", "constant"}:
            raise ValueError("scale_mode must be 'predicted' or 'constant'")
        self.alpha = float(alpha)
        self.scale_mode = scale_mode
        self.quantiles: dict[int, float] = {}
        self.counts: dict[int, int] = {}

    def fit(
        self,
        predicted_progress: np.ndarray,
        observed_progress: np.ndarray,
        durations: np.ndarray,
        predicted_scale: np.ndarray | None = None,
    ) -> ReachabilityAdvantageCalibrator:
        predicted = np.asarray(predicted_progress, dtype=np.float64).reshape(-1)
        observed = np.asarray(observed_progress, dtype=np.float64).reshape(-1)
        duration = np.asarray(durations, dtype=np.int64).reshape(-1)
        scale = np.ones_like(predicted)
        if self.scale_mode == "predicted" and predicted_scale is not None:
            scale = np.asarray(predicted_scale, dtype=np.float64).reshape(-1)
        if not (len(predicted) == len(observed) == len(duration) == len(scale)):
            raise ValueError("calibration arrays must have equal lengths")
        if not len(predicted):
            raise ValueError("at least one calibration example is required")
        if np.any(~np.isfinite(predicted)) or np.any(~np.isfinite(observed)):
            raise ValueError("progress calibration values must be finite")
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("predicted scales must be finite and positive")
        self.quantiles.clear()
        self.counts.clear()
        # Positive scores mean the planner was over-optimistic.  Their
        # conformal upper quantile is subtracted to form a progress lower bound.
        scores = (predicted - observed) / scale
        for value in sorted(set(duration.tolist())):
            group = scores[duration == value]
            self.quantiles[int(value)] = conformal_quantile(group, self.alpha)
            self.counts[int(value)] = int(len(group))
        return self

    def _quantile(self, duration: int) -> float:
        try:
            return self.quantiles[int(duration)]
        except KeyError as error:
            raise ValueError(f"duration {duration} has no advantage calibration stratum") from error

    def quantile(self, duration: int) -> float:
        return self._quantile(duration)

    def lower_bound(
        self,
        predicted_progress: float | np.ndarray,
        predicted_scale: float | np.ndarray,
        duration: int,
    ) -> float | np.ndarray:
        predicted = np.asarray(predicted_progress, dtype=np.float64)
        scale = np.asarray(predicted_scale, dtype=np.float64)
        if np.any(scale <= 0.0):
            raise ValueError("predicted scale must be positive")
        if self.scale_mode == "constant":
            scale = np.ones_like(scale)
        result = predicted - self._quantile(duration) * scale
        return float(result) if result.ndim == 0 else result

    def as_dict(self) -> dict:
        if not self.quantiles:
            raise RuntimeError("advantage calibrator has not been fitted")
        return {
            "method": self.method,
            "alpha": self.alpha,
            "scale_mode": self.scale_mode,
            "by_duration": {
                str(duration): {"quantile": quantile, "count": self.counts[duration]}
                for duration, quantile in self.quantiles.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ReachabilityAdvantageCalibrator:
        if payload.get("method") != cls.method:
            raise ValueError("unsupported reachability advantage calibration artifact")
        result = cls(
            alpha=float(payload["alpha"]),
            scale_mode=str(payload.get("scale_mode", "predicted")),
        )
        for duration, item in payload["by_duration"].items():
            result.quantiles[int(duration)] = float(item["quantile"])
            result.counts[int(duration)] = int(item["count"])
        if not result.quantiles:
            raise ValueError("calibration artifact has no duration strata")
        return result

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> ReachabilityAdvantageCalibrator:
        return cls.from_dict(json.loads(Path(path).read_text()))


class CalibratedReachabilityScorer:
    """Score shared action prefixes by conformal reachability progress rate."""

    def __init__(
        self,
        model: DirectedReachabilityDistribution | ContinuousReachabilityDistribution,
        calibrator: ReachabilityAdvantageCalibrator,
        durations: tuple[int, ...],
    ) -> None:
        if not durations or any(value <= 0 for value in durations):
            raise ValueError("durations must be positive")
        if tuple(sorted(set(durations))) != tuple(durations):
            raise ValueError("durations must be strictly increasing")
        missing = set(durations) - set(calibrator.quantiles)
        if missing:
            raise ValueError(f"missing conformal strata for durations {sorted(missing)}")
        self.model = model
        self.calibrator = calibrator
        self.durations = tuple(map(int, durations))

    @torch.inference_mode()
    def score_paths(
        self,
        current: Tensor,
        goal: Tensor,
        paths: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return rates, lower bounds, predicted progress, and uncertainty scale.

        ``paths`` has shape ``[candidates, primitive_steps, latent_dim]``.
        Every duration is evaluated on a prefix of the same imagined action
        sequence; no alternate controller or topology branch is involved.
        """

        if paths.ndim != 3 or paths.shape[-1] != self.model.latent_dim:
            raise ValueError("paths must have shape [candidates, steps, latent_dim]")
        if paths.shape[1] < self.durations[-1]:
            raise ValueError("imagined paths are shorter than the largest duration")
        candidate_count = paths.shape[0]
        current = current.reshape(1, -1).expand(candidate_count, -1)
        goal = goal.reshape(1, -1).expand(candidate_count, -1)
        current_mean, current_std = self.model.expected_and_std(current, goal)
        progress_columns: list[Tensor] = []
        lower_columns: list[Tensor] = []
        scale_columns: list[Tensor] = []
        for duration in self.durations:
            endpoint = paths[:, duration - 1]
            endpoint_mean, endpoint_std = self.model.expected_and_std(endpoint, goal)
            progress = current_mean - endpoint_mean
            if self.calibrator.scale_mode == "predicted":
                scale = torch.sqrt(current_std.square() + endpoint_std.square()).clamp_min(1e-6)
            else:
                scale = torch.ones_like(progress)
            quantile = torch.as_tensor(
                self.calibrator.quantile(duration), device=paths.device, dtype=paths.dtype
            )
            lower = progress - quantile * scale
            progress_columns.append(progress)
            lower_columns.append(lower)
            scale_columns.append(scale)
        predicted = torch.stack(progress_columns, dim=-1)
        lower = torch.stack(lower_columns, dim=-1)
        scales = torch.stack(scale_columns, dim=-1)
        duration_tensor = torch.as_tensor(self.durations, device=paths.device, dtype=paths.dtype)
        rates = lower / duration_tensor
        return rates, lower, predicted, scales

    @torch.inference_mode()
    def select(self, current: Tensor, goal: Tensor, paths: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Jointly select one candidate sequence and one execution duration."""

        _, lower, _, _ = self.score_paths(current, goal, paths)
        # Total certified progress is the shared decision objective. Dividing
        # by duration makes every smooth trajectory prefer its shortest prefix
        # and collapses temporal abstraction to duration 5. Horizon-dependent
        # conformal corrections already penalize unreliable long commitments.
        flat_index = lower.reshape(-1).argmax()
        candidate_index = torch.div(flat_index, len(self.durations), rounding_mode="floor")
        duration_index = flat_index % len(self.durations)
        duration = torch.as_tensor(
            self.durations[int(duration_index)], device=paths.device, dtype=torch.long
        )
        return candidate_index, duration, lower[candidate_index, duration_index]


@dataclass(frozen=True, slots=True)
class ReachabilityActionPlan:
    actions: Array
    predicted_path: Array
    duration: int
    certificate: float
    planning_cost: int
    selected_candidate: int
    rates: Array
    predicted_progress: Array
    predicted_scale: Array


class MultiHorizonReachabilityCEM:
    """One CEM search whose shared action prefixes define all temporal scales."""

    def __init__(
        self,
        world_model: WorldModelAdapter,
        scorer: CalibratedReachabilityScorer,
        *,
        action_block: int,
        samples: int = 300,
        elites: int = 30,
        iterations: int = 30,
        variance_scale: float = 1.0,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        if action_block <= 0:
            raise ValueError("action block must be positive")
        if any(duration % action_block for duration in scorer.durations):
            raise ValueError("every duration must be divisible by the action block")
        if not 1 < elites <= samples:
            raise ValueError("elites must lie in [2, samples]")
        self.world_model = world_model
        self.scorer = scorer
        self.action_block = int(action_block)
        self.samples = int(samples)
        self.elites = int(elites)
        self.iterations = int(iterations)
        self.variance_scale = float(variance_scale)
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))

    @torch.inference_mode()
    def plan(self, current: Array, goal: Array) -> ReachabilityActionPlan:
        max_duration = self.scorer.durations[-1]
        primitive_dim = int(np.prod(self.world_model.action_shape))
        mean = torch.zeros(max_duration, primitive_dim, device=self.device)
        std = torch.full_like(mean, self.variance_scale)
        current_tensor = torch.as_tensor(current, dtype=torch.float32, device=self.device)
        goal_tensor = torch.as_tensor(goal, dtype=torch.float32, device=self.device)
        last_rates: Tensor | None = None
        tensor_rollout = getattr(self.world_model, "batch_rollout_tensor", None)
        for _ in range(self.iterations):
            candidates = torch.randn(
                self.samples,
                max_duration,
                primitive_dim,
                device=self.device,
                generator=self.generator,
            )
            candidates = candidates * std.unsqueeze(0) + mean.unsqueeze(0)
            candidates[0] = mean
            if callable(tensor_rollout):
                paths = tensor_rollout(current_tensor, candidates)
            else:
                paths_np = self.world_model.batch_rollout(
                    np.asarray(current, dtype=np.float32),
                    candidates.detach().cpu().numpy().astype(np.float32),
                )
                paths = torch.as_tensor(paths_np, dtype=torch.float32, device=self.device)
            rates, lower, _, _ = self.scorer.score_paths(current_tensor, goal_tensor, paths)
            candidate_scores = lower.max(dim=-1).values
            elite_indices = torch.topk(candidate_scores, self.elites, largest=True).indices
            elite = candidates[elite_indices]
            mean = elite.mean(dim=0)
            std = elite.std(dim=0).clamp_min(1e-4)
            last_rates = rates

        actions = mean.detach().cpu().numpy().astype(np.float32)
        if callable(tensor_rollout):
            path_tensor = tensor_rollout(current_tensor, mean.unsqueeze(0))
            path = path_tensor[0].detach().cpu().numpy().astype(np.float32)
        else:
            path = self.world_model.rollout(np.asarray(current, dtype=np.float32), actions)
            path_tensor = torch.as_tensor(path[None], dtype=torch.float32, device=self.device)
        rates, lower, predicted, scales = self.scorer.score_paths(
            current_tensor, goal_tensor, path_tensor
        )
        duration_index = int(lower[0].argmax().item())
        duration = self.scorer.durations[duration_index]
        block_horizon = max_duration // self.action_block
        calls = self.samples * self.iterations * block_horizon + block_horizon
        del last_rates
        return ReachabilityActionPlan(
            actions=actions[:duration],
            predicted_path=np.asarray(path[:duration], dtype=np.float32),
            duration=duration,
            certificate=float(lower[0, duration_index].item()),
            planning_cost=int(calls),
            selected_candidate=0,
            rates=rates[0].detach().cpu().numpy().astype(np.float32),
            predicted_progress=predicted[0].detach().cpu().numpy().astype(np.float32),
            predicted_scale=scales[0].detach().cpu().numpy().astype(np.float32),
        )


class ReachabilityAdvantagePlanner:
    """Single-path MPC planner with no topology, trusted anchor, or fallback policy."""

    def __init__(
        self,
        world_model: WorldModelAdapter,
        controller: MultiHorizonReachabilityCEM,
    ) -> None:
        self.model = world_model
        self.controller = controller
        self.reset()

    def reset(self) -> None:
        self._goal_latent: Array | None = None
        self._active: ReachabilityActionPlan | None = None
        self._elapsed = 0
        self._plan_index = 0
        self._recorded_durations: set[int] = set()
        self._start_expected: float | None = None
        self.calibration_records: list[dict[str, float | int]] = []

    @torch.inference_mode()
    def _expected_steps(self, latent: Array, goal: Array) -> float:
        mean, _ = self.controller.scorer.model.expected_and_std(
            torch.as_tensor(
                latent, dtype=torch.float32, device=self.controller.device
            ).reshape(1, -1),
            torch.as_tensor(goal, dtype=torch.float32, device=self.controller.device).reshape(
                1, -1
            ),
        )
        return float(mean.item())

    def _record_executed_prefix(self, current: Array, goal: Array) -> None:
        if self._active is None or self._start_expected is None:
            return
        if self._elapsed not in self.controller.scorer.durations:
            return
        if self._elapsed > self._active.duration or self._elapsed in self._recorded_durations:
            return
        duration_index = self.controller.scorer.durations.index(self._elapsed)
        observed = self._start_expected - self._expected_steps(current, goal)
        self.calibration_records.append(
            {
                "plan_index": self._plan_index,
                "duration": self._elapsed,
                "predicted_progress": float(
                    self._active.predicted_progress[duration_index]
                ),
                "observed_progress": float(observed),
                "predicted_scale": float(self._active.predicted_scale[duration_index]),
            }
        )
        self._recorded_durations.add(self._elapsed)

    def plan(self, observation: Any, goal_image: Any) -> tuple[Array, PlanDiagnostics]:
        current = ensure_flat_latent(self.model.encode(observation))
        if self._goal_latent is None:
            self._goal_latent = ensure_flat_latent(self.model.encode(goal_image))
        goal = self._goal_latent
        self._record_executed_prefix(current, goal)
        new_plan = self._active is None or self._elapsed >= self._active.duration
        if new_plan:
            self._active = self.controller.plan(current, goal)
            self._elapsed = 0
            self._plan_index += 1
            self._recorded_durations.clear()
            self._start_expected = self._expected_steps(current, goal)
        assert self._active is not None
        action = np.asarray(self._active.actions[self._elapsed], dtype=np.float32)
        self._elapsed += 1
        expected_steps = self._expected_steps(current, goal)
        diagnostics = PlanDiagnostics(
            event=ExecutionEvent.NEW_PLAN if new_plan else ExecutionEvent.CONTINUE,
            trigger_event=None,
            chosen_duration=self._active.duration,
            active_step=self._elapsed,
            max_duration=self.controller.scorer.durations[-1],
            candidate_count=self.controller.samples if new_plan else 0,
            feasible_count=self.controller.samples if new_plan else 0,
            goal_cost=expected_steps,
            step_planning_cost=float(self._active.planning_cost if new_plan else 0.0),
            fallback_used=False,
            candidates=(
                [
                    {
                        "generator": "calibrated_reachability_advantage",
                        "duration": self._active.duration,
                        "certificate": self._active.certificate,
                        "rates": self._active.rates.tolist(),
                    }
                ]
                if new_plan
                else []
            ),
        )
        return action, diagnostics
