from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .interfaces import CandidateGenerator, RiskEstimator, WorldModelAdapter, ensure_flat_latent
from .types import Array, CandidatePlan, ExecutionEvent, PlanDiagnostics


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    durations: tuple[int, ...] = (5, 10, 20, 40)
    alpha: float = 0.1
    executability_threshold: float = 0.15
    goal_threshold: float = 0.05
    subgoal_threshold: float = 0.05
    lambda_compute: float = 0.01
    lambda_risk: float = 0.05
    compute_cost_scale: float = 1e-5
    min_progress: float = 1e-4
    minimum_candidate_progress: float = 0.0
    stall_patience: int = 2
    successes_to_expand: int = 2
    replan_low_level: bool = True
    adaptive_duration: bool = True
    risk_filter_enabled: bool = True
    endpoint_bound_filter_enabled: bool = True
    tube_trigger_enabled: bool = True
    stall_trigger_enabled: bool = True
    subgoal_trigger_enabled: bool = True
    remaining_risk_trigger_enabled: bool = True
    minimum_success_probability: float = 0.0
    anchor_advantage_margin: float = 0.0
    monitor_interval: int = 1

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.durations))) != self.durations:
            raise ValueError("durations must be positive, unique, and sorted")
        if not self.durations or self.durations[0] <= 0:
            raise ValueError("durations must be positive")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one")
        if not 0.0 <= self.minimum_success_probability <= 1.0:
            raise ValueError("minimum_success_probability must lie in [0, 1]")
        if self.anchor_advantage_margin < 0.0:
            raise ValueError("anchor_advantage_margin must be non-negative")
        if self.stall_patience <= 0 or self.successes_to_expand <= 0:
            raise ValueError("event patience values must be positive")
        if self.monitor_interval <= 0:
            raise ValueError("monitor_interval must be positive")


class CAPEPlanner:
    """Closed-loop temporal commitment with calibrated interruption and backoff."""

    FAILURE_EVENTS = {
        ExecutionEvent.TUBE_VIOLATION,
        ExecutionEvent.STALLED,
        ExecutionEvent.RISK_INCREASED,
    }
    SUCCESS_EVENTS = {
        ExecutionEvent.SUBGOAL_REACHED,
        ExecutionEvent.DURATION_EXHAUSTED,
    }

    def __init__(
        self,
        model: WorldModelAdapter,
        generator: CandidateGenerator,
        risk: RiskEstimator,
        config: PlannerConfig | None = None,
    ) -> None:
        self.model = model
        self.generator = generator
        self.risk = risk
        self.config = config or PlannerConfig()
        calibrated_alpha = getattr(risk, "alpha", None)
        if calibrated_alpha is not None and not np.isclose(calibrated_alpha, self.config.alpha):
            raise ValueError("planner alpha does not match the calibration artifact")
        self.reset()

    def reset(self) -> None:
        reset_generator = getattr(self.generator, "reset", None)
        if callable(reset_generator):
            reset_generator()
        self._active: CandidatePlan | None = None
        self._elapsed = 0
        self._max_duration_index = len(self.config.durations) - 1
        self._success_streak = 0
        self._stall_count = 0
        self._previous_goal_cost: float | None = None
        self._expected_next: Array | None = None
        self._expected_threshold: float | None = None
        self._commitment_start: Array | None = None
        self._goal_latent: Array | None = None
        self.last_current_latent: Array | None = None
        self.last_candidates: list[CandidatePlan] = []
        self.last_selected_index: int | None = None

    @property
    def max_duration(self) -> int:
        return self.config.durations[self._max_duration_index]

    def plan(self, observation: Any, goal_image: Any) -> tuple[Array, PlanDiagnostics]:
        """Return one primitive action and closed-loop planning diagnostics."""

        return self.act(observation, goal_image)

    def act(self, observation: Any, goal_image: Any) -> tuple[Array, PlanDiagnostics]:
        current = ensure_flat_latent(self.model.encode(observation))
        self.last_current_latent = current.copy()
        if self._goal_latent is None:
            self._goal_latent = ensure_flat_latent(self.model.encode(goal_image))
        goal = self._goal_latent
        current_cost = float(self.model.goal_cost(current, goal))

        if current_cost <= self.config.goal_threshold:
            self._active = None
            self._expected_next = None
            self._expected_threshold = None
            self._commitment_start = None
            action = np.zeros(self.model.action_shape, dtype=np.float32)
            return action, self._diagnostics(
                ExecutionEvent.GOAL_REACHED,
                current_cost,
                [],
                False,
                None,
            )

        trigger_event: ExecutionEvent | None = None
        step_planning_cost = 0.0
        residual: float | None = None
        threshold: float | None = None
        event = ExecutionEvent.NEW_PLAN if self._active is None else ExecutionEvent.CONTINUE
        monitored_boundary = (
            self._active is not None and self._elapsed % self.config.monitor_interval == 0
        )
        if monitored_boundary:
            monitored, residual, threshold = self._monitor_active(current, goal, current_cost)
            if monitored != ExecutionEvent.CONTINUE:
                trigger_event = monitored
                self._update_duration_policy(monitored)
                self._active = None
                self._elapsed = 0
                self._expected_next = None
                self._expected_threshold = None
                self._commitment_start = None

        candidates: list[CandidatePlan] = []
        fallback = False
        if self._active is None:
            candidates = self._score_candidates(current, goal)
            self.last_candidates = candidates
            self.last_selected_index = None
            step_planning_cost += sum(candidate.planning_cost for candidate in candidates)
            feasible = [candidate for candidate in candidates if candidate.feasible]
            if feasible:
                best = max(feasible, key=lambda candidate: candidate.score)
                topology = [
                    candidate
                    for candidate in feasible
                    if candidate.metadata.get("preferred_topology")
                ]
                if topology:
                    best = max(topology, key=lambda candidate: candidate.score)
                anchors = [
                    candidate for candidate in feasible if candidate.metadata.get("trusted_anchor")
                ]
                if anchors and not topology:
                    anchor = max(anchors, key=lambda candidate: candidate.score)
                    if best is anchor or (
                        best.score <= anchor.score + self.config.anchor_advantage_margin
                    ):
                        best = anchor
                self._active = best
                self.last_selected_index = next(
                    index for index, candidate in enumerate(candidates) if candidate is self._active
                )
                self._commitment_start = current.copy()
                event = ExecutionEvent.NEW_PLAN
            else:
                fallback = True
                fallback_candidate = self.generator.fallback(
                    current, goal, self.config.durations[0]
                )
                if fallback_candidate is None:
                    raise RuntimeError("candidate generator returned no plans")
                predicted_miss, upper, probability = self.risk.assess(
                    current, fallback_candidate.subgoal, fallback_candidate.duration
                )
                self._active = replace(
                    fallback_candidate,
                    predicted_miss=predicted_miss,
                    miss_upper_bound=upper,
                    success_probability=probability,
                    feasible=False,
                    score=float("-inf"),
                )
                self._commitment_start = current.copy()
                step_planning_cost += fallback_candidate.planning_cost
                event = ExecutionEvent.FALLBACK

        assert self._active is not None
        if len(self._active.actions) == 0:
            raise RuntimeError("selected candidate contains no primitive actions")
        remaining = max(1, self._active.duration - self._elapsed)
        if self.config.replan_low_level and self._elapsed > 0:
            previous_cost = self._active.planning_cost
            self._active = self.generator.refine(current, self._active, remaining)
            step_planning_cost += max(0.0, self._active.planning_cost - previous_cost)

        action_index = 0 if self.config.replan_low_level else self._elapsed
        action_index = min(action_index, len(self._active.actions) - 1)
        action = np.asarray(self._active.actions[action_index], dtype=np.float32)
        if self._active.predicted_path.size:
            path_index = min(action_index, len(self._active.predicted_path) - 1)
            self._expected_next = np.asarray(
                self._active.predicted_path[path_index], dtype=np.float32
            )
            tube_source = current if self.config.replan_low_level else self._commitment_start
            assert tube_source is not None
            tube_step = 1 if self.config.replan_low_level else self._elapsed + 1
            tube_duration = remaining if self.config.replan_low_level else self._active.duration
            if self._active.metadata.get("trusted_anchor") or self._active.metadata.get(
                "trusted_topology"
            ):
                self._expected_threshold = float("inf")
            else:
                self._expected_threshold = self.risk.tube_threshold(
                    tube_source,
                    self._active.subgoal,
                    tube_duration,
                    tube_step,
                )
        else:
            self._expected_next = None
            self._expected_threshold = None
        self._elapsed += 1
        if self._previous_goal_cost is None or monitored_boundary:
            self._previous_goal_cost = current_cost

        diagnostics = self._diagnostics(
            event,
            current_cost,
            candidates,
            fallback,
            trigger_event,
            step_planning_cost,
        )
        diagnostics.tube_residual = residual
        diagnostics.tube_threshold = threshold
        return action, diagnostics

    def _monitor_active(
        self, current: Array, goal: Array, current_cost: float
    ) -> tuple[ExecutionEvent, float | None, float | None]:
        if self._active is None:
            return ExecutionEvent.NEW_PLAN, None, None

        if self._active.metadata.get("trusted_anchor") or self._active.metadata.get(
            "trusted_topology"
        ):
            if self._elapsed >= self._active.duration:
                return ExecutionEvent.DURATION_EXHAUSTED, None, None
            return ExecutionEvent.CONTINUE, None, None

        residual: float | None = None
        threshold: float | None = None
        if self.config.tube_trigger_enabled and self._expected_next is not None:
            residual = float(np.linalg.norm(current - self._expected_next))
            threshold = self._expected_threshold
            if threshold is None:
                raise RuntimeError("expected latent exists without a precommitted tube threshold")
            if residual > threshold:
                return ExecutionEvent.TUBE_VIOLATION, residual, threshold

        subgoal_cost = self.model.goal_cost(current, self._active.subgoal)
        if self.config.subgoal_trigger_enabled and subgoal_cost <= self.config.subgoal_threshold:
            return ExecutionEvent.SUBGOAL_REACHED, residual, threshold
        if self._elapsed >= self._active.duration:
            return ExecutionEvent.DURATION_EXHAUSTED, residual, threshold

        if self.config.stall_trigger_enabled and self._previous_goal_cost is not None:
            progress = self._previous_goal_cost - current_cost
            self._stall_count = self._stall_count + 1 if progress < self.config.min_progress else 0
            if self._stall_count >= self.config.stall_patience:
                return ExecutionEvent.STALLED, residual, threshold

        if self.config.remaining_risk_trigger_enabled:
            remaining = max(1, self._active.duration - self._elapsed)
            _, upper, probability = self.risk.assess(current, self._active.subgoal, remaining)
            minimum_probability = getattr(
                self.risk, "minimum_success_probability", lambda _, default: default
            )(remaining, self.config.minimum_success_probability)
            if (
                self.config.endpoint_bound_filter_enabled
                and upper > self.config.executability_threshold
            ) or probability < minimum_probability:
                return ExecutionEvent.RISK_INCREASED, residual, threshold
        return ExecutionEvent.CONTINUE, residual, threshold

    def _score_candidates(self, current: Array, goal: Array) -> list[CandidatePlan]:
        proposed = self.generator.propose(
            current,
            goal,
            self.config.durations,
            self.max_duration,
        )
        scored: list[CandidatePlan] = []
        for candidate in proposed:
            trusted = bool(
                candidate.metadata.get("trusted_anchor")
                or candidate.metadata.get("trusted_topology")
            )
            if trusted:
                predicted_miss, upper, probability = 0.0, 0.0, 1.0
            else:
                predicted_miss, upper, probability = self.risk.assess(
                    current, candidate.subgoal, candidate.duration
                )
            minimum_probability = getattr(
                self.risk, "minimum_success_probability", lambda _, default: default
            )(candidate.duration, self.config.minimum_success_probability)
            risk_feasible = (
                trusted
                or not self.config.risk_filter_enabled
                or (
                    (
                        not self.config.endpoint_bound_filter_enabled
                        or upper <= self.config.executability_threshold
                    )
                    and probability >= minimum_probability
                )
            )
            feasible = risk_feasible and (
                trusted or candidate.progress > self.config.minimum_candidate_progress
            )
            score = (
                candidate.progress / candidate.duration
                - self.config.lambda_compute
                * candidate.planning_cost
                * self.config.compute_cost_scale
                - self.config.lambda_risk * upper
            )
            scored.append(
                replace(
                    candidate,
                    predicted_miss=predicted_miss,
                    miss_upper_bound=upper,
                    success_probability=probability,
                    feasible=feasible,
                    score=float(score),
                )
            )
        return scored

    def _update_duration_policy(self, event: ExecutionEvent) -> None:
        self._stall_count = 0
        if not self.config.adaptive_duration:
            return
        if event in self.FAILURE_EVENTS:
            self._max_duration_index = max(0, self._max_duration_index - 1)
            self._success_streak = 0
        elif event in self.SUCCESS_EVENTS:
            self._success_streak += 1
            if self._success_streak >= self.config.successes_to_expand:
                self._max_duration_index = min(
                    len(self.config.durations) - 1, self._max_duration_index + 1
                )
                self._success_streak = 0

    def _diagnostics(
        self,
        event: ExecutionEvent,
        goal_cost: float,
        candidates: list[CandidatePlan],
        fallback: bool,
        trigger_event: ExecutionEvent | None,
        step_planning_cost: float = 0.0,
    ) -> PlanDiagnostics:
        compact_candidates = [
            {
                "duration": c.duration,
                "progress": c.progress,
                "miss_upper_bound": c.miss_upper_bound,
                "success_probability": c.success_probability,
                "planning_cost": c.planning_cost,
                "feasible": c.feasible,
                "score": c.score,
                **c.metadata,
            }
            for c in candidates
        ]
        return PlanDiagnostics(
            event=event,
            trigger_event=trigger_event,
            chosen_duration=self._active.duration if self._active is not None else None,
            active_step=self._elapsed,
            max_duration=self.max_duration,
            candidate_count=len(candidates),
            feasible_count=sum(candidate.feasible for candidate in candidates),
            goal_cost=goal_cost,
            step_planning_cost=step_planning_cost,
            fallback_used=fallback,
            candidates=compact_candidates,
        )
