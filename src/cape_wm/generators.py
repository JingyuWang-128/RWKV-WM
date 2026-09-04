from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

import numpy as np
import torch

from .cem import CEMConfig, cem_optimize
from .interfaces import CandidateGenerator, WorldModelAdapter
from .types import Array, CandidatePlan


class MacroDynamics(Protocol):
    @property
    def macro_dim(self) -> int: ...

    def predict(self, latent: Array, macro_action: Array, duration: int) -> Array: ...


def _batched_rollout_cost(
    model: WorldModelAdapter,
    current: Array,
    goal: Array,
    action_population: Array,
    path_cost_weight: float = 0.1,
    action_cost_weight: float = 1e-4,
) -> Array:
    batch_method = getattr(model, "batch_rollout", None)
    if callable(batch_method):
        paths = np.asarray(batch_method(current, action_population), dtype=np.float32)
        batch_cost = getattr(model, "batch_goal_cost", None)
        if callable(batch_cost):
            all_costs = np.asarray(
                batch_cost(paths.reshape(-1, paths.shape[-1]), goal),
                dtype=np.float64,
            ).reshape(paths.shape[:2])
            terminal = all_costs[:, -1]
            stage = all_costs.mean(axis=1)
        else:
            terminal = np.asarray(
                [model.goal_cost(path[-1], goal) for path in paths],
                dtype=np.float64,
            )
            stage = np.asarray(
                [np.mean([model.goal_cost(state, goal) for state in path]) for path in paths],
                dtype=np.float64,
            )
        effort = np.mean(np.square(action_population), axis=tuple(range(1, action_population.ndim)))
        return terminal + path_cost_weight * stage + action_cost_weight * effort
    scores = []
    for actions in action_population:
        path = model.rollout(current, actions)
        terminal = model.goal_cost(path[-1], goal)
        stage = np.mean([model.goal_cost(state, goal) for state in path])
        effort = np.mean(np.square(actions))
        scores.append(terminal + path_cost_weight * stage + action_cost_weight * effort)
    return np.asarray(scores, dtype=np.float64)


class CEMLowLevelController:
    """Receding-horizon continuous controller over a frozen latent world model."""

    def __init__(
        self,
        model: WorldModelAdapter,
        horizon: int = 10,
        cem: CEMConfig | None = None,
        path_cost_weight: float = 0.1,
        action_cost_weight: float = 1e-4,
    ) -> None:
        self.model = model
        self.horizon = horizon
        self.cem = cem or CEMConfig()
        self.path_cost_weight = path_cost_weight
        self.action_cost_weight = action_cost_weight

    def plan(
        self,
        current: Array,
        goal: Array,
        horizon: int | None = None,
        transition_budget: int | None = None,
    ) -> tuple[Array, Array, int]:
        planning_horizon = max(1, min(horizon or self.horizon, self.horizon))
        shape = (planning_horizon, *self.model.action_shape)
        cem_config = resize_cem_for_transition_budget(self.cem, planning_horizon, transition_budget)
        result = cem_optimize(
            lambda population: _batched_rollout_cost(
                self.model,
                current,
                goal,
                population,
                self.path_cost_weight,
                self.action_cost_weight,
            ),
            shape=shape,
            low=self.model.action_low,
            high=self.model.action_high,
            config=cem_config,
        )
        path = self.model.rollout(current, result.value)
        model_calls = result.evaluations * planning_horizon + planning_horizon
        return result.value, path, model_calls


class OfficialCEMLowLevelController:
    """Flat+TRM-compatible CEM over LeWM action blocks.

    This intentionally mirrors stable-worldmodel's continuous CEM update:
    unit initial variance, an explicit mean candidate, unbiased elite standard
    deviation, no clipping or momentum, and returning the final mean.  A
    shared torch generator preserves the upstream batch-size-one RNG stream
    across vectorized environments.
    """

    def __init__(
        self,
        model: WorldModelAdapter,
        *,
        horizon: int,
        action_block: int,
        samples: int = 300,
        elites: int = 30,
        iterations: int = 30,
        variance_scale: float = 1.0,
        generator: torch.Generator,
        fixed_horizon: bool = True,
    ) -> None:
        if horizon <= 0 or horizon % action_block:
            raise ValueError("horizon must be a positive multiple of action_block")
        if not 1 < elites <= samples:
            raise ValueError("elites must lie in [2, samples]")
        self.model = model
        self.horizon = int(horizon)
        self.action_block = int(action_block)
        self.samples = int(samples)
        self.elites = int(elites)
        self.iterations = int(iterations)
        self.variance_scale = float(variance_scale)
        self.generator = generator
        self.fixed_horizon = bool(fixed_horizon)

    @torch.inference_mode()
    def plan(
        self,
        current: Array,
        goal: Array,
        horizon: int | None = None,
        transition_budget: int | None = None,
    ) -> tuple[Array, Array, int]:
        del transition_budget
        planning_horizon = int(horizon or self.horizon)
        if self.fixed_horizon and planning_horizon != self.horizon:
            raise ValueError("official-compatible controller uses one fixed horizon")
        if (
            planning_horizon <= 0
            or planning_horizon > self.horizon
            or planning_horizon % self.action_block
        ):
            raise ValueError(
                "planning horizon must be a positive action-block multiple "
                "not exceeding the configured horizon"
            )
        primitive_dim = int(np.prod(self.model.action_shape))
        block_horizon = planning_horizon // self.action_block
        block_dim = primitive_dim * self.action_block
        device = torch.device(getattr(self.model, "device", "cpu"))
        mean = torch.zeros(block_horizon, block_dim, device=device)
        std = torch.full_like(mean, self.variance_scale)
        for _ in range(self.iterations):
            candidates = torch.randn(
                self.samples,
                block_horizon,
                block_dim,
                generator=self.generator,
                device=device,
            )
            candidates = candidates * std[None] + mean[None]
            candidates[0] = mean
            primitive = candidates.reshape(self.samples, planning_horizon, primitive_dim)
            paths = self.model.batch_rollout(current, primitive)
            scores = self.model.batch_goal_cost(paths[:, -1], goal)
            score_tensor = torch.as_tensor(scores, dtype=mean.dtype, device=device)
            elite_indices = torch.topk(score_tensor, self.elites, largest=False).indices
            elite = candidates[elite_indices]
            mean = elite.mean(dim=0)
            std = elite.std(dim=0)
        actions = mean.reshape(planning_horizon, primitive_dim).cpu().numpy().astype(np.float32)
        path = self.model.rollout(current, actions)
        model_transitions = self.samples * self.iterations * block_horizon + block_horizon
        return actions, path, model_transitions


class DirectRolloutGenerator(CandidateGenerator):
    """A useful flat baseline that exposes multiple commitment lengths."""

    def __init__(
        self,
        model: WorldModelAdapter,
        low_level: CEMLowLevelController,
        model_transition_budget: int | None = None,
        fallback_budget_fraction: float = 0.1,
    ) -> None:
        self.model = model
        self.low_level = low_level
        self.model_transition_budget = model_transition_budget
        if not 0.0 <= fallback_budget_fraction < 1.0:
            raise ValueError("fallback_budget_fraction must lie in [0, 1)")
        self.fallback_budget_fraction = fallback_budget_fraction

    def propose(
        self,
        current_latent: Array,
        goal_latent: Array,
        durations: tuple[int, ...],
        max_duration: int,
    ) -> list[CandidatePlan]:
        candidates: list[CandidatePlan] = []
        current_cost = self.model.goal_cost(current_latent, goal_latent)
        allowed = [duration for duration in durations if duration <= max_duration]
        candidate_budget = (
            int(self.model_transition_budget * (1.0 - self.fallback_budget_fraction))
            if self.model_transition_budget is not None
            else None
        )
        per_scale_budget = (
            candidate_budget // len(allowed) if candidate_budget is not None and allowed else None
        )
        for duration in allowed:
            actions, path, calls = self.low_level.plan(
                current_latent,
                goal_latent,
                horizon=duration,
                transition_budget=per_scale_budget,
            )
            subgoal = path[-1]
            candidates.append(
                CandidatePlan(
                    duration=duration,
                    subgoal=subgoal,
                    actions=actions,
                    predicted_path=path,
                    current_goal_cost=current_cost,
                    subgoal_goal_cost=self.model.goal_cost(subgoal, goal_latent),
                    predicted_miss=0.0,
                    miss_upper_bound=float("inf"),
                    success_probability=0.0,
                    planning_cost=float(calls),
                    metadata={"generator": "direct_rollout"},
                )
            )
        if (
            candidate_budget is not None
            and sum(candidate.planning_cost for candidate in candidates) > candidate_budget
        ):
            raise RuntimeError("direct candidate generation exceeded transition budget")
        return candidates

    def fallback(self, current_latent: Array, goal_latent: Array, duration: int) -> CandidatePlan:
        budget = (
            self.model_transition_budget
            - int(self.model_transition_budget * (1.0 - self.fallback_budget_fraction))
            if self.model_transition_budget is not None
            else None
        )
        actions, path, calls = self.low_level.plan(
            current_latent, goal_latent, horizon=duration, transition_budget=budget
        )
        current_cost = self.model.goal_cost(current_latent, goal_latent)
        subgoal = path[-1]
        return CandidatePlan(
            duration=duration,
            subgoal=subgoal,
            actions=actions,
            predicted_path=path,
            current_goal_cost=current_cost,
            subgoal_goal_cost=self.model.goal_cost(subgoal, goal_latent),
            predicted_miss=0.0,
            miss_upper_bound=float("inf"),
            success_probability=0.0,
            planning_cost=float(calls),
            metadata={"generator": "direct_low_level_fallback"},
        )

    def refine(
        self,
        current_latent: Array,
        candidate: CandidatePlan,
        remaining_duration: int,
    ) -> CandidatePlan:
        actions, path, calls = self.low_level.plan(
            current_latent, candidate.subgoal, horizon=remaining_duration
        )
        return replace(
            candidate,
            actions=actions,
            predicted_path=path,
            planning_cost=candidate.planning_cost + calls,
        )


class MacroCEMGenerator(CandidateGenerator):
    """Matched-physical-horizon CEM over learned macro actions.

    The high-level model only proposes a latent subgoal. Primitive actions are
    always regenerated by the frozen low-level world model controller.
    """

    def __init__(
        self,
        model: WorldModelAdapter,
        macro_dynamics: MacroDynamics,
        low_level: CEMLowLevelController,
        physical_horizon: int = 80,
        macro_cem: CEMConfig | None = None,
        macro_low: float = -3.0,
        macro_high: float = 3.0,
        model_transition_budget: int | None = None,
        high_level_budget_fraction: float = 0.6,
        fallback_budget_fraction: float = 0.1,
        empirical_banks: dict[int, Array] | None = None,
        empirical_residual_scale: float = 0.25,
        empirical_discrete: bool = False,
    ) -> None:
        self.model = model
        self.macro_dynamics = macro_dynamics
        self.low_level = low_level
        self.physical_horizon = physical_horizon
        self.macro_cem = macro_cem or CEMConfig(samples=128, elites=16, iterations=4)
        self.macro_low = macro_low
        self.macro_high = macro_high
        self.model_transition_budget = model_transition_budget
        if not 0.0 < high_level_budget_fraction < 1.0:
            raise ValueError("high_level_budget_fraction must lie in (0, 1)")
        self.high_level_budget_fraction = high_level_budget_fraction
        if not 0.0 <= fallback_budget_fraction < 1.0:
            raise ValueError("fallback_budget_fraction must lie in [0, 1)")
        self.fallback_budget_fraction = fallback_budget_fraction
        self.empirical_banks = empirical_banks
        self.empirical_residual_scale = float(empirical_residual_scale)
        self.empirical_discrete = bool(empirical_discrete)
        if self.empirical_discrete and self.empirical_banks is None:
            raise ValueError("discrete empirical search requires empirical macro banks")

    def _macro_rollout(self, current: Array, macros: Array, duration: int) -> Array:
        latent = np.asarray(current, dtype=np.float32)
        predictions = []
        for macro in macros:
            latent = np.asarray(
                self.macro_dynamics.predict(latent, macro, duration), dtype=np.float32
            )
            predictions.append(latent)
        return np.stack(predictions)

    def _macro_rollout_population(
        self, current: Array, macro_population: Array, duration: int
    ) -> Array:
        batch = len(macro_population)
        latent = np.broadcast_to(
            np.asarray(current, dtype=np.float32), (batch, np.asarray(current).size)
        ).copy()
        predictions = []
        batch_predict = getattr(self.macro_dynamics, "batch_predict", None)
        for step in range(macro_population.shape[1]):
            if callable(batch_predict):
                latent = np.asarray(
                    batch_predict(latent, macro_population[:, step], duration),
                    dtype=np.float32,
                )
            else:
                latent = np.stack(
                    [
                        self.macro_dynamics.predict(state, macro, duration)
                        for state, macro in zip(latent, macro_population[:, step], strict=True)
                    ]
                ).astype(np.float32)
            predictions.append(latent.copy())
        return np.stack(predictions, axis=1)

    def propose(
        self,
        current_latent: Array,
        goal_latent: Array,
        durations: tuple[int, ...],
        max_duration: int,
    ) -> list[CandidatePlan]:
        candidates: list[CandidatePlan] = []
        current_cost = self.model.goal_cost(current_latent, goal_latent)
        allowed = [duration for duration in durations if duration <= max_duration]
        candidate_budget = (
            int(self.model_transition_budget * (1.0 - self.fallback_budget_fraction))
            if self.model_transition_budget is not None
            else None
        )
        per_scale_budget = (
            candidate_budget // len(allowed) if candidate_budget is not None and allowed else None
        )
        for duration in allowed:
            macro_steps = max(1, int(np.ceil(self.physical_horizon / duration)))
            high_budget = (
                int(per_scale_budget * self.high_level_budget_fraction)
                if per_scale_budget is not None
                else None
            )
            low_budget = (
                per_scale_budget - high_budget
                if per_scale_budget is not None and high_budget is not None
                else None
            )

            def objective(population: Array, scale: int = duration) -> Array:
                terminals = self._macro_rollout_population(current_latent, population, scale)[:, -1]
                batch_cost = getattr(self.model, "batch_goal_cost", None)
                if callable(batch_cost):
                    return np.asarray(batch_cost(terminals, goal_latent), dtype=np.float64)
                return np.asarray(
                    [self.model.goal_cost(terminal, goal_latent) for terminal in terminals],
                    dtype=np.float64,
                )

            macro_config = resize_cem_for_transition_budget(
                replace_seed(self.macro_cem, self.macro_cem.seed + duration),
                macro_steps,
                high_budget,
            )
            initial_population = None
            if self.empirical_banks is not None:
                try:
                    bank = np.asarray(self.empirical_banks[duration], dtype=np.float32)
                except KeyError as error:
                    raise ValueError(f"empirical macro bank has no duration {duration}") from error
                rng = np.random.default_rng(macro_config.seed)
                if self.empirical_discrete:
                    macro_config = CEMConfig(
                        samples=macro_config.samples * macro_config.iterations,
                        elites=min(
                            macro_config.elites,
                            macro_config.samples * macro_config.iterations,
                        ),
                        iterations=1,
                        momentum=macro_config.momentum,
                        min_std=macro_config.min_std,
                        seed=macro_config.seed,
                        clip=False,
                    )
                anchors = bank[
                    rng.integers(
                        0,
                        len(bank),
                        size=(macro_config.samples, macro_steps),
                    )
                ]
                if self.empirical_discrete:
                    initial_population = anchors
                else:
                    initial_population = (
                        anchors
                        + self.empirical_residual_scale
                        * rng.standard_normal(anchors.shape, dtype=np.float32)
                    )
            result = cem_optimize(
                objective,
                shape=(macro_steps, self.macro_dynamics.macro_dim),
                low=self.macro_low,
                high=self.macro_high,
                config=macro_config,
                initial_population=initial_population,
            )
            macro_path = self._macro_rollout(current_latent, result.value, duration)
            subgoal = macro_path[0]
            actions, low_path, low_calls = self.low_level.plan(
                current_latent,
                subgoal,
                horizon=duration,
                transition_budget=low_budget,
            )
            high_calls = result.evaluations * macro_steps + macro_steps
            candidates.append(
                CandidatePlan(
                    duration=duration,
                    subgoal=subgoal,
                    actions=actions,
                    predicted_path=low_path,
                    current_goal_cost=current_cost,
                    subgoal_goal_cost=self.model.goal_cost(subgoal, goal_latent),
                    predicted_miss=0.0,
                    miss_upper_bound=float("inf"),
                    success_probability=0.0,
                    planning_cost=float(high_calls + low_calls),
                    metadata={
                        "generator": "macro_cem",
                        "macro_objective": result.objective,
                        "macro_steps": macro_steps,
                        "allocated_transition_budget": per_scale_budget,
                    },
                )
            )
        if (
            candidate_budget is not None
            and sum(candidate.planning_cost for candidate in candidates) > candidate_budget
        ):
            raise RuntimeError("macro candidate generation exceeded transition budget")
        return candidates

    def refine(
        self,
        current_latent: Array,
        candidate: CandidatePlan,
        remaining_duration: int,
    ) -> CandidatePlan:
        actions, path, calls = self.low_level.plan(
            current_latent, candidate.subgoal, horizon=remaining_duration
        )
        return replace(
            candidate,
            actions=actions,
            predicted_path=path,
            planning_cost=candidate.planning_cost + calls,
        )

    def fallback(
        self,
        current_latent: Array,
        goal_latent: Array,
        duration: int,
    ) -> CandidatePlan:
        budget = (
            self.model_transition_budget
            - int(self.model_transition_budget * (1.0 - self.fallback_budget_fraction))
            if self.model_transition_budget is not None
            else None
        )
        actions, path, calls = self.low_level.plan(
            current_latent,
            goal_latent,
            horizon=duration,
            transition_budget=budget,
        )
        current_cost = self.model.goal_cost(current_latent, goal_latent)
        subgoal = path[-1]
        return CandidatePlan(
            duration=duration,
            subgoal=subgoal,
            actions=actions,
            predicted_path=path,
            current_goal_cost=current_cost,
            subgoal_goal_cost=self.model.goal_cost(subgoal, goal_latent),
            predicted_miss=0.0,
            miss_upper_bound=float("inf"),
            success_probability=0.0,
            planning_cost=float(calls),
            metadata={"generator": "low_level_fallback"},
        )


class AnchoredMacroGenerator(CandidateGenerator):
    """Add an exact Flat+TRM candidate to learned macro proposals."""

    def __init__(
        self,
        macro_generator: MacroCEMGenerator,
        anchor_controller: OfficialCEMLowLevelController,
        anchor_duration: int,
    ) -> None:
        self.macro_generator = macro_generator
        self.anchor_controller = anchor_controller
        self.anchor_duration = int(anchor_duration)

    def propose(
        self,
        current_latent: Array,
        goal_latent: Array,
        durations: tuple[int, ...],
        max_duration: int,
    ) -> list[CandidatePlan]:
        candidates = self.macro_generator.propose(
            current_latent, goal_latent, durations, max_duration
        )
        actions, path, calls = self.anchor_controller.plan(
            current_latent, goal_latent, horizon=self.anchor_duration
        )
        current_cost = self.macro_generator.model.goal_cost(current_latent, goal_latent)
        subgoal = path[-1]
        candidates.append(
            CandidatePlan(
                duration=self.anchor_duration,
                subgoal=subgoal,
                actions=actions,
                predicted_path=path,
                current_goal_cost=current_cost,
                subgoal_goal_cost=self.macro_generator.model.goal_cost(subgoal, goal_latent),
                predicted_miss=0.0,
                miss_upper_bound=0.0,
                success_probability=1.0,
                planning_cost=float(calls),
                metadata={
                    "generator": "flat_trm_anchor",
                    "trusted_anchor": True,
                },
            )
        )
        return candidates

    def fallback(
        self,
        current_latent: Array,
        goal_latent: Array,
        duration: int,
    ) -> CandidatePlan:
        return self.macro_generator.fallback(current_latent, goal_latent, duration)

    def refine(
        self,
        current_latent: Array,
        candidate: CandidatePlan,
        remaining_duration: int,
    ) -> CandidatePlan:
        if candidate.metadata.get("trusted_anchor"):
            return candidate
        return self.macro_generator.refine(current_latent, candidate, remaining_duration)


class TrustedAnchorGenerator(CandidateGenerator):
    """Expose only the official Flat+TRM plan as a trusted candidate."""

    def __init__(
        self,
        model: WorldModelAdapter,
        controller: OfficialCEMLowLevelController,
        duration: int,
    ) -> None:
        self.model = model
        self.controller = controller
        self.duration = int(duration)

    def propose(
        self,
        current_latent: Array,
        goal_latent: Array,
        durations: tuple[int, ...],
        max_duration: int,
    ) -> list[CandidatePlan]:
        del durations, max_duration
        actions, path, calls = self.controller.plan(
            current_latent, goal_latent, horizon=self.duration
        )
        subgoal = path[-1]
        return [
            CandidatePlan(
                duration=self.duration,
                subgoal=subgoal,
                actions=actions,
                predicted_path=path,
                current_goal_cost=self.model.goal_cost(current_latent, goal_latent),
                subgoal_goal_cost=self.model.goal_cost(subgoal, goal_latent),
                predicted_miss=0.0,
                miss_upper_bound=0.0,
                success_probability=1.0,
                planning_cost=float(calls),
                metadata={
                    "generator": "flat_trm_anchor",
                    "trusted_anchor": True,
                },
            )
        ]

    def fallback(
        self,
        current_latent: Array,
        goal_latent: Array,
        duration: int,
    ) -> CandidatePlan:
        del duration
        return self.propose(current_latent, goal_latent, (), self.duration)[0]

    def refine(
        self,
        current_latent: Array,
        candidate: CandidatePlan,
        remaining_duration: int,
    ) -> CandidatePlan:
        del current_latent, remaining_duration
        return candidate


class TopologyWaypointGenerator(CandidateGenerator):
    """Add a train-derived door-crossing waypoint when goal rooms differ."""

    def __init__(
        self,
        base_generator: CandidateGenerator,
        model: WorldModelAdapter,
        controller: OfficialCEMLowLevelController | None,
        *,
        duration: int,
        latent_mean: Array,
        latent_scale: Array,
        position_weight: Array,
        position_mean: Array,
        wall_position: float,
        door_position: float,
        side_offset: float,
        door_left_latent: Array,
        door_right_latent: Array,
        direct_action_transform: Callable[[Array], Array] | None = None,
        direct_goal_completion: bool = False,
    ) -> None:
        self.base_generator = base_generator
        self.model = model
        self.controller = controller
        self.duration = int(duration)
        self.latent_mean = np.asarray(latent_mean, dtype=np.float32)
        self.latent_scale = np.asarray(latent_scale, dtype=np.float32)
        self.position_weight = np.asarray(position_weight, dtype=np.float32)
        self.position_mean = np.asarray(position_mean, dtype=np.float32)
        self.wall_position = float(wall_position)
        self.door_position = float(door_position)
        self.side_offset = float(side_offset)
        self.door_left_latent = np.asarray(door_left_latent, dtype=np.float32)
        self.door_right_latent = np.asarray(door_right_latent, dtype=np.float32)
        self.direct_action_transform = direct_action_transform
        self.direct_goal_completion = bool(direct_goal_completion)
        self._route_active = False
        self._route_goal_left = False
        self._route_source_left = False
        self._wall_goal_route = False
        self._clearance_emitted = False
        if self.controller is None and self.direct_action_transform is None:
            raise ValueError("topology waypoint requires CEM or direct control")

    def reset(self) -> None:
        self._route_active = False
        self._route_goal_left = False
        self._route_source_left = False
        self._wall_goal_route = False
        self._clearance_emitted = False

    def decode_position(self, latent: Array) -> Array:
        standardized = (np.asarray(latent, dtype=np.float32) - self.latent_mean) / self.latent_scale
        return standardized @ self.position_weight + self.position_mean

    def propose(
        self,
        current_latent: Array,
        goal_latent: Array,
        durations: tuple[int, ...],
        max_duration: int,
    ) -> list[CandidatePlan]:
        current_position = self.decode_position(current_latent)
        goal_position = self.decode_position(goal_latent)
        current_left = bool(current_position[0] < self.wall_position)
        goal_left = bool(goal_position[0] < self.wall_position)
        goal_in_door_band = bool(
            abs(float(goal_position[0]) - self.wall_position) <= self.side_offset
            and abs(float(goal_position[1]) - self.door_position) <= self.side_offset
        )
        if self.direct_goal_completion and not self._route_active:
            if current_left != goal_left or goal_in_door_band:
                self._route_active = True
                self._route_goal_left = goal_left
                self._route_source_left = current_left
                self._wall_goal_route = current_left == goal_left
                self._clearance_emitted = False

        if not self.direct_goal_completion and current_left == goal_left:
            return self.base_generator.propose(current_latent, goal_latent, durations, max_duration)
        if self.direct_goal_completion and not self._route_active:
            return self.base_generator.propose(current_latent, goal_latent, durations, max_duration)

        route_to_goal = False
        route_from_left = current_left
        if self.direct_goal_completion:
            route_from_left = self._route_source_left
            if self._wall_goal_route:
                approach_position = np.asarray(
                    [
                        self.wall_position
                        + (-self.side_offset if route_from_left else self.side_offset),
                        self.door_position,
                    ],
                    dtype=np.float32,
                )
                route_to_goal = bool(
                    np.linalg.norm(current_position - approach_position) <= self.side_offset
                )
            elif current_left == self._route_goal_left:
                if self._clearance_emitted:
                    route_to_goal = True
                else:
                    self._clearance_emitted = True

        subgoal = (
            goal_latent
            if route_to_goal
            else (self.door_right_latent if route_from_left else self.door_left_latent)
        )
        if self.direct_action_transform is None:
            if route_to_goal:
                return self.base_generator.propose(
                    current_latent, goal_latent, durations, max_duration
                )
            assert self.controller is not None
            actions, path, calls = self.controller.plan(
                current_latent, subgoal, horizon=self.duration
            )
            control_mode = "cem"
        else:
            target_position = (
                goal_position
                if route_to_goal
                else np.asarray(
                    [
                        self.wall_position
                        + (self.side_offset if route_from_left else -self.side_offset),
                        self.door_position,
                    ],
                    dtype=np.float32,
                )
            )
            direction = target_position - current_position
            norm = float(np.linalg.norm(direction))
            if norm > 1e-6:
                direction /= norm
            normalized_action = np.asarray(
                self.direct_action_transform(direction[None])[0], dtype=np.float32
            )
            actions = np.repeat(normalized_action[None], self.duration, axis=0)
            path = np.empty((0, np.asarray(current_latent).size), dtype=np.float32)
            calls = 0
            control_mode = "direct_probe"
        # A preferred topology candidate always wins while the decoded states
        # are in different rooms.  Do not eagerly run the expensive Flat+TRM
        # anchor only to discard it; generate that anchor after crossing.
        return [
            CandidatePlan(
                duration=self.duration,
                subgoal=subgoal,
                actions=actions,
                predicted_path=path,
                current_goal_cost=self.model.goal_cost(current_latent, goal_latent),
                subgoal_goal_cost=self.model.goal_cost(subgoal, goal_latent),
                predicted_miss=0.0,
                miss_upper_bound=0.0,
                success_probability=1.0,
                planning_cost=float(calls),
                metadata={
                    "generator": (
                        "topology_goal_completion" if route_to_goal else "topology_waypoint"
                    ),
                    "trusted_topology": True,
                    "preferred_topology": True,
                    "topology_control_mode": control_mode,
                    "decoded_current_position": current_position.tolist(),
                    "decoded_goal_position": goal_position.tolist(),
                },
            )
        ]

    def fallback(
        self,
        current_latent: Array,
        goal_latent: Array,
        duration: int,
    ) -> CandidatePlan:
        return self.base_generator.fallback(current_latent, goal_latent, duration)

    def refine(
        self,
        current_latent: Array,
        candidate: CandidatePlan,
        remaining_duration: int,
    ) -> CandidatePlan:
        if candidate.metadata.get("trusted_topology"):
            return candidate
        return self.base_generator.refine(current_latent, candidate, remaining_duration)


def replace_seed(config: CEMConfig, seed: int) -> CEMConfig:
    return CEMConfig(
        samples=config.samples,
        elites=config.elites,
        iterations=config.iterations,
        momentum=config.momentum,
        min_std=config.min_std,
        seed=seed,
        clip=config.clip,
    )


def resize_cem_for_transition_budget(
    config: CEMConfig,
    transition_horizon: int,
    transition_budget: int | None,
) -> CEMConfig:
    """Reduce a CEM population so predicted latent transitions fit a hard budget."""

    if transition_budget is None:
        return config
    minimum = (config.iterations + 1) * transition_horizon
    if transition_budget < minimum:
        raise ValueError(
            f"transition budget {transition_budget} is below the CEM minimum {minimum}"
        )
    samples = max(
        1,
        min(
            config.samples,
            (transition_budget - transition_horizon) // (config.iterations * transition_horizon),
        ),
    )
    return CEMConfig(
        samples=samples,
        elites=min(config.elites, samples),
        iterations=config.iterations,
        momentum=config.momentum,
        min_std=config.min_std,
        seed=config.seed,
        clip=config.clip,
    )
