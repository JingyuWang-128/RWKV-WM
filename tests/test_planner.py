from __future__ import annotations

from dataclasses import replace

import numpy as np

from cape_wm.interfaces import CandidateGenerator, WorldModelAdapter
from cape_wm.planner import CAPEPlanner, PlannerConfig
from cape_wm.risk import ConstantRiskEstimator
from cape_wm.types import CandidatePlan, ExecutionEvent


class IdentityWorldModel(WorldModelAdapter):
    def encode(self, observation):
        return np.asarray(observation, dtype=np.float32)

    def rollout(self, latent, actions):
        return np.asarray(latent)[None] + np.cumsum(actions, axis=0)

    def goal_cost(self, latent, goal_latent):
        return float(np.linalg.norm(np.asarray(latent) - np.asarray(goal_latent)))

    @property
    def action_shape(self):
        return (1,)


class FixedGenerator(CandidateGenerator):
    def __init__(self):
        self.model = IdentityWorldModel()

    def propose(self, current_latent, goal_latent, durations, max_duration):
        candidates = []
        current_cost = self.model.goal_cost(current_latent, goal_latent)
        progress_by_duration = {5: 0.25, 10: 0.8, 20: 0.9, 40: 0.95}
        for duration in durations:
            if duration > max_duration:
                continue
            progress = progress_by_duration[duration]
            subgoal = np.asarray([current_cost - progress], dtype=np.float32)
            candidates.append(
                CandidatePlan(
                    duration=duration,
                    subgoal=subgoal,
                    actions=np.asarray([[-0.1]], dtype=np.float32),
                    predicted_path=np.asarray([[current_latent[0] - 0.1]], dtype=np.float32),
                    current_goal_cost=current_cost,
                    subgoal_goal_cost=current_cost - progress,
                    predicted_miss=0.0,
                    miss_upper_bound=0.0,
                    success_probability=1.0,
                    planning_cost=1.0,
                )
            )
        return candidates

    def refine(self, current_latent, candidate, remaining_duration):
        return replace(
            candidate,
            actions=np.asarray([[-0.1]], dtype=np.float32),
            predicted_path=np.asarray([[current_latent[0] - 0.1]], dtype=np.float32),
        )


class CurrentDependentTubeRisk(ConstantRiskEstimator):
    def tube_threshold(self, current_latent, subgoal, duration, step):
        del subgoal, duration, step
        return float(current_latent[0])


class SequenceGenerator(FixedGenerator):
    def propose(self, current_latent, goal_latent, durations, max_duration):
        candidate = super().propose(current_latent, goal_latent, (5,), 5)[0]
        return [
            replace(
                candidate,
                actions=np.asarray([[-0.1], [-0.2], [-0.3]], dtype=np.float32),
                predicted_path=np.asarray([[0.9], [0.7], [0.4]], dtype=np.float32),
                duration=3,
            )
        ]


class AnchoredGenerator(FixedGenerator):
    def propose(self, current_latent, goal_latent, durations, max_duration):
        macro = super().propose(current_latent, goal_latent, (5,), 5)[0]
        anchor = replace(
            macro,
            duration=25,
            subgoal=np.asarray([2.0], dtype=np.float32),
            actions=np.full((25, 1), -0.04, dtype=np.float32),
            predicted_path=np.linspace(0.96, 0.0, 25, dtype=np.float32)[:, None],
            subgoal_goal_cost=2.0,
            metadata={"trusted_anchor": True},
        )
        return [macro, anchor]


def test_planner_selects_progress_per_duration_and_emits_diagnostics():
    planner = CAPEPlanner(
        IdentityWorldModel(),
        FixedGenerator(),
        ConstantRiskEstimator(tube_threshold=1.0),
        PlannerConfig(goal_threshold=0.01, subgoal_threshold=0.01),
    )
    action, diagnostics = planner.plan(np.asarray([1.0]), np.asarray([0.0]))
    assert action.shape == (1,)
    assert diagnostics.event == ExecutionEvent.NEW_PLAN
    assert diagnostics.chosen_duration == 10
    assert diagnostics.feasible_count == 4


def test_tube_violation_backs_off_duration_and_is_reported():
    planner = CAPEPlanner(
        IdentityWorldModel(),
        FixedGenerator(),
        ConstantRiskEstimator(tube_threshold=0.01),
        PlannerConfig(goal_threshold=0.01, subgoal_threshold=0.01),
    )
    planner.plan(np.asarray([1.0]), np.asarray([0.0]))
    _, diagnostics = planner.plan(np.asarray([1.0]), np.asarray([0.0]))
    assert diagnostics.trigger_event == ExecutionEvent.TUBE_VIOLATION
    assert diagnostics.max_duration == 20


def test_tube_threshold_is_committed_before_observing_next_state():
    planner = CAPEPlanner(
        IdentityWorldModel(),
        FixedGenerator(),
        CurrentDependentTubeRisk(),
        PlannerConfig(goal_threshold=0.01, subgoal_threshold=0.01),
    )
    planner.plan(np.asarray([1.0]), np.asarray([0.0]))
    _, diagnostics = planner.plan(np.asarray([0.9]), np.asarray([0.0]))
    assert diagnostics.tube_threshold == 1.0


def test_open_loop_commitment_consumes_action_sequence_in_order():
    planner = CAPEPlanner(
        IdentityWorldModel(),
        SequenceGenerator(),
        ConstantRiskEstimator(tube_threshold=10.0),
        PlannerConfig(
            durations=(5,),
            goal_threshold=0.01,
            subgoal_threshold=0.01,
            replan_low_level=False,
            stall_patience=10,
        ),
    )
    first, _ = planner.plan(np.asarray([1.0]), np.asarray([0.0]))
    second, _ = planner.plan(np.asarray([0.9]), np.asarray([0.0]))
    assert np.isclose(first.item(), -0.1)
    assert np.isclose(second.item(), -0.2)


def test_monitor_interval_checks_tube_only_at_model_block_boundaries():
    planner = CAPEPlanner(
        IdentityWorldModel(),
        FixedGenerator(),
        ConstantRiskEstimator(tube_threshold=0.01),
        PlannerConfig(
            goal_threshold=0.01,
            subgoal_threshold=0.01,
            monitor_interval=5,
            stall_patience=10,
        ),
    )
    planner.plan(np.asarray([1.0]), np.asarray([0.0]))
    for _ in range(4):
        _, diagnostics = planner.plan(np.asarray([1.0]), np.asarray([0.0]))
        assert diagnostics.trigger_event is None
    _, diagnostics = planner.plan(np.asarray([1.0]), np.asarray([0.0]))
    assert diagnostics.trigger_event == ExecutionEvent.TUBE_VIOLATION


def test_trusted_anchor_margin_and_commitment_bypass_risk_interruptions():
    planner = CAPEPlanner(
        IdentityWorldModel(),
        AnchoredGenerator(),
        ConstantRiskEstimator(
            predicted_miss=100.0,
            miss_upper_bound=100.0,
            success_probability=0.0,
            tube_threshold=0.0,
        ),
        PlannerConfig(
            durations=(5,),
            goal_threshold=-1.0,
            subgoal_threshold=10.0,
            anchor_advantage_margin=1.0,
            replan_low_level=False,
        ),
    )

    _, first = planner.plan(np.asarray([1.0]), np.asarray([0.0]))
    _, second = planner.plan(np.asarray([1.0]), np.asarray([0.0]))

    assert first.chosen_duration == 25
    assert first.feasible_count == 1
    assert second.trigger_event is None
    assert second.event == ExecutionEvent.CONTINUE
