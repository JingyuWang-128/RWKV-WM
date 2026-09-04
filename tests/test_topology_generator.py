from __future__ import annotations

import numpy as np

from cape_wm.generators import TopologyWaypointGenerator
from cape_wm.interfaces import CandidateGenerator, WorldModelAdapter
from cape_wm.types import CandidatePlan


class PositionWorldModel(WorldModelAdapter):
    def encode(self, observation):
        return np.asarray(observation, dtype=np.float32)

    def rollout(self, latent, actions):
        return np.repeat(np.asarray(latent)[None], len(actions), axis=0)

    def goal_cost(self, latent, goal_latent):
        return float(np.linalg.norm(np.asarray(latent) - np.asarray(goal_latent)))

    @property
    def action_shape(self):
        return (2,)


class CountingBaseGenerator(CandidateGenerator):
    def __init__(self):
        self.calls = 0

    def propose(self, current_latent, goal_latent, durations, max_duration):
        self.calls += 1
        return [
            CandidatePlan(
                duration=max_duration,
                subgoal=np.asarray(goal_latent, dtype=np.float32),
                actions=np.zeros((max_duration, 2), dtype=np.float32),
                predicted_path=np.empty((0, 2), dtype=np.float32),
                current_goal_cost=1.0,
                subgoal_goal_cost=0.0,
                predicted_miss=0.0,
                miss_upper_bound=0.0,
                success_probability=1.0,
                planning_cost=1.0,
                metadata={"generator": "base"},
            )
        ]


def make_generator(base, *, direct_goal_completion=False):
    return TopologyWaypointGenerator(
        base,
        PositionWorldModel(),
        None,
        duration=5,
        latent_mean=np.zeros(2, dtype=np.float32),
        latent_scale=np.ones(2, dtype=np.float32),
        position_weight=np.eye(2, dtype=np.float32),
        position_mean=np.zeros(2, dtype=np.float32),
        wall_position=0.0,
        door_position=0.0,
        side_offset=1.0,
        door_left_latent=np.asarray([-1.0, 0.0], dtype=np.float32),
        door_right_latent=np.asarray([1.0, 0.0], dtype=np.float32),
        direct_action_transform=lambda value: value,
        direct_goal_completion=direct_goal_completion,
    )


def test_cross_room_proposal_skips_unused_base_planning():
    base = CountingBaseGenerator()
    generator = make_generator(base)

    candidates = generator.propose(
        np.asarray([-2.0, 2.0], dtype=np.float32),
        np.asarray([2.0, 2.0], dtype=np.float32),
        (25,),
        25,
    )

    assert base.calls == 0
    assert len(candidates) == 1
    assert candidates[0].metadata["preferred_topology"] is True
    assert candidates[0].planning_cost == 0.0


def test_same_room_proposal_delegates_to_base_planning():
    base = CountingBaseGenerator()
    generator = make_generator(base)

    candidates = generator.propose(
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.asarray([2.0, 2.0], dtype=np.float32),
        (25,),
        25,
    )

    assert base.calls == 1
    assert candidates[0].metadata["generator"] == "base"


def test_direct_completion_clears_door_then_routes_to_goal():
    base = CountingBaseGenerator()
    generator = make_generator(base, direct_goal_completion=True)
    goal = np.asarray([2.0, 2.0], dtype=np.float32)

    first = generator.propose(np.asarray([-2.0, 2.0], dtype=np.float32), goal, (25,), 25)
    clearance = generator.propose(np.asarray([0.2, 0.1], dtype=np.float32), goal, (25,), 25)
    completion = generator.propose(np.asarray([1.1, 0.1], dtype=np.float32), goal, (25,), 25)

    assert first[0].metadata["generator"] == "topology_waypoint"
    assert clearance[0].metadata["generator"] == "topology_waypoint"
    assert completion[0].metadata["generator"] == "topology_goal_completion"
    assert np.allclose(completion[0].subgoal, goal)
    assert base.calls == 0


def test_direct_completion_routes_wall_band_goal_via_door_approach():
    base = CountingBaseGenerator()
    generator = make_generator(base, direct_goal_completion=True)
    goal = np.asarray([-0.5, 0.0], dtype=np.float32)

    approach = generator.propose(np.asarray([-2.0, 2.0], dtype=np.float32), goal, (25,), 25)
    completion = generator.propose(np.asarray([-1.0, 0.0], dtype=np.float32), goal, (25,), 25)

    assert approach[0].metadata["generator"] == "topology_waypoint"
    assert completion[0].metadata["generator"] == "topology_goal_completion"


def test_direct_completion_state_resets_between_episodes():
    base = CountingBaseGenerator()
    generator = make_generator(base, direct_goal_completion=True)
    generator.propose(
        np.asarray([-2.0, 2.0], dtype=np.float32),
        np.asarray([2.0, 2.0], dtype=np.float32),
        (25,),
        25,
    )

    generator.reset()
    candidates = generator.propose(
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.asarray([2.0, 2.0], dtype=np.float32),
        (25,),
        25,
    )

    assert base.calls == 1
    assert candidates[0].metadata["generator"] == "base"
