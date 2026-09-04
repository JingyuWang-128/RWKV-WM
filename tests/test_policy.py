import numpy as np

from cape_wm.adapters.toy import PointImageEnv
from cape_wm.cli import build_toy_planner
from cape_wm.policy import GoalImagePolicy, StableWorldModelCAPEPolicy


def test_goal_image_policy_accepts_mapping_and_resets():
    environment = PointImageEnv()
    observation, _ = environment.reset(
        options={"start": np.asarray([0.1, 0.1]), "goal": np.asarray([0.8, 0.8])}
    )
    policy = GoalImagePolicy(build_toy_planner(0))
    action = policy(
        {
            "pixels": observation,
            "goal_pixels": environment.render_state(environment.goal),
        }
    )
    assert action.shape == (2,)
    assert policy.last_diagnostics is not None
    policy.reset()
    assert policy.last_diagnostics is None


class FakePlanner:
    def __init__(self):
        self.resets = 0
        self.model = type("Model", (), {"action_shape": (1,)})()

    def reset(self):
        self.resets += 1

    def plan(self, observation, goal):
        del goal
        return np.asarray([np.mean(observation)], dtype=np.float32), object()


def test_stable_worldmodel_bridge_handles_vectorized_time_axis():
    policy = StableWorldModelCAPEPolicy(FakePlanner, num_envs=2)
    policy.set_env(type("Env", (), {"num_envs": 2})())
    info = {
        "pixels": np.ones((2, 1, 4, 4, 3), dtype=np.float32),
        "goal": np.zeros((2, 1, 4, 4, 3), dtype=np.float32),
        "_needs_flush": np.asarray([True, False]),
    }
    actions = policy.get_action(info)
    assert actions.shape == (2, 1)
    assert policy.planners[0].resets == 1
