import numpy as np
import torch

from cape_wm.generators import OfficialCEMLowLevelController
from cape_wm.interfaces import WorldModelAdapter


class QuadraticBlockModel(WorldModelAdapter):
    device = torch.device("cpu")
    action_block = 2

    def encode(self, observation):
        return np.asarray(observation, dtype=np.float32)

    def rollout(self, latent, actions):
        return self.batch_rollout(latent, np.asarray(actions)[None])[0]

    def batch_rollout(self, latent, action_population):
        actions = np.asarray(action_population, dtype=np.float32)
        return np.asarray(latent, dtype=np.float32)[None, None] + np.cumsum(
            actions, axis=1
        )

    def goal_cost(self, latent, goal_latent):
        return float(np.square(np.asarray(latent) - np.asarray(goal_latent)).sum())

    def batch_goal_cost(self, latents, goal_latent):
        return np.square(np.asarray(latents) - np.asarray(goal_latent)).sum(axis=-1)

    @property
    def action_shape(self):
        return (1,)


def test_official_cem_controller_returns_fixed_horizon_mean_plan():
    model = QuadraticBlockModel()
    controller = OfficialCEMLowLevelController(
        model,
        horizon=4,
        action_block=2,
        samples=64,
        elites=8,
        iterations=6,
        generator=torch.Generator().manual_seed(3),
    )

    actions, path, transitions = controller.plan(
        np.asarray([0.0], dtype=np.float32), np.asarray([1.0], dtype=np.float32)
    )

    assert actions.shape == (4, 1)
    assert path.shape == (4, 1)
    assert abs(path[-1, 0] - 1.0) < 0.25
    assert transitions == 64 * 6 * 2 + 2


def test_official_cem_controller_can_support_macro_duration_grid():
    model = QuadraticBlockModel()
    controller = OfficialCEMLowLevelController(
        model,
        horizon=6,
        action_block=2,
        samples=32,
        elites=4,
        iterations=3,
        generator=torch.Generator().manual_seed(4),
        fixed_horizon=False,
    )

    actions, path, transitions = controller.plan(
        np.asarray([0.0], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        horizon=2,
    )

    assert actions.shape == (2, 1)
    assert path.shape == (2, 1)
    assert transitions == 32 * 3 + 1
