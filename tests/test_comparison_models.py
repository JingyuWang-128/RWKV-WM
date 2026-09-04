import numpy as np
import torch
from torch import nn

from cape_wm.comparison_baselines import HybridTRMObjective, VLWMDynamics
from cape_wm.comparison_models import (
    PairwiseReachabilityMetric,
    VariableLengthPredictor,
    long_to_short_schedule,
)
from cape_wm.tworoom_data import make_split, sample_segments, sample_temporal_pairs


def test_comparison_models_shapes_and_schedule():
    trm = PairwiseReachabilityMetric(latent_dim=6, hidden_dim=8)
    values = trm(torch.randn(4, 6), torch.randn(4, 6))
    assert values.shape == (4,)
    assert torch.all(values >= 0)

    predictor = VariableLengthPredictor(
        latent_dim=6,
        action_dim=4,
        model_dim=12,
        depth=1,
        heads=3,
        mlp_dim=24,
        max_horizon=5,
    )
    prediction = predictor(
        torch.randn(3, 6),
        torch.randn(3, 5, 4),
        torch.tensor([1, 3, 5]),
    )
    assert prediction.shape == (3, 6)
    assert sum(long_to_short_schedule(5, 3)) == 5
    assert long_to_short_schedule(5, 3)[0] >= long_to_short_schedule(5, 3)[-1]


def test_tworoom_split_and_samplers_exclude_test_episodes():
    episodes = np.arange(8)
    split = make_split(episodes, (6, 7), seed=4, train_fraction=0.6)
    assert set(split.test) == {6, 7}
    assert not set(split.train) & set(split.test)

    offsets = np.arange(8) * 51
    lengths = np.full(8, 51)
    actions = np.random.default_rng(2).normal(size=(8 * 51, 2)).astype(np.float32)
    pairs = sample_temporal_pairs(
        split.train,
        episodes,
        offsets,
        lengths,
        count=20,
        seed=5,
        block_size=5,
    )
    assert set(pairs["episode_ids"]).issubset(split.train)
    assert np.all(pairs["separation"] % 5 == 0)

    segments = sample_segments(
        split.train,
        episodes,
        offsets,
        lengths,
        actions,
        count=20,
        seed=6,
        durations=(5, 10, 20, 40),
        block_size=5,
    )
    assert segments["actions"].shape == (20, 40, 2)
    assert set(segments["episode_ids"]).issubset(split.train)
    assert np.all(segments["target_rows"] - segments["source_rows"] == segments["durations"])


class _FakeEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.latent_dim = latent_dim

    def encode(self, info):
        pixels = info["pixels"]
        shape = (*pixels.shape[:2], self.latent_dim)
        info["emb"] = torch.zeros(shape, device=pixels.device)
        return info


def test_hybrid_objective_and_vlwm_rollout_shapes():
    trm = PairwiseReachabilityMetric(latent_dim=6, hidden_dim=8)
    objective = HybridTRMObjective(trm)
    info = {
        "predicted_emb": torch.randn(2, 7, 3, 6),
        "goal_emb": torch.randn(2, 1, 6),
    }
    cost = objective(info)
    assert cost.shape == (2, 7)
    assert torch.allclose(cost.mean(dim=1), torch.zeros(2), atol=1e-5)

    predictor = VariableLengthPredictor(
        latent_dim=6,
        action_dim=4,
        model_dim=12,
        depth=1,
        heads=3,
        mlp_dim=24,
        max_horizon=5,
    )
    dynamics = VLWMDynamics(_FakeEncoder(6), predictor, (3, 1, 1))
    rollout = dynamics.rollout(
        {"pixels": torch.randn(2, 7, 1, 3, 4, 4)},
        torch.randn(2, 7, 5, 4),
    )
    assert rollout["predicted_emb"].shape == (2, 7, 4, 6)
