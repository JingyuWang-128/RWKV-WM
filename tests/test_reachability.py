from __future__ import annotations

import numpy as np
import pytest
import torch

from cape_wm.interfaces import WorldModelAdapter
from cape_wm.models import DirectedReachabilityDistribution
from cape_wm.reachability import (
    CalibratedReachabilityScorer,
    MultiHorizonReachabilityCEM,
    ReachabilityAdvantageCalibrator,
    ReachabilityAdvantagePlanner,
)


class LinearWorldModel(WorldModelAdapter):
    def encode(self, observation):
        return np.asarray(observation, dtype=np.float32).reshape(-1)

    def rollout(self, latent, actions):
        increments = np.asarray(actions, dtype=np.float32).reshape(len(actions), -1).mean(axis=1)
        return np.asarray(latent, dtype=np.float32) + np.cumsum(increments)[:, None]

    def batch_rollout(self, latent, action_population):
        return np.stack([self.rollout(latent, actions) for actions in action_population])

    def goal_cost(self, latent, goal_latent):
        return float(np.square(np.asarray(latent) - np.asarray(goal_latent)).mean())

    @property
    def action_shape(self):
        return (1,)


def test_advantage_calibration_is_duration_conditional_and_round_trips():
    calibrator = ReachabilityAdvantageCalibrator(alpha=0.5).fit(
        predicted_progress=np.asarray([3.0, 5.0, 8.0, 13.0]),
        observed_progress=np.asarray([2.0, 4.0, 4.0, 8.0]),
        durations=np.asarray([5, 5, 10, 10]),
        predicted_scale=np.ones(4),
    )
    assert calibrator.quantile(5) == 1.0
    assert calibrator.quantile(10) == 5.0
    assert calibrator.lower_bound(7.0, 2.0, 5) == 5.0
    restored = ReachabilityAdvantageCalibrator.from_dict(calibrator.as_dict())
    assert restored.as_dict() == calibrator.as_dict()
    with pytest.raises(ValueError, match="no advantage calibration"):
        restored.quantile(20)


def test_constant_scale_calibration_does_not_reintroduce_model_spread():
    calibrator = ReachabilityAdvantageCalibrator(
        alpha=0.5, scale_mode="constant"
    ).fit(
        predicted_progress=np.asarray([3.0, 5.0]),
        observed_progress=np.asarray([2.0, 4.0]),
        durations=np.asarray([5, 5]),
        predicted_scale=np.asarray([100.0, 0.01]),
    )
    assert calibrator.quantile(5) == 1.0
    assert calibrator.lower_bound(7.0, 100.0, 5) == 6.0
    assert calibrator.scale_mode == "constant"


def test_shared_prefix_scorer_has_one_continuous_decision_path():
    torch.manual_seed(1)
    model = DirectedReachabilityDistribution(
        latent_dim=3, horizon_bins=(2, 4, 8), hidden_dim=12, depth=1
    )
    calibrator = ReachabilityAdvantageCalibrator(alpha=0.5).fit(
        predicted_progress=np.zeros(6),
        observed_progress=np.zeros(6),
        durations=np.asarray([2, 2, 4, 4, 8, 8]),
        predicted_scale=np.ones(6),
    )
    scorer = CalibratedReachabilityScorer(model, calibrator, durations=(2, 4, 8))
    paths = torch.randn(5, 8, 3)
    rates, lower, predicted, scales = scorer.score_paths(
        torch.zeros(3), torch.ones(3), paths
    )
    assert rates.shape == lower.shape == predicted.shape == scales.shape == (5, 3)
    candidate, duration, certificate = scorer.select(torch.zeros(3), torch.ones(3), paths)
    assert 0 <= int(candidate) < len(paths)
    assert int(duration) in (2, 4, 8)
    assert torch.isfinite(certificate)


def test_reachability_planner_never_emits_fallback_or_trusted_branch():
    torch.manual_seed(2)
    world_model = LinearWorldModel()
    model = DirectedReachabilityDistribution(
        latent_dim=1, horizon_bins=(2, 4), hidden_dim=8, depth=1
    )
    calibrator = ReachabilityAdvantageCalibrator(alpha=0.5).fit(
        predicted_progress=np.zeros(4),
        observed_progress=np.zeros(4),
        durations=np.asarray([2, 2, 4, 4]),
    )
    scorer = CalibratedReachabilityScorer(model, calibrator, durations=(2, 4))
    controller = MultiHorizonReachabilityCEM(
        world_model,
        scorer,
        action_block=1,
        samples=8,
        elites=2,
        iterations=2,
        seed=3,
    )
    planner = ReachabilityAdvantagePlanner(world_model, controller)
    action, diagnostics = planner.plan(np.asarray([0.0]), np.asarray([1.0]))
    assert action.shape == (1,)
    assert diagnostics.fallback_used is False
    assert diagnostics.candidates[0]["generator"] == "calibrated_reachability_advantage"
    assert "trusted" not in str(diagnostics.candidates).lower()
