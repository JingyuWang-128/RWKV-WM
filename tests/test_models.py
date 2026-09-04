import torch

from cape_wm.models import (
    ContinuousReachabilityDistribution,
    DirectedReachabilityDistribution,
    DurationConditionedMacroPredictor,
    ExecutabilityRiskHead,
    MacroActionEncoder,
    continuous_reachability_loss,
    reachability_distribution_loss,
    reachability_semigroup_loss,
    risk_head_loss,
)


def test_macro_models_support_variable_and_odd_durations():
    torch.manual_seed(0)
    encoder = MacroActionEncoder(2, macro_dim=4, model_dim=16, depth=1, max_duration=7)
    predictor = DurationConditionedMacroPredictor(
        latent_dim=6, macro_dim=4, hidden_dim=16, depth=2, max_duration=7
    )
    actions = torch.randn(3, 7, 2)
    durations = torch.tensor([3, 5, 7])
    mean, log_std = encoder(actions, durations)
    output = predictor(torch.randn(3, 6), encoder.sample(mean, log_std), durations)
    assert mean.shape == (3, 4)
    assert log_std.shape == (3, 4)
    assert output.shape == (3, 6)
    output.mean().backward()


def test_risk_head_produces_masked_per_step_scales():
    model = ExecutabilityRiskHead(latent_dim=5, hidden_dim=16, depth=2, max_duration=8)
    duration = torch.tensor([3, 8])
    output = model(torch.randn(2, 5), torch.randn(2, 5), duration)
    assert output[0].shape == (2,)
    assert output[1].shape == (2,)
    assert output[2].shape == (2, 8)
    mask = torch.arange(8)[None] < duration[:, None]
    loss = risk_head_loss(
        output,
        torch.rand(2),
        torch.tensor([1.0, 0.0]),
        torch.rand(2, 8).clamp_min(1e-3),
        mask,
    )
    loss.backward()
    assert torch.isfinite(loss)


def test_directed_reachability_distribution_is_monotone_and_trainable():
    torch.manual_seed(0)
    model = DirectedReachabilityDistribution(
        latent_dim=4, horizon_bins=(5, 10, 20), hidden_dim=16, depth=2
    )
    source = torch.randn(6, 4)
    goal = torch.randn(6, 4)
    cdf = model.cdf(source, goal)
    assert cdf.shape == (6, 3)
    assert torch.all(cdf[:, 1:] >= cdf[:, :-1])
    mean, std = model.expected_and_std(source, goal)
    assert mean.shape == std.shape == (6,)
    assert torch.all(std > 0)

    separation = torch.tensor([1, 5, 6, 10, 20, 21])
    assert model.target_class(separation).tolist() == [0, 0, 1, 1, 2, 3]
    loss = reachability_distribution_loss(model, source, goal, separation)
    loss.backward()
    assert torch.isfinite(loss)


def test_reachability_semigroup_loss_uses_only_registered_segments():
    loss = reachability_semigroup_loss(
        torch.tensor([10.0, 30.0]),
        torch.tensor([5.0, 20.0]),
        torch.tensor([5.0, 5.0]),
        torch.tensor([True, False]),
    )
    assert loss.item() == 0.0


def test_continuous_reachability_is_monotone_precise_and_trainable():
    torch.manual_seed(13)
    model = ContinuousReachabilityDistribution(
        latent_dim=4,
        horizon_bins=(5, 10, 20, 40),
        hidden_dim=16,
        depth=1,
    )
    source = torch.randn(6, 4)
    goal = torch.randn(6, 4)
    separation = torch.tensor([5.0, 10.0, 15.0, 20.0, 30.0, 40.0])
    cdf = model.cdf(source, goal)
    assert cdf.shape == (6, 4)
    assert torch.all(cdf[:, 1:] >= cdf[:, :-1])
    mean, scale = model.expected_and_std(source, goal)
    assert mean.shape == scale.shape == (6,)
    assert torch.all(mean > 0.0)
    assert torch.all(scale > 0.0)
    loss = continuous_reachability_loss(model, source, goal, separation)
    loss.backward()
    assert torch.isfinite(loss)
