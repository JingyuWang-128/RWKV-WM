import pytest
import torch

from cape_wm.cc_rwkv.cell import (
    CC_DECAY_LOG_HAZARD,
    CC_DECAY_OFFICIAL_LOGIT,
    RWKV7_DECAY_SCALE,
    counterfactual_rwkv7_decay,
    counterfactual_rwkv7_matrix_step,
)
from cape_wm.cc_rwkv.counterfactual import (
    CounterfactualRWKV7Config,
    CounterfactualRWKV7WorldPredictor,
)
from cape_wm.cc_rwkv.predictor import RWKV7WorldModelConfig, VanillaRWKV7WorldPredictor


def _config(*, centered=True, channel_mlp_dim=32):
    return CounterfactualRWKV7Config(
        latent_dim=6,
        action_dim=3,
        model_dim=8,
        num_layers=2,
        num_heads=2,
        channel_mlp_dim=channel_mlp_dim,
        decay_lora_dim=4,
        aaa_lora_dim=4,
        value_lora_dim=4,
        gate_lora_dim=4,
        action_hidden_dim=5,
        centered=centered,
    )


def _model(*, centered=True):
    return CounterfactualRWKV7WorldPredictor(_config(centered=centered))


def _enable_action_effect(model):
    for block in model.blocks:
        block.time_mix.output.weight.data.normal_(std=0.1)
        block.time_mix.action_parameter_network.delta_head.weight.data.normal_(std=0.1)


def test_log_hazard_double_exp_matches_oracle_and_exact_official_zero_delta():
    logits = torch.tensor([[-8.0, -6.0, -3.0, 0.0, 3.0, 5.0]])
    zero = torch.zeros_like(logits)
    actual, official, rate = counterfactual_rwkv7_decay(logits, zero)
    expected_rate = RWKV7_DECAY_SCALE * torch.sigmoid(logits)
    expected_official = torch.exp(-expected_rate)
    assert torch.equal(actual, official)
    assert torch.equal(official, expected_official)
    assert torch.equal(rate, expected_rate)

    residual = torch.tensor([[-9.0, -3.0, -1.0, 0.5, 3.0, 7.0]])
    actual, _, rate = counterfactual_rwkv7_decay(logits, residual)
    bounded = residual.clamp(-8.0, 5.0)
    world_log_hazard = torch.log(expected_rate)
    expected_rate = torch.exp(world_log_hazard + bounded)
    torch.testing.assert_close(rate, expected_rate)
    torch.testing.assert_close(actual, torch.exp(-torch.exp(world_log_hazard + bounded)))


def test_log_hazard_mode_restores_decay_range_beyond_official_logit_ablation():
    logits = torch.zeros(3)
    residual = torch.tensor([-5.0, 0.0, 5.0])
    hazard, official, _ = counterfactual_rwkv7_decay(
        logits, residual, mode=CC_DECAY_LOG_HAZARD
    )
    old_ablation, old_official, _ = counterfactual_rwkv7_decay(
        logits, residual, mode=CC_DECAY_OFFICIAL_LOGIT
    )
    assert torch.equal(official, old_official)
    assert hazard[0] > old_ablation[0]
    assert hazard[2] < 1e-10
    assert old_ablation[2] > torch.exp(torch.tensor(-RWKV7_DECAY_SCALE))
    with pytest.raises(ValueError, match="unsupported"):
        counterfactual_rwkv7_decay(logits, residual, mode="not-a-decay")


def test_rank_two_decomposition_matches_explicit_transition_matrix():
    torch.manual_seed(101)
    batch, heads, width = 2, 3, 4
    matrix = torch.randn(batch, heads, width, width)
    decay = torch.sigmoid(torch.randn(batch, heads, width))
    world_erase = torch.randn(batch, heads, width)
    world_erase_key = torch.randn(batch, heads, width)
    world_value = torch.randn(batch, heads, width)
    world_write_key = torch.randn(batch, heads, width)
    action_erase = torch.randn(batch, heads, width)
    action_erase_key = torch.randn(batch, heads, width)
    action_write = torch.randn(batch, heads, width)
    action_write_key = torch.randn(batch, heads, width)

    actual, world, action = counterfactual_rwkv7_matrix_step(
        matrix,
        decay,
        world_erase,
        world_erase_key,
        world_value,
        world_write_key,
        action_erase,
        action_erase_key,
        action_write,
        action_write_key,
    )
    world_transition = torch.diag_embed(decay) + world_erase.unsqueeze(-1) * (
        world_erase_key.unsqueeze(-2)
    )
    action_transition = action_erase.unsqueeze(-1) * action_erase_key.unsqueeze(-2)
    expected_world = matrix @ world_transition + world_value.unsqueeze(-1) * (
        world_write_key.unsqueeze(-2)
    )
    expected_action = matrix @ action_transition + action_write.unsqueeze(-1) * (
        action_write_key.unsqueeze(-2)
    )
    torch.testing.assert_close(world, expected_world, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(action, expected_action, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual, expected_world + expected_action, atol=1e-6, rtol=1e-5)


def test_centered_zero_delta_is_exact_official_rwkv7_reduction():
    torch.manual_seed(103)
    vanilla = VanillaRWKV7WorldPredictor(
        RWKV7WorldModelConfig(
            latent_dim=6,
            action_dim=3,
            model_dim=8,
            num_layers=2,
            num_heads=2,
            channel_mlp_dim=32,
            decay_lora_dim=4,
            aaa_lora_dim=4,
            value_lora_dim=4,
            gate_lora_dim=4,
        )
    )
    centered = _model()
    migration = centered.load_world_from_vanilla(vanilla)
    assert migration["ignored"] == ["action_projection.weight"]

    latent = torch.randn(3, 6)
    zero = torch.zeros(3, 3)
    arbitrary_equal_action = torch.randn(3, 3)
    expected, expected_state, _ = vanilla.step(latent, zero, vanilla.init_state(3))
    actual, actual_state, diagnostics = centered.step(
        latent,
        arbitrary_equal_action,
        centered.init_state(3),
        arbitrary_equal_action.clone(),
        return_diagnostics=True,
    )
    assert torch.equal(expected, actual)
    assert torch.equal(expected_state.matrix, actual_state.matrix)
    assert torch.equal(expected_state.time_shift, actual_state.time_shift)
    assert torch.count_nonzero(diagnostics["action_decay_delta"]) == 0
    assert torch.count_nonzero(diagnostics["action_erase_delta"]) == 0
    assert torch.count_nonzero(diagnostics["action_write_delta"]) == 0
    assert torch.count_nonzero(diagnostics["matrix_update_action_norm"]) == 0


def test_actual_reference_swap_negates_all_raw_action_deltas():
    torch.manual_seed(107)
    model = _model()
    _enable_action_effect(model)
    module = model.blocks[0].time_mix
    world = torch.randn(2, 8)
    previous_world = torch.randn(2, 8)
    matrix = torch.randn(2, 2, 4, 4)
    actual = torch.randn(2, 3)
    reference = torch.randn(2, 3)
    forward = module(world, previous_world, matrix, None, actual, reference)[-1]
    reverse = module(world, previous_world, matrix, None, reference, actual)[-1]
    for name in ("action_decay_delta", "action_erase_delta", "action_write_delta"):
        torch.testing.assert_close(forward[name], -reverse[name], atol=1e-7, rtol=1e-6)


def test_raw_action_has_no_prediction_bypass_but_can_act_through_matrix_update():
    torch.manual_seed(109)
    model = _model()
    assert not hasattr(model, "action_projection")
    action_consumers = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and module.in_features == model.config.action_dim
    ]
    assert action_consumers
    assert all("action_parameter_network.action_projection" in name for name in action_consumers)

    latent = torch.randn(2, 6)
    first = torch.randn(2, 3)
    second = torch.randn(2, 3)
    same_first, same_state, _ = model.step(
        latent, first, model.init_state(2), first.clone()
    )
    same_second, second_state, _ = model.step(
        latent, second, model.init_state(2), second.clone()
    )
    assert torch.equal(same_first, same_second)
    assert torch.equal(same_state.matrix, second_state.matrix)

    _enable_action_effect(model)
    reference = torch.zeros_like(first)
    factual, factual_state, _ = model.step(latent, first, model.init_state(2), reference)
    noop, noop_state, _ = model.step(latent, reference, model.init_state(2), reference)
    assert not torch.allclose(factual_state.matrix, noop_state.matrix)
    assert not torch.allclose(factual, noop)


def test_reference_subtraction_uses_shared_weights_and_has_finite_gradients():
    torch.manual_seed(113)
    model = _model()
    _enable_action_effect(model)
    latent = torch.randn(2, 6, requires_grad=True)
    actual = torch.randn(2, 3, requires_grad=True)
    reference = torch.randn(2, 3, requires_grad=True)
    prediction, state, _ = model.step(
        latent, actual, model.init_state(2), reference
    )
    (prediction.square().mean() + state.matrix.square().mean()).backward()
    assert torch.isfinite(actual.grad).all() and torch.count_nonzero(actual.grad)
    assert torch.isfinite(reference.grad).all() and torch.count_nonzero(reference.grad)
    action_nets = [block.time_mix.action_parameter_network for block in model.blocks]
    assert len(action_nets) == model.config.num_layers
    for network in action_nets:
        gradients = [p.grad for p in network.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_b4_b6_capacity_is_identical_and_main_width_matches_b0_budget():
    b4 = CounterfactualRWKV7WorldPredictor(
        CounterfactualRWKV7Config(channel_mlp_dim=3728, centered=False)
    )
    b6 = CounterfactualRWKV7WorldPredictor(
        CounterfactualRWKV7Config(channel_mlp_dim=3728, centered=True)
    )
    assert b4.parameter_count() == b6.parameter_count() == 10_783_296
    official_b0_parameters = 10_791_360
    assert abs(b6.parameter_count() / official_b0_parameters - 1.0) < 0.05


def test_initial_intervention_gate_is_active_and_not_saturated():
    model = _model()
    _, _, diagnostics = model.step(
        torch.randn(4, 6),
        torch.randn(4, 3),
        model.init_state(4),
        torch.zeros(4, 3),
        return_diagnostics=True,
    )
    gate = diagnostics["intervention_gate"]
    torch.testing.assert_close(gate, torch.full_like(gate, 0.1), atol=1e-7, rtol=1e-6)
    assert not ((gate < 0.01) | (gate > 0.99)).any()
