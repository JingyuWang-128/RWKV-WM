import pytest
import torch

from cape_wm.cc_rwkv.predictor import (
    RWKV7WorldModelConfig,
    VanillaRWKV7WorldPredictor,
    rwkv7_optimizer_groups,
)


def _predictor():
    config = RWKV7WorldModelConfig(
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
    return VanillaRWKV7WorldPredictor(config)


def test_sequential_step_and_sequence_wrapper_are_identical():
    torch.manual_seed(7)
    model = _predictor()
    latents = torch.randn(2, 5, 6)
    actions = torch.randn(2, 5, 3)
    sequence, sequence_state, _ = model.forward_sequence(latents, actions)
    state = model.init_state(2, device=latents.device, dtype=latents.dtype)
    step_predictions = []
    for index in range(5):
        prediction, state, _ = model.step(latents[:, index], actions[:, index], state)
        step_predictions.append(prediction)
    torch.testing.assert_close(sequence, torch.stack(step_predictions, dim=1))
    torch.testing.assert_close(sequence_state.matrix, state.matrix)
    torch.testing.assert_close(sequence_state.time_shift, state.time_shift)
    assert sequence_state.steps.tolist() == [5, 5]


def test_padding_does_not_update_any_recurrent_state():
    torch.manual_seed(9)
    model = _predictor()
    latents = torch.randn(2, 4, 6)
    actions = torch.randn(2, 4, 3)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    _, masked_state, _ = model.forward_sequence(latents, actions, mask=mask)
    _, prefix_state, _ = model.forward_sequence(latents[:1, :2], actions[:1, :2])
    torch.testing.assert_close(masked_state.matrix[:1], prefix_state.matrix)
    torch.testing.assert_close(masked_state.time_shift[:1], prefix_state.time_shift)
    torch.testing.assert_close(masked_state.channel_shift[:1], prefix_state.channel_shift)
    assert masked_state.steps.tolist() == [2, 4]


def test_rollout_clones_branches_and_is_fully_free_running():
    torch.manual_seed(11)
    model = _predictor()
    history_latents = torch.randn(2, 3, 6)
    history_actions = torch.randn(2, 3, 3)
    state = model.consume_history(history_latents, history_actions)
    original = state.clone()
    initial = history_latents[:, -1]
    actions = torch.randn(2, 3, 20, 3)
    result = model.rollout(initial, actions, state)
    assert result["latents"].shape == (2, 3, 20, 6)
    assert result["final_state"].matrix.shape[:2] == (6, 2)
    assert result["final_state"].steps.tolist() == [23] * 6
    torch.testing.assert_close(state.matrix, original.matrix)
    assert torch.isfinite(result["latents"]).all()
    assert torch.isfinite(result["final_state"].matrix).all()


def test_persistent_state_changes_output_relative_to_forced_reset():
    torch.manual_seed(13)
    model = _predictor()
    for block in model.blocks:
        block.time_mix.output.weight.data.normal_(std=0.2)
        block.channel_mix.value.weight.data.normal_(std=0.2)
    latent = torch.randn(1, 6)
    actions = torch.randn(1, 2, 3)
    state = model.init_state(1, device=latent.device, dtype=latent.dtype)
    _, state, _ = model.step(latent, actions[:, 0], state)
    persistent, _, _ = model.step(latent, actions[:, 1], state)
    reset, _, _ = model.step(
        latent,
        actions[:, 1],
        model.init_state(1, device=latent.device, dtype=latent.dtype),
    )
    assert not torch.allclose(persistent, reset)


def test_bfloat16_twenty_step_recurrence_is_finite_when_supported():
    model = _predictor().to(dtype=torch.bfloat16)
    latents = torch.randn(2, 20, 6, dtype=torch.bfloat16)
    actions = torch.randn(2, 20, 3, dtype=torch.bfloat16)
    try:
        output, state, _ = model.forward_sequence(latents, actions)
    except RuntimeError as error:
        pytest.skip(f"this torch backend lacks CPU bfloat16 support: {error}")
    assert torch.isfinite(output).all()
    assert torch.isfinite(state.matrix).all()
    assert state.matrix.dtype == torch.float32


def test_optimizer_groups_isolate_decay_logits_and_large_weights():
    model = _predictor()
    groups = rwkv7_optimizer_groups(model, weight_decay=1e-3)
    assert [group["lr_scale"] for group in groups] == [1.0, 2.0, 1.0]
    assert groups[0]["weight_decay"] == groups[1]["weight_decay"] == 0
    assert groups[2]["weight_decay"] == 1e-3
    w0_ids = {id(block.time_mix.w0) for block in model.blocks}
    assert {id(parameter) for parameter in groups[1]["params"]} == w0_ids

