from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import h5py
import numpy as np
import pytest
import torch

from cape_wm.cc_rwkv.dwm import DWMOutputBaseline
from cape_wm.cc_rwkv.per_step_training import (
    PerStepBatch,
    _position_balanced_mean,
    _read_hdf5_rows,
    per_step_loss,
    per_step_rollout,
)
from cape_wm.cc_rwkv.predictor import (
    RWKV7WorldModelConfig,
    VanillaRWKV7WorldPredictor,
)
from scripts.train_per_step_pairs import (
    OPTIMIZER_POLICY,
    require_finite_tensor,
    verify_data_artifacts,
    verify_protocol_manifest,
)


def _predictor() -> VanillaRWKV7WorldPredictor:
    return VanillaRWKV7WorldPredictor(
        RWKV7WorldModelConfig(
            latent_dim=6,
            action_dim=3,
            model_dim=8,
            num_layers=2,
            num_heads=2,
            channel_mlp_dim=16,
            decay_lora_dim=4,
            aaa_lora_dim=4,
            value_lora_dim=4,
            gate_lora_dim=4,
        )
    )


def _batch(batch_size: int = 4, horizon: int = 3) -> PerStepBatch:
    torch.manual_seed(401)
    factual_latents = torch.randn(batch_size, horizon + 1, 6)
    pulse_latents = torch.randn(batch_size, horizon, horizon, 6)
    mask = torch.zeros(batch_size, horizon, horizon, dtype=torch.bool)
    for position in range(horizon):
        mask[:, position, : horizon - position] = True
    future = factual_latents[:, 1:]
    aligned = torch.zeros_like(pulse_latents)
    for position in range(horizon):
        length = horizon - position
        aligned[:, position, :length] = future[:, position : position + length]
    effect = (pulse_latents - aligned) * mask[..., None]
    factual_actions = torch.randn(batch_size, horizon, 3)
    return PerStepBatch(
        history_latents=torch.randn(batch_size, 2, 6),
        history_actions=torch.randn(batch_size, 2, 3),
        history_mask=torch.ones(batch_size, 2, dtype=torch.bool),
        factual_actions=factual_actions,
        factual_latents=factual_latents,
        pulse_noop_actions=torch.zeros(batch_size, horizon, horizon, 3),
        pulse_noop_latents=pulse_latents,
        pulse_noop_mask=mask,
        effect_latents=effect,
        sample_ids=[f"sample-{index}" for index in range(batch_size)],
    )


def test_position_balanced_mean_does_not_overweight_early_positions() -> None:
    values = torch.tensor([[[1.0, 1.0, 1.0], [3.0, 0.0, 0.0]]])
    mask = torch.tensor([[[True, True, True], [True, False, False]]])
    torch.testing.assert_close(_position_balanced_mean(values, mask), torch.tensor(2.0))


@pytest.mark.parametrize("horizon", [1, 5, 10, 20])
def test_curriculum_mask_only_contains_generated_predictions(horizon) -> None:
    model = _predictor().eval()
    batch = _batch(batch_size=2, horizon=20)
    with torch.no_grad():
        output = per_step_rollout(model, batch, horizon=horizon)
    positions = torch.arange(horizon)
    expected = (positions[:, None] + positions[None, :] < horizon)[None].expand(2, -1, -1)
    torch.testing.assert_close(output["mask"], expected)
    assert output["mask"][0].sum() == horizon * (horizon + 1) // 2


@pytest.mark.parametrize("horizon", [5, 10])
@pytest.mark.parametrize("paired", [False, True])
def test_padding_labels_cannot_change_curriculum_loss_or_gradients(horizon, paired) -> None:
    model = _predictor().eval()
    batch = _batch(batch_size=2, horizon=20)
    positions = torch.arange(20)
    outside = (positions[:, None] + positions[None, :] >= horizon)[None, :, :, None]
    changed = replace(
        batch,
        pulse_noop_latents=batch.pulse_noop_latents + outside * 100,
        effect_latents=batch.effect_latents + outside * 50,
    )
    original, components = per_step_loss(
        model, batch, horizon=horizon, allow_paired_loss=paired, effect_weight=0.5
    )
    original_grad = torch.autograd.grad(original, tuple(model.parameters()), allow_unused=True)
    modified, changed_components = per_step_loss(
        model, changed, horizon=horizon, allow_paired_loss=paired, effect_weight=0.5
    )
    changed_grad = torch.autograd.grad(modified, tuple(model.parameters()), allow_unused=True)
    torch.testing.assert_close(original, modified)
    for key in components:
        torch.testing.assert_close(components[key], changed_components[key])
    for first, second in zip(original_grad, changed_grad, strict=True):
        if first is None:
            assert second is None
        else:
            torch.testing.assert_close(first, second)


def test_f4_disables_effect_computation_even_with_configured_weight_half(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("F4 must not calculate direction loss")

    monkeypatch.setattr(torch.nn.functional, "cosine_similarity", forbidden)
    model = _predictor().eval()
    batch = _batch(batch_size=2, horizon=5)
    batch = replace(batch, effect_latents=torch.full_like(batch.effect_latents, float("nan")))
    loss, components = per_step_loss(
        model, batch, horizon=3, allow_paired_loss=False, effect_weight=0.5
    )
    assert torch.isfinite(loss)
    for key in ("paired_effect", "effect_direction", "effect_magnitude"):
        assert components[key] == 0
    torch.testing.assert_close(
        loss, components["factual_prediction"] + components["noop_prediction"]
    )


def test_per_step_b3_trains_dwm_world_head_at_every_position() -> None:
    torch.manual_seed(403)
    model = DWMOutputBaseline(_predictor(), world_head_hidden_dim=16)
    total, components = per_step_loss(
        model,
        _batch(),
        horizon=3,
        allow_paired_loss=False,
    )
    total.backward()
    world_head_gradients = [
        parameter.grad for parameter in model.world_head.parameters() if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in world_head_gradients)
    assert sum(float(gradient.norm()) for gradient in world_head_gradients) > 0
    assert components["dwm_world_contrastive"] > 0
    assert components["dwm_orthogonality"] >= 0
    assert components["paired_effect"] == 0


def test_b3_does_not_consume_paired_effect_labels() -> None:
    torch.manual_seed(409)
    model = DWMOutputBaseline(_predictor(), world_head_hidden_dim=16).eval()
    batch = _batch()
    changed = replace(batch, effect_latents=torch.full_like(batch.effect_latents, 1e4))
    with torch.no_grad():
        original, _ = per_step_loss(model, batch, allow_paired_loss=False)
        modified, _ = per_step_loss(model, changed, allow_paired_loss=False)
    torch.testing.assert_close(original, modified)


def test_paired_effect_loss_consumes_every_intervention_position() -> None:
    """Every factual prefix must contribute its pulse-noop effect target."""

    torch.manual_seed(419)
    model = _predictor().eval()
    batch = _batch()
    with torch.no_grad():
        _, baseline = per_step_loss(model, batch, horizon=3, allow_paired_loss=True)
        for position in range(batch.horizon):
            changed_effect = batch.effect_latents.clone()
            changed_effect[:, position, 0] += 10.0
            changed = replace(batch, effect_latents=changed_effect)
            _, components = per_step_loss(
                model, changed, horizon=3, allow_paired_loss=True
            )
            assert not torch.isclose(
                baseline["paired_effect"], components["paired_effect"]
            )


def test_vectorized_pulse_noop_suffix_matches_positionwise_rollout() -> None:
    """The batched path must preserve zero-at-t then factual-suffix semantics."""

    torch.manual_seed(421)
    model = _predictor().eval()
    batch = _batch()
    with torch.no_grad():
        vectorized = per_step_rollout(
            model, batch, horizon=3, vectorized_branches=True
        )
        positionwise = per_step_rollout(
            model, batch, horizon=3, vectorized_branches=False
        )
    torch.testing.assert_close(
        vectorized["factual_predicted"], positionwise["factual_predicted"]
    )
    torch.testing.assert_close(
        vectorized["pulse_predicted"], positionwise["pulse_predicted"]
    )


def test_formal_data_verification_binds_manifest_and_passed_audit(tmp_path) -> None:
    data = tmp_path / "pairs.h5"
    data.write_bytes(b"pair-data")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "pair_dataset_sha256": "frozen-sha",
                "bytes": data.stat().st_size,
                "samples": 2,
            }
        )
    )
    (tmp_path / "audit.json").write_text(
        json.dumps(
            {
                "dataset": str(data),
                "samples": 2,
                "failures": [],
                "status": "pass",
            }
        )
    )
    manifest = verify_data_artifacts(data, expected_sha256="frozen-sha", formal=True)
    assert manifest["samples"] == 2
    with pytest.raises(ValueError, match="frozen protocol"):
        verify_data_artifacts(data, expected_sha256="wrong", formal=True)


def test_protocol_manifest_verifies_frozen_code_and_dataset(tmp_path) -> None:
    data = tmp_path / "pairs.h5"
    data.write_bytes(b"data")
    code = tmp_path / "training.py"
    code.write_text("frozen = True\n")
    manifest = tmp_path / "protocol.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol_id": "formal_test",
                "data_status": "audited_v2_primitive_pass",
                "optimizer_policy": OPTIMIZER_POLICY,
                "datasets": {"test": {"path": str(data), "sha256": "data-sha"}},
                "code_sha256": {
                    str(code): hashlib.sha256(code.read_bytes()).hexdigest(),
                },
            }
        )
    )
    assert verify_protocol_manifest(
        manifest,
        protocol_id="formal_test",
        data=data,
        data_sha256="data-sha",
    ) == hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_hdf5_split_reader_preserves_large_split_row_order(tmp_path) -> None:
    source = np.arange(60, dtype=np.float32).reshape(20, 3)
    path = tmp_path / "rows.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("values", data=source)
    indices = np.asarray([0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19])
    with h5py.File(path, "r") as handle:
        selected = _read_hdf5_rows(handle["values"], indices)
    np.testing.assert_array_equal(selected, source[indices])


def test_training_fail_fast_rejects_nan_and_inf_before_backward() -> None:
    require_finite_tensor(torch.tensor(1.0), name="loss", step=7)
    with pytest.raises(FloatingPointError, match="non-finite loss at step 7"):
        require_finite_tensor(torch.tensor(float("nan")), name="loss", step=7)
    with pytest.raises(FloatingPointError, match="non-finite gradient norm at step 8"):
        require_finite_tensor(
            torch.tensor(float("inf")), name="gradient norm", step=8
        )
