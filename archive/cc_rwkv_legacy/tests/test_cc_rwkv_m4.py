import copy
import json

import pytest
import torch

from cape_wm.cc_rwkv.counterfactual import (
    CounterfactualRWKV7Config,
    CounterfactualRWKV7WorldPredictor,
)
from cape_wm.cc_rwkv.dwm import (
    DWM_IMPLEMENTATION_STATUS,
    DWMOutputBaseline,
    dwm_auxiliary_losses,
    symmetric_info_nce,
)
from cape_wm.cc_rwkv.fairness import (
    FairnessViolation,
    RunRecord,
    audit_run_matrix,
)
from cape_wm.cc_rwkv.metrics import (
    counterfactual_curves,
    fit_ridge_probe,
    normalized_curve_auc,
    probe_r2,
    select_effect_weight,
)
from cape_wm.cc_rwkv.predictor import (
    RWKV7WorldModelConfig,
    VanillaRWKV7WorldPredictor,
)
from cape_wm.cc_rwkv.trainer import M4Trainer, M4TrainerConfig
from cape_wm.cc_rwkv.training import (
    BranchBatch,
    HorizonCurriculum,
    LossWeights,
    RolloutBatch,
    m4_counterfactual_loss,
    rollout_branch_batch,
)
from scripts.train_cc_rwkv import restore_best_for_horizon_transition


def _batch(batch_size=4, horizon=2):
    torch.manual_seed(211)
    history_latents = torch.randn(batch_size, 2, 6)
    history_actions = torch.randn(batch_size, 2, 3)
    initial = torch.randn(batch_size, 1, 1, 6).expand(-1, 4, -1, -1).clone()
    future = torch.randn(batch_size, 4, horizon, 6)
    branch_latents = torch.cat((initial, future), dim=2)
    actions = torch.randn(batch_size, 4, horizon, 3)
    actions[:, 2, 0] = 0
    return BranchBatch(
        history_latents=history_latents,
        history_actions=history_actions,
        history_mask=torch.ones(batch_size, 2, dtype=torch.bool),
        branch_actions=actions,
        branch_latents=branch_latents,
        branch_mask=torch.ones(batch_size, 4, horizon + 1, dtype=torch.bool),
        sample_ids=[f"sample-{index}" for index in range(batch_size)],
    )


def _vanilla():
    return VanillaRWKV7WorldPredictor(
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


def _counterfactual(centered=True):
    return CounterfactualRWKV7WorldPredictor(
        CounterfactualRWKV7Config(
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
            action_hidden_dim=5,
            centered=centered,
        )
    )


def test_perfect_effects_have_zero_effect_direction_and_magnitude_losses():
    target = torch.randn(3, 4, 2, 6)
    rollout = RolloutBatch(
        predicted=target.clone(),
        target=target,
        target_initial=torch.randn(3, 4, 6),
        valid_mask=torch.ones(3, 4, 2, dtype=torch.bool),
        final_state=None,
        diagnostics={},
    )
    output = m4_counterfactual_loss(
        rollout,
        weights=LossWeights(gate=0),
        effect_threshold=0,
        allow_paired_loss=True,
    )
    for name in ("prediction", "effect", "direction", "magnitude", "world"):
        torch.testing.assert_close(output.components[name], torch.zeros(()), atol=1e-6, rtol=0)
    torch.testing.assert_close(output.total, torch.zeros(()), atol=1e-6, rtol=0)


def test_near_zero_effect_mask_is_finite_and_unpaired_method_rejects_labels():
    target = torch.zeros(2, 4, 1, 3)
    rollout = RolloutBatch(
        predicted=torch.randn_like(target),
        target=target,
        target_initial=torch.zeros(2, 4, 3),
        valid_mask=torch.ones(2, 4, 1, dtype=torch.bool),
        final_state=None,
        diagnostics={},
    )
    output = m4_counterfactual_loss(
        rollout,
        weights=LossWeights(),
        effect_threshold=0.1,
        allow_paired_loss=True,
    )
    assert output.components["direction"] == 0
    assert output.components["magnitude"] == 0
    assert torch.isfinite(output.total)
    with pytest.raises(ValueError, match="not allowed"):
        m4_counterfactual_loss(
            rollout,
            weights=LossWeights(effect=1),
            effect_threshold=0,
            allow_paired_loss=False,
        )


def test_b6_one_step_paired_loss_does_not_consume_future_effect_labels():
    target = torch.zeros(1, 4, 2, 3)
    predicted = target.clone()
    predicted[:, 1, 1] = 10.0
    rollout = RolloutBatch(
        predicted=predicted,
        target=target,
        target_initial=torch.zeros(1, 4, 3),
        valid_mask=torch.ones(1, 4, 2, dtype=torch.bool),
        final_state=None,
        diagnostics={},
    )
    one_step = m4_counterfactual_loss(
        rollout,
        weights=LossWeights(gate=0),
        effect_threshold=0,
        allow_paired_loss=True,
        paired_horizon=1,
    )
    all_steps = m4_counterfactual_loss(
        rollout,
        weights=LossWeights(gate=0),
        effect_threshold=0,
        allow_paired_loss=True,
    )
    assert one_step.components["prediction"] > 0
    assert one_step.components["effect"] == 0
    assert all_steps.components["effect"] > 0


def test_counterfactual_metrics_and_cer_are_stable_and_interpretable():
    target = torch.zeros(2, 4, 2, 3)
    target[:, 1] = 1
    target[:, 3] = 2
    predicted = 2 * target
    curves = counterfactual_curves(
        predicted,
        target,
        torch.zeros(2, 4, 3),
        effect_threshold=0.1,
    )
    torch.testing.assert_close(curves.ced, torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(curves.cer, torch.full((2,), 2.0))
    assert torch.isfinite(curves.cee).all()
    torch.testing.assert_close(
        normalized_curve_auc(torch.tensor([1.0, 3.0, 5.0])), torch.tensor(3.0)
    )


def test_effect_weight_selection_enforces_one_step_degradation_guard():
    selected = select_effect_weight(
        [
            {"effect_weight": 0.5, "validation_score": 2.0, "one_step_error": 1.0},
            {"effect_weight": 1.0, "validation_score": 1.0, "one_step_error": 1.04},
            {"effect_weight": 2.0, "validation_score": 1.5, "one_step_error": 1.02},
        ],
        b2_one_step_error=1.0,
    )
    assert selected["effect_weight"] == 2.0
    with pytest.raises(ValueError, match="all effect weights"):
        select_effect_weight(
            [{"effect_weight": 1.0, "validation_score": 1.0, "one_step_error": 2.0}],
            b2_one_step_error=1.0,
        )


def test_free_running_predictions_do_not_read_true_future_latents():
    model = _vanilla()
    batch = _batch()
    original = rollout_branch_batch(model, batch).predicted
    changed = _batch()
    changed.history_latents = batch.history_latents
    changed.history_actions = batch.history_actions
    changed.branch_actions = batch.branch_actions
    changed.branch_latents[:, :, 0] = batch.branch_latents[:, :, 0]
    changed.branch_latents[:, :, 1:] = 10_000
    modified = rollout_branch_batch(model, changed).predicted
    assert torch.equal(original, modified)


def test_dwm_paper_spec_losses_and_inference_boundary():
    torch.manual_seed(223)
    first = torch.eye(4)
    aligned = symmetric_info_nce(first, first)
    permuted = symmetric_info_nce(first, first.roll(1, 0))
    assert aligned < permuted
    losses = dwm_auxiliary_losses(
        prediction=torch.tensor([[1.0, 1.0], [1.0, -1.0]]),
        world=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        perturbed_world=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    )
    assert torch.isfinite(losses.world_contrastive)
    assert losses.orthogonality == 0
    model = DWMOutputBaseline(_vanilla(), world_head_hidden_dim=16)
    assert model.implementation_status == DWM_IMPLEMENTATION_STATUS
    assert model.inference_predictor() is model.predictor
    assert model.parameter_count() > model.parameter_count(inference_only=True)


def test_curriculum_advances_only_after_budget_and_plateau():
    curriculum = HorizonCurriculum(
        (1, 2, 4), minimum_steps=2, plateau_evaluations=2
    )
    curriculum.observe(10.0)
    curriculum.step()
    assert not curriculum.observe(9.0)
    curriculum.step()
    assert not curriculum.observe(9.0)
    assert curriculum.observe(9.0)
    assert curriculum.current_horizon == 2


def test_curriculum_replays_shorter_horizons_at_registered_fraction():
    curriculum = HorizonCurriculum((1, 2, 4), minimum_steps=1)
    curriculum.level_index = 2
    generator = torch.Generator().manual_seed(3072)
    sampled = [curriculum.sample_horizon(generator) for _ in range(4000)]
    short_fraction = sum(horizon < 4 for horizon in sampled) / len(sampled)
    assert 0.22 < short_fraction < 0.28
    assert set(sampled) == {1, 2, 4}


def _record(method, **overrides):
    payload = dict(
        method=method,
        seed=0,
        encoder_sha256="encoder",
        normalizer_sha256="normalizer",
        dataset_sha256="dataset",
        split_sha256="split",
        branch_trajectories=128,
        optimizer_steps=20,
        effective_batch_size=8,
        curriculum=(1, 2, 4),
        predictor_parameters=1000,
        paired_loss=method == "b6",
        implementation_status=(
            "paper_spec_reimplementation" if method == "b3" else "native"
        ),
    )
    payload.update(overrides)
    return RunRecord(**payload)


def test_fairness_audit_passes_complete_matrix_and_rejects_mismatch():
    records = [_record(method) for method in ("b2", "b3", "b4", "b6")]
    assert audit_run_matrix(records)["status"] == "pass"
    bad = records[:-1] + [_record("b6", dataset_sha256="wrong")]
    with pytest.raises(FairnessViolation, match="dataset_sha256"):
        audit_run_matrix(bad)
    bad_parameters = records[:-1] + [_record("b6", predictor_parameters=1200)]
    with pytest.raises(FairnessViolation, match="parameter"):
        audit_run_matrix(bad_parameters)


def test_ridge_matrix_probe_recovers_linear_action():
    torch.manual_seed(227)
    features = torch.randn(64, 5)
    targets = features @ torch.randn(5, 3)
    probe = fit_ridge_probe(features[:48], targets[:48])
    assert probe_r2(probe, features[48:], targets[48:]) > 0.999


@pytest.mark.parametrize("method", ["b2", "b3", "b4", "b6"])
def test_unified_trainer_all_methods_have_finite_gradients(method):
    torch.manual_seed(229)
    if method == "b2":
        model = _vanilla()
    elif method == "b3":
        model = DWMOutputBaseline(_vanilla(), world_head_hidden_dim=16)
    else:
        model = _counterfactual(centered=method == "b6")
    trainer = M4Trainer(
        model,
        M4TrainerConfig(
            method=method,
            max_steps=3,
            curriculum_levels=(1, 2),
            minimum_horizon_steps=1,
            warmup_steps=0,
            stage_b_freeze_steps=1,
        ),
        effect_threshold=0.1,
    )
    record = trainer.train_step(_batch())
    assert torch.isfinite(torch.tensor(record["total"]))
    assert record["gradient_norm"] >= 0


def test_m4_checkpoint_resume_restores_step_optimizer_and_predictions(tmp_path):
    torch.manual_seed(233)
    trainer = M4Trainer(
        _vanilla(),
        M4TrainerConfig(
            method="b2",
            max_steps=4,
            curriculum_levels=(1, 2),
            minimum_horizon_steps=1,
            warmup_steps=0,
        ),
        effect_threshold=0.1,
    )
    trainer.train_step(_batch())
    generator_state = trainer.generator.get_state()
    expected_generator_sample = torch.rand(4, generator=trainer.generator)
    trainer.generator.set_state(generator_state)
    path = tmp_path / "m4.pt"
    data = {"dataset_sha256": "data", "normalizer_sha256": "norm"}
    encoder = {"encoder_sha256": "encoder"}
    trainer.save(path, data_provenance=data, encoder_provenance=encoder)
    restored = M4Trainer.resume(
        path,
        device="cpu",
        effect_threshold=0.1,
        expected_provenance={**data, **encoder},
    )
    assert restored.global_step == trainer.global_step == 1
    assert len(restored.optimizer.state) == len(trainer.optimizer.state)
    actual_generator_sample = torch.rand(4, generator=restored.generator)
    assert torch.equal(actual_generator_sample, expected_generator_sample)
    trainer.model.eval()
    restored.model.eval()
    batch = _batch()
    expected = rollout_branch_batch(trainer.model, batch).predicted
    actual = rollout_branch_batch(restored.model, batch).predicted
    assert torch.equal(expected, actual)
    payload = json.loads(json.dumps({"step": restored.global_step}))
    assert payload["step"] == 1


def test_horizon_transition_restores_best_without_rewinding_budget_or_scheduler(tmp_path):
    torch.manual_seed(239)
    trainer = M4Trainer(
        _vanilla(),
        M4TrainerConfig(
            method="b2",
            max_steps=4,
            curriculum_levels=(1, 2),
            minimum_horizon_steps=1,
            warmup_steps=0,
        ),
        effect_threshold=0.1,
    )
    trainer.train_step(_batch())
    data = {"dataset_sha256": "data", "normalizer_sha256": "norm"}
    encoder = {"encoder_sha256": "encoder"}
    checkpoint = tmp_path / "best_h1.pt"
    trainer.save(checkpoint, data_provenance=data, encoder_provenance=encoder)
    expected = {
        name: value.detach().clone() for name, value in trainer.model.state_dict().items()
    }
    trainer.train_step(_batch())
    preserved_step = trainer.global_step
    preserved_scheduler = copy.deepcopy(trainer.scheduler.state_dict())
    preserved_lrs = [group["lr"] for group in trainer.optimizer.param_groups]

    audit = restore_best_for_horizon_transition(
        trainer,
        checkpoint,
        expected_provenance={**data, **encoder},
    )

    assert trainer.global_step == preserved_step == 2
    assert trainer.scheduler.state_dict() == preserved_scheduler
    assert [group["lr"] for group in trainer.optimizer.param_groups] == preserved_lrs
    assert audit["source_global_step"] == 1
    for name, value in trainer.model.state_dict().items():
        assert torch.equal(value, expected[name])


@pytest.mark.parametrize("method", ["b2", "b6"])
def test_mixed_precision_free_running_reenters_persistent_state_dtype(method):
    model = _vanilla() if method == "b2" else _counterfactual()
    trainer = M4Trainer(
        model,
        M4TrainerConfig(
            method=method,
            max_steps=2,
            curriculum_levels=(2,),
            minimum_horizon_steps=1,
            warmup_steps=0,
            precision="bfloat16",
        ),
        effect_threshold=0.1,
    )
    try:
        record = trainer.train_step(_batch(horizon=2))
    except RuntimeError as error:
        pytest.skip(f"this torch backend lacks CPU bfloat16 support: {error}")
    assert record["horizon"] == 2
    assert torch.isfinite(torch.tensor(record["total"]))
