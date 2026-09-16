from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from cape_wm.cc_rwkv.m5 import M5Run, aggregate_gate_ab, select_effect_weight_across_seeds
from scripts.evaluate_cc_rwkv_m5 import audit_formal_training_protocol


def _run(
    method: str,
    seed: int,
    *,
    task: str = "tworoom_w",
    split: str = "test",
    weight: float = 1.0,
    cee: float | None = None,
    samples: int = 20000,
    horizon: int = 50,
) -> M5Run:
    cee = (0.5 if method == "b6" else 1.0) if cee is None else cee
    sample_rows = tuple(
        {
            "sample_id": sample_id,
            "cee_auc": cee + offset,
            "trajectory_auc": cee + offset,
            "one_step_rmse": 0.5 if method == "b6" else 0.51,
            "reference_one_step_rmse": 0.5 if method == "b6" else 0.51,
            "rollout_endpoint_rmse": cee + offset,
            "rollout_at_20_rmse": cee + offset,
            "rollout_at_50_rmse": cee + offset,
        }
        for sample_id, offset in (("a", 0.0), ("b", 0.1), ("c", -0.1))
    )
    return M5Run(
        task=task,
        method=method,
        seed=seed,
        effect_weight=weight,
        action_block=5,
        data_samples=samples,
        trained_max_primitive_horizon=horizon,
        max_primitive_horizon=horizon,
        fairness="pass",
        split=split,
        summary={
            "cee_auc": cee,
            "validation_score": cee,
            "one_step_rmse": 0.5 if method == "b6" else 0.51,
            "gate": {"saturated_fraction": 0.0} if method == "b6" else None,
        },
        samples=sample_rows,
        mechanism={
            "actual_equals_reference_max_delta": 0.0,
            "action_entry": {"action_update_only": method == "b6"},
            "probe": {
                "evaluation_r2": 0.5 if method == "b6" else -1.0,
                "probe_mse_minus_mean_baseline_ci": {
                    "estimate": -0.2,
                    "low": -0.3,
                    "high": -0.1,
                    "confidence": 0.95,
                },
            },
        },
        source=f"/{task}/{method}/{seed}/{weight}",
        formal_training_protocol={"status": "pass"},
    )


def test_effect_weight_selection_uses_all_seeds_and_one_step_guard():
    runs = [_run("b2", seed, split="validation") for seed in range(3)]
    for weight, score, one_step in ((0.5, 0.8, 0.50), (1.0, 0.6, 0.54), (2.0, 0.7, 0.51)):
        for seed in range(3):
            run = _run("b6", seed, split="validation", weight=weight, cee=score)
            run.summary["one_step_rmse"] = one_step
            runs.append(run)
    selection = select_effect_weight_across_seeds(runs)
    assert selection["selected_effect_weight"] == 2.0
    assert not selection["candidates"][1]["eligible"]


def test_gate_ab_uses_seed_averaged_paired_samples_and_passes_complete_matrix():
    runs = [_run(method, seed) for method in ("b2", "b3", "b4", "b6") for seed in range(3)]
    result = aggregate_gate_ab(runs, bootstrap_samples=1000)
    assert result["overall_status"] == "PASS"
    task = result["tasks"]["tworoom_w"]
    assert task["gate_a"]["status"] == "PASS"
    assert task["gate_b_at_20"]["status"] == "PASS"
    assert task["gate_b"]["status"] == "PASS"
    assert result["m5_at_20_overall_status"] == "PASS"
    assert task["comparisons"]["cee_auc"]["paired_sample_ids"] == 3


def test_m5_gate_b_at_20_is_assessable_without_claiming_full_gate_b():
    runs = [
        _run(method, seed, samples=5000, horizon=20)
        for method in ("b2", "b3", "b4", "b6")
        for seed in range(3)
    ]
    result = aggregate_gate_ab(runs, bootstrap_samples=100)
    task = result["tasks"]["tworoom_w"]
    assert task["gate_a"]["status"] == "PASS"
    assert task["gate_b_at_20"] == {
        "status": "PASS",
        "provisional": True,
        "reasons": [],
    }
    assert task["gate_b"]["status"] == "NOT_ASSESSABLE"
    assert result["m5_at_20_overall_status"] == "PASS"
    assert result["overall_status"] == "NOT_ASSESSABLE"


def test_gate_is_not_assessable_for_smoke_data_or_missing_rollout_50():
    runs = [
        _run(method, seed, samples=32, horizon=20)
        for method in ("b2", "b3", "b4", "b6")
        for seed in range(3)
    ]
    result = aggregate_gate_ab(runs, bootstrap_samples=100)
    assert result["overall_status"] == "NOT_ASSESSABLE"
    task = result["tasks"]["tworoom_w"]
    assert task["gate_b_at_20"]["status"] == "NOT_ASSESSABLE"
    assert "dataset_below_5000_snapshots" in task["gate_b_at_20"]["reasons"]
    reasons = task["gate_b"]["reasons"]
    assert "dataset_below_20000_snapshots" in reasons
    assert "rollout_50_unavailable" in reasons


def test_formal_protocol_requires_best_final_checkpoint_and_transition_audit(tmp_path):
    checkpoint = tmp_path / "best_rollout_h2.pt"
    checkpoint.touch()
    (tmp_path / "curriculum_transitions.jsonl").write_text(
        json.dumps(
            {
                "from_horizon": 1,
                "to_horizon": 2,
                "source_checkpoint": str(tmp_path / "best_h1.pt"),
                "source_global_step": 5,
                "preserved_global_step": 6,
                "restored_model": True,
                "restored_optimizer": True,
                "preserved_scheduler": True,
            }
        )
        + "\n"
    )
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            curriculum_levels=(1, 2), method="b2", seed=0, max_steps=10
        ),
        curriculum=SimpleNamespace(current_horizon=2),
    )
    torch.save(
        {"method": "b2", "global_step": 10, "curriculum": {"level_index": 1}},
        tmp_path / "latest.pt",
    )
    manifest = {"model_horizon": 2, "samples": 5000}
    assert audit_formal_training_protocol(
        SimpleNamespace(checkpoint=checkpoint), manifest, trainer
    )["status"] == "pass"

    checkpoint = tmp_path / "latest.pt"
    with pytest.raises(ValueError, match="checkpoint_is_not_final_horizon_best"):
        audit_formal_training_protocol(
            SimpleNamespace(checkpoint=checkpoint), manifest, trainer
        )
