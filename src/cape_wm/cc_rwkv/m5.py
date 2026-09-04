from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from cape_wm.stats import BootstrapInterval, paired_bootstrap_difference

from .metrics import DEFAULT_EFFECT_PAIRS
from .trainer import M4Trainer
from .training import BranchBatch, rollout_branch_batch

M5_SCHEMA_VERSION = "cc_rwkv_m5_v2"
REQUIRED_METHODS = frozenset({"b2", "b3", "b4", "b6"})
REQUIRED_SEEDS = frozenset({0, 1, 2})


def _auc(values: Tensor) -> Tensor:
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("sample curves must have shape [samples, horizon]")
    if values.shape[1] == 1:
        return values[:, 0]
    return torch.trapezoid(values, dx=1.0, dim=1) / (values.shape[1] - 1)


@torch.no_grad()
def sample_rollout_records(
    trainer: M4Trainer,
    batches: Iterable[BranchBatch],
    *,
    horizon: int,
    action_block: int,
) -> list[dict[str, Any]]:
    """Evaluate paired samples without losing sample IDs or seed pairing."""

    if horizon <= 0 or action_block <= 0:
        raise ValueError("horizon and action_block must be positive")
    trainer.model.eval()
    records: list[dict[str, Any]] = []
    for batch in batches:
        used_horizon = min(horizon, batch.horizon)
        rollout = rollout_branch_batch(
            trainer.model,
            batch.truncate(used_horizon),
            return_diagnostics=False,
        )
        predicted, target = rollout.predicted, rollout.target
        predicted_effect = torch.stack(
            [predicted[:, left] - predicted[:, right] for left, right in DEFAULT_EFFECT_PAIRS],
            dim=1,
        )
        true_effect = torch.stack(
            [target[:, left] - target[:, right] for left, right in DEFAULT_EFFECT_PAIRS],
            dim=1,
        )
        effect_rmse = (predicted_effect - true_effect).square().mean(dim=-1).sqrt().mean(dim=1)
        true_effect_norm = true_effect.norm(dim=-1).mean(dim=1)
        effect_valid = true_effect.norm(dim=-1) > trainer.effect_threshold
        trajectory_rmse = (predicted - target).square().mean(dim=-1).sqrt().mean(dim=1)
        cee_auc = _auc(effect_rmse)
        trajectory_auc = _auc(trajectory_rmse)
        reference_one_step = (predicted[:, 2, 0] - target[:, 2, 0]).square().mean(dim=-1).sqrt()
        one_step = (predicted[:, :, 0] - target[:, :, 0]).square().mean(dim=-1).sqrt().mean(dim=1)
        for index, sample_id in enumerate(batch.sample_ids):
            record = {
                "schema_version": M5_SCHEMA_VERSION,
                "sample_id": sample_id,
                "model_horizon": used_horizon,
                "primitive_horizon": used_horizon * action_block,
                "cee": effect_rmse[index].cpu().tolist(),
                "trajectory_rmse": trajectory_rmse[index].cpu().tolist(),
                "true_effect_norm": true_effect_norm[index].cpu().tolist(),
                "effect_valid": effect_valid[index].cpu().tolist(),
                "effect_valid_fraction": float(effect_valid[index].float().mean()),
                "effect_threshold": trainer.effect_threshold,
                "cee_auc": float(cee_auc[index]),
                "trajectory_auc": float(trajectory_auc[index]),
                "one_step_rmse": float(one_step[index]),
                "reference_one_step_rmse": float(reference_one_step[index]),
                "rollout_endpoint_rmse": float(trajectory_rmse[index, -1]),
            }
            for primitive_horizon in (20, 50):
                if primitive_horizon % action_block == 0:
                    model_index = primitive_horizon // action_block - 1
                    if model_index < used_horizon:
                        record[f"rollout_at_{primitive_horizon}_rmse"] = float(
                            trajectory_rmse[index, model_index]
                        )
            records.append(record)
    if len({record["sample_id"] for record in records}) != len(records):
        raise ValueError("sample IDs must be unique within one evaluation run")
    return records


def action_entry_audit(model: nn.Module) -> dict[str, Any]:
    """Structural audit: CC raw-action linear consumers must all be update networks."""

    predictor = getattr(model, "predictor", model)
    action_dim = int(predictor.config.action_dim)
    consumers = [
        name
        for name, module in predictor.named_modules()
        if isinstance(module, nn.Linear) and module.in_features == action_dim
    ]
    centered = bool(getattr(predictor.config, "centered", False))
    allowed = bool(consumers) and all(
        "time_mix.action_parameter_network.action_projection" in name for name in consumers
    )
    return {
        "centered_model": centered,
        "raw_action_consumers": consumers,
        "has_top_level_action_projection": hasattr(predictor, "action_projection"),
        "action_update_only": centered and allowed and not hasattr(predictor, "action_projection"),
    }


def write_sample_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def read_sample_records(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


@dataclass(frozen=True, slots=True)
class M5Run:
    task: str
    method: str
    seed: int
    effect_weight: float
    action_block: int
    data_samples: int
    trained_max_primitive_horizon: int
    max_primitive_horizon: int
    fairness: str
    split: str
    summary: dict[str, Any]
    samples: tuple[dict[str, Any], ...]
    mechanism: dict[str, Any]
    source: str
    formal_training_protocol: dict[str, Any] = field(default_factory=dict)


def load_m5_run(path: str | Path) -> M5Run:
    root = Path(path)
    metadata = json.loads((root / "m5_run.json").read_text())
    records = tuple(read_sample_records(root / "sample_metrics.jsonl"))
    if len(records) != metadata["test_samples"]:
        raise ValueError(f"sample count mismatch in {root}")
    return M5Run(
        task=str(metadata["task"]),
        method=str(metadata["method"]),
        seed=int(metadata["seed"]),
        effect_weight=float(metadata["effect_weight"]),
        action_block=int(metadata["action_block"]),
        data_samples=int(metadata["data_samples"]),
        trained_max_primitive_horizon=int(metadata["trained_max_primitive_horizon"]),
        max_primitive_horizon=int(metadata["max_primitive_horizon"]),
        fairness=str(metadata["fairness"]),
        split=str(metadata["split"]),
        summary=json.loads((root / "summary.json").read_text()),
        samples=records,
        mechanism=json.loads((root / "mechanism_audit.json").read_text()),
        source=str(root),
        formal_training_protocol=metadata.get(
            "formal_training_protocol", {"status": "not_recorded"}
        ),
    )


def _interval(value: BootstrapInterval) -> dict[str, float]:
    return asdict(value)


def _paired_values(
    treatment: list[M5Run], baseline: list[M5Run], metric: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Average training seeds per sample, then bootstrap independent sample IDs."""

    def collect(runs: list[M5Run]) -> dict[str, list[float]]:
        output: dict[str, list[float]] = defaultdict(list)
        for run in runs:
            for record in run.samples:
                output[str(record["sample_id"])].append(float(record[metric]))
        return output

    left, right = collect(treatment), collect(baseline)
    if set(left) != set(right):
        raise ValueError("paired methods do not contain identical sample IDs")
    expected_repeats = len(treatment)
    if any(len(values) != expected_repeats for values in left.values()):
        raise ValueError("treatment sample IDs are not shared across all seeds")
    if any(len(values) != len(baseline) for values in right.values()):
        raise ValueError("baseline sample IDs are not shared across all seeds")
    identifiers = sorted(left)
    return (
        np.asarray([np.mean(left[key]) for key in identifiers]),
        np.asarray([np.mean(right[key]) for key in identifiers]),
        identifiers,
    )


def select_effect_weight_across_seeds(
    runs: list[M5Run], *, maximum_one_step_degradation: float = 0.03
) -> dict[str, Any]:
    b2 = [run for run in runs if run.method == "b2"]
    candidates = [run for run in runs if run.method == "b6"]
    if {run.seed for run in b2} != REQUIRED_SEEDS:
        raise ValueError("effect selection requires B2 seeds 0/1/2")
    grouped: dict[float, list[M5Run]] = defaultdict(list)
    for run in candidates:
        grouped[run.effect_weight].append(run)
    rows = []
    b2_one_step = float(np.mean([run.summary["one_step_rmse"] for run in b2]))
    for weight in (0.5, 1.0, 2.0):
        group = grouped.get(weight, [])
        if {run.seed for run in group} != REQUIRED_SEEDS:
            raise ValueError(f"effect weight {weight} requires seeds 0/1/2")
        one_step = float(np.mean([run.summary["one_step_rmse"] for run in group]))
        row = {
            "effect_weight": weight,
            "validation_score": float(np.mean([run.summary["validation_score"] for run in group])),
            "one_step_error": one_step,
            "eligible": one_step <= b2_one_step * (1 + maximum_one_step_degradation),
        }
        rows.append(row)
    eligible = [row for row in rows if row["eligible"]]
    return {
        "status": "selected" if eligible else "no_eligible_candidate",
        "b2_one_step_error": b2_one_step,
        "maximum_one_step_degradation": maximum_one_step_degradation,
        "candidates": rows,
        "selected_effect_weight": (
            min(eligible, key=lambda row: row["validation_score"])["effect_weight"]
            if eligible
            else None
        ),
    }


def aggregate_gate_ab(
    runs: list[M5Run],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 3072,
) -> dict[str, Any]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    tasks = sorted({run.task for run in runs})
    result: dict[str, Any] = {
        "schema_version": M5_SCHEMA_VERSION,
        "tasks": {},
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }
    for task in tasks:
        task_runs = [run for run in runs if run.task == task]
        methods = {run.method for run in task_runs}
        seeds_by_method = {
            method: {run.seed for run in task_runs if run.method == method} for method in methods
        }
        completeness = {
            "required_methods": sorted(REQUIRED_METHODS),
            "observed_methods": sorted(methods),
            "required_seeds": sorted(REQUIRED_SEEDS),
            "seeds_by_method": {key: sorted(value) for key, value in seeds_by_method.items()},
            "gate_a_minimum_data_samples": 5000,
            "gate_a_minimum_primitive_horizon": 20,
            "gate_b_minimum_data_samples": 20000,
            "gate_b_minimum_primitive_horizon": 50,
        }
        common_reasons = []
        if methods != REQUIRED_METHODS:
            common_reasons.append("incomplete_method_matrix")
        if any(seeds_by_method.get(method, set()) != REQUIRED_SEEDS for method in REQUIRED_METHODS):
            common_reasons.append("incomplete_seed_matrix")
        if any(run.fairness != "pass" for run in task_runs):
            common_reasons.append("fairness_audit_not_passed")
        if any(
            run.formal_training_protocol.get("status") != "pass" for run in task_runs
        ):
            common_reasons.append("formal_training_protocol_audit_not_passed")

        gate_a_prerequisite_reasons = list(common_reasons)
        if any(run.data_samples < 5000 for run in task_runs):
            gate_a_prerequisite_reasons.append("dataset_below_5000_snapshots")
        if any(run.max_primitive_horizon < 20 for run in task_runs):
            gate_a_prerequisite_reasons.append("rollout_20_unavailable")
        if any(run.trained_max_primitive_horizon < 20 for run in task_runs):
            gate_a_prerequisite_reasons.append("training_did_not_reach_20_primitive_steps")
        gate_b_prerequisite_reasons = list(common_reasons)
        if any(run.data_samples < 20000 for run in task_runs):
            gate_b_prerequisite_reasons.append("dataset_below_20000_snapshots")
        if any(run.max_primitive_horizon < 50 for run in task_runs):
            gate_b_prerequisite_reasons.append("rollout_50_unavailable")
        if any(run.trained_max_primitive_horizon < 50 for run in task_runs):
            gate_b_prerequisite_reasons.append("training_did_not_reach_50_primitive_steps")

        # M5 is deliberately capped at 5,000 snapshots and 20 primitive steps.
        # Keep its preregistered candidate decision separate from the full Gate B,
        # which remains unavailable until the later 20,000/50 protocol.
        gate_b_at_20_prerequisite_reasons = list(gate_a_prerequisite_reasons)

        comparisons: dict[str, Any] = {}
        gate_b_reasons = list(gate_b_prerequisite_reasons)
        gate_b_at_20_reasons = list(gate_b_at_20_prerequisite_reasons)
        if not {"b3", "b4", "b6"} <= methods or any(
            seeds_by_method.get(method, set()) != REQUIRED_SEEDS for method in ("b3", "b4", "b6")
        ):
            gate_b_reasons.append("comparison_runs_missing")
            gate_b_at_20_reasons.append("comparison_runs_missing")
        else:
            b6 = [run for run in task_runs if run.method == "b6"]
            baseline_means = {
                method: float(
                    np.mean([run.summary["cee_auc"] for run in task_runs if run.method == method])
                )
                for method in ("b3", "b4")
            }
            strongest = min(baseline_means, key=baseline_means.get)
            baseline = [run for run in task_runs if run.method == strongest]
            metrics = [
                "cee_auc",
                "trajectory_auc",
                "one_step_rmse",
                "rollout_endpoint_rmse",
            ]
            for primitive_horizon in (20, 50):
                name = f"rollout_at_{primitive_horizon}_rmse"
                if all(name in record for run in (*b6, *baseline) for record in run.samples):
                    metrics.append(name)
            for metric in metrics:
                left, right, identifiers = _paired_values(b6, baseline, metric)
                interval = paired_bootstrap_difference(
                    left, right, resamples=bootstrap_samples, seed=bootstrap_seed
                )
                comparisons[metric] = {
                    "b6_mean": float(left.mean()),
                    "baseline_mean": float(right.mean()),
                    "difference_b6_minus_baseline": _interval(interval),
                    "paired_sample_ids": len(identifiers),
                }
            cee = comparisons["cee_auc"]
            relative = (cee["baseline_mean"] - cee["b6_mean"]) / max(
                abs(cee["baseline_mean"]), 1e-12
            )
            comparisons["strongest_baseline"] = strongest
            comparisons["cee_relative_improvement"] = relative
            if relative < 0.15:
                gate_b_reasons.append("cee_improvement_below_15pct")
                gate_b_at_20_reasons.append("cee_improvement_below_15pct")
            if cee["difference_b6_minus_baseline"]["high"] >= 0:
                gate_b_reasons.append("cee_bootstrap_ci_crosses_zero")
                gate_b_at_20_reasons.append("cee_bootstrap_ci_crosses_zero")
            one_step = comparisons["one_step_rmse"]
            if one_step["b6_mean"] > one_step["baseline_mean"] * 1.03:
                gate_b_reasons.append("one_step_degradation_above_3pct")
                gate_b_at_20_reasons.append("one_step_degradation_above_3pct")
            for primitive_horizon in (20, 50):
                name = f"rollout_at_{primitive_horizon}_rmse"
                if name not in comparisons:
                    gate_b_reasons.append(f"rollout_{primitive_horizon}_unavailable")
                    if primitive_horizon == 20:
                        gate_b_at_20_reasons.append("rollout_20_unavailable")
                elif comparisons[name]["difference_b6_minus_baseline"]["high"] >= 0:
                    gate_b_reasons.append(f"rollout_{primitive_horizon}_not_improved")
                    if primitive_horizon == 20:
                        gate_b_at_20_reasons.append("rollout_20_not_improved")

        b6_runs = [run for run in task_runs if run.method == "b6"]
        gate_a_checks = {
            "no_op_tolerance": bool(b6_runs)
            and all(
                run.mechanism.get("actual_equals_reference_max_delta", 1.0) < 1e-7
                for run in b6_runs
            ),
            "action_update_only": bool(b6_runs)
            and all(
                run.mechanism.get("action_entry", {}).get("action_update_only", False)
                for run in b6_runs
            ),
            "probe_above_mean_baseline": bool(b6_runs)
            and all(
                run.mechanism.get("probe", {})
                .get("probe_mse_minus_mean_baseline_ci", {})
                .get("high", 0.0)
                < 0
                for run in b6_runs
            ),
            "gate_not_collapsed": bool(b6_runs)
            and all(
                run.summary.get("gate", {}).get("saturated_fraction", 1.0) < 0.95 for run in b6_runs
            ),
        }
        if {"b2", "b6"} <= methods and all(
            seeds_by_method.get(method, set()) == REQUIRED_SEEDS for method in ("b2", "b6")
        ):
            b2 = [run for run in task_runs if run.method == "b2"]
            left, right, _ = _paired_values(b6_runs, b2, "reference_one_step_rmse")
            gate_a_checks["world_noop_noninferior_to_b2"] = bool(left.mean() <= right.mean() * 1.03)
        else:
            gate_a_checks["world_noop_noninferior_to_b2"] = False
        gate_a_reasons = list(gate_a_prerequisite_reasons)
        gate_a_reasons.extend(
            f"failed_{name}" for name, passed in gate_a_checks.items() if not passed
        )
        effect_rows = [record for run in b6_runs for record in run.samples]
        effect_mask_audit = {
            "records": len(effect_rows),
            "mean_valid_fraction": (
                float(
                    np.mean(
                        [
                            float(record["effect_valid_fraction"])
                            for record in effect_rows
                            if "effect_valid_fraction" in record
                        ]
                    )
                )
                if any("effect_valid_fraction" in record for record in effect_rows)
                else None
            ),
            "zero_valid_records": sum(
                float(record.get("effect_valid_fraction", 1.0)) == 0.0 for record in effect_rows
            ),
            "thresholds": sorted(
                {
                    float(record["effect_threshold"])
                    for record in effect_rows
                    if "effect_threshold" in record
                }
            ),
        }
        gate_a_assessable = not gate_a_prerequisite_reasons
        gate_b_at_20_assessable = not gate_b_at_20_prerequisite_reasons
        gate_b_assessable = not gate_b_prerequisite_reasons
        result["tasks"][task] = {
            "completeness": completeness,
            "comparisons": comparisons,
            "effect_mask_audit": effect_mask_audit,
            "gate_a": {
                "status": "PASS"
                if gate_a_assessable and not gate_a_reasons
                else ("FAIL" if gate_a_assessable else "NOT_ASSESSABLE"),
                "checks": gate_a_checks,
                "reasons": gate_a_reasons,
            },
            "gate_b_at_20": {
                "status": "PASS"
                if gate_b_at_20_assessable and not gate_b_at_20_reasons
                else ("FAIL" if gate_b_at_20_assessable else "NOT_ASSESSABLE"),
                "provisional": True,
                "reasons": list(dict.fromkeys(gate_b_at_20_reasons)),
            },
            "gate_b": {
                "status": "PASS"
                if gate_b_assessable and not gate_b_reasons
                else ("FAIL" if gate_b_assessable else "NOT_ASSESSABLE"),
                "reasons": list(dict.fromkeys(gate_b_reasons)),
            },
        }
    statuses = [
        result["tasks"][task][gate]["status"] for task in tasks for gate in ("gate_a", "gate_b")
    ]
    result["overall_status"] = (
        "PASS"
        if statuses and all(status == "PASS" for status in statuses)
        else "FAIL"
        if statuses and all(status != "NOT_ASSESSABLE" for status in statuses)
        else "NOT_ASSESSABLE"
    )
    m5_at_20_statuses = [
        result["tasks"][task][gate]["status"]
        for task in tasks
        for gate in ("gate_a", "gate_b_at_20")
    ]
    result["m5_at_20_overall_status"] = (
        "PASS"
        if m5_at_20_statuses and all(status == "PASS" for status in m5_at_20_statuses)
        else "FAIL"
        if m5_at_20_statuses and all(status != "NOT_ASSESSABLE" for status in m5_at_20_statuses)
        else "NOT_ASSESSABLE"
    )
    return result
