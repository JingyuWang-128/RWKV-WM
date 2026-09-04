from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


M5_SCHEMA = "cc_rwkv_m5_v2"
AGGREGATE_SCHEMA = "cc_rwkv_m5_aggregate_v2"
REQUIRED_SEEDS = (0, 1, 2)
METHOD_WEIGHTS = {
    "b2": (1.0,),
    "b3": (1.0,),
    "b4": (1.0,),
    "b6": (0.5, 1.0, 2.0),
}
EVALUATION_FILES = (
    "m5_run.json",
    "sample_metrics.jsonl",
    "mechanism_audit.json",
    "metrics.jsonl",
    "summary.json",
    "resolved_config.yaml",
    "provenance.json",
    "failed_samples.jsonl",
    "gate_report.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit formal CC-RWKV M5 completion")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tworoom-manifest", type=Path, required=True)
    parser.add_argument("--action-delay-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=3072)
    return parser.parse_args()


def _weight_tag(weight: float) -> str:
    if weight == 1.0:
        return "1"
    return str(weight).replace(".", "p")


def _training_dir(task_root: Path, seed: int, method: str, weight: float) -> Path:
    if method != "b6" or weight == 1.0:
        return task_root / f"seed_{seed}" / "train" / method
    return task_root / f"seed_{seed}" / f"train_b6_w{_weight_tag(weight)}" / "b6"


def _evaluation_dir(
    task_root: Path, seed: int, split: str, method: str, weight: float
) -> Path:
    return task_root / f"seed_{seed}" / split / f"{method}_w{_weight_tag(weight)}"


def _read_json(path: Path, failures: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        failures.append(f"missing:{path}")
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        failures.append(f"invalid_json:{path}:{error}")
        return None
    if not isinstance(value, dict):
        failures.append(f"json_not_object:{path}")
        return None
    return value


def _finite_history(payload: dict[str, Any]) -> bool:
    for row in payload.get("history", []):
        if not isinstance(row, dict):
            return False
        for value in row.values():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                return False
    return True


def _audit_training(
    directory: Path,
    *,
    method: str,
    seed: int,
    weight: float,
    max_steps: int,
    final_model_horizon: int,
    curriculum: tuple[int, ...],
    failures: list[str],
) -> dict[str, Any]:
    latest = directory / "latest.pt"
    selected = directory / f"best_rollout_h{final_model_horizon}.pt"
    evidence: dict[str, Any] = {
        "directory": str(directory),
        "latest": str(latest),
        "selected": str(selected),
    }
    if not selected.is_file():
        failures.append(f"missing:{selected}")
    if not latest.is_file():
        failures.append(f"missing:{latest}")
        return evidence
    try:
        payload = torch.load(latest, map_location="cpu", weights_only=True)
    except Exception as error:  # checkpoint corruption must be surfaced verbatim
        failures.append(f"invalid_checkpoint:{latest}:{error}")
        return evidence
    global_step = int(payload.get("global_step", -1))
    stored_method = payload.get("method")
    level_index = int(payload.get("curriculum", {}).get("level_index", -1))
    scheduler_step = int(payload.get("scheduler", {}).get("last_epoch", -1))
    stored = tuple(
        int(item)
        for item in payload.get("training_config", {})
        .get("trainer", {})
        .get("curriculum_levels", [])
    )
    evidence.update(
        {
            "global_step": global_step,
            "method": stored_method,
            "curriculum": list(stored),
            "curriculum_level_index": level_index,
            "scheduler_last_epoch": scheduler_step,
            "history_rows": len(payload.get("history", [])),
        }
    )
    if stored_method != method:
        failures.append(f"method_mismatch:{latest}:{stored_method}!={method}")
    if global_step != max_steps:
        failures.append(f"optimizer_budget:{latest}:{global_step}!={max_steps}")
    if scheduler_step != max_steps:
        failures.append(f"scheduler_budget:{latest}:{scheduler_step}!={max_steps}")
    if stored != curriculum or level_index != len(curriculum) - 1:
        failures.append(f"curriculum_incomplete:{latest}:{stored}:{level_index}")
    if len(payload.get("history", [])) != max_steps or not _finite_history(payload):
        failures.append(f"training_history_invalid:{latest}")
    transition_path = directory / "curriculum_transitions.jsonl"
    transitions = (
        [json.loads(line) for line in transition_path.read_text().splitlines() if line]
        if transition_path.is_file()
        else []
    )
    expected_transitions = list(zip(curriculum[:-1], curriculum[1:]))
    observed = [
        (int(row.get("from_horizon", -1)), int(row.get("to_horizon", -1)))
        for row in transitions
    ]
    evidence["transitions"] = observed
    if observed != expected_transitions or any(
        not row.get("restored_model")
        or not row.get("restored_optimizer")
        or not row.get("preserved_scheduler")
        for row in transitions
    ):
        failures.append(f"transition_audit_failed:{transition_path}")
    if method == "b6":
        initialization = _read_json(directory / "initialization.json", failures)
        if initialization is not None:
            evidence["initialization"] = {
                "scheme": initialization.get("scheme"),
                "source_seed": initialization.get("source_seed"),
                "target_seed": initialization.get("target_seed"),
                "effect_weight": initialization.get("effect_weight"),
                "copied_tensor_count": initialization.get("copied_tensor_count"),
            }
            if (
                initialization.get("scheme") != "b2_final_horizon_rollout_best"
                or int(initialization.get("source_seed", -1)) != seed
                or int(initialization.get("target_seed", -1)) != seed
                or float(initialization.get("effect_weight", -1.0)) != weight
                or int(initialization.get("copied_tensor_count", 0)) <= 0
            ):
                failures.append(f"b6_initialization_invalid:{directory}")
    return evidence


def _audit_evaluation(
    directory: Path,
    *,
    task: str,
    seed: int,
    split: str,
    method: str,
    weight: float,
    manifest: dict[str, Any],
    failures: list[str],
) -> tuple[dict[str, Any], set[str]]:
    for name in EVALUATION_FILES:
        if not (directory / name).is_file():
            failures.append(f"missing:{directory / name}")
    run = _read_json(directory / "m5_run.json", failures)
    provenance = _read_json(directory / "provenance.json", failures)
    mechanism = _read_json(directory / "mechanism_audit.json", failures)
    sample_path = directory / "sample_metrics.jsonl"
    samples: list[dict[str, Any]] = []
    if sample_path.is_file():
        try:
            samples = [json.loads(line) for line in sample_path.read_text().splitlines() if line]
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"invalid_sample_metrics:{sample_path}:{error}")
    expected_samples = int(manifest["split_counts"][split])
    sample_ids = {str(row.get("sample_id")) for row in samples}
    if len(samples) != expected_samples or len(sample_ids) != expected_samples:
        failures.append(
            f"sample_count_or_uniqueness:{sample_path}:{len(samples)}:{len(sample_ids)}"
        )
    evidence: dict[str, Any] = {
        "directory": str(directory),
        "m5_run_present": run is not None,
        "sample_rows": len(samples),
        "unique_sample_ids": len(sample_ids),
    }
    if run is not None:
        checks = {
            "schema_version": run.get("schema_version") == M5_SCHEMA,
            "task": run.get("task") == task,
            "method": run.get("method") == method,
            "seed": int(run.get("seed", -1)) == seed,
            "effect_weight": float(run.get("effect_weight", -1.0))
            == (weight if method == "b6" else 0.0),
            "split": run.get("split") == split,
            "test_samples": int(run.get("test_samples", -1)) == expected_samples,
            "data_samples": int(run.get("data_samples", -1)) == int(manifest["samples"]),
            "action_block": int(run.get("action_block", -1)) == int(manifest["action_block"]),
            "trained_horizon": int(run.get("trained_max_primitive_horizon", -1)) >= 20,
            "evaluated_horizon": int(run.get("max_primitive_horizon", -1)) == 20,
            "fairness": run.get("fairness") == "pass",
            "protocol": run.get("formal_training_protocol", {}).get("status") == "pass",
        }
        evidence["run_checks"] = checks
        for name, passed in checks.items():
            if not passed:
                failures.append(f"run_check:{directory}:{name}")
    if provenance is not None:
        observed_sha = provenance.get("data", {}).get("dataset_sha256")
        if observed_sha != manifest["branch_dataset_sha256"]:
            failures.append(f"dataset_provenance:{directory}:{observed_sha}")
    if method == "b6" and mechanism is not None:
        no_op = mechanism.get("actual_equals_reference_max_delta")
        entry = mechanism.get("action_entry", {})
        gate = mechanism.get("gate") or {}
        if no_op is None or float(no_op) >= 1e-7:
            failures.append(f"b6_no_op_invariant:{directory}:{no_op}")
        if not entry.get("action_update_only"):
            failures.append(f"b6_action_entry:{directory}")
        if not gate or float(gate.get("saturated_fraction", 1.0)) >= 1.0:
            failures.append(f"b6_gate_invalid:{directory}")
    return evidence, sample_ids


def main() -> None:
    args = parse_args()
    task_specs = {
        "tworoom_w": {
            "manifest_path": args.tworoom_manifest,
            "max_steps": 30_000,
            "curriculum": (1, 2, 4),
        },
        "action_delay": {
            "manifest_path": args.action_delay_manifest,
            "max_steps": 40_000,
            "curriculum": (1, 5, 10, 20),
        },
    }
    failures: list[str] = []
    inventory: list[dict[str, Any]] = []
    sample_sets: dict[tuple[str, str], list[tuple[str, set[str]]]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for task, spec in task_specs.items():
        manifest = _read_json(spec["manifest_path"], failures)
        if manifest is None:
            continue
        manifests[task] = manifest
        manifest_checks = {
            "variant": manifest.get("variant") == task,
            "samples": int(manifest.get("samples", -1)) == 5000,
            "splits": manifest.get("split_counts")
            == {"train": 4000, "validation": 500, "test": 500},
            "primitive_horizon": int(manifest.get("primitive_horizon", -1)) == 20,
            "restore_failures": int(manifest.get("restore_failure_count", -1)) == 0,
        }
        for name, passed in manifest_checks.items():
            if not passed:
                failures.append(f"manifest_check:{task}:{name}")
        task_root = args.root / task
        final_model_horizon = int(manifest["model_horizon"])
        for seed in REQUIRED_SEEDS:
            for method, weights in METHOD_WEIGHTS.items():
                for weight in weights:
                    training = _audit_training(
                        _training_dir(task_root, seed, method, weight),
                        method=method,
                        seed=seed,
                        weight=weight,
                        max_steps=int(spec["max_steps"]),
                        final_model_horizon=final_model_horizon,
                        curriculum=tuple(spec["curriculum"]),
                        failures=failures,
                    )
                    for split in ("validation", "test"):
                        evaluation, sample_ids = _audit_evaluation(
                            _evaluation_dir(task_root, seed, split, method, weight),
                            task=task,
                            seed=seed,
                            split=split,
                            method=method,
                            weight=weight,
                            manifest=manifest,
                            failures=failures,
                        )
                        label = f"{task}:seed{seed}:{method}:w{weight}:{split}"
                        if len(sample_ids) == int(manifest["split_counts"][split]):
                            sample_sets.setdefault((task, split), []).append(
                                (label, sample_ids)
                            )
                        inventory.append(
                            {
                                "task": task,
                                "seed": seed,
                                "method": method,
                                "effect_weight": weight,
                                "split": split,
                                "training": training,
                                "evaluation": evaluation,
                            }
                        )
    for (task, split), labelled_sets in sample_sets.items():
        if not labelled_sets:
            continue
        expected = labelled_sets[0][1]
        for label, observed in labelled_sets[1:]:
            if observed != expected:
                failures.append(f"sample_pairing:{task}:{split}:{label}")

    aggregate_path = args.root / "aggregate" / "summary.json"
    aggregate = _read_json(aggregate_path, failures)
    aggregate_evidence: dict[str, Any] = {"path": str(aggregate_path)}
    if aggregate is not None:
        gates = aggregate.get("gates", {})
        aggregate_evidence.update(
            {
                "schema_version": aggregate.get("schema_version"),
                "bootstrap_samples": gates.get("bootstrap_samples"),
                "bootstrap_seed": gates.get("bootstrap_seed"),
                "run_inventory": len(aggregate.get("run_inventory", [])),
                "m5_at_20_overall_status": gates.get("m5_at_20_overall_status"),
                "full_gate_overall_status": gates.get("overall_status"),
            }
        )
        if aggregate.get("schema_version") != AGGREGATE_SCHEMA:
            failures.append("aggregate_schema")
        if int(gates.get("bootstrap_samples", -1)) != args.bootstrap_samples:
            failures.append("aggregate_bootstrap_samples")
        if int(gates.get("bootstrap_seed", -1)) != args.bootstrap_seed:
            failures.append("aggregate_bootstrap_seed")
        if len(aggregate.get("run_inventory", [])) != len(inventory):
            failures.append("aggregate_inventory_size")
        m5_at_20_status = str(gates.get("m5_at_20_overall_status", "")).lower()
        if m5_at_20_status not in {"pass", "fail"}:
            failures.append("gate_ab_at_20_not_assessed")
        for task in task_specs:
            item = gates.get("tasks", {}).get(task, {})
            gate_a_status = str(item.get("gate_a", {}).get("status", "")).lower()
            if gate_a_status not in {"pass", "fail"}:
                failures.append(f"gate_a_not_assessed:{task}")
            gate_b_at_20_status = str(
                item.get("gate_b_at_20", {}).get("status", "")
            ).lower()
            if gate_b_at_20_status not in {"pass", "fail"}:
                failures.append(f"gate_b_at_20_not_assessed:{task}")
        for name in ("gate_report.md", "failed_samples.jsonl"):
            if not (args.root / "aggregate" / name).is_file():
                failures.append(f"missing:{args.root / 'aggregate' / name}")

    payload = {
        "schema_version": "cc_rwkv_m5_completion_audit_v1",
        "status": "pass" if not failures else "incomplete",
        "expected_evaluation_units": 72,
        "observed_evaluation_units": sum(
            int(item["evaluation"].get("m5_run_present", False)) for item in inventory
        ),
        "manifests": {
            task: {
                "path": str(task_specs[task]["manifest_path"]),
                "dataset_sha256": manifest["branch_dataset_sha256"],
                "samples": manifest["samples"],
                "split_counts": manifest["split_counts"],
                "primitive_horizon": manifest["primitive_horizon"],
            }
            for task, manifest in manifests.items()
        },
        "aggregate": aggregate_evidence,
        "failures": failures,
        "inventory": inventory,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in payload if key != "inventory"}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
