from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cape_wm.cc_rwkv.evaluation import write_evaluation_artifacts
from cape_wm.cc_rwkv.m4_data import InMemoryBranchSplit, m4_data_provenance
from cape_wm.cc_rwkv.m5 import (
    M5_SCHEMA_VERSION,
    action_entry_audit,
    sample_rollout_records,
    write_sample_records,
)
from cape_wm.cc_rwkv.trainer import M4Trainer
from cape_wm.cc_rwkv.training import effect_norm_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M5 paired checkpoint evaluator")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=Path("artifacts/results/cc_rwkv/tworoom/m0/protocol/manifest.json"),
    )
    parser.add_argument("--task", choices=("tworoom_w", "action_delay"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--train-limit",
        type=int,
        help="Optional training-split cap used to fit the frozen effect threshold; "
        "the formal default uses the complete training split.",
    )
    parser.add_argument("--evaluation-limit", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fairness", choices=("pass", "fail", "not_run"), default="not_run")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def audit_formal_training_protocol(args, manifest: dict, trainer: M4Trainer) -> dict:
    final_model_horizon = int(manifest["model_horizon"])
    expected_checkpoint = f"best_rollout_h{final_model_horizon}.pt"
    levels = tuple(int(level) for level in trainer.config.curriculum_levels)
    expected_transitions = list(zip(levels[:-1], levels[1:]))
    transition_path = args.checkpoint.parent / "curriculum_transitions.jsonl"
    transitions = (
        [
            json.loads(line)
            for line in transition_path.read_text().splitlines()
            if line.strip()
        ]
        if transition_path.is_file()
        else []
    )
    latest_path = args.checkpoint.parent / "latest.pt"
    latest_payload = (
        torch.load(latest_path, map_location="cpu", weights_only=True)
        if latest_path.is_file()
        else None
    )
    observed_transitions = [
        (int(row["from_horizon"]), int(row["to_horizon"])) for row in transitions
    ]
    reasons = []
    if args.checkpoint.name != expected_checkpoint:
        reasons.append("checkpoint_is_not_final_horizon_best")
    if latest_payload is None:
        reasons.append("completed_training_checkpoint_missing")
    else:
        if latest_payload.get("method") != trainer.config.method:
            reasons.append("completed_training_checkpoint_method_mismatch")
        if int(latest_payload.get("global_step", -1)) != trainer.config.max_steps:
            reasons.append("optimizer_step_budget_incomplete")
        if int(latest_payload.get("curriculum", {}).get("level_index", -1)) != len(levels) - 1:
            reasons.append("completed_training_checkpoint_not_at_final_horizon")
    if trainer.curriculum.current_horizon != final_model_horizon:
        reasons.append("checkpoint_did_not_reach_final_model_horizon")
    if observed_transitions != expected_transitions:
        reasons.append("curriculum_transition_audit_incomplete")
    if any(
        not row.get("restored_model")
        or not row.get("restored_optimizer")
        or not row.get("preserved_scheduler")
        for row in transitions
    ):
        reasons.append("curriculum_best_restore_not_proven")
    if any(
        Path(str(row.get("source_checkpoint", ""))).name
        != f"best_h{int(row.get('from_horizon', -1))}.pt"
        or int(row.get("source_global_step", -1))
        > int(row.get("preserved_global_step", -1))
        for row in transitions
    ):
        reasons.append("curriculum_best_restore_source_invalid")

    initialization = None
    if trainer.config.method == "b6":
        initialization_path = args.checkpoint.parent / "initialization.json"
        if initialization_path.is_file():
            initialization = json.loads(initialization_path.read_text())
        if initialization is None:
            reasons.append("b6_b2_initialization_audit_missing")
        else:
            if initialization.get("scheme") != "b2_final_horizon_rollout_best":
                reasons.append("b6_b2_initialization_scheme_invalid")
            if int(initialization.get("source_seed", -1)) != trainer.config.seed:
                reasons.append("b6_b2_initialization_seed_mismatch")
            if int(initialization.get("target_seed", -1)) != trainer.config.seed:
                reasons.append("b6_target_seed_mismatch")
            if int(initialization.get("source_final_model_horizon", -1)) != final_model_horizon:
                reasons.append("b6_b2_initialization_horizon_mismatch")
            if (
                Path(str(initialization.get("source_checkpoint", ""))).name
                != expected_checkpoint
            ):
                reasons.append("b6_b2_initialization_checkpoint_not_final_best")
            if int(initialization.get("copied_tensor_count", 0)) <= 0:
                reasons.append("b6_b2_initialization_copied_nothing")

    audit = {
        "status": "pass" if not reasons else "fail",
        "reasons": reasons,
        "checkpoint_selection": "best_rollout_final_horizon"
        if args.checkpoint.name == expected_checkpoint
        else "invalid",
        "expected_checkpoint": expected_checkpoint,
        "observed_checkpoint": args.checkpoint.name,
        "expected_transitions": expected_transitions,
        "observed_transitions": observed_transitions,
        "transition_audit": str(transition_path),
        "completed_training_checkpoint": str(latest_path),
        "completed_optimizer_steps": (
            int(latest_payload["global_step"]) if latest_payload is not None else None
        ),
        "b6_initialization": initialization,
    }
    if int(manifest["samples"]) >= 5000 and reasons:
        raise ValueError(f"formal training protocol audit failed: {', '.join(reasons)}")
    return audit


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.data_manifest.read_text())
    if manifest["variant"] != args.task:
        raise ValueError("task does not match the branch dataset manifest")
    data_provenance, encoder_provenance = m4_data_provenance(
        args.data, args.data_manifest, args.protocol_manifest
    )
    train = InMemoryBranchSplit(args.data, split="train", limit=args.train_limit)
    evaluation = InMemoryBranchSplit(args.data, split=args.split, limit=args.evaluation_limit)
    threshold = effect_norm_threshold(train.tensors["branch_latents"])
    trainer = M4Trainer.resume(
        args.checkpoint,
        device=args.device,
        effect_threshold=threshold,
        expected_provenance={**data_provenance, **encoder_provenance},
    )
    protocol_audit = audit_formal_training_protocol(args, manifest, trainer)
    available = evaluation.tensors["branch_actions_raw"].shape[2]
    horizons = tuple(level for level in (1, 2, 4, 10, 20) if level <= available)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = write_evaluation_artifacts(
        args.output,
        trainer,
        list(evaluation.batches(args.batch_size, device=args.device)),
        list(train.batches(args.batch_size, device=args.device)),
        horizons=horizons,
        provenance={
            "data": data_provenance,
            "encoder": encoder_provenance,
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "m5_schema": M5_SCHEMA_VERSION,
        },
        fairness_status=args.fairness,
    )
    records = sample_rollout_records(
        trainer,
        evaluation.batches(args.batch_size, device=args.device),
        horizon=available,
        action_block=int(manifest["action_block"]),
    )
    write_sample_records(args.output / "sample_metrics.jsonl", records)
    mechanism = {
        "actual_equals_reference_max_delta": summary.get("actual_equals_reference_max_delta"),
        "action_entry": action_entry_audit(trainer.model),
        "probe": summary["probe"],
        "gate": summary.get("gate"),
        "formal_training_protocol": protocol_audit,
    }
    (args.output / "mechanism_audit.json").write_text(json.dumps(mechanism, indent=2) + "\n")
    metadata = {
        "schema_version": M5_SCHEMA_VERSION,
        "task": args.task,
        "method": trainer.config.method,
        "seed": trainer.config.seed,
        "effect_weight": trainer.loss_weights.effect,
        "split": args.split,
        "test_samples": len(records),
        "data_samples": int(manifest["samples"]),
        "action_block": int(manifest["action_block"]),
        "trained_max_model_horizon": trainer.curriculum.current_horizon,
        "trained_max_primitive_horizon": (
            trainer.curriculum.current_horizon * int(manifest["action_block"])
        ),
        "max_model_horizon": available,
        "max_primitive_horizon": available * int(manifest["action_block"]),
        "fairness": args.fairness,
        "checkpoint": str(args.checkpoint),
        "selected_checkpoint_global_step": trainer.global_step,
        "formal_training_protocol": protocol_audit,
    }
    (args.output / "m5_run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"run": metadata, "summary": summary, "mechanism": mechanism}, indent=2))


if __name__ == "__main__":
    main()
