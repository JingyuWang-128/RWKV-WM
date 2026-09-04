from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from cape_wm.cc_rwkv.counterfactual import (
    CounterfactualRWKV7Config,
    CounterfactualRWKV7WorldPredictor,
)
from cape_wm.cc_rwkv.checkpoint import load_m4_checkpoint
from cape_wm.cc_rwkv.dwm import DWMOutputBaseline
from cape_wm.cc_rwkv.evaluation import gate_report_markdown, write_evaluation_artifacts
from cape_wm.cc_rwkv.fairness import RunRecord, audit_run_matrix
from cape_wm.cc_rwkv.m4_data import InMemoryBranchSplit, m4_data_provenance
from cape_wm.cc_rwkv.predictor import RWKV7WorldModelConfig, VanillaRWKV7WorldPredictor
from cape_wm.cc_rwkv.trainer import M4Trainer, M4TrainerConfig
from cape_wm.cc_rwkv.training import LossWeights, effect_norm_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified M4 B2/B3/B4/B6 trainer")
    parser.add_argument("--config", type=Path, default=Path("configs/cc_rwkv/tworoom.yaml"))
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("artifacts/cache/cc_rwkv/tworoom/mvp5000/branches.h5"),
    )
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=Path("artifacts/cache/cc_rwkv/tworoom/mvp5000/manifest.json"),
    )
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=Path("artifacts/results/cc_rwkv/tworoom/m0/protocol/manifest.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/results/cc_rwkv/tworoom/m4_smoke")
    )
    parser.add_argument("--methods", default="b2,b3,b4,b6")
    parser.add_argument("--method", choices=("b2", "b3", "b4", "b6"))
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--train-limit",
        type=int,
        help="optional debug cap; omitted means the complete training split",
    )
    parser.add_argument(
        "--validation-limit",
        type=int,
        help="optional debug cap; omitted means the complete validation split",
    )
    parser.add_argument("--profile", choices=("smoke", "main"), default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluation-interval", type=int, default=20)
    parser.add_argument("--allow-early-stop", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--precision", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--effect-weight", type=float, choices=(0.5, 1.0, 2.0), default=1.0)
    parser.add_argument(
        "--b2-checkpoint",
        type=Path,
        help="B2 final-horizon best checkpoint used to initialize a main-profile B6 run; "
        "defaults to <output>/b2/best_rollout_h<final>.pt.",
    )
    parser.add_argument(
        "--curriculum",
        help="comma-separated model-step horizons; defaults to 1,2,4,10,20",
    )
    return parser.parse_args()


def build_model(method: str, profile: str, latent_dim: int, action_dim: int):
    if profile == "main":
        model_dim, layers, heads = 192, 6, 6
        vanilla_width, counterfactual_width = 4096, 3728
        action_hidden, dwm_hidden = 64, 2048
    else:
        model_dim, layers, heads = 32, 2, 4
        vanilla_width, counterfactual_width = 512, 432
        action_hidden, dwm_hidden = 16, 64
    if method in {"b2", "b3"}:
        predictor = VanillaRWKV7WorldPredictor(
            RWKV7WorldModelConfig(
                latent_dim=latent_dim,
                action_dim=action_dim,
                model_dim=model_dim,
                num_layers=layers,
                num_heads=heads,
                channel_mlp_dim=vanilla_width,
            )
        )
        return (
            DWMOutputBaseline(predictor, world_head_hidden_dim=dwm_hidden)
            if method == "b3"
            else predictor
        )
    return CounterfactualRWKV7WorldPredictor(
        CounterfactualRWKV7Config(
            latent_dim=latent_dim,
            action_dim=action_dim,
            model_dim=model_dim,
            num_layers=layers,
            num_heads=heads,
            channel_mlp_dim=counterfactual_width,
            action_hidden_dim=action_hidden,
            centered=method == "b6",
        )
    )


def restore_best_for_horizon_transition(
    trainer: M4Trainer,
    checkpoint: Path,
    *,
    expected_provenance: dict,
) -> dict:
    """Restore best model/Adam state without rewinding the global budget or scheduler."""

    current_lrs = [float(group["lr"]) for group in trainer.optimizer.param_groups]
    best_model, payload = load_m4_checkpoint(
        checkpoint,
        map_location="cpu",
        expected_provenance=expected_provenance,
    )
    if payload.get("method") != trainer.config.method:
        raise ValueError("curriculum transition checkpoint method mismatch")
    trainer.model.load_state_dict(best_model.state_dict())
    if payload.get("optimizer") is None:
        raise ValueError("curriculum transition checkpoint has no optimizer state")
    trainer.optimizer.load_state_dict(payload["optimizer"])
    for group, learning_rate in zip(trainer.optimizer.param_groups, current_lrs, strict=True):
        group["lr"] = learning_rate
    return {
        "source_checkpoint": str(checkpoint),
        "source_global_step": int(payload["global_step"]),
        "restored_model": True,
        "restored_optimizer": True,
        "preserved_global_step": trainer.global_step,
        "preserved_scheduler": True,
    }


def main() -> None:
    args = parse_args()
    methods = (
        (args.method,)
        if args.method is not None
        else tuple(item.strip().lower() for item in args.methods.split(",") if item.strip())
    )
    if len(set(methods)) != len(methods) or not methods:
        raise ValueError("methods must be a non-empty unique comma-separated list")
    if args.batch_size < 2 and "b3" in methods:
        raise ValueError("B3 InfoNCE and BatchNorm require batch-size >= 2")
    resolved_user_config = yaml.safe_load(args.config.read_text())
    train = InMemoryBranchSplit(args.data, split="train", limit=args.train_limit)
    validation = InMemoryBranchSplit(args.data, split="validation", limit=args.validation_limit)
    data_provenance, encoder_provenance = m4_data_provenance(
        args.data, args.data_manifest, args.protocol_manifest
    )
    threshold = effect_norm_threshold(train.tensors["branch_latents"])
    available_horizon = train.tensors["branch_actions_raw"].shape[2]
    requested_curriculum = (
        tuple(int(item) for item in args.curriculum.split(","))
        if args.curriculum
        else (1, 2, 4, 10, 20)
    )
    if (
        not requested_curriculum
        or any(level <= 0 for level in requested_curriculum)
        or tuple(sorted(set(requested_curriculum))) != requested_curriculum
    ):
        raise ValueError("curriculum must contain unique increasing positive integers")
    curriculum = tuple(level for level in requested_curriculum if level <= available_horizon)
    if not curriculum:
        raise ValueError("no requested curriculum horizon is available in the dataset")
    precision = (
        "bfloat16"
        if args.precision == "auto" and str(args.device).startswith("cuda")
        else "float32"
        if args.precision == "auto"
        else args.precision
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "input_config.yaml").write_text(
        yaml.safe_dump(resolved_user_config, sort_keys=True)
    )
    records: list[RunRecord] = []
    summaries: dict[str, dict] = {}
    for method in methods:
        torch.manual_seed(args.seed)
        trainer_config = M4TrainerConfig(
            method=method,
            max_steps=args.max_steps,
            curriculum_levels=curriculum,
            minimum_horizon_steps=(
                5000 if args.profile == "main" else max(10, args.max_steps // 4)
            ),
            maximum_horizon_steps=(
                10000 if args.profile == "main" else max(20, args.max_steps // 2)
            ),
            warmup_steps=min(2000, max(args.max_steps // 10, 0)),
            learning_rate=(5e-5 if args.profile == "main" else 3e-3),
            stage_b_freeze_steps=(1000 if args.profile == "main" else 1),
            seed=args.seed,
            precision=precision,
        )
        method_output = args.output / method
        latest_checkpoint = method_output / "latest.pt"
        initialization_path = method_output / "initialization.json"
        if args.resume_existing and latest_checkpoint.exists():
            if method == "b6" and args.profile == "main":
                if not initialization_path.is_file():
                    raise ValueError(
                        "refusing to resume a main-profile B6 checkpoint without "
                        "a B2 initialization audit"
                    )
                initialization = json.loads(initialization_path.read_text())
                if initialization.get("scheme") != "b2_final_horizon_rollout_best":
                    raise ValueError("B6 initialization audit has an unsupported scheme")
            trainer = M4Trainer.resume(
                latest_checkpoint,
                device=args.device,
                effect_threshold=threshold,
                expected_provenance={**data_provenance, **encoder_provenance},
            )
            if trainer.config != trainer_config:
                raise ValueError(
                    f"resume configuration mismatch for {method}: "
                    f"stored={trainer.config!r}, requested={trainer_config!r}"
                )
        else:
            model = build_model(
                method,
                args.profile,
                int(data_provenance["latent_dim"]),
                int(data_provenance["action_dim"]),
            )
            if method == "b6" and args.profile == "main":
                b2_checkpoint = args.b2_checkpoint or (
                    args.output / "b2" / f"best_rollout_h{curriculum[-1]}.pt"
                )
                vanilla, b2_payload = load_m4_checkpoint(
                    b2_checkpoint,
                    map_location="cpu",
                    expected_provenance={**data_provenance, **encoder_provenance},
                )
                if b2_payload.get("method") != "b2" or not isinstance(
                    vanilla, VanillaRWKV7WorldPredictor
                ):
                    raise ValueError("B6 initialization requires a B2 vanilla RWKV checkpoint")
                copy_audit = model.load_world_from_vanilla(vanilla)
                initialization_path.parent.mkdir(parents=True, exist_ok=True)
                initialization_path.write_text(
                    json.dumps(
                        {
                            "scheme": "b2_final_horizon_rollout_best",
                            "source_checkpoint": str(b2_checkpoint),
                            "source_method": "b2",
                            "source_seed": int(
                                b2_payload["training_config"]["trainer"]["seed"]
                            ),
                            "target_seed": args.seed,
                            "source_global_step": int(b2_payload["global_step"]),
                            "source_checkpoint_selection": "best_rollout_final_horizon",
                            "source_final_model_horizon": curriculum[-1],
                            "source_curriculum_level_index": int(
                                b2_payload.get("curriculum", {}).get("level_index", -1)
                            ),
                            "source_curriculum_steps_at_level": int(
                                b2_payload.get("curriculum", {}).get("steps_at_level", 0)
                            ),
                            "effect_weight": args.effect_weight,
                            "copied_tensor_count": len(copy_audit["copied"]),
                            "copied": copy_audit["copied"],
                            "ignored": copy_audit["ignored"],
                            "new": copy_audit["new"],
                        },
                        indent=2,
                    )
                    + "\n"
                )
            trainer = M4Trainer(
                model,
                trainer_config,
                loss_weights=LossWeights(effect=args.effect_weight),
                effect_threshold=threshold,
                device=args.device,
            )
        best_ledger = method_output / "best_by_horizon.json"
        if args.resume_existing and best_ledger.exists():
            best_by_horizon = {
                int(horizon): float(score)
                for horizon, score in json.loads(best_ledger.read_text()).items()
            }
        else:
            best_by_horizon = {}
        best_rollout_ledger = method_output / "best_rollout_by_horizon.json"
        if args.resume_existing and best_rollout_ledger.exists():
            best_rollout_by_horizon = {
                int(horizon): float(score)
                for horizon, score in json.loads(best_rollout_ledger.read_text()).items()
            }
        else:
            best_rollout_by_horizon = {}
        best = min(best_by_horizon.values(), default=float("inf"))
        while trainer.global_step < args.max_steps:
            batch = train.random_batch(
                args.batch_size,
                generator=trainer.generator,
                device=args.device,
            )
            record = trainer.train_step(batch)
            if trainer.global_step % args.evaluation_interval == 0:
                evaluation_batches = list(validation.batches(args.batch_size, device=args.device))
                evaluation_horizon = trainer.curriculum.current_horizon
                result = trainer.evaluate(evaluation_batches, horizon=evaluation_horizon)
                score = float(result["validation_score"])
                rollout_endpoint = float(result["rollout_rmse"][-1])
                if score < best_by_horizon.get(evaluation_horizon, float("inf")):
                    best_by_horizon[evaluation_horizon] = score
                    trainer.save(
                        method_output / f"best_h{evaluation_horizon}.pt",
                        data_provenance=data_provenance,
                        encoder_provenance=encoder_provenance,
                    )
                    best_ledger.write_text(
                        json.dumps(best_by_horizon, indent=2, sort_keys=True) + "\n"
                    )
                if rollout_endpoint < best_rollout_by_horizon.get(
                    evaluation_horizon, float("inf")
                ):
                    best_rollout_by_horizon[evaluation_horizon] = rollout_endpoint
                    trainer.save(
                        method_output / f"best_rollout_h{evaluation_horizon}.pt",
                        data_provenance=data_provenance,
                        encoder_provenance=encoder_provenance,
                    )
                    best_rollout_ledger.write_text(
                        json.dumps(best_rollout_by_horizon, indent=2, sort_keys=True) + "\n"
                    )
                if score < best:
                    best = score
                    trainer.save(
                        method_output / "best.pt",
                        data_provenance=data_provenance,
                        encoder_provenance=encoder_provenance,
                    )
                advanced, should_stop = trainer.observe_validation(score)
                if advanced:
                    transition = restore_best_for_horizon_transition(
                        trainer,
                        method_output / f"best_h{evaluation_horizon}.pt",
                        expected_provenance={**data_provenance, **encoder_provenance},
                    )
                    transition.update(
                        {
                            "from_horizon": evaluation_horizon,
                            "to_horizon": trainer.curriculum.current_horizon,
                            "validation_score_at_transition": score,
                            "best_validation_score": best_by_horizon[evaluation_horizon],
                        }
                    )
                    with (method_output / "curriculum_transitions.jsonl").open("a") as handle:
                        handle.write(json.dumps(transition) + "\n")
                trainer.save(
                    latest_checkpoint,
                    data_provenance=data_provenance,
                    encoder_provenance=encoder_provenance,
                )
                print(
                    json.dumps(
                        {
                            "event": "validation",
                            "method": method,
                            "seed": args.seed,
                            "global_step": trainer.global_step,
                            "evaluated_horizon": evaluation_horizon,
                            "next_horizon": trainer.curriculum.current_horizon,
                            "validation_score": score,
                            "rollout_endpoint_rmse": rollout_endpoint,
                            "curriculum_advanced": advanced,
                            "early_stop_requested": should_stop,
                        }
                    ),
                    flush=True,
                )
                if should_stop and args.allow_early_stop:
                    break
            if record.get("skipped_nonfinite"):
                continue
        trainer.save(
            latest_checkpoint,
            data_provenance=data_provenance,
            encoder_provenance=encoder_provenance,
        )
        evaluation_batches = list(validation.batches(args.batch_size, device=args.device))
        probe_batches = list(train.batches(args.batch_size, device=args.device))
        summary = write_evaluation_artifacts(
            args.output / method / "evaluation",
            trainer,
            evaluation_batches,
            probe_batches,
            horizons=curriculum,
            provenance={
                "data": data_provenance,
                "encoder": encoder_provenance,
                "checkpoint": str(args.output / method / "latest.pt"),
                "implementation_status": (
                    "paper_spec_reimplementation" if method == "b3" else "native"
                ),
            },
            fairness_status="pending matrix audit",
        )
        summaries[method] = summary
        predictor = getattr(trainer.model, "predictor", trainer.model)
        predictor_parameters = sum(parameter.numel() for parameter in predictor.parameters())
        records.append(
            RunRecord(
                method=method,
                seed=args.seed,
                encoder_sha256=str(encoder_provenance["encoder_sha256"]),
                normalizer_sha256=str(data_provenance["normalizer_sha256"]),
                dataset_sha256=str(data_provenance["dataset_sha256"]),
                split_sha256=str(data_provenance["split_sha256"]),
                branch_trajectories=len(train) * 4,
                optimizer_steps=trainer.global_step,
                effective_batch_size=args.batch_size,
                curriculum=curriculum,
                predictor_parameters=predictor_parameters,
                paired_loss=method == "b6",
                implementation_status=(
                    "paper_spec_reimplementation" if method == "b3" else "native"
                ),
                precision=precision,
            )
        )
    fairness = audit_run_matrix(
        records,
        require_complete=set(methods) == {"b2", "b3", "b4", "b6"},
    )
    (args.output / "fairness.json").write_text(json.dumps(fairness, indent=2) + "\n")
    for method, summary in summaries.items():
        (args.output / method / "evaluation" / "gate_report.md").write_text(
            gate_report_markdown(
                method=method,
                summary=summary,
                no_op_max=summary.get("actual_equals_reference_max_delta"),
                fairness_status=fairness["status"],
            )
        )
    aggregate = {
        "schema_version": "cc_rwkv_m4_matrix_v1",
        "methods": list(methods),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "train_samples": len(train),
        "validation_samples": len(validation),
        "effect_threshold_train_q10": threshold,
        "fairness": fairness["status"],
        "summaries": summaries,
    }
    (args.output / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
