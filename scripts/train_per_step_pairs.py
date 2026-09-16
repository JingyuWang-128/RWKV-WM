from __future__ import annotations

import argparse
import hashlib
import json
import signal
from pathlib import Path

import torch

from cape_wm.cc_rwkv.model_factory import build_rwkv_world_model
from cape_wm.cc_rwkv.per_step_checkpoint import (
    restore_per_step_checkpoint,
    save_per_step_checkpoint,
)
from cape_wm.cc_rwkv.per_step_training import (
    ROLLOUT_MASK_POLICY,
    InMemoryPerStepSplit,
    per_step_loss,
)
from cape_wm.cc_rwkv.stability import (
    StabilityGateError,
    gradient_window_report,
    learning_rate_factor,
)
from cape_wm.cc_rwkv.training import make_adamw

OPTIMIZER_POLICY = "rwkv7_official_style_adamw_v1"


def require_finite_tensor(value: torch.Tensor, *, name: str, step: int) -> None:
    """Fail before a non-finite value can contaminate an optimizer update."""

    if bool(torch.isfinite(value).all()):
        return
    nonfinite = int((~torch.isfinite(value)).sum().item())
    raise FloatingPointError(
        f"non-finite {name} at step {step}: {nonfinite}/{value.numel()} values"
    )


def verify_protocol_manifest(
    path: Path,
    *,
    protocol_id: str,
    data: Path,
    data_sha256: str,
) -> str:
    """Verify the frozen formal protocol and every registered code artifact."""
    if not path.is_file():
        raise FileNotFoundError(f"formal protocol manifest is missing: {path}")
    payload = json.loads(path.read_text())
    if payload.get("protocol_id") != protocol_id:
        raise ValueError("formal protocol ID does not match its manifest")
    if payload.get("data_status") != "audited_v2_primitive_pass":
        raise ValueError("formal protocol data status is not PASS")
    if payload.get("optimizer_policy") != OPTIMIZER_POLICY:
        raise ValueError("formal protocol optimizer policy mismatch")
    matches = [
        record
        for record in payload.get("datasets", {}).values()
        if Path(str(record.get("path", ""))).resolve() == data.resolve()
    ]
    if len(matches) != 1 or matches[0].get("sha256") != data_sha256:
        raise ValueError("formal dataset is not frozen in the protocol manifest")
    for source, expected in payload.get("code_sha256", {}).items():
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"frozen code artifact is missing: {source_path}")
        actual = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"frozen code SHA256 mismatch: {source}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_data_artifacts(
    data: Path, *, expected_sha256: str | None, formal: bool
) -> dict[str, object]:
    """Fail closed on stale, unaudited, or unexpectedly replaced pair data."""
    if not data.is_file():
        raise FileNotFoundError(data)
    manifest_path = data.with_name("manifest.json")
    audit_path = data.with_name("audit.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"pair dataset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    declared_sha256 = str(manifest.get("pair_dataset_sha256", ""))
    if not declared_sha256:
        raise ValueError("pair dataset manifest does not declare pair_dataset_sha256")
    if int(manifest.get("bytes", -1)) != data.stat().st_size:
        raise ValueError("pair dataset byte size does not match its manifest")
    if formal and not expected_sha256:
        raise ValueError("formal training requires --data-sha256")
    if expected_sha256 is not None and declared_sha256 != expected_sha256:
        raise ValueError("pair dataset SHA256 does not match the frozen protocol")
    if formal:
        if not audit_path.is_file():
            raise FileNotFoundError(f"formal pair dataset audit is missing: {audit_path}")
        audit = json.loads(audit_path.read_text())
        if audit.get("status") != "pass" or audit.get("failures"):
            raise ValueError("formal pair dataset audit did not pass")
        if Path(str(audit.get("dataset", ""))).resolve() != data.resolve():
            raise ValueError("pair dataset audit points to a different dataset")
        if int(audit.get("samples", -1)) != int(manifest.get("samples", -2)):
            raise ValueError("pair dataset audit/manifest sample counts differ")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RWKV on per-step paired suffix data")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-id", default="development")
    parser.add_argument(
        "--data-sha256",
        help="frozen SHA256 required by non-development protocols",
    )
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        help="frozen manifest required by non-development protocols",
    )
    parser.add_argument("--method", choices=("b2", "b3", "b4", "b6"), default="b6")
    parser.add_argument("--profile", choices=("smoke", "main"), default="smoke")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=64,
        help="evaluation-only batch size; does not change optimization",
    )
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--decay-lr-multiplier", type=float, default=2.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=1.0)
    parser.add_argument("--stability-window", type=int, default=200)
    parser.add_argument("--stability-grace-steps", type=int, default=200)
    parser.add_argument(
        "--max-clip-fraction",
        type=float,
        help="enable fail-closed quality gate at fixed clip norm 1 (e.g. 0.2)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--effect-weight", type=float, default=1.0)
    parser.add_argument(
        "--disable-paired-loss",
        action="store_true",
        help="run the B6 centered architecture without reading paired-effect labels",
    )
    parser.add_argument("--dwm-contrastive-weight", type=float, default=0.3)
    parser.add_argument("--dwm-orthogonality-weight", type=float, default=0.5)
    parser.add_argument("--dwm-temperature", type=float, default=0.07)
    parser.add_argument("--curriculum", help="comma-separated model-step horizons")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="optimizer steps between atomic latest.pt updates",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=5000,
        help="optimizer steps between retained checkpoints/step_XXXXXXXX.pt snapshots",
    )
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume",
        action="store_true",
        help="resume exactly from <output>/latest.pt",
    )
    resume.add_argument(
        "--resume-from",
        type=Path,
        help="resume exactly from an explicit v3 per-step checkpoint",
    )
    parser.add_argument("--teacher-forcing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        min(
            args.max_steps,
            args.batch_size,
            args.validation_batch_size,
            args.log_interval,
            args.checkpoint_interval,
            args.snapshot_interval,
            args.gradient_accumulation,
            args.stability_window,
        )
        <= 0
    ):
        raise ValueError("step, batch, log, and checkpoint intervals must be positive")
    if args.learning_rate <= 0 or args.decay_lr_multiplier <= 0:
        raise ValueError("learning rates must be positive")
    learning_rate_factor(
        1,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        min_ratio=args.min_lr_ratio,
    )
    if args.stability_grace_steps < 0:
        raise ValueError("stability grace steps must be non-negative")
    if args.max_clip_fraction is not None:
        if not 0 <= args.max_clip_fraction <= 1:
            raise ValueError("max-clip-fraction must be in [0, 1]")
    if args.method == "b3" and args.batch_size < 2:
        raise ValueError("B3 DWM loss requires batch-size >= 2")
    if min(args.dwm_contrastive_weight, args.dwm_orthogonality_weight) < 0:
        raise ValueError("DWM loss weights must be non-negative")
    if args.dwm_temperature <= 0:
        raise ValueError("DWM temperature must be positive")
    if args.disable_paired_loss and args.method != "b6":
        raise ValueError("--disable-paired-loss is only valid for method b6")
    if args.effect_weight < 0:
        raise ValueError("effect-weight must be non-negative")
    data_manifest = verify_data_artifacts(
        args.data,
        expected_sha256=args.data_sha256,
        formal=args.protocol_id != "development",
    )
    protocol_manifest_sha256 = None
    if args.protocol_id != "development":
        if args.protocol_manifest is None:
            raise ValueError("formal training requires --protocol-manifest")
        protocol_manifest_sha256 = verify_protocol_manifest(
            args.protocol_manifest,
            protocol_id=args.protocol_id,
            data=args.data,
            data_sha256=str(data_manifest["pair_dataset_sha256"]),
        )
    torch.manual_seed(args.seed)
    uses_paired_loss = args.method == "b6" and not args.disable_paired_loss
    train = InMemoryPerStepSplit(
        args.data, split="train", limit=args.train_limit, load_effect=uses_paired_loss
    )
    validation = InMemoryPerStepSplit(
        args.data, split="validation", limit=args.validation_limit, load_effect=uses_paired_loss
    )
    latent_dim = int(train.tensors["factual_latents"].shape[-1])
    action_dim = int(train.tensors["factual_actions"].shape[-1])
    model = build_rwkv_world_model(args.method, args.profile, latent_dim, action_dim).to(
        args.device
    )
    optimizer = make_adamw(model, learning_rate=args.learning_rate, weight_decay=1e-3)
    # make_adamw places decay logits in group 1 with the official-style x2 LR.
    optimizer.param_groups[1]["lr"] = args.learning_rate * args.decay_lr_multiplier
    base_learning_rates = [group["lr"] for group in optimizer.param_groups]
    generator = torch.Generator().manual_seed(args.seed + 1701)
    if uses_paired_loss:
        effect = train.tensors["effect_latents"]
        mask = train.tensors["pulse_noop_mask"]
        threshold = (
            float(torch.quantile(effect.float().norm(dim=-1)[mask], 0.1)) if mask.any() else 0.0
        )
    else:
        threshold = 0.0
    requested = (
        tuple(int(item) for item in args.curriculum.split(","))
        if args.curriculum
        else (1, 2, 4, 10, 20)
    )
    curriculum = tuple(level for level in requested if level <= train.horizon)
    if not curriculum:
        raise ValueError("curriculum has no horizon available in the dataset")
    if args.max_clip_fraction is not None and (
        args.max_steps // len(curriculum) < args.stability_grace_steps + args.stability_window
    ):
        raise ValueError("each curriculum phase must include grace plus a full stability window")
    args.output.mkdir(parents=True, exist_ok=True)
    run_config = {
        "method": args.method,
        "protocol_id": args.protocol_id,
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "profile": args.profile,
        "data": str(args.data),
        "data_sha256": str(data_manifest["pair_dataset_sha256"]),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "validation_batch_size": args.validation_batch_size,
        "learning_rate": args.learning_rate,
        "gradient_accumulation": args.gradient_accumulation,
        "decay_lr_multiplier": args.decay_lr_multiplier,
        "warmup_steps": args.warmup_steps,
        "min_lr_ratio": args.min_lr_ratio,
        "stability_window": args.stability_window,
        "stability_grace_steps": args.stability_grace_steps,
        "max_clip_fraction": args.max_clip_fraction,
        "clip_norm": 1.0,
        "seed": args.seed,
        "effect_weight": args.effect_weight,
        "effective_effect_weight": args.effect_weight if uses_paired_loss else 0.0,
        "rollout_mask_policy": ROLLOUT_MASK_POLICY,
        "paired_loss": uses_paired_loss,
        "dwm_contrastive_weight": args.dwm_contrastive_weight,
        "dwm_orthogonality_weight": args.dwm_orthogonality_weight,
        "dwm_temperature": args.dwm_temperature,
        "loss_normalization": "position_balanced_v1",
        "curriculum": list(curriculum),
        "teacher_forcing": args.teacher_forcing,
        "optimizer_policy": OPTIMIZER_POLICY,
        "latent_dim": latent_dim,
        "action_dim": action_dim,
    }
    resume_path = (
        args.resume_from
        if args.resume_from is not None
        else args.output / "latest.pt"
        if args.resume
        else None
    )
    history: list[dict[str, float]] = []
    completed_step = 0
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
        payload = restore_per_step_checkpoint(
            resume_path,
            model,
            optimizer,
            generator,
            expected_run_config=run_config,
            expected_effect_threshold=threshold,
            map_location=args.device,
        )
        completed_step = int(payload["step"])
        history = [dict(row) for row in payload.get("history", [])]
        if completed_step >= args.max_steps:
            raise ValueError(f"checkpoint is already at step {completed_step}/{args.max_steps}")
        print(
            json.dumps(
                {
                    "resumed_from": str(resume_path),
                    "completed_step": completed_step,
                    "remaining_steps": args.max_steps - completed_step,
                }
            ),
            flush=True,
        )
    elif (args.output / "latest.pt").exists():
        raise FileExistsError(
            f"{args.output / 'latest.pt'} already exists; use --resume or a new output directory"
        )

    def handle_termination(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_termination)

    active_step = completed_step
    active_phase = "setup"
    failure_path = args.output / "failure.json"
    failure_path.unlink(missing_ok=True)
    try:
        for step in range(completed_step + 1, args.max_steps + 1):
            active_step = step
            model.train()
            level_index = min(
                (step - 1) * len(curriculum) // args.max_steps,
                len(curriculum) - 1,
            )
            train_horizon = curriculum[level_index]
            lr_factor = learning_rate_factor(
                step,
                max_steps=args.max_steps,
                warmup_steps=args.warmup_steps,
                min_ratio=args.min_lr_ratio,
            )
            for group, base_lr in zip(optimizer.param_groups, base_learning_rates, strict=True):
                group["lr"] = base_lr * lr_factor
            optimizer.zero_grad(set_to_none=True)
            total = torch.zeros((), device=args.device)
            components: dict[str, torch.Tensor] = {}
            for _ in range(args.gradient_accumulation):
                batch = train.random_batch(args.batch_size, generator=generator, device=args.device)
                active_phase = "training_forward"
                micro_loss, micro_components = per_step_loss(
                    model,
                    batch,
                    horizon=train_horizon,
                    effect_threshold=threshold,
                    allow_paired_loss=uses_paired_loss,
                    effect_weight=args.effect_weight,
                    dwm_contrastive_weight=args.dwm_contrastive_weight,
                    dwm_orthogonality_weight=args.dwm_orthogonality_weight,
                    dwm_temperature=args.dwm_temperature,
                    teacher_forcing=args.teacher_forcing,
                )
                require_finite_tensor(micro_loss, name="training loss", step=step)
                for name, value in micro_components.items():
                    require_finite_tensor(value, name=f"training component {name}", step=step)
                    components[name] = components.get(name, 0) + (
                        value.detach() / args.gradient_accumulation
                    )
                active_phase = "training_backward"
                (micro_loss / args.gradient_accumulation).backward()
                total += micro_loss.detach() / args.gradient_accumulation
            try:
                grad = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
            except RuntimeError as error:
                raise FloatingPointError(f"non-finite gradient norm at step {step}") from error
            require_finite_tensor(grad, name="gradient norm", step=step)
            active_phase = "stability_gate"
            if args.max_clip_fraction is not None:
                # Reconstruct the window from checkpoint history for exact resume.
                phase_start = (level_index * args.max_steps + len(curriculum) - 1) // len(
                    curriculum
                )
                phase_step = step - phase_start
                observed = phase_step - args.stability_grace_steps
                if observed >= args.stability_window and observed % args.stability_window == 0:
                    norms = [r["grad_norm"] for r in history[-(args.stability_window - 1) :]]
                    if args.stability_window == 1:
                        norms = []
                    report = gradient_window_report(
                        norms + [float(grad)], max_clip_fraction=args.max_clip_fraction
                    )
                    report.update({"step": step, "horizon": train_horizon})
                    with (args.output / "stability_windows.jsonl").open("a") as handle:
                        handle.write(json.dumps(report) + "\n")
                    print(json.dumps({"stability_window": report}), flush=True)
                    if not report["passed"]:
                        raise StabilityGateError(f"gradient quality gate failed: {report}")
            active_phase = "optimizer_step"
            optimizer.step()
            row = {
                "step": float(step),
                "loss": float(total.detach()),
                "grad_norm": float(grad),
                "horizon": float(train_horizon),
                "clip_scale": min(1.0, 1.0 / (float(grad) + 1e-6)),
                "clipped": float(float(grad) > 1.0),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            row.update({name: float(value.detach()) for name, value in components.items()})
            history.append(row)
            completed_step = step
            validate_now = step % args.log_interval == 0 or step == args.max_steps
            if validate_now:
                active_phase = "validation"
                model.eval()
                with torch.no_grad():
                    val_sum = total.new_zeros(())
                    val_count = 0
                    val_components: dict[str, torch.Tensor] = {}
                    for val_batch in validation.batches(
                        args.validation_batch_size, device=args.device
                    ):
                        val_total, current_val_components = per_step_loss(
                            model,
                            val_batch,
                            horizon=val_batch.horizon,
                            effect_threshold=threshold,
                            allow_paired_loss=uses_paired_loss,
                            effect_weight=args.effect_weight,
                            dwm_contrastive_weight=args.dwm_contrastive_weight,
                            dwm_orthogonality_weight=args.dwm_orthogonality_weight,
                            dwm_temperature=args.dwm_temperature,
                            teacher_forcing=args.teacher_forcing,
                        )
                        require_finite_tensor(val_total, name="validation loss", step=step)
                        val_sum += val_total * val_batch.batch_size
                        for name, value in current_val_components.items():
                            require_finite_tensor(value, name=f"validation {name}", step=step)
                            val_components[name] = val_components.get(name, 0) + (
                                value * val_batch.batch_size
                            )
                        val_count += val_batch.batch_size
                    val_mean = val_sum / val_count
                    require_finite_tensor(val_mean, name="validation mean", step=step)
                    val_loss = float(val_mean)
                row["validation_loss"] = val_loss
                row.update(
                    {
                        f"validation_{name}": float(value / val_count)
                        for name, value in val_components.items()
                    }
                )
            if validate_now or step == 1:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "train_loss": row["loss"],
                            "validation_loss": row.get("validation_loss"),
                        }
                    ),
                    flush=True,
                )
            if (
                step % args.checkpoint_interval == 0
                or step % args.snapshot_interval == 0
                or step == args.max_steps
            ):
                active_phase = "checkpoint"
                keep_numbered = step % args.snapshot_interval == 0 or step == args.max_steps
                latest, numbered = save_per_step_checkpoint(
                    args.output,
                    model,
                    optimizer,
                    step=step,
                    effect_threshold=threshold,
                    history=history,
                    batch_generator=generator,
                    run_config=run_config,
                    status="complete" if step == args.max_steps else "running",
                    keep_numbered=keep_numbered,
                )
                (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "checkpoint": str(latest),
                            "numbered_checkpoint": (
                                str(numbered) if numbered is not None else None
                            ),
                            "step": step,
                        }
                    ),
                    flush=True,
                )
            active_phase = "training"
    except (FloatingPointError, StabilityGateError) as error:
        failure = {
            "status": (
                "failed_stability" if isinstance(error, StabilityGateError) else "failed_nonfinite"
            ),
            "step": int(active_step),
            "last_completed_step": int(completed_step),
            "phase": active_phase,
            "error": str(error),
            "method": args.method,
            "seed": args.seed,
            "data": str(args.data),
            "data_sha256": str(data_manifest["pair_dataset_sha256"]),
            "protocol_id": args.protocol_id,
            "protocol_manifest_sha256": protocol_manifest_sha256,
        }
        failure_path.write_text(json.dumps(failure, indent=2) + "\n")
        (args.output / "failed_history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(failure), flush=True)
        raise
    except KeyboardInterrupt:
        if completed_step > 0:
            latest, numbered = save_per_step_checkpoint(
                args.output,
                model,
                optimizer,
                step=completed_step,
                effect_threshold=threshold,
                history=history,
                batch_generator=generator,
                run_config=run_config,
                status="interrupted",
                keep_numbered=True,
            )
            (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "interrupted_checkpoint": str(latest),
                        "numbered_checkpoint": str(numbered),
                        "step": completed_step,
                    }
                ),
                flush=True,
            )
        raise

    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "cc_rwkv_per_step_training_summary_v2",
                "method": args.method,
                "protocol_id": args.protocol_id,
                "protocol_manifest_sha256": protocol_manifest_sha256,
                "profile": args.profile,
                "data": str(args.data),
                "data_sha256": str(data_manifest["pair_dataset_sha256"]),
                "steps": completed_step,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "validation_batch_size": args.validation_batch_size,
                "learning_rate": args.learning_rate,
                "gradient_accumulation": args.gradient_accumulation,
                "decay_lr_multiplier": args.decay_lr_multiplier,
                "warmup_steps": args.warmup_steps,
                "min_lr_ratio": args.min_lr_ratio,
                "stability_window": args.stability_window,
                "stability_grace_steps": args.stability_grace_steps,
                "max_clip_fraction": args.max_clip_fraction,
                "clip_norm": 1.0,
                "train_samples": len(train),
                "validation_samples": len(validation),
                "curriculum": list(curriculum),
                "effect_weight": args.effect_weight,
                "effective_effect_weight": args.effect_weight if uses_paired_loss else 0.0,
                "rollout_mask_policy": ROLLOUT_MASK_POLICY,
                "paired_loss": uses_paired_loss,
                "dwm_contrastive_weight": args.dwm_contrastive_weight,
                "dwm_orthogonality_weight": args.dwm_orthogonality_weight,
                "dwm_temperature": args.dwm_temperature,
                "loss_normalization": "position_balanced_v1",
                "teacher_forcing": args.teacher_forcing,
                "optimizer_policy": OPTIMIZER_POLICY,
                "effect_threshold": threshold,
                "final_train_loss": history[-1]["loss"],
                "checkpoint_interval": args.checkpoint_interval,
                "snapshot_interval": args.snapshot_interval,
                "checkpoint_schema": "cc_rwkv_per_step_checkpoint_v3",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
