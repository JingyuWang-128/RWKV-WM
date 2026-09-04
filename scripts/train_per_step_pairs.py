from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.train_cc_rwkv import build_model
from cape_wm.cc_rwkv.per_step_training import InMemoryPerStepSplit, per_step_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RWKV on per-step paired suffix data")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("b2", "b3", "b4", "b6"), default="b6")
    parser.add_argument("--profile", choices=("smoke", "main"), default="smoke")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--effect-weight", type=float, default=1.0)
    parser.add_argument("--curriculum", help="comma-separated model-step horizons")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--teacher-forcing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0 or args.batch_size <= 0:
        raise ValueError("max_steps and batch_size must be positive")
    torch.manual_seed(args.seed)
    uses_paired_loss = args.method == "b6"
    train = InMemoryPerStepSplit(
        args.data, split="train", limit=args.train_limit, load_effect=uses_paired_loss
    )
    validation = InMemoryPerStepSplit(
        args.data, split="validation", limit=args.validation_limit, load_effect=uses_paired_loss
    )
    latent_dim = int(train.tensors["factual_latents"].shape[-1])
    action_dim = int(train.tensors["factual_actions"].shape[-1])
    model = build_model(args.method, args.profile, latent_dim, action_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(args.seed + 1701)
    if uses_paired_loss:
        effect = train.tensors["effect_latents"]
        mask = train.tensors["pulse_noop_mask"]
        threshold = (
            float(torch.quantile(effect.float().norm(dim=-1)[mask], 0.1))
            if mask.any()
            else 0.0
        )
    else:
        threshold = 0.0
    history: list[dict[str, float]] = []
    requested = (
        tuple(int(item) for item in args.curriculum.split(","))
        if args.curriculum
        else (1, 2, 4, 10, 20)
    )
    curriculum = tuple(level for level in requested if level <= train.horizon)
    if not curriculum:
        raise ValueError("curriculum has no horizon available in the dataset")
    args.output.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.max_steps + 1):
        model.train()
        batch = train.random_batch(args.batch_size, generator=generator, device=args.device)
        level_index = min((step - 1) * len(curriculum) // args.max_steps, len(curriculum) - 1)
        train_horizon = curriculum[level_index]
        total, components = per_step_loss(
            model,
            batch,
            horizon=train_horizon,
            effect_threshold=threshold,
            allow_paired_loss=uses_paired_loss,
            effect_weight=args.effect_weight,
            teacher_forcing=args.teacher_forcing,
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        row = {
            "step": float(step), "loss": float(total.detach()),
            "grad_norm": float(grad), "horizon": float(train_horizon),
        }
        row.update({name: float(value.detach()) for name, value in components.items()})
        history.append(row)
        if step % args.log_interval == 0 or step == 1 or step == args.max_steps:
            model.eval()
            with torch.no_grad():
                val_rows = []
                for val_batch in validation.batches(args.batch_size, device=args.device):
                    val_total, val_components = per_step_loss(
                        model,
                        val_batch,
                        horizon=val_batch.horizon,
                        effect_threshold=threshold,
                        allow_paired_loss=uses_paired_loss,
                        effect_weight=args.effect_weight,
                        teacher_forcing=args.teacher_forcing,
                    )
                    val_rows.append((val_total, val_components))
                val_loss = float(torch.stack([item[0] for item in val_rows]).mean())
            print(json.dumps({"step": step, "train_loss": row["loss"], "validation_loss": val_loss}), flush=True)
    torch.save(
        {
            "schema_version": "cc_rwkv_per_step_checkpoint_v1",
            "method": args.method,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": args.max_steps,
            "effect_threshold": threshold,
            "data": str(args.data),
            "history": history,
        },
        args.output / "latest.pt",
    )
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "cc_rwkv_per_step_training_summary_v1",
                "method": args.method,
                "steps": args.max_steps,
                "train_samples": len(train),
                "validation_samples": len(validation),
                "curriculum": list(curriculum),
                "effect_weight": args.effect_weight,
                "teacher_forcing": args.teacher_forcing,
                "effect_threshold": threshold,
                "final_train_loss": history[-1]["loss"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
