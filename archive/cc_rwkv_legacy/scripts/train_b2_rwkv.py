from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cape_wm.cc_rwkv.checkpoint import save_b2_checkpoint
from cape_wm.cc_rwkv.predictor import RWKV7WorldModelConfig, VanillaRWKV7WorldPredictor
from cape_wm.cc_rwkv.training import (
    free_running_b2_loss,
    make_adamw,
    teacher_forced_b2_loss,
)


def _synthetic_sequences(
    samples: int,
    horizon: int,
    latent_dim: int,
    action_dim: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    actions = torch.randn(samples, horizon, action_dim, generator=generator) * 0.2
    action_map = torch.randn(action_dim, latent_dim, generator=generator) / action_dim**0.5
    latent = torch.randn(samples, latent_dim, generator=generator) * 0.2
    sequence = [latent]
    for index in range(horizon):
        latent = 0.85 * latent + actions[:, index] @ action_map
        sequence.append(latent)
    return torch.stack(sequence, dim=1), actions


def _load_sequences(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    arrays = np.load(path, allow_pickle=False)
    if "latents" not in arrays or "actions" not in arrays:
        raise ValueError("NPZ must contain latents [N,T+1,D] and actions [N,T,A]")
    latents = torch.from_numpy(np.asarray(arrays["latents"], dtype=np.float32))
    actions = torch.from_numpy(np.asarray(arrays["actions"], dtype=np.float32))
    if latents.ndim != 3 or actions.ndim != 3 or latents.shape[1] != actions.shape[1] + 1:
        raise ValueError("invalid latent/action sequence shapes")
    if "split" not in arrays:
        raise ValueError("NPZ must include a per-sequence split array to prevent leakage")
    raw_split = np.asarray(arrays["split"])
    train_mask = torch.from_numpy(np.isin(raw_split.astype(str), ["train", "0"]))
    if train_mask.shape != (latents.shape[0],) or not train_mask.any() or train_mask.all():
        raise ValueError("split must contain non-empty train and validation/test subsets")
    return latents, actions, train_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the M1 vanilla RWKV-7 B2 predictor")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--smoke-synthetic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--latent-dim", type=int, default=192)
    parser.add_argument("--action-dim", type=int, default=10)
    parser.add_argument("--model-dim", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--channel-mlp-dim", type=int, default=768)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_synthetic == (args.data is not None):
        raise ValueError("choose exactly one of --data or --smoke-synthetic")
    torch.manual_seed(args.seed)
    if args.smoke_synthetic:
        # Keep the acceptance smoke small while exercising a full 20-step graph.
        if args.latent_dim == 192:
            args.latent_dim, args.action_dim = 8, 3
            args.model_dim, args.num_layers, args.num_heads = 16, 2, 2
            args.channel_mlp_dim = 64
        if not 0 < args.validation_fraction < 1:
            raise ValueError("validation-fraction must be between zero and one")
        latents, actions = _synthetic_sequences(
            max(args.batch_size * 4, 32),
            args.horizon,
            args.latent_dim,
            args.action_dim,
            seed=args.seed,
        )
        validation_count = max(1, round(latents.shape[0] * args.validation_fraction))
        train_mask = torch.ones(latents.shape[0], dtype=torch.bool)
        train_mask[-validation_count:] = False
    else:
        latents, actions, train_mask = _load_sequences(args.data)
        if (args.latent_dim, args.action_dim) != (latents.shape[-1], actions.shape[-1]):
            raise ValueError("CLI latent/action dimensions do not match the NPZ")
    if actions.shape[1] < args.horizon:
        raise ValueError("dataset sequence is shorter than requested horizon")
    latents = latents[:, : args.horizon + 1].to(args.device)
    actions = actions[:, : args.horizon].to(args.device)
    train_mask = train_mask.to(args.device)
    train_latents, validation_latents = latents[train_mask], latents[~train_mask]
    train_actions, validation_actions = actions[train_mask], actions[~train_mask]

    config = RWKV7WorldModelConfig(
        latent_dim=args.latent_dim,
        action_dim=args.action_dim,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        channel_mlp_dim=args.channel_mlp_dim,
    )
    model = VanillaRWKV7WorldPredictor(config).to(args.device)
    optimizer = make_adamw(
        model, learning_rate=args.learning_rate, weight_decay=args.weight_decay
    )
    with torch.no_grad():
        initial_train = float(free_running_b2_loss(model, train_latents, train_actions))
        initial_validation = float(
            free_running_b2_loss(model, validation_latents, validation_actions)
        )
    generator = torch.Generator().manual_seed(args.seed + 1)
    history = []
    for step in range(1, args.steps + 1):
        indices = torch.randint(
            train_latents.shape[0], (args.batch_size,), generator=generator
        ).to(train_latents.device)
        batch_latents = train_latents.index_select(0, indices)
        batch_actions = train_actions.index_select(0, indices)
        teacher = teacher_forced_b2_loss(model, batch_latents, batch_actions)
        free = free_running_b2_loss(model, batch_latents, batch_actions)
        loss = 0.5 * teacher + 0.5 * free
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 20 == 0 or step == args.steps:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "teacher_forced_loss": float(teacher.detach()),
                    "free_running_loss": float(free.detach()),
                }
            )
    with torch.no_grad():
        final_train = float(free_running_b2_loss(model, train_latents, train_actions))
        final_validation = float(
            free_running_b2_loss(model, validation_latents, validation_actions)
        )
    args.output.mkdir(parents=True, exist_ok=True)
    save_b2_checkpoint(
        args.output / "b2_m1.pt",
        model,
        optimizer=optimizer,
        global_step=args.steps,
        best_metric=final_validation,
        training_config=vars(args),
        history=history,
    )
    summary = {
        "schema_version": "cc_rwkv_m1_smoke_v1",
        "protocol_status": "engineering_smoke_only",
        "horizon": args.horizon,
        "train_samples": int(train_mask.sum()),
        "validation_samples": int((~train_mask).sum()),
        "initial_train_free_running_loss": initial_train,
        "final_train_free_running_loss": final_train,
        "train_relative_reduction": (
            (initial_train - final_train) / max(initial_train, 1e-12)
        ),
        "initial_validation_free_running_loss": initial_validation,
        "final_validation_free_running_loss": final_validation,
        "validation_relative_reduction": (
            (initial_validation - final_validation) / max(initial_validation, 1e-12)
        ),
        "finite": bool(np.isfinite(final_train) and np.isfinite(final_validation)),
        "parameter_count": model.parameter_count(),
        "history": history,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
