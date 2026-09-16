from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from cape_wm.cc_rwkv.counterfactual import (
    CounterfactualRWKV7Config,
    CounterfactualRWKV7WorldPredictor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M3 real-pair CC-RWKV overfit smoke")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("artifacts/cache/cc_rwkv/tworoom/mvp5000/branches.h5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/cc_rwkv/tworoom/m3_smoke_log_hazard"),
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_real_pairs(path: Path, samples: int) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"M2 branch cache is missing: {path}")
    with h5py.File(path, "r") as handle:
        split = np.asarray(handle["samples/split"])
        train = np.flatnonzero(split == 0)[:samples]
        if len(train) != samples:
            raise ValueError("not enough training samples in M2 cache")
        group = handle["samples"]
        arrays = {
            "history_latents": np.asarray(group["history_latents"][train], dtype=np.float32),
            "history_actions": np.asarray(
                group["history_actions_raw"][train], dtype=np.float32
            ),
            "initial": np.asarray(group["branch_latents"][train, 1, 0], dtype=np.float32),
            "factual_action": np.asarray(
                group["branch_actions_raw"][train, 1, 0], dtype=np.float32
            ),
            "noop_action": np.asarray(
                group["branch_actions_raw"][train, 2, 0], dtype=np.float32
            ),
            "factual_target": np.asarray(
                group["branch_latents"][train, 1, 1], dtype=np.float32
            ),
            "noop_target": np.asarray(
                group["branch_latents"][train, 2, 1], dtype=np.float32
            ),
        }
    if not np.array_equal(arrays["noop_action"], np.zeros_like(arrays["noop_action"])):
        raise RuntimeError("PULSE_NOOP branch is not a zero-action reference")
    return {name: torch.from_numpy(value) for name, value in arrays.items()}


def paired_predictions(
    model: CounterfactualRWKV7WorldPredictor,
    batch: dict[str, torch.Tensor],
    *,
    diagnostics: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    state = model.consume_history(batch["history_latents"], batch["history_actions"])
    factual, _, info = model.step(
        batch["initial"],
        batch["factual_action"],
        state,
        batch["noop_action"],
        return_diagnostics=diagnostics,
    )
    noop, _, _ = model.step(
        batch["initial"],
        batch["noop_action"],
        state,
        batch["noop_action"],
    )
    return factual, noop, info


def losses(
    factual: torch.Tensor,
    noop: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ordinary = 0.5 * (
        F.mse_loss(factual, batch["factual_target"])
        + F.mse_loss(noop, batch["noop_target"])
    )
    predicted_effect = factual - noop
    true_effect = batch["factual_target"] - batch["noop_target"]
    effect = F.mse_loss(predicted_effect, true_effect)
    return ordinary + effect, ordinary, effect


@torch.no_grad()
def evaluate(
    model: CounterfactualRWKV7WorldPredictor,
    data: dict[str, torch.Tensor],
    batch_size: int,
) -> dict[str, float]:
    totals = {"ordinary": 0.0, "effect": 0.0, "gate": 0.0, "saturated": 0.0}
    samples = next(iter(data.values())).shape[0]
    for start in range(0, samples, batch_size):
        batch = {name: value[start : start + batch_size] for name, value in data.items()}
        factual, noop, info = paired_predictions(model, batch, diagnostics=True)
        _, ordinary, effect = losses(factual, noop, batch)
        weight = factual.shape[0]
        gate = info["intervention_gate"]
        totals["ordinary"] += float(ordinary) * weight
        totals["effect"] += float(effect) * weight
        totals["gate"] += float(gate.mean()) * weight
        totals["saturated"] += float(((gate < 0.01) | (gate > 0.99)).float().mean()) * weight
    return {name: value / samples for name, value in totals.items()}


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("steps, samples, and batch-size must be positive")
    torch.manual_seed(args.seed)
    data = {name: value.to(args.device) for name, value in load_real_pairs(
        args.data, args.samples
    ).items()}
    config = CounterfactualRWKV7Config(
        latent_dim=data["initial"].shape[-1],
        action_dim=data["factual_action"].shape[-1],
        model_dim=32,
        num_layers=2,
        num_heads=4,
        channel_mlp_dim=128,
        action_hidden_dim=16,
        centered=True,
    )
    model = CounterfactualRWKV7WorldPredictor(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    initial = evaluate(model, data, args.batch_size)
    generator = torch.Generator().manual_seed(args.seed + 1)
    history = []
    for step in range(1, args.steps + 1):
        indices = torch.randint(args.samples, (args.batch_size,), generator=generator).to(
            args.device
        )
        batch = {name: value.index_select(0, indices) for name, value in data.items()}
        factual, noop, _ = paired_predictions(model, batch, diagnostics=False)
        loss, ordinary, effect = losses(factual, noop, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 20 == 0 or step == args.steps:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "ordinary_mse": float(ordinary.detach()),
                    "effect_mse": float(effect.detach()),
                    "gradient_norm": float(gradient_norm),
                }
            )
    final = evaluate(model, data, args.batch_size)
    summary = {
        "schema_version": "cc_rwkv_m3_log_hazard_smoke_v1",
        "protocol_status": "engineering_smoke_real_m2_pairs_not_paper_result",
        "data": str(args.data),
        "steps": args.steps,
        "paired_samples": args.samples,
        "initial": initial,
        "final": final,
        "effect_relative_reduction": (
            initial["effect"] - final["effect"]
        ) / max(initial["effect"], 1e-12),
        "ordinary_relative_reduction": (
            initial["ordinary"] - final["ordinary"]
        ) / max(initial["ordinary"], 1e-12),
        "finite": bool(all(np.isfinite(value) for value in final.values())),
        "gate_not_saturated": final["saturated"] < 1.0,
        "parameter_count": model.parameter_count(),
        "config": config.as_dict(),
        "history": history,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "config": config.as_dict(), "summary": summary},
        args.output / "b6_m3_smoke.pt",
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
