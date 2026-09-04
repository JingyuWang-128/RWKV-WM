from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.train_cc_rwkv import build_model
from cape_wm.cc_rwkv.per_step_training import InMemoryPerStepSplit, per_step_rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a per-step paired suffix checkpoint")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("b2", "b3", "b4", "b6"), required=True)
    parser.add_argument("--profile", choices=("smoke", "main"), default="main")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float(value[mask].mean())


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    dataset = InMemoryPerStepSplit(args.data, split=args.split)
    model = build_model(
        args.method,
        args.profile,
        int(dataset.tensors["factual_latents"].shape[-1]),
        int(dataset.tensors["factual_actions"].shape[-1]),
    ).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
    rows: list[dict] = []
    position_errors: list[list[float]] = []
    with torch.no_grad():
        for batch in dataset.batches(args.batch_size, device=args.device):
            output = per_step_rollout(model, batch, teacher_forcing=False)
            factual = output["factual_predicted"]
            factual_target = output["factual_target"]
            pulse = output["pulse_predicted"]
            pulse_target = output["pulse_target"]
            effect_target = output["effect_target"]
            mask = output["mask"]
            predicted_effect = pulse.new_zeros(pulse.shape)
            h = factual.shape[1]
            for position in range(h):
                length = h - position
                predicted_effect[:, position, :length] = (
                    pulse[:, position, :length]
                    - factual[:, position : position + length]
                )
            factual_err = (factual - factual_target).pow(2).mean(-1).sqrt()
            pulse_err = (pulse - pulse_target).pow(2).mean(-1).sqrt()
            effect_err = (predicted_effect - effect_target).pow(2).mean(-1).sqrt()
            for row_index, sample_id in enumerate(batch.sample_ids):
                sample_mask = mask[row_index]
                rows.append(
                    {
                        "sample_id": sample_id,
                        "factual_rmse": _mean(factual_err[row_index], torch.ones_like(factual_err[row_index], dtype=torch.bool)),
                        "noop_rmse": _mean(pulse_err[row_index], sample_mask),
                        "effect_rmse": _mean(effect_err[row_index], sample_mask),
                        "factual_one_step_rmse": float(factual_err[row_index, 0]),
                        "effect_one_step_rmse": _mean(effect_err[row_index, :, :1], sample_mask[:, :1]),
                        "effect_valid_fraction": float(sample_mask.float().mean()),
                    }
                )
            position_errors.extend(
                [
                    [
                        _mean(effect_err[:, position, : h - position], mask[:, position, : h - position])
                    ]
                    for position in range(h)
                ]
            )
    metrics = {
        "factual_rmse": sum(row["factual_rmse"] for row in rows) / len(rows),
        "noop_rmse": sum(row["noop_rmse"] for row in rows) / len(rows),
        "effect_rmse": sum(row["effect_rmse"] for row in rows) / len(rows),
        "factual_one_step_rmse": sum(row["factual_one_step_rmse"] for row in rows) / len(rows),
        "effect_one_step_rmse": sum(row["effect_one_step_rmse"] for row in rows) / len(rows),
        "effect_valid_fraction": sum(row["effect_valid_fraction"] for row in rows) / len(rows),
    }
    result = {
        "schema_version": "cc_rwkv_per_step_eval_v1",
        "method": args.method,
        "split": args.split,
        "samples": len(rows),
        "horizon": dataset.horizon,
        "metrics": metrics,
        "per_sample": rows,
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "effect_threshold": payload.get("effect_threshold"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("schema_version", "method", "split", "samples", "horizon", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
