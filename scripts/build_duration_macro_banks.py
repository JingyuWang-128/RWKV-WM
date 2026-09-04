#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cape_wm.models import MacroActionEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build duration-specific empirical macro banks")
    parser.add_argument("segments", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-per-duration", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=8101)
    args = parser.parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = payload["config"]
    encoder = MacroActionEncoder(
        action_dim=int(config["action_dim"]),
        macro_dim=int(config["macro_dim"]),
        model_dim=int(config["model_dim"]),
        depth=2,
        heads=4,
        max_duration=int(config["max_duration"]),
    ).to(device)
    encoder.load_state_dict(payload["modules"]["macro_encoder"])
    encoder.eval().requires_grad_(False)
    archive = np.load(args.segments)
    actions = archive["actions"]
    durations = archive["durations"].astype(np.int64)
    rng = np.random.default_rng(args.seed)
    banks = {}
    metadata = {}
    with torch.inference_mode():
        for duration in sorted(set(durations.tolist())):
            indices = np.flatnonzero(durations == duration)
            if len(indices) > args.max_per_duration:
                indices = np.sort(
                    rng.choice(indices, args.max_per_duration, replace=False)
                )
            values = []
            for start in range(0, len(indices), args.batch_size):
                batch_indices = indices[start : start + args.batch_size]
                batch_actions = torch.as_tensor(
                    actions[batch_indices], dtype=torch.float32, device=device
                )
                batch_duration = torch.full(
                    (len(batch_indices),), duration, dtype=torch.long, device=device
                )
                mean, _ = encoder(batch_actions, batch_duration)
                values.append(mean.cpu().numpy().astype(np.float32))
            bank = np.concatenate(values)
            banks[str(duration)] = bank
            metadata[str(duration)] = {
                "records": len(bank),
                "mean_abs": float(np.abs(bank.mean(axis=0)).mean()),
                "mean_std": float(bank.std(axis=0).mean()),
                "abs_q95": float(np.quantile(np.abs(bank), 0.95)),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **banks)
    summary = {
        "status": "complete",
        "segments": str(args.segments.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "output": str(args.output.resolve()),
        "by_duration": metadata,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
