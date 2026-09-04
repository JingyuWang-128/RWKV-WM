#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cape_wm.checkpoints import load_risk_model
from cape_wm.conformal import conformal_quantile
from cape_wm.data import load_calibration_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit duration-conditional failure control")
    parser.add_argument("records", type=Path)
    parser.add_argument("risk_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    records = load_calibration_records(args.records)
    model, _ = load_risk_model(args.risk_checkpoint, args.device)
    current = np.stack([record.current_latent for record in records])
    subgoal = np.stack([record.subgoal_latent for record in records])
    duration = np.asarray([record.duration for record in records], dtype=np.int64)
    success = np.asarray([record.success for record in records], dtype=bool)
    probability = []
    device = torch.device(args.device)
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            stop = min(start + args.batch_size, len(records))
            _, batch_probability, _ = model(
                torch.as_tensor(current[start:stop], device=device),
                torch.as_tensor(subgoal[start:stop], device=device),
                torch.as_tensor(duration[start:stop], device=device),
            )
            probability.extend(batch_probability.cpu().numpy().tolist())
    probability = np.asarray(probability)
    by_duration = {}
    for value in sorted(set(duration.tolist())):
        mask = duration == value
        failures = mask & ~success
        threshold = conformal_quantile(probability[failures], args.alpha)
        accepted = mask & (probability > threshold)
        by_duration[str(value)] = {
            "threshold": threshold,
            "n": int(mask.sum()),
            "n_success": int((mask & success).sum()),
            "n_failure": int(failures.sum()),
            "accepted": int(accepted.sum()),
            "accepted_precision": float(success[accepted].mean()) if accepted.any() else 0.0,
            "success_recall": float(
                (accepted & success).sum() / max(1, (mask & success).sum())
            ),
        }
    payload = {
        "alpha": args.alpha,
        "method": "split_conformal_duration_conditional_failure_control_v1",
        "records": str(args.records.resolve()),
        "risk_checkpoint": str(args.risk_checkpoint.resolve()),
        "by_duration": by_duration,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
