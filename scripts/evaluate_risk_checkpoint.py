#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from cape_wm.checkpoints import load_risk_model
from cape_wm.conformal import conformal_quantile
from cape_wm.data import load_calibration_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate risk heads on candidate records")
    parser.add_argument("records", type=Path)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = load_calibration_records(args.records)
    current = np.stack([record.current_latent for record in records])
    subgoal = np.stack([record.subgoal_latent for record in records])
    duration = np.asarray([record.duration for record in records], dtype=np.int64)
    observed = np.asarray([record.observed_miss for record in records], dtype=np.float32)
    labels = np.asarray([record.success for record in records], dtype=bool)
    device = torch.device(args.device)
    results = []
    for checkpoint in args.checkpoints:
        model, config = load_risk_model(checkpoint, device)
        predicted_miss = []
        predicted_probability = []
        with torch.inference_mode():
            for start in range(0, len(records), args.batch_size):
                stop = min(start + args.batch_size, len(records))
                miss, probability, _ = model(
                    torch.as_tensor(current[start:stop], device=device),
                    torch.as_tensor(subgoal[start:stop], device=device),
                    torch.as_tensor(duration[start:stop], device=device),
                )
                predicted_miss.extend(
                    (miss * float(config.get("miss_scale", 1.0))).cpu().numpy().tolist()
                )
                predicted_probability.extend(probability.cpu().numpy().tolist())
        miss = np.asarray(predicted_miss)
        probability = np.asarray(predicted_probability)
        groups = {}
        for value in sorted(set(duration.tolist())):
            mask = duration == value
            groups[str(value)] = {
                "records": int(mask.sum()),
                "success_rate": float(labels[mask].mean()),
                "success_auc": float(roc_auc_score(labels[mask], probability[mask])),
                "brier": float(np.mean(np.square(probability[mask] - labels[mask]))),
                "miss_mae": float(np.mean(np.abs(miss[mask] - observed[mask]))),
                "miss_bias": float(np.mean(miss[mask] - observed[mask])),
            }
            failure_threshold = conformal_quantile(
                probability[mask & ~labels], alpha=0.1
            )
            accepted = mask & (probability > failure_threshold)
            groups[str(value)].update(
                {
                    "failure_control_probability_threshold": failure_threshold,
                    "accepted": int(accepted.sum()),
                    "accepted_precision": (
                        float(labels[accepted].mean()) if accepted.any() else 0.0
                    ),
                    "success_recall": float(
                        (accepted & labels).sum() / max(1, (mask & labels).sum())
                    ),
                }
            )
        results.append(
            {
                "checkpoint": str(checkpoint.resolve()),
                "records": len(records),
                "success_auc": float(roc_auc_score(labels, probability)),
                "brier": float(np.mean(np.square(probability - labels))),
                "miss_mae": float(np.mean(np.abs(miss - observed))),
                "miss_bias": float(np.mean(miss - observed)),
                "by_duration": groups,
            }
        )
    payload = {"status": "complete", "dataset": str(args.records.resolve()), "models": results}
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
