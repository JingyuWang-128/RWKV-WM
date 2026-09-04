#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cape_wm.calibration import fit_from_records, save_calibration_artifact
from cape_wm.config import load_config
from cape_wm.data import load_calibration_records
from cape_wm.device import resolve_device
from cape_wm.models import ExecutabilityRiskHead
from cape_wm.torch_wrappers import TorchRiskPredictor
from cape_wm.training import save_checkpoint, set_reproducible_seed, train_risk_epoch


class RiskDataset(Dataset):
    def __init__(self, records, max_duration: int, miss_scale: float = 1.0) -> None:
        self.records = records
        self.max_duration = max_duration
        self.miss_scale = float(miss_scale)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        residuals = np.asarray(record.step_residuals, dtype=np.float32)
        if residuals.size > self.max_duration:
            raise ValueError("record exceeds the configured maximum duration")
        scales = np.ones(self.max_duration, dtype=np.float32)
        mask = np.zeros(self.max_duration, dtype=np.bool_)
        scales[: residuals.size] = np.maximum(residuals, 1e-6)
        mask[: residuals.size] = True
        return {
            "current": torch.from_numpy(np.asarray(record.current_latent, dtype=np.float32)),
            "subgoal": torch.from_numpy(np.asarray(record.subgoal_latent, dtype=np.float32)),
            "duration": torch.as_tensor(record.duration, dtype=torch.long),
            "observed_miss": torch.as_tensor(
                record.observed_miss / self.miss_scale, dtype=torch.float32
            ),
            "success": torch.as_tensor(record.success, dtype=torch.float32),
            "residual_scale": torch.from_numpy(scales),
            "residual_mask": torch.from_numpy(mask),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and calibrate the CAPE-WM risk head")
    parser.add_argument("train_records", type=Path)
    parser.add_argument("calibration_records", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/runtime.yaml"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--miss-scale",
        type=float,
        default=1.0,
        help="positive label scale; predictions are converted back to environment units",
    )
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    if args.miss_scale <= 0:
        raise SystemExit("--miss-scale must be positive")

    runtime_config = load_config(args.runtime_config)
    runtime = runtime_config["runtime"]
    training_config = runtime_config["training"]
    device = resolve_device(args.device or runtime["device"])
    batch_size = args.batch_size or int(training_config["risk_batch_size"])
    num_workers = args.num_workers if args.num_workers is not None else int(runtime["num_workers"])
    pin_memory = (
        device.type == "cuda"
        if runtime.get("pin_memory", "auto") == "auto"
        else bool(runtime["pin_memory"])
    )
    print(json.dumps({"event": "runtime", "device": str(device)}))
    set_reproducible_seed(args.seed, args.deterministic)
    training_records = load_calibration_records(args.train_records)
    calibration_records = load_calibration_records(args.calibration_records)
    if not training_records or not calibration_records:
        raise SystemExit("both training and calibration attempt sets must be non-empty")
    latent_dim = int(np.asarray(training_records[0].current_latent).size)
    max_duration = max(record.duration for record in training_records + calibration_records)
    model = ExecutabilityRiskHead(latent_dim, max_duration=max_duration).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loader = DataLoader(
        RiskDataset(training_records, max_duration, args.miss_scale),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(args.seed),
    )
    history = []
    for epoch in range(args.epochs):
        loss = train_risk_epoch(model, loader, optimizer, device)
        metrics = {"epoch": epoch, "loss": loss}
        history.append(metrics)
        print(json.dumps(metrics))

    predictor = TorchRiskPredictor(model, device, miss_scale=args.miss_scale)
    calibrator = fit_from_records(calibration_records, predictor, alpha=args.alpha)
    save_checkpoint(
        args.output,
        {"risk_head": model},
        {
            "latent_dim": latent_dim,
            "max_duration": max_duration,
            "alpha": args.alpha,
            "miss_scale": args.miss_scale,
            "seed": args.seed,
            "device": str(device),
        },
        optimizer,
        {"history": history, "calibration": calibrator.artifact.as_dict()},
    )
    save_calibration_artifact(args.output.with_suffix(".calibration.json"), calibrator.artifact)


if __name__ == "__main__":
    main()
