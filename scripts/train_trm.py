#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cape_wm.config import load_config
from cape_wm.data import load_npz_trajectories
from cape_wm.device import resolve_device
from cape_wm.models import TrajectoryReachabilityMetric
from cape_wm.training import save_checkpoint, set_reproducible_seed, train_trm_epoch


class ReachabilityDataset(Dataset):
    def __init__(self, trajectories, max_horizon: int, stride: int) -> None:
        self.examples = []
        for trajectory_index, trajectory in enumerate(trajectories):
            if trajectory.latents is None:
                raise ValueError("TRM training requires cached latents")
            for start in range(0, len(trajectory.actions), stride):
                final = min(len(trajectory.actions), start + max_horizon)
                for goal in range(start + 1, final + 1, stride):
                    self.examples.append((trajectory_index, trajectory, start, goal))
        self.max_horizon = max_horizon

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        trajectory_index, trajectory, start, goal = self.examples[index]
        separation = goal - start
        return {
            "source": torch.from_numpy(np.asarray(trajectory.latents[start], dtype=np.float32)),
            "goal": torch.from_numpy(np.asarray(trajectory.latents[goal], dtype=np.float32)),
            "horizon": torch.as_tensor(separation, dtype=torch.long),
            "temporal_distance": torch.as_tensor(
                separation / self.max_horizon, dtype=torch.float32
            ),
            "trajectory_index": torch.as_tensor(trajectory_index, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the shared horizon-matched TRM")
    parser.add_argument("trajectories", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-horizon", type=int, default=100)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/runtime.yaml"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    runtime_config = load_config(args.runtime_config)
    runtime = runtime_config["runtime"]
    training_config = runtime_config["training"]
    device = resolve_device(args.device or runtime["device"])
    batch_size = args.batch_size or int(training_config["trm_batch_size"])
    num_workers = args.num_workers if args.num_workers is not None else int(runtime["num_workers"])
    pin_memory = (
        device.type == "cuda"
        if runtime.get("pin_memory", "auto") == "auto"
        else bool(runtime["pin_memory"])
    )
    print(json.dumps({"event": "runtime", "device": str(device)}))
    set_reproducible_seed(args.seed, args.deterministic)
    dataset = ReachabilityDataset(
        load_npz_trajectories(args.trajectories), args.max_horizon, args.stride
    )
    if not dataset:
        raise SystemExit("no TRM training pairs were generated")
    latent_dim = int(dataset[0]["source"].numel())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = TrajectoryReachabilityMetric(latent_dim=latent_dim, max_horizon=args.max_horizon).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history = []
    for epoch in range(args.epochs):
        loss = train_trm_epoch(model, loader, optimizer, device)
        metrics = {"epoch": epoch, "loss": loss}
        history.append(metrics)
        print(json.dumps(metrics))
    save_checkpoint(
        args.output,
        {"trm": model},
        {
            "latent_dim": latent_dim,
            "max_horizon": args.max_horizon,
            "stride": args.stride,
            "seed": args.seed,
            "device": str(device),
        },
        optimizer,
        {"history": history},
    )


if __name__ == "__main__":
    main()
