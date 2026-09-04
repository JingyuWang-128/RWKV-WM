#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cape_wm.config import load_config
from cape_wm.data import iter_macro_segments, load_npz_trajectories
from cape_wm.device import resolve_device
from cape_wm.models import DurationConditionedMacroPredictor, MacroActionEncoder
from cape_wm.training import save_checkpoint, set_reproducible_seed, train_macro_epoch


class SegmentDataset(Dataset):
    def __init__(self, segments, max_duration: int) -> None:
        self.segments = list(segments)
        self.max_duration = max_duration

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int):
        segment = self.segments[index]
        actions = np.zeros((self.max_duration, segment.actions.shape[-1]), dtype=np.float32)
        actions[: segment.duration] = segment.actions
        return {
            "current": torch.from_numpy(np.asarray(segment.start_latent, dtype=np.float32)),
            "target": torch.from_numpy(np.asarray(segment.target_latent, dtype=np.float32)),
            "actions": torch.from_numpy(actions),
            "duration": torch.as_tensor(segment.duration, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CAPE-WM duration-conditioned macro dynamics"
    )
    parser.add_argument("trajectories", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--durations", type=int, nargs="+", default=[5, 10, 20, 40])
    parser.add_argument("--macro-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
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
    batch_size = args.batch_size or int(training_config["macro_batch_size"])
    num_workers = args.num_workers if args.num_workers is not None else int(runtime["num_workers"])
    pin_memory = (
        device.type == "cuda"
        if runtime.get("pin_memory", "auto") == "auto"
        else bool(runtime["pin_memory"])
    )
    print(json.dumps({"event": "runtime", "device": str(device)}))
    set_reproducible_seed(args.seed, args.deterministic)
    trajectories = load_npz_trajectories(args.trajectories)
    if args.manifest:
        train_ids = set(json.loads(args.manifest.read_text())["train"])
        trajectories = [item for item in trajectories if item.trajectory_id in train_ids]
    segments = list(iter_macro_segments(trajectories, tuple(args.durations)))
    if not segments:
        raise SystemExit("no valid macro segments were found")
    latent_dim = int(np.asarray(segments[0].start_latent).size)
    action_dim = int(segments[0].actions.shape[-1])
    max_duration = max(args.durations)
    dataset = SegmentDataset(segments, max_duration)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        drop_last=False,
    )
    action_encoder = MacroActionEncoder(
        action_dim=action_dim,
        macro_dim=args.macro_dim,
        max_duration=max_duration,
    ).to(device)
    predictor = DurationConditionedMacroPredictor(
        latent_dim=latent_dim,
        macro_dim=args.macro_dim,
        max_duration=max_duration,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(action_encoder.parameters()) + list(predictor.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    history = []
    for epoch in range(args.epochs):
        metrics = train_macro_epoch(action_encoder, predictor, loader, optimizer, device)
        metrics["epoch"] = epoch
        history.append(metrics)
        print(json.dumps(metrics))
    save_checkpoint(
        args.output,
        {"action_encoder": action_encoder, "macro_predictor": predictor},
        {
            "latent_dim": latent_dim,
            "action_dim": action_dim,
            "macro_dim": args.macro_dim,
            "durations": args.durations,
            "max_duration": max_duration,
            "seed": args.seed,
            "device": str(device),
        },
        optimizer,
        {"history": history},
    )


if __name__ == "__main__":
    main()
