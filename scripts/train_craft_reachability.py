#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from cape_wm.config import load_config
from cape_wm.device import resolve_device
from cape_wm.models import (
    ContinuousReachabilityDistribution,
    DirectedReachabilityDistribution,
    continuous_reachability_loss,
)
from cape_wm.training import set_reproducible_seed, train_reachability_epoch


class ReachabilityPairDataset(Dataset):
    """Trajectory-only supervision; no positions or environment geometry."""

    def __init__(
        self, archive: Path, latents: np.ndarray, latent_rows: np.ndarray
    ) -> None:
        values = np.load(archive)
        required = {"source", "goal", "separation", "episode_ids"}
        missing = required - set(values.files)
        if missing:
            raise ValueError(f"reachability archive is missing {sorted(missing)}")
        self.values = {key: values[key] for key in values.files}
        self.latents = latents
        self.latent_rows = np.asarray(latent_rows, dtype=np.int64)
        self.row_to_latent = {
            int(raw_row): index for index, raw_row in enumerate(self.latent_rows.tolist())
        }

    def __len__(self) -> int:
        return len(self.values["source"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_index = int(self.values["source"][index])
        goal_index = int(self.values["goal"][index])
        # Existing TRM archives include both temporal orientations. CRAFT uses
        # the recorded forward order so the learned distribution remains
        # directed and can later represent irreversible manipulation.
        if self.latent_rows[source_index] > self.latent_rows[goal_index]:
            source_index, goal_index = goal_index, source_index
        source = np.array(self.latents[source_index], dtype=np.float32, copy=True)
        goal = np.array(self.latents[goal_index], dtype=np.float32, copy=True)
        separation = int(self.values["separation"][index])
        source_row = int(self.latent_rows[source_index])
        elapsed = max(5, (separation // 2 // 5) * 5)
        successor_index = self.row_to_latent.get(source_row + elapsed)
        has_interior_successor = successor_index is not None and elapsed < separation
        successor = (
            np.array(self.latents[successor_index], dtype=np.float32, copy=True)
            if has_interior_successor
            else source.copy()
        )
        return {
            "source": torch.from_numpy(source),
            "goal": torch.from_numpy(goal),
            "successor": torch.from_numpy(successor),
            "elapsed": torch.as_tensor(elapsed, dtype=torch.long),
            "semigroup_mask": torch.as_tensor(has_interior_successor),
            "separation": torch.as_tensor(separation, dtype=torch.long),
            "trajectory_index": torch.as_tensor(
                int(self.values["episode_ids"][index]), dtype=torch.long
            ),
        }


def _loader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.inference_mode()
def evaluate(
    model: DirectedReachabilityDistribution | ContinuousReachabilityDistribution,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_nll = 0.0
    total_mae = 0.0
    correct = 0
    count = 0
    for raw in loader:
        source = raw["source"].to(device)
        goal = raw["goal"].to(device)
        separation = raw["separation"].to(device)
        target = model.target_class(separation)
        if isinstance(model, ContinuousReachabilityDistribution):
            probabilities = model.bin_probabilities(source, goal)
            total_nll += float(
                continuous_reachability_loss(
                    model,
                    source,
                    goal,
                    separation,
                    mean_weight=0.0,
                )
                * len(source)
            )
            prediction_class = probabilities.argmax(dim=-1)
        else:
            logits = model(source, goal)
            total_nll += float(F.cross_entropy(logits, target, reduction="sum"))
            prediction_class = logits.argmax(dim=-1)
        mean, _ = model.expected_and_std(source, goal)
        total_mae += float((mean - separation.float()).abs().sum())
        correct += int((prediction_class == target).sum())
        count += len(source)
    return {
        "nll": total_nll / max(count, 1),
        "class_accuracy": correct / max(count, 1),
        "expected_step_mae": total_mae / max(count, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the CRAFT directed hitting-time distribution"
    )
    parser.add_argument(
        "--cache", type=Path, default=Path("artifacts/cache/tworoom_baselines")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/craft_phase_b.yaml"))
    parser.add_argument(
        "--runtime-config", type=Path, default=Path("configs/runtime.yaml")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/checkpoints/tworoom_craft/reachability.pt"),
    )
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    section = config["reachability"]
    runtime = load_config(args.runtime_config)["runtime"]
    device = resolve_device(args.device or runtime.get("device", "auto"))
    seed = int(section["seed"])
    set_reproducible_seed(seed)
    torch.set_num_threads(int(runtime.get("torch_num_threads", torch.get_num_threads())))

    latents = np.load(args.cache / "latents.npy", mmap_mode="r")
    latent_rows = np.load(args.cache / "latent_rows.npy", mmap_mode="r")
    train = ReachabilityPairDataset(args.cache / "trm_train.npz", latents, latent_rows)
    validation = ReachabilityPairDataset(
        args.cache / "trm_validation.npz", latents, latent_rows
    )
    pin_memory = device.type == "cuda"
    batch_size = int(section["batch_size"])
    train_loader = _loader(
        train,
        batch_size,
        shuffle=True,
        seed=seed,
        workers=int(runtime["num_workers"]),
        pin_memory=pin_memory,
    )
    validation_loader = _loader(
        validation,
        batch_size,
        shuffle=False,
        seed=seed,
        workers=int(runtime["num_workers"]),
        pin_memory=pin_memory,
    )
    family = str(section.get("family", "categorical"))
    if family == "continuous_lognormal":
        model = ContinuousReachabilityDistribution(
            latent_dim=int(latents.shape[1]),
            horizon_bins=tuple(map(int, section["horizon_bins"])),
            hidden_dim=int(section["hidden_dim"]),
            depth=int(section["depth"]),
            min_log_scale=float(section["min_log_scale"]),
            max_log_scale=float(section["max_log_scale"]),
        ).to(device)
    elif family == "categorical":
        model = DirectedReachabilityDistribution(
            latent_dim=int(latents.shape[1]),
            horizon_bins=tuple(map(int, section["horizon_bins"])),
            hidden_dim=int(section["hidden_dim"]),
            depth=int(section["depth"]),
            overflow_horizon=int(section["overflow_horizon"]),
        ).to(device)
    else:
        raise ValueError(f"unsupported reachability family {family!r}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    epochs = int(args.epochs or section["epochs"])
    best = float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    history: list[dict] = []
    started = time.time()
    for epoch in range(epochs):
        train_metrics = train_reachability_epoch(
            model,
            train_loader,
            optimizer,
            device,
            negative_weight=float(section["negative_weight"]),
            semigroup_weight=float(section["semigroup_weight"]),
            mean_weight=float(section.get("mean_weight", 1.0)),
            label_scale=float(section.get("label_scale", 100.0)),
        )
        validation_metrics = evaluate(model, validation_loader, device)
        metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if validation_metrics["nll"] < best:
            best = validation_metrics["nll"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(section["patience"]):
            break
    if best_state is None:
        raise RuntimeError("reachability training produced no checkpoint")
    model.load_state_dict(best_state)
    final_validation = evaluate(model, validation_loader, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "craft_reachability_v2",
            "method": "craft_wm",
            "modules": {"reachability": model.state_dict()},
            "config": {
                "latent_dim": int(latents.shape[1]),
                **section,
                "position_labels": False,
                "topology_artifact": False,
            },
            "history": history,
            "best_epoch": best_epoch,
            "validation": final_validation,
            "training_wall_seconds": time.time() - started,
        },
        args.output,
    )
    summary = {
        "status": "complete",
        "checkpoint": str(args.output.resolve()),
        "device": str(device),
        "best_epoch": best_epoch,
        "train_pairs": len(train),
        "validation_pairs": len(validation),
        "validation": final_validation,
        "uses_position_labels": False,
        "uses_topology": False,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
