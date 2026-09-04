#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from cape_wm.comparison_models import PairwiseReachabilityMetric, VariableLengthPredictor
from cape_wm.config import load_config
from cape_wm.device import resolve_device
from cape_wm.models import DurationConditionedMacroPredictor, LossWeights, MacroActionEncoder
from cape_wm.training import macro_kl_loss, set_reproducible_seed


class ArrayDataset(Dataset):
    def __init__(self, archive: Path, latents: np.ndarray, method: str) -> None:
        values = np.load(archive)
        self.values = {key: values[key] for key in values.files}
        self.latents = latents
        self.method = method

    def __len__(self) -> int:
        first = next(iter(self.values.values()))
        return len(first)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {
            "source": torch.from_numpy(
                np.asarray(self.latents[int(self.values["source"][index])], dtype=np.float32)
            ),
        }
        if self.method == "trm":
            result["goal"] = torch.from_numpy(
                np.asarray(self.latents[int(self.values["goal"][index])], dtype=np.float32)
            )
            result["target"] = torch.as_tensor(self.values["targets"][index], dtype=torch.float32)
        else:
            result["target"] = torch.from_numpy(
                np.asarray(self.latents[int(self.values["target"][index])], dtype=np.float32)
            )
            result["actions"] = torch.from_numpy(
                np.asarray(self.values["actions"][index], dtype=np.float32)
            )
            length_key = "durations" if self.method == "macro" else "lengths"
            result["length"] = torch.as_tensor(self.values[length_key][index], dtype=torch.long)
        return result


def _loader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,
    )


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.inference_mode()
def _trm_loss(
    model: PairwiseReachabilityMetric,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for raw in loader:
        batch = _move(raw, device)
        loss = F.smooth_l1_loss(model(batch["source"], batch["goal"]), batch["target"])
        total += float(loss) * len(batch["target"])
        count += len(batch["target"])
    return total / max(count, 1)


@torch.inference_mode()
def _macro_loss(
    encoder: MacroActionEncoder,
    predictor: DurationConditionedMacroPredictor,
    loader: DataLoader,
    device: torch.device,
) -> float:
    encoder.eval()
    predictor.eval()
    total = 0.0
    count = 0
    for raw in loader:
        batch = _move(raw, device)
        mean, _ = encoder(batch["actions"], batch["length"])
        prediction = predictor(batch["source"], mean, batch["length"])
        loss = F.smooth_l1_loss(prediction, batch["target"])
        total += float(loss) * len(batch["target"])
        count += len(batch["target"])
    return total / max(count, 1)


@torch.inference_mode()
def _vlwm_loss(
    model: VariableLengthPredictor,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for raw in loader:
        batch = _move(raw, device)
        prediction = model(batch["source"], batch["actions"], batch["length"])
        loss = F.mse_loss(prediction, batch["target"])
        total += float(loss) * len(batch["target"])
        count += len(batch["target"])
    return total / max(count, 1)


def _checkpoint(
    path: Path,
    *,
    method: str,
    modules: dict[str, nn.Module],
    config: dict,
    history: list[dict],
    best_epoch: int,
    started: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "cape_wm_comparison_v1",
            "method": method,
            "reproduction_level": "paper_spec_tworoom_adaptation",
            "modules": {name: module.state_dict() for name, module in modules.items()},
            "config": config,
            "history": history,
            "best_epoch": best_epoch,
            "training_wall_seconds": time.time() - started,
            "finished_at_unix": time.time(),
        },
        path,
    )


def train_trm(
    cache: Path,
    output: Path,
    config: dict,
    device: torch.device,
    runtime: dict,
    seed: int,
) -> None:
    section = config["trm"]
    latents = np.load(cache / "latents.npy", mmap_mode="r")
    train = ArrayDataset(cache / "trm_train.npz", latents, "trm")
    validation = ArrayDataset(cache / "trm_validation.npz", latents, "trm")
    batch_size = int(section["batch_size"])
    pin = device.type == "cuda" if runtime["pin_memory"] == "auto" else bool(runtime["pin_memory"])
    train_loader = _loader(
        train,
        batch_size,
        shuffle=True,
        seed=seed,
        num_workers=int(runtime["num_workers"]),
        pin_memory=pin,
    )
    validation_loader = _loader(
        validation,
        batch_size,
        shuffle=False,
        seed=seed,
        num_workers=int(runtime["num_workers"]),
        pin_memory=pin,
    )
    model = PairwiseReachabilityMetric(
        latent_dim=latents.shape[1], hidden_dim=int(section["hidden_dim"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    history: list[dict] = []
    best = float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    started = time.time()
    for epoch in range(int(section["epochs"])):
        model.train()
        total = 0.0
        count = 0
        for raw in train_loader:
            batch = _move(raw, device)
            loss = F.smooth_l1_loss(model(batch["source"], batch["goal"]), batch["target"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total += float(loss) * len(batch["target"])
            count += len(batch["target"])
        metrics = {
            "epoch": epoch,
            "train_loss": total / max(count, 1),
            "validation_loss": _trm_loss(model, validation_loader, device),
        }
        history.append(metrics)
        print(json.dumps({"method": "flat_trm", **metrics}), flush=True)
        if metrics["validation_loss"] < best:
            best = metrics["validation_loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(section["patience"]):
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    _checkpoint(
        output / "trm.pt",
        method="flat_trm",
        modules={"trm": model},
        config={
            "latent_dim": int(latents.shape[1]),
            "hidden_dim": int(section["hidden_dim"]),
            "label_scale": float(section["label_scale"]),
            "hybrid_trm_weight": float(section["hybrid_trm_weight"]),
            "hybrid_l2_weight": float(section["hybrid_l2_weight"]),
            "seed": seed,
            "device": str(device),
        },
        history=history,
        best_epoch=best_epoch,
        started=started,
    )


def train_macro(
    cache: Path,
    output: Path,
    config: dict,
    device: torch.device,
    runtime: dict,
    seed: int,
) -> None:
    section = config["macro"]
    latents = np.load(cache / "latents.npy", mmap_mode="r")
    train = ArrayDataset(cache / "macro_train.npz", latents, "macro")
    validation = ArrayDataset(cache / "macro_validation.npz", latents, "macro")
    pin = device.type == "cuda" if runtime["pin_memory"] == "auto" else bool(runtime["pin_memory"])
    train_loader = _loader(
        train,
        int(section["batch_size"]),
        shuffle=True,
        seed=seed,
        num_workers=int(runtime["num_workers"]),
        pin_memory=pin,
    )
    validation_loader = _loader(
        validation,
        int(section["batch_size"]),
        shuffle=False,
        seed=seed,
        num_workers=int(runtime["num_workers"]),
        pin_memory=pin,
    )
    encoder = MacroActionEncoder(
        action_dim=2,
        macro_dim=int(section["macro_dim"]),
        model_dim=192,
        depth=2,
        heads=4,
        max_duration=int(section["max_duration"]),
    ).to(device)
    predictor = DurationConditionedMacroPredictor(
        latent_dim=latents.shape[1],
        macro_dim=int(section["macro_dim"]),
        hidden_dim=512,
        depth=3,
        max_duration=int(section["max_duration"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    history: list[dict] = []
    best = float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    started = time.time()
    weights = LossWeights()
    for epoch in range(int(section["epochs"])):
        encoder.train()
        predictor.train()
        total = 0.0
        count = 0
        for raw in train_loader:
            batch = _move(raw, device)
            mean, log_std = encoder(batch["actions"], batch["length"])
            macro = encoder.sample(mean, log_std)
            prediction = predictor(batch["source"], macro, batch["length"])
            prediction_loss = F.smooth_l1_loss(prediction, batch["target"])
            half = torch.div(batch["length"], 2, rounding_mode="floor").clamp_min(1)
            second_lengths = batch["length"] - half
            first_actions = batch["actions"].new_zeros(batch["actions"].shape)
            second_actions = batch["actions"].new_zeros(batch["actions"].shape)
            for item, (duration, split) in enumerate(
                zip(batch["length"].tolist(), half.tolist(), strict=True)
            ):
                first_actions[item, :split] = batch["actions"][item, :split]
                second_actions[item, : duration - split] = batch["actions"][item, split:duration]
            first_mean, _ = encoder(first_actions, half)
            second_mean, _ = encoder(second_actions, second_lengths)
            midpoint = predictor(batch["source"], first_mean, half)
            composed = predictor(midpoint, second_mean, second_lengths)
            consistency_loss = F.smooth_l1_loss(prediction, composed)
            loss = (
                weights.prediction * prediction_loss
                + weights.cross_scale * consistency_loss
                + weights.macro_kl * macro_kl_loss(mean, log_std)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(predictor.parameters()), 1.0
            )
            optimizer.step()
            total += float(loss) * len(batch["target"])
            count += len(batch["target"])
        metrics = {
            "epoch": epoch,
            "train_loss": total / max(count, 1),
            "validation_loss": _macro_loss(encoder, predictor, validation_loader, device),
        }
        history.append(metrics)
        print(json.dumps({"method": "macro", **metrics}), flush=True)
        if metrics["validation_loss"] < best:
            best = metrics["validation_loss"]
            best_epoch = epoch
            best_state = {
                "macro_encoder": copy.deepcopy(encoder.state_dict()),
                "macro_predictor": copy.deepcopy(predictor.state_dict()),
            }
            stale = 0
        else:
            stale += 1
        if stale >= int(section["patience"]):
            break
    assert best_state is not None
    encoder.load_state_dict(best_state["macro_encoder"])
    predictor.load_state_dict(best_state["macro_predictor"])
    _checkpoint(
        output / "macro.pt",
        method="hwm_shared_macro",
        modules={"macro_encoder": encoder, "macro_predictor": predictor},
        config={
            "latent_dim": int(latents.shape[1]),
            "action_dim": 2,
            "macro_dim": int(section["macro_dim"]),
            "model_dim": 192,
            "max_duration": int(section["max_duration"]),
            "seed": seed,
            "device": str(device),
        },
        history=history,
        best_epoch=best_epoch,
        started=started,
    )
    archive = np.load(cache / "macro_train.npz")
    fixed = np.flatnonzero(archive["durations"] == int(section["fixed_duration"]))
    rng = np.random.default_rng(seed)
    selected = rng.choice(
        fixed,
        size=min(int(section["empirical_bank_size"]), len(fixed)),
        replace=False,
    )
    bank_loader = _loader(
        Subset(train, selected.tolist()),
        int(section["batch_size"]),
        shuffle=False,
        seed=seed,
        num_workers=int(runtime["num_workers"]),
        pin_memory=pin,
    )
    anchors = []
    encoder.eval()
    with torch.inference_mode():
        for raw in bank_loader:
            batch = _move(raw, device)
            mean, _ = encoder(batch["actions"], batch["length"])
            anchors.append(mean.cpu().numpy().astype(np.float32))
    np.save(output / "empirical_macro_bank.npy", np.concatenate(anchors))


def train_vlwm(
    cache: Path,
    output: Path,
    config: dict,
    device: torch.device,
    runtime: dict,
    seed: int,
) -> None:
    section = config["vlwm"]
    latents = np.load(cache / "latents.npy", mmap_mode="r")
    train = ArrayDataset(cache / "vlwm_train.npz", latents, "vlwm")
    validation = ArrayDataset(cache / "vlwm_validation.npz", latents, "vlwm")
    pin = device.type == "cuda" if runtime["pin_memory"] == "auto" else bool(runtime["pin_memory"])
    validation_loader = _loader(
        validation,
        int(section["batch_size"]),
        shuffle=False,
        seed=seed,
        num_workers=int(runtime["num_workers"]),
        pin_memory=pin,
    )
    model = VariableLengthPredictor(
        latent_dim=latents.shape[1],
        action_dim=10,
        model_dim=int(section["model_dim"]),
        depth=int(section["depth"]),
        heads=int(section["heads"]),
        mlp_dim=int(section["mlp_dim"]),
        max_horizon=int(section["max_horizon_blocks"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    all_lengths = train.values["lengths"]
    history: list[dict] = []
    best = float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    started = time.time()
    epochs = int(section["epochs"])
    maximum = int(section["max_horizon_blocks"])
    for epoch in range(epochs):
        stage = min(maximum, max(1, int(np.ceil(maximum * (epoch + 1) / epochs))))
        indices = np.flatnonzero(all_lengths <= stage).tolist()
        train_loader = _loader(
            Subset(train, indices),
            int(section["batch_size"]),
            shuffle=True,
            seed=seed + epoch,
            num_workers=int(runtime["num_workers"]),
            pin_memory=pin,
        )
        model.train()
        total = 0.0
        count = 0
        for raw in train_loader:
            batch = _move(raw, device)
            prediction = model(batch["source"], batch["actions"], batch["length"])
            loss = F.mse_loss(prediction, batch["target"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss) * len(batch["target"])
            count += len(batch["target"])
        metrics = {
            "epoch": epoch,
            "curriculum_max_horizon": stage,
            "train_loss": total / max(count, 1),
            "validation_loss": _vlwm_loss(model, validation_loader, device),
        }
        history.append(metrics)
        print(json.dumps({"method": "vlwm", **metrics}), flush=True)
        if metrics["validation_loss"] < best:
            best = metrics["validation_loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        # Curriculum must reach its final stage before early stopping.
        if stage == maximum and stale >= int(section["patience"]):
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    _checkpoint(
        output / "vlwm.pt",
        method="vlwm_frozen_lewm_encoder",
        modules={"vlwm": model},
        config={
            "latent_dim": int(latents.shape[1]),
            "action_dim": 10,
            "model_dim": int(section["model_dim"]),
            "depth": int(section["depth"]),
            "heads": int(section["heads"]),
            "mlp_dim": int(section["mlp_dim"]),
            "max_horizon": maximum,
            "seed": seed,
            "device": str(device),
        },
        history=history,
        best_epoch=best_epoch,
        started=started,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Two-Room comparison baselines")
    parser.add_argument("--cache", type=Path, default=Path("artifacts/cache/tworoom_baselines"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/checkpoints/tworoom_baselines")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/tworoom_baselines.yaml"))
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/runtime.yaml"))
    parser.add_argument(
        "--methods", nargs="+", choices=("trm", "macro", "vlwm"), default=("trm", "macro", "vlwm")
    )
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--macro-epochs", type=int)
    parser.add_argument("--macro-patience", type=int)
    parser.add_argument("--macro-batch-size", type=int)
    parser.add_argument("--macro-learning-rate", type=float)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.macro_epochs is not None:
        config["macro"]["epochs"] = int(args.macro_epochs)
    if args.macro_patience is not None:
        config["macro"]["patience"] = int(args.macro_patience)
    if args.macro_batch_size is not None:
        config["macro"]["batch_size"] = int(args.macro_batch_size)
    if args.macro_learning_rate is not None:
        config["macro"]["learning_rate"] = float(args.macro_learning_rate)
    runtime = load_config(args.runtime_config)["runtime"]
    torch.set_num_threads(int(runtime.get("torch_num_threads", torch.get_num_threads())))
    if "torch_num_interop_threads" in runtime:
        torch.set_num_interop_threads(int(runtime["torch_num_interop_threads"]))
    device = resolve_device(args.device or runtime["device"])
    set_reproducible_seed(args.seed)
    print(json.dumps({"event": "runtime", "device": str(device)}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if "trm" in args.methods:
        train_trm(args.cache, args.output, config, device, runtime, args.seed)
    if "macro" in args.methods:
        train_macro(args.cache, args.output, config, device, runtime, args.seed)
    if "vlwm" in args.methods:
        train_vlwm(args.cache, args.output, config, device, runtime, args.seed)


if __name__ == "__main__":
    main()
