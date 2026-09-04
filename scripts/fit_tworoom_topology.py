#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - registers the dataset compression filter
import numpy as np


def _unique_indices(archive: np.lib.npyio.NpzFile) -> np.ndarray:
    return np.unique(np.concatenate((archive["source"], archive["target"]))).astype(
        np.int64
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a train-only latent topology probe and door prototypes"
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-position", type=float, default=112.0)
    parser.add_argument("--door-position", type=float, default=49.0)
    parser.add_argument("--side-offset", type=float, default=14.0)
    parser.add_argument("--ridge", type=float, default=1e-2)
    args = parser.parse_args()

    latents = np.load(args.cache / "latents.npy", mmap_mode="r")
    latent_rows = np.load(args.cache / "latent_rows.npy", mmap_mode="r")
    train_archive = np.load(args.cache / "macro_train.npz")
    validation_archive = np.load(args.cache / "macro_validation.npz")
    train_indices = _unique_indices(train_archive)
    validation_indices = _unique_indices(validation_archive)
    with h5py.File(args.data, "r") as handle:
        train_positions = np.asarray(
            handle["pos_agent"][np.asarray(latent_rows[train_indices])],
            dtype=np.float64,
        )
        validation_positions = np.asarray(
            handle["pos_agent"][np.asarray(latent_rows[validation_indices])],
            dtype=np.float64,
        )

    train_latents = np.asarray(latents[train_indices], dtype=np.float64)
    mean = train_latents.mean(axis=0)
    scale = train_latents.std(axis=0).clip(1e-6)
    standardized = (train_latents - mean) / scale
    target_mean = train_positions.mean(axis=0)
    centered_target = train_positions - target_mean
    gram = standardized.T @ standardized
    gram.flat[:: gram.shape[0] + 1] += float(args.ridge)
    weight = np.linalg.solve(gram, standardized.T @ centered_target)

    def predict(values: np.ndarray) -> np.ndarray:
        return ((values - mean) / scale) @ weight + target_mean

    train_rmse = float(np.sqrt(np.mean(np.square(predict(train_latents) - train_positions))))
    validation_latents = np.asarray(latents[validation_indices], dtype=np.float64)
    validation_rmse = float(
        np.sqrt(np.mean(np.square(predict(validation_latents) - validation_positions)))
    )

    left_target = np.asarray(
        [args.wall_position - args.side_offset, args.door_position], dtype=np.float64
    )
    right_target = np.asarray(
        [args.wall_position + args.side_offset, args.door_position], dtype=np.float64
    )

    def nearest(target: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        index = int(np.argmin(np.linalg.norm(train_positions - target, axis=1)))
        return (
            np.asarray(train_latents[index], dtype=np.float32),
            np.asarray(train_positions[index], dtype=np.float32),
            int(train_indices[index]),
        )

    left_latent, left_position, left_index = nearest(left_target)
    right_latent, right_position, right_index = nearest(right_target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        latent_mean=mean.astype(np.float32),
        latent_scale=scale.astype(np.float32),
        position_weight=weight.astype(np.float32),
        position_mean=target_mean.astype(np.float32),
        door_left_latent=left_latent,
        door_right_latent=right_latent,
        wall_position=np.float32(args.wall_position),
        door_position=np.float32(args.door_position),
        side_offset=np.float32(args.side_offset),
    )
    summary = {
        "status": "complete",
        "training_latents": len(train_indices),
        "validation_latents": len(validation_indices),
        "train_position_rmse": train_rmse,
        "validation_position_rmse": validation_rmse,
        "left_prototype_position": left_position.tolist(),
        "right_prototype_position": right_position.tolist(),
        "left_prototype_latent_index": left_index,
        "right_prototype_latent_index": right_index,
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
