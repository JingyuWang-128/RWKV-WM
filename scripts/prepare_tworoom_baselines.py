#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from cape_wm.baseline import prepare_resources
from cape_wm.config import load_config
from cape_wm.tworoom_data import (
    episode_tables,
    make_split,
    sample_segments,
    sample_temporal_pairs,
    test_episode_ids,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _map_rows(rows: np.ndarray, selected: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(rows, selected)
    if np.any(positions >= len(rows)) or not np.array_equal(rows[positions], selected):
        raise RuntimeError("latent row mapping is incomplete")
    return positions.astype(np.int64)


def _save_trm(path: Path, pairs: dict[str, np.ndarray], rows: np.ndarray) -> None:
    np.savez(
        path,
        source=_map_rows(rows, pairs["source_rows"]),
        goal=_map_rows(rows, pairs["goal_rows"]),
        targets=pairs["targets"].astype(np.float32),
        separation=pairs["separation"].astype(np.int64),
        episode_ids=pairs["episode_ids"].astype(np.int64),
    )


def _standardize_actions(
    segments: dict[str, np.ndarray], mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    actions = segments["actions"].copy()
    durations = segments["durations"]
    for index, duration in enumerate(durations):
        actions[index, : int(duration)] = (actions[index, : int(duration)] - mean) / scale
    return actions


def _save_macro(
    path: Path,
    segments: dict[str, np.ndarray],
    rows: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> None:
    np.savez(
        path,
        source=_map_rows(rows, segments["source_rows"]),
        target=_map_rows(rows, segments["target_rows"]),
        durations=segments["durations"].astype(np.int64),
        episode_ids=segments["episode_ids"].astype(np.int64),
        actions=_standardize_actions(segments, mean, scale),
    )


def _save_vlwm(
    path: Path,
    segments: dict[str, np.ndarray],
    rows: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    block_size: int,
    max_blocks: int,
) -> None:
    standardized = _standardize_actions(segments, mean, scale)
    block_actions = np.zeros(
        (len(standardized), max_blocks, block_size * standardized.shape[-1]),
        dtype=np.float32,
    )
    lengths = segments["durations"] // block_size
    for index, blocks in enumerate(lengths):
        block_actions[index, :blocks] = standardized[index, : int(blocks) * block_size].reshape(
            int(blocks), -1
        )
    np.savez(
        path,
        source=_map_rows(rows, segments["source_rows"]),
        target=_map_rows(rows, segments["target_rows"]),
        lengths=lengths.astype(np.int64),
        episode_ids=segments["episode_ids"].astype(np.int64),
        actions=block_actions,
    )


def _cache_latents(
    output: Path,
    data: Path,
    rows: np.ndarray,
    resources,
    batch_size: int,
) -> dict[str, object]:
    rows_path = output / "latent_rows.npy"
    latents_path = output / "latents.npy"
    progress_path = output / "latent_progress.json"
    fingerprint = hashlib.sha256(rows.tobytes()).hexdigest()
    if rows_path.exists():
        existing = np.load(rows_path, mmap_mode="r")
        if not np.array_equal(existing, rows):
            raise RuntimeError("existing latent row manifest does not match this protocol")
    else:
        np.save(rows_path, rows)

    completed = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("row_fingerprint") != fingerprint:
            raise RuntimeError("latent progress belongs to a different row manifest")
        completed = int(progress.get("completed", 0))
    if latents_path.exists():
        latents = np.lib.format.open_memmap(latents_path, mode="r+")
        if latents.shape != (len(rows), 192):
            raise RuntimeError("existing latent cache has the wrong shape")
    else:
        latents = np.lib.format.open_memmap(
            latents_path, mode="w+", dtype=np.float32, shape=(len(rows), 192)
        )
        completed = 0

    started = time.time()
    with h5py.File(data, "r") as handle, torch.inference_mode():
        pixels = handle["pixels"]
        for start in range(completed, len(rows), batch_size):
            end = min(len(rows), start + batch_size)
            row_batch = rows[start:end]
            images = np.asarray(pixels[row_batch])
            tensor = torch.stack([resources.image_transform(image) for image in images]).to(
                resources.device
            )
            encoded = resources.model.encode({"pixels": tensor[:, None]})["emb"][:, -1]
            latents[start:end] = encoded.detach().float().cpu().numpy()
            latents.flush()
            _write_json(
                progress_path,
                {
                    "status": "running" if end < len(rows) else "complete",
                    "completed": end,
                    "total": len(rows),
                    "row_fingerprint": fingerprint,
                    "updated_at_unix": time.time(),
                },
            )
            if end == len(rows) or end % (10 * batch_size) == 0:
                elapsed = max(time.time() - started, 1e-9)
                print(
                    json.dumps(
                        {
                            "event": "latent_cache_progress",
                            "completed": end,
                            "total": len(rows),
                            "images_per_second": (end - completed) / elapsed,
                        }
                    ),
                    flush=True,
                )
    return {
        "rows": len(rows),
        "latent_dim": int(latents.shape[1]),
        "row_fingerprint": fingerprint,
        "latents_sha256": _sha256(latents_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare leak-free Two-Room baseline data")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--matrix-episodes",
        type=Path,
        default=Path("artifacts/results/lewm_tworooms_long_matrix/matrix_episodes.jsonl"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/tworoom_baselines.yaml"))
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/lewm_tworooms_baseline.yaml"),
    )
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/runtime.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/cache/tworoom_baselines"))
    parser.add_argument("--device")
    args = parser.parse_args()

    config = load_config(args.config)
    runtime = load_config(args.runtime_config)["runtime"]
    torch.set_num_threads(int(runtime.get("torch_num_threads", torch.get_num_threads())))
    if "torch_num_interop_threads" in runtime:
        torch.set_num_interop_threads(int(runtime["torch_num_interop_threads"]))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = config["protocol"]
    data_config = config["data"]
    block_size = int(protocol["action_block"])

    with h5py.File(args.data, "r") as handle:
        all_episodes, offsets, lengths = episode_tables(handle)
        actions = np.asarray(handle["action"], dtype=np.float32)
        episode_column = np.asarray(handle["ep_idx"], dtype=np.int64)
    held_out = test_episode_ids(args.matrix_episodes)
    split = make_split(
        all_episodes,
        held_out,
        seed=int(protocol["split_seed"]),
        train_fraction=float(protocol["train_fraction"]),
    )
    _write_json(output / "split_manifest.json", split.as_dict())

    valid_train_actions = actions[
        np.isin(episode_column, np.asarray(split.train)) & np.isfinite(actions).all(axis=1)
    ]
    action_mean = valid_train_actions.mean(axis=0).astype(np.float32)
    action_scale = valid_train_actions.std(axis=0).astype(np.float32)
    if np.any(action_scale <= 0):
        raise RuntimeError("training action scale is degenerate")

    trm_train = sample_temporal_pairs(
        split.train,
        all_episodes,
        offsets,
        lengths,
        count=int(data_config["trm_train_pairs"]),
        seed=int(protocol["split_seed"]),
        block_size=block_size,
        label_scale=float(config["trm"]["label_scale"]),
    )
    trm_validation = sample_temporal_pairs(
        split.validation,
        all_episodes,
        offsets,
        lengths,
        count=int(data_config["trm_validation_pairs"]),
        seed=int(protocol["split_seed"]) + 1,
        block_size=block_size,
        label_scale=float(config["trm"]["label_scale"]),
    )
    macro_train = sample_segments(
        split.train,
        all_episodes,
        offsets,
        lengths,
        actions,
        count=int(data_config["dynamics_train_segments"]),
        seed=int(protocol["split_seed"]) + 2,
        durations=tuple(map(int, data_config["durations"])),
        block_size=block_size,
    )
    macro_validation = sample_segments(
        split.validation,
        all_episodes,
        offsets,
        lengths,
        actions,
        count=int(data_config["dynamics_validation_segments"]),
        seed=int(protocol["split_seed"]) + 3,
        durations=tuple(map(int, data_config["durations"])),
        block_size=block_size,
    )
    vlwm_durations = tuple(
        block_size * item for item in range(1, int(config["vlwm"]["max_horizon_blocks"]) + 1)
    )
    vlwm_train = sample_segments(
        split.train,
        all_episodes,
        offsets,
        lengths,
        actions,
        count=int(data_config["dynamics_train_segments"]),
        seed=int(protocol["split_seed"]) + 4,
        durations=vlwm_durations,
        block_size=block_size,
    )
    vlwm_validation = sample_segments(
        split.validation,
        all_episodes,
        offsets,
        lengths,
        actions,
        count=int(data_config["dynamics_validation_segments"]),
        seed=int(protocol["split_seed"]) + 5,
        durations=vlwm_durations,
        block_size=block_size,
    )
    required_rows = np.unique(
        np.concatenate(
            [
                trm_train["source_rows"],
                trm_train["goal_rows"],
                trm_validation["source_rows"],
                trm_validation["goal_rows"],
                macro_train["source_rows"],
                macro_train["target_rows"],
                macro_validation["source_rows"],
                macro_validation["target_rows"],
                vlwm_train["source_rows"],
                vlwm_train["target_rows"],
                vlwm_validation["source_rows"],
                vlwm_validation["target_rows"],
            ]
        )
    )
    resources = prepare_resources(
        data=args.data,
        weights=args.weights,
        experiment_config=args.experiment_config,
        runtime_config=args.runtime_config,
        cache_dir=output / "stablewm_cache",
        device_override=args.device,
    )
    latent_metadata = _cache_latents(
        output,
        args.data,
        required_rows,
        resources,
        int(runtime.get("latent_batch_size", 64)),
    )
    _save_trm(output / "trm_train.npz", trm_train, required_rows)
    _save_trm(output / "trm_validation.npz", trm_validation, required_rows)
    _save_macro(output / "macro_train.npz", macro_train, required_rows, action_mean, action_scale)
    _save_macro(
        output / "macro_validation.npz",
        macro_validation,
        required_rows,
        action_mean,
        action_scale,
    )
    _save_vlwm(
        output / "vlwm_train.npz",
        vlwm_train,
        required_rows,
        action_mean,
        action_scale,
        block_size,
        int(config["vlwm"]["max_horizon_blocks"]),
    )
    _save_vlwm(
        output / "vlwm_validation.npz",
        vlwm_validation,
        required_rows,
        action_mean,
        action_scale,
        block_size,
        int(config["vlwm"]["max_horizon_blocks"]),
    )
    metadata = {
        "status": "complete",
        "reproduction_level": protocol["reproduction_level"],
        "data": str(args.data.resolve()),
        "weights": str(args.weights.resolve()),
        "matrix_episodes": str(args.matrix_episodes.resolve()),
        "resolved_device": str(resources.device),
        "torch_num_threads": torch.get_num_threads(),
        "test_episode_count": len(split.test),
        "train_episode_count": len(split.train),
        "validation_episode_count": len(split.validation),
        "calibration_episode_count": len(split.calibration),
        "action_mean": action_mean.tolist(),
        "action_scale": action_scale.tolist(),
        "latent_cache": latent_metadata,
        "counts": {
            "trm_train": len(trm_train["targets"]),
            "trm_validation": len(trm_validation["targets"]),
            "macro_train": len(macro_train["durations"]),
            "macro_validation": len(macro_validation["durations"]),
            "vlwm_train": len(vlwm_train["durations"]),
            "vlwm_validation": len(vlwm_validation["durations"]),
        },
        "finished_at_unix": time.time(),
    }
    _write_json(output / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
