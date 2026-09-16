#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from cape_wm.baseline import prepare_resources
from cape_wm.cc_rwkv.b0 import open_loop_rollout_latents
from cape_wm.cc_rwkv.protocol import HorizonSpec, array_sha256, file_sha256
from cape_wm.config import load_config


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _encode_rows(
    resources,
    data_path: Path,
    rows: np.ndarray,
    output: Path,
    batch_size: int,
    encoder_weights_sha256: str,
    rebuild: bool = False,
) -> np.ndarray:
    flat_rows = rows.reshape(-1)
    if np.any(flat_rows[1:] < flat_rows[:-1]):
        raise ValueError("rollout rows must be globally sorted for deterministic HDF5 reads")
    fingerprint = array_sha256(rows)
    latents_path = output / "target_latents.npy"
    metadata_path = output / "target_latents.json"
    if not rebuild and latents_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        latents = np.load(latents_path, mmap_mode="r")
        if metadata.get("row_fingerprint") != fingerprint:
            raise RuntimeError("existing latent cache belongs to different rollout rows")
        if metadata.get("encoder_weights_sha256") != encoder_weights_sha256:
            raise RuntimeError("existing latent cache belongs to different encoder weights")
        if tuple(latents.shape[:2]) != rows.shape:
            raise RuntimeError("existing latent cache has the wrong shape")
        return np.array(latents, copy=True)

    output.mkdir(parents=True, exist_ok=True)
    encoded_batches: list[np.ndarray] = []
    started = time.time()
    with h5py.File(data_path, "r") as handle, torch.inference_mode():
        pixels = handle["pixels"]
        for start in range(0, len(flat_rows), batch_size):
            batch_rows = flat_rows[start : start + batch_size]
            images = np.asarray(pixels[batch_rows])
            tensor = torch.stack([resources.image_transform(image) for image in images]).to(
                resources.device
            )
            latent = resources.model.encode({"pixels": tensor[:, None]})["emb"][:, -1]
            encoded_batches.append(latent.detach().float().cpu().numpy())
    encoded = np.concatenate(encoded_batches).reshape(*rows.shape, -1).astype(np.float32)
    np.save(latents_path, encoded)
    _write_json(
        metadata_path,
        {
            "status": "complete",
            "row_fingerprint": fingerprint,
            "encoder_weights_sha256": encoder_weights_sha256,
            "shape": list(encoded.shape),
            "dtype": str(encoded.dtype),
            "sha256": file_sha256(latents_path),
            "wall_seconds": time.time() - started,
        },
    )
    return encoded


def _summary_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    primitive_steps: np.ndarray,
    requested_horizons: tuple[int, ...],
) -> tuple[dict, dict[str, np.ndarray]]:
    pred = torch.from_numpy(predictions)
    target = torch.from_numpy(targets)
    squared = (pred - target).square()
    mse = squared.mean(dim=-1).numpy()
    l2 = squared.sum(dim=-1).sqrt().numpy()
    cosine = (1.0 - F.cosine_similarity(pred, target, dim=-1)).numpy()
    step_lookup = {int(step): index for index, step in enumerate(primitive_steps)}
    per_horizon: dict[str, dict[str, float]] = {}
    for horizon in requested_horizons:
        index = step_lookup[horizon]
        per_horizon[str(horizon)] = {
            "model_steps": index + 1,
            "primitive_steps": horizon,
            "latent_mse_mean": float(mse[:, index].mean()),
            "latent_mse_median": float(np.median(mse[:, index])),
            "latent_l2_mean": float(l2[:, index].mean()),
            "latent_l2_median": float(np.median(l2[:, index])),
            "cosine_distance_mean": float(cosine[:, index].mean()),
        }
    curve_steps = np.concatenate(([0], primitive_steps)).astype(np.float64)
    curve_l2 = np.concatenate(([0.0], l2.mean(axis=0)))
    curve_mse = np.concatenate(([0.0], mse.mean(axis=0)))
    summary = {
        "per_horizon": per_horizon,
        "trajectory_l2_auc": float(
            np.trapezoid(curve_l2, curve_steps) / float(curve_steps[-1])
        ),
        "trajectory_mse_auc": float(
            np.trapezoid(curve_mse, curve_steps) / float(curve_steps[-1])
        ),
        "per_model_step": {
            "primitive_steps": primitive_steps.tolist(),
            "latent_mse_mean": mse.mean(axis=0).tolist(),
            "latent_l2_mean": l2.mean(axis=0).tolist(),
            "cosine_distance_mean": cosine.mean(axis=0).tolist(),
        },
    }
    return summary, {"latent_mse": mse, "latent_l2": l2, "cosine_distance": cosine}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate official B0 LeWM open-loop")
    parser.add_argument("--config", type=Path, default=Path("configs/cc_rwkv/tworoom.yaml"))
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/runtime.yaml"))
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rebuild-latents", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    protocol = config["protocol"]
    evaluation = config["evaluation"]
    assets = {key: Path(value).resolve() for key, value in config["assets"].items()}
    manifest_dir = (args.manifest_dir or Path(config["paths"]["manifest_dir"])).resolve()
    output = (args.output or Path(config["paths"]["b0_output_dir"])).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_metadata = json.loads((manifest_dir / "manifest.json").read_text())
    if file_sha256(manifest_dir / "open_loop_manifest.npz") != manifest_metadata[
        "manifest_npz_sha256"
    ]:
        raise RuntimeError("open-loop manifest hash does not match frozen metadata")
    values = np.load(manifest_dir / "open_loop_manifest.npz")
    count = (
        len(values["episode_id"])
        if args.limit is None
        else min(args.limit, len(values["episode_id"]))
    )
    rows = np.asarray(values["rollout_rows"][:count], dtype=np.int64)
    raw_actions = np.asarray(values["action_blocks_raw"][:count], dtype=np.float32)
    action_mean = np.asarray(values["action_mean"], dtype=np.float32)
    action_scale = np.asarray(values["action_scale"], dtype=np.float32)
    repeat = raw_actions.shape[-1] // len(action_mean)
    normalized_actions = (raw_actions - np.tile(action_mean, repeat)) / np.tile(
        action_scale, repeat
    )

    horizon = HorizonSpec(
        primitive_horizons=tuple(map(int, protocol["primitive_horizons"])),
        action_block=int(protocol["action_block"]),
        observation_stride=int(protocol["observation_stride"]),
    )
    resources = prepare_resources(
        data=assets["data"],
        weights=assets["weights"],
        experiment_config=Path("configs/lewm_tworooms_baseline.yaml"),
        runtime_config=args.runtime_config,
        cache_dir=output / "stablewm_cache",
        device_override=args.device,
    )
    target_latents = _encode_rows(
        resources,
        assets["data"],
        rows,
        output,
        int(evaluation["encoder_batch_size"]),
        manifest_metadata["asset_hashes"]["weights"],
        rebuild=args.rebuild_latents,
    )

    predictions: list[np.ndarray] = []
    rollout_started = time.time()
    batch_size = int(evaluation["rollout_batch_size"])
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            end = min(count, start + batch_size)
            context = torch.from_numpy(target_latents[start:end, :1]).to(resources.device)
            actions = torch.from_numpy(normalized_actions[start:end]).to(resources.device)
            prediction = open_loop_rollout_latents(resources.model, context, actions)
            predictions.append(prediction.detach().float().cpu().numpy())
    predicted = np.concatenate(predictions).astype(np.float32)
    targets = np.asarray(target_latents[:, 1:], dtype=np.float32)
    primitive_steps = np.asarray(values["primitive_steps"][1:], dtype=np.int64)
    metrics, error_arrays = _summary_metrics(
        predicted, targets, primitive_steps, horizon.primitive_horizons
    )
    paired_array_hashes = {
        name: array_sha256(value) for name, value in error_arrays.items()
    }
    np.savez_compressed(
        output / "paired_errors.npz",
        sample_id=values["sample_id"][:count],
        episode_id=values["episode_id"][:count],
        primitive_steps=primitive_steps,
        **error_arrays,
    )
    predictor_parameters = sum(
        parameter.numel() for parameter in resources.model.predictor.parameters()
    )
    total_parameters = sum(parameter.numel() for parameter in resources.model.parameters())
    summary = {
        "status": "complete",
        "method": "B0 official Transformer-LeWM",
        "protocol": {
            "no_intermediate_observations": True,
            "controller": "none",
            "history_frames": int(protocol["history_frames"]),
            "horizon": horizon.as_dict(),
            "samples": count,
            "full_manifest_samples": len(values["episode_id"]),
        },
        "metrics": metrics,
        "model": {
            "predictor_parameters": predictor_parameters,
            "total_parameters": total_parameters,
            "predicted_transitions": int(count * horizon.max_model_horizon),
        },
        "runtime": {
            "device": str(resources.device),
            "rollout_wall_seconds": time.time() - rollout_started,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(resources.device))
                if resources.device.type == "cuda"
                else 0
            ),
        },
        "provenance": {
            "manifest_npz_sha256": manifest_metadata["manifest_npz_sha256"],
            "weights_sha256": manifest_metadata["asset_hashes"]["weights"],
            "data_sha256": manifest_metadata["asset_hashes"]["data"],
            "target_latents_sha256": file_sha256(output / "target_latents.npy"),
            "paired_errors_sha256": file_sha256(output / "paired_errors.npz"),
            "paired_array_hashes": paired_array_hashes,
        },
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "resolved_config.json", config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
