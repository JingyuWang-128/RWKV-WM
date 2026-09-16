"""Read-only checkpoint gradient attribution; never creates an optimizer or trains."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from cape_wm.cc_rwkv.model_factory import build_rwkv_world_model
from cape_wm.cc_rwkv.per_step_training import PerStepBatch, per_step_loss


def load_batch(samples, rows):
    unique, inverse = np.unique(rows, return_inverse=True)
    names = {
        "history_latents": "history_latents",
        "history_actions": "history_actions_raw",
        "history_mask": "history_mask",
        "factual_actions": "factual_actions",
        "factual_latents": "factual_latents",
        "pulse_noop_actions": "pulse_noop_actions",
        "pulse_noop_latents": "pulse_noop_latents",
        "pulse_noop_mask": "pulse_noop_mask",
        "effect_latents": "effect_latents",
    }
    payload = {
        key: torch.from_numpy(np.asarray(samples[name][unique])[inverse])
        for key, name in names.items()
    }
    payload["sample_id"] = [str(int(row)) for row in rows]
    return PerStepBatch.from_mapping(payload, "cpu")


def group_name(name):
    if "action_" in name:
        return "action_branch"
    if "prediction_head" in name:
        return "prediction_head"
    if "latent_projection" in name:
        return "latent_projection"
    if "time_mix" in name:
        return "world_time_mix"
    if "channel_mix" in name:
        return "channel_mix"
    return "normalization_other"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", default="1,5,10,20")
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    payload = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    cfg = payload["run_config"]
    model = build_rwkv_world_model(
        cfg["method"], cfg["profile"], cfg["latent_dim"], cfg["action_dim"]
    ).eval()
    model.load_state_dict(payload["model"])
    del payload["optimizer"]
    parameter_list = list(model.named_parameters())
    parameters = [p for _, p in parameter_list]
    fingerprint = hashlib.sha256()
    for parameter in parameters:
        fingerprint.update(parameter.detach().numpy().tobytes())
    before = fingerprint.hexdigest()
    sizes = [p.numel() for p in parameters]
    names = [name for name, _ in parameter_list]
    weights = {
        "factual_prediction": 1.0,
        "noop_prediction": 1.0,
        "paired_effect": cfg["effect_weight"],
        "effect_direction": cfg["effect_weight"] * 0.1,
        "effect_magnitude": cfg["effect_weight"] * 0.1,
    }
    generator = torch.Generator()
    generator.set_state(payload["rng_states"]["batch_generator"].cpu())
    accumulation = cfg.get("gradient_accumulation", 1)
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": payload["step"],
        "method": cfg["method"],
        "effect_weight": cfg["effect_weight"],
        "effect_threshold": payload["effect_threshold"],
        "effective_batch": cfg["batch_size"] * accumulation,
        "device": "cpu",
        "updates": 0,
        "results": [],
        "note": "frozen checkpoint, future training sample RNG, not exact failure-step weights",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(cfg["data"], "r") as handle:
        samples = handle["samples"]
        train_rows = np.flatnonzero(np.asarray(samples["split"]) == 0)
        for batch_index in range(args.batches):
            microbatches = []
            for _ in range(accumulation):
                indices = torch.randint(len(train_rows), (cfg["batch_size"],), generator=generator)
                microbatches.append(load_batch(samples, train_rows[indices.numpy()]))
            for horizon in [int(item) for item in args.horizons.split(",")]:
                started = time.monotonic()
                vectors = {key: torch.zeros(sum(sizes)) for key in weights}
                losses = dict.fromkeys(weights, 0.0)
                for batch in microbatches:
                    total, components = per_step_loss(
                        model,
                        batch,
                        horizon=horizon,
                        effect_threshold=payload["effect_threshold"],
                        effect_weight=cfg["effect_weight"],
                        allow_paired_loss=True,
                    )
                    for key, weight in weights.items():
                        value = components[key]
                        losses[key] += float(value.detach()) / accumulation
                        if value.requires_grad:
                            gradients = torch.autograd.grad(
                                value * (weight / accumulation),
                                parameters,
                                retain_graph=True,
                                allow_unused=True,
                            )
                            flat = torch.cat(
                                [
                                    torch.zeros_like(parameter).reshape(-1)
                                    if gradient is None
                                    else gradient.detach().reshape(-1)
                                    for parameter, gradient in zip(
                                        parameters, gradients, strict=True
                                    )
                                ]
                            )
                            vectors[key] += flat
                    del total, components
                combined = sum(vectors.values())
                base = vectors["factual_prediction"] + vectors["noop_prediction"]
                base_effect = base + vectors["paired_effect"]
                def norm(value):
                    return float(torch.linalg.vector_norm(value, dtype=torch.float64))
                total_norm = norm(combined)
                grouped = {}
                top = []
                for name, piece in zip(names, combined.split(sizes), strict=True):
                    squared = norm(piece) ** 2
                    group = group_name(name)
                    grouped[group] = grouped.get(group, 0.0) + squared
                    top.append((name, squared))
                weighted_norms = {key: norm(value) for key, value in vectors.items()}
                projection = {
                    key: float(torch.dot(value.double(), combined.double()))
                    / max(total_norm**2, 1e-30)
                    for key, value in vectors.items()
                }
                record = {
                    "batch_index": batch_index,
                    "horizon": horizon,
                    "losses": losses,
                    "weighted_component_norms": weighted_norms,
                    "total_grad_norm": total_norm,
                    "base_only_grad_norm": norm(base),
                    "base_plus_effect_grad_norm": norm(base_effect),
                    "clip_scale_at_1": min(1.0, 1.0 / (total_norm + 1e-6)),
                    "component_projection_share": projection,
                    "group_squared_norm_share": {
                        key: value / max(total_norm**2, 1e-30) for key, value in grouped.items()
                    },
                    "top_parameters": sorted(top, key=lambda item: -item[1])[:8],
                    "seconds": time.monotonic() - started,
                }
                report["results"].append(record)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(record), flush=True)
    fingerprint = hashlib.sha256()
    for parameter in parameters:
        fingerprint.update(parameter.detach().numpy().tobytes())
    assert fingerprint.hexdigest() == before, "diagnostics must never change model parameters"
    report["parameter_fingerprint_unchanged"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
