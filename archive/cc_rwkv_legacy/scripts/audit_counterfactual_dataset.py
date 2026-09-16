from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from cape_wm.cc_rwkv.branch_dataset import CounterfactualBranchDataset, FrozenImageEncoder
from cape_wm.cc_rwkv.lewm import FrozenLeWMEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit an M2 counterfactual HDF5 cache")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--latent-atol", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import stable_worldmodel as swm

    dataset = CounterfactualBranchDataset(args.dataset)
    dataset.close()
    with h5py.File(args.dataset, "r") as handle:
        samples = handle["samples"]
        actions = np.asarray(samples["branch_actions_raw"])
        branch_latents = np.asarray(samples["branch_latents"])
        split = np.asarray(samples["split"])
        episodes = np.asarray(samples["episode_id"])
        support = np.asarray(samples["action_support_score"])
        threshold = float(handle["metadata"].attrs["support_threshold"])
        initial_branch_max_abs_error = float(
            np.max(
                np.abs(
                    branch_latents[:, :, 0].astype(np.float32)
                    - branch_latents[:, :1, 0].astype(np.float32)
                )
            )
        )
        checks = {
            "finite_latents": bool(np.isfinite(branch_latents).all()),
            "finite_actions": bool(np.isfinite(actions).all()),
            "restore_consistent": bool(np.asarray(samples["restore_consistent"]).all()),
            "reference_is_zero": bool(np.all(actions[:, 0] == 0)),
            "pulse_noop_first_is_zero": bool(np.all(actions[:, 2, 0] == 0)),
            "pulse_noop_suffix_matches": bool(np.array_equal(actions[:, 2, 1:], actions[:, 1, 1:])),
            "pulse_local_suffix_matches": bool(
                np.array_equal(actions[:, 3, 1:], actions[:, 1, 1:])
            ),
            "pulse_local_supported": bool(np.all(support[:, 3, 0] >= threshold)),
            "initial_branch_latents_close": initial_branch_max_abs_error <= args.latent_atol,
        }
        groups = [set(episodes[split == code].tolist()) for code in (0, 1, 2)]
        checks["split_disjoint"] = not any(
            groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)
        )

        audit_indices = np.asarray(handle["audit/sample_index"], dtype=np.int64)
        raw_images = np.asarray(handle["audit/raw_images"])
        mean = torch.tensor(json.loads(handle["metadata"].attrs["action_scaler_mean"]))
        scale = torch.tensor(json.loads(handle["metadata"].attrs["action_scaler_scale"]))
        official = swm.wm.utils.load_pretrained(
            str(args.weights), cache_dir=str(args.cache_dir)
        ).to(args.device)
        bridge = FrozenLeWMEncoder.from_official_model(
            official, action_mean=mean, action_scale=scale, source_weights=args.weights
        )
        encoder = FrozenImageEncoder(bridge, device=torch.device(args.device), batch_size=128)
        online = encoder(raw_images.reshape(-1, 224, 224, 3)).reshape(
            *raw_images.shape[:3], 192
        ).astype(np.float16)
        cached = branch_latents[audit_indices]
        cache_max_abs_error = float(
            np.max(np.abs(online.astype(np.float32) - cached.astype(np.float32)))
        )
        checks["raw_audit_latents_close"] = cache_max_abs_error <= args.latent_atol

    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"counterfactual dataset audit failed: {failed}")
    summary = {
        "schema_version": "cc_rwkv_m2_audit_v1",
        "samples": len(actions),
        "checks": checks,
        "latent_atol": args.latent_atol,
        "initial_branch_max_abs_error": initial_branch_max_abs_error,
        "raw_audit_cache_max_abs_error": cache_max_abs_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
