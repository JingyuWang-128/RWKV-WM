from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - registers filters before h5py reads compressed pixels.
import numpy as np
import torch

from cape_wm.cc_rwkv.branch_dataset import (
    BranchDatasetLayout,
    BranchHDF5Writer,
    FrozenImageEncoder,
    branch_dataset_sha256,
    freeze_episode_split,
    sample_snapshot_specs,
    split_sha256,
)
from cape_wm.cc_rwkv.branches import (
    ActionDelayTwoRoomAdapter,
    ActionSupportFilter,
    TwoRoomBranchAdapter,
    rollout_branches,
)
from cape_wm.cc_rwkv.lewm import FrozenLeWMEncoder
from cape_wm.cc_rwkv.protocol import file_sha256, load_frozen_test_episode_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect M2 true-intervention branches")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--frozen-test-episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("tworoom", "tworoom_w", "action_delay"), default="tworoom"
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--history-steps", type=int, default=3)
    parser.add_argument("--future-primitive-steps", type=int, default=20)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=3072)
    parser.add_argument("--perturb-seed", type=int, default=1000)
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    parser.add_argument("--flush-samples", type=int, default=16)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--raw-audit-fraction", type=float, default=0.01)
    parser.add_argument("--support-threshold", type=float)
    parser.add_argument("--candidate-oversample-fraction", type=float, default=0.1)
    return parser.parse_args()


def _spec_hash(specs) -> str:
    payload = json.dumps(
        [
            {
                "sample_id": item.sample_id,
                "episode_id": item.episode_id,
                "start_step": item.start_step,
                "start_row": item.start_row,
                "split": item.split,
            }
            for item in specs
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    if args.future_primitive_steps % args.action_block:
        raise ValueError("future primitive horizon must be divisible by action block")
    if not 0 <= args.raw_audit_fraction <= 1:
        raise ValueError("raw audit fraction must be in [0,1]")
    if not 0 <= args.candidate_oversample_fraction <= 1:
        raise ValueError("candidate oversample fraction must be in [0,1]")
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    import gymnasium as gym
    import stable_worldmodel as swm

    future_steps = args.future_primitive_steps // args.action_block
    frozen_test = load_frozen_test_episode_ids(args.frozen_test_episodes)
    with h5py.File(args.data, "r") as source:
        episode_ids = np.unique(np.asarray(source["ep_idx"], dtype=np.int64))
        offsets = np.asarray(source["ep_offset"], dtype=np.int64)
        lengths = np.asarray(source["ep_len"], dtype=np.int64)
        split = freeze_episode_split(episode_ids, frozen_test, seed=args.split_seed)
        candidate_count = max(
            args.samples + 20,
            int(np.ceil(args.samples * (1.0 + args.candidate_oversample_fraction))),
        )
        candidate_specs = sample_snapshot_specs(
            split,
            episode_ids,
            offsets,
            lengths,
            count=candidate_count,
            history_steps=args.history_steps,
            future_steps=future_steps,
            action_block=args.action_block,
            seed=args.split_seed,
        )
        train_rows = np.isin(np.asarray(source["ep_idx"]), np.asarray(split["train"]))
        behavior_actions = np.asarray(source["action"])[train_rows]
        support = ActionSupportFilter(behavior_actions, threshold=args.support_threshold)
        finite_actions = np.asarray(source["action"])
        finite_actions = finite_actions[np.isfinite(finite_actions).all(axis=1)]
        action_mean = finite_actions.mean(axis=0).astype(np.float32)
        action_scale = finite_actions.std(axis=0).astype(np.float32)

        official = swm.wm.utils.load_pretrained(
            str(args.weights), cache_dir=str(args.cache_dir)
        ).to(args.device)
        bridge = FrozenLeWMEncoder.from_official_model(
            official,
            action_mean=torch.from_numpy(action_mean),
            action_scale=torch.from_numpy(action_scale),
            source_weights=args.weights,
        )
        encoder = FrozenImageEncoder(
            bridge, device=torch.device(args.device), batch_size=args.encoder_batch_size
        )

        raw_env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array").unwrapped
        if args.variant == "action_delay":
            env = ActionDelayTwoRoomAdapter(raw_env, delay=5)
            state_dim = 13
        elif args.variant == "tworoom_w":
            env = TwoRoomBranchAdapter(raw_env, drift_amplitude=0.5, drift_seed=args.perturb_seed)
            state_dim = 2
        else:
            env = TwoRoomBranchAdapter(raw_env)
            state_dim = 2

        audit_count = int(np.ceil(args.samples * args.raw_audit_fraction))
        layout = BranchDatasetLayout(
            samples=args.samples,
            history_steps=args.history_steps,
            branches=4,
            future_steps=future_steps,
            latent_dim=192,
            action_block_dim=args.action_block * 2,
            state_dim=state_dim,
            audit_samples=audit_count,
        )
        metadata = {
            "environment": args.variant,
            "dataset_sha256": file_sha256(args.data),
            "encoder_weights_sha256": file_sha256(args.weights),
            "model_config_sha256": file_sha256(args.model_config),
            "split_sha256": split_sha256(split),
            "action_block": args.action_block,
            "observation_stride": args.action_block,
            "primitive_horizon": args.future_primitive_steps,
            "model_horizon": future_steps,
            "reference_action_semantics": "valid zero-vector no-op submission",
            "action_scaler_mean": action_mean.tolist(),
            "action_scaler_scale": action_scale.tolist(),
            "support_threshold": support.threshold,
        }
        output_h5 = args.output / "branches.h5"
        rng = np.random.default_rng(args.perturb_seed)
        factual_replay_max_error = 0.0
        pending = []
        selected_specs = []
        discarded_samples = []
        split_targets = {
            "train": int(np.floor(args.samples * 0.8)),
            "validation": int(np.floor(args.samples * 0.1)),
            "test": int(np.floor(args.samples * 0.1)),
        }
        split_targets["train"] += args.samples - sum(split_targets.values())
        split_successes = {name: 0 for name in split_targets}

        def flush(writer: BranchHDF5Writer) -> None:
            nonlocal pending
            if not pending:
                return
            flat_images = []
            for _, _, history_images, _, rollout in pending:
                flat_images.extend(history_images)
                flat_images.extend(rollout.images.reshape(-1, 224, 224, 3))
            all_latents = encoder(np.stack(flat_images))
            cursor = 0
            for index, spec, history_images, history_actions, rollout in pending:
                history_count = len(history_images)
                history_latents = all_latents[cursor : cursor + history_count]
                cursor += history_count
                branch_count = 4 * (future_steps + 1)
                branch_latents = all_latents[cursor : cursor + branch_count].reshape(
                    4, future_steps + 1, 192
                )
                cursor += branch_count
                writer.write(
                    index,
                    spec,
                    snapshot_hash=rollout.snapshot_hash,
                    history_latents=history_latents,
                    history_actions=history_actions,
                    rollout=rollout,
                    branch_latents=branch_latents,
                    audit=index < audit_count,
                )
            pending = []

        with BranchHDF5Writer(output_h5, layout, metadata=metadata) as writer:
            for spec in candidate_specs:
                if split_successes[spec.split] >= split_targets[spec.split]:
                    continue
                index = len(selected_specs)
                start = spec.start_row
                reset_options = {
                    "state": np.asarray(source["pos_agent"][start], dtype=np.float32),
                    "target_state": np.asarray(source["pos_target"][start], dtype=np.float32),
                }
                if args.variant == "action_delay":
                    reset_options["action_history"] = np.asarray(
                        source["action"][start - 5 : start], dtype=np.float32
                    )
                env.reset(seed=spec.episode_id, options=reset_options)
                if args.variant == "tworoom" and not np.array_equal(
                    env.render(), np.asarray(source["pixels"][start])
                ):
                    raise RuntimeError(f"offline render mismatch for {spec.sample_id}")
                dense_actions = np.asarray(
                    source["action"][start : start + args.future_primitive_steps],
                    dtype=np.float32,
                )
                factual = dense_actions.reshape(future_steps, args.action_block * 2)
                try:
                    rollout = rollout_branches(
                        env, factual, support, action_block=args.action_block, rng=rng
                    )
                except RuntimeError as error:
                    if "failed to sample a support-valid local action perturbation" not in str(
                        error
                    ):
                        raise
                    discarded_samples.append(
                        {
                            "sample_id": spec.sample_id,
                            "split": spec.split,
                            "reason": "local_perturbation_out_of_support",
                        }
                    )
                    continue
                if not rollout.restore_consistent:
                    raise RuntimeError(f"snapshot restore mismatch for {spec.sample_id}")
                if args.variant == "tworoom":
                    target_rows = start + np.arange(
                        0, args.future_primitive_steps + 1, args.action_block
                    )
                    target_states = np.asarray(source["pos_agent"][target_rows])
                    error = float(np.max(np.abs(rollout.states[1] - target_states)))
                    factual_replay_max_error = max(factual_replay_max_error, error)
                    if error != 0:
                        raise RuntimeError(f"factual replay mismatch {error} for {spec.sample_id}")
                history_rows = start + np.arange(
                    -args.history_steps * args.action_block, 0, args.action_block
                )
                history_images = np.asarray(source["pixels"][history_rows])
                history_actions = np.stack(
                    [
                        np.asarray(
                            source["action"][row : row + args.action_block],
                            dtype=np.float32,
                        ).reshape(-1)
                        for row in history_rows
                    ]
                )
                pending.append((index, spec, history_images, history_actions, rollout))
                selected_specs.append(spec)
                split_successes[spec.split] += 1
                if len(pending) >= args.flush_samples:
                    flush(writer)
                if (index + 1) % 100 == 0 or index + 1 == args.samples:
                    print(f"collected {index + 1}/{args.samples}", flush=True)
            flush(writer)
            if len(selected_specs) != args.samples or split_successes != split_targets:
                raise RuntimeError(
                    "candidate oversampling did not yield the requested split counts: "
                    f"observed={split_successes}, expected={split_targets}"
                )
        env.close()

    specs = selected_specs

    manifest = {
        "schema_version": "cc_rwkv_m2_manifest_v1",
        "variant": args.variant,
        "samples": args.samples,
        "split_counts": {
            name: sum(spec.split == name for spec in specs)
            for name in ("train", "validation", "test")
        },
        "spec_sha256": _spec_hash(specs),
        "split_sha256": split_sha256(split),
        "branch_dataset_sha256": branch_dataset_sha256(output_h5),
        "source_dataset_sha256": file_sha256(args.data),
        "encoder_weights_sha256": file_sha256(args.weights),
        "model_config_sha256": file_sha256(args.model_config),
        "support_threshold": support.threshold,
        "discarded_sample_count": len(discarded_samples),
        "discarded_samples": discarded_samples,
        "restore_failure_count": 0,
        "factual_replay_checked": args.variant == "tworoom",
        "factual_replay_max_error": (
            factual_replay_max_error if args.variant == "tworoom" else None
        ),
        "action_block": args.action_block,
        "primitive_horizon": args.future_primitive_steps,
        "model_horizon": future_steps,
        "history_steps": args.history_steps,
        "raw_audit_samples": audit_count,
        "bytes": output_h5.stat().st_size,
        "command_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
