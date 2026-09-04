from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from cape_wm.cc_rwkv.branch_dataset import FrozenImageEncoder, freeze_episode_split, sample_snapshot_specs, split_sha256
from cape_wm.cc_rwkv.branches import ActionDelayTwoRoomAdapter, TwoRoomBranchAdapter, snapshot_sha256
from cape_wm.cc_rwkv.lewm import FrozenLeWMEncoder
from cape_wm.cc_rwkv.per_step_pairs import SCHEMA_VERSION, PerStepPairWriter
from cape_wm.cc_rwkv.protocol import file_sha256, load_frozen_test_episode_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect per-position factual/pulse-noop counterfactual suffixes"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--frozen-test-episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("tworoom_w", "action_delay"), required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--history-steps", type=int, default=None)
    parser.add_argument("--future-primitive-steps", type=int, default=None)
    parser.add_argument("--action-block", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=3072)
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="snapshot sampling seed; defaults to split-seed",
    )
    parser.add_argument("--perturb-seed", type=int, default=1000)
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--raw-audit-fraction", type=float, default=0.01)
    parser.add_argument(
        "--allow-multiple-per-episode",
        action="store_true",
        help="sample multiple distinct start positions while keeping episode-level splits",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def _spec_hash(specs: list[Any]) -> str:
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


def _hash_json(value: Any) -> str:
    def normalise(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, dict):
            return {str(k): normalise(v) for k, v in sorted(item.items())}
        if isinstance(item, (list, tuple)):
            return [normalise(v) for v in item]
        return item

    return hashlib.sha256(
        json.dumps(normalise(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _step_block(env: Any, block: np.ndarray, action_block: int) -> None:
    primitive = np.asarray(block, dtype=np.float32).reshape(action_block, 2)
    for action in primitive:
        env.step(action)


def _make_item(
    *,
    env: Any,
    source: h5py.File,
    spec: Any,
    action_block: int,
    model_horizon: int,
    history_steps: int,
    encoder: FrozenImageEncoder,
    audit: bool,
) -> dict[str, Any]:
    start = int(spec.start_row)
    dense_actions = np.asarray(
        source["action"][start : start + model_horizon * action_block], dtype=np.float32
    )
    factual_actions = dense_actions.reshape(model_horizon, action_block * 2)
    history_rows = start + np.arange(
        -history_steps * action_block, 0, action_block, dtype=np.int64
    )
    history_images = np.asarray(source["pixels"][history_rows], dtype=np.uint8)
    history_actions = np.stack(
        [
            np.asarray(source["action"][row : row + action_block], dtype=np.float32).reshape(-1)
            for row in history_rows
        ]
    )

    reset_options: dict[str, Any] = {
        "state": np.asarray(source["pos_agent"][start], dtype=np.float32),
        "target_state": np.asarray(source["pos_target"][start], dtype=np.float32),
    }
    if isinstance(env, ActionDelayTwoRoomAdapter):
        reset_options["action_history"] = np.asarray(
            source["action"][start - env.delay : start], dtype=np.float32
        )
    env.reset(seed=int(spec.episode_id), options=reset_options)
    base_snapshot = env.snapshot()
    base_hash = snapshot_sha256(base_snapshot)

    factual_snapshots: list[dict[str, Any]] = []
    source_hashes: list[str] = []
    noise_hashes: list[str] = []
    factual_images = [env.render()]
    factual_states = [env.state_vector().copy()]
    for position in range(model_horizon):
        snapshot = env.snapshot()
        factual_snapshots.append(snapshot)
        source_hashes.append(snapshot_sha256(snapshot))
        noise_hashes.append(_hash_json(env.external_noise_state()))
        _step_block(env, factual_actions[position], action_block)
        factual_images.append(env.render())
        factual_states.append(env.state_vector().copy())
    factual_images_array = np.stack(factual_images).astype(np.uint8)
    factual_states_array = np.stack(factual_states).astype(np.float32)

    # Replay the factual suffix from the same snapshot. This is the deterministic
    # replay audit; perturbed tasks are not required to match the unperturbed
    # source HDF5 positions.
    env.restore(base_snapshot)
    replay_states = [env.state_vector().copy()]
    replay_images = [env.render()]
    for position in range(model_horizon):
        _step_block(env, factual_actions[position], action_block)
        replay_states.append(env.state_vector().copy())
        replay_images.append(env.render())
    replay_states_array = np.stack(replay_states).astype(np.float32)
    factual_replay_error = np.max(
        np.abs(replay_states_array[1:] - factual_states_array[1:]), axis=1
    ).astype(np.float32)
    factual_replay_image_error = np.asarray(
        [
            float(np.max(np.abs(a.astype(np.int16) - b.astype(np.int16))))
            for a, b in zip(replay_images[1:], factual_images[1:])
        ],
        dtype=np.float32,
    )
    if snapshot_sha256(env.snapshot()) != base_hash:
        # The replay ended at a future state, so restore explicitly before the
        # final base-snapshot invariant check.
        env.restore(base_snapshot)
    if snapshot_sha256(env.snapshot()) != base_hash:
        raise RuntimeError(f"base snapshot restore mismatch for {spec.sample_id}")

    pulse_images = np.zeros(
        (model_horizon, model_horizon, 224, 224, 3), dtype=np.uint8
    )
    pulse_states = np.zeros(
        (model_horizon, model_horizon, factual_states_array.shape[-1]), dtype=np.float32
    )
    pulse_actions = np.zeros(
        (model_horizon, model_horizon, action_block * 2), dtype=np.float32
    )
    restore_consistent = np.ones(model_horizon, dtype=bool)
    for position, snapshot in enumerate(factual_snapshots):
        env.restore(snapshot)
        initial_hash = snapshot_sha256(env.snapshot())
        initial_state = env.state_vector().copy()
        restore_consistent[position] &= initial_hash == source_hashes[position]
        restore_consistent[position] &= np.array_equal(
            initial_state, factual_states_array[position]
        )
        for offset in range(model_horizon - position):
            absolute = position + offset
            action = (
                np.zeros_like(factual_actions[absolute])
                if offset == 0
                else factual_actions[absolute]
            )
            pulse_actions[position, offset] = action
            _step_block(env, action, action_block)
            pulse_images[position, offset] = env.render()
            pulse_states[position, offset] = env.state_vector().copy()
        env.restore(snapshot)
        restore_consistent[position] &= snapshot_sha256(env.snapshot()) == source_hashes[position]

    image_list = [factual_images_array]
    for position in range(model_horizon):
        image_list.append(pulse_images[position, : model_horizon - position])
    encoded = encoder(np.concatenate(image_list, axis=0))
    cursor = 0
    factual_latents = encoded[cursor : cursor + model_horizon + 1]
    cursor += model_horizon + 1
    pulse_latents = np.zeros((model_horizon, model_horizon, factual_latents.shape[-1]), dtype=np.float32)
    for position in range(model_horizon):
        length = model_horizon - position
        pulse_latents[position, :length] = encoded[cursor : cursor + length]
        cursor += length
    pulse_mask = np.zeros((model_horizon, model_horizon), dtype=bool)
    for position in range(model_horizon):
        pulse_mask[position, : model_horizon - position] = True
    future_targets = np.zeros_like(pulse_latents)
    for position in range(model_horizon):
        length = model_horizon - position
        future_targets[position, :length] = factual_latents[position + 1 :]
    effect_latents = pulse_latents - future_targets
    effect_latents *= pulse_mask[..., None]
    env.restore(base_snapshot)
    if snapshot_sha256(env.snapshot()) != base_hash:
        raise RuntimeError(f"final snapshot restore mismatch for {spec.sample_id}")

    item: dict[str, Any] = {
        "sample_id": spec.sample_id,
        "episode_id": int(spec.episode_id),
        "start_step": int(spec.start_step),
        "split": spec.split,
        "source_snapshot_hash": base_hash,
        "source_snapshot_hashes": source_hashes,
        "external_noise_hashes": noise_hashes,
        "history_latents": encoder(history_images),
        "history_actions_raw": history_actions,
        "history_mask": np.ones(history_steps, dtype=bool),
        "factual_actions": factual_actions,
        "factual_latents": factual_latents,
        "factual_states": factual_states_array,
        "pulse_noop_actions": pulse_actions,
        "pulse_noop_latents": pulse_latents,
        "pulse_noop_states": pulse_states,
        "pulse_noop_mask": pulse_mask,
        "effect_latents": effect_latents,
        "factual_replay_error": factual_replay_error,
        "factual_replay_image_error": factual_replay_image_error,
        "restore_consistent": restore_consistent,
    }
    if audit:
        item["factual_images"] = factual_images_array
        item["pulse_noop_images"] = pulse_images
    return item


def main() -> None:
    args = parse_args()
    defaults = {
        "tworoom_w": (3, 20, 5),
        "action_delay": (5, 20, 1),
    }[args.variant]
    history_steps = args.history_steps or defaults[0]
    future_primitive_steps = args.future_primitive_steps or defaults[1]
    action_block = args.action_block or defaults[2]
    if future_primitive_steps % action_block:
        raise ValueError("future primitive horizon must be divisible by action block")
    if args.samples <= 0 or not 0 <= args.raw_audit_fraction <= 1:
        raise ValueError("samples must be positive and raw audit fraction must be in [0,1]")
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    model_horizon = future_primitive_steps // action_block

    import gymnasium as gym
    import stable_worldmodel as swm

    frozen_test = load_frozen_test_episode_ids(args.frozen_test_episodes)
    with h5py.File(args.data, "r") as source:
        episode_ids = np.unique(np.asarray(source["ep_idx"], dtype=np.int64))
        offsets = np.asarray(source["ep_offset"], dtype=np.int64)
        lengths = np.asarray(source["ep_len"], dtype=np.int64)
        split = freeze_episode_split(episode_ids, frozen_test, seed=args.split_seed)
        specs = sample_snapshot_specs(
            split,
            episode_ids,
            offsets,
            lengths,
            count=args.samples,
            history_steps=history_steps,
            future_steps=model_horizon,
            action_block=action_block,
            seed=args.sample_seed if args.sample_seed is not None else args.split_seed,
            allow_multiple_per_episode=args.allow_multiple_per_episode,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
        finite_actions = np.asarray(source["action"], dtype=np.float32)
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
        else:
            env = TwoRoomBranchAdapter(
                raw_env, drift_amplitude=0.5, drift_seed=args.perturb_seed
            )
            state_dim = 2

        local_samples = len(specs)
        audit_count = int(np.ceil(local_samples * args.raw_audit_fraction)) if local_samples else 0
        metadata = {
            "variant": args.variant,
            "source_dataset_sha256": file_sha256(args.data),
            "encoder_weights_sha256": file_sha256(args.weights),
            "model_config_sha256": file_sha256(args.model_config),
            "split_sha256": split_sha256(split),
            "spec_sha256": _spec_hash(specs),
            "action_block": action_block,
            "observation_stride": action_block,
            "primitive_horizon": future_primitive_steps,
            "model_horizon": model_horizon,
            "history_steps": history_steps,
            "reference_action_semantics": "raw-space zero action at each intervention position",
            "action_scaler_mean": action_mean.tolist(),
            "action_scaler_scale": action_scale.tolist(),
            "common_random_numbers": True,
            "triangular_mask": True,
            "factual_replay_definition": "deterministic replay from identical snapshot",
        }
        output_h5 = args.output / "pairs.h5"
        with PerStepPairWriter(
            output_h5,
            samples=local_samples,
            history_steps=history_steps,
            model_horizon=model_horizon,
            latent_dim=192,
            action_dim=action_block * 2,
            state_dim=state_dim,
            audit_samples=audit_count,
            metadata=metadata,
        ) as writer:
            for index, spec in enumerate(specs):
                # Make the drift draw independent of collection order. This is
                # important when a later formal run is resumed or repartitioned.
                if isinstance(env, TwoRoomBranchAdapter) and args.variant == "tworoom_w":
                    env._drift_rng = np.random.default_rng(
                        args.perturb_seed + int(spec.episode_id)
                    )
                item = _make_item(
                    env=env,
                    source=source,
                    spec=spec,
                    action_block=action_block,
                    model_horizon=model_horizon,
                    history_steps=history_steps,
                    encoder=encoder,
                    audit=index < audit_count,
                )
                if not np.asarray(item["restore_consistent"]).all():
                    raise RuntimeError(f"restore audit failed for {spec.sample_id}")
                if float(np.max(item["factual_replay_error"])) != 0.0:
                    raise RuntimeError(f"factual state replay audit failed for {spec.sample_id}")
                if float(np.max(item["factual_replay_image_error"])) != 0.0:
                    raise RuntimeError(f"factual image replay audit failed for {spec.sample_id}")
                writer.write(index, item, audit=index < audit_count)
                if (index + 1) % 10 == 0 or index + 1 == len(specs):
                    print(f"collected {index + 1}/{len(specs)}", flush=True)
        env.close()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "variant": args.variant,
        "samples": len(specs),
        "split_counts": {
            name: sum(spec.split == name for spec in specs)
            for name in ("train", "validation", "test")
        },
        "spec_sha256": _spec_hash(specs),
        "split_sha256": split_sha256(split),
        "pair_dataset_sha256": file_sha256(output_h5),
        "source_dataset_sha256": file_sha256(args.data),
        "encoder_weights_sha256": file_sha256(args.weights),
        "model_config_sha256": file_sha256(args.model_config),
        "action_block": action_block,
        "primitive_horizon": future_primitive_steps,
        "model_horizon": model_horizon,
        "history_steps": history_steps,
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
