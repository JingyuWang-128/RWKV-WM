from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from cape_wm.cc_rwkv.per_step_pairs import PerStepPairWriter
from scripts import (
    audit_per_step_paired_counterfactual,
    merge_per_step_pair_shards,
)


def _write_source(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ep_idx", data=np.repeat([10, 20], 10))
        handle.create_dataset("ep_offset", data=np.asarray([0, 10], dtype=np.int64))
        handle.create_dataset("ep_len", data=np.asarray([10, 10], dtype=np.int64))
        handle.create_dataset("step_idx", data=np.tile(np.arange(10), 2))


def _item(episode: int, start: int, source_row: int, split: str) -> dict[str, object]:
    horizon, latent_dim, action_dim, state_dim = 2, 3, 2, 2
    factual_latents = np.arange((horizon + 1) * latent_dim, dtype=np.float32).reshape(
        horizon + 1, latent_dim
    )
    pulse_latents = np.zeros((horizon, horizon, latent_dim), dtype=np.float32)
    mask = np.zeros((horizon, horizon), dtype=bool)
    for position in range(horizon):
        length = horizon - position
        mask[position, :length] = True
        pulse_latents[position, :length] = factual_latents[position + 1 :] + 1
    effect_latents = np.zeros_like(pulse_latents)
    for position in range(horizon):
        length = horizon - position
        effect_latents[position, :length] = (
            pulse_latents[position, :length] - factual_latents[position + 1 :]
        )
    factual_images = np.zeros((horizon + 1, 224, 224, 3), dtype=np.uint8)
    return {
        "sample_id": f"test:train:ep{episode}:start{start}:h2",
        "episode_id": episode,
        "start_step": start,
        "source_row": source_row,
        "split": split,
        "source_snapshot_hash": f"base-{episode}",
        "source_snapshot_hashes": [f"s-{episode}-0", f"s-{episode}-1"],
        "restored_snapshot_hashes": [f"s-{episode}-0", f"s-{episode}-1"],
        "external_noise_hashes": [f"n-{episode}-0", f"n-{episode}-1"],
        "history_latents": np.zeros((1, latent_dim), dtype=np.float32),
        "history_actions_raw": np.zeros((1, action_dim), dtype=np.float32),
        "history_mask": np.ones(1, dtype=bool),
        "factual_actions": np.ones((horizon, action_dim), dtype=np.float32),
        "factual_latents": factual_latents,
        "factual_states": np.zeros((horizon + 1, state_dim), dtype=np.float32),
        "pulse_noop_actions": np.asarray(
            [[[[0, 0], [1, 1]][offset] for offset in range(horizon)] for _ in range(horizon)],
            dtype=np.float32,
        ),
        "pulse_noop_latents": pulse_latents,
        "pulse_noop_states": np.zeros((horizon, horizon, state_dim), dtype=np.float32),
        "pulse_noop_mask": mask,
        "effect_latents": effect_latents,
        "factual_replay_error": np.zeros(horizon, dtype=np.float32),
        "factual_replay_image_error": np.zeros(horizon, dtype=np.float32),
        "restore_consistent": np.ones(horizon, dtype=bool),
        "factual_images": factual_images,
        "pulse_noop_images": np.zeros((horizon, horizon, 224, 224, 3), dtype=np.uint8),
    }


def _write_shard(path: Path, item: dict[str, object]) -> None:
    metadata = {
        "variant": "test",
        "source_dataset_sha256": "source-hash",
        "encoder_weights_sha256": "weights-hash",
        "model_config_sha256": "config-hash",
        "split_sha256": "split-hash",
        "model_horizon": 2,
        "primitive_horizon": 2,
        "history_steps": 1,
        "action_block": 1,
        "observation_stride": 1,
        "action_scaler_mean": [0.0, 0.0],
        "action_scaler_scale": [1.0, 1.0],
        "reference_action_semantics": "raw-space zero action at each intervention position",
        "common_random_numbers": True,
        "triangular_mask": True,
    }
    with PerStepPairWriter(
        path,
        samples=1,
        history_steps=1,
        model_horizon=2,
        latent_dim=3,
        action_dim=2,
        state_dim=2,
        audit_samples=1,
        metadata=metadata,
    ) as writer:
        writer.write(0, item, audit=True)


def test_merge_remaps_audit_indices_and_audit_verifies_source_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.h5"
    _write_source(source)
    shard0 = tmp_path / "shard0" / "pairs.h5"
    shard1 = tmp_path / "shard1" / "pairs.h5"
    # Deliberately merge source rows out of order. h5py rejects direct fancy
    # indexing with [14, 3], while the audit must preserve merged row order.
    _write_shard(shard0, _item(20, 4, 14, "validation"))
    _write_shard(shard1, _item(10, 3, 3, "train"))
    merged = tmp_path / "merged" / "pairs.h5"
    monkeypatch.setattr(
        sys,
        "argv",
        ["merge", "--shards", str(shard0), str(shard1), "--output", str(merged)],
    )
    merge_per_step_pair_shards.main()
    with h5py.File(merged, "r") as handle:
        assert np.array_equal(handle["samples/source_row"][:], [14, 3])
        assert np.array_equal(handle["audit/sample_index"][:], [0, 1])

    report = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--dataset",
            str(merged),
            "--source",
            str(source),
            "--output",
            str(report),
        ],
    )
    audit_per_step_paired_counterfactual.main()


def test_audit_rejects_incorrect_episode_start_source_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.h5"
    _write_source(source)
    pairs = tmp_path / "pairs.h5"
    _write_shard(pairs, _item(10, 3, 4, "train"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--dataset",
            str(pairs),
            "--source",
            str(source),
            "--output",
            str(tmp_path / "audit.json"),
        ],
    )
    with pytest.raises(RuntimeError, match="source_row_matches_episode_start"):
        audit_per_step_paired_counterfactual.main()
