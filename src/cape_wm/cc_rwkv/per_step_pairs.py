from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SCHEMA_VERSION = "cc_rwkv_per_step_pairs_v1"
SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class PerStepPairWriter:
    """Fixed-shape HDF5 writer for factual/pulse-noop suffix pairs.

    The pulse arrays use a padded [intervention_position, future_offset] layout;
    ``pulse_noop_mask`` is the authoritative triangular mask.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        samples: int,
        history_steps: int,
        model_horizon: int,
        latent_dim: int,
        action_dim: int,
        state_dim: int,
        audit_samples: int,
        metadata: dict[str, Any],
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.horizon = int(model_horizon)
        self.audit_samples = int(audit_samples)
        self.handle = h5py.File(self.path, "w")
        meta = self.handle.create_group("metadata")
        meta.attrs["schema_version"] = SCHEMA_VERSION
        for key, value in metadata.items():
            meta.attrs[key] = _json(value) if isinstance(value, (dict, list, tuple)) else value

        group = self.handle.create_group("samples")
        string = h5py.string_dtype("utf-8")
        n, h = int(samples), self.horizon
        group.create_dataset("sample_id", (n,), dtype=string)
        group.create_dataset("episode_id", (n,), dtype="i8")
        group.create_dataset("start_step", (n,), dtype="i8")
        group.create_dataset("split", (n,), dtype="u1")
        group.create_dataset("source_snapshot_hash", (n,), dtype=string)
        group.create_dataset("source_snapshot_hashes", (n, h), dtype=string)
        group.create_dataset("external_noise_hashes", (n, h), dtype=string)
        group.create_dataset(
            "history_latents", (n, history_steps, latent_dim), dtype="f2", chunks=True
        )
        group.create_dataset(
            "history_actions_raw", (n, history_steps, action_dim), dtype="f4", chunks=True
        )
        group.create_dataset("history_mask", (n, history_steps), dtype="?", chunks=True)
        group.create_dataset("factual_actions", (n, h, action_dim), dtype="f4", chunks=True)
        group.create_dataset(
            "factual_latents", (n, h + 1, latent_dim), dtype="f2", chunks=True
        )
        group.create_dataset("factual_states", (n, h + 1, state_dim), dtype="f4", chunks=True)
        group.create_dataset(
            "pulse_noop_actions", (n, h, h, action_dim), dtype="f4", chunks=True
        )
        group.create_dataset(
            "pulse_noop_latents", (n, h, h, latent_dim), dtype="f2", chunks=True
        )
        group.create_dataset(
            "pulse_noop_states", (n, h, h, state_dim), dtype="f4", chunks=True
        )
        group.create_dataset("pulse_noop_mask", (n, h, h), dtype="?", chunks=True)
        group.create_dataset(
            "effect_latents", (n, h, h, latent_dim), dtype="f2", chunks=True
        )
        group.create_dataset("factual_replay_error", (n, h), dtype="f4", chunks=True)
        group.create_dataset("factual_replay_image_error", (n, h), dtype="f4", chunks=True)
        group.create_dataset("restore_consistent", (n, h), dtype="?", chunks=True)

        if self.audit_samples:
            audit = self.handle.create_group("audit")
            audit.create_dataset("sample_index", (self.audit_samples,), dtype="i8")
            audit.create_dataset(
                "factual_images",
                (self.audit_samples, h + 1, 224, 224, 3),
                dtype="u1",
                compression="gzip",
                compression_opts=1,
                chunks=(1, 1, 224, 224, 3),
            )
            audit.create_dataset(
                "pulse_noop_images",
                (self.audit_samples, h, h, 224, 224, 3),
                dtype="u1",
                compression="gzip",
                compression_opts=1,
                chunks=(1, 1, 1, 224, 224, 3),
            )
        self._audit_written = 0

    def write(self, index: int, item: dict[str, Any], *, audit: bool = False) -> None:
        group = self.handle["samples"]
        index = int(index)
        group["sample_id"][index] = item["sample_id"]
        group["episode_id"][index] = int(item["episode_id"])
        group["start_step"][index] = int(item["start_step"])
        group["split"][index] = SPLIT_CODES[item["split"]]
        group["source_snapshot_hash"][index] = item["source_snapshot_hash"]
        group["source_snapshot_hashes"][index] = item["source_snapshot_hashes"]
        group["external_noise_hashes"][index] = item["external_noise_hashes"]
        for name in (
            "history_latents",
            "history_actions_raw",
            "history_mask",
            "factual_actions",
            "factual_latents",
            "factual_states",
            "pulse_noop_actions",
            "pulse_noop_latents",
            "pulse_noop_states",
            "pulse_noop_mask",
            "effect_latents",
            "factual_replay_error",
            "factual_replay_image_error",
            "restore_consistent",
        ):
            group[name][index] = item[name]
        if audit:
            audit_group = self.handle["audit"]
            slot = self._audit_written
            audit_group["sample_index"][slot] = index
            audit_group["factual_images"][slot] = item["factual_images"]
            audit_group["pulse_noop_images"][slot] = item["pulse_noop_images"]
            self._audit_written += 1

    def close(self) -> None:
        self.handle.flush()
        self.handle.close()

    def __enter__(self) -> "PerStepPairWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
