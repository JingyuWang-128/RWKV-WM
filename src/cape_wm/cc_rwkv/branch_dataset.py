from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from .protocol import file_sha256

SCHEMA_VERSION = "cc_rwkv_branches_v1"
SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}


@dataclass(frozen=True, slots=True)
class SnapshotSpec:
    sample_id: str
    episode_id: int
    start_step: int
    start_row: int
    split: str


@dataclass(frozen=True, slots=True)
class BranchDatasetLayout:
    samples: int
    history_steps: int
    branches: int
    future_steps: int
    latent_dim: int
    action_block_dim: int
    state_dim: int
    audit_samples: int


def freeze_episode_split(
    episode_ids: np.ndarray,
    frozen_test: tuple[int, ...],
    *,
    seed: int,
) -> dict[str, tuple[int, ...]]:
    identifiers = np.asarray(sorted(map(int, episode_ids)), dtype=np.int64)
    test = set(map(int, frozen_test))
    if not test <= set(identifiers.tolist()):
        raise ValueError("frozen test split contains unknown episodes")
    available = identifiers[~np.isin(identifiers, np.asarray(sorted(test)))]
    rng = np.random.default_rng(seed)
    available = available[rng.permutation(len(available))]
    desired_train = min(int(round(0.8 * len(identifiers))), len(available) - 1)
    split = {
        "train": tuple(sorted(map(int, available[:desired_train]))),
        "validation": tuple(sorted(map(int, available[desired_train:]))),
        "test": tuple(sorted(test)),
    }
    groups = [set(split[name]) for name in ("train", "validation", "test")]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("episode split leakage detected")
    return split


def split_sha256(split: dict[str, tuple[int, ...]]) -> str:
    payload = json.dumps(split, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def sample_snapshot_specs(
    split: dict[str, tuple[int, ...]],
    all_episode_ids: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    *,
    count: int,
    history_steps: int,
    future_steps: int,
    action_block: int,
    seed: int,
    allow_multiple_per_episode: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
) -> list[SnapshotSpec]:
    if count <= 0 or history_steps <= 0 or future_steps <= 0 or action_block <= 0:
        raise ValueError("sampling counts and horizons must be positive")
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    lookup = {int(episode): index for index, episode in enumerate(all_episode_ids)}
    proportions = {"train": 0.8, "validation": 0.1, "test": 0.1}
    counts = {name: int(np.floor(count * value)) for name, value in proportions.items()}
    counts["train"] += count - sum(counts.values())
    rng = np.random.default_rng(seed)
    specs: list[SnapshotSpec] = []
    minimum_start = history_steps * action_block
    future_primitive = future_steps * action_block
    for split_name in ("train", "validation", "test"):
        eligible = [
            episode
            for episode in split[split_name]
            if int(lengths[lookup[episode]]) >= minimum_start + future_primitive + 1
        ]
        requested = counts[split_name]
        if allow_multiple_per_episode:
            candidates: list[tuple[int, int]] = []
            for episode in eligible:
                table_index = lookup[int(episode)]
                maximum_start = int(lengths[table_index]) - future_primitive - 1
                candidates.extend(
                    (int(episode), int(start))
                    for start in range(minimum_start, maximum_start + 1, action_block)
                )
            if requested > len(candidates):
                raise ValueError(
                    f"not enough eligible episode/start pairs for {split_name}: "
                    f"requested {requested}, available {len(candidates)}"
                )
            selected_indices = rng.choice(len(candidates), size=requested, replace=False)
            shard_indices = np.array_split(np.asarray(selected_indices), num_shards)[shard_index]
            selected_pairs = [candidates[int(index)] for index in shard_indices]
        else:
            if requested > len(eligible):
                raise ValueError(f"not enough eligible {split_name} episodes for unique sampling")
            selected = rng.choice(np.asarray(eligible), size=requested, replace=False)
            selected = np.array_split(np.asarray(selected), num_shards)[shard_index]
            selected_pairs = []
            for episode in selected:
                table_index = lookup[int(episode)]
                maximum_start = int(lengths[table_index]) - future_primitive - 1
                candidates = np.arange(minimum_start, maximum_start + 1, action_block)
                selected_pairs.append((int(episode), int(rng.choice(candidates))))
        for episode, start in selected_pairs:
            specs.append(
                SnapshotSpec(
                    sample_id=(
                        f"tworoom:{split_name}:ep{int(episode)}:"
                        f"start{start}:h{future_primitive}"
                    ),
                    episode_id=int(episode),
                    start_step=start,
                    start_row=int(offsets[table_index]) + start,
                    split=split_name,
                )
            )
    specs.sort(key=lambda item: (SPLIT_CODES[item.split], item.episode_id))
    return specs


class FrozenImageEncoder:
    """Batch image preprocessing and encoding matching the released LeWM evaluator."""

    def __init__(self, bridge: torch.nn.Module, *, device: torch.device, batch_size: int):
        self.bridge = bridge.to(device).eval()
        self.device = device
        self.batch_size = int(batch_size)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, images: np.ndarray) -> np.ndarray:
        source = np.asarray(images, dtype=np.uint8)
        outputs = []
        for start in range(0, len(source), self.batch_size):
            tensor = torch.from_numpy(source[start : start + self.batch_size].copy())
            tensor = tensor.to(self.device).permute(0, 3, 1, 2).float() / 255.0
            tensor = (tensor - self.mean) / self.std
            outputs.append(self.bridge.encode_images(tensor)[:, 0].cpu())
        return torch.cat(outputs).numpy().astype(np.float32)


class BranchHDF5Writer:
    def __init__(
        self,
        path: str | Path,
        layout: BranchDatasetLayout,
        *,
        metadata: dict[str, Any],
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.layout = layout
        self.handle = h5py.File(self.path, "w")
        meta = self.handle.create_group("metadata")
        meta.attrs["schema_version"] = SCHEMA_VERSION
        for key, value in metadata.items():
            meta.attrs[key] = json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
        samples = self.handle.create_group("samples")
        string = h5py.string_dtype("utf-8")
        n, th, k, tf = layout.samples, layout.history_steps, layout.branches, layout.future_steps
        samples.create_dataset("sample_id", (n,), dtype=string)
        samples.create_dataset("episode_id", (n,), dtype="i8")
        samples.create_dataset("start_step", (n,), dtype="i8")
        samples.create_dataset("split", (n,), dtype="u1")
        samples.create_dataset("snapshot_hash", (n,), dtype=string)
        samples.create_dataset("common_noise_id", (n,), dtype=string)
        samples.create_dataset(
            "history_latents", (n, th, layout.latent_dim), dtype="f2", chunks=True
        )
        samples.create_dataset(
            "history_actions_raw", (n, th, layout.action_block_dim), dtype="f4", chunks=True
        )
        samples.create_dataset("history_mask", (n, th), dtype="?", chunks=True)
        samples.create_dataset("branch_type", (n, k), dtype="u1", chunks=True)
        samples.create_dataset(
            "branch_actions_raw", (n, k, tf, layout.action_block_dim), dtype="f4", chunks=True
        )
        samples.create_dataset(
            "branch_latents", (n, k, tf + 1, layout.latent_dim), dtype="f2", chunks=True
        )
        samples.create_dataset(
            "branch_states", (n, k, tf + 1, layout.state_dim), dtype="f4", chunks=True
        )
        samples.create_dataset("branch_mask", (n, k, tf + 1), dtype="?", chunks=True)
        samples.create_dataset("action_support_score", (n, k, tf), dtype="f4", chunks=True)
        samples.create_dataset("restore_consistent", (n,), dtype="?", chunks=True)
        if layout.audit_samples:
            audit = self.handle.create_group("audit")
            audit.create_dataset("sample_index", (layout.audit_samples,), dtype="i8")
            audit.create_dataset(
                "raw_images",
                (layout.audit_samples, k, tf + 1, 224, 224, 3),
                dtype="u1",
                compression="gzip",
                compression_opts=1,
                chunks=(1, 1, 1, 224, 224, 3),
            )
        self.audit_written = 0

    def write(
        self,
        index: int,
        spec: SnapshotSpec,
        *,
        snapshot_hash: str,
        history_latents: np.ndarray,
        history_actions: np.ndarray,
        rollout: Any,
        branch_latents: np.ndarray,
        audit: bool,
    ) -> None:
        samples = self.handle["samples"]
        samples["sample_id"][index] = spec.sample_id
        samples["episode_id"][index] = spec.episode_id
        samples["start_step"][index] = spec.start_step
        samples["split"][index] = SPLIT_CODES[spec.split]
        samples["snapshot_hash"][index] = snapshot_hash
        samples["common_noise_id"][index] = f"deterministic:{snapshot_hash}"
        samples["history_latents"][index] = history_latents.astype(np.float16)
        samples["history_actions_raw"][index] = history_actions
        samples["history_mask"][index] = True
        samples["branch_type"][index] = rollout.branch_type
        samples["branch_actions_raw"][index] = rollout.actions
        samples["branch_latents"][index] = branch_latents.astype(np.float16)
        samples["branch_states"][index] = rollout.states
        samples["branch_mask"][index] = True
        samples["action_support_score"][index] = rollout.support_score
        samples["restore_consistent"][index] = rollout.restore_consistent
        if audit:
            audit_group = self.handle["audit"]
            audit_group["sample_index"][self.audit_written] = index
            audit_group["raw_images"][self.audit_written] = rollout.images
            self.audit_written += 1

    def close(self) -> None:
        self.handle.flush()
        self.handle.close()

    def __enter__(self) -> BranchHDF5Writer:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class CounterfactualBranchDataset(torch.utils.data.Dataset):
    def __init__(self, path: str | Path, *, expected_metadata: dict[str, Any] | None = None):
        self.path = Path(path)
        self.handle = h5py.File(self.path, "r")
        metadata = self.handle["metadata"].attrs
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported counterfactual branch schema")
        for key, expected in (expected_metadata or {}).items():
            if metadata.get(key) != expected:
                raise ValueError(f"branch dataset metadata mismatch: {key}")
        sample = self.handle["samples"]
        split = np.asarray(sample["split"])
        episodes = np.asarray(sample["episode_id"])
        groups = [set(episodes[split == code].tolist()) for code in SPLIT_CODES.values()]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("episode leakage across branch dataset splits")
        if not np.asarray(sample["restore_consistent"]).all():
            raise ValueError("dataset contains snapshot restoration failures")

    def __len__(self) -> int:
        return len(self.handle["samples/sample_id"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        samples = self.handle["samples"]
        tensor_keys = (
            "history_latents",
            "history_actions_raw",
            "history_mask",
            "branch_type",
            "branch_actions_raw",
            "branch_latents",
            "branch_states",
            "branch_mask",
            "action_support_score",
        )
        return {
            "sample_id": samples["sample_id"][index].decode(),
            "episode_id": int(samples["episode_id"][index]),
            "start_step": int(samples["start_step"][index]),
            "split": int(samples["split"][index]),
            **{key: torch.from_numpy(np.asarray(samples[key][index])) for key in tensor_keys},
        }

    def close(self) -> None:
        self.handle.close()


def branch_dataset_sha256(path: str | Path) -> str:
    return file_sha256(path)
