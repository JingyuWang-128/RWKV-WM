from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
import torch

from .branch_dataset import SCHEMA_VERSION, SPLIT_CODES
from .training import BranchBatch


def m4_data_provenance(
    branch_path: str | Path,
    branch_manifest_path: str | Path,
    protocol_manifest_path: str | Path,
) -> tuple[dict[str, str | int], dict[str, str]]:
    branch_manifest = json.loads(Path(branch_manifest_path).read_text())
    protocol = json.loads(Path(protocol_manifest_path).read_text())
    normalizer_payload = (
        protocol["array_hashes"]["action_mean"]
        + ":"
        + protocol["array_hashes"]["action_scale"]
    ).encode()
    normalizer_sha = hashlib.sha256(normalizer_payload).hexdigest()
    with h5py.File(branch_path, "r") as handle:
        samples = handle["samples"]
        if handle["metadata"].attrs["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported branch data schema")
        latent_dim = int(samples["branch_latents"].shape[-1])
        action_dim = int(samples["branch_actions_raw"].shape[-1])
    data = {
        "schema_version": branch_manifest["schema_version"],
        "dataset_sha256": branch_manifest["branch_dataset_sha256"],
        "split_sha256": branch_manifest["split_sha256"],
        "normalizer_sha256": normalizer_sha,
        "latent_dim": latent_dim,
        "action_dim": action_dim,
        "action_block": int(branch_manifest["action_block"]),
    }
    encoder = {
        "encoder_sha256": branch_manifest["encoder_weights_sha256"],
        "model_config_sha256": branch_manifest["model_config_sha256"],
    }
    return data, encoder


class InMemoryBranchSplit:
    def __init__(
        self,
        path: str | Path,
        *,
        split: str,
        limit: int | None = None,
    ) -> None:
        if split not in SPLIT_CODES:
            raise ValueError(f"unknown split: {split}")
        with h5py.File(path, "r") as handle:
            samples = handle["samples"]
            indices = np.flatnonzero(np.asarray(samples["split"]) == SPLIT_CODES[split])
            if limit is not None:
                if limit <= 0:
                    raise ValueError("limit must be positive")
                indices = indices[:limit]
            if len(indices) == 0:
                raise ValueError(f"branch split is empty: {split}")
            self.sample_ids = [item.decode() for item in samples["sample_id"][indices]]
            self.tensors = {
                "history_latents": torch.from_numpy(
                    np.asarray(samples["history_latents"][indices], dtype=np.float32)
                ),
                "history_actions_raw": torch.from_numpy(
                    np.asarray(samples["history_actions_raw"][indices], dtype=np.float32)
                ),
                "history_mask": torch.from_numpy(
                    np.asarray(samples["history_mask"][indices], dtype=np.bool_)
                ),
                "branch_actions_raw": torch.from_numpy(
                    np.asarray(samples["branch_actions_raw"][indices], dtype=np.float32)
                ),
                "branch_latents": torch.from_numpy(
                    np.asarray(samples["branch_latents"][indices], dtype=np.float32)
                ),
                "branch_mask": torch.from_numpy(
                    np.asarray(samples["branch_mask"][indices], dtype=np.bool_)
                ),
            }

    def __len__(self) -> int:
        return len(self.sample_ids)

    def batch(
        self, indices: torch.Tensor, *, device: torch.device | str
    ) -> BranchBatch:
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("batch indices must be an int64 vector")
        payload = {
            name: tensor.index_select(0, indices)
            for name, tensor in self.tensors.items()
        }
        payload["sample_id"] = [self.sample_ids[index] for index in indices.tolist()]
        return BranchBatch.from_mapping(payload, device)

    def batches(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
    ) -> Iterator[BranchBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(self), batch_size):
            indices = torch.arange(start, min(start + batch_size, len(self)))
            yield self.batch(indices, device=device)

    def random_batch(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        device: torch.device | str,
    ) -> BranchBatch:
        indices = torch.randint(len(self), (batch_size,), generator=generator)
        return self.batch(indices, device=device)
