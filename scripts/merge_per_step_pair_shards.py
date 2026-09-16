from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from cape_wm.cc_rwkv.per_step_pairs import SCHEMA_VERSION
from cape_wm.cc_rwkv.protocol import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge and deduplicate per-step pair shards")
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _attrs(handle: h5py.File) -> dict[str, object]:
    return {str(k): v for k, v in handle["metadata"].attrs.items()}


def main() -> None:
    args = parse_args()
    if len(args.shards) < 2:
        raise ValueError("at least two shards are required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[Path, int]] = []
    selected_by_shard: dict[Path, list[int]] = {}
    seen: set[str] = set()
    base_attrs: dict[str, object] | None = None
    dataset_names: list[str] | None = None
    audit_names: list[str] = []
    for shard in args.shards:
        with h5py.File(shard, "r") as handle:
            attrs = _attrs(handle)
            if attrs.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"schema mismatch in {shard}")
            if base_attrs is None:
                base_attrs = attrs
                dataset_names = sorted(handle["samples"].keys())
                audit_names = sorted(handle.get("audit", {}).keys()) if "audit" in handle else []
            else:
                for key in (
                    "schema_version",
                    "variant",
                    "source_dataset_sha256",
                    "encoder_weights_sha256",
                    "model_config_sha256",
                    "split_sha256",
                    "model_horizon",
                    "primitive_horizon",
                    "history_steps",
                    "action_block",
                    "observation_stride",
                    "action_scaler_mean",
                    "action_scaler_scale",
                    "reference_action_semantics",
                    "common_random_numbers",
                    "triangular_mask",
                ):
                    if attrs.get(key) != base_attrs.get(key):
                        raise ValueError(f"metadata mismatch for {key} in {shard}")
                if sorted(handle["samples"].keys()) != dataset_names:
                    raise ValueError(f"dataset layout mismatch in {shard}")
            sample_ids = handle["samples"]["sample_id"][:]
            for index, value in enumerate(sample_ids):
                key = value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
                if key not in seen:
                    seen.add(key)
                    rows.append((shard, index))
                    selected_by_shard.setdefault(shard, []).append(index)

    if not rows or dataset_names is None or base_attrs is None:
        raise ValueError("no rows to merge")
    with h5py.File(args.shards[0], "r") as first:
        sample_group = first["samples"]
        n = len(rows)
        output_index_by_source: dict[tuple[Path, int], int] = {}
        with h5py.File(args.output, "w") as out:
            metadata = out.create_group("metadata")
            for key, value in base_attrs.items():
                metadata.attrs[key] = value
            metadata.attrs["merged_shards"] = json.dumps([str(p) for p in args.shards])
            metadata.attrs["deduplicated_samples"] = n
            target = out.create_group("samples")
            for name in dataset_names:
                source = sample_group[name]
                target.create_dataset(
                    name, shape=(n,) + source.shape[1:], dtype=source.dtype, chunks=True
                )
            output_positions = 0
            for shard in args.shards:
                indices = np.asarray(selected_by_shard.get(shard, []), dtype=np.int64)
                if not len(indices):
                    continue
                for local_position, source_index in enumerate(indices):
                    output_index_by_source[(shard, int(source_index))] = (
                        output_positions + local_position
                    )
                with h5py.File(shard, "r") as handle:
                    for name in dataset_names:
                        target[name][output_positions : output_positions + len(indices)] = handle[
                            "samples"
                        ][name][indices]
                output_positions += len(indices)
            # Preserve audited raw examples and remap shard-local sample_index
            # values to the corresponding row in the merged sample table.
            audit_records: list[tuple[Path, int, int]] = []
            for shard in args.shards:
                with h5py.File(shard, "r") as handle:
                    if "audit" not in handle:
                        continue
                    if sorted(handle["audit"].keys()) != audit_names:
                        raise ValueError(f"audit layout mismatch in {shard}")
                    for audit_slot, source_index in enumerate(
                        np.asarray(handle["audit/sample_index"], dtype=np.int64)
                    ):
                        merged_index = output_index_by_source.get((shard, int(source_index)))
                        if merged_index is not None:
                            audit_records.append((shard, audit_slot, merged_index))
            if audit_records:
                audit_group = out.create_group("audit")
                audit_group.create_dataset("sample_index", (len(audit_records),), dtype="i8")
                image_names = [name for name in audit_names if name != "sample_index"]
                for name in image_names:
                    source = first["audit"][name]
                    audit_group.create_dataset(
                        name,
                        shape=(len(audit_records),) + source.shape[1:],
                        dtype=source.dtype,
                        chunks=True,
                    )
                for output_slot, (shard, audit_slot, merged_index) in enumerate(audit_records):
                    audit_group["sample_index"][output_slot] = merged_index
                    with h5py.File(shard, "r") as handle:
                        for name in image_names:
                            audit_group[name][output_slot] = handle["audit"][name][audit_slot]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "variant": base_attrs.get("variant"),
        "samples": len(rows),
        "split_counts": {},
        "split_sha256": base_attrs.get("split_sha256"),
        "source_dataset_sha256": base_attrs.get("source_dataset_sha256"),
        "encoder_weights_sha256": base_attrs.get("encoder_weights_sha256"),
        "model_config_sha256": base_attrs.get("model_config_sha256"),
        "pair_dataset_sha256": file_sha256(args.output),
        "bytes": args.output.stat().st_size,
        "shards": [str(p) for p in args.shards],
        "deduplicated": True,
    }
    with h5py.File(args.output, "r") as handle:
        splits = handle["samples"]["split"][:]
        manifest["split_counts"] = {
            name: int(np.sum(splits == code))
            for name, code in (("train", 0), ("validation", 1), ("test", 2))
        }
    args.output.with_name("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
