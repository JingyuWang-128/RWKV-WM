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
                for key in ("schema_version", "variant", "split_sha256", "model_horizon", "history_steps", "action_block"):
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
        with h5py.File(args.output, "w") as out:
            metadata = out.create_group("metadata")
            for key, value in base_attrs.items():
                metadata.attrs[key] = value
            metadata.attrs["merged_shards"] = json.dumps([str(p) for p in args.shards])
            metadata.attrs["deduplicated_samples"] = n
            target = out.create_group("samples")
            for name in dataset_names:
                source = sample_group[name]
                target.create_dataset(name, shape=(n,) + source.shape[1:], dtype=source.dtype, chunks=True)
            output_positions = 0
            for shard in args.shards:
                indices = np.asarray(selected_by_shard.get(shard, []), dtype=np.int64)
                if not len(indices):
                    continue
                with h5py.File(shard, "r") as handle:
                    for name in dataset_names:
                        target[name][output_positions : output_positions + len(indices)] = handle["samples"][name][indices]
                output_positions += len(indices)
            # Preserve all audited raw examples, if present in any shard.
            audits: list[tuple[Path, str, int]] = []
            for shard in args.shards:
                with h5py.File(shard, "r") as handle:
                    if "audit" not in handle:
                        continue
                    for name in audit_names:
                        if name in handle["audit"]:
                            audits.append((shard, name, handle["audit"][name].shape[0]))
            if audits:
                audit_group = out.create_group("audit")
                for name in audit_names:
                    parts = [(path, size) for path, candidate, size in audits if candidate == name]
                    if not parts:
                        continue
                    source = first["audit"][name]
                    total = sum(size for _, size in parts)
                    audit_group.create_dataset(name, shape=(total,) + source.shape[1:], dtype=source.dtype, chunks=True)
                    cursor = 0
                    for path, size in parts:
                        with h5py.File(path, "r") as handle:
                            audit_group[name][cursor : cursor + size] = handle["audit"][name][:]
                        cursor += size
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "variant": base_attrs.get("variant"),
        "samples": len(rows),
        "split_counts": {},
        "split_sha256": base_attrs.get("split_sha256"),
        "pair_dataset_sha256": file_sha256(args.output),
        "bytes": args.output.stat().st_size,
        "shards": [str(p) for p in args.shards],
        "deduplicated": True,
    }
    with h5py.File(args.output, "r") as handle:
        splits = handle["samples"]["split"][:]
        manifest["split_counts"] = {name: int(np.sum(splits == code)) for name, code in (("train", 0), ("validation", 1), ("test", 2))}
    args.output.with_name("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
