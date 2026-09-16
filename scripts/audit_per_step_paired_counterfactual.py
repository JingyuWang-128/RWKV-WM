from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from cape_wm.cc_rwkv.per_step_pairs import SCHEMA_VERSION, SPLIT_CODES


def _take_rows(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    """Read arbitrary HDF5 rows while preserving order and duplicates.

    h5py requires fancy indices to be strictly increasing. Merged shards retain
    their deterministic shard order, so source rows are intentionally not
    guaranteed to be monotonic.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return np.empty((0,) + dataset.shape[1:], dtype=dataset.dtype)
    unique_indices, inverse = np.unique(indices, return_inverse=True)
    return np.asarray(dataset[unique_indices])[inverse]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a per-step paired counterfactual cache")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="original LeWM HDF5 used to verify episode/start/source-row provenance",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effect-atol", type=float, default=1e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks: dict[str, bool] = {}
    failures: list[str] = []
    with h5py.File(args.dataset, "r") as handle:
        metadata = handle["metadata"].attrs
        samples = handle["samples"]
        schema = metadata.get("schema_version")
        checks["schema_version"] = schema == SCHEMA_VERSION
        n, h, h2 = samples["pulse_noop_mask"].shape
        if h != h2:
            raise ValueError("pulse_noop_mask must have square position/offset axes")
        checks["finite_factual"] = bool(
            np.isfinite(np.asarray(samples["factual_latents"])).all()
            and np.isfinite(np.asarray(samples["factual_states"])).all()
        )
        checks["finite_pulse"] = bool(
            np.isfinite(np.asarray(samples["pulse_noop_latents"])).all()
            and np.isfinite(np.asarray(samples["pulse_noop_states"])).all()
        )
        mask = np.asarray(samples["pulse_noop_mask"], dtype=bool)
        expected_mask = np.zeros_like(mask)
        for position in range(h):
            expected_mask[:, position, : h - position] = True
        checks["triangular_mask"] = bool(np.array_equal(mask, expected_mask))

        factual_actions = np.asarray(samples["factual_actions"])
        pulse_actions = np.asarray(samples["pulse_noop_actions"])
        factual = np.asarray(samples["factual_latents"])
        pulse = np.asarray(samples["pulse_noop_latents"])
        effect = np.asarray(samples["effect_latents"])
        current_zero = []
        suffix_equal = []
        effect_error = []
        for position in range(h):
            current_zero.append(np.all(pulse_actions[:, position, 0] == 0))
            length = h - position
            if length > 1:
                suffix_equal.append(
                    np.array_equal(
                        pulse_actions[:, position, 1:length], factual_actions[:, position + 1 :]
                    )
                )
            else:
                suffix_equal.append(True)
            expected = pulse[:, position, :length] - factual[:, position + 1 :]
            effect_error.append(float(np.max(np.abs(effect[:, position, :length] - expected))))
        checks["current_action_is_raw_zero"] = bool(all(current_zero))
        checks["factual_suffix_is_shared"] = bool(all(suffix_equal))
        checks["effect_target_aligned"] = bool(max(effect_error, default=0.0) <= args.effect_atol)
        checks["restore_consistent"] = bool(np.asarray(samples["restore_consistent"]).all())
        checks["branch_starts_from_factual_snapshot"] = bool(
            np.array_equal(
                np.asarray(samples["source_snapshot_hashes"]),
                np.asarray(samples["restored_snapshot_hashes"]),
            )
        )
        checks["factual_state_replay_exact"] = bool(
            np.max(np.asarray(samples["factual_replay_error"])) == 0.0
        )
        checks["factual_image_replay_exact"] = bool(
            np.max(np.asarray(samples["factual_replay_image_error"])) == 0.0
        )
        split = np.asarray(samples["split"])
        episodes = np.asarray(samples["episode_id"])
        starts = np.asarray(samples["start_step"])
        source_rows = np.asarray(samples["source_row"])
        sample_ids = [
            item.decode() if isinstance(item, bytes) else str(item)
            for item in samples["sample_id"][:]
        ]
        groups = [set(episodes[split == code].tolist()) for code in SPLIT_CODES.values()]
        checks["split_disjoint"] = not any(
            groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)
        )
        checks["finite_actions"] = bool(
            np.isfinite(factual_actions).all() and np.isfinite(pulse_actions).all()
        )
        source_offsets = source_rows - starts
        checks["source_rows_nonnegative"] = bool(np.all(source_rows >= 0) and np.all(starts >= 0))
        checks["episode_source_offset_consistent"] = bool(
            all(
                np.unique(source_offsets[episodes == episode]).size == 1
                for episode in np.unique(episodes)
            )
        )
        checks["sample_ids_unique"] = len(set(sample_ids)) == len(sample_ids)
        checks["sample_id_provenance"] = bool(
            all(
                f":ep{int(episode)}:start{int(start)}:" in sample_id
                for sample_id, episode, start in zip(sample_ids, episodes, starts, strict=True)
            )
        )
        with h5py.File(args.source, "r") as source:
            source_episode_ids = np.unique(np.asarray(source["ep_idx"], dtype=np.int64))
            source_offsets_table = np.asarray(source["ep_offset"], dtype=np.int64)
            source_lengths = np.asarray(source["ep_len"], dtype=np.int64)
            lookup = {int(episode): index for index, episode in enumerate(source_episode_ids)}
            known_episodes = np.asarray(
                [int(episode) in lookup for episode in episodes], dtype=bool
            )
            expected_rows = np.full_like(source_rows, -1)
            within_episode = np.zeros_like(known_episodes)
            for index, (episode, start) in enumerate(zip(episodes, starts, strict=True)):
                table_index = lookup.get(int(episode))
                if table_index is None:
                    continue
                expected_rows[index] = source_offsets_table[table_index] + int(start)
                within_episode[index] = 0 <= int(start) < int(source_lengths[table_index])
            checks["source_episode_known"] = bool(known_episodes.all())
            checks["source_row_matches_episode_start"] = bool(
                np.array_equal(source_rows, expected_rows) and within_episode.all()
            )
            checks["source_row_direct_episode_match"] = bool(
                np.array_equal(
                    np.asarray(_take_rows(source["ep_idx"], source_rows), dtype=np.int64),
                    episodes,
                )
            )
            if "step_idx" in source:
                checks["source_row_direct_step_match"] = bool(
                    np.array_equal(
                        np.asarray(_take_rows(source["step_idx"], source_rows), dtype=np.int64),
                        starts,
                    )
                )
        if "audit" in handle:
            audit_indices = np.asarray(handle["audit/sample_index"], dtype=np.int64)
            checks["raw_audit_present"] = bool(len(audit_indices) > 0)
            checks["raw_audit_aligned"] = bool(
                len(audit_indices)
                == handle["audit/factual_images"].shape[0]
                == handle["audit/pulse_noop_images"].shape[0]
                and len(np.unique(audit_indices)) == len(audit_indices)
                and np.all((0 <= audit_indices) & (audit_indices < n))
            )
        else:
            checks["raw_audit_present"] = False
            checks["raw_audit_aligned"] = False
        for name, passed in checks.items():
            if not passed:
                failures.append(name)

    result = {
        "schema_version": "cc_rwkv_per_step_pairs_audit_v1",
        "dataset": str(args.dataset),
        "source": str(args.source),
        "samples": int(n),
        "model_horizon": int(h),
        "effect_atol": args.effect_atol,
        "checks": checks,
        "max_effect_alignment_error": max(effect_error, default=0.0),
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if failures:
        raise RuntimeError(f"per-step paired audit failed: {failures}")


if __name__ == "__main__":
    main()
