from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True, slots=True)
class HorizonSpec:
    """Unambiguous mapping between model transitions and primitive actions."""

    primitive_horizons: tuple[int, ...]
    action_block: int
    observation_stride: int

    def __post_init__(self) -> None:
        if self.action_block <= 0:
            raise ValueError("action_block must be positive")
        if self.observation_stride <= 0:
            raise ValueError("observation_stride must be positive")
        if not self.primitive_horizons:
            raise ValueError("primitive_horizons cannot be empty")
        if tuple(sorted(set(self.primitive_horizons))) != self.primitive_horizons:
            raise ValueError("primitive_horizons must be sorted and unique")
        invalid = [
            horizon
            for horizon in self.primitive_horizons
            if horizon <= 0 or horizon % self.action_block
        ]
        if invalid:
            raise ValueError(
                "every primitive horizon must be positive and divisible by "
                f"action_block={self.action_block}: {invalid}"
            )
        if self.observation_stride != self.action_block:
            raise ValueError(
                "M0 requires observation_stride == action_block so every target "
                "corresponds to one predictor transition"
            )

    @property
    def model_horizons(self) -> tuple[int, ...]:
        return tuple(item // self.action_block for item in self.primitive_horizons)

    @property
    def max_primitive_horizon(self) -> int:
        return self.primitive_horizons[-1]

    @property
    def max_model_horizon(self) -> int:
        return self.model_horizons[-1]

    def as_dict(self) -> dict[str, Any]:
        return {
            "primitive_horizons": list(self.primitive_horizons),
            "model_horizons": list(self.model_horizons),
            "action_block": self.action_block,
            "observation_stride": self.observation_stride,
        }


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def load_frozen_test_episode_ids(path: str | Path) -> tuple[int, ...]:
    source = Path(path)
    records = [json.loads(line) for line in source.read_text().splitlines() if line]
    if not records:
        raise ValueError(f"frozen test manifest is empty: {source}")
    episodes = tuple(sorted({int(record["episode_index"]) for record in records}))
    if len(episodes) == 0:
        raise ValueError("frozen test manifest contains no episode IDs")
    return episodes


def _episode_tables(handle: h5py.File) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    episode_ids = np.unique(np.asarray(handle["ep_idx"], dtype=np.int64))
    offsets = np.asarray(handle["ep_offset"], dtype=np.int64)
    lengths = np.asarray(handle["ep_len"], dtype=np.int64)
    if not (len(episode_ids) == len(offsets) == len(lengths)):
        raise ValueError("inconsistent HDF5 episode tables")
    return episode_ids, offsets, lengths


def build_tworoom_open_loop_arrays(
    data_path: str | Path,
    frozen_test_episodes: tuple[int, ...],
    horizon: HorizonSpec,
    *,
    sample_count: int,
    selection_seed: int,
    history_frames: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Freeze held-out factual actions and target rows for open-loop B0.

    M0 intentionally uses one observed start frame. This keeps a complete
    100-primitive-step future inside the official 101-frame TwoRoom episodes
    and prevents a Transformer history window from being confused with the
    long-memory mechanism evaluated later in Gate D.
    """

    if history_frames != 1:
        raise ValueError("M0 open-loop protocol requires history_frames=1")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    test_set = set(map(int, frozen_test_episodes))
    with h5py.File(data_path, "r") as handle:
        episode_ids, offsets, lengths = _episode_tables(handle)
        lookup = {int(episode): index for index, episode in enumerate(episode_ids)}
        unknown = test_set - set(lookup)
        if unknown:
            raise ValueError(f"test manifest contains unknown episodes: {sorted(unknown)[:5]}")
        required_length = horizon.max_primitive_horizon + 1
        eligible = np.asarray(
            [
                episode
                for episode in sorted(test_set)
                if lengths[lookup[episode]] >= required_length
            ],
            dtype=np.int64,
        )
        if sample_count > len(eligible):
            raise ValueError(
                f"requested {sample_count} samples but only {len(eligible)} held-out "
                f"episodes contain {horizon.max_primitive_horizon} future actions"
            )
        rng = np.random.default_rng(selection_seed)
        selected = np.sort(rng.choice(eligible, size=sample_count, replace=False))

        model_primitive_steps = np.arange(
            0,
            horizon.max_primitive_horizon + horizon.observation_stride,
            horizon.observation_stride,
            dtype=np.int64,
        )
        rollout_rows = np.empty((sample_count, len(model_primitive_steps)), dtype=np.int64)
        action_blocks = np.empty(
            (sample_count, horizon.max_model_horizon, horizon.action_block * 2),
            dtype=np.float32,
        )
        evaluator_states = np.empty(
            (sample_count, len(model_primitive_steps), 2), dtype=np.float32
        )
        start_steps = np.zeros(sample_count, dtype=np.int64)

        actions_source = handle["action"]
        states_source = handle["pos_agent"]
        for item, episode in enumerate(selected):
            index = lookup[int(episode)]
            start_row = int(offsets[index])
            rows = start_row + model_primitive_steps
            dense_actions = np.asarray(
                actions_source[start_row : start_row + horizon.max_primitive_horizon],
                dtype=np.float32,
            )
            if dense_actions.shape != (horizon.max_primitive_horizon, 2):
                raise RuntimeError("open-loop action suffix has an unexpected shape")
            if not np.all(np.isfinite(dense_actions)):
                raise ValueError(f"episode {episode} contains non-finite open-loop actions")
            rollout_rows[item] = rows
            action_blocks[item] = dense_actions.reshape(horizon.max_model_horizon, -1)
            evaluator_states[item] = np.asarray(states_source[rows], dtype=np.float32)

        all_actions = np.asarray(handle["action"], dtype=np.float32)
        finite_actions = all_actions[np.isfinite(all_actions).all(axis=1)]
        # This intentionally matches both the released LeWM training script
        # (normalizer fit before train/validation split) and the official
        # evaluator. Changing it to a train-only fit would alter B0 inputs.
        action_mean = finite_actions.mean(axis=0).astype(np.float32)
        action_scale = finite_actions.std(axis=0).astype(np.float32)
        if np.any(action_scale <= 0):
            raise RuntimeError("training action scale is degenerate")

    sample_ids = np.asarray(
        [
            f"tworoom:ep{int(episode)}:start0:h{horizon.max_primitive_horizon}"
            for episode in selected
        ]
    )
    arrays = {
        "sample_id": sample_ids,
        "episode_id": selected,
        "start_step": start_steps,
        "primitive_steps": model_primitive_steps,
        "rollout_rows": rollout_rows,
        "action_blocks_raw": action_blocks,
        "evaluator_states": evaluator_states,
        "action_mean": action_mean,
        "action_scale": action_scale,
    }
    metadata = {
        "sample_count": sample_count,
        "selection_seed": selection_seed,
        "history_frames": history_frames,
        "eligible_test_episodes": int(len(eligible)),
        "frozen_test_episode_count": len(test_set),
        "horizon": horizon.as_dict(),
        "array_hashes": {name: array_sha256(value) for name, value in arrays.items()},
        "action_normalizer_fit": (
            "all finite primitive actions; matches released LeWM training/evaluation"
        ),
    }
    return arrays, metadata
