from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True, slots=True)
class TwoRoomSplit:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    calibration: tuple[int, ...]
    test: tuple[int, ...]
    seed: int
    content_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "calibration": list(self.calibration),
            "test": list(self.test),
            "seed": self.seed,
            "content_hash": self.content_hash,
        }


def test_episode_ids(matrix_episodes: Path) -> tuple[int, ...]:
    rows = [json.loads(line) for line in matrix_episodes.read_text().splitlines() if line]
    return tuple(sorted({int(row["episode_index"]) for row in rows}))


def make_split(
    episode_ids: np.ndarray,
    held_out_test: tuple[int, ...],
    *,
    seed: int = 3072,
    train_fraction: float = 0.8,
) -> TwoRoomSplit:
    identifiers = sorted(map(int, np.asarray(episode_ids).reshape(-1)))
    test = set(held_out_test)
    unknown = test - set(identifiers)
    if unknown:
        raise ValueError(f"test manifest references unknown episodes: {sorted(unknown)[:5]}")
    available = np.asarray([item for item in identifiers if item not in test], dtype=np.int64)
    if len(available) < 3:
        raise ValueError("not enough non-test episodes for train/validation/calibration")
    rng = np.random.default_rng(seed)
    available = available[rng.permutation(len(available))]
    train_count = int(np.floor(train_fraction * len(available)))
    remaining = len(available) - train_count
    validation_count = remaining // 2
    train = tuple(sorted(map(int, available[:train_count])))
    validation = tuple(sorted(map(int, available[train_count : train_count + validation_count])))
    calibration = tuple(sorted(map(int, available[train_count + validation_count :])))
    payload = json.dumps(
        {
            "episodes": identifiers,
            "test": sorted(test),
            "seed": seed,
            "train_fraction": train_fraction,
        },
        separators=(",", ":"),
    ).encode()
    split = TwoRoomSplit(
        train=train,
        validation=validation,
        calibration=calibration,
        test=tuple(sorted(test)),
        seed=seed,
        content_hash=hashlib.sha256(payload).hexdigest(),
    )
    groups = [set(split.train), set(split.validation), set(split.calibration), set(split.test)]
    for index, left in enumerate(groups):
        for right in groups[index + 1 :]:
            if left & right:
                raise RuntimeError("trajectory split leakage detected")
    return split


def episode_tables(handle: h5py.File) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.asarray(handle["ep_len"], dtype=np.int64)
    offsets = np.asarray(handle["ep_offset"], dtype=np.int64)
    episodes = np.unique(np.asarray(handle["ep_idx"], dtype=np.int64))
    if not (len(lengths) == len(offsets) == len(episodes)):
        raise ValueError("inconsistent HDF5 episode tables")
    return episodes, offsets, lengths


def block_rows(
    episode_ids: tuple[int, ...] | list[int],
    all_episode_ids: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    *,
    block_size: int = 5,
) -> np.ndarray:
    lookup = {int(ep): index for index, ep in enumerate(all_episode_ids)}
    rows: list[np.ndarray] = []
    for episode in episode_ids:
        index = lookup[int(episode)]
        # The final HDF5 record carries a NaN terminal action. It remains a
        # valid latent target, but no action block may start from it.
        local = np.arange(0, int(lengths[index]), block_size, dtype=np.int64)
        rows.append(int(offsets[index]) + local)
    return np.unique(np.concatenate(rows)) if rows else np.empty(0, dtype=np.int64)


def sample_temporal_pairs(
    episode_ids: tuple[int, ...],
    all_episode_ids: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    *,
    count: int,
    seed: int,
    block_size: int = 5,
    label_scale: float = 224.0,
) -> dict[str, np.ndarray]:
    lookup = {int(ep): index for index, ep in enumerate(all_episode_ids)}
    eligible = [ep for ep in episode_ids if lengths[lookup[int(ep)]] > block_size]
    if not eligible:
        raise ValueError("no episodes are long enough for temporal pairs")
    rng = np.random.default_rng(seed)
    source = np.empty(count, dtype=np.int64)
    goal = np.empty(count, dtype=np.int64)
    separation = np.empty(count, dtype=np.int64)
    sampled_episode = np.empty(count, dtype=np.int64)
    # Separation quantiles are sampled uniformly first, which prevents short
    # local pairs from dominating the full-horizon selector objective.
    for item in range(count):
        episode = int(eligible[int(rng.integers(len(eligible)))])
        index = lookup[episode]
        max_blocks = max(1, (int(lengths[index]) - 1) // block_size)
        delta_blocks = int(rng.integers(1, max_blocks + 1))
        start_block = int(rng.integers(0, max_blocks - delta_blocks + 1))
        left = int(offsets[index]) + start_block * block_size
        right = left + delta_blocks * block_size
        if rng.random() < 0.5:
            left, right = right, left
        source[item] = left
        goal[item] = right
        separation[item] = delta_blocks * block_size
        sampled_episode[item] = episode
    return {
        "source_rows": source,
        "goal_rows": goal,
        "separation": separation,
        "targets": separation.astype(np.float32) / float(label_scale),
        "episode_ids": sampled_episode,
    }


def sample_segments(
    episode_ids: tuple[int, ...],
    all_episode_ids: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    actions: np.ndarray,
    *,
    count: int,
    seed: int,
    durations: tuple[int, ...] = (5, 10, 20, 40),
    block_size: int = 5,
) -> dict[str, np.ndarray]:
    if any(duration <= 0 or duration % block_size for duration in durations):
        raise ValueError("durations must be positive multiples of block_size")
    lookup = {int(ep): index for index, ep in enumerate(all_episode_ids)}
    eligible: dict[int, list[int]] = {
        duration: [ep for ep in episode_ids if int(lengths[lookup[int(ep)]]) > duration]
        for duration in durations
    }
    if any(not values for values in eligible.values()):
        raise ValueError("at least one duration has no eligible episodes")
    rng = np.random.default_rng(seed)
    max_duration = max(durations)
    source_rows = np.empty(count, dtype=np.int64)
    target_rows = np.empty(count, dtype=np.int64)
    sampled_durations = np.empty(count, dtype=np.int64)
    sampled_episodes = np.empty(count, dtype=np.int64)
    action_chunks = np.zeros((count, max_duration, actions.shape[-1]), dtype=np.float32)
    for item in range(count):
        duration = int(durations[item % len(durations)])
        choices = eligible[duration]
        episode = int(choices[int(rng.integers(len(choices)))])
        index = lookup[episode]
        max_start_block = (int(lengths[index]) - duration - 1) // block_size
        start = int(rng.integers(max_start_block + 1)) * block_size
        source = int(offsets[index]) + start
        target = source + duration
        chunk = np.asarray(actions[source:target], dtype=np.float32)
        if not np.all(np.isfinite(chunk)):
            raise ValueError("sampled an invalid terminal action")
        source_rows[item] = source
        target_rows[item] = target
        sampled_durations[item] = duration
        sampled_episodes[item] = episode
        action_chunks[item, :duration] = chunk
    return {
        "source_rows": source_rows,
        "target_rows": target_rows,
        "durations": sampled_durations,
        "episode_ids": sampled_episodes,
        "actions": action_chunks,
    }
