from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .types import Array


@dataclass(frozen=True, slots=True)
class SplitManifest:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    calibration: tuple[str, ...]
    test: tuple[str, ...]
    seed: int
    content_hash: str = ""

    def assert_disjoint(self) -> None:
        groups = [set(self.train), set(self.validation), set(self.calibration), set(self.test)]
        for index, left in enumerate(groups):
            for right in groups[index + 1 :]:
                if left & right:
                    raise ValueError("trajectory split manifest contains leakage")

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n")


def deterministic_group_split(
    trajectory_ids: list[str] | tuple[str, ...],
    seed: int = 0,
    fractions: tuple[float, float, float, float] = (0.7, 0.1, 0.1, 0.1),
) -> SplitManifest:
    """Create trajectory-level train/validation/calibration/test splits."""

    if len(fractions) != 4 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("four split fractions must sum to one")
    identifiers = sorted(set(trajectory_ids))
    if len(identifiers) < 4:
        raise ValueError("at least four trajectories are required")
    raw_counts = np.asarray(fractions, dtype=np.float64) * len(identifiers)
    counts = np.maximum(1, np.floor(raw_counts).astype(int))
    while counts.sum() < len(identifiers):
        index = int(np.argmax(raw_counts - counts))
        counts[index] += 1
    while counts.sum() > len(identifiers):
        removable = np.where(counts > 1, counts - raw_counts, -np.inf)
        counts[int(np.argmax(removable))] -= 1

    rng = np.random.default_rng(seed)
    shuffled = [identifiers[index] for index in rng.permutation(len(identifiers))]
    boundaries = np.cumsum(counts)
    parts = np.split(np.asarray(shuffled, dtype=object), boundaries[:-1])
    digest_payload = json.dumps(
        {"trajectory_ids": identifiers, "fractions": fractions, "seed": seed},
        separators=(",", ":"),
    ).encode()
    manifest = SplitManifest(
        train=tuple(parts[0].tolist()),
        validation=tuple(parts[1].tolist()),
        calibration=tuple(parts[2].tolist()),
        test=tuple(parts[3].tolist()),
        seed=seed,
        content_hash=hashlib.sha256(digest_payload).hexdigest(),
    )
    if any(len(part) == 0 for part in parts):
        raise ValueError("dataset is too small for the requested four-way split")
    manifest.assert_disjoint()
    return manifest


def stable_pair_id(environment: str, seed: int, start: int, goal: int) -> str:
    payload = f"{environment}:{seed}:{start}:{goal}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(slots=True)
class Trajectory:
    trajectory_id: str
    observations: Array
    actions: Array
    latents: Array | None = None
    metadata: dict[str, Any] | None = None

    def validate(self) -> None:
        if self.observations.shape[0] != self.actions.shape[0] + 1:
            raise ValueError("a trajectory requires one more observation than action")
        if self.latents is not None and self.latents.shape[0] != self.observations.shape[0]:
            raise ValueError("latent and observation sequence lengths must match")


@dataclass(slots=True)
class MacroSegment:
    trajectory_id: str
    start: int
    duration: int
    start_latent: Array
    target_latent: Array
    actions: Array


def iter_macro_segments(
    trajectories: list[Trajectory],
    durations: tuple[int, ...] = (5, 10, 20, 40),
    stride: int = 1,
) -> Iterator[MacroSegment]:
    for trajectory in trajectories:
        trajectory.validate()
        if trajectory.latents is None:
            raise ValueError("latents must be cached before constructing macro segments")
        for duration in durations:
            for start in range(0, len(trajectory.actions) - duration + 1, stride):
                yield MacroSegment(
                    trajectory_id=trajectory.trajectory_id,
                    start=start,
                    duration=duration,
                    start_latent=trajectory.latents[start],
                    target_latent=trajectory.latents[start + duration],
                    actions=trajectory.actions[start : start + duration],
                )


@dataclass(slots=True)
class CalibrationRecord:
    record_id: str
    current_latent: Array
    subgoal_latent: Array
    duration: int
    observed_miss: float
    success: bool
    step_residuals: Array

    @property
    def max_residual(self) -> float:
        return float(np.max(self.step_residuals))


@dataclass(frozen=True, slots=True)
class GoalPair:
    pair_id: str
    trajectory_id: str
    start: int
    goal: int
    offset: int
    seed: int


def make_paired_test_list(
    environment: str,
    trajectory_lengths: dict[str, int],
    test_trajectory_ids: tuple[str, ...] | list[str],
    offsets: tuple[int, ...] = (25, 50, 75, 100),
    seeds: tuple[int, ...] = (0, 1, 2),
    episodes_per_condition: int = 100,
    sampling_seed: int = 0,
) -> list[GoalPair]:
    """Freeze identical start-goal pairs for all methods before evaluation."""

    if episodes_per_condition <= 0:
        raise ValueError("episodes_per_condition must be positive")
    rng = np.random.default_rng(sampling_seed)
    pairs: list[GoalPair] = []
    for seed in seeds:
        for offset in offsets:
            eligible = [
                (trajectory_id, start, start + offset)
                for trajectory_id in sorted(test_trajectory_ids)
                for start in range(max(0, trajectory_lengths[trajectory_id] - offset + 1))
            ]
            if len(eligible) < episodes_per_condition:
                raise ValueError(
                    f"only {len(eligible)} pairs are available for offset {offset}; "
                    f"requested {episodes_per_condition}"
                )
            chosen = rng.choice(len(eligible), size=episodes_per_condition, replace=False)
            for index in chosen:
                trajectory_id, start, goal = eligible[int(index)]
                pairs.append(
                    GoalPair(
                        pair_id=stable_pair_id(environment, seed, start, goal)
                        + "-"
                        + hashlib.sha256(trajectory_id.encode()).hexdigest()[:8],
                        trajectory_id=trajectory_id,
                        start=start,
                        goal=goal,
                        offset=offset,
                        seed=seed,
                    )
                )
    return pairs


def save_goal_pairs(path: str | Path, pairs: list[GoalPair]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps([asdict(pair) for pair in pairs], indent=2, ensure_ascii=False) + "\n"
    )


def save_calibration_records(path: str | Path, records: list[CalibrationRecord]) -> None:
    payload: dict[str, Any] = {"count": np.asarray(len(records), dtype=np.int64)}
    for index, record in enumerate(records):
        prefix = f"record_{index}"
        payload[f"{prefix}_id"] = np.asarray(record.record_id)
        payload[f"{prefix}_current"] = record.current_latent
        payload[f"{prefix}_subgoal"] = record.subgoal_latent
        payload[f"{prefix}_duration"] = np.asarray(record.duration, dtype=np.int64)
        payload[f"{prefix}_miss"] = np.asarray(record.observed_miss, dtype=np.float32)
        payload[f"{prefix}_success"] = np.asarray(record.success, dtype=np.bool_)
        payload[f"{prefix}_residuals"] = record.step_residuals
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **payload)


def load_calibration_records(path: str | Path) -> list[CalibrationRecord]:
    archive = np.load(path, allow_pickle=False)
    records: list[CalibrationRecord] = []
    for index in range(int(archive["count"])):
        prefix = f"record_{index}"
        records.append(
            CalibrationRecord(
                record_id=str(archive[f"{prefix}_id"]),
                current_latent=archive[f"{prefix}_current"],
                subgoal_latent=archive[f"{prefix}_subgoal"],
                duration=int(archive[f"{prefix}_duration"]),
                observed_miss=float(archive[f"{prefix}_miss"]),
                success=bool(archive[f"{prefix}_success"]),
                step_residuals=archive[f"{prefix}_residuals"],
            )
        )
    return records


def save_npz_trajectories(path: str | Path, trajectories: list[Trajectory]) -> None:
    """Portable artifact format used by training scripts and synthetic tests."""

    payload: dict[str, Any] = {"count": np.asarray(len(trajectories), dtype=np.int64)}
    for index, trajectory in enumerate(trajectories):
        trajectory.validate()
        prefix = f"trajectory_{index}"
        payload[f"{prefix}_id"] = np.asarray(trajectory.trajectory_id)
        payload[f"{prefix}_observations"] = trajectory.observations
        payload[f"{prefix}_actions"] = trajectory.actions
        if trajectory.latents is not None:
            payload[f"{prefix}_latents"] = trajectory.latents
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **payload)


def load_npz_trajectories(path: str | Path) -> list[Trajectory]:
    archive = np.load(path, allow_pickle=False)
    trajectories: list[Trajectory] = []
    for index in range(int(archive["count"])):
        prefix = f"trajectory_{index}"
        latent_key = f"{prefix}_latents"
        trajectories.append(
            Trajectory(
                trajectory_id=str(archive[f"{prefix}_id"]),
                observations=archive[f"{prefix}_observations"],
                actions=archive[f"{prefix}_actions"],
                latents=archive[latent_key] if latent_key in archive else None,
            )
        )
    return trajectories
