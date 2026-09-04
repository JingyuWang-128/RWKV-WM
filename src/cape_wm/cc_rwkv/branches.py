from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BranchableEnv(Protocol):
    def snapshot(self) -> Mapping[str, Any]: ...

    def restore(self, snapshot: Mapping[str, Any]) -> None: ...

    def external_noise_state(self) -> Any: ...

    def set_external_noise_state(self, state: Any) -> None: ...


_VARIATION_PATHS = (
    "agent.color",
    "agent.radius",
    "agent.position",
    "agent.speed",
    "target.color",
    "target.radius",
    "target.position",
    "wall.color",
    "wall.thickness",
    "wall.axis",
    "wall.border_color",
    "door.color",
    "door.number",
    "door.size",
    "door.position",
    "background.color",
    "rendering.render_target",
    "task.min_steps",
)


def _variation_node(space: Any, path: str) -> Any:
    node = space
    for part in path.split("."):
        node = node[part]
    return node


def _copy_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def snapshot_sha256(snapshot: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _jsonable(snapshot), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class TwoRoomBranchAdapter:
    """Complete snapshot/restore wrapper for the official deterministic TwoRoom."""

    def __init__(
        self,
        env: Any,
        *,
        drift: np.ndarray | None = None,
        drift_amplitude: float = 0.0,
        drift_seed: int = 0,
    ) -> None:
        self.env = getattr(env, "unwrapped", env)
        self.elapsed_steps = 0
        self._drift_rng = np.random.default_rng(drift_seed)
        if drift is not None and drift_amplitude:
            raise ValueError("provide fixed drift or drift_amplitude, not both")
        self._fixed_drift = (
            np.asarray(drift, dtype=np.float32).copy() if drift is not None else None
        )
        if self._fixed_drift is not None and self._fixed_drift.shape != (2,):
            raise ValueError("drift must be a two-dimensional vector")
        self.drift_amplitude = float(drift_amplitude)
        self.drift = np.zeros(2, dtype=np.float32)

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.elapsed_steps = 0
        if self._fixed_drift is not None:
            self.drift = self._fixed_drift.copy()
        elif self.drift_amplitude:
            direction = int(self._drift_rng.integers(8))
            angle = direction * math.pi / 4
            self.drift = np.asarray(
                [math.cos(angle), math.sin(angle)], dtype=np.float32
            ) * self.drift_amplitude
        else:
            self.drift = np.zeros(2, dtype=np.float32)
        return observation, info

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if np.any(self.drift):
            import torch

            current = self.env.agent_position
            proposed = current + torch.as_tensor(self.drift, dtype=torch.float32)
            self.env.agent_position = self.env._apply_collisions(current, proposed)
            observation = self.env._get_obs()
            distance = float(
                torch.norm(self.env.agent_position - self.env.target_position)
            )
            terminated = distance < 16.0
            info = self.env._get_info()
            info["distance_to_target"] = distance
        self.elapsed_steps += 1
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return np.asarray(self.env.render()).copy()

    def state_vector(self) -> np.ndarray:
        return np.asarray(self.env.agent_position.detach().cpu(), dtype=np.float32)

    def snapshot(self) -> dict[str, Any]:
        variations = {
            path: _copy_value(_variation_node(self.env.variation_space, path).value)
            for path in _VARIATION_PATHS
        }
        return {
            "schema_version": "tworoom_snapshot_v1",
            "agent_position": np.asarray(
                self.env.agent_position.detach().cpu(), dtype=np.float32
            ),
            "target_position": np.asarray(
                self.env.target_position.detach().cpu(), dtype=np.float32
            ),
            "variations": variations,
            "wall_axis": int(self.env.wall_axis),
            "wall_thickness": int(self.env.wall_thickness),
            "num_doors": int(self.env.num_doors),
            "door_positions": np.asarray(
                self.env.door_positions.detach().cpu(), dtype=np.float32
            ),
            "door_sizes": np.asarray(
                self.env.door_sizes.detach().cpu(), dtype=np.float32
            ),
            "wall_pos": float(self.env.wall_pos),
            "elapsed_steps": int(self.elapsed_steps),
            "drift": self.drift.copy(),
            "env_rng": copy.deepcopy(self.env.np_random.bit_generator.state),
            "drift_rng": copy.deepcopy(self._drift_rng.bit_generator.state),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("schema_version") != "tworoom_snapshot_v1":
            raise ValueError("unsupported TwoRoom snapshot schema")
        for path, value in snapshot["variations"].items():
            _variation_node(self.env.variation_space, path).set_value(
                np.asarray(value).copy()
            )
        import torch

        self.env.agent_position = torch.as_tensor(
            snapshot["agent_position"], dtype=torch.float32
        ).clone()
        self.env.target_position = torch.as_tensor(
            snapshot["target_position"], dtype=torch.float32
        ).clone()
        self.env._cache_params()
        self.env._target_img = self.env._render_frame(
            agent_pos=self.env.target_position
        )
        self.elapsed_steps = int(snapshot["elapsed_steps"])
        self.drift = np.asarray(snapshot["drift"], dtype=np.float32).copy()
        self.env.np_random.bit_generator.state = copy.deepcopy(snapshot["env_rng"])
        self._drift_rng.bit_generator.state = copy.deepcopy(snapshot["drift_rng"])

        # Cached fields are redundant but audited: a mismatch means the snapshot
        # was made by an incompatible environment version.
        checks = (
            int(self.env.wall_axis) == int(snapshot["wall_axis"]),
            int(self.env.wall_thickness) == int(snapshot["wall_thickness"]),
            int(self.env.num_doors) == int(snapshot["num_doors"]),
            np.array_equal(self.env.door_positions.numpy(), snapshot["door_positions"]),
            np.array_equal(self.env.door_sizes.numpy(), snapshot["door_sizes"]),
            float(self.env.wall_pos) == float(snapshot["wall_pos"]),
        )
        if not all(checks):
            raise RuntimeError("restored TwoRoom cached geometry does not match snapshot")

    def external_noise_state(self) -> dict[str, Any]:
        return {
            "env_rng": copy.deepcopy(self.env.np_random.bit_generator.state),
            "drift_rng": copy.deepcopy(self._drift_rng.bit_generator.state),
        }

    def set_external_noise_state(self, state: Any) -> None:
        self.env.np_random.bit_generator.state = copy.deepcopy(state["env_rng"])
        self._drift_rng.bit_generator.state = copy.deepcopy(state["drift_rng"])

    def close(self) -> None:
        self.env.close()


class ActionDelayTwoRoomAdapter(TwoRoomBranchAdapter):
    """TwoRoom where the submitted action is executed after a fixed FIFO delay."""

    def __init__(self, env: Any, *, delay: int = 5, **kwargs: Any) -> None:
        if delay <= 0:
            raise ValueError("delay must be positive")
        super().__init__(env, **kwargs)
        self.delay = int(delay)
        self.action_fifo = np.zeros((delay, 2), dtype=np.float32)
        self.fifo_pointer = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        options = dict(options or {})
        if "action_history" not in options:
            raise ValueError("ActionDelay reset requires non-empty action_history")
        history = np.asarray(options.pop("action_history"), dtype=np.float32)
        if history.shape != (self.delay, 2) or not np.isfinite(history).all():
            raise ValueError(f"action_history must have shape [{self.delay}, 2]")
        output = super().reset(seed=seed, options=options)
        self.action_fifo = history.copy()
        self.fifo_pointer = 0
        return output

    def step(self, action: np.ndarray):
        submitted = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        executed = self.action_fifo[self.fifo_pointer].copy()
        self.action_fifo[self.fifo_pointer] = submitted
        self.fifo_pointer = (self.fifo_pointer + 1) % self.delay
        return super().step(executed)

    def state_vector(self) -> np.ndarray:
        ordered_fifo = np.concatenate(
            (
                self.action_fifo[self.fifo_pointer :],
                self.action_fifo[: self.fifo_pointer],
            ),
            axis=0,
        )
        return np.concatenate(
            (
                super().state_vector(),
                ordered_fifo.reshape(-1),
                np.asarray([self.fifo_pointer], dtype=np.float32),
            )
        ).astype(np.float32)

    def snapshot(self) -> dict[str, Any]:
        base = super().snapshot()
        base["schema_version"] = "action_delay_tworoom_snapshot_v1"
        base["action_fifo"] = self.action_fifo.copy()
        base["fifo_pointer"] = int(self.fifo_pointer)
        base["delay"] = self.delay
        return base

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("schema_version") != "action_delay_tworoom_snapshot_v1":
            raise ValueError("unsupported ActionDelay snapshot schema")
        if int(snapshot["delay"]) != self.delay:
            raise ValueError("snapshot action delay does not match adapter")
        base = dict(snapshot)
        base["schema_version"] = "tworoom_snapshot_v1"
        super().restore(base)
        self.action_fifo = np.asarray(snapshot["action_fifo"], dtype=np.float32).copy()
        self.fifo_pointer = int(snapshot["fifo_pointer"])


class BranchType(IntEnum):
    REFERENCE = 0
    FACTUAL = 1
    PULSE_NOOP = 2
    PULSE_LOCAL = 3


BRANCH_ORDER = (
    BranchType.REFERENCE,
    BranchType.FACTUAL,
    BranchType.PULSE_NOOP,
    BranchType.PULSE_LOCAL,
)


class ActionSupportFilter:
    """Nearest-neighbour support audit for primitive or blocked actions."""

    def __init__(
        self,
        behavior_actions: np.ndarray,
        *,
        quantile: float = 0.05,
        action_low: float = -1.0,
        action_high: float = 1.0,
        threshold: float | None = None,
    ) -> None:
        actions = np.asarray(behavior_actions, dtype=np.float32)
        actions = actions[np.isfinite(actions).all(axis=-1)]
        if actions.ndim != 2 or len(actions) < 3:
            raise ValueError("behavior_actions must contain at least three finite actions")
        if not 0 <= quantile < 1:
            raise ValueError("support quantile must be in [0, 1)")
        from scipy.spatial import cKDTree

        self.behavior_actions = actions
        self.tree = cKDTree(actions)
        if threshold is None:
            second_neighbor = self.tree.query(actions, k=2)[0][:, 1]
            self.threshold = float(np.quantile(-second_neighbor, quantile))
        else:
            self.threshold = float(threshold)
        self.action_low = float(action_low)
        self.action_high = float(action_high)

    def score(self, action_blocks: np.ndarray) -> np.ndarray:
        blocks = np.asarray(action_blocks, dtype=np.float32)
        if blocks.shape[-1] % self.behavior_actions.shape[-1]:
            raise ValueError("action block has an incompatible primitive action dimension")
        primitive = blocks.reshape(*blocks.shape[:-1], -1, self.behavior_actions.shape[-1])
        distances = self.tree.query(primitive.reshape(-1, primitive.shape[-1]), k=1)[0]
        distances = distances.reshape(primitive.shape[:-1])
        return -distances.mean(axis=-1).astype(np.float32)

    def supported(self, action_blocks: np.ndarray) -> np.ndarray:
        return self.score(action_blocks) >= self.threshold

    def local_perturbation(
        self,
        factual_block: np.ndarray,
        *,
        rng: np.random.Generator,
        fraction: float = 0.1,
        max_attempts: int = 64,
    ) -> tuple[np.ndarray, float]:
        factual = np.asarray(factual_block, dtype=np.float32)
        magnitude = fraction * (self.action_high - self.action_low)
        for _ in range(max_attempts):
            signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=factual.shape)
            candidate = np.clip(
                factual + signs * magnitude, self.action_low, self.action_high
            ).astype(np.float32)
            score = float(self.score(candidate[None])[0])
            if score >= self.threshold and not np.array_equal(candidate, factual):
                return candidate, score
        raise RuntimeError("failed to sample a support-valid local action perturbation")


@dataclass(slots=True)
class BranchRollout:
    branch_type: np.ndarray
    actions: np.ndarray
    images: np.ndarray
    states: np.ndarray
    support_score: np.ndarray
    snapshot_hash: str
    restore_consistent: bool


def build_branch_actions(
    factual_actions: np.ndarray,
    support: ActionSupportFilter,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    factual = np.asarray(factual_actions, dtype=np.float32)
    if factual.ndim != 2 or factual.shape[0] <= 0:
        raise ValueError("factual_actions must have shape [model_steps, action_block_dim]")
    local, _ = support.local_perturbation(factual[0], rng=rng)
    actions = np.stack((np.zeros_like(factual), factual, factual.copy(), factual.copy()))
    actions[BranchType.PULSE_NOOP, 0] = 0
    actions[BranchType.PULSE_LOCAL, 0] = local
    scores = support.score(actions)
    return actions, scores


def rollout_branches(
    env: BranchableEnv,
    factual_actions: np.ndarray,
    support: ActionSupportFilter,
    *,
    action_block: int,
    rng: np.random.Generator,
) -> BranchRollout:
    if not hasattr(env, "render") or not hasattr(env, "step") or not hasattr(env, "state_vector"):
        raise TypeError("branch environment lacks render/step/state_vector")
    actions, scores = build_branch_actions(factual_actions, support, rng=rng)
    if actions.shape[-1] % action_block:
        raise ValueError("blocked action width is not divisible by action_block")
    primitive_dim = actions.shape[-1] // action_block
    source_snapshot = env.snapshot()
    source_hash = snapshot_sha256(source_snapshot)
    noise_state = env.external_noise_state()
    branch_images: list[np.ndarray] = []
    branch_states: list[np.ndarray] = []
    consistent = True
    initial_render: np.ndarray | None = None
    for branch_index in range(len(BRANCH_ORDER)):
        env.restore(source_snapshot)
        env.set_external_noise_state(noise_state)
        image = env.render()
        if initial_render is None:
            initial_render = image.copy()
        else:
            consistent = consistent and np.array_equal(image, initial_render)
        images = [image]
        states = [env.state_vector().copy()]
        for model_step in range(actions.shape[1]):
            primitive_actions = actions[branch_index, model_step].reshape(
                action_block, primitive_dim
            )
            for primitive_action in primitive_actions:
                env.step(primitive_action)
            images.append(env.render())
            states.append(env.state_vector().copy())
        branch_images.append(np.stack(images))
        branch_states.append(np.stack(states))
    env.restore(source_snapshot)
    consistent = consistent and snapshot_sha256(env.snapshot()) == source_hash
    return BranchRollout(
        branch_type=np.asarray(BRANCH_ORDER, dtype=np.uint8),
        actions=actions,
        images=np.stack(branch_images),
        states=np.stack(branch_states).astype(np.float32),
        support_score=scores,
        snapshot_hash=source_hash,
        restore_consistent=bool(consistent),
    )
