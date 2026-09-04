from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .baseline import _final_positions, _write_json
from .data import CalibrationRecord, save_calibration_records


@dataclass(frozen=True, slots=True)
class AttemptPair:
    record_id: str
    episode_index: int
    start_step: int
    duration: int


def select_split_attempts(
    episode_indices: np.ndarray,
    step_indices: np.ndarray,
    episode_lengths: np.ndarray,
    allowed_episodes: tuple[int, ...] | list[int],
    *,
    duration: int,
    count: int,
    seed: int,
) -> list[AttemptPair]:
    """Sample dataset starts from one declared trajectory split only."""

    episodes = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
    steps = np.asarray(step_indices, dtype=np.int64).reshape(-1)
    unique = np.unique(episodes)
    lengths = np.asarray(episode_lengths, dtype=np.int64).reshape(-1)
    if len(unique) != len(lengths):
        raise ValueError("episode length table does not match episode indices")
    length_by_episode = {int(ep): int(lengths[i]) for i, ep in enumerate(unique)}
    allowed = set(map(int, allowed_episodes))
    unknown = allowed - set(length_by_episode)
    if unknown:
        raise ValueError(f"split references unknown episodes: {sorted(unknown)[:5]}")
    valid = np.fromiter(
        (
            int(ep) in allowed
            and int(step) <= length_by_episode[int(ep)] - duration - 1
            for ep, step in zip(episodes, steps, strict=True)
        ),
        dtype=bool,
        count=len(episodes),
    )
    rows = np.flatnonzero(valid)
    if count > len(rows):
        raise ValueError(f"requested {count} attempts but only {len(rows)} are valid")
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(rows, size=count, replace=False))
    return [
        AttemptPair(
            record_id=(
                f"tworoom-risk:ep{int(episodes[row])}:start{int(steps[row])}:duration{duration}"
            ),
            episode_index=int(episodes[row]),
            start_step=int(steps[row]),
            duration=int(duration),
        )
        for row in selected
    ]


def _last_frame(value: Any) -> np.ndarray:
    array = np.asarray(value)
    return array[:, -1] if array.ndim >= 5 else array


@torch.inference_mode()
def _encode_raw_images(resources: Any, images: np.ndarray) -> np.ndarray:
    tensor = torch.stack([resources.image_transform(image) for image in images]).to(
        resources.device
    )
    latent = resources.model.encode({"pixels": tensor[:, None]})["emb"][:, -1]
    return latent.detach().float().cpu().numpy().astype(np.float32)


class AttemptLoggingSolver:
    """Record the selected low-level plan without changing CEM decisions."""

    def __init__(self, solver: Any, model: torch.nn.Module, device: torch.device) -> None:
        self.solver = solver
        self.model = model
        self.device = device
        self.initial_latents: np.ndarray | None = None
        self.goal_latents: np.ndarray | None = None
        self.predicted_paths: np.ndarray | None = None

    def configure(self, **kwargs: Any) -> None:
        self.solver.configure(**kwargs)

    @property
    def action_dim(self) -> int:
        return self.solver.action_dim

    @property
    def n_envs(self) -> int:
        return self.solver.n_envs

    @property
    def horizon(self) -> int:
        return self.solver.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        return self.solve(*args, **kwargs)

    @torch.inference_mode()
    def solve(
        self, info: dict[str, Any], init_action: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        outputs = self.solver.solve(info, init_action=init_action)
        if self.initial_latents is None:
            model_info = {
                key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in info.items()
            }
            current = self.model.encode({"pixels": model_info["pixels"]})["emb"][:, -1]
            goal = self.model.encode({"pixels": model_info["goal"]})["emb"][:, -1]
            actions = outputs["actions"].to(self.device)
            action_embeddings = self.model.action_encoder(actions)
            history_size = int(getattr(self.model.predictor, "num_frames", 3))
            states = [current]
            for step in range(actions.shape[1]):
                lo = max(0, len(states) - history_size)
                state_history = torch.stack(states[lo:], dim=1)
                action_history = action_embeddings[:, lo : step + 1]
                states.append(self.model.predict(state_history, action_history)[:, -1])
            self.initial_latents = current.detach().float().cpu().numpy().astype(np.float32)
            self.goal_latents = goal.detach().float().cpu().numpy().astype(np.float32)
            self.predicted_paths = (
                torch.stack(states[1:], dim=1).detach().float().cpu().numpy().astype(np.float32)
            )
        return outputs


class AttemptTrackingPolicy:
    """Track block-boundary latent residuals around an upstream MPC policy."""

    def __init__(
        self,
        policy: Any,
        logging_solver: AttemptLoggingSolver,
        resources: Any,
        num_envs: int,
        action_block: int,
    ) -> None:
        self.policy = policy
        self.logging_solver = logging_solver
        self.resources = resources
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.action_block = int(action_block)
        self.residuals: list[list[float]] = [[] for _ in range(num_envs)]
        self._recorded_blocks = np.zeros(num_envs, dtype=np.int64)

    def set_env(self, env: Any) -> None:
        self.env = env
        self.policy.set_env(env)

    @staticmethod
    def _flag(info: dict[str, Any], key: str, size: int) -> np.ndarray:
        value = np.asarray(info.get(key, np.zeros(size, dtype=bool)), dtype=bool)
        return value.reshape(size, -1)[:, -1]

    def _record_boundaries(self, info: dict[str, Any]) -> None:
        paths = self.logging_solver.predicted_paths
        if paths is None:
            return
        due = [
            index
            for index, step in enumerate(self.steps)
            if step > 0
            and step % self.action_block == 0
            and self._recorded_blocks[index] < step // self.action_block
        ]
        if not due:
            return
        images = _last_frame(info["pixels"])[due]
        actual = _encode_raw_images(self.resources, images)
        for row, index in enumerate(due):
            block = min(int(self.steps[index] // self.action_block - 1), paths.shape[1] - 1)
            self.residuals[index].append(float(np.linalg.norm(actual[row] - paths[index, block])))
            self._recorded_blocks[index] += 1

    def get_action(self, info: dict[str, Any]) -> np.ndarray:
        size = len(self.steps)
        self._record_boundaries(info)
        dead = self._flag(info, "terminated", size) | self._flag(info, "truncated", size)
        action = self.policy.get_action(info)
        self.steps[~dead] += 1
        return action

    def finalize(self, info: dict[str, Any]) -> None:
        paths = self.logging_solver.predicted_paths
        if paths is None:
            raise RuntimeError("low-level solver did not produce a plan")
        images = _last_frame(info["pixels"])
        actual = _encode_raw_images(self.resources, images)
        for index, step in enumerate(self.steps):
            required = max(1, int(np.ceil(step / self.action_block)))
            if self._recorded_blocks[index] >= required:
                continue
            block = min(required - 1, paths.shape[1] - 1)
            self.residuals[index].append(float(np.linalg.norm(actual[index] - paths[index, block])))
            self._recorded_blocks[index] += 1


def collect_attempt_group(
    resources: Any,
    pairs: list[AttemptPair],
    *,
    experiment: dict[str, Any],
    planner: dict[str, Any],
    cem: dict[str, Any],
    success_threshold: float,
) -> tuple[list[CalibrationRecord], dict[str, Any]]:
    """Run one duration-homogeneous batch of closed-loop MPC attempts."""

    import stable_worldmodel as swm

    if not pairs:
        return [], {}
    duration = pairs[0].duration
    if any(pair.duration != duration for pair in pairs):
        raise ValueError("an attempt group must have one duration")
    action_block = int(planner["action_block"])
    if duration % action_block:
        raise ValueError("attempt duration must be divisible by action_block")
    horizon = duration // action_block
    objective = swm.planning.GoalMSE()
    cost = swm.planning.ShootingCostEvaluator(resources.model, objective)
    base_solver = swm.planning.CEMSolver(
        cost=cost,
        batch_size=int(cem["batch_size"]),
        num_samples=int(cem["num_samples"]),
        var_scale=float(cem["variance_scale"]),
        n_steps=int(cem["iterations"]),
        topk=int(cem["topk"]),
        device=resources.device,
        seed=int(experiment["selection_seed"]) + duration,
    )
    logging_solver = AttemptLoggingSolver(base_solver, resources.model, resources.device)
    plan_config = swm.PlanConfig(
        horizon=horizon,
        receding_horizon=horizon,
        history_len=int(planner["history_len"]),
        action_block=action_block,
        warm_start=False,
    )
    base_policy = swm.policy.WorldModelPolicy(
        solver=logging_solver,
        config=plan_config,
        process=resources.process,
        transform={"pixels": resources.image_transform, "goal": resources.image_transform},
    )
    policy = AttemptTrackingPolicy(
        base_policy, logging_solver, resources, len(pairs), action_block
    )
    world = swm.World(
        env_name=str(experiment["environment"]),
        num_envs=len(pairs),
        max_episode_steps=2 * duration,
        image_shape=(int(experiment["image_size"]), int(experiment["image_size"])),
    )
    world.set_policy(policy)
    started = time.perf_counter()
    try:
        metrics = world.evaluate(
            dataset=resources.dataset,
            episodes_idx=[pair.episode_index for pair in pairs],
            start_steps=[pair.start_step for pair in pairs],
            goal_offset=duration,
            eval_budget=duration,
            callables=[
                {"method": "_set_state", "args": {"state": {"value": "pos_agent"}}},
                {
                    "method": "_set_goal_state",
                    "args": {"goal_state": {"value": "goal_pos_agent"}},
                },
            ],
            video=None,
        )
        _, _, distances = _final_positions(world)
        policy.finalize(world.infos)
    finally:
        world.close()
    initial = logging_solver.initial_latents
    goals = logging_solver.goal_latents
    if initial is None or goals is None:
        raise RuntimeError("attempt logger did not capture initial and goal latents")
    successes = np.asarray(metrics["episode_successes"], dtype=bool)
    records = [
        CalibrationRecord(
            record_id=pair.record_id,
            current_latent=initial[index],
            subgoal_latent=goals[index],
            duration=duration,
            observed_miss=float(distances[index]),
            success=bool(successes[index] and distances[index] <= success_threshold),
            step_residuals=np.asarray(policy.residuals[index], dtype=np.float32),
        )
        for index, pair in enumerate(pairs)
    ]
    return records, {
        "duration": duration,
        "attempts": len(records),
        "successes": int(sum(record.success for record in records)),
        "mean_endpoint_miss": float(np.mean(distances)),
        "wall_seconds": time.perf_counter() - started,
        "mean_executed_steps": float(np.mean(policy.steps)),
    }


def collect_split(
    resources: Any,
    split_name: str,
    allowed_episodes: tuple[int, ...] | list[int],
    *,
    durations: tuple[int, ...],
    attempts_per_duration: int,
    seed: int,
    output: Path,
    experiment: dict[str, Any],
    planner: dict[str, Any],
    cem: dict[str, Any],
) -> dict[str, Any]:
    all_records: list[CalibrationRecord] = []
    groups: list[dict[str, Any]] = []
    all_pairs: list[AttemptPair] = []
    for duration in durations:
        pairs = select_split_attempts(
            resources.dataset.get_col_data("ep_idx"),
            resources.dataset.get_col_data("step_idx"),
            resources.dataset.lengths,
            allowed_episodes,
            duration=duration,
            count=attempts_per_duration,
            seed=seed + duration,
        )
        all_pairs.extend(pairs)
        records, summary = collect_attempt_group(
            resources,
            pairs,
            experiment={**experiment, "selection_seed": seed},
            planner=planner,
            cem=cem,
            success_threshold=float(experiment["success_threshold"]),
        )
        all_records.extend(records)
        groups.append(summary)
        save_calibration_records(output / f"{split_name}_records.npz", all_records)
        _write_json(
            output / f"{split_name}_pairs.json",
            {
                "split": split_name,
                "durations": list(durations),
                "attempts_per_duration": attempts_per_duration,
                "pairs": [asdict(pair) for pair in all_pairs],
                "completed_groups": groups,
            },
        )
        print(json.dumps({"event": "risk_collection", "split": split_name, **summary}), flush=True)
    result = {
        "split": split_name,
        "records": len(all_records),
        "successes": int(sum(record.success for record in all_records)),
        "groups": groups,
    }
    _write_json(output / f"{split_name}_summary.json", result)
    return result
