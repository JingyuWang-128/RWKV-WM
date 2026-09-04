from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn import preprocessing

from .config import load_config
from .device import resolve_device


@dataclass(frozen=True, slots=True)
class DatasetPair:
    pair_id: str
    episode_index: int
    start_step: int
    goal_offset: int


@dataclass(slots=True)
class PlanningCounters:
    solver_invocations: int = 0
    planned_environments: int = 0
    cost_calls: int = 0
    candidate_sequences: int = 0
    predicted_transitions: int = 0
    solver_seconds: float = 0.0
    policy_seconds: float = 0.0
    high_level_invocations: int = 0
    high_level_candidates: int = 0
    high_level_predicted_transitions: int = 0


@dataclass(slots=True)
class BaselineResources:
    dataset: Any
    model: torch.nn.Module
    process: dict[str, Any]
    image_transform: Any
    device: torch.device
    requested_device: str
    data_path: Path
    weights_path: Path
    model_config_path: Path
    weights_sha256: str
    model_config_sha256: str


def select_pairs(
    episode_indices: np.ndarray,
    step_indices: np.ndarray,
    episode_lengths: np.ndarray,
    *,
    goal_offset: int,
    num_eval: int,
    seed: int,
) -> tuple[list[DatasetPair], np.ndarray]:
    """Reproduce stable-worldmodel's seeded dataset-pair selection."""

    episodes = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
    steps = np.asarray(step_indices, dtype=np.int64).reshape(-1)
    lengths = np.asarray(episode_lengths, dtype=np.int64).reshape(-1)
    unique_episodes = np.unique(episodes)
    if len(unique_episodes) != len(lengths):
        raise ValueError("episode length table does not match the episode index column")
    if goal_offset <= 0 or num_eval <= 0:
        raise ValueError("goal_offset and num_eval must be positive")

    max_start = {int(ep): int(lengths[i] - goal_offset - 1) for i, ep in enumerate(unique_episodes)}
    valid = np.fromiter(
        (steps[i] <= max_start[int(ep)] for i, ep in enumerate(episodes)),
        dtype=bool,
        count=len(episodes),
    )
    valid_rows = np.flatnonzero(valid)
    if num_eval > len(valid_rows):
        raise ValueError(f"requested {num_eval} tasks but only {len(valid_rows)} are valid")

    rng = np.random.default_rng(seed)
    chosen_rows = np.sort(rng.choice(valid_rows, size=num_eval, replace=False))
    pairs = [
        DatasetPair(
            pair_id=f"tworoom:ep{int(episodes[row])}:start{int(steps[row])}:offset{goal_offset}",
            episode_index=int(episodes[row]),
            start_step=int(steps[row]),
            goal_offset=goal_offset,
        )
        for row in chosen_rows
    ]
    return pairs, chosen_rows


class CountingCost(torch.nn.Module):
    def __init__(self, cost: torch.nn.Module, counters: PlanningCounters) -> None:
        super().__init__()
        self.cost = cost
        self.counters = counters

    def get_cost(self, info_dict: dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        batch, samples, horizon = actions.shape[:3]
        self.counters.cost_calls += 1
        self.counters.candidate_sequences += int(batch * samples)
        self.counters.predicted_transitions += int(batch * samples * horizon)
        return self.cost.get_cost(info_dict, actions)


class TimedSolver:
    def __init__(self, solver: Any, counters: PlanningCounters) -> None:
        self.solver = solver
        self.counters = counters

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

    def solve(self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None) -> dict:
        start = time.perf_counter()
        self.counters.solver_invocations += 1
        self.counters.planned_environments += len(next(iter(info_dict.values())))
        try:
            return self.solver.solve(info_dict, init_action=init_action)
        finally:
            self.counters.solver_seconds += time.perf_counter() - start

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)


class TrackingPolicy:
    def __init__(self, policy: Any, num_envs: int, counters: PlanningCounters) -> None:
        self.policy = policy
        self.counters = counters
        self.steps = np.zeros(num_envs, dtype=np.int64)

    def set_env(self, env: Any) -> None:
        self.env = env
        self.policy.set_env(env)

    @staticmethod
    def _flag(info: dict[str, Any], key: str, size: int) -> np.ndarray:
        value = np.asarray(info.get(key, np.zeros(size, dtype=bool)), dtype=bool)
        return value.reshape(size, -1)[:, -1]

    def get_action(self, info: dict[str, Any]) -> np.ndarray:
        size = len(self.steps)
        dead = self._flag(info, "terminated", size) | self._flag(info, "truncated", size)
        self.steps[~dead] += 1
        start = time.perf_counter()
        try:
            return self.policy.get_action(info)
        finally:
            self.counters.policy_seconds += time.perf_counter() - start


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _final_positions(world: Any) -> tuple[list[list[float]], list[list[float]], list[float]]:
    agents: list[list[float]] = []
    goals: list[list[float]] = []
    distances: list[float] = []
    for wrapped in world.envs.envs:
        env = wrapped.unwrapped
        agent = np.asarray(env.agent_position.detach().cpu(), dtype=np.float64)
        goal = np.asarray(env.target_position.detach().cpu(), dtype=np.float64)
        agents.append(agent.tolist())
        goals.append(goal.tolist())
        distances.append(float(np.linalg.norm(agent - goal)))
    return agents, goals, distances


def prepare_resources(
    *,
    data: Path,
    weights: Path,
    experiment_config: Path,
    runtime_config: Path,
    cache_dir: Path,
    device_override: str | None = None,
) -> BaselineResources:
    import hdf5plugin  # noqa: F401 -- registers HDF5 compression filters
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from torchvision.transforms import v2 as transforms

    experiment = load_config(experiment_config)["experiment"]
    runtime = load_config(runtime_config)["runtime"]
    # Evaluation is often launched once per accelerator.  Apply the registered
    # host-thread limits before loading the model so concurrent GPU jobs do not
    # each inherit PyTorch's machine-wide CPU default.
    torch.set_num_threads(int(runtime.get("torch_num_threads", torch.get_num_threads())))
    if "torch_num_interop_threads" in runtime:
        interop_threads = int(runtime["torch_num_interop_threads"])
        if torch.get_num_interop_threads() != interop_threads:
            torch.set_num_interop_threads(interop_threads)
    requested_device = device_override or runtime["device"]
    device = resolve_device(requested_device)
    data_path = data.resolve()
    weights_path = weights.resolve()
    model_config_path = weights_path.parent / "config.json"
    for required in (weights_path, model_config_path, data_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = swm.data.load_dataset(
        str(data_path),
        cache_dir=str(cache_dir),
        keys_to_cache=["action", "proprio"],
    )
    model = swm.wm.utils.load_pretrained(str(weights_path), cache_dir=str(cache_dir))
    model = model.to(device).eval()
    model.requires_grad_(False)
    image_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=int(experiment["image_size"])),
        ]
    )
    process: dict[str, Any] = {}
    for key in ("action", "proprio"):
        values = dataset.get_col_data(key)
        values = values[~np.isnan(values).any(axis=1)]
        process[key] = preprocessing.StandardScaler().fit(values)
        if key != "action":
            process[f"goal_{key}"] = process[key]
    return BaselineResources(
        dataset=dataset,
        model=model,
        process=process,
        image_transform=image_transform,
        device=device,
        requested_device=str(requested_device),
        data_path=data_path,
        weights_path=weights_path,
        model_config_path=model_config_path,
        weights_sha256=_sha256(weights_path),
        model_config_sha256=_sha256(model_config_path),
    )


def run_baseline(
    args: argparse.Namespace, resources: BaselineResources | None = None
) -> dict[str, Any]:
    import stable_worldmodel as swm

    experiment_config = load_config(args.experiment_config)
    runtime_config = load_config(args.runtime_config)
    exp = dict(experiment_config["experiment"])
    planner_cfg = experiment_config["planner"]
    cem_cfg = experiment_config["cem"]
    if args.selection_seed is not None:
        exp["selection_seed"] = int(args.selection_seed)
    if args.goal_offset is not None:
        exp["goal_offset"] = int(args.goal_offset)
    if args.eval_budget is not None:
        exp["eval_budget"] = int(args.eval_budget)
    num_eval = args.num_eval if args.num_eval is not None else int(exp["num_eval"])
    exp["num_eval"] = num_eval
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    resources = resources or prepare_resources(
        data=args.data,
        weights=args.weights,
        experiment_config=args.experiment_config,
        runtime_config=args.runtime_config,
        cache_dir=cache_dir,
        device_override=args.device,
    )
    dataset = resources.dataset
    model = resources.model
    process = resources.process
    image_transform = resources.image_transform
    device = resources.device
    requested_device = resources.requested_device
    data_path = resources.data_path
    weights_path = resources.weights_path
    model_config_path = resources.model_config_path
    pairs, selected_rows = select_pairs(
        dataset.get_col_data("ep_idx"),
        dataset.get_col_data("step_idx"),
        dataset.lengths,
        goal_offset=int(exp["goal_offset"]),
        num_eval=num_eval,
        seed=int(exp["selection_seed"]),
    )
    pair_payload = {
        "selection_seed": int(exp["selection_seed"]),
        "dataset": str(data_path),
        "selected_rows": selected_rows.tolist(),
        "pairs": [asdict(pair) for pair in pairs],
    }
    _write_json(output_dir / "pairs.json", pair_payload)

    counters = PlanningCounters()
    objective = swm.planning.GoalMSE()
    base_cost = swm.planning.ShootingCostEvaluator(model, objective)
    counted_cost = CountingCost(base_cost, counters)
    base_solver = swm.planning.CEMSolver(
        cost=counted_cost,
        batch_size=int(cem_cfg["batch_size"]),
        num_samples=int(cem_cfg["num_samples"]),
        var_scale=float(cem_cfg["variance_scale"]),
        n_steps=int(cem_cfg["iterations"]),
        topk=int(cem_cfg["topk"]),
        device=device,
        seed=int(exp["selection_seed"]),
    )
    solver = TimedSolver(base_solver, counters)
    plan_config = swm.PlanConfig(
        horizon=int(planner_cfg["horizon"]),
        receding_horizon=int(planner_cfg["receding_horizon"]),
        history_len=int(planner_cfg["history_len"]),
        action_block=int(planner_cfg["action_block"]),
        warm_start=bool(planner_cfg["warm_start"]),
    )
    base_policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan_config,
        process=process,
        transform={"pixels": image_transform, "goal": image_transform},
    )
    policy = TrackingPolicy(base_policy, num_eval, counters)

    world = swm.World(
        env_name=str(exp["environment"]),
        num_envs=num_eval,
        max_episode_steps=2 * int(exp["eval_budget"]),
        image_shape=(int(exp["image_size"]), int(exp["image_size"])),
    )
    world.set_policy(policy)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started_at = time.time()
    wall_start = time.perf_counter()
    try:
        metrics = world.evaluate(
            dataset=dataset,
            episodes_idx=[pair.episode_index for pair in pairs],
            start_steps=[pair.start_step for pair in pairs],
            goal_offset=int(exp["goal_offset"]),
            eval_budget=int(exp["eval_budget"]),
            callables=[
                {"method": "_set_state", "args": {"state": {"value": "pos_agent"}}},
                {
                    "method": "_set_goal_state",
                    "args": {"goal_state": {"value": "goal_pos_agent"}},
                },
            ],
            video=output_dir / "videos" if bool(exp["save_video"]) else None,
        )
        wall_seconds = time.perf_counter() - wall_start
        agents, goals, distances = _final_positions(world)
    finally:
        world.close()

    successes = np.asarray(metrics["episode_successes"], dtype=bool)
    success_interval = binomtest(int(successes.sum()), num_eval).proportion_ci(
        confidence_level=0.95, method="exact"
    )
    episode_records = []
    for i, pair in enumerate(pairs):
        episode_records.append(
            {
                **asdict(pair),
                "selection_seed": int(exp["selection_seed"]),
                "success": bool(successes[i]),
                "steps": int(policy.steps[i]),
                "final_state_error": distances[i],
                "final_agent_position": agents[i],
                "goal_position": goals[i],
            }
        )
    with (output_dir / "episodes.jsonl").open("w") as handle:
        for record in episode_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    peak_cuda_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    summary = {
        "status": "complete",
        "experiment": exp,
        "planner": planner_cfg,
        "cem": cem_cfg,
        "runtime": {
            "requested_device": str(requested_device),
            "resolved_device": str(device),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": _package_version("torchvision"),
            "transformers": _package_version("transformers"),
            "stable_worldmodel": _package_version("stable-worldmodel"),
            "stable_pretraining": _package_version("stable-pretraining"),
        },
        "assets": {
            "dataset": str(data_path),
            "dataset_bytes": data_path.stat().st_size,
            "weights": str(weights_path),
            "weights_sha256": resources.weights_sha256,
            "model_config": str(model_config_path),
            "model_config_sha256": resources.model_config_sha256,
        },
        "results": {
            "num_tasks": num_eval,
            "successes": int(successes.sum()),
            "success_rate": float(successes.mean()),
            "success_rate_percent": float(metrics["success_rate"]),
            "success_rate_exact_95_ci": [
                float(success_interval.low),
                float(success_interval.high),
            ],
            "mean_final_state_error": float(np.mean(distances)),
            "median_final_state_error": float(np.median(distances)),
            "mean_steps": float(np.mean(policy.steps)),
            "wall_time_seconds": wall_seconds,
            "peak_cuda_memory_bytes": peak_cuda_memory,
        },
        "planning": asdict(counters),
        "started_at_unix": started_at,
        "finished_at_unix": time.time(),
    }
    _write_json(output_dir / "summary.json", summary)
    resolved_experiment_config = dict(experiment_config)
    resolved_experiment_config["experiment"] = exp
    _write_json(
        output_dir / "resolved_config.json",
        {"experiment": resolved_experiment_config, "runtime": runtime_config},
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the official LeWM Two-Room MPC baseline")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/lewm_tworooms_baseline.yaml"),
    )
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/runtime.yaml"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/results/lewm_tworooms_official")
    )
    parser.add_argument("--device", default=None, help="optional runtime.yaml override")
    parser.add_argument("--num-eval", type=int, default=None, help="optional pilot override")
    parser.add_argument("--selection-seed", type=int, default=None)
    parser.add_argument("--goal-offset", type=int, default=None)
    parser.add_argument("--eval-budget", type=int, default=None)
    run_baseline(parser.parse_args())


if __name__ == "__main__":
    main()
