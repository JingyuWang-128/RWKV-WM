from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import binomtest

from .baseline import (
    CountingCost,
    PlanningCounters,
    TimedSolver,
    TrackingPolicy,
    _final_positions,
    _package_version,
    _write_json,
    prepare_resources,
    select_pairs,
)
from .comparison_baselines import (
    HierarchicalMacroSolver,
    HybridTRMObjective,
    VLWMDynamics,
)
from .comparison_models import PairwiseReachabilityMetric, VariableLengthPredictor
from .config import load_config
from .tworoom_risk_collection import select_split_attempts
from .models import DurationConditionedMacroPredictor

METHOD_LABELS = {
    "flat_mpc": "Flat MPC + latent L2",
    "flat_trm": "Flat MPC + horizon-matched TRM",
    "hwm": "fixed-scale HWM Two-Room adaptation",
    "vlwm": "VLWM Two-Room adaptation",
    "hilewm_c": "Hi-LeWM-C Two-Room adaptation",
}


def _load_payload(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != "cape_wm_comparison_v1":
        raise ValueError(f"unsupported comparison checkpoint: {path}")
    return payload


def _load_trm(path: Path, device: torch.device) -> tuple[PairwiseReachabilityMetric, dict]:
    payload = _load_payload(path, device)
    cfg = payload["config"]
    model = PairwiseReachabilityMetric(
        latent_dim=int(cfg["latent_dim"]), hidden_dim=int(cfg["hidden_dim"])
    ).to(device)
    model.load_state_dict(payload["modules"]["trm"])
    model.eval().requires_grad_(False)
    return model, payload


def _load_macro(path: Path, device: torch.device) -> tuple[DurationConditionedMacroPredictor, dict]:
    payload = _load_payload(path, device)
    cfg = payload["config"]
    model = DurationConditionedMacroPredictor(
        latent_dim=int(cfg["latent_dim"]),
        macro_dim=int(cfg["macro_dim"]),
        hidden_dim=512,
        depth=3,
        max_duration=int(cfg["max_duration"]),
    ).to(device)
    model.load_state_dict(payload["modules"]["macro_predictor"])
    model.eval().requires_grad_(False)
    return model, payload


def _load_vlwm(path: Path, device: torch.device) -> tuple[VariableLengthPredictor, dict]:
    payload = _load_payload(path, device)
    cfg = payload["config"]
    model = VariableLengthPredictor(
        latent_dim=int(cfg["latent_dim"]),
        action_dim=int(cfg["action_dim"]),
        model_dim=int(cfg["model_dim"]),
        depth=int(cfg["depth"]),
        heads=int(cfg["heads"]),
        mlp_dim=int(cfg["mlp_dim"]),
        max_horizon=int(cfg["max_horizon"]),
    ).to(device)
    model.load_state_dict(payload["modules"]["vlwm"])
    model.eval().requires_grad_(False)
    return model, payload


def _assert_reference_pairs(
    reference_matrix: Path | None,
    goal_offset: int,
    seed: int,
    pairs: list[Any],
) -> str | None:
    if reference_matrix is None:
        return None
    path = reference_matrix / f"offset_{goal_offset}" / f"seed_{seed}" / "pairs.json"
    reference = json.loads(path.read_text())
    expected = [item["pair_id"] for item in reference["pairs"]]
    actual = [item.pair_id for item in pairs]
    if actual != expected:
        raise RuntimeError(f"comparison pairs differ from official Flat MPC manifest: {path}")
    return str(path.resolve())


def run_comparison(args: argparse.Namespace, resources: Any | None = None) -> dict[str, Any]:
    import stable_worldmodel as swm

    experiment_config = load_config(args.experiment_config)
    comparison_config = load_config(args.comparison_config)
    runtime_config = load_config(args.runtime_config)
    exp = dict(experiment_config["experiment"])
    exp.update(
        {
            "name": f"lewm_tworooms_{args.method}",
            "selection_seed": int(args.selection_seed),
            "goal_offset": int(args.goal_offset),
            "eval_budget": int(args.eval_budget),
            "num_eval": int(args.num_eval),
        }
    )
    planner_cfg = dict(experiment_config["planner"])
    cem_cfg = experiment_config["cem"]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resources = resources or prepare_resources(
        data=args.data,
        weights=args.weights,
        experiment_config=args.experiment_config,
        runtime_config=args.runtime_config,
        cache_dir=output_dir / "cache",
        device_override=args.device,
    )
    split_name = getattr(args, "split_name", None)
    skip_reference_check = bool(getattr(args, "skip_reference_check", False))
    if split_name is None:
        pairs, selected_rows = select_pairs(
            resources.dataset.get_col_data("ep_idx"),
            resources.dataset.get_col_data("step_idx"),
            resources.dataset.lengths,
            goal_offset=int(exp["goal_offset"]),
            num_eval=int(exp["num_eval"]),
            seed=int(exp["selection_seed"]),
        )
    else:
        if not skip_reference_check:
            raise ValueError("selection-split runs must use --skip-reference-check")
        split_manifest = getattr(
            args,
            "split_manifest",
            Path("artifacts/cache/tworoom_baselines/split_manifest.json"),
        )
        split = json.loads(Path(split_manifest).read_text())
        attempts = select_split_attempts(
            resources.dataset.get_col_data("ep_idx"),
            resources.dataset.get_col_data("step_idx"),
            resources.dataset.lengths,
            split[split_name],
            duration=int(exp["goal_offset"]),
            count=int(exp["num_eval"]),
            seed=int(exp["selection_seed"]),
        )
        from .baseline import DatasetPair

        pairs = [
            DatasetPair(
                pair_id=item.record_id.replace(
                    "tworoom-risk", f"tworoom-{split_name}"
                ),
                episode_index=item.episode_index,
                start_step=item.start_step,
                goal_offset=item.duration,
            )
            for item in attempts
        ]
        selected_rows = np.empty(0, dtype=np.int64)
    reference_path = (
        None
        if skip_reference_check
        else _assert_reference_pairs(
            args.reference_matrix,
            int(exp["goal_offset"]),
            int(exp["selection_seed"]),
            pairs,
        )
    )
    _write_json(
        output_dir / "pairs.json",
        {
            "selection_seed": int(exp["selection_seed"]),
            "dataset": str(resources.data_path),
            "reference_flat_mpc_manifest": reference_path,
            "selected_rows": selected_rows.tolist(),
            "pairs": [asdict(pair) for pair in pairs],
        },
    )

    trm, trm_payload = _load_trm(args.checkpoint_dir / "trm.pt", resources.device)
    trm_cfg = comparison_config["trm"]
    objective = HybridTRMObjective(
        trm,
        trm_weight=(
            0.0 if args.method == "flat_mpc" else float(trm_cfg["hybrid_trm_weight"])
        ),
        l2_weight=(
            1.0 if args.method == "flat_mpc" else float(trm_cfg["hybrid_l2_weight"])
        ),
    )
    counters = PlanningCounters()
    model_for_cost: torch.nn.Module = resources.model
    checkpoint_metadata: dict[str, Any] = {
        "trm": {
            "path": str((args.checkpoint_dir / "trm.pt").resolve()),
            "best_epoch": trm_payload["best_epoch"],
        }
    }
    if args.method == "vlwm":
        predictor, payload = _load_vlwm(args.checkpoint_dir / "vlwm.pt", resources.device)
        schedule = tuple(map(int, comparison_config["vlwm"]["schedules"][str(exp["goal_offset"])]))
        model_for_cost = VLWMDynamics(resources.model, predictor, schedule)
        checkpoint_metadata["vlwm"] = {
            "path": str((args.checkpoint_dir / "vlwm.pt").resolve()),
            "best_epoch": payload["best_epoch"],
            "schedule": list(schedule),
        }

    base_cost = swm.planning.ShootingCostEvaluator(model_for_cost, objective)
    counted_cost = CountingCost(base_cost, counters)
    low_solver = swm.planning.CEMSolver(
        cost=counted_cost,
        batch_size=int(cem_cfg["batch_size"]),
        num_samples=int(cem_cfg["num_samples"]),
        var_scale=float(cem_cfg["variance_scale"]),
        n_steps=int(cem_cfg["iterations"]),
        topk=int(cem_cfg["topk"]),
        device=resources.device,
        seed=int(exp["selection_seed"]),
    )
    if args.method in {"hwm", "hilewm_c"}:
        macro, payload = _load_macro(args.checkpoint_dir / "macro.pt", resources.device)
        macro_cfg = comparison_config["macro"]
        # The low-level MPC is directed to a pre-encoded macro subgoal.
        low_cost = swm.planning.ShootingCostEvaluator(resources.model, objective, encode_goal=None)
        low_solver = swm.planning.CEMSolver(
            cost=CountingCost(low_cost, counters),
            batch_size=int(cem_cfg["batch_size"]),
            num_samples=int(cem_cfg["num_samples"]),
            var_scale=float(cem_cfg["variance_scale"]),
            n_steps=int(cem_cfg["iterations"]),
            topk=int(cem_cfg["topk"]),
            device=resources.device,
            seed=int(exp["selection_seed"]),
        )
        bank = None
        if args.method == "hilewm_c":
            bank = torch.as_tensor(
                np.load(args.checkpoint_dir / "empirical_macro_bank.npy"),
                dtype=next(macro.parameters()).dtype,
                device=resources.device,
            )
        base_solver: Any = HierarchicalMacroSolver(
            base_model=resources.model,
            macro_predictor=macro,
            low_solver=low_solver,
            objective=objective,
            macro_dim=int(macro_cfg["macro_dim"]),
            duration=int(macro_cfg["fixed_duration"]),
            high_horizon=int(macro_cfg["high_level_horizon"]),
            num_samples=int(macro_cfg["cem_samples"]),
            iterations=int(macro_cfg["cem_iterations"]),
            elites=int(macro_cfg["cem_elites"]),
            device=resources.device,
            seed=int(exp["selection_seed"]),
            mode=args.method,
            empirical_bank=bank,
            empirical_residual_scale=float(macro_cfg["empirical_residual_scale"]),
            planning_counters=counters,
        )
        planner_cfg["receding_horizon"] = int(macro_cfg["fixed_duration"]) // int(
            planner_cfg["action_block"]
        )
        checkpoint_metadata["macro"] = {
            "path": str((args.checkpoint_dir / "macro.pt").resolve()),
            "best_epoch": payload["best_epoch"],
            "empirical_bank": (
                str((args.checkpoint_dir / "empirical_macro_bank.npy").resolve())
                if bank is not None
                else None
            ),
        }
    else:
        base_solver = low_solver
    solver = TimedSolver(base_solver, counters)
    plan_config = swm.PlanConfig(
        horizon=int(planner_cfg["horizon"]),
        receding_horizon=int(planner_cfg["receding_horizon"]),
        history_len=int(planner_cfg["history_len"]),
        action_block=int(planner_cfg["action_block"]),
        warm_start=bool(planner_cfg["warm_start"]),
    )
    policy = TrackingPolicy(
        swm.policy.WorldModelPolicy(
            solver=solver,
            config=plan_config,
            process=resources.process,
            transform={"pixels": resources.image_transform, "goal": resources.image_transform},
        ),
        int(exp["num_eval"]),
        counters,
    )
    world = swm.World(
        env_name=str(exp["environment"]),
        num_envs=int(exp["num_eval"]),
        max_episode_steps=2 * int(exp["eval_budget"]),
        image_shape=(int(exp["image_size"]), int(exp["image_size"])),
    )
    world.set_policy(policy)
    if resources.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resources.device)

    started_at = time.time()
    wall_start = time.perf_counter()
    try:
        metrics = world.evaluate(
            dataset=resources.dataset,
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
            video=None,
        )
        wall_seconds = time.perf_counter() - wall_start
        agents, goals, distances = _final_positions(world)
    finally:
        world.close()

    successes = np.asarray(metrics["episode_successes"], dtype=bool)
    interval = binomtest(int(successes.sum()), len(successes)).proportion_ci(
        confidence_level=0.95, method="exact"
    )
    with (output_dir / "episodes.jsonl").open("w") as handle:
        for index, pair in enumerate(pairs):
            handle.write(
                json.dumps(
                    {
                        **asdict(pair),
                        "selection_seed": int(exp["selection_seed"]),
                        "success": bool(successes[index]),
                        "steps": int(policy.steps[index]),
                        "final_state_error": distances[index],
                        "final_agent_position": agents[index],
                        "goal_position": goals[index],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    summary = {
        "status": "complete",
        "method": args.method,
        "method_label": METHOD_LABELS[args.method],
        "reproduction_level": "paper_spec_tworoom_adaptation",
        "official_implementation": False,
        "data_split": split_name or "final_test_pairs",
        "experiment": exp,
        "planner": planner_cfg,
        "cem": cem_cfg,
        "runtime": {
            "requested_device": resources.requested_device,
            "resolved_device": str(resources.device),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "stable_worldmodel": _package_version("stable-worldmodel"),
        },
        "assets": {
            "dataset": str(resources.data_path),
            "weights": str(resources.weights_path),
            "weights_sha256": resources.weights_sha256,
            "checkpoints": checkpoint_metadata,
        },
        "results": {
            "num_tasks": len(successes),
            "successes": int(successes.sum()),
            "success_rate": float(successes.mean()),
            "success_rate_percent": 100.0 * float(successes.mean()),
            "success_rate_exact_95_ci": [float(interval.low), float(interval.high)],
            "mean_final_state_error": float(np.mean(distances)),
            "median_final_state_error": float(np.median(distances)),
            "mean_steps": float(np.mean(policy.steps)),
            "wall_time_seconds": wall_seconds,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(resources.device))
                if resources.device.type == "cuda"
                else 0
            ),
        },
        "planning": asdict(counters),
        "started_at_unix": started_at,
        "finished_at_unix": time.time(),
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "resolved_config.json",
        {
            "experiment": experiment_config,
            "comparison": comparison_config,
            "runtime": runtime_config,
            "resolved_experiment": exp,
            "resolved_planner": planner_cfg,
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Two-Room comparison baseline")
    parser.add_argument("--method", required=True, choices=tuple(METHOD_LABELS))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--goal-offset", type=int, required=True)
    parser.add_argument("--selection-seed", type=int, required=True)
    parser.add_argument("--num-eval", type=int, default=100)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--experiment-config", type=Path, default=Path("configs/lewm_tworooms_baseline.yaml")
    )
    parser.add_argument(
        "--comparison-config", type=Path, default=Path("configs/tworoom_baselines.yaml")
    )
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/runtime.yaml"))
    parser.add_argument(
        "--reference-matrix",
        type=Path,
        default=Path("artifacts/results/lewm_tworooms_long_matrix"),
    )
    parser.add_argument("--device")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("artifacts/cache/tworoom_baselines/split_manifest.json"),
    )
    parser.add_argument("--split-name", choices=("train", "validation", "calibration"))
    parser.add_argument("--skip-reference-check", action="store_true")
    return parser


def main() -> None:
    run_comparison(build_parser().parse_args())


if __name__ == "__main__":
    main()
