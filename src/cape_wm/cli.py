from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .adapters.toy import (
    PointImageEnv,
    PointImageWorldModel,
    PointMacroDynamics,
    PointRiskPredictor,
)
from .cem import CEMConfig
from .conformal import SplitConformalCalibrator
from .evaluation import EpisodeSpec, evaluate_episode, read_jsonl
from .generators import CEMLowLevelController, MacroCEMGenerator
from .planner import CAPEPlanner, PlannerConfig
from .risk import CalibratedRiskEstimator
from .stats import mechanism_metrics


def build_toy_planner(seed: int = 0) -> CAPEPlanner:
    model = PointImageWorldModel()
    low_level = CEMLowLevelController(
        model,
        horizon=10,
        cem=CEMConfig(samples=192, elites=24, iterations=4, seed=seed),
        path_cost_weight=0.5,
    )
    generator = MacroCEMGenerator(
        model,
        PointMacroDynamics(),
        low_level,
        physical_horizon=40,
        macro_cem=CEMConfig(samples=96, elites=12, iterations=4, seed=seed),
    )
    predictor = PointRiskPredictor()
    rng = np.random.default_rng(seed)
    predicted_miss = np.full(128, 0.01)
    observed_miss = predicted_miss + np.abs(rng.normal(0.0, 0.004, size=128))
    predicted_scales = [np.full(10, 0.014) for _ in range(128)]
    residuals = [np.abs(rng.normal(0.0, 0.006, size=10)) for _ in range(128)]
    calibrator = SplitConformalCalibrator(alpha=0.1)
    calibrator.fit(predicted_miss, observed_miss, predicted_scales, residuals)
    risk = CalibratedRiskEstimator(predictor, calibrator)
    return CAPEPlanner(
        model,
        generator,
        risk,
        PlannerConfig(
            durations=(5, 10, 20, 40),
            executability_threshold=0.09,
            goal_threshold=0.045,
            subgoal_threshold=0.06,
            lambda_compute=0.01,
            lambda_risk=0.05,
            min_progress=1e-3,
        ),
    )


def toy_main() -> None:
    parser = argparse.ArgumentParser(description="Run the CAPE-WM image-only toy smoke test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    environment = PointImageEnv()
    start = np.asarray((0.1, 0.15), dtype=np.float32)
    goal = np.asarray((0.85, 0.8), dtype=np.float32)
    result = evaluate_episode(
        environment,
        build_toy_planner(args.seed),
        EpisodeSpec(
            pair_id=f"toy-{args.seed}",
            seed=args.seed,
            goal_image=environment.render_state(goal),
            reset_options={"start": start, "goal": goal},
            offset=40,
        ),
        max_steps=args.max_steps,
    )
    payload = {
        "success": result.success,
        "steps": result.steps,
        "final_goal_cost": result.final_goal_cost,
        "events": result.planning_events,
        "durations": result.duration_histogram,
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print("CAPE-WM toy smoke test")
        print(json.dumps(payload, indent=2))


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CAPE-WM JSONL evaluation artifacts")
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    results = read_jsonl(args.results)
    if not results:
        raise SystemExit("result file is empty")
    summary = {
        "episodes": len(results),
        "success_rate": float(np.mean([result.success for result in results])),
        "mean_steps": float(np.mean([result.steps for result in results])),
        "mean_wall_time_seconds": float(np.mean([result.wall_time_seconds for result in results])),
        "mean_planning_time_seconds": float(
            np.mean([result.planning_time_seconds for result in results])
        ),
        "mean_model_calls": float(np.mean([result.model_calls for result in results])),
        "peak_cuda_memory_bytes": max(result.peak_cuda_memory_bytes for result in results),
        "mechanism": asdict(mechanism_metrics([result.planning_trace for result in results])),
    }
    print(json.dumps(summary, indent=2))
