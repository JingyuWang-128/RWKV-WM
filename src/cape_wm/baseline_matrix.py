from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.stats import binomtest

from .baseline import prepare_resources, run_baseline


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _interval(successes: int, tasks: int) -> list[float]:
    result = binomtest(successes, tasks).proportion_ci(confidence_level=0.95, method="exact")
    return [float(result.low), float(result.high)]


def summarize_matrix(
    condition_summaries: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    offsets: list[int],
    seeds: list[int],
) -> dict[str, Any]:
    offset_results: dict[str, Any] = {}
    pooled_rates: list[float] = []
    for offset in offsets:
        selected = [row for row in episodes if row["goal_offset"] == offset]
        successes = int(sum(row["success"] for row in selected))
        seed_rates = [
            float(np.mean([row["success"] for row in selected if row["evaluation_seed"] == seed]))
            for seed in seeds
        ]
        pooled_rate = successes / len(selected)
        pooled_rates.append(pooled_rate)
        offset_results[str(offset)] = {
            "tasks": len(selected),
            "successes": successes,
            "pooled_success_rate": pooled_rate,
            "pooled_success_rate_exact_95_ci": _interval(successes, len(selected)),
            "seed_success_rates": dict(zip(map(str, seeds), seed_rates, strict=True)),
            "seed_success_rate_mean": float(np.mean(seed_rates)),
            "seed_success_rate_std": (
                float(np.std(seed_rates, ddof=1)) if len(seed_rates) > 1 else 0.0
            ),
            "mean_final_state_error": float(
                np.mean([row["final_state_error"] for row in selected])
            ),
            "median_final_state_error": float(
                np.median([row["final_state_error"] for row in selected])
            ),
            "mean_steps": float(np.mean([row["steps"] for row in selected])),
        }

    long_offsets = offsets[-2:]
    long_rows = [row for row in episodes if row["goal_offset"] in long_offsets]
    long_successes = int(sum(row["success"] for row in long_rows))
    auc = (
        float(np.trapezoid(pooled_rates, offsets) / (offsets[-1] - offsets[0]))
        if len(offsets) > 1
        else pooled_rates[0]
    )
    total_wall = float(sum(item["results"]["wall_time_seconds"] for item in condition_summaries))
    return {
        "status": "complete",
        "method": "LeWM + Flat MPC (official latent GoalMSE)",
        "offsets": offsets,
        "evaluation_seeds": seeds,
        "conditions": len(offsets) * len(seeds),
        "tasks_per_condition": len(episodes) // (len(offsets) * len(seeds)),
        "total_tasks": len(episodes),
        "offset_results": offset_results,
        "long_horizon": {
            "offsets": long_offsets,
            "tasks": len(long_rows),
            "successes": long_successes,
            "pooled_success_rate": long_successes / len(long_rows),
            "pooled_success_rate_exact_95_ci": _interval(long_successes, len(long_rows)),
            "mean_final_state_error": float(
                np.mean([row["final_state_error"] for row in long_rows])
            ),
        },
        "offset_success_auc": auc,
        "success_drop_25_to_100": pooled_rates[0] - pooled_rates[-1],
        "compute": {
            "wall_time_seconds": total_wall,
            "solver_seconds": float(
                sum(item["planning"]["solver_seconds"] for item in condition_summaries)
            ),
            "cost_calls": int(sum(item["planning"]["cost_calls"] for item in condition_summaries)),
            "candidate_sequences": int(
                sum(item["planning"]["candidate_sequences"] for item in condition_summaries)
            ),
            "predicted_transitions": int(
                sum(item["planning"]["predicted_transitions"] for item in condition_summaries)
            ),
            "peak_cuda_memory_bytes": int(
                max(item["results"]["peak_cuda_memory_bytes"] for item in condition_summaries)
            ),
        },
        "protocol_note": (
            "Seeds vary pair selection and CEM sampling for one official pretrained "
            "checkpoint; they are not independent backbone-training seeds."
        ),
    }


def _is_complete(path: Path, seed: int, offset: int, num_eval: int) -> bool:
    summary_path = path / "summary.json"
    episodes_path = path / "episodes.jsonl"
    if not summary_path.is_file() or not episodes_path.is_file():
        return False
    summary = json.loads(summary_path.read_text())
    lines = [line for line in episodes_path.read_text().splitlines() if line]
    return bool(
        summary.get("status") == "complete"
        and summary["experiment"]["selection_seed"] == seed
        and summary["experiment"]["goal_offset"] == offset
        and summary["results"]["num_tasks"] == num_eval
        and len(lines) == num_eval
    )


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    summaries: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    resources = prepare_resources(
        data=args.data,
        weights=args.weights,
        experiment_config=args.experiment_config,
        runtime_config=args.runtime_config,
        cache_dir=root / "shared_cache",
        device_override=args.device,
    )

    for offset in args.offsets:
        for seed in args.seeds:
            condition_dir = root / f"offset_{offset}" / f"seed_{seed}"
            if args.resume and _is_complete(condition_dir, seed, offset, args.num_eval):
                print(f"[matrix] reuse complete offset={offset} seed={seed}", flush=True)
            else:
                print(f"[matrix] run offset={offset} seed={seed}", flush=True)
                namespace = SimpleNamespace(
                    data=args.data,
                    weights=args.weights,
                    experiment_config=args.experiment_config,
                    runtime_config=args.runtime_config,
                    output_dir=condition_dir,
                    device=args.device,
                    num_eval=args.num_eval,
                    selection_seed=seed,
                    goal_offset=offset,
                    eval_budget=args.eval_budget,
                )
                run_baseline(namespace, resources=resources)

            summary = json.loads((condition_dir / "summary.json").read_text())
            summaries.append(summary)
            for line in (condition_dir / "episodes.jsonl").read_text().splitlines():
                if not line:
                    continue
                record = json.loads(line)
                record["evaluation_seed"] = seed
                record["matrix_pair_id"] = f"seed{seed}:{record['pair_id']}"
                episodes.append(record)

            partial = {
                "status": "running",
                "completed_conditions": len(summaries),
                "total_conditions": len(args.offsets) * len(args.seeds),
                "last_offset": offset,
                "last_seed": seed,
                "updated_at_unix": time.time(),
            }
            _write_json(root / "progress.json", partial)

    matrix_summary = summarize_matrix(summaries, episodes, args.offsets, args.seeds)
    matrix_summary["started_at_unix"] = started
    matrix_summary["finished_at_unix"] = time.time()
    matrix_summary["eval_budget"] = args.eval_budget
    matrix_summary["resolved_devices"] = sorted(
        {item["runtime"]["resolved_device"] for item in summaries}
    )
    _write_json(root / "matrix_summary.json", matrix_summary)
    with (root / "matrix_episodes.jsonl").open("w") as handle:
        for record in episodes:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    _write_json(root / "condition_summaries.json", summaries)
    _write_json(
        root / "progress.json",
        {
            "status": "complete",
            "completed_conditions": len(summaries),
            "total_conditions": len(summaries),
            "updated_at_unix": time.time(),
        },
    )
    print(json.dumps(matrix_summary, indent=2, ensure_ascii=False), flush=True)
    return matrix_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full LeWM Two-Room baseline matrix")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/lewm_tworooms_baseline.yaml"),
    )
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/runtime.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/results/lewm_tworooms_long_matrix"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--offsets", type=int, nargs="+", default=[25, 50, 75, 100])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--num-eval", type=int, default=100)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.set_defaults(resume=True)
    run_matrix(parser.parse_args())


if __name__ == "__main__":
    main()
