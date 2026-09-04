#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cape_wm.baseline import prepare_resources
from cape_wm.tworoom_cape_eval import build_parser, run_tworoom_cape


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one shard of paired Two-Room CAPE final matrix"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/results/tworoom_cape_matrix_gpu_final")
    )
    parser.add_argument(
        "--selected",
        type=Path,
        default=Path("artifacts/checkpoints/tworoom_cape/selected_hyperparameters.json"),
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--offsets", type=int, nargs="+", default=[25, 50, 75, 100])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--num-eval", type=int, default=100)
    parser.add_argument(
        "--reference-matrix",
        type=Path,
        default=Path("artifacts/results/lewm_tworooms_long_matrix_gpu_final"),
    )
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid shard index")
    frozen = json.loads(args.selected.read_text())
    if frozen.get("status") not in {
        "frozen_before_final_test",
        "frozen_on_validation_after_primary",
        "frozen_before_third_confirmatory",
        "frozen_before_craft_confirmatory",
    }:
        raise SystemExit("hyperparameters are not frozen")
    params = frozen["selected"]
    conditions = [(offset, seed) for offset in args.offsets for seed in args.seeds]
    selected_conditions = conditions[args.shard_index :: args.shard_count]
    args.output.mkdir(parents=True, exist_ok=True)
    resources = prepare_resources(
        data=args.data,
        weights=args.weights,
        experiment_config=Path("configs/lewm_tworooms_baseline.yaml"),
        runtime_config=Path("configs/runtime.yaml"),
        cache_dir=args.output / f"shared_cache_gpu_{args.shard_index}",
        device_override=args.device,
    )
    base_parser = build_parser()
    completed = []
    for offset, seed in selected_conditions:
        output_dir = args.output / f"offset_{offset}" / f"seed_{seed}"
        summary_path = output_dir / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text())
        else:
            eval_args = base_parser.parse_args(
                [
                    "--data",
                    str(args.data),
                    "--weights",
                    str(args.weights),
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    args.device,
                    "--selection-seed",
                    str(seed),
                    "--goal-offset",
                    str(offset),
                    "--num-eval",
                    str(args.num_eval),
                    "--reference-matrix",
                    str(args.reference_matrix),
                    "--lambda-compute",
                    str(params["lambda_compute"]),
                    "--lambda-risk",
                    str(params["lambda_risk"]),
                    "--planner-mode",
                    str(params.get("planner_mode", "cape")),
                    *(
                        [
                            "--craft-checkpoint",
                            str(params["craft_checkpoint"]),
                            "--craft-calibration",
                            str(params["craft_calibration"]),
                            "--craft-config",
                            str(params["craft_config"]),
                        ]
                        if params.get("planner_mode") == "craft"
                        else []
                    ),
                    *(
                        [
                            "--topology-artifact",
                            str(params["topology_artifact"]),
                            "--topology-duration",
                            str(params.get("topology_duration", 5)),
                            "--topology-control-mode",
                            str(params.get("topology_control_mode", "direct")),
                            *(
                                ["--topology-direct-completion"]
                                if params.get("topology_direct_completion", False)
                                else []
                            ),
                        ]
                        if params.get("topology_artifact")
                        else []
                    ),
                ]
            )
            summary = run_tworoom_cape(eval_args, resources=resources)
        if summary.get("data_split") != "final_test_pairs":
            raise RuntimeError("final matrix condition did not use final test pairs")
        if summary["assets"].get("reference_pairs") is None:
            raise RuntimeError("final matrix condition skipped paired reference validation")
        if params.get("planner_mode") == "craft":
            forbidden = (
                "trm_checkpoint",
                "macro_checkpoint",
                "risk_checkpoint",
                "topology_artifact",
            )
            loaded = [name for name in forbidden if summary["assets"].get(name) is not None]
            if loaded or summary["planning"]["fallback_decisions"]:
                raise RuntimeError(
                    f"CRAFT confirmatory run used a forbidden branch: {loaded}"
                )
        completed.append(
            {
                "offset": offset,
                "seed": seed,
                "success_rate": summary["results"]["success_rate"],
                "wall_time_seconds": summary["results"]["wall_time_seconds"],
            }
        )
    manifest = {
        "status": "complete",
        "device": args.device,
        "shard_index": args.shard_index,
        "frozen_hyperparameters": params,
        "frozen_status": frozen["status"],
        "conditions": completed,
    }
    (args.output / f"shard_{args.shard_index}.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
