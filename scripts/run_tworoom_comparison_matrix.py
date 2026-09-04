#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cape_wm.baseline import prepare_resources
from cape_wm.comparison_eval import run_comparison
from cape_wm.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paired Two-Room comparison matrix")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("artifacts/checkpoints/tworoom_baselines"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/results/tworoom_comparison_matrix"),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("flat_trm", "hwm", "vlwm", "hilewm_c"),
        default=("flat_trm", "hwm", "vlwm", "hilewm_c"),
    )
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
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--offsets", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--num-eval", type=int)
    parser.add_argument("--eval-budget", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--skip-reference-check", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid shard index")

    protocol = load_config(args.comparison_config)["protocol"]
    offsets = tuple(args.offsets or map(int, protocol["offsets"]))
    seeds = tuple(args.seeds or map(int, protocol["evaluation_seeds"]))
    num_eval = int(args.num_eval or protocol["tasks_per_condition"])
    eval_budget = int(args.eval_budget or protocol["evaluation_budget"])
    conditions = [
        (method, offset, seed) for method in args.methods for offset in offsets for seed in seeds
    ][args.shard_index :: args.shard_count]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    resources = prepare_resources(
        data=args.data,
        weights=args.weights,
        experiment_config=args.experiment_config,
        runtime_config=args.runtime_config,
        cache_dir=output_root / f"cache_shard_{args.shard_index}",
        device_override=args.device,
    )
    runs = []
    for method, offset, seed in conditions:
        output = output_root / method / f"offset_{offset}" / f"seed_{seed}"
        summary_path = output / "summary.json"
        if summary_path.is_file() and not args.force:
            status = json.loads(summary_path.read_text()).get("status")
            if status == "complete":
                print(
                    json.dumps(
                        {
                            "event": "skip_complete",
                            "method": method,
                            "offset": offset,
                            "seed": seed,
                        }
                    ),
                    flush=True,
                )
                runs.append(str(summary_path))
                continue
        condition = argparse.Namespace(
            method=method,
            data=args.data,
            weights=args.weights,
            checkpoint_dir=args.checkpoint_dir,
            goal_offset=offset,
            selection_seed=seed,
            num_eval=num_eval,
            eval_budget=eval_budget,
            output_dir=output,
            experiment_config=args.experiment_config,
            comparison_config=args.comparison_config,
            runtime_config=args.runtime_config,
            reference_matrix=(None if args.skip_reference_check else args.reference_matrix),
            device=args.device,
        )
        run_comparison(condition, resources=resources)
        runs.append(str(summary_path))
    manifest = {
        "status": "complete",
        "methods": list(args.methods),
        "runs": runs,
        "reproduction_level": "paper_spec_tworoom_adaptation",
    }
    manifest["shard_index"] = args.shard_index
    manifest["shard_count"] = args.shard_count
    (output_root / f"run_manifest_shard_{args.shard_index}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
