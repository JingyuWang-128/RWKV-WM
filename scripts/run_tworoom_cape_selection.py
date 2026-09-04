#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from cape_wm.tworoom_cape_eval import build_parser, run_tworoom_cape


GRID = (0.0, 0.01, 0.05, 0.1)


def _tag(value: float) -> str:
    return str(value).replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one shard of Two-Room CAPE selection")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/results/tworoom_cape_selection")
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--num-eval", type=int, default=20)
    parser.add_argument("--selection-seed", type=int, default=3072)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid shard index")
    conditions = list(itertools.product(GRID, GRID))
    selected = conditions[args.shard_index :: args.shard_count]
    base_parser = build_parser()
    completed = []
    for lambda_compute, lambda_risk in selected:
        condition = f"lc_{_tag(lambda_compute)}_lr_{_tag(lambda_risk)}"
        for offset in (25, 100):
            output_dir = args.output / condition / f"offset_{offset}"
            summary_path = output_dir / "summary.json"
            if summary_path.is_file():
                completed.append(json.loads(summary_path.read_text()))
                continue
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
                    str(args.selection_seed),
                    "--goal-offset",
                    str(offset),
                    "--num-eval",
                    str(args.num_eval),
                    "--lambda-compute",
                    str(lambda_compute),
                    "--lambda-risk",
                    str(lambda_risk),
                    "--split-name",
                    "validation",
                    "--skip-reference-check",
                ]
            )
            completed.append(run_tworoom_cape(eval_args))
    print(
        json.dumps(
            {
                "status": "complete",
                "device": args.device,
                "shard_index": args.shard_index,
                "conditions": len(selected),
                "runs": len(completed),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
