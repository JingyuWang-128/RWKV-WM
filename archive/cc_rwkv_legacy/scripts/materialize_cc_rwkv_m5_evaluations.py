from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an M5 seed/method matrix")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--task", choices=("tworoom_w", "action_delay"), required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--methods", default="b2,b3,b4,b6")
    parser.add_argument("--splits", default="validation,test")
    parser.add_argument(
        "--train-limit",
        type=int,
        help="Optional training-split cap; omit it for the complete formal split.",
    )
    parser.add_argument("--evaluation-limit", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("matrix lists cannot be empty")
    return result


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.data_manifest.read_text())
    if manifest["variant"] != args.task:
        raise ValueError("task does not match the branch dataset manifest")
    final_model_horizon = int(manifest["model_horizon"])
    seeds = tuple(int(item) for item in _csv(args.seeds))
    methods = _csv(args.methods)
    splits = _csv(args.splits)
    completed = []
    for seed in seeds:
        fairness_path = args.root / f"seed_{seed}" / "train" / "fairness.json"
        fairness = json.loads(fairness_path.read_text())["status"]
        for method in methods:
            checkpoint = (
                args.root
                / f"seed_{seed}"
                / "train"
                / method
                / f"best_rollout_h{final_model_horizon}.pt"
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            for split in splits:
                output = args.root / f"seed_{seed}" / split / f"{method}_w1"
                command = [
                    sys.executable,
                    "scripts/evaluate_cc_rwkv_m5.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--data",
                    str(args.data),
                    "--data-manifest",
                    str(args.data_manifest),
                    "--task",
                    args.task,
                    "--split",
                    split,
                    "--batch-size",
                    str(args.batch_size),
                    "--device",
                    args.device,
                    "--fairness",
                    fairness,
                    "--output",
                    str(output),
                ]
                if args.train_limit is not None:
                    command.extend(["--train-limit", str(args.train_limit)])
                if args.evaluation_limit is not None:
                    command.extend(["--evaluation-limit", str(args.evaluation_limit)])
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                output.mkdir(parents=True, exist_ok=True)
                (output / "command_stdout.json").write_text(result.stdout)
                completed.append(str(output))
                print(f"evaluated {args.task} seed={seed} method={method} split={split}")
    (args.root / "evaluation_inventory.json").write_text(
        json.dumps({"completed": completed}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
