from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate M5 B6 effect-weight candidates")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--task", choices=("tworoom_w", "action_delay"), required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--weights", default="0.5,2.0")
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--evaluation-interval", type=int, default=500)
    parser.add_argument("--profile", choices=("smoke", "main"), default="main")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    return parser.parse_args()


def _tag(weight: float) -> str:
    return str(weight).replace(".", "p")


def main() -> None:
    args = parse_args()
    seeds = tuple(int(item) for item in args.seeds.split(","))
    weights = tuple(float(item) for item in args.weights.split(","))
    curriculum = tuple(int(item) for item in args.curriculum.split(","))
    if not curriculum:
        raise ValueError("curriculum cannot be empty")
    final_model_horizon = curriculum[-1]
    for seed in seeds:
        b2_latest = args.root / f"seed_{seed}" / "train" / "b2" / "latest.pt"
        if not b2_latest.is_file():
            raise FileNotFoundError(b2_latest)
        b2_latest_payload = torch.load(b2_latest, map_location="cpu", weights_only=True)
        stored_levels = tuple(
            int(level)
            for level in b2_latest_payload.get("training_config", {})
            .get("trainer", {})
            .get("curriculum_levels", [])
        )
        final_level_index = int(b2_latest_payload.get("curriculum", {}).get("level_index", -1))
        if (
            b2_latest_payload.get("method") != "b2"
            or int(b2_latest_payload.get("global_step", -1)) != args.max_steps
            or stored_levels != curriculum
            or final_level_index != len(curriculum) - 1
        ):
            raise RuntimeError(
                f"B2 seed {seed} is not frozen at the required {args.max_steps} steps "
                "and final curriculum horizon"
            )
        for weight in weights:
            tag = _tag(weight)
            train_root = args.root / f"seed_{seed}" / f"train_b6_w{tag}"
            b2_checkpoint = (
                args.root
                / f"seed_{seed}"
                / "train"
                / "b2"
                / f"best_rollout_h{final_model_horizon}.pt"
            )
            command = [
                sys.executable,
                "scripts/train_cc_rwkv.py",
                "--config",
                str(args.config),
                "--data",
                str(args.data),
                "--data-manifest",
                str(args.data_manifest),
                "--output",
                str(train_root),
                "--method",
                "b6",
                "--max-steps",
                str(args.max_steps),
                "--batch-size",
                str(args.batch_size),
                "--profile",
                args.profile,
                "--device",
                args.device,
                "--seed",
                str(seed),
                "--evaluation-interval",
                str(args.evaluation_interval),
                "--effect-weight",
                str(weight),
                "--b2-checkpoint",
                str(b2_checkpoint),
                "--curriculum",
                args.curriculum,
            ]
            if args.train_limit is not None:
                command.extend(["--train-limit", str(args.train_limit)])
            if args.validation_limit is not None:
                command.extend(["--validation-limit", str(args.validation_limit)])
            if args.resume_existing:
                command.append("--resume-existing")
            if not args.evaluate_only:
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            checkpoint = train_root / "b6" / f"best_rollout_h{final_model_horizon}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            for split in ("validation", "test"):
                evaluation = args.root / f"seed_{seed}" / split / f"b6_w{tag}"
                evaluate = [
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
                    "pass",
                    "--output",
                    str(evaluation),
                ]
                if args.train_limit is not None:
                    evaluate.extend(["--train-limit", str(args.train_limit)])
                if split == "validation" and args.validation_limit is not None:
                    evaluate.extend(["--evaluation-limit", str(args.validation_limit)])
                subprocess.run(evaluate, check=True, stdout=subprocess.DEVNULL)
            print(f"selected-candidate input ready: {args.task} seed={seed} weight={weight}")


if __name__ == "__main__":
    main()
