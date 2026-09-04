#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from cape_wm.baseline import _write_json, prepare_resources
from cape_wm.config import load_config
from cape_wm.tworoom_risk_collection import collect_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect leak-free Two-Room CAPE risk data")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("artifacts/cache/tworoom_baselines/split_manifest.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/cache/tworoom_cape_risk")
    )
    parser.add_argument(
        "--experiment-config", type=Path, default=Path("configs/lewm_tworooms_baseline.yaml")
    )
    parser.add_argument("--phase-config", type=Path, default=Path("configs/cape_phase_b.yaml"))
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/runtime.yaml"))
    parser.add_argument("--train-per-duration", type=int, default=128)
    parser.add_argument("--calibration-per-duration", type=int, default=64)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=3072)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    experiment_config = load_config(args.experiment_config)
    experiment = dict(experiment_config["experiment"])
    baseline_planner = dict(experiment_config["planner"])
    cem = dict(experiment_config["cem"])
    phase = load_config(args.phase_config)
    durations = tuple(map(int, phase["planner"]["durations"]))
    runtime = load_config(args.runtime_config)["runtime"]
    torch.set_num_threads(int(runtime.get("torch_num_threads", torch.get_num_threads())))
    if "torch_num_interop_threads" in runtime:
        interop = int(runtime["torch_num_interop_threads"])
        if torch.get_num_interop_threads() != interop:
            torch.set_num_interop_threads(interop)
    split = json.loads(args.split_manifest.read_text())
    train = set(map(int, split["train"]))
    calibration = set(map(int, split["calibration"]))
    test = set(map(int, split["test"]))
    if train & calibration or train & test or calibration & test:
        raise SystemExit("split manifest contains trajectory leakage")
    resources = prepare_resources(
        data=args.data,
        weights=args.weights,
        experiment_config=args.experiment_config,
        runtime_config=args.runtime_config,
        cache_dir=output / "stablewm_cache",
        device_override=args.device,
    )
    started = time.time()
    summaries = {
        "train": collect_split(
            resources,
            "train",
            sorted(train),
            durations=durations,
            attempts_per_duration=args.train_per_duration,
            seed=args.seed,
            output=output,
            experiment=experiment,
            planner=baseline_planner,
            cem=cem,
        ),
        "calibration": collect_split(
            resources,
            "calibration",
            sorted(calibration),
            durations=durations,
            attempts_per_duration=args.calibration_per_duration,
            seed=args.seed + 100_000,
            output=output,
            experiment=experiment,
            planner=baseline_planner,
            cem=cem,
        ),
    }
    metadata = {
        "status": "complete",
        "data": str(resources.data_path),
        "weights": str(resources.weights_path),
        "split_manifest": str(args.split_manifest.resolve()),
        "split_content_hash": split["content_hash"],
        "durations": list(durations),
        "action_block": int(baseline_planner["action_block"]),
        "resolved_device": str(resources.device),
        "summaries": summaries,
        "started_at_unix": started,
        "finished_at_unix": time.time(),
    }
    _write_json(output / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
