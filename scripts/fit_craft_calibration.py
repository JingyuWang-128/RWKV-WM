#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cape_wm.reachability import ReachabilityAdvantageCalibrator


def load_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        records.extend(
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        )
    return records


def episode_max_scores(
    records: list[dict], scale_mode: str = "constant"
) -> dict[int, list[float]]:
    """One simultaneous score per task and duration, never per video frame."""

    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for record in records:
        scale = (
            float(record["predicted_scale"]) if scale_mode == "predicted" else 1.0
        )
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("progress records require positive finite scales")
        score = (
            float(record["predicted_progress"]) - float(record["observed_progress"])
        ) / scale
        grouped[(str(record["pair_id"]), int(record["duration"]))].append(score)
    by_duration: dict[int, list[float]] = defaultdict(list)
    for (_, duration), scores in grouped.items():
        by_duration[duration].append(max(scores))
    return dict(by_duration)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit task-level conformal CRAFT reachability advantage bounds"
    )
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--scale-mode", choices=("constant", "predicted"), default="constant"
    )
    parser.add_argument("--durations", type=int, nargs="+", default=[5, 10, 20, 40])
    parser.add_argument("--minimum-tasks-per-duration", type=int, default=64)
    args = parser.parse_args()

    records = load_records(args.records)
    by_duration = episode_max_scores(records, args.scale_mode)
    missing = set(args.durations) - set(by_duration)
    if missing:
        raise ValueError(f"missing calibration records for durations {sorted(missing)}")
    for duration in args.durations:
        count = len(by_duration[duration])
        if count < args.minimum_tasks_per_duration:
            raise ValueError(
                f"duration {duration} has {count} task-level scores; "
                f"requires {args.minimum_tasks_per_duration}"
            )

    normalized_scores = np.concatenate(
        [np.asarray(by_duration[duration], dtype=np.float64) for duration in args.durations]
    )
    durations = np.concatenate(
        [
            np.full(len(by_duration[duration]), duration, dtype=np.int64)
            for duration in args.durations
        ]
    )
    calibrator = ReachabilityAdvantageCalibrator(
        alpha=args.alpha, scale_mode=args.scale_mode
    ).fit(
        predicted_progress=normalized_scores,
        observed_progress=np.zeros_like(normalized_scores),
        durations=durations,
        predicted_scale=np.ones_like(normalized_scores),
    )
    calibrator.save(args.output)
    summary = {
        "status": "complete",
        "method": calibrator.method,
        "alpha": calibrator.alpha,
        "scale_mode": calibrator.scale_mode,
        "input_records": len(records),
        "task_level_scores": {
            str(duration): len(by_duration[duration]) for duration in args.durations
        },
        "quantiles": {
            str(duration): calibrator.quantile(duration) for duration in args.durations
        },
        "empirical_simultaneous_coverage": {
            str(duration): float(
                np.mean(
                    np.asarray(by_duration[duration]) <= calibrator.quantile(duration)
                )
            )
            for duration in args.durations
        },
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
