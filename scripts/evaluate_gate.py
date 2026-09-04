#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cape_wm.evaluation import read_jsonl
from cape_wm.stats import evaluate_phase_gate


def _paired(cape_path: Path, baseline_path: Path):
    cape_results = read_jsonl(cape_path)
    baseline_results = read_jsonl(baseline_path)
    cape = {result.pair_id: result for result in cape_results}
    baseline = {result.pair_id: result for result in baseline_results}
    if len(cape) != len(cape_results) or len(baseline) != len(baseline_results):
        raise SystemExit("duplicate pair IDs would invalidate paired statistics")
    shared = sorted(cape.keys() & baseline.keys())
    if not shared:
        raise SystemExit("no paired episode IDs are shared by the two result files")
    if len(shared) != len(cape) or len(shared) != len(baseline):
        raise SystemExit("result files must contain exactly the same paired task IDs")
    return [cape[key] for key in shared], [baseline[key] for key in shared]


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the preregistered CAPE-WM Phase-B gate")
    parser.add_argument("cape_long", type=Path)
    parser.add_argument("baseline_long", type=Path)
    parser.add_argument("cape_short", type=Path)
    parser.add_argument("baseline_short", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cape_long, baseline_long = _paired(args.cape_long, args.baseline_long)
    cape_short, baseline_short = _paired(args.cape_short, args.baseline_short)
    result = evaluate_phase_gate(
        np.asarray([item.success for item in cape_long]),
        np.asarray([item.success for item in baseline_long]),
        np.asarray([item.success for item in cape_short]),
        np.asarray([item.success for item in baseline_short]),
        cape_time=sum(item.planning_time_seconds for item in cape_long),
        baseline_time=sum(item.planning_time_seconds for item in baseline_long),
        seed=args.seed,
    )
    print(json.dumps(asdict(result), indent=2))
    raise SystemExit(0 if result.passed else 2)


if __name__ == "__main__":
    main()
