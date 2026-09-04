#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cape_wm.data import make_paired_test_list, save_goal_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a paired final-test task list")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("trajectory_lengths", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--offsets", type=int, nargs="+", default=[25, 50, 75, 100])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--sampling-seed", type=int, default=0)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    lengths = {
        str(key): int(value)
        for key, value in json.loads(args.trajectory_lengths.read_text()).items()
    }
    pairs = make_paired_test_list(
        args.environment,
        lengths,
        manifest["test"],
        tuple(args.offsets),
        tuple(args.seeds),
        args.episodes,
        args.sampling_seed,
    )
    save_goal_pairs(args.output, pairs)
    print(f"wrote {len(pairs)} paired tasks to {args.output}")


if __name__ == "__main__":
    main()
