#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from cape_wm.data import deterministic_group_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a leakage-free trajectory split manifest")
    parser.add_argument("trajectory_ids", type=Path, help="one trajectory ID per line")
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    identifiers = [
        line.strip() for line in args.trajectory_ids.read_text().splitlines() if line.strip()
    ]
    manifest = deterministic_group_split(identifiers, seed=args.seed)
    manifest.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
