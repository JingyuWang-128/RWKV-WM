#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import numpy as np

from cape_wm.data import load_npz_trajectories, save_npz_trajectories


def _load_factory(specification: str):
    module_name, separator, attribute = specification.partition(":")
    if not separator:
        raise ValueError("factory must use the form package.module:function")
    return getattr(importlib.import_module(module_name), attribute)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen world-model latents")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--adapter-factory",
        required=True,
        help="importable zero-argument factory, e.g. experiment.make_adapter:build",
    )
    args = parser.parse_args()

    adapter: Any = _load_factory(args.adapter_factory)()
    trajectories = load_npz_trajectories(args.input)
    for trajectory in trajectories:
        trajectory.latents = np.stack(
            [adapter.encode(observation) for observation in trajectory.observations]
        ).astype(np.float32)
    save_npz_trajectories(args.output, trajectories)


if __name__ == "__main__":
    main()
