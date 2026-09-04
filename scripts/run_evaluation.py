#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from cape_wm.evaluation import evaluate_episode, write_jsonl


def _factory(specification: str):
    module_name, separator, attribute = specification.partition(":")
    if not separator:
        raise ValueError("factory must use the form package.module:function")
    return getattr(importlib.import_module(module_name), attribute)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a paired CAPE-WM evaluation list")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--factory",
        required=True,
        help=("zero-argument factory returning (environment, planner, episode_specs, max_steps)"),
    )
    args = parser.parse_args()
    environment, planner, episode_specs, max_steps = _factory(args.factory)()
    results = [evaluate_episode(environment, planner, spec, max_steps) for spec in episode_specs]
    if not results:
        raise SystemExit("factory returned no episode specifications")
    write_jsonl(args.output, results)


if __name__ == "__main__":
    main()
