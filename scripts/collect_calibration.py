#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from cape_wm.collection import collect_closed_loop_attempt
from cape_wm.data import save_calibration_records


def _factory(specification: str):
    module_name, separator, attribute = specification.partition(":")
    if not separator:
        raise ValueError("factory must use the form package.module:function")
    return getattr(importlib.import_module(module_name), attribute)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect independent closed-loop executability/calibration records"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--factory",
        required=True,
        help=(
            "zero-argument factory returning "
            "(environment, world_model, low_level_controller, attempt_specs)"
        ),
    )
    parser.add_argument("--success-threshold", type=float, required=True)
    args = parser.parse_args()

    components = _factory(args.factory)()
    if len(components) == 4:
        environment, model, controller, attempts = components
        endpoint_error = None
        success_evaluator = None
    elif len(components) == 5:
        environment, model, controller, attempts, endpoint_error = components
        success_evaluator = None
    elif len(components) == 6:
        (
            environment,
            model,
            controller,
            attempts,
            endpoint_error,
            success_evaluator,
        ) = components
    else:
        raise ValueError("factory must return four, five, or six components")
    records = [
        collect_closed_loop_attempt(
            environment,
            model,
            controller,
            attempt,
            args.success_threshold,
            endpoint_error=endpoint_error,
            success_evaluator=success_evaluator,
        )
        for attempt in attempts
    ]
    if not records:
        raise SystemExit("factory returned no calibration attempts")
    save_calibration_records(args.output, records)


if __name__ == "__main__":
    main()
