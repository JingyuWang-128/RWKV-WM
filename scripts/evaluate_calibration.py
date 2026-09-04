#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from cape_wm.calibration import load_calibrator
from cape_wm.data import load_calibration_records
from cape_wm.stats import (
    binary_auroc,
    brier_score,
    endpoint_coverage,
    expected_calibration_error,
    sequence_coverage,
)


def _factory(specification: str):
    module_name, separator, attribute = specification.partition(":")
    if not separator:
        raise ValueError("factory must use the form package.module:function")
    return getattr(importlib.import_module(module_name), attribute)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate held-out CAPE-WM calibration")
    parser.add_argument("records", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--predictor-factory", required=True)
    args = parser.parse_args()

    records = load_calibration_records(args.records)
    if not records:
        raise SystemExit("evaluation record set is empty")
    predictor = _factory(args.predictor_factory)()
    calibrator = load_calibrator(args.artifact)
    misses = []
    probabilities = []
    upper_bounds = []
    residuals = []
    tubes = []
    for record in records:
        miss, probability, scales = predictor(
            record.current_latent, record.subgoal_latent, record.duration
        )
        scale_sequence = np.asarray(scales, dtype=np.float64).reshape(-1)
        if scale_sequence.size == 1:
            scale_sequence = np.full(len(record.step_residuals), scale_sequence.item())
        scale_sequence = scale_sequence[: len(record.step_residuals)]
        if len(scale_sequence) != len(record.step_residuals):
            raise ValueError("predictor returned fewer scales than observed residuals")
        misses.append(record.observed_miss)
        probabilities.append(probability)
        upper_bounds.append(calibrator.miss_upper_bound(miss))
        residuals.append(record.step_residuals)
        tubes.append(np.asarray(calibrator.simultaneous_tube(scale_sequence)))
    targets = np.asarray([record.success for record in records])
    probabilities_array = np.asarray(probabilities)
    report = {
        "records": len(records),
        "brier": brier_score(probabilities_array, targets),
        "ece": expected_calibration_error(probabilities_array, targets),
        "auroc": binary_auroc(probabilities_array, targets),
        "endpoint_coverage": endpoint_coverage(np.asarray(misses), np.asarray(upper_bounds)),
        "sequence_coverage": sequence_coverage(residuals, tubes),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
