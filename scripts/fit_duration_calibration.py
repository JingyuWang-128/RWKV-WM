#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cape_wm.calibration import (
    fit_duration_conditional_from_records,
    save_duration_calibrator,
)
from cape_wm.checkpoints import load_risk_model
from cape_wm.data import load_calibration_records
from cape_wm.torch_wrappers import TorchRiskPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit duration-conditional CAPE calibration")
    parser.add_argument("records", type=Path)
    parser.add_argument("risk_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model, config = load_risk_model(args.risk_checkpoint, args.device)
    predictor = TorchRiskPredictor(
        model, args.device, miss_scale=float(config.get("miss_scale", 1.0))
    )
    calibrator = fit_duration_conditional_from_records(
        load_calibration_records(args.records), predictor, alpha=args.alpha
    )
    save_duration_calibrator(args.output, calibrator)
    print(json.dumps(calibrator.as_dict(), indent=2), flush=True)


if __name__ == "__main__":
    main()
