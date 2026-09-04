#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the best Two-Room CAPE validation grid")
    parser.add_argument(
        "--input", type=Path, default=Path("artifacts/results/tworoom_cape_selection")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/checkpoints/tworoom_cape/selected_hyperparameters.json"),
    )
    args = parser.parse_args()
    rows = []
    for condition in sorted(args.input.glob("lc_*_lr_*")):
        summaries = [
            json.loads((condition / f"offset_{offset}" / "summary.json").read_text())
            for offset in (25, 100)
        ]
        rates = [item["results"]["success_rate"] for item in summaries]
        walls = [item["results"]["wall_time_seconds"] for item in summaries]
        params = summaries[0]["selected_hyperparameters"]
        rows.append(
            {
                **params,
                "short_success_rate": rates[0],
                "long_success_rate": rates[1],
                "mean_success_rate": float(np.mean(rates)),
                "wall_time_seconds": float(np.sum(walls)),
                "condition": condition.name,
            }
        )
    if len(rows) != 16:
        raise SystemExit(f"selection grid incomplete: found {len(rows)}/16 conditions")
    # Long horizon is primary. Mean success, short success, then lower wall
    # time are deterministic tie breakers fixed before final-test evaluation.
    ranked = sorted(
        rows,
        key=lambda row: (
            -row["long_success_rate"],
            -row["mean_success_rate"],
            -row["short_success_rate"],
            row["wall_time_seconds"],
            row["lambda_compute"],
            row["lambda_risk"],
        ),
    )
    payload = {
        "status": "frozen_before_final_test",
        "selection_split": "validation",
        "selection_offsets": [25, 100],
        "ranking_rule": "long, mean, short descending; wall ascending",
        "selected": ranked[0],
        "grid": ranked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
