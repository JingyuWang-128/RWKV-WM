#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cape_wm.stats import evaluate_phase_gate, paired_mcnemar_exact


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _condition(root: Path, offset: int, seed: int) -> tuple[list[dict], dict]:
    path = root / f"offset_{offset}" / f"seed_{seed}"
    return _read_jsonl(path / "episodes.jsonl"), json.loads((path / "summary.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate paired Two-Room CAPE Phase-B gate")
    parser.add_argument(
        "--cape-root", type=Path, default=Path("artifacts/results/tworoom_cape_matrix_gpu_final")
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("artifacts/results/tworoom_comparison_matrix_gpu_final/flat_trm"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/results/tworoom_cape_phase_b")
    )
    parser.add_argument(
        "--cape-wall-root",
        type=Path,
        help="optional equivalent rerun root used only for measured wall time",
    )
    parser.add_argument("--offsets", type=int, nargs="+", default=[25, 50, 75, 100])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--tasks-per-condition", type=int, default=100)
    args = parser.parse_args()
    offsets = tuple(args.offsets)
    seeds = tuple(args.seeds)
    long_offsets = offsets[-2:]
    cape_by_offset: dict[int, list[dict]] = {offset: [] for offset in offsets}
    base_by_offset: dict[int, list[dict]] = {offset: [] for offset in offsets}
    cape_wall: dict[int, float] = {offset: 0.0 for offset in offsets}
    base_wall: dict[int, float] = {offset: 0.0 for offset in offsets}
    planning = {
        "fallback_decisions": 0,
        "candidate_decisions": 0,
        "feasible_candidates": 0,
        "reported_model_transitions": 0.0,
        "decisions": 0,
        "events": {},
        "triggers": {},
        "chosen_durations": {},
    }
    for offset in offsets:
        for seed in seeds:
            cape_rows, cape_summary = _condition(args.cape_root, offset, seed)
            base_rows, base_summary = _condition(args.baseline_root, offset, seed)
            wall_summary = cape_summary
            if args.cape_wall_root is not None:
                wall_path = (
                    args.cape_wall_root
                    / f"offset_{offset}"
                    / f"seed_{seed}"
                    / "summary.json"
                )
                if wall_path.is_file():
                    wall_rows, wall_summary = _condition(
                        args.cape_wall_root, offset, seed
                    )
                    if wall_rows != cape_rows:
                        raise RuntimeError(
                            "wall-time rerun changed task outcomes at "
                            f"offset={offset}, seed={seed}"
                        )
            cape = {row["pair_id"]: row for row in cape_rows}
            base = {row["pair_id"]: row for row in base_rows}
            if set(cape) != set(base) or len(cape) != args.tasks_per_condition:
                raise RuntimeError(f"pair mismatch at offset={offset}, seed={seed}")
            keys = sorted(cape)
            cape_by_offset[offset].extend(cape[key] for key in keys)
            base_by_offset[offset].extend(base[key] for key in keys)
            cape_wall[offset] += float(wall_summary["results"]["wall_time_seconds"])
            base_wall[offset] += float(base_summary["results"]["wall_time_seconds"])
            item = cape_summary["planning"]
            for key in (
                "fallback_decisions",
                "candidate_decisions",
                "feasible_candidates",
                "reported_model_transitions",
                "decisions",
            ):
                planning[key] += item[key]
            for key in ("events", "triggers", "chosen_durations"):
                for name, value in item[key].items():
                    planning[key][name] = planning[key].get(name, 0) + value
    cape_long = np.asarray(
        [row["success"] for offset in long_offsets for row in cape_by_offset[offset]], dtype=bool
    )
    base_long = np.asarray(
        [row["success"] for offset in long_offsets for row in base_by_offset[offset]], dtype=bool
    )
    cape_short = np.asarray([row["success"] for row in cape_by_offset[25]], dtype=bool)
    base_short = np.asarray([row["success"] for row in base_by_offset[25]], dtype=bool)
    gate = evaluate_phase_gate(
        cape_long,
        base_long,
        cape_short,
        base_short,
        cape_time=sum(cape_wall[offset] for offset in long_offsets),
        baseline_time=sum(base_wall[offset] for offset in long_offsets),
        seed=3072,
    )
    per_offset = {
        str(offset): {
            "tasks": len(cape_by_offset[offset]),
            "cape_success_rate": float(np.mean([row["success"] for row in cape_by_offset[offset]])),
            "flat_trm_success_rate": float(
                np.mean([row["success"] for row in base_by_offset[offset]])
            ),
            "delta": float(
                np.mean([row["success"] for row in cape_by_offset[offset]])
                - np.mean([row["success"] for row in base_by_offset[offset]])
            ),
            "cape_wall_seconds": cape_wall[offset],
            "flat_trm_wall_seconds": base_wall[offset],
        }
        for offset in offsets
    }
    payload = {
        "status": "complete",
        "gate": asdict(gate),
        "paired_mcnemar_long_pvalue": paired_mcnemar_exact(cape_long, base_long),
        "per_offset": per_offset,
        "long_horizon": {
            "tasks": len(cape_long),
            "cape_success_rate": float(cape_long.mean()),
            "flat_trm_success_rate": float(base_long.mean()),
            "delta": float(cape_long.mean() - base_long.mean()),
        },
        "planning": planning,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Two-Room CAPE-WM Phase-B gate",
        "",
        f"Gate: **{'PASS' if gate.passed else 'FAIL'}**",
        "",
        "| Offset | CAPE-WM | Flat+TRM | Delta |",
        "|---:|---:|---:|---:|",
    ]
    for offset in offsets:
        item = per_offset[str(offset)]
        lines.append(
            f"| {offset} | {100 * item['cape_success_rate']:.2f}% | "
            f"{100 * item['flat_trm_success_rate']:.2f}% | {100 * item['delta']:+.2f}pp |"
        )
    lines.extend(
        [
            "",
            f"Long-horizon delta: {100 * gate.long_horizon_gain:+.2f}pp; "
            f"95% paired bootstrap [{100 * gate.confidence_interval.low:+.2f}, "
            f"{100 * gate.confidence_interval.high:+.2f}]pp.",
            f"Planning overhead: {100 * gate.overhead_ratio:+.2f}%.",
            f"Short-horizon regression: {100 * gate.short_horizon_regression:.2f}pp.",
            f"Reasons: {', '.join(gate.reasons)}.",
        ]
    )
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
