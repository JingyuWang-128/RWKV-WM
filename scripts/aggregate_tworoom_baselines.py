#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

LABELS = {
    "flat_mpc": "Flat MPC (official)",
    "flat_trm": "Flat+TRM",
    "hwm": "fixed-scale HWM",
    "vlwm": "VLWM",
    "hilewm_c": "Hi-LeWM-C",
}


def _episodes(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _load_method(
    method: str,
    root: Path,
    offsets: tuple[int, ...],
    seeds: tuple[int, ...],
) -> dict[tuple[int, int, str], dict[str, Any]]:
    rows = {}
    for offset in offsets:
        for seed in seeds:
            condition = root / f"offset_{offset}" / f"seed_{seed}"
            summary = json.loads((condition / "summary.json").read_text())
            if summary.get("status") != "complete":
                raise RuntimeError(f"incomplete result: {condition}")
            for episode in _episodes(condition / "episodes.jsonl"):
                key = (offset, seed, episode["pair_id"])
                rows[key] = {"method": method, **episode, "summary": summary}
    return rows


def _paired_bootstrap(
    baseline: np.ndarray,
    comparison: np.ndarray,
    *,
    seed: int = 3072,
    samples: int = 10000,
) -> list[float]:
    rng = np.random.default_rng(seed)
    delta = comparison.astype(float) - baseline.astype(float)
    chunk = 1000
    means = []
    for start in range(0, samples, chunk):
        count = min(chunk, samples - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        means.append(delta[indices].mean(axis=1))
    values = np.concatenate(means)
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _mcnemar(baseline: np.ndarray, comparison: np.ndarray) -> dict[str, Any]:
    baseline_only = int(np.sum(baseline & ~comparison))
    comparison_only = int(np.sum(~baseline & comparison))
    discordant = baseline_only + comparison_only
    pvalue = (
        float(binomtest(min(baseline_only, comparison_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "baseline_only_success": baseline_only,
        "comparison_only_success": comparison_only,
        "discordant": discordant,
        "exact_two_sided_pvalue": pvalue,
    }


def _summary(method: str, records: dict, offsets: tuple[int, ...]) -> dict[str, Any]:
    per_offset = {}
    for offset in offsets:
        subset = [row for (item_offset, _, _), row in records.items() if item_offset == offset]
        success = np.asarray([row["success"] for row in subset], dtype=bool)
        errors = np.asarray([row["final_state_error"] for row in subset], dtype=float)
        per_offset[str(offset)] = {
            "tasks": len(subset),
            "successes": int(success.sum()),
            "success_rate": float(success.mean()),
            "mean_final_state_error": float(errors.mean()),
            "median_final_state_error": float(np.median(errors)),
        }
    long = [row for (offset, _, _), row in records.items() if offset >= 75]
    long_success = np.asarray([row["success"] for row in long], dtype=bool)
    rates = np.asarray([per_offset[str(offset)]["success_rate"] for offset in offsets])
    auc = float(np.trapezoid(rates, np.asarray(offsets)) / (offsets[-1] - offsets[0]))
    wall = sum(row["summary"]["results"]["wall_time_seconds"] for row in records.values()) / len(
        records
    )
    return {
        "method": method,
        "label": LABELS[method],
        "per_offset": per_offset,
        "long_horizon": {
            "tasks": len(long),
            "successes": int(long_success.sum()),
            "success_rate": float(long_success.mean()),
        },
        "offset_auc": auc,
        "mean_condition_wall_seconds": wall,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate paired Two-Room baselines")
    parser.add_argument(
        "--flat-root",
        type=Path,
        default=Path("artifacts/results/lewm_tworooms_long_matrix"),
    )
    parser.add_argument(
        "--comparison-root",
        type=Path,
        default=Path("artifacts/results/tworoom_comparison_matrix"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/results/tworoom_all_baselines")
    )
    args = parser.parse_args()
    offsets = (25, 50, 75, 100)
    seeds = (0, 1, 2)
    methods = tuple(LABELS)
    records = {
        "flat_mpc": _load_method("flat_mpc", args.flat_root, offsets, seeds),
        **{
            method: _load_method(method, args.comparison_root / method, offsets, seeds)
            for method in methods
            if method != "flat_mpc"
        },
    }
    baseline_keys = sorted(records["flat_mpc"])
    comparisons = {}
    for method in methods[1:]:
        if sorted(records[method]) != baseline_keys:
            raise RuntimeError(f"{method} does not contain the exact paired Flat MPC tasks")
        long_keys = [key for key in baseline_keys if key[0] >= 75]
        flat = np.asarray([records["flat_mpc"][key]["success"] for key in long_keys], dtype=bool)
        current = np.asarray([records[method][key]["success"] for key in long_keys], dtype=bool)
        comparisons[method] = {
            "long_horizon_success_rate_delta": float(current.mean() - flat.mean()),
            "paired_bootstrap_95_ci": _paired_bootstrap(flat, current),
            "mcnemar": _mcnemar(flat, current),
        }
    summaries = {method: _summary(method, records[method], offsets) for method in methods}
    payload = {
        "status": "complete",
        "protocol": {
            "offsets": list(offsets),
            "evaluation_seeds": list(seeds),
            "tasks_per_condition": 100,
            "pairing": "exact start-goal pair and evaluation seed",
        },
        "methods": summaries,
        "paired_vs_official_flat_mpc": comparisons,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    with (args.output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["method", "offset", "tasks", "successes", "success_rate", "mean_state_error"]
        )
        for method in methods:
            for offset in offsets:
                item = summaries[method]["per_offset"][str(offset)]
                writer.writerow(
                    [
                        method,
                        offset,
                        item["tasks"],
                        item["successes"],
                        item["success_rate"],
                        item["mean_final_state_error"],
                    ]
                )
    lines = [
        "# Two-Room 全部基线配对评测",
        "",
        "| 方法 | offset 25 | offset 50 | offset 75 | offset 100 | 长程 75+100 | AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        item = summaries[method]
        rates = [100 * item["per_offset"][str(offset)]["success_rate"] for offset in offsets]
        lines.append(
            f"| {item['label']} | {rates[0]:.2f}% | {rates[1]:.2f}% | {rates[2]:.2f}% | "
            f"{rates[3]:.2f}% | {100 * item['long_horizon']['success_rate']:.2f}% | "
            f"{item['offset_auc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Flat MPC 使用官方 LeWM 实现；其余方法均为同一冻结 LeWM 骨干上的论文规格 "
            "Two-Room 适配，不等同于作者官方复现。所有比较严格共享任务对、动作预算与评测种子。",
            "",
            "配对 bootstrap 与 McNemar 统计见 `summary.json`。",
        ]
    )
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
