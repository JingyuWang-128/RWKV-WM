from __future__ import annotations

import argparse
import json
from pathlib import Path

from cape_wm.cc_rwkv.m5 import (
    REQUIRED_METHODS,
    REQUIRED_SEEDS,
    aggregate_gate_ab,
    load_m5_run,
    select_effect_weight_across_seeds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate M5 Gate A/B paired runs")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=3072)
    return parser.parse_args()


def _failure_rows(runs, gate_summary):
    rows = []
    for task, task_summary in gate_summary["tasks"].items():
        strongest = task_summary["comparisons"].get("strongest_baseline")
        if strongest is None:
            continue
        task_runs = [run for run in runs if run.task == task]
        for seed in sorted(REQUIRED_SEEDS):
            b6 = next((run for run in task_runs if run.method == "b6" and run.seed == seed), None)
            base = next(
                (run for run in task_runs if run.method == strongest and run.seed == seed),
                None,
            )
            if b6 is None or base is None:
                continue
            b6_samples = {row["sample_id"]: row for row in b6.samples}
            base_samples = {row["sample_id"]: row for row in base.samples}
            if set(b6_samples) != set(base_samples):
                raise ValueError("failure review requires exact paired sample IDs")
            for sample_id in b6_samples:
                delta = b6_samples[sample_id]["cee_auc"] - base_samples[sample_id]["cee_auc"]
                if delta > 0:
                    rows.append(
                        {
                            "task": task,
                            "seed": seed,
                            "sample_id": sample_id,
                            "strongest_baseline": strongest,
                            "b6_cee_auc": b6_samples[sample_id]["cee_auc"],
                            "baseline_cee_auc": base_samples[sample_id]["cee_auc"],
                            "b6_minus_baseline": delta,
                            "true_effect_norm": b6_samples[sample_id].get("true_effect_norm"),
                            "effect_valid_fraction": b6_samples[sample_id].get(
                                "effect_valid_fraction"
                            ),
                            "effect_threshold": b6_samples[sample_id].get("effect_threshold"),
                            "reason": "b6_counterfactual_effect_error_worse",
                        }
                    )
    return sorted(rows, key=lambda row: row["b6_minus_baseline"], reverse=True)


def _report(payload: dict) -> str:
    lines = [
        "# CC-RWKV M5 Gate A/B report",
        "",
        f"M5@20 overall: **{payload['gates']['m5_at_20_overall_status']}**",
        "",
        f"Full Gate A/B overall: **{payload['gates']['overall_status']}**",
        "",
        "| Task | Gate A | Gate B@20 | Full Gate B | Samples | Max primitive horizon |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, item in payload["gates"]["tasks"].items():
        completeness = item["completeness"]
        task_runs = [run for run in payload["run_inventory"] if run["task"] == task]
        samples = max((run["data_samples"] for run in task_runs), default=0)
        horizon = max((run["max_primitive_horizon"] for run in task_runs), default=0)
        lines.append(
            f"| {task} | {item['gate_a']['status']} | "
            f"{item['gate_b_at_20']['status']} | {item['gate_b']['status']} | "
            f"{samples} | {horizon} |"
        )
        del completeness
    lines.extend(["", "## Effect-weight selection", ""])
    if payload["effect_weight_selection"]:
        for task, selection in payload["effect_weight_selection"].items():
            lines.append(
                f"- {task}: `{selection['status']}`, selected="
                f"`{selection['selected_effect_weight']}`."
            )
    else:
        lines.append("- NOT_ASSESSABLE: validation candidate matrix is incomplete.")
    lines.extend(["", "## Blocking reasons", ""])
    for task, item in payload["gates"]["tasks"].items():
        m5_reasons = sorted(
            set(item["gate_a"]["reasons"] + item["gate_b_at_20"]["reasons"])
        )
        full_reasons = sorted(set(item["gate_b"]["reasons"]))
        lines.append(
            f"- {task} M5@20: {', '.join(m5_reasons) if m5_reasons else 'none'}"
        )
        lines.append(
            f"- {task} full Gate B: "
            f"{', '.join(full_reasons) if full_reasons else 'none'}"
        )
    lines.extend(
        [
            "",
            "> Gate A PASS requires 5,000 snapshots and 20 primitive steps; full Gate B",
            "> additionally requires 20,000 snapshots, 50 primitive steps, seeds 0/1/2,",
            "> B2/B3/B4/B6 and paired 10,000-bootstrap intervals.",
            "> Gate B@20 is the explicit provisional M5 decision and never substitutes for",
            "> the full 20,000-snapshot/50-step Gate B.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    paths = sorted({path.parent for path in args.root.rglob("m5_run.json")})
    if not paths:
        raise FileNotFoundError(f"no M5 runs found below {args.root}")
    all_runs = [load_m5_run(path) for path in paths]
    validation = [run for run in all_runs if run.split == "validation"]
    test = [run for run in all_runs if run.split == "test"]
    selections = {}
    selected_test = []
    for task in sorted({run.task for run in all_runs}):
        candidates = [run for run in validation if run.task == task]
        try:
            selection = select_effect_weight_across_seeds(candidates)
        except ValueError as error:
            selection = {
                "status": "not_assessable",
                "reason": str(error),
                "selected_effect_weight": None,
            }
        selections[task] = selection
        selected_weight = selection.get("selected_effect_weight")
        task_test = [run for run in test if run.task == task]
        selected_test.extend(
            run
            for run in task_test
            if run.method != "b6" or selected_weight is None or run.effect_weight == selected_weight
        )
    gates = aggregate_gate_ab(
        selected_test,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    inventory = [
        {
            "task": run.task,
            "method": run.method,
            "seed": run.seed,
            "effect_weight": run.effect_weight,
            "split": run.split,
            "data_samples": run.data_samples,
            "trained_max_primitive_horizon": run.trained_max_primitive_horizon,
            "max_primitive_horizon": run.max_primitive_horizon,
            "fairness": run.fairness,
            "source": run.source,
        }
        for run in all_runs
    ]
    payload = {
        "schema_version": "cc_rwkv_m5_aggregate_v2",
        "required_methods": sorted(REQUIRED_METHODS),
        "required_seeds": sorted(REQUIRED_SEEDS),
        "effect_weight_selection": selections,
        "gates": gates,
        "run_inventory": inventory,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    failures = _failure_rows(selected_test, gates)
    with (args.output / "failed_samples.jsonl").open("w") as handle:
        for row in failures:
            handle.write(json.dumps(row) + "\n")
    (args.output / "gate_report.md").write_text(_report(payload))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
