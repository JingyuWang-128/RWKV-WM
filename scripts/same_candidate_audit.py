#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from cape_wm.audit import (
    AuditedCandidate,
    CandidateAuditRecord,
    classify_same_candidate_failure,
    summarize_same_candidate_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the preregistered same-candidate audit")
    parser.add_argument("records", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.records.read_text())
    records = [
        CandidateAuditRecord(
            candidates=tuple(
                AuditedCandidate(
                    oracle_executable=bool(item["oracle_executable"]),
                    oracle_goal_progress=float(item["oracle_goal_progress"]),
                )
                for item in record["candidates"]
            ),
            selected_index=record.get("selected_index"),
            execution_succeeded=bool(record["execution_succeeded"]),
        )
        for record in payload
    ]
    counts = summarize_same_candidate_audit(records)
    total = len(records)
    viable = [
        [candidate.oracle_executable and candidate.oracle_goal_progress > 0.0
         for candidate in record.candidates]
        for record in records
    ]
    summary = {
        "status": "complete",
        "audit_scope": "frozen validation candidate sets",
        "total_candidate_decisions": total,
        "counts": counts,
        "percentages": {
            key: 100.0 * value / total if total else 0.0
            for key, value in counts.items()
        },
        "risk_ranking_breakdown": {
            "viable_but_fallback": sum(
                any(flags) and record.selected_index is None
                for record, flags in zip(records, viable, strict=True)
            ),
            "selected_nonviable": sum(
                classify_same_candidate_failure(record) == "risk_ranking"
                and record.selected_index is not None
                for record in records
            ),
        },
        "candidate_statistics": {
            "decisions_with_oracle_viable_candidate": sum(map(any, viable)),
            "mean_oracle_viable_candidates": (
                sum(map(sum, viable)) / total if total else 0.0
            ),
        },
    }
    raw_candidates = [item for record in payload for item in record["candidates"]]
    if raw_candidates and all("feasible" in item for item in raw_candidates):
        oracle_viable = np.asarray(
            [
                bool(item["oracle_executable"])
                and float(item["oracle_goal_progress"]) > 0.0
                for item in raw_candidates
            ],
            dtype=bool,
        )
        risk_scores = -np.asarray(
            [float(item["miss_upper_bound"]) for item in raw_candidates]
        )
        positive_ranks = rankdata(risk_scores)[oracle_viable].sum()
        positives = int(oracle_viable.sum())
        negatives = int((~oracle_viable).sum())
        auc = (
            (positive_ranks - positives * (positives + 1) / 2)
            / (positives * negatives)
            if positives and negatives
            else float("nan")
        )
        feasible = np.asarray(
            [bool(item["feasible"]) for item in raw_candidates], dtype=bool
        )
        durations = sorted({int(item["duration"]) for item in raw_candidates})
        summary["risk_filter_diagnostics"] = {
            "total_candidates": len(raw_candidates),
            "oracle_viable_candidates": positives,
            "oracle_viable_candidates_filtered": int(
                (oracle_viable & ~feasible).sum()
            ),
            "risk_upper_viability_auc": float(auc),
            "by_duration": {
                str(duration): {
                    "candidates": sum(
                        int(item["duration"]) == duration for item in raw_candidates
                    ),
                    "predicted_feasible_rate": float(
                        np.mean(
                            [
                                bool(item["feasible"])
                                for item in raw_candidates
                                if int(item["duration"]) == duration
                            ]
                        )
                    ),
                }
                for duration in durations
            },
        }
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        percentages = summary["percentages"]
        breakdown = summary["risk_ranking_breakdown"]
        filter_diagnostics = summary.get("risk_filter_diagnostics")
        filter_note = ""
        if filter_diagnostics is not None:
            filter_note = (
                f" Across {filter_diagnostics['total_candidates']} candidates, "
                f"{filter_diagnostics['oracle_viable_candidates_filtered']} of "
                f"{filter_diagnostics['oracle_viable_candidates']} oracle-viable "
                "candidates were filtered. The risk upper bound's viability AUC was "
                f"{filter_diagnostics['risk_upper_viability_auc']:.3f}."
            )
        args.report.write_text(
            "# Two-Room same-candidate audit\n\n"
            "The audit reuses the frozen validation candidate sets. Candidate "
            "reachability is evaluated from the true room geometry; frozen low-level "
            "actions are replayed independently. Latent subgoals are mapped to physical "
            "positions by nearest neighbor in the frozen dataset cache.\n\n"
            f"Audited candidate decisions: {total}.\n\n"
            "| Attribution | Count | Share |\n"
            "|---|---:|---:|\n"
            + "".join(
                f"| {key} | {value} | {percentages[key]:.2f}% |\n"
                for key, value in counts.items()
            )
            + "\n"
            f"Risk/ranking failures split into {breakdown['viable_but_fallback']} "
            "decisions that fell back despite an oracle-viable candidate and "
            f"{breakdown['selected_nonviable']} decisions that selected a nonviable "
            f"candidate.{filter_note} The dominant next target is therefore feasibility filtering "
            "and candidate ranking, not a new low-level controller.\n"
        )
    print(rendered, end="")


if __name__ == "__main__":
    main()
