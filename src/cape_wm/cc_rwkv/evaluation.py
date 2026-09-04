from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from cape_wm.stats import paired_bootstrap_difference

from .metrics import fit_ridge_probe, probe_r2
from .trainer import M4Trainer, matrix_action_probe_features
from .training import BranchBatch


@torch.no_grad()
def no_op_intervention_max(trainer: M4Trainer, batch: BranchBatch) -> float | None:
    if trainer.config.method != "b6":
        return None
    predictor = trainer.model
    state = predictor.consume_history(
        batch.history_latents,
        batch.history_actions,
        mask=batch.history_mask,
    )
    initial = batch.branch_latents[:, 2, 0]
    action = batch.branch_actions[:, 2, 0]
    _, _, diagnostics = predictor.step(
        initial, action, state, action.clone(), return_diagnostics=True
    )
    names = ("action_decay_delta", "action_erase_delta", "action_write_delta")
    return max(float(diagnostics[name].abs().max()) for name in names)


def action_probe_report(
    trainer: M4Trainer,
    train_batches: list[BranchBatch],
    evaluation_batches: list[BranchBatch],
) -> dict[str, float]:
    trainer.model.eval()
    train_features, train_targets = zip(
        *(matrix_action_probe_features(trainer.model, batch) for batch in train_batches),
        strict=True,
    )
    test_features, test_targets = zip(
        *(matrix_action_probe_features(trainer.model, batch) for batch in evaluation_batches),
        strict=True,
    )
    train_x, train_y = torch.cat(train_features), torch.cat(train_targets)
    test_x, test_y = torch.cat(test_features), torch.cat(test_targets)
    probe = fit_ridge_probe(train_x, train_y, regularization=1e-2)
    prediction_mse = (probe.predict(test_x) - test_y).square().mean(dim=1)
    mean_baseline_mse = (train_y.mean(dim=0) - test_y).square().mean(dim=1)
    interval = paired_bootstrap_difference(
        prediction_mse.detach().cpu().numpy(),
        mean_baseline_mse.detach().cpu().numpy(),
        resamples=10_000,
        seed=3072,
    )
    return {
        "train_r2": float(probe_r2(probe, train_x, train_y)),
        "evaluation_r2": float(probe_r2(probe, test_x, test_y)),
        "feature_dim": int(train_x.shape[1]),
        "target_dim": int(train_y.shape[1]),
        "probe_mse": float(prediction_mse.mean()),
        "mean_baseline_mse": float(mean_baseline_mse.mean()),
        "probe_mse_minus_mean_baseline_ci": asdict(interval),
    }


def gate_report_markdown(
    *,
    method: str,
    summary: dict[str, Any],
    no_op_max: float | None,
    fairness_status: str,
) -> str:
    gate = summary.get("gate")
    lines = [
        f"# {method.upper()} automatic gate report",
        "",
        f"- Fairness audit: **{fairness_status}**",
        f"- Validation score: `{summary['validation_score']:.6g}`",
        f"- One-step RMSE: `{summary['one_step_rmse']:.6g}`",
        f"- CEE AUC: `{summary['cee_auc']:.6g}`",
        f"- Trajectory AUC: `{summary['trajectory_auc']:.6g}`",
    ]
    if no_op_max is not None:
        lines.append(f"- actual=reference max delta: `{no_op_max:.6g}`")
        lines.append(f"- No-op tolerance `<1e-7`: **{'PASS' if no_op_max < 1e-7 else 'FAIL'}**")
    if gate is not None:
        lines.extend(
            [
                f"- Gate mean/std: `{gate['mean']:.6g}` / `{gate['std']:.6g}`",
                f"- Gate saturated fraction: `{gate['saturated_fraction']:.6g}`",
                "- Gate non-collapse: **"
                + ("PASS" if gate["saturated_fraction"] < 1.0 else "FAIL")
                + "**",
            ]
        )
    lines.extend(
        [
            "",
            "> This engineering report does not make a Gate B comparison claim; M5 supplies",
            "> three-seed paired confidence intervals and the B2/B3/B4 comparison.",
            "",
        ]
    )
    return "\n".join(lines)


def write_evaluation_artifacts(
    output: str | Path,
    trainer: M4Trainer,
    evaluation_batches: list[BranchBatch],
    train_probe_batches: list[BranchBatch],
    *,
    horizons: tuple[int, ...],
    provenance: dict[str, Any],
    fairness_status: str,
) -> dict[str, Any]:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    available = min(batch.horizon for batch in evaluation_batches)
    horizons = tuple(sorted({min(horizon, available) for horizon in horizons if horizon > 0}))
    metrics = []
    for horizon in horizons:
        result = trainer.evaluate(evaluation_batches, horizon=horizon)
        metrics.append({"horizon": horizon, **result})
    summary = dict(metrics[-1])
    summary["method"] = trainer.config.method
    summary["effect_threshold"] = trainer.effect_threshold
    summary["probe"] = action_probe_report(trainer, train_probe_batches, evaluation_batches)
    no_op_max = no_op_intervention_max(trainer, evaluation_batches[0])
    summary["actual_equals_reference_max_delta"] = no_op_max
    with (destination / "metrics.jsonl").open("w") as handle:
        for record in metrics:
            handle.write(json.dumps(record) + "\n")
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (destination / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "trainer": trainer.config.__dict__
                if hasattr(trainer.config, "__dict__")
                else {
                    name: getattr(trainer.config, name)
                    for name in trainer.config.__dataclass_fields__
                },
                "loss_weights": {
                    name: getattr(trainer.loss_weights, name)
                    for name in trainer.loss_weights.__dataclass_fields__
                },
            },
            sort_keys=True,
        )
    )
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (destination / "failed_samples.jsonl").write_text("")
    (destination / "gate_report.md").write_text(
        gate_report_markdown(
            method=trainer.config.method,
            summary=summary,
            no_op_max=no_op_max,
            fairness_status=fairness_status,
        )
    )
    return summary
