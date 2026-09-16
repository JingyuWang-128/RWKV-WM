#!/usr/bin/env python3
"""Plot completed preflight updates, including history saved at gate failure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worldmodel_long_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(
    "artifacts/runs/per_step_pairs/formal_corrected_v1/mask_fixed_seed0/preflight_lr1e4_acc4"
)
OUTPUT = Path("artifacts/results/cc_rwkv/per_step_pairs/formal_corrected_v1/diagnostics")
TASKS = (("tworoom_w", "TwoRoom-W"), ("action_delay", "Action-Delay"))


def segments(rows):
    horizons = np.array([r["horizon"] for r in rows])
    boundaries = np.r_[0, np.flatnonzero(np.diff(horizons)) + 1, len(rows)]
    return list(zip(boundaries[:-1], boundaries[1:], strict=True))


def smooth(rows, values, window):
    """Trailing means reset at every curriculum boundary; never mix horizons."""
    result = np.full(len(values), np.nan)
    for start, end in segments(rows):
        width = min(window, end - start)
        result[start + width - 1 : end] = np.convolve(
            values[start:end], np.ones(width) / width, mode="valid"
        )
    return result


def decorate(axis, rows, failure):
    steps = [r["step"] for r in rows]
    for index, (start, end) in enumerate(segments(rows)):
        left = steps[start] - 0.5
        right = steps[end - 1] + 0.5
        axis.axvspan(left, right, color="#4788aa", alpha=0.035 + 0.035 * (index % 2))
        axis.text(
            (left + right) / 2,
            0.98,
            f"Train H={int(rows[start]['horizon'])}",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
        )
        if start:
            axis.axvline(left, color="#777777", linestyle=":", linewidth=1)
    axis.axvline(failure["step"], color="#bb3333", linestyle="--", linewidth=1)
    axis.set_xlim(0, failure["step"] * 1.025)
    axis.set_xlabel("Completed optimizer updates")
    axis.grid(alpha=0.2)


def series(axis, rows, key, label, color, window, scale=1):
    steps = [r["step"] for r in rows]
    values = np.array([r[key] for r in rows]) * scale
    axis.plot(steps, values, color=color, alpha=0.2, linewidth=0.6)
    axis.plot(steps, smooth(rows, values, window), color=color, linewidth=1.7, label=label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--smooth-window", type=int, default=25)
    args = parser.parse_args()
    if args.smooth_window < 1:
        parser.error("smooth-window must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.3), layout="constrained")
    components, comp_axes = plt.subplots(2, 2, figsize=(13, 8.3), layout="constrained")
    manifest = {}
    for index, (task, title) in enumerate(TASKS):
        source = args.run_root / task / "failed_history.json"
        rows = json.loads(source.read_text())
        failure = json.loads((source.parent / "failure.json").read_text())
        assert rows[-1]["step"] == failure["last_completed_step"]
        expected = np.array(
            [
                r["factual_prediction"]
                + r["noop_prediction"]
                + 0.5 * r["paired_effect"]
                + 0.05 * r["effect_direction"]
                + 0.05 * r["effect_magnitude"]
                for r in rows
            ]
        )
        np.testing.assert_allclose(expected, [r["loss"] for r in rows], rtol=2e-6, atol=1e-6)
        train, val = axes[index]
        series(
            train,
            rows,
            "loss",
            f"Train total: {args.smooth_window}-step mean",
            "#0072B2",
            args.smooth_window,
        )
        train.set_title(f"{title} | training loss (last update {int(rows[-1]['step'])})")
        train.set_ylabel("Training total loss")
        validation = [r for r in rows if "validation_loss" in r]
        val.plot(
            [r["step"] for r in validation],
            [r["validation_loss"] for r in validation],
            "o-",
            color="#D55E00",
            label="Fixed H=20 validation total",
            markersize=6,
        )
        for r in validation:
            val.annotate(
                f"{r['validation_loss']:.4f}",
                (r["step"], r["validation_loss"]),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        if len(validation) == 1:
            val.text(
                0.5,
                0.3,
                "Only one validation point; no trend can be inferred",
                transform=val.transAxes,
                ha="center",
                fontsize=9,
            )
        val.margins(y=0.3)
        val.set_title(f"{title} | validation (256 samples, H=20)")
        val.set_ylabel("Validation total loss")
        base, aux = comp_axes[index]
        base.set_title(f"{title} | prediction loss contributions")
        aux.set_title(f"{title} | weighted auxiliary loss contributions")
        for key, label, color in (
            ("factual_prediction", "Factual prediction", "#0072B2"),
            ("noop_prediction", "No-op prediction", "#009E73"),
        ):
            series(base, rows, key, label, color, args.smooth_window)
        for key, label, scale, color in (
            ("paired_effect", "0.5 x Effect Smooth-L1", 0.5, "#D55E00"),
            ("effect_direction", "0.05 x Direction", 0.05, "#CC79A7"),
            ("effect_magnitude", "0.05 x Magnitude", 0.05, "#E69F00"),
        ):
            series(aux, rows, key, label, color, args.smooth_window, scale)
        if task == "action_delay":
            aux.text(
                0.5,
                0.7,
                "Direction and magnitude = 0 at logged H=1 / H=5;\n"
                "effect contribution is near zero",
                transform=aux.transAxes,
                ha="center",
                fontsize=9,
            )
        for axis in (train, val, base, aux):
            decorate(axis, rows, failure)
            axis.legend(loc="best", fontsize=8)
        base.set_ylabel("Loss contribution")
        aux.set_ylabel("Loss contribution (weighted)")
        manifest[task] = {
            "source": str(source),
            "completed_updates": len(rows),
            "last_completed_step": rows[-1]["step"],
            "failure_step": failure["step"],
            "validation": [{"step": r["step"], "loss": r["validation_loss"]} for r in validation],
        }
    note = (
        "seed=0 | effect weight=0.5 | LR=1e-4 | effective batch=32 | "
        f"faint=raw, bold={args.smooth_window}-step mean\n"
        "Red dashed line: clipping gate stopped run BEFORE this update; not formal F3/F4 results"
    )
    fig.suptitle("Mask-fixed preflight: training and validation loss\n" + note, fontsize=11)
    components.suptitle("Mask-fixed preflight: contributions to total loss\n" + note, fontsize=11)
    for figure, name in (
        (fig, "mask_fixed_preflight_loss_curves"),
        (components, "mask_fixed_preflight_loss_components"),
    ):
        for suffix in ("png", "svg"):
            path = args.output_dir / f"{name}.{suffix}"
            figure.savefig(path, dpi=170, bbox_inches="tight")
            print(path)
        plt.close(figure)
    (args.output_dir / "mask_fixed_preflight_loss_manifest.json").write_text(
        json.dumps({"smoothing_window": args.smooth_window, "tasks": manifest}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
