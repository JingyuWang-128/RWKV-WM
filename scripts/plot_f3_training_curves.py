#!/usr/bin/env python3
"""Plot formal_corrected_v1 F3 training and validation loss curves."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worldmodel_long_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TASKS = (("tworoom_w", "TwoRoom-W"), ("action_delay", "Action-Delay"))
WEIGHTS = (("0p5", 0.5), ("1p0", 1.0), ("2p0", 2.0))
SEED_COLORS = {0: "#0072B2", 1: "#D55E00", 2: "#009E73"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/runs/per_step_pairs/formal_corrected_v1/f3"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/results/cc_rwkv/per_step_pairs/formal_corrected_v1/diagnostics"
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=500,
        help="Moving-average window in optimizer steps.",
    )
    return parser.parse_args()


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values.copy()
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    averaged = (cumulative[window:] - cumulative[:-window]) / window
    prefix = np.full(window - 1, np.nan)
    return np.concatenate((prefix, averaged))


def load_history(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as handle:
        history = json.load(handle)
    if not isinstance(history, list) or not history:
        raise ValueError(f"empty or invalid history: {path}")
    return history


def main() -> None:
    args = parse_args()
    if args.smooth_window <= 0:
        raise ValueError("--smooth-window must be positive")

    figure, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True, constrained_layout=True)
    completed = 0
    discovered = 0

    for row, (task_dir, task_name) in enumerate(TASKS):
        for column, (weight_tag, weight) in enumerate(WEIGHTS):
            axis = axes[row, column]
            for left, right, horizon in (
                (0, 10_000, 1),
                (10_000, 20_000, 5),
                (20_000, 30_000, 10),
                (30_000, 40_000, 20),
            ):
                axis.axvspan(
                    left,
                    right,
                    color="#777777",
                    alpha=0.035 if horizon in (1, 10) else 0.075,
                    zorder=0,
                )
                axis.text(
                    (left + right) / 2,
                    0.985,
                    f"H={horizon}",
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="#666666",
                )
            for boundary in (10_000, 20_000, 30_000):
                axis.axvline(boundary, color="#888888", linewidth=0.8, linestyle="--")

            for seed in range(3):
                run_dir = args.run_root / task_dir / f"b6_w{weight_tag}_seed{seed}"
                history_path = run_dir / "history.json"
                if not history_path.exists():
                    continue
                discovered += 1
                history = load_history(history_path)
                steps = np.asarray([row_["step"] for row_ in history], dtype=np.float64)
                losses = np.asarray([row_["loss"] for row_ in history], dtype=np.float64)
                smoothed = moving_average(losses, args.smooth_window)
                is_complete = (run_dir / "summary.json").exists() and steps[-1] >= 40_000
                completed += int(is_complete)
                status = "complete" if is_complete else f"active: {int(steps[-1]):,}"
                axis.plot(
                    steps,
                    smoothed,
                    color=SEED_COLORS[seed],
                    linewidth=1.45,
                    label=f"seed {seed} ({status})",
                )

                validation = [
                    (row_["step"], row_["validation_loss"])
                    for row_ in history
                    if "validation_loss" in row_
                ]
                if validation:
                    val_steps, val_losses = zip(*validation, strict=True)
                    axis.plot(
                        val_steps,
                        val_losses,
                        color=SEED_COLORS[seed],
                        linewidth=0.8,
                        linestyle=":",
                        marker="o",
                        markersize=2.2,
                        alpha=0.72,
                    )

            axis.set_title(f"{task_name} | effect weight={weight:g}", fontsize=11)
            axis.set_xlim(0, 40_000)
            axis.grid(axis="y", alpha=0.2)
            axis.legend(loc="best", fontsize=7.5, framealpha=0.86)
            if column == 0:
                axis.set_ylabel("Loss")
            if row == 1:
                axis.set_xlabel("Optimizer step")

    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    expected = len(TASKS) * len(WEIGHTS) * 3
    active = discovered - completed
    queued = expected - discovered
    figure.suptitle(
        "formal_corrected_v1 F3 convergence\n"
        f"solid: training loss ({args.smooth_window}-step moving average); "
        f"dotted: full-H20 validation loss | "
        f"{completed}/{expected} complete, {active} active, {queued} queued | {timestamp}",
        fontsize=14,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "f3_training_loss_curves.png"
    svg_path = args.output_dir / "f3_training_loss_curves.svg"
    figure.savefig(png_path, dpi=180)
    figure.savefig(svg_path)
    plt.close(figure)
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
