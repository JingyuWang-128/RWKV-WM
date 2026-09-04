from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest, rankdata


def brier_score(probabilities: np.ndarray, targets: np.ndarray) -> float:
    probability = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if probability.shape != target.shape:
        raise ValueError("probabilities and targets must have identical shapes")
    return float(np.mean((probability - target) ** 2))


def expected_calibration_error(
    probabilities: np.ndarray, targets: np.ndarray, bins: int = 10
) -> float:
    probability = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    if probability.shape != target.shape:
        raise ValueError("probabilities and targets must have identical shapes")
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for index in range(bins):
        mask = indices == index
        if np.any(mask):
            error += mask.mean() * abs(probability[mask].mean() - target[mask].mean())
    return float(error)


def binary_auroc(scores: np.ndarray, targets: np.ndarray) -> float:
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    target = np.asarray(targets, dtype=np.int64).reshape(-1)
    positives = target == 1
    negatives = target == 0
    if not np.any(positives) or not np.any(negatives):
        return float("nan")
    ranks = rankdata(score, method="average")
    positive_count = positives.sum()
    negative_count = negatives.sum()
    value = (ranks[positives].sum() - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )
    return float(value)


def endpoint_coverage(observed_miss: np.ndarray, upper_bound: np.ndarray) -> float:
    observed = np.asarray(observed_miss, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper_bound, dtype=np.float64).reshape(-1)
    if observed.shape != upper.shape or observed.size == 0:
        raise ValueError("coverage arrays must have the same non-zero shape")
    return float(np.mean(observed <= upper))


def sequence_coverage(
    observed_residuals: list[np.ndarray], tube_thresholds: list[np.ndarray]
) -> float:
    if len(observed_residuals) != len(tube_thresholds) or not observed_residuals:
        raise ValueError("residual and threshold sequence collections must match")
    covered = []
    for residual, threshold in zip(observed_residuals, tube_thresholds, strict=True):
        left = np.asarray(residual, dtype=np.float64).reshape(-1)
        right = np.asarray(threshold, dtype=np.float64).reshape(-1)
        if left.shape != right.shape or left.size == 0:
            raise ValueError("each residual and threshold sequence must match and be non-empty")
        covered.append(bool(np.all(left <= right)))
    return float(np.mean(covered))


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    estimate: float
    low: float
    high: float
    confidence: float


def paired_bootstrap_difference(
    treatment: np.ndarray,
    baseline: np.ndarray,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapInterval:
    left = np.asarray(treatment, dtype=np.float64).reshape(-1)
    right = np.asarray(baseline, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("paired samples must have the same non-zero length")
    difference = left - right
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, difference.size, size=(resamples, difference.size))
    estimates = difference[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=float(difference.mean()),
        low=float(np.quantile(estimates, tail)),
        high=float(np.quantile(estimates, 1.0 - tail)),
        confidence=confidence,
    )


def paired_mcnemar_exact(treatment: np.ndarray, baseline: np.ndarray) -> float:
    left = np.asarray(treatment, dtype=bool).reshape(-1)
    right = np.asarray(baseline, dtype=bool).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("paired outcomes must have identical shapes")
    treatment_only = int(np.sum(left & ~right))
    baseline_only = int(np.sum(~left & right))
    discordant = treatment_only + baseline_only
    if discordant == 0:
        return 1.0
    return float(binomtest(min(treatment_only, baseline_only), discordant, 0.5).pvalue)


def horizon_auc(offsets: np.ndarray, success_rates: np.ndarray) -> float:
    horizon = np.asarray(offsets, dtype=np.float64)
    rates = np.asarray(success_rates, dtype=np.float64)
    order = np.argsort(horizon)
    width = horizon[order][-1] - horizon[order][0]
    if width <= 0:
        raise ValueError("at least two distinct horizon offsets are required")
    return float(np.trapezoid(rates[order], horizon[order]) / width)


@dataclass(frozen=True, slots=True)
class PhaseGateResult:
    passed: bool
    long_horizon_gain: float
    overhead_ratio: float
    short_horizon_regression: float
    confidence_interval: BootstrapInterval
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MechanismMetrics:
    infeasible_candidate_rate: float
    fallback_rate: float
    mean_commitment_duration: float
    failure_trigger_count: int
    trigger_precision: float
    failure_recovery_rate: float


def mechanism_metrics(
    traces: list[list[dict]],
    minimum_post_trigger_progress: float = 1e-4,
    recovery_window: int = 5,
) -> MechanismMetrics:
    """Aggregate auditable selection and event-trigger metrics from raw traces.

    A trigger is precise when goal cost improves within ``recovery_window``
    controller calls. A recovery additionally requires a successful commitment
    event or goal reach in that window. These definitions are fixed here so the
    paper cannot tune them after inspecting results.
    """

    failure_events = {"tube_violation", "stalled", "risk_increased"}
    recovery_events = {"subgoal_reached", "duration_exhausted", "goal_reached"}
    candidate_total = 0
    infeasible_total = 0
    action_total = 0
    fallback_total = 0
    durations: list[int] = []
    precise = 0
    recovered = 0
    trigger_count = 0
    for trace in traces:
        for index, item in enumerate(trace):
            candidates = item.get("candidates", [])
            candidate_total += len(candidates)
            infeasible_total += sum(
                not candidate.get("feasible", False) for candidate in candidates
            )
            action_total += 1
            fallback_total += int(item.get("fallback_used", False))
            if item.get("chosen_duration") is not None:
                durations.append(int(item["chosen_duration"]))
            trigger = item.get("trigger_event")
            trigger_value = trigger.value if hasattr(trigger, "value") else trigger
            if trigger_value not in failure_events:
                continue
            trigger_count += 1
            future = trace[index + 1 : index + 1 + recovery_window]
            if future:
                start_cost = float(item["goal_cost"])
                if min(float(next_item["goal_cost"]) for next_item in future) < (
                    start_cost - minimum_post_trigger_progress
                ):
                    precise += 1
                future_events = {
                    event.value if hasattr(event, "value") else event
                    for event in (
                        value
                        for next_item in future
                        for value in (
                            next_item.get("event"),
                            next_item.get("trigger_event"),
                        )
                    )
                    if event is not None
                }
                if future_events & recovery_events:
                    recovered += 1
    return MechanismMetrics(
        infeasible_candidate_rate=infeasible_total / max(candidate_total, 1),
        fallback_rate=fallback_total / max(action_total, 1),
        mean_commitment_duration=float(np.mean(durations)) if durations else float("nan"),
        failure_trigger_count=trigger_count,
        trigger_precision=precise / max(trigger_count, 1),
        failure_recovery_rate=recovered / max(trigger_count, 1),
    )


def evaluate_phase_gate(
    cape_long: np.ndarray,
    baseline_long: np.ndarray,
    cape_short: np.ndarray,
    baseline_short: np.ndarray,
    cape_time: float,
    baseline_time: float,
    seed: int = 0,
) -> PhaseGateResult:
    if baseline_time <= 0.0:
        raise ValueError("baseline planning time must be positive")
    interval = paired_bootstrap_difference(cape_long, baseline_long, seed=seed)
    gain = float(np.mean(cape_long) - np.mean(baseline_long))
    overhead = float(cape_time / baseline_time - 1.0)
    regression = float(np.mean(baseline_short) - np.mean(cape_short))
    reasons: list[str] = []
    if gain < 0.10:
        reasons.append("long_horizon_gain_below_10pp")
    if overhead > 0.25:
        reasons.append("planning_overhead_above_25pct")
    if regression > 0.03:
        reasons.append("short_horizon_regression_above_3pp")
    if interval.low <= 0.0:
        reasons.append("paired_bootstrap_interval_crosses_zero")
    return PhaseGateResult(
        passed=not reasons,
        long_horizon_gain=gain,
        overhead_ratio=overhead,
        short_horizon_regression=regression,
        confidence_interval=interval,
        reasons=tuple(reasons),
    )
