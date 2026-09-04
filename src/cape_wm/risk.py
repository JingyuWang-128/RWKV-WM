from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .conformal import SplitConformalCalibrator
from .interfaces import RiskEstimator
from .types import Array

RiskPrediction = tuple[float, float, float | np.ndarray]


class RawRiskEstimator(RiskEstimator):
    """No-conformal ablation using the head's point estimates directly."""

    def __init__(
        self,
        predictor: Callable[[Array, Array, int], RiskPrediction],
        tube_multiplier: float = 1.0,
    ) -> None:
        self.predictor = predictor
        self.tube_multiplier = tube_multiplier

    def assess(
        self, current_latent: Array, subgoal: Array, duration: int
    ) -> tuple[float, float, float]:
        miss, probability, _ = self.predictor(current_latent, subgoal, duration)
        return float(miss), float(miss), float(np.clip(probability, 0.0, 1.0))

    def tube_threshold(
        self,
        current_latent: Array,
        subgoal: Array,
        duration: int,
        step: int,
    ) -> float:
        _, _, scale = self.predictor(current_latent, subgoal, duration)
        values = np.asarray(scale, dtype=np.float64).reshape(-1)
        if values.size == 0:
            raise ValueError("risk predictor returned an empty residual-scale sequence")
        return float(self.tube_multiplier * values[min(max(step - 1, 0), values.size - 1)])


class CalibratedRiskEstimator(RiskEstimator):
    """Apply split-conformal corrections to an arbitrary trained risk predictor."""

    def __init__(
        self,
        predictor: Callable[[Array, Array, int], RiskPrediction],
        calibrator: SplitConformalCalibrator,
    ) -> None:
        self.predictor = predictor
        self.calibrator = calibrator

    @property
    def alpha(self) -> float:
        return self.calibrator.alpha

    def assess(
        self, current_latent: Array, subgoal: Array, duration: int
    ) -> tuple[float, float, float]:
        predicted_miss, success_probability, _ = self.predictor(current_latent, subgoal, duration)
        upper = self.calibrator.miss_upper_bound(float(predicted_miss))
        return float(predicted_miss), upper, float(np.clip(success_probability, 0.0, 1.0))

    def tube_threshold(
        self,
        current_latent: Array,
        subgoal: Array,
        duration: int,
        step: int,
    ) -> float:
        _, _, predicted_scale = self.predictor(current_latent, subgoal, duration)
        scales = np.asarray(predicted_scale, dtype=np.float64).reshape(-1)
        if scales.size == 0:
            raise ValueError("risk predictor returned an empty residual-scale sequence")
        index = min(max(step - 1, 0), scales.size - 1)
        return float(self.calibrator.simultaneous_tube(float(scales[index])))


class DurationCalibratedRiskEstimator(RiskEstimator):
    """Apply a predeclared duration-conditional split-conformal artifact."""

    def __init__(self, predictor, calibrator, probability_thresholds=None) -> None:
        self.predictor = predictor
        self.calibrator = calibrator
        self.probability_thresholds = (
            dict(sorted((int(key), float(value)) for key, value in probability_thresholds.items()))
            if probability_thresholds is not None
            else None
        )

    @property
    def alpha(self) -> float:
        return self.calibrator.alpha

    def assess(
        self, current_latent: Array, subgoal: Array, duration: int
    ) -> tuple[float, float, float]:
        predicted_miss, success_probability, _ = self.predictor(
            current_latent, subgoal, duration
        )
        upper = self.calibrator.miss_upper_bound(float(predicted_miss), duration)
        return (
            float(predicted_miss),
            float(upper),
            float(np.clip(success_probability, 0.0, 1.0)),
        )

    def minimum_success_probability(self, duration: int, default: float = 0.0) -> float:
        if self.probability_thresholds is None:
            return float(default)
        durations = sorted(self.probability_thresholds)
        selected = next((item for item in durations if item >= duration), durations[-1])
        return self.probability_thresholds[selected]

    def tube_threshold(
        self,
        current_latent: Array,
        subgoal: Array,
        duration: int,
        step: int,
    ) -> float:
        _, _, predicted_scale = self.predictor(current_latent, subgoal, duration)
        scales = np.asarray(predicted_scale, dtype=np.float64).reshape(-1)
        if scales.size == 0:
            raise ValueError("risk predictor returned an empty residual-scale sequence")
        index = min(max(step - 1, 0), scales.size - 1)
        return float(
            self.calibrator.simultaneous_tube(float(scales[index]), duration)
        )


class ConstantRiskEstimator(RiskEstimator):
    """Transparent risk estimator for smoke tests and calibration ablations."""

    def __init__(
        self,
        predicted_miss: float = 0.0,
        miss_upper_bound: float = 0.0,
        success_probability: float = 1.0,
        tube_threshold: float = float("inf"),
    ) -> None:
        self.predicted_miss = predicted_miss
        self.upper = miss_upper_bound
        self.probability = success_probability
        self.threshold = tube_threshold

    def assess(
        self, current_latent: Array, subgoal: Array, duration: int
    ) -> tuple[float, float, float]:
        del current_latent, subgoal, duration
        return self.predicted_miss, self.upper, self.probability

    def tube_threshold(
        self,
        current_latent: Array,
        subgoal: Array,
        duration: int,
        step: int,
    ) -> float:
        del current_latent, subgoal, duration, step
        return self.threshold
