from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .conformal import SplitConformalCalibrator
from .data import CalibrationRecord
from .types import CalibrationArtifact


class DurationConditionalCalibrator:
    """Mondrian split conformal with one predeclared stratum per duration."""

    method = "split_conformal_duration_conditional_sequence_max_v1"

    def __init__(self, calibrators: dict[int, SplitConformalCalibrator]) -> None:
        if not calibrators:
            raise ValueError("at least one duration calibrator is required")
        alphas = {item.alpha for item in calibrators.values()}
        if len(alphas) != 1:
            raise ValueError("duration calibrators must use one alpha")
        self.calibrators = dict(sorted(calibrators.items()))
        self.alpha = next(iter(alphas))

    def _for(self, duration: int) -> SplitConformalCalibrator:
        try:
            return self.calibrators[int(duration)]
        except KeyError as error:
            raise ValueError(f"duration {duration} has no calibration stratum") from error

    def miss_upper_bound(self, predicted_miss: float, duration: int) -> float:
        return self._for(duration).miss_upper_bound(predicted_miss)

    def simultaneous_tube(
        self, predicted_scale: float | np.ndarray, duration: int
    ) -> float | np.ndarray:
        return self._for(duration).simultaneous_tube(predicted_scale)

    def as_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "method": self.method,
            "by_duration": {
                str(duration): calibrator._require_fitted().as_dict()
                for duration, calibrator in self.calibrators.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DurationConditionalCalibrator":
        if payload.get("method") != cls.method:
            raise ValueError("unsupported duration-conditional calibration artifact")
        calibrators = {
            int(duration): SplitConformalCalibrator.from_artifact(CalibrationArtifact(**artifact))
            for duration, artifact in payload["by_duration"].items()
        }
        result = cls(calibrators)
        if not np.isclose(result.alpha, float(payload["alpha"])):
            raise ValueError("duration calibration alpha mismatch")
        return result


def fit_from_records(
    records: list[CalibrationRecord],
    risk_predictor,
    alpha: float = 0.1,
) -> SplitConformalCalibrator:
    """Fit endpoint and sequence-level calibration from closed-loop attempts."""

    if not records:
        raise ValueError("at least one closed-loop calibration record is required")
    predicted_miss: list[float] = []
    observed_miss: list[float] = []
    predicted_scales: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    for record in records:
        miss, _, scale = risk_predictor(
            record.current_latent, record.subgoal_latent, record.duration
        )
        predicted_miss.append(float(miss))
        observed_miss.append(float(record.observed_miss))
        scale_sequence = np.asarray(scale, dtype=np.float64).reshape(-1)
        if scale_sequence.size == 1:
            scale_sequence = np.full_like(
                record.step_residuals, scale_sequence.item(), dtype=np.float64
            )
        elif scale_sequence.size < len(record.step_residuals):
            raise ValueError("risk predictor returned fewer scales than executed steps")
        else:
            scale_sequence = scale_sequence[: len(record.step_residuals)]
        predicted_scales.append(scale_sequence)
        residuals.append(np.asarray(record.step_residuals, dtype=np.float64))
    calibrator = SplitConformalCalibrator(alpha=alpha)
    calibrator.fit(
        np.asarray(predicted_miss),
        np.asarray(observed_miss),
        predicted_scales,
        residuals,
    )
    return calibrator


def fit_duration_conditional_from_records(
    records: list[CalibrationRecord],
    risk_predictor,
    alpha: float = 0.1,
) -> DurationConditionalCalibrator:
    """Fit valid split-conformal strata for the registered duration set."""

    if not records:
        raise ValueError("at least one closed-loop calibration record is required")
    calibrators = {}
    for duration in sorted({record.duration for record in records}):
        group = [record for record in records if record.duration == duration]
        calibrators[duration] = fit_from_records(group, risk_predictor, alpha)
    return DurationConditionalCalibrator(calibrators)


def save_calibration_artifact(path: str | Path, artifact: CalibrationArtifact) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact.as_dict(), indent=2) + "\n")


def load_calibration_artifact(path: str | Path) -> CalibrationArtifact:
    payload = json.loads(Path(path).read_text())
    return CalibrationArtifact(**payload)


def load_calibrator(path: str | Path) -> SplitConformalCalibrator:
    return SplitConformalCalibrator.from_artifact(load_calibration_artifact(path))


def save_duration_calibrator(
    path: str | Path, calibrator: DurationConditionalCalibrator
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(calibrator.as_dict(), indent=2) + "\n")


def load_duration_calibrator(path: str | Path) -> DurationConditionalCalibrator:
    return DurationConditionalCalibrator.from_dict(json.loads(Path(path).read_text()))
