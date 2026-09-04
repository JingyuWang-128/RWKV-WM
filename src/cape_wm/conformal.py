from __future__ import annotations

from dataclasses import replace

import numpy as np

from .types import CalibrationArtifact


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile using the conservative ``higher`` rule."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("at least one calibration score is required")
    if not np.all(np.isfinite(values)):
        raise ValueError("calibration scores must be finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")

    rank = int(np.ceil((values.size + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), values.size)
    return float(np.partition(values, rank - 1)[rank - 1])


class SplitConformalCalibrator:
    """Calibrate endpoint miss and whole-sequence prediction-tube residuals."""

    def __init__(self, alpha: float = 0.1, eps: float = 1e-8) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one")
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.artifact: CalibrationArtifact | None = None

    def fit(
        self,
        predicted_miss: np.ndarray,
        observed_miss: np.ndarray,
        predicted_scales: list[np.ndarray] | np.ndarray,
        observed_residuals: list[np.ndarray] | np.ndarray,
    ) -> CalibrationArtifact:
        pred = np.asarray(predicted_miss, dtype=np.float64).reshape(-1)
        obs = np.asarray(observed_miss, dtype=np.float64).reshape(-1)
        if pred.shape != obs.shape:
            raise ValueError("predicted and observed miss arrays must have identical shapes")
        miss_scores = obs - pred

        scale_sequences = self._as_sequences(predicted_scales)
        residual_sequences = self._as_sequences(observed_residuals)
        if len(scale_sequences) != len(residual_sequences):
            raise ValueError("scale and residual sequence counts must match")
        sequence_scores: list[float] = []
        for scale, residual in zip(scale_sequences, residual_sequences, strict=True):
            if scale.shape != residual.shape:
                raise ValueError("each residual sequence must match its scale sequence")
            if scale.size == 0:
                raise ValueError("calibration residual sequences must be non-empty")
            sequence_scores.append(float(np.max(residual / np.maximum(scale, self.eps))))

        self.artifact = CalibrationArtifact(
            alpha=self.alpha,
            miss_quantile=conformal_quantile(miss_scores, self.alpha),
            tube_quantile=conformal_quantile(np.asarray(sequence_scores), self.alpha),
            n_miss=miss_scores.size,
            n_tube=len(sequence_scores),
        )
        return replace(self.artifact)

    def miss_upper_bound(self, predicted_miss: float) -> float:
        artifact = self._require_fitted()
        return float(max(0.0, predicted_miss + artifact.miss_quantile))

    def simultaneous_tube(self, predicted_scale: float | np.ndarray) -> float | np.ndarray:
        artifact = self._require_fitted()
        result = artifact.tube_quantile * np.maximum(predicted_scale, self.eps)
        if np.ndim(result) == 0:
            return float(result)
        return result

    def _require_fitted(self) -> CalibrationArtifact:
        if self.artifact is None:
            raise RuntimeError("calibrator has not been fitted")
        return self.artifact

    @classmethod
    def from_artifact(cls, artifact: CalibrationArtifact) -> SplitConformalCalibrator:
        calibrator = cls(alpha=artifact.alpha)
        calibrator.artifact = replace(artifact)
        return calibrator

    @staticmethod
    def _as_sequences(values: list[np.ndarray] | np.ndarray) -> list[np.ndarray]:
        if isinstance(values, np.ndarray) and values.ndim == 2:
            return [np.asarray(row, dtype=np.float64) for row in values]
        return [np.asarray(row, dtype=np.float64).reshape(-1) for row in values]
