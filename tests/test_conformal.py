import numpy as np

from cape_wm.calibration import (
    DurationConditionalCalibrator,
    fit_duration_conditional_from_records,
)
from cape_wm.conformal import SplitConformalCalibrator, conformal_quantile
from cape_wm.data import CalibrationRecord
from cape_wm.risk import DurationCalibratedRiskEstimator


def test_finite_sample_quantile_uses_conservative_rank():
    scores = np.arange(10, dtype=np.float64)
    assert conformal_quantile(scores, alpha=0.1) == 9.0
    assert conformal_quantile(scores, alpha=0.5) == 5.0


def test_sequence_calibration_uses_maximum_normalized_residual():
    calibrator = SplitConformalCalibrator(alpha=0.25)
    artifact = calibrator.fit(
        predicted_miss=np.zeros(4),
        observed_miss=np.asarray([0.1, 0.2, 0.3, 0.4]),
        predicted_scales=np.ones((4, 3)),
        observed_residuals=np.asarray(
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.1, 0.1, 0.5], [0.6, 0.1, 0.1]]
        ),
    )
    assert artifact.miss_quantile == 0.4
    assert artifact.tube_quantile == 0.6
    assert np.isclose(calibrator.miss_upper_bound(0.2), 0.6)
    restored = SplitConformalCalibrator.from_artifact(artifact)
    assert np.isclose(restored.simultaneous_tube(2.0), 1.2)


def test_duration_conditional_calibration_keeps_registered_strata_separate():
    records = [
        CalibrationRecord(
            record_id=str(index),
            current_latent=np.asarray([0.0]),
            subgoal_latent=np.asarray([1.0]),
            duration=duration,
            observed_miss=miss,
            success=True,
            step_residuals=np.asarray([miss]),
        )
        for index, (duration, miss) in enumerate(
            [(5, 1.0), (5, 2.0), (40, 10.0), (40, 20.0)]
        )
    ]
    calibrator = fit_duration_conditional_from_records(
        records,
        lambda current, subgoal, duration: (0.0, 1.0, np.ones(duration)),
        alpha=0.5,
    )
    assert calibrator.miss_upper_bound(0.0, 5) == 2.0
    assert calibrator.miss_upper_bound(0.0, 40) == 20.0
    restored = DurationConditionalCalibrator.from_dict(calibrator.as_dict())
    assert restored.simultaneous_tube(1.0, 5) == 2.0


def test_duration_risk_returns_assessment_and_uses_ceiling_probability_threshold():
    class Predictor:
        def __call__(self, current, subgoal, duration):
            del current, subgoal, duration
            return 2.0, 0.8, np.ones(4, dtype=np.float32)

    class Calibrator:
        alpha = 0.1

        def miss_upper_bound(self, miss, duration):
            return miss + duration

        def simultaneous_tube(self, scale, duration):
            return scale + duration

    risk = DurationCalibratedRiskEstimator(
        Predictor(), Calibrator(), probability_thresholds={5: 0.9, 10: 0.7}
    )

    assert risk.assess(np.zeros(1), np.ones(1), 5) == (2.0, 7.0, 0.8)
    assert risk.minimum_success_probability(6) == 0.7
