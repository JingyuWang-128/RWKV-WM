import numpy as np

from cape_wm.stats import (
    binary_auroc,
    brier_score,
    evaluate_phase_gate,
    expected_calibration_error,
    paired_mcnemar_exact,
)


def test_probability_metrics_have_expected_extremes():
    targets = np.asarray([0, 0, 1, 1])
    perfect = np.asarray([0.0, 0.0, 1.0, 1.0])
    assert brier_score(perfect, targets) == 0.0
    assert expected_calibration_error(perfect, targets) == 0.0
    assert binary_auroc(perfect, targets) == 1.0


def test_phase_gate_applies_all_preregistered_thresholds():
    baseline_long = np.zeros(100)
    cape_long = np.r_[np.ones(20), np.zeros(80)]
    baseline_short = np.ones(100)
    cape_short = np.ones(100)
    result = evaluate_phase_gate(
        cape_long,
        baseline_long,
        cape_short,
        baseline_short,
        cape_time=1.2,
        baseline_time=1.0,
    )
    assert result.passed
    assert result.long_horizon_gain == 0.2
    assert result.confidence_interval.low > 0.0
    assert paired_mcnemar_exact(cape_long, baseline_long) < 0.001
