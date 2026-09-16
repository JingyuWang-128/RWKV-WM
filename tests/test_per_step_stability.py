import math

import pytest

from cape_wm.cc_rwkv.stability import gradient_window_report, learning_rate_factor


def test_stability_gate_rejects_widespread_clipping():
    report = gradient_window_report([2.0] * 200)
    assert report["passed"] is False
    assert report["clip_fraction"] == 1


def test_stability_gate_rejects_isolated_extreme_gradient():
    report = gradient_window_report([0.1] * 199 + [1e6])
    assert report["clip_fraction"] < 0.2
    assert report["passed"] is False


def test_stability_gate_accepts_small_gradients_with_occasional_mild_clipping():
    report = gradient_window_report([0.4] * 180 + [1.1] * 20)
    assert report["passed"] is True
    assert report["clip_norm"] == 1


@pytest.mark.parametrize("bad", [math.inf, math.nan, -1.0])
def test_stability_gate_rejects_invalid_norms(bad):
    with pytest.raises(FloatingPointError):
        gradient_window_report([bad])


def test_warmup_and_decay_schedule_is_step_based_and_resumable():
    kwargs = dict(max_steps=1000, warmup_steps=100, min_ratio=0.1)
    assert learning_rate_factor(1, **kwargs) == pytest.approx(0.01)
    assert learning_rate_factor(100, **kwargs) == 1
    assert learning_rate_factor(1000, **kwargs) == pytest.approx(0.1)
    assert 0.1 < learning_rate_factor(500, **kwargs) < 1
    assert learning_rate_factor(500, **kwargs) == learning_rate_factor(500, **kwargs)
