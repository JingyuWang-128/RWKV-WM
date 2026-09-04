import numpy as np

from cape_wm.online import AdaptiveImaginationScheduler


def test_online_scheduler_truncates_and_recovers_with_hysteresis():
    scheduler = AdaptiveImaginationScheduler(disagreement_threshold=0.1)
    decision = scheduler.select(np.full(30, 0.2))
    assert decision.length == 5
    assert decision.truncated
    assert scheduler.maximum_length == 15
    scheduler.report_outcome(True)
    scheduler.report_outcome(True)
    assert scheduler.maximum_length == 30


def test_online_scheduler_selects_longest_safe_prefix():
    scheduler = AdaptiveImaginationScheduler(disagreement_threshold=0.1)
    disagreement = np.concatenate((np.full(15, 0.05), np.full(15, 0.2)))
    decision = scheduler.select(disagreement)
    assert decision.length == 15
    assert decision.truncated
