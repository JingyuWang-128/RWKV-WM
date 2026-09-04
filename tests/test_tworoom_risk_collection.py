import numpy as np

from cape_wm.tworoom_risk_collection import select_split_attempts


def test_attempt_sampler_uses_only_declared_split_and_is_reproducible():
    episodes = np.repeat(np.arange(4), 10)
    steps = np.tile(np.arange(10), 4)
    lengths = np.full(4, 10)
    first = select_split_attempts(
        episodes, steps, lengths, [1, 3], duration=5, count=4, seed=7
    )
    second = select_split_attempts(
        episodes, steps, lengths, [1, 3], duration=5, count=4, seed=7
    )
    assert first == second
    assert {item.episode_index for item in first} <= {1, 3}
    assert all(item.start_step <= 4 for item in first)
