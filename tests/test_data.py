import numpy as np

from cape_wm.data import (
    Trajectory,
    deterministic_group_split,
    iter_macro_segments,
    make_paired_test_list,
)


def test_trajectory_split_is_disjoint_and_deterministic():
    identifiers = [f"trajectory-{index}" for index in range(20)]
    first = deterministic_group_split(identifiers, seed=11)
    second = deterministic_group_split(identifiers, seed=11)
    assert first == second
    first.assert_disjoint()
    assert set(first.train + first.validation + first.calibration + first.test) == set(identifiers)


def test_macro_segments_never_cross_trajectory_boundaries():
    trajectory = Trajectory(
        trajectory_id="episode-1",
        observations=np.zeros((11, 2)),
        actions=np.zeros((10, 1)),
        latents=np.arange(22).reshape(11, 2),
    )
    segments = list(iter_macro_segments([trajectory], durations=(5, 10)))
    assert len(segments) == 7
    assert all(segment.trajectory_id == "episode-1" for segment in segments)
    assert segments[-1].duration == 10


def test_final_pair_list_is_fixed_and_balanced():
    lengths = {"trajectory-a": 20, "trajectory-b": 20}
    pairs = make_paired_test_list(
        "toy",
        lengths,
        tuple(lengths),
        offsets=(5, 10),
        seeds=(0, 1),
        episodes_per_condition=3,
        sampling_seed=9,
    )
    assert len(pairs) == 12
    assert len({pair.pair_id for pair in pairs}) == 12
    assert {pair.offset for pair in pairs} == {5, 10}
