import numpy as np

from cape_wm.baseline import select_pairs
from cape_wm.baseline_matrix import summarize_matrix


def test_select_pairs_is_seeded_and_respects_goal_offset():
    episode_indices = np.repeat(np.arange(3), 6)
    step_indices = np.tile(np.arange(6), 3)
    lengths = np.full(3, 6)
    left, rows_left = select_pairs(
        episode_indices,
        step_indices,
        lengths,
        goal_offset=2,
        num_eval=4,
        seed=7,
    )
    right, rows_right = select_pairs(
        episode_indices,
        step_indices,
        lengths,
        goal_offset=2,
        num_eval=4,
        seed=7,
    )
    assert left == right
    assert np.array_equal(rows_left, rows_right)
    assert all(pair.start_step <= 3 for pair in left)
    assert len({pair.pair_id for pair in left}) == 4


def test_summarize_matrix_reports_long_horizon_and_auc():
    summaries = []
    episodes = []
    for offset, successes in ((25, 2), (50, 1), (75, 1), (100, 0)):
        for seed in (0, 1):
            summaries.append(
                {
                    "results": {"wall_time_seconds": 1.0, "peak_cuda_memory_bytes": 0},
                    "planning": {
                        "solver_seconds": 0.5,
                        "cost_calls": 2,
                        "candidate_sequences": 3,
                        "predicted_transitions": 4,
                    },
                }
            )
            for index in range(2):
                episodes.append(
                    {
                        "goal_offset": offset,
                        "evaluation_seed": seed,
                        "success": index < successes,
                        "final_state_error": float(offset),
                        "steps": offset,
                    }
                )
    result = summarize_matrix(summaries, episodes, [25, 50, 75, 100], [0, 1])
    assert result["total_tasks"] == 16
    assert result["long_horizon"]["pooled_success_rate"] == 0.25
    assert result["success_drop_25_to_100"] == 1.0
    assert 0.0 <= result["offset_success_auc"] <= 1.0
