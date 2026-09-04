import numpy as np

from cape_wm.adapters.toy import PointImageEnv
from cape_wm.cli import build_toy_planner
from cape_wm.evaluation import EpisodeSpec, evaluate_episode, read_jsonl, write_jsonl


def test_image_only_toy_reaches_goal(tmp_path):
    environment = PointImageEnv()
    start = np.asarray((0.1, 0.15), dtype=np.float32)
    goal = np.asarray((0.85, 0.8), dtype=np.float32)
    result = evaluate_episode(
        environment,
        build_toy_planner(seed=0),
        EpisodeSpec(
            pair_id="toy-test",
            seed=0,
            goal_image=environment.render_state(goal),
            reset_options={"start": start, "goal": goal},
            offset=40,
        ),
        max_steps=30,
    )
    assert result.success
    assert result.steps <= 30
    assert set(result.duration_histogram).issubset({"5", "10", "20", "40"})
    artifact = tmp_path / "toy.jsonl"
    write_jsonl(artifact, [result])
    restored = read_jsonl(artifact)[0]
    assert restored.success
    assert restored.planning_trace
