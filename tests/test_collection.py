import numpy as np

from cape_wm.adapters.toy import PointImageEnv, PointImageWorldModel
from cape_wm.cem import CEMConfig
from cape_wm.collection import CalibrationAttemptSpec, collect_closed_loop_attempt
from cape_wm.generators import CEMLowLevelController


def test_closed_loop_collection_records_whole_sequence():
    environment = PointImageEnv()
    model = PointImageWorldModel()
    controller = CEMLowLevelController(
        model,
        horizon=5,
        cem=CEMConfig(samples=16, elites=4, iterations=2, seed=0),
    )
    goal = np.asarray([0.35, 0.35], dtype=np.float32)
    record = collect_closed_loop_attempt(
        environment,
        model,
        controller,
        CalibrationAttemptSpec(
            record_id="attempt-0",
            seed=0,
            subgoal_image=environment.render_state(goal),
            duration=5,
            reset_options={"start": np.asarray([0.1, 0.1]), "goal": goal},
        ),
        success_threshold=0.1,
        endpoint_error=lambda info, current, subgoal: info["state_distance"],
    )
    assert record.current_latent.shape == (2,)
    assert record.step_residuals.shape == (5,)
    assert np.all(record.step_residuals >= 0.0)
    assert np.isclose(record.observed_miss, np.linalg.norm(environment.state - goal))
