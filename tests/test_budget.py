import numpy as np
import pytest

from cape_wm.adapters.toy import PointImageWorldModel
from cape_wm.cem import CEMConfig
from cape_wm.generators import (
    CEMLowLevelController,
    DirectRolloutGenerator,
    resize_cem_for_transition_budget,
)


def test_cem_population_is_reduced_to_transition_budget():
    original = CEMConfig(samples=256, elites=32, iterations=4)
    resized = resize_cem_for_transition_budget(original, 10, 4010)
    assert resized.samples == 100
    assert resized.elites == 32
    assert resized.samples * resized.iterations * 10 + 10 <= 4010


def test_impossible_budget_is_rejected_instead_of_silently_exceeded():
    with pytest.raises(ValueError, match="below the CEM minimum"):
        resize_cem_for_transition_budget(CEMConfig(samples=8, elites=4, iterations=2), 10, 5)


def test_multi_duration_generator_enforces_total_candidate_budget():
    model = PointImageWorldModel()
    controller = CEMLowLevelController(
        model, horizon=10, cem=CEMConfig(samples=32, elites=4, iterations=2)
    )
    generator = DirectRolloutGenerator(model, controller, model_transition_budget=1000)
    candidates = generator.propose(
        current_latent=np.asarray([0.1, 0.1], dtype=np.float32),
        goal_latent=np.asarray([0.8, 0.8], dtype=np.float32),
        durations=(5, 10),
        max_duration=10,
    )
    assert sum(candidate.planning_cost for candidate in candidates) <= 900
