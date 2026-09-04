from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import CalibrationRecord
from .generators import CEMLowLevelController
from .interfaces import WorldModelAdapter, ensure_flat_latent

EndpointError = Callable[[dict[str, Any], np.ndarray, np.ndarray], float]
SuccessEvaluator = Callable[[dict[str, Any], float], bool]


@dataclass(frozen=True, slots=True)
class CalibrationAttemptSpec:
    record_id: str
    seed: int
    subgoal_image: Any
    duration: int
    reset_options: dict[str, Any] | None = None


def collect_closed_loop_attempt(
    environment,
    model: WorldModelAdapter,
    low_level: CEMLowLevelController,
    spec: CalibrationAttemptSpec,
    success_threshold: float,
    endpoint_error: EndpointError | None = None,
    success_evaluator: SuccessEvaluator | None = None,
) -> CalibrationRecord:
    """Execute frozen low-level MPC and retain the complete latent residual path.

    Optional evaluators may inspect true state in final environment info for
    labels. The model, controller, and risk-head inputs remain image latents.
    """

    reset_result = environment.reset(seed=spec.seed, options=spec.reset_options)
    observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    final_info = reset_result[1] if isinstance(reset_result, tuple) else {}
    initial = ensure_flat_latent(model.encode(observation))
    current = initial.copy()
    subgoal = ensure_flat_latent(model.encode(spec.subgoal_image))
    residuals: list[float] = []
    for elapsed in range(spec.duration):
        remaining = spec.duration - elapsed
        actions, predicted_path, _ = low_level.plan(current, subgoal, horizon=remaining)
        transition = environment.step(actions[0])
        observation = transition[0]
        final_info = transition[-1]
        current = ensure_flat_latent(model.encode(observation))
        residuals.append(float(np.linalg.norm(current - predicted_path[0])))
        if len(transition) == 5:
            _, _, terminated, truncated, _ = transition
            if terminated or truncated:
                break
        elif transition[2]:
            break
    observed_miss = (
        float(endpoint_error(final_info, current, subgoal))
        if endpoint_error is not None
        else float(model.goal_cost(current, subgoal))
    )
    success = (
        bool(success_evaluator(final_info, observed_miss))
        if success_evaluator is not None
        else observed_miss <= success_threshold
    )
    return CalibrationRecord(
        record_id=spec.record_id,
        current_latent=initial,
        subgoal_latent=subgoal,
        duration=spec.duration,
        observed_miss=observed_miss,
        success=success,
        step_residuals=np.asarray(residuals, dtype=np.float32),
    )
