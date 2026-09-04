from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np

Array = np.ndarray


class ExecutionEvent(str, Enum):
    """Reason that the active temporal commitment changed."""

    CONTINUE = "continue"
    NEW_PLAN = "new_plan"
    GOAL_REACHED = "goal_reached"
    SUBGOAL_REACHED = "subgoal_reached"
    TUBE_VIOLATION = "tube_violation"
    STALLED = "stalled"
    RISK_INCREASED = "risk_increased"
    DURATION_EXHAUSTED = "duration_exhausted"
    FALLBACK = "fallback"


@dataclass(slots=True)
class CandidatePlan:
    """A duration-specific high-level proposal and its low-level execution plan."""

    duration: int
    subgoal: Array
    actions: Array
    predicted_path: Array
    current_goal_cost: float
    subgoal_goal_cost: float
    predicted_miss: float
    miss_upper_bound: float
    success_probability: float
    planning_cost: float
    feasible: bool = True
    score: float = float("-inf")
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def progress(self) -> float:
        return self.current_goal_cost - self.subgoal_goal_cost


@dataclass(slots=True)
class PlanDiagnostics:
    """Serializable diagnostics emitted at each call to the planner."""

    event: ExecutionEvent
    trigger_event: ExecutionEvent | None
    chosen_duration: int | None
    active_step: int
    max_duration: int
    candidate_count: int
    feasible_count: int
    goal_cost: float
    step_planning_cost: float = 0.0
    tube_residual: float | None = None
    tube_threshold: float | None = None
    fallback_used: bool = False
    candidates: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CalibrationArtifact:
    """Versioned calibration values saved independently of trainable weights."""

    alpha: float
    miss_quantile: float
    tube_quantile: float
    n_miss: int
    n_tube: int
    method: str = "split_conformal_sequence_max_v1"

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)
