from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .planner import CAPEPlanner
from .types import ExecutionEvent


class GoalEnvironment(Protocol):
    def reset(self, *, seed: int, options: dict[str, Any] | None = None): ...

    def step(self, action: np.ndarray): ...


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    pair_id: str
    seed: int
    goal_image: Any
    reset_options: dict[str, Any] | None = None
    offset: int | None = None


@dataclass(slots=True)
class EpisodeResult:
    pair_id: str
    seed: int
    offset: int | None
    success: bool
    steps: int
    wall_time_seconds: float
    planning_time_seconds: float
    final_goal_cost: float
    planning_events: dict[str, int]
    duration_histogram: dict[str, int]
    fallback_count: int
    model_calls: float
    environment_metrics: dict[str, float] = field(default_factory=dict)
    planning_trace: list[dict[str, Any]] = field(default_factory=list)
    peak_cuda_memory_bytes: int = 0


def evaluate_episode(
    environment: GoalEnvironment,
    planner: CAPEPlanner,
    spec: EpisodeSpec,
    max_steps: int,
) -> EpisodeResult:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    planner.reset()
    reset_result = environment.reset(seed=spec.seed, options=spec.reset_options)
    observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    event_counts: dict[str, int] = {}
    durations: dict[str, int] = {}
    fallback_count = 0
    model_calls = 0.0
    final_cost = float("inf")
    start = time.perf_counter()
    success = False
    environment_metrics: dict[str, float] = {}
    planning_trace: list[dict[str, Any]] = []
    planning_time = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            for device_index in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(device_index)
    except ImportError:
        torch = None  # type: ignore[assignment]

    for _step in range(1, max_steps + 1):
        planning_start = time.perf_counter()
        action, diagnostics = planner.plan(observation, spec.goal_image)
        planning_time += time.perf_counter() - planning_start
        event_counts[diagnostics.event.value] = event_counts.get(diagnostics.event.value, 0) + 1
        if diagnostics.trigger_event is not None and diagnostics.trigger_event != diagnostics.event:
            trigger_key = f"trigger:{diagnostics.trigger_event.value}"
            event_counts[trigger_key] = event_counts.get(trigger_key, 0) + 1
        if diagnostics.chosen_duration is not None:
            key = str(diagnostics.chosen_duration)
            durations[key] = durations.get(key, 0) + 1
        fallback_count += int(diagnostics.fallback_used)
        model_calls += diagnostics.step_planning_cost
        final_cost = diagnostics.goal_cost
        planning_trace.append(diagnostics.as_dict())
        if diagnostics.event == ExecutionEvent.GOAL_REACHED:
            success = True
            break

        transition = environment.step(action)
        if len(transition) == 5:
            observation, _, terminated, truncated, info = transition
        else:
            observation, _, done, info = transition
            terminated, truncated = done, False
        environment_metrics.update(
            {key: float(value) for key, value in info.items() if np.isscalar(value)}
        )
        success = bool(info.get("success", False))
        if success or terminated or truncated:
            break
    elapsed = time.perf_counter() - start
    peak_memory = 0
    if torch is not None and torch.cuda.is_available():
        peak_memory = max(
            torch.cuda.max_memory_allocated(index) for index in range(torch.cuda.device_count())
        )
    return EpisodeResult(
        pair_id=spec.pair_id,
        seed=spec.seed,
        offset=spec.offset,
        success=success,
        steps=_step,
        wall_time_seconds=elapsed,
        planning_time_seconds=planning_time,
        final_goal_cost=final_cost,
        planning_events=event_counts,
        duration_histogram=durations,
        fallback_count=fallback_count,
        model_calls=model_calls,
        environment_metrics=environment_metrics,
        planning_trace=planning_trace,
        peak_cuda_memory_bytes=peak_memory,
    )


def write_jsonl(path: str | Path, results: list[EpisodeResult]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        for result in results:
            handle.write(json.dumps(_jsonable(asdict(result)), ensure_ascii=False) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def read_jsonl(path: str | Path) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                payload.setdefault("planning_time_seconds", payload.get("wall_time_seconds", 0.0))
                payload.setdefault("planning_trace", [])
                payload.setdefault("peak_cuda_memory_bytes", 0)
                results.append(EpisodeResult(**payload))
    return results
