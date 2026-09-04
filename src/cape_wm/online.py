from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ImaginationDecision:
    length: int
    truncated: bool
    maximum_disagreement: float


class AdaptiveImaginationScheduler:
    """Phase-D bridge for Dreamer/R2I ensemble-disagreement horizons."""

    def __init__(
        self,
        lengths: tuple[int, ...] = (5, 15, 30),
        disagreement_threshold: float = 0.1,
        successes_to_expand: int = 2,
    ) -> None:
        if tuple(sorted(set(lengths))) != lengths or not lengths:
            raise ValueError("imagination lengths must be sorted and unique")
        self.lengths = lengths
        self.threshold = disagreement_threshold
        self.successes_to_expand = successes_to_expand
        self.reset()

    def reset(self) -> None:
        self._max_index = len(self.lengths) - 1
        self._successes = 0

    def select(self, per_step_ensemble_disagreement: np.ndarray) -> ImaginationDecision:
        disagreement = np.asarray(per_step_ensemble_disagreement, dtype=np.float64).reshape(-1)
        if disagreement.size < self.lengths[0]:
            raise ValueError("disagreement trace is shorter than the minimum imagination")
        allowed = self.lengths[: self._max_index + 1]
        safe = [
            length
            for length in allowed
            if disagreement.size >= length
            and float(np.max(disagreement[:length])) <= self.threshold
        ]
        length = max(safe) if safe else self.lengths[0]
        maximum = float(np.max(disagreement[:length]))
        truncated = not safe or length < allowed[-1]
        if truncated:
            self.report_outcome(False)
        return ImaginationDecision(length, truncated, maximum)

    def report_outcome(self, successful: bool) -> None:
        if not successful:
            self._max_index = max(0, self._max_index - 1)
            self._successes = 0
            return
        self._successes += 1
        if self._successes >= self.successes_to_expand:
            self._max_index = min(len(self.lengths) - 1, self._max_index + 1)
            self._successes = 0

    @property
    def maximum_length(self) -> int:
        return self.lengths[self._max_index]
