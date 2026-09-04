from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AuditedCandidate:
    oracle_executable: bool
    oracle_goal_progress: float


@dataclass(frozen=True, slots=True)
class CandidateAuditRecord:
    candidates: tuple[AuditedCandidate, ...]
    selected_index: int | None
    execution_succeeded: bool


def classify_same_candidate_failure(record: CandidateAuditRecord) -> str:
    """Attribute failure without regenerating or changing the candidate set."""

    viable = [
        candidate.oracle_executable and candidate.oracle_goal_progress > 0.0
        for candidate in record.candidates
    ]
    if not any(viable):
        return "subgoal_generation"
    if record.selected_index is None or not 0 <= record.selected_index < len(viable):
        return "risk_ranking"
    if not viable[record.selected_index]:
        return "risk_ranking"
    if not record.execution_succeeded:
        return "low_level_execution"
    return "success"


def summarize_same_candidate_audit(records: list[CandidateAuditRecord]) -> dict[str, int]:
    return dict(Counter(classify_same_candidate_failure(record) for record in records))
