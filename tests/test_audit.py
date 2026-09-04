from cape_wm.audit import (
    AuditedCandidate,
    CandidateAuditRecord,
    classify_same_candidate_failure,
)


def test_same_candidate_audit_distinguishes_registered_failure_sources():
    unsafe = AuditedCandidate(False, 0.5)
    viable = AuditedCandidate(True, 0.5)
    assert (
        classify_same_candidate_failure(CandidateAuditRecord((unsafe,), 0, False))
        == "subgoal_generation"
    )
    assert (
        classify_same_candidate_failure(CandidateAuditRecord((unsafe, viable), 0, False))
        == "risk_ranking"
    )
    assert (
        classify_same_candidate_failure(CandidateAuditRecord((viable,), 0, False))
        == "low_level_execution"
    )
