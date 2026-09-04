from __future__ import annotations

from dataclasses import asdict, dataclass

REQUIRED_M4_METHODS = frozenset({"b2", "b3", "b4", "b6"})


class FairnessViolation(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RunRecord:
    method: str
    seed: int
    encoder_sha256: str
    normalizer_sha256: str
    dataset_sha256: str
    split_sha256: str
    branch_trajectories: int
    optimizer_steps: int
    effective_batch_size: int
    curriculum: tuple[int, ...]
    predictor_parameters: int
    paired_loss: bool
    implementation_status: str = "native"
    precision: str = "float32"

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["curriculum"] = list(self.curriculum)
        return payload


def audit_run_matrix(
    records: list[RunRecord],
    *,
    parameter_tolerance: float = 0.05,
    require_complete: bool = True,
) -> dict[str, object]:
    if not records:
        raise FairnessViolation("run matrix is empty")
    if not 0 <= parameter_tolerance < 1:
        raise ValueError("parameter_tolerance must be in [0,1)")
    methods = {record.method for record in records}
    unknown = methods - REQUIRED_M4_METHODS
    if unknown:
        raise FairnessViolation(f"unsupported methods in matrix: {sorted(unknown)}")
    if require_complete and methods != REQUIRED_M4_METHODS:
        missing = REQUIRED_M4_METHODS - methods
        raise FairnessViolation(f"incomplete method matrix: missing {sorted(missing)}")

    expected_paired = {"b2": False, "b3": False, "b4": False, "b6": True}
    violations: list[str] = []
    for record in records:
        if record.paired_loss != expected_paired[record.method]:
            violations.append(f"{record.method}: paired-loss policy")
        if record.method == "b3" and record.implementation_status != (
            "paper_spec_reimplementation"
        ):
            violations.append("b3: missing paper-spec label")

    invariant_fields = (
        "encoder_sha256",
        "normalizer_sha256",
        "dataset_sha256",
        "split_sha256",
        "branch_trajectories",
        "optimizer_steps",
        "effective_batch_size",
        "curriculum",
        "precision",
    )
    for seed in sorted({record.seed for record in records}):
        group = [record for record in records if record.seed == seed]
        if require_complete and {record.method for record in group} != REQUIRED_M4_METHODS:
            violations.append(f"seed {seed}: incomplete methods")
            continue
        for field in invariant_fields:
            if len({getattr(record, field) for record in group}) != 1:
                violations.append(f"seed {seed}: mismatched {field}")
        counts = [record.predictor_parameters for record in group]
        if max(counts) / min(counts) - 1.0 > parameter_tolerance:
            violations.append(f"seed {seed}: predictor parameter mismatch")
    if violations:
        raise FairnessViolation("fairness audit failed: " + "; ".join(violations))
    return {
        "status": "pass",
        "methods": sorted(methods),
        "seeds": sorted({record.seed for record in records}),
        "parameter_tolerance": parameter_tolerance,
        "records": [record.as_dict() for record in records],
    }
