from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import torch

PER_STEP_CHECKPOINT_SCHEMA = "cc_rwkv_per_step_checkpoint_v3"


def _first_nonfinite_path(value: Any, path: str) -> str | None:
    """Return the first floating tensor path containing NaN/Inf."""

    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            return path
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            found = _first_nonfinite_path(item, f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _first_nonfinite_path(item, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def require_finite_checkpoint_state(
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    *,
    context: str,
) -> None:
    """Reject a checkpoint before non-finite state can be saved or restored."""

    for name, state in (("model", model_state), ("optimizer", optimizer_state)):
        found = _first_nonfinite_path(state, name)
        if found is not None:
            raise FloatingPointError(f"non-finite {context} checkpoint tensor: {found}")


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, torch.Tensor)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    raise TypeError(f"unsupported checkpoint metadata type: {type(value).__name__}")


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.unlink(missing_ok=True)
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_per_step_checkpoint(
    output: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    effect_threshold: float,
    history: list[dict[str, float]],
    batch_generator: torch.Generator,
    run_config: dict[str, Any],
    status: str = "running",
    keep_numbered: bool = False,
) -> tuple[Path, Path | None]:
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    if status not in {"running", "complete", "interrupted"}:
        raise ValueError(f"unsupported checkpoint status: {status}")
    output = Path(output)
    latest = output / "latest.pt"
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    require_finite_checkpoint_state(
        model_state, optimizer_state, context=f"step {step}"
    )
    payload = {
        "schema_version": PER_STEP_CHECKPOINT_SCHEMA,
        "method": str(run_config["method"]),
        "model": model_state,
        "optimizer": optimizer_state,
        "step": int(step),
        "effect_threshold": float(effect_threshold),
        "data": str(run_config["data"]),
        "history": _safe(history),
        "run_config": _safe(run_config),
        "status": status,
        "rng_states": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "batch_generator": batch_generator.get_state(),
        },
    }
    _atomic_torch_save(payload, latest)
    numbered: Path | None = None
    if keep_numbered:
        numbered = output / "checkpoints" / f"step_{step:08d}.pt"
        _link_or_copy(latest, numbered)
    return latest, numbered


def restore_per_step_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_generator: torch.Generator,
    *,
    expected_run_config: dict[str, Any],
    expected_effect_threshold: float,
    map_location: str | torch.device,
) -> dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("schema_version") != PER_STEP_CHECKPOINT_SCHEMA:
        raise ValueError(
            "checkpoint does not support exact per-step resume; expected "
            f"{PER_STEP_CHECKPOINT_SCHEMA}"
        )
    stored_config = payload.get("run_config", {})
    expected = _safe(expected_run_config)
    mismatches = [key for key, value in expected.items() if stored_config.get(key) != value]
    if mismatches:
        raise ValueError("resume configuration mismatch: " + ", ".join(sorted(mismatches)))
    stored_threshold = float(payload.get("effect_threshold", float("nan")))
    if not torch.isclose(
        torch.tensor(stored_threshold),
        torch.tensor(float(expected_effect_threshold)),
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError("resume effect threshold mismatch")
    require_finite_checkpoint_state(
        payload["model"], payload["optimizer"], context="resume"
    )
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    rng = payload.get("rng_states", {})
    if "torch" not in rng or "batch_generator" not in rng:
        raise ValueError("resume checkpoint is missing required RNG states")
    torch.set_rng_state(rng["torch"].cpu())
    if torch.cuda.is_available() and rng.get("cuda"):
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
    batch_generator.set_state(rng["batch_generator"].cpu())
    return payload
