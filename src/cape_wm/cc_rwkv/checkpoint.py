from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .predictor import RWKV7WorldModelConfig, VanillaRWKV7WorldPredictor

CHECKPOINT_FORMAT = "cc_rwkv_checkpoint_v1"
M4_METHODS = frozenset({"b2", "b3", "b4", "b6"})


def _weights_only_safe(value: Any) -> Any:
    """Convert checkpoint metadata to weights_only-compatible primitives."""

    if value is None or isinstance(value, (str, int, float, bool, torch.Tensor)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _weights_only_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_weights_only_safe(item) for item in value]
    return str(value)


def save_b2_checkpoint(
    path: str | Path,
    predictor: VanillaRWKV7WorldPredictor,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    global_step: int = 0,
    best_metric: float = float("inf"),
    training_config: dict[str, Any] | None = None,
    encoder_provenance: dict[str, Any] | None = None,
    data_manifest_sha256: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "method": "b2",
        "model_config": predictor.config.as_dict(),
        "training_config": _weights_only_safe(training_config or {}),
        "modules": {"predictor": predictor.state_dict()},
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "encoder_provenance": _weights_only_safe(encoder_provenance or {}),
        "data_manifest_sha256": data_manifest_sha256,
        "history": _weights_only_safe(history or []),
        "rng_states": {"torch": torch.get_rng_state()},
    }
    torch.save(payload, destination)


def load_b2_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    expected_encoder_sha256: str | None = None,
    expected_data_manifest_sha256: str | None = None,
    allow_provenance_mismatch: bool = False,
) -> tuple[VanillaRWKV7WorldPredictor, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("method") != "b2":
        raise ValueError("checkpoint is not a CC-RWKV B2 checkpoint")
    actual_encoder = payload.get("encoder_provenance", {}).get("source_weights_sha256")
    mismatches: list[str] = []
    if expected_encoder_sha256 is not None and actual_encoder != expected_encoder_sha256:
        mismatches.append("encoder SHA256")
    actual_manifest = payload.get("data_manifest_sha256")
    if (
        expected_data_manifest_sha256 is not None
        and actual_manifest != expected_data_manifest_sha256
    ):
        mismatches.append("data manifest SHA256")
    if mismatches and not allow_provenance_mismatch:
        raise ValueError(f"checkpoint provenance mismatch: {', '.join(mismatches)}")
    payload["invalid_for_paper"] = bool(mismatches)
    predictor = VanillaRWKV7WorldPredictor(
        RWKV7WorldModelConfig(**payload["model_config"])
    )
    predictor.load_state_dict(payload["modules"]["predictor"])
    return predictor.to(map_location), payload


def save_m4_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    method: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    curriculum_state: dict[str, Any] | None = None,
    early_stopping_state: dict[str, Any] | None = None,
    epoch: int = 0,
    global_step: int = 0,
    best_metric: float = float("inf"),
    training_config: dict[str, Any] | None = None,
    data_provenance: dict[str, Any] | None = None,
    encoder_provenance: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    trainer_rng_state: torch.Tensor | None = None,
) -> None:
    if method not in M4_METHODS:
        raise ValueError(f"unsupported M4 method: {method}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictor = getattr(model, "predictor", model)
    model_config: dict[str, Any] = {"predictor": predictor.config.as_dict()}
    if method == "b3":
        model_config["world_head_hidden_dim"] = int(model.world_head_hidden_dim)
        model_config["implementation_status"] = str(model.implementation_status)
        model_config["source_arxiv"] = str(model.source_arxiv)
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "method": method,
        "model_config": model_config,
        "training_config": _weights_only_safe(training_config or {}),
        "modules": {"model": model.state_dict()},
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "curriculum": _weights_only_safe(curriculum_state or {}),
        "early_stopping": _weights_only_safe(early_stopping_state or {}),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "encoder_provenance": _weights_only_safe(encoder_provenance or {}),
        "data_provenance": _weights_only_safe(data_provenance or {}),
        "history": _weights_only_safe(history or []),
        "rng_states": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "trainer": trainer_rng_state,
        },
    }
    torch.save(payload, destination)


def _build_m4_model(method: str, config: dict[str, Any]) -> torch.nn.Module:
    predictor_config = config["predictor"]
    if method in {"b2", "b3"}:
        predictor = VanillaRWKV7WorldPredictor(
            RWKV7WorldModelConfig(**predictor_config)
        )
        if method == "b2":
            return predictor
        from .dwm import DWMOutputBaseline

        return DWMOutputBaseline(
            predictor,
            world_head_hidden_dim=int(config["world_head_hidden_dim"]),
        )
    from .counterfactual import (
        CounterfactualRWKV7Config,
        CounterfactualRWKV7WorldPredictor,
    )

    return CounterfactualRWKV7WorldPredictor(
        CounterfactualRWKV7Config(**predictor_config)
    )


def load_m4_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    expected_provenance: dict[str, Any] | None = None,
    allow_provenance_mismatch: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    method = payload.get("method")
    if payload.get("format") != CHECKPOINT_FORMAT or method not in M4_METHODS:
        raise ValueError("checkpoint is not a supported CC-RWKV M4 checkpoint")
    actual = {
        **payload.get("encoder_provenance", {}),
        **payload.get("data_provenance", {}),
    }
    mismatches = [
        name
        for name, expected in (expected_provenance or {}).items()
        if actual.get(name) != expected
    ]
    if mismatches and not allow_provenance_mismatch:
        raise ValueError(f"checkpoint provenance mismatch: {', '.join(sorted(mismatches))}")
    payload["invalid_for_paper"] = bool(mismatches)
    model = _build_m4_model(method, payload["model_config"])
    model.load_state_dict(payload["modules"]["model"])
    return model.to(map_location), payload


def restore_m4_training_state(
    payload: dict[str, Any],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> None:
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    rng = payload.get("rng_states", {})
    if "torch" in rng:
        torch.set_rng_state(rng["torch"].cpu())
    if torch.cuda.is_available() and rng.get("cuda"):
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
