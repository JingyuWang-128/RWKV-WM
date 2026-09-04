from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .models import (
    DurationConditionedMacroPredictor,
    ExecutabilityRiskHead,
    MacroActionEncoder,
    TrajectoryReachabilityMetric,
)


def load_payload(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("format") != "cape_wm_checkpoint_v1":
        raise ValueError("unsupported CAPE-WM checkpoint format")
    return payload


def load_macro_models(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[MacroActionEncoder, DurationConditionedMacroPredictor, dict[str, Any]]:
    payload = load_payload(path, map_location)
    config = payload["config"]
    encoder = MacroActionEncoder(
        action_dim=config["action_dim"],
        macro_dim=config["macro_dim"],
        max_duration=config["max_duration"],
    )
    predictor = DurationConditionedMacroPredictor(
        latent_dim=config["latent_dim"],
        macro_dim=config["macro_dim"],
        max_duration=config["max_duration"],
    )
    encoder.load_state_dict(payload["modules"]["action_encoder"])
    predictor.load_state_dict(payload["modules"]["macro_predictor"])
    return encoder.to(map_location).eval(), predictor.to(map_location).eval(), config


def load_risk_model(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[ExecutabilityRiskHead, dict[str, Any]]:
    payload = load_payload(path, map_location)
    config = payload["config"]
    model = ExecutabilityRiskHead(
        latent_dim=config["latent_dim"], max_duration=config["max_duration"]
    )
    model.load_state_dict(payload["modules"]["risk_head"])
    return model.to(map_location).eval(), config


def load_trm_model(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[TrajectoryReachabilityMetric, dict[str, Any]]:
    payload = load_payload(path, map_location)
    config = payload["config"]
    model = TrajectoryReachabilityMetric(
        latent_dim=config["latent_dim"], max_horizon=config["max_horizon"]
    )
    model.load_state_dict(payload["modules"]["trm"])
    return model.to(map_location).eval(), config
