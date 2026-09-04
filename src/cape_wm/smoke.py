from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import load_config
from .device import resolve_device
from .models import (
    DurationConditionedMacroPredictor,
    ExecutabilityRiskHead,
    MacroActionEncoder,
    TrajectoryReachabilityMetric,
)
from .training import train_macro_epoch, train_risk_epoch, train_trm_epoch


def run_training_smoke(requested_device: str = "auto") -> dict[str, float | str]:
    """Run one tiny synthetic optimization step through every trainable module."""

    device = resolve_device(requested_device)
    torch.manual_seed(0)

    action_encoder = MacroActionEncoder(
        action_dim=2,
        macro_dim=4,
        model_dim=16,
        depth=1,
        heads=2,
        max_duration=5,
    ).to(device)
    macro_predictor = DurationConditionedMacroPredictor(
        latent_dim=6,
        macro_dim=4,
        hidden_dim=32,
        depth=2,
        max_duration=5,
    ).to(device)
    macro_optimizer = torch.optim.AdamW(
        list(action_encoder.parameters()) + list(macro_predictor.parameters()),
        lr=1e-3,
    )
    macro_batch = {
        "current": torch.randn(2, 6),
        "target": torch.randn(2, 6),
        "actions": torch.randn(2, 5, 2),
        "duration": torch.tensor([3, 5]),
    }
    macro_metrics = train_macro_epoch(
        action_encoder,
        macro_predictor,
        [macro_batch],
        macro_optimizer,
        device,
    )

    risk = ExecutabilityRiskHead(latent_dim=6, hidden_dim=32, depth=2, max_duration=5).to(device)
    risk_optimizer = torch.optim.AdamW(risk.parameters(), lr=1e-3)
    duration = torch.tensor([3, 5])
    residual_mask = torch.arange(5)[None] < duration[:, None]
    risk_batch = {
        "current": torch.randn(2, 6),
        "subgoal": torch.randn(2, 6),
        "duration": duration,
        "observed_miss": torch.rand(2),
        "success": torch.tensor([1.0, 0.0]),
        "residual_scale": torch.rand(2, 5).clamp_min(1e-3),
        "residual_mask": residual_mask,
    }
    risk_loss = train_risk_epoch(risk, [risk_batch], risk_optimizer, device)

    trm = TrajectoryReachabilityMetric(latent_dim=6, hidden_dim=32, max_horizon=10).to(device)
    trm_optimizer = torch.optim.AdamW(trm.parameters(), lr=1e-3)
    trm_batch = {
        "source": torch.randn(2, 6),
        "goal": torch.randn(2, 6),
        "horizon": torch.tensor([4, 8]),
        "temporal_distance": torch.tensor([0.4, 0.8]),
        "trajectory_index": torch.tensor([0, 1]),
    }
    trm_loss = train_trm_epoch(trm, [trm_batch], trm_optimizer, device)
    return {
        "device": str(device),
        "macro_loss": macro_metrics["loss"],
        "risk_loss": risk_loss,
        "trm_loss": trm_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a portable CAPE-WM training smoke test")
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/runtime.yaml"))
    parser.add_argument("--device")
    args = parser.parse_args()
    configured_device = load_config(args.runtime_config)["runtime"]["device"]
    print(json.dumps(run_training_smoke(args.device or configured_device), indent=2))


if __name__ == "__main__":
    main()
