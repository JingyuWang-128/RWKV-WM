from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .protocol import file_sha256


class FrozenLeWMEncoder(nn.Module):
    """Frozen official LeWM encoder/projector with reproducible action scaling."""

    def __init__(
        self,
        encoder: nn.Module,
        projector: nn.Module,
        *,
        action_mean: Tensor,
        action_scale: Tensor,
        source_weights: str | Path | None = None,
    ) -> None:
        super().__init__()
        if action_mean.ndim != 1 or action_scale.shape != action_mean.shape:
            raise ValueError("action_mean and action_scale must be matching vectors")
        if not torch.isfinite(action_mean).all() or not torch.isfinite(action_scale).all():
            raise ValueError("action normalization statistics must be finite")
        if torch.any(action_scale <= 0):
            raise ValueError("action_scale must be positive")
        self.encoder = encoder.eval().requires_grad_(False)
        self.projector = projector.eval().requires_grad_(False)
        self.register_buffer("action_mean", action_mean.detach().float().clone())
        self.register_buffer("action_scale", action_scale.detach().float().clone())
        self.source_weights = str(source_weights) if source_weights is not None else None
        self.source_weights_sha256 = (
            file_sha256(source_weights) if source_weights is not None else None
        )

    @classmethod
    def from_official_model(
        cls,
        model: nn.Module,
        *,
        action_mean: Tensor,
        action_scale: Tensor,
        source_weights: str | Path | None = None,
    ) -> FrozenLeWMEncoder:
        missing = [name for name in ("encoder", "projector") if not hasattr(model, name)]
        if missing:
            raise TypeError(f"official LeWM model is missing modules: {missing}")
        return cls(
            model.encoder,
            model.projector,
            action_mean=action_mean,
            action_scale=action_scale,
            source_weights=source_weights,
        )

    @torch.inference_mode()
    def encode_images(self, images: Tensor) -> Tensor:
        """Encode preprocessed images shaped [B,C,H,W] or [B,T,C,H,W]."""

        if images.ndim == 4:
            images = images[:, None]
        if images.ndim != 5:
            raise ValueError("images must have shape [B,C,H,W] or [B,T,C,H,W]")
        batch, steps = images.shape[:2]
        flat = images.reshape(batch * steps, *images.shape[2:]).float()
        output = self.encoder(flat, interpolate_pos_encoding=True)
        if not hasattr(output, "last_hidden_state"):
            raise TypeError("official encoder output must expose last_hidden_state")
        latent = self.projector(output.last_hidden_state[:, 0])
        return latent.reshape(batch, steps, -1)

    def normalize_action(self, raw_action_block: Tensor) -> Tensor:
        primitive_dim = self.action_mean.numel()
        if raw_action_block.shape[-1] % primitive_dim:
            raise ValueError("action block width must be divisible by primitive action dimension")
        block = raw_action_block.shape[-1] // primitive_dim
        shaped = raw_action_block.float().reshape(
            *raw_action_block.shape[:-1], block, primitive_dim
        )
        normalized = (shaped - self.action_mean) / self.action_scale
        return normalized.reshape_as(raw_action_block)

    @staticmethod
    def decode_prediction(predictor_hidden: Tensor) -> Tensor:
        """M1 predictor emits the frozen LeWM latent space directly."""

        return predictor_hidden

    def train(self, mode: bool = True) -> FrozenLeWMEncoder:
        # Keep the frozen feature path in eval mode even while the predictor trains.
        super().train(False)
        return self

    def provenance(self) -> dict[str, Any]:
        return {
            "source_weights": self.source_weights,
            "source_weights_sha256": self.source_weights_sha256,
            "action_mean": self.action_mean.detach().cpu().tolist(),
            "action_scale": self.action_scale.detach().cpu().tolist(),
            "encoder_frozen": True,
            "projector_frozen": True,
        }
