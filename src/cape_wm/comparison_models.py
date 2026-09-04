from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PairwiseReachabilityMetric(nn.Module):
    """TRM pair head from arXiv:2605.22164.

    The model intentionally has no planning-horizon input. Horizon matching is
    implemented by the temporal-pair sampler, as specified by the paper.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(4 * latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, source: Tensor, goal: Tensor) -> Tensor:
        delta = source - goal
        features = torch.cat((source, goal, delta, delta.abs()), dim=-1)
        return F.softplus(self.network(features)).squeeze(-1)


class VariableLengthPredictor(nn.Module):
    """Action-token predictor for a frozen LeWM latent space.

    This is the Two-Room adaptation of VLWM's variable-length action-token
    interface. The frozen official LeWM encoder supplies state latents; this
    lightweight predictor is trained with the cumulative horizon curriculum.
    """

    def __init__(
        self,
        latent_dim: int = 192,
        action_dim: int = 10,
        model_dim: int = 192,
        depth: int = 6,
        heads: int = 12,
        mlp_dim: int = 768,
        max_horizon: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.model_dim = model_dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.max_horizon = max_horizon
        self.state_projection = nn.Linear(latent_dim, model_dim)
        self.action_projection = nn.Linear(action_dim, model_dim)
        self.state_type = nn.Parameter(torch.zeros(model_dim))
        self.action_type = nn.Parameter(torch.zeros(model_dim))
        self.position = nn.Parameter(torch.randn(max_horizon + 1, model_dim) * 0.01)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(model_dim)
        self.delta = nn.Linear(model_dim, latent_dim)
        self.gate = nn.Linear(model_dim, latent_dim)

    def forward(self, current: Tensor, actions: Tensor, lengths: Tensor) -> Tensor:
        if actions.ndim != 3:
            raise ValueError("actions must have shape [batch, horizon, action_dim]")
        batch, horizon, _ = actions.shape
        if horizon > self.max_horizon:
            raise ValueError("action sequence exceeds maximum VLWM horizon")
        if lengths.shape != (batch,):
            raise ValueError("lengths must have shape [batch]")
        state = self.state_projection(current).unsqueeze(1) + self.state_type
        action = self.action_projection(actions) + self.action_type
        tokens = torch.cat((state, action), dim=1) + self.position[: horizon + 1]
        positions = torch.arange(horizon + 1, device=actions.device).unsqueeze(0)
        padding = positions > lengths.unsqueeze(1)
        # A causal mask lets every action token use the current state and all
        # preceding actions while preventing access to future action tokens.
        causal = torch.full(
            (horizon + 1, horizon + 1),
            float("-inf"),
            device=actions.device,
            dtype=tokens.dtype,
        ).triu(1)
        encoded = self.transformer(tokens, mask=causal, src_key_padding_mask=padding)
        batch_index = torch.arange(batch, device=actions.device)
        final = self.norm(encoded[batch_index, lengths])
        return current + torch.sigmoid(self.gate(final)) * self.delta(final)


def long_to_short_schedule(horizon: int, max_chunk: int) -> tuple[int, ...]:
    """Return a deterministic coarse-to-fine partition for reproducible CEM."""

    if horizon <= 0 or max_chunk <= 0:
        raise ValueError("horizon and max_chunk must be positive")
    remaining = horizon
    result: list[int] = []
    while remaining:
        progress = len(result) / max(1, horizon - 1)
        preferred = max(1, int(round(max_chunk * math.exp(-2.0 * progress))))
        chunk = min(remaining, preferred)
        result.append(chunk)
        remaining -= chunk
    return tuple(result)
