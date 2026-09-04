from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .predictor import VanillaRWKV7WorldPredictor

DWM_ARXIV_ID = "2607.18715"
DWM_IMPLEMENTATION_STATUS = "paper_spec_reimplementation"


class DWMWorldHead(nn.Module):
    """Paper-spec two-layer MLP world head with BatchNorm."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 2048) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        if hidden.ndim != 2 or hidden.shape[-1] != self.input_dim:
            raise ValueError("DWM world head input must be [batch, predictor_dim]")
        return self.network(hidden)


class DWMOutputBaseline(nn.Module):
    """B3: vanilla RWKV inference path plus a training-only DWM world head."""

    implementation_status = DWM_IMPLEMENTATION_STATUS
    source_arxiv = DWM_ARXIV_ID

    def __init__(
        self,
        predictor: VanillaRWKV7WorldPredictor,
        *,
        world_head_hidden_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.world_head = DWMWorldHead(
            predictor.config.model_dim,
            predictor.config.latent_dim,
            world_head_hidden_dim,
        )
        self.world_head_hidden_dim = world_head_hidden_dim

    @property
    def config(self):
        return self.predictor.config

    def inference_predictor(self) -> VanillaRWKV7WorldPredictor:
        return self.predictor

    def world_views(
        self, hidden: Tensor, perturbed_hidden: Tensor
    ) -> tuple[Tensor, Tensor]:
        if hidden.shape != perturbed_hidden.shape:
            raise ValueError("DWM views must have matching shapes")
        combined = self.world_head(torch.cat((hidden, perturbed_hidden), dim=0))
        return combined.chunk(2, dim=0)

    def parameter_count(self, *, inference_only: bool = False) -> int:
        module = self.predictor if inference_only else self
        return sum(parameter.numel() for parameter in module.parameters())


@dataclass(frozen=True, slots=True)
class DWMLosses:
    world_contrastive: Tensor
    orthogonality: Tensor


def symmetric_info_nce(first: Tensor, second: Tensor, *, temperature: float = 0.07) -> Tensor:
    if first.ndim != 2 or second.shape != first.shape:
        raise ValueError("InfoNCE views must be matching matrices")
    if first.shape[0] < 2:
        raise ValueError("InfoNCE requires at least two contexts")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    first = F.normalize(first, dim=-1)
    second = F.normalize(second, dim=-1)
    logits = first @ second.T / temperature
    labels = torch.arange(first.shape[0], device=first.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )


def dwm_auxiliary_losses(
    prediction: Tensor,
    world: Tensor,
    perturbed_world: Tensor,
    *,
    temperature: float = 0.07,
    eps: float = 1e-6,
) -> DWMLosses:
    if prediction.shape != world.shape or perturbed_world.shape != world.shape:
        raise ValueError("DWM prediction/world tensors must match")
    action_component = prediction - world
    orthogonality = F.cosine_similarity(
        world, action_component, dim=-1, eps=eps
    ).abs().mean()
    return DWMLosses(
        world_contrastive=symmetric_info_nce(
            world, perturbed_world, temperature=temperature
        ),
        orthogonality=orthogonality,
    )
