from __future__ import annotations

from torch import nn

from .counterfactual import CounterfactualRWKV7Config, CounterfactualRWKV7WorldPredictor
from .dwm import DWMOutputBaseline
from .predictor import RWKV7WorldModelConfig, VanillaRWKV7WorldPredictor


def build_rwkv_world_model(
    method: str,
    profile: str,
    latent_dim: int,
    action_dim: int,
) -> nn.Module:
    if method not in {"b2", "b3", "b4", "b6"}:
        raise ValueError(f"unsupported method: {method}")
    if profile not in {"smoke", "main"}:
        raise ValueError(f"unsupported profile: {profile}")
    if profile == "main":
        model_dim, layers, heads = 192, 6, 6
        vanilla_width, counterfactual_width = 4096, 3728
        action_hidden, dwm_hidden = 64, 2048
    else:
        model_dim, layers, heads = 32, 2, 4
        vanilla_width, counterfactual_width = 512, 432
        action_hidden, dwm_hidden = 16, 64
    if method in {"b2", "b3"}:
        predictor = VanillaRWKV7WorldPredictor(
            RWKV7WorldModelConfig(
                latent_dim=latent_dim,
                action_dim=action_dim,
                model_dim=model_dim,
                num_layers=layers,
                num_heads=heads,
                channel_mlp_dim=vanilla_width,
            )
        )
        return (
            DWMOutputBaseline(predictor, world_head_hidden_dim=dwm_hidden)
            if method == "b3"
            else predictor
        )
    return CounterfactualRWKV7WorldPredictor(
        CounterfactualRWKV7Config(
            latent_dim=latent_dim,
            action_dim=action_dim,
            model_dim=model_dim,
            num_layers=layers,
            num_heads=heads,
            channel_mlp_dim=counterfactual_width,
            action_hidden_dim=action_hidden,
            centered=method == "b6",
        )
    )
