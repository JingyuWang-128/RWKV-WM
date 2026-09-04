from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .cell import RWKV7BlockConfig, VanillaRWKV7Cell
from .state import RWKVMatrixState


@dataclass(frozen=True, slots=True)
class RWKV7WorldModelConfig:
    latent_dim: int = 192
    action_dim: int = 10
    model_dim: int = 192
    num_layers: int = 6
    num_heads: int = 6
    channel_mlp_dim: int = 768
    decay_lora_dim: int | None = None
    aaa_lora_dim: int | None = None
    value_lora_dim: int | None = None
    gate_lora_dim: int | None = None

    def __post_init__(self) -> None:
        if min(self.latent_dim, self.action_dim, self.model_dim) <= 0:
            raise ValueError("latent_dim, action_dim, and model_dim must be positive")
        if self.num_layers <= 1:
            raise ValueError("num_layers must be greater than one")
        if self.num_heads <= 0 or self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if self.channel_mlp_dim <= 0:
            raise ValueError("channel_mlp_dim must be positive")

    @property
    def head_dim(self) -> int:
        return self.model_dim // self.num_heads

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class VanillaRWKV7WorldPredictor(nn.Module):
    """Action-conditioned stateful RWKV-7 latent predictor (B2).

    The task adapter maps continuous latent and action vectors to an RWKV token.
    Every recurrent block after that adapter follows the official RWKV-7 x070
    equations.  This vanilla B2 model deliberately mixes action into the token;
    world/action separation belongs to M3, not M1.
    """

    def __init__(self, config: RWKV7WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.latent_projection = nn.Linear(config.latent_dim, config.model_dim, bias=False)
        self.action_projection = nn.Linear(config.action_dim, config.model_dim, bias=False)
        self.blocks = nn.ModuleList(
            [
                VanillaRWKV7Cell(
                    RWKV7BlockConfig(
                        model_dim=config.model_dim,
                        num_layers=config.num_layers,
                        num_heads=config.num_heads,
                        layer_id=layer_id,
                        channel_mlp_dim=config.channel_mlp_dim,
                        decay_lora_dim=config.decay_lora_dim,
                        aaa_lora_dim=config.aaa_lora_dim,
                        value_lora_dim=config.value_lora_dim,
                        gate_lora_dim=config.gate_lora_dim,
                    )
                )
                for layer_id in range(config.num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.model_dim)
        self.prediction_head = nn.Linear(config.model_dim, config.latent_dim)

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> RWKVMatrixState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        parameter = next(self.parameters())
        device = parameter.device if device is None else torch.device(device)
        dtype = parameter.dtype if dtype is None else dtype
        cfg = self.config
        return RWKVMatrixState(
            matrix=torch.zeros(
                batch_size,
                cfg.num_layers,
                cfg.num_heads,
                cfg.head_dim,
                cfg.head_dim,
                device=device,
                dtype=torch.float32,
            ),
            time_shift=torch.zeros(
                batch_size, cfg.num_layers, cfg.model_dim, device=device, dtype=dtype
            ),
            channel_shift=torch.zeros(
                batch_size, cfg.num_layers, cfg.model_dim, device=device, dtype=dtype
            ),
            steps=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def _validate_step_inputs(
        self, latent: Tensor, action: Tensor, state: RWKVMatrixState
    ) -> None:
        if latent.ndim != 2 or latent.shape[-1] != self.config.latent_dim:
            raise ValueError("latent must have shape [batch, latent_dim]")
        if action.ndim != 2 or action.shape != (latent.shape[0], self.config.action_dim):
            raise ValueError("action must have shape [batch, action_dim]")
        if state.batch_size != latent.shape[0]:
            raise ValueError("state and input batch sizes must match")
        expected_matrix = (
            latent.shape[0],
            self.config.num_layers,
            self.config.num_heads,
            self.config.head_dim,
            self.config.head_dim,
        )
        if state.matrix.shape != expected_matrix:
            raise ValueError("state matrix shape is incompatible with predictor config")
        if state.dtype != latent.dtype:
            raise ValueError("latent and shift state dtypes must match")
        if state.device != latent.device or action.device != latent.device:
            raise ValueError("latent, action, and state must share a device")

    @staticmethod
    def _masked(new: Tensor, old: Tensor, update_mask: Tensor) -> Tensor:
        shape = (update_mask.shape[0],) + (1,) * (new.ndim - 1)
        return torch.where(update_mask.reshape(shape), new, old)

    def _step(
        self,
        latent: Tensor,
        action: Tensor,
        state: RWKVMatrixState,
        *,
        update_mask: Tensor | None,
        return_diagnostics: bool,
    ) -> tuple[Tensor, RWKVMatrixState, dict[str, Tensor]]:
        self._validate_step_inputs(latent, action, state)
        if update_mask is None:
            update_mask = torch.ones(latent.shape[0], device=latent.device, dtype=torch.bool)
        if update_mask.shape != (latent.shape[0],) or update_mask.dtype != torch.bool:
            raise ValueError("update_mask must be boolean with shape [batch]")

        x = self.latent_projection(latent) + self.action_projection(action)
        next_time: list[Tensor] = []
        next_channel: list[Tensor] = []
        next_matrix: list[Tensor] = []
        layer_diagnostics: dict[str, list[Tensor]] = {}
        v_first: Tensor | None = None
        for layer_id, block in enumerate(self.blocks):
            x, time_x, channel_x, matrix, v_first, diagnostics = block(
                x,
                state.time_shift[:, layer_id],
                state.channel_shift[:, layer_id],
                state.matrix[:, layer_id],
                v_first,
            )
            next_time.append(
                self._masked(time_x, state.time_shift[:, layer_id], update_mask)
            )
            next_channel.append(
                self._masked(channel_x, state.channel_shift[:, layer_id], update_mask)
            )
            next_matrix.append(
                self._masked(matrix, state.matrix[:, layer_id], update_mask)
            )
            if return_diagnostics:
                for name, value in diagnostics.items():
                    layer_diagnostics.setdefault(name, []).append(value)

        predictor_hidden = self.output_norm(x)
        prediction = self.prediction_head(predictor_hidden)
        if return_diagnostics:
            layer_diagnostics["predictor_hidden"] = [predictor_hidden]
        next_state = RWKVMatrixState(
            matrix=torch.stack(next_matrix, dim=1).float(),
            time_shift=torch.stack(next_time, dim=1),
            channel_shift=torch.stack(next_channel, dim=1),
            steps=state.steps + update_mask.long(),
        )
        stacked = (
            {name: torch.stack(values, dim=1) for name, values in layer_diagnostics.items()}
            if return_diagnostics
            else {}
        )
        return prediction, next_state, stacked

    def step(
        self,
        latent: Tensor,
        action: Tensor,
        state: RWKVMatrixState,
        reference_action: Tensor | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, RWKVMatrixState, dict[str, Tensor]]:
        """Consume ``(latent_t, action_t)`` and predict ``latent_{t+1}``."""

        del reference_action  # B2 is action-conditioned but not counterfactual-centered.
        return self._step(
            latent,
            action,
            state,
            update_mask=None,
            return_diagnostics=return_diagnostics,
        )

    def forward_sequence(
        self,
        latents: Tensor,
        actions: Tensor,
        state: RWKVMatrixState | None = None,
        mask: Tensor | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, RWKVMatrixState, dict[str, Tensor]]:
        if latents.ndim != 3 or actions.ndim != 3:
            raise ValueError("latents and actions must be rank-3")
        if latents.shape[:2] != actions.shape[:2]:
            raise ValueError("latents and actions must share batch/time dimensions")
        batch, steps = latents.shape[:2]
        if steps <= 0:
            raise ValueError("sequence length must be positive")
        if mask is None:
            mask = torch.ones(batch, steps, device=latents.device, dtype=torch.bool)
        if mask.shape != (batch, steps) or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean with shape [batch, time]")
        if state is None:
            state = self.init_state(batch, device=latents.device, dtype=latents.dtype)

        predictions: list[Tensor] = []
        sequence_diagnostics: dict[str, list[Tensor]] = {}
        for time_index in range(steps):
            prediction, state, diagnostics = self._step(
                latents[:, time_index],
                actions[:, time_index],
                state,
                update_mask=mask[:, time_index],
                return_diagnostics=return_diagnostics,
            )
            predictions.append(prediction)
            for name, value in diagnostics.items():
                sequence_diagnostics.setdefault(name, []).append(value)
        stacked = {
            name: torch.stack(values, dim=1)
            for name, values in sequence_diagnostics.items()
        }
        return torch.stack(predictions, dim=1), state, stacked

    def consume_history(
        self,
        latents: Tensor,
        actions: Tensor,
        mask: Tensor | None = None,
    ) -> RWKVMatrixState:
        _, state, _ = self.forward_sequence(latents, actions, mask=mask)
        return state

    def rollout(
        self,
        initial_latent: Tensor,
        future_actions: Tensor,
        state: RWKVMatrixState,
        reference_actions: Tensor | None = None,
    ) -> dict[str, Any]:
        """Clone history state over branches and free-run without observations."""

        del reference_actions
        if initial_latent.ndim != 2:
            raise ValueError("initial_latent must have shape [batch, latent_dim]")
        if future_actions.ndim != 4:
            raise ValueError("future_actions must have shape [batch, branches, time, action_dim]")
        batch, branches, horizon, action_dim = future_actions.shape
        if batch != initial_latent.shape[0] or action_dim != self.config.action_dim:
            raise ValueError("rollout inputs are incompatible with predictor config")
        if horizon <= 0 or branches <= 0:
            raise ValueError("rollout branches and horizon must be positive")
        if state.batch_size != batch:
            raise ValueError("history state batch size does not match initial_latent")

        branch_state = state.clone_branches(branches)
        latent = initial_latent.repeat_interleave(branches, dim=0)
        actions = future_actions.reshape(batch * branches, horizon, action_dim)
        predictions: list[Tensor] = []
        for time_index in range(horizon):
            prediction, branch_state, _ = self.step(
                latent, actions[:, time_index], branch_state
            )
            predictions.append(prediction)
            latent = prediction.to(branch_state.dtype)
        trajectory = torch.stack(predictions, dim=1).reshape(
            batch, branches, horizon, self.config.latent_dim
        )
        return {"latents": trajectory, "final_state": branch_state}

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def rwkv7_optimizer_groups(
    model: nn.Module,
    *,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Official-style RWKV-7 AdamW grouping.

    Large projection weights receive weight decay, vector/normalization/LoRA
    parameters do not, and decay base logits use the official 2x LR scale.
    """

    decay: list[nn.Parameter] = []
    regular: list[nn.Parameter] = []
    decay_logits: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("time_mix.w0"):
            decay_logits.append(parameter)
        elif parameter.squeeze().ndim >= 2 and name.endswith(".weight") and weight_decay > 0:
            decay.append(parameter)
        else:
            regular.append(parameter)
    groups: list[dict[str, Any]] = [
        {"params": regular, "weight_decay": 0.0, "lr_scale": 1.0},
        {"params": decay_logits, "weight_decay": 0.0, "lr_scale": 2.0},
    ]
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay, "lr_scale": 1.0})
    return groups
