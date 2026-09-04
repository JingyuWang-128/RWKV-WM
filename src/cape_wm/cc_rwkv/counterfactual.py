from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .cell import (
    CC_DECAY_LOG_HAZARD,
    CC_DECAY_MODES,
    CounterfactualCenteredRWKV7Cell,
    RWKV7BlockConfig,
)
from .predictor import RWKV7WorldModelConfig, VanillaRWKV7WorldPredictor
from .state import RWKVMatrixState


@dataclass(frozen=True, slots=True)
class CounterfactualRWKV7Config(RWKV7WorldModelConfig):
    """Configuration shared by B4 (uncentered) and B6 (centered)."""

    action_hidden_dim: int = 64
    centered: bool = True
    decay_mode: str = CC_DECAY_LOG_HAZARD

    def __post_init__(self) -> None:
        super(CounterfactualRWKV7Config, self).__post_init__()
        if self.action_hidden_dim <= 0:
            raise ValueError("action_hidden_dim must be positive")
        if self.decay_mode not in CC_DECAY_MODES:
            raise ValueError(f"unsupported counterfactual decay mode: {self.decay_mode}")


class CounterfactualRWKV7WorldPredictor(nn.Module):
    """World/action-separated RWKV-7 predictor used for B4 and B6.

    Raw actions are never added to the latent token.  They are consumed only
    by each block's shared action-parameter network and can influence output
    only through the recurrent matrix update.
    """

    def __init__(self, config: CounterfactualRWKV7Config) -> None:
        super().__init__()
        self.config = config
        self.latent_projection = nn.Linear(config.latent_dim, config.model_dim, bias=False)
        self.blocks = nn.ModuleList(
            [
                CounterfactualCenteredRWKV7Cell(
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
                    ),
                    action_dim=config.action_dim,
                    action_hidden_dim=config.action_hidden_dim,
                    centered=config.centered,
                    decay_mode=config.decay_mode,
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
        self,
        latent: Tensor,
        action: Tensor,
        reference_action: Tensor,
        state: RWKVMatrixState,
    ) -> None:
        if latent.ndim != 2 or latent.shape[-1] != self.config.latent_dim:
            raise ValueError("latent must have shape [batch, latent_dim]")
        action_shape = (latent.shape[0], self.config.action_dim)
        if action.shape != action_shape or reference_action.shape != action_shape:
            raise ValueError("actual/reference actions must have shape [batch, action_dim]")
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
        devices = {latent.device, action.device, reference_action.device, state.device}
        if len(devices) != 1:
            raise ValueError("latent, actions, and state must share a device")

    @staticmethod
    def _masked(new: Tensor, old: Tensor, update_mask: Tensor) -> Tensor:
        shape = (update_mask.shape[0],) + (1,) * (new.ndim - 1)
        return torch.where(update_mask.reshape(shape), new, old)

    def _step(
        self,
        latent: Tensor,
        action: Tensor,
        reference_action: Tensor,
        state: RWKVMatrixState,
        *,
        update_mask: Tensor | None,
        return_diagnostics: bool,
    ) -> tuple[Tensor, RWKVMatrixState, dict[str, Tensor]]:
        self._validate_step_inputs(latent, action, reference_action, state)
        if update_mask is None:
            update_mask = torch.ones(latent.shape[0], device=latent.device, dtype=torch.bool)
        if update_mask.shape != (latent.shape[0],) or update_mask.dtype != torch.bool:
            raise ValueError("update_mask must be boolean with shape [batch]")

        # Deliberately no action_projection: raw action enters only the update.
        x = self.latent_projection(latent)
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
                action,
                reference_action,
            )
            next_time.append(self._masked(time_x, state.time_shift[:, layer_id], update_mask))
            next_channel.append(
                self._masked(channel_x, state.channel_shift[:, layer_id], update_mask)
            )
            next_matrix.append(self._masked(matrix, state.matrix[:, layer_id], update_mask))
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
        diagnostics = (
            {name: torch.stack(values, dim=1) for name, values in layer_diagnostics.items()}
            if return_diagnostics
            else {}
        )
        return prediction, next_state, diagnostics

    def step(
        self,
        latent: Tensor,
        action: Tensor,
        state: RWKVMatrixState,
        reference_action: Tensor | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, RWKVMatrixState, dict[str, Tensor]]:
        if reference_action is None:
            reference_action = torch.zeros_like(action)
        return self._step(
            latent,
            action,
            reference_action,
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
        reference_actions: Tensor | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, RWKVMatrixState, dict[str, Tensor]]:
        if latents.ndim != 3 or actions.ndim != 3:
            raise ValueError("latents and actions must be rank-3")
        if latents.shape[:2] != actions.shape[:2]:
            raise ValueError("latents and actions must share batch/time dimensions")
        if reference_actions is None:
            reference_actions = torch.zeros_like(actions)
        if reference_actions.shape != actions.shape:
            raise ValueError("reference_actions must match actions")
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
                reference_actions[:, time_index],
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
        reference_actions: Tensor | None = None,
    ) -> RWKVMatrixState:
        _, state, _ = self.forward_sequence(
            latents,
            actions,
            mask=mask,
            reference_actions=reference_actions,
        )
        return state

    def rollout(
        self,
        initial_latent: Tensor,
        future_actions: Tensor,
        state: RWKVMatrixState,
        reference_actions: Tensor | None = None,
    ) -> dict[str, Any]:
        if initial_latent.ndim != 2:
            raise ValueError("initial_latent must have shape [batch, latent_dim]")
        if future_actions.ndim != 4:
            raise ValueError("future_actions must have shape [batch, branches, time, action_dim]")
        if reference_actions is None:
            reference_actions = torch.zeros_like(future_actions)
        if reference_actions.shape != future_actions.shape:
            raise ValueError("reference_actions must match future_actions")
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
        references = reference_actions.reshape(batch * branches, horizon, action_dim)
        predictions: list[Tensor] = []
        for time_index in range(horizon):
            prediction, branch_state, _ = self.step(
                latent,
                actions[:, time_index],
                branch_state,
                references[:, time_index],
            )
            predictions.append(prediction)
            latent = prediction.to(branch_state.dtype)
        trajectory = torch.stack(predictions, dim=1).reshape(
            batch, branches, horizon, self.config.latent_dim
        )
        return {"latents": trajectory, "final_state": branch_state}

    def load_world_from_vanilla(
        self, vanilla: VanillaRWKV7WorldPredictor
    ) -> dict[str, list[str]]:
        """Copy all shape-compatible world-path weights from a B2 predictor."""

        source = vanilla.state_dict()
        target = self.state_dict()
        copied: list[str] = []
        for name, value in source.items():
            if name in target and target[name].shape == value.shape:
                target[name] = value.detach().clone()
                copied.append(name)
        self.load_state_dict(target)
        ignored = sorted(set(source) - set(copied))
        new = sorted(set(target) - set(copied))
        return {"copied": sorted(copied), "ignored": ignored, "new": new}

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
