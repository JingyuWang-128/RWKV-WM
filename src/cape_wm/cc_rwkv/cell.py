from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

RWKV7_GROUP_NORM_EPS = 64e-5
RWKV7_DECAY_SCALE = 0.606531  # exp(-0.5), as used by the official RNN reference.
RWKV7_NORMALIZE_EPS = 1e-12
CC_DECAY_LOG_HAZARD = "log_hazard"
CC_DECAY_OFFICIAL_LOGIT = "official_logit_residual"
CC_DECAY_MODES = frozenset({CC_DECAY_LOG_HAZARD, CC_DECAY_OFFICIAL_LOGIT})


@dataclass(frozen=True, slots=True)
class RWKV7BlockConfig:
    model_dim: int
    num_layers: int
    num_heads: int
    layer_id: int
    channel_mlp_dim: int
    decay_lora_dim: int | None = None
    aaa_lora_dim: int | None = None
    value_lora_dim: int | None = None
    gate_lora_dim: int | None = None

    def __post_init__(self) -> None:
        if self.model_dim <= 1 or self.num_layers <= 1:
            raise ValueError("official RWKV-7 initialization requires model_dim and num_layers > 1")
        if self.num_heads <= 0 or self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if self.head_dim <= 1:
            raise ValueError("head_dim must be greater than one")
        if not 0 <= self.layer_id < self.num_layers:
            raise ValueError("layer_id is out of range")
        if self.channel_mlp_dim <= 0:
            raise ValueError("channel_mlp_dim must be positive")

    @property
    def head_dim(self) -> int:
        return self.model_dim // self.num_heads

    @staticmethod
    def _suggested_lora(model_dim: int, scale: float) -> int:
        return max(32, int(round((scale * math.sqrt(model_dim)) / 32) * 32))

    @property
    def resolved_decay_lora_dim(self) -> int:
        return self.decay_lora_dim or self._suggested_lora(self.model_dim, 2.5)

    @property
    def resolved_aaa_lora_dim(self) -> int:
        return self.aaa_lora_dim or self._suggested_lora(self.model_dim, 2.5)

    @property
    def resolved_value_lora_dim(self) -> int:
        return self.value_lora_dim or self._suggested_lora(self.model_dim, 1.7)

    @property
    def resolved_gate_lora_dim(self) -> int:
        return self.gate_lora_dim or self._suggested_lora(self.model_dim, 5.0)


def _orthogonal_parameter(rows: int, columns: int, scale: float) -> nn.Parameter:
    value = torch.zeros(rows, columns)
    gain = math.sqrt(rows / columns) if rows > columns else 1.0
    nn.init.orthogonal_(value, gain=gain * scale)
    return nn.Parameter(value)


def rwkv7_matrix_step(
    matrix: Tensor,
    decay: Tensor,
    key: Tensor,
    value: Tensor,
    normalized_key: Tensor,
    learning_rate: Tensor,
) -> Tensor:
    """Official RWKV-7 generalized delta-rule recurrence in decomposed form."""

    erased = torch.einsum("bhij,bhj->bhi", matrix, normalized_key)
    erase_target = normalized_key * learning_rate
    return (
        matrix * decay.unsqueeze(-2)
        - erased.unsqueeze(-1) * erase_target.unsqueeze(-2)
        + value.unsqueeze(-1) * key.unsqueeze(-2)
    )


def counterfactual_rwkv7_decay(
    world_decay_logit: Tensor,
    gated_action_delta: Tensor,
    *,
    mode: str = CC_DECAY_LOG_HAZARD,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return counterfactual decay, official world decay, and CF decay rate.

    In ``log_hazard`` mode the official x070 decay rate is the base hazard:

        rate_world = 0.606531 * sigmoid(world_logit)
        rate_cf = rate_world * exp(action_log_rate_delta)
        decay_cf = exp(-rate_cf)

    This is equivalent to a double exponential after reparameterizing the
    world log-hazard, and is exactly official x070 when the action delta is
    zero.  ``official_logit_residual`` retains the earlier M3 ablation.
    """

    if mode not in CC_DECAY_MODES:
        raise ValueError(f"unsupported counterfactual decay mode: {mode}")
    if world_decay_logit.shape != gated_action_delta.shape:
        raise ValueError("world_decay_logit and gated_action_delta must match")
    world_rate = RWKV7_DECAY_SCALE * torch.sigmoid(world_decay_logit.float())
    world_decay = torch.exp(-world_rate)
    bounded_delta = gated_action_delta.float().clamp(-8.0, 5.0)
    if mode == CC_DECAY_LOG_HAZARD:
        counterfactual_rate = world_rate * torch.exp(bounded_delta)
    else:
        counterfactual_rate = RWKV7_DECAY_SCALE * torch.sigmoid(
            world_decay_logit.float() + bounded_delta
        )
    return torch.exp(-counterfactual_rate), world_decay, counterfactual_rate


def counterfactual_rwkv7_matrix_step(
    matrix: Tensor,
    decay: Tensor,
    world_erase: Tensor,
    world_erase_key: Tensor,
    world_value: Tensor,
    world_write_key: Tensor,
    action_erase_delta: Tensor,
    action_erase_key: Tensor,
    action_write_delta: Tensor,
    action_write_key: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Rank-two centered update without materializing its transition matrix.

    All vectors use the official RWKV row-value/column-key convention.  The
    first outer product is exactly the x070 generalized delta rule when
    ``world_erase=-normalized_key`` and
    ``world_erase_key=normalized_key*learning_rate``.  The second transition
    outer product and second write outer product are action residuals.
    """

    matrix = matrix.float()
    world_next = (
        matrix * decay.float().unsqueeze(-2)
        + torch.einsum("bhij,bhj->bhi", matrix, world_erase.float()).unsqueeze(-1)
        * world_erase_key.float().unsqueeze(-2)
        + world_value.float().unsqueeze(-1) * world_write_key.float().unsqueeze(-2)
    )
    action_update = (
        torch.einsum("bhij,bhj->bhi", matrix, action_erase_delta.float()).unsqueeze(-1)
        * action_erase_key.float().unsqueeze(-2)
        + action_write_delta.float().unsqueeze(-1)
        * action_write_key.float().unsqueeze(-2)
    )
    return world_next + action_update, world_next, action_update


class RWKV7ActionParameterNetwork(nn.Module):
    """The only module allowed to consume raw actual/reference actions."""

    def __init__(self, model_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if action_dim <= 0 or hidden_dim <= 0:
            raise ValueError("action_dim and hidden_dim must be positive")
        self.model_dim = model_dim
        self.action_dim = action_dim
        self.world_projection = nn.Linear(model_dim, hidden_dim, bias=False)
        self.action_projection = nn.Linear(action_dim, hidden_dim, bias=False)
        self.trunk = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.delta_head = nn.Linear(hidden_dim, 3 * model_dim)
        self.gate_head = nn.Linear(hidden_dim, model_dim)

        # At migration time CC-RWKV must be an exact functional reduction of
        # vanilla x070.  Zero delta heads make that true before Stage-B learns.
        self.delta_head.weight.data.zero_()
        self.delta_head.bias.data.zero_()
        self.gate_head.weight.data.zero_()
        self.gate_head.bias.data.fill_(math.log(0.1 / 0.9))

    def forward(self, world: Tensor, action: Tensor) -> dict[str, Tensor]:
        if world.ndim != 2 or world.shape[-1] != self.model_dim:
            raise ValueError("world must have shape [batch, model_dim]")
        if action.shape != (world.shape[0], self.action_dim):
            raise ValueError("action must have shape [batch, action_dim]")
        hidden = F.silu(self.world_projection(world) + self.action_projection(action))
        hidden = F.silu(self.trunk(hidden))
        decay, erase, write = self.delta_head(hidden).chunk(3, dim=-1)
        return {
            "decay": decay,
            "erase": erase,
            "write": write,
            "gate": torch.sigmoid(self.gate_head(hidden)),
        }


class CounterfactualCenteredRWKV7TimeMix(nn.Module):
    """Official x070 world path plus a centered, action-only rank-two update."""

    def __init__(
        self,
        config: RWKV7BlockConfig,
        *,
        action_dim: int,
        action_hidden_dim: int,
        centered: bool,
        decay_mode: str = CC_DECAY_LOG_HAZARD,
    ) -> None:
        super().__init__()
        self.config = config
        c = config.model_dim
        h = config.num_heads
        n = config.head_dim
        layer_id = config.layer_id
        ratio_0_to_1 = layer_id / (config.num_layers - 1)
        ratio_1_to_almost0 = 1.0 - layer_id / config.num_layers

        position = torch.arange(c, dtype=torch.float32).reshape(1, c) / c
        self.x_r = nn.Parameter(1.0 - position.pow(0.2 * ratio_1_to_almost0))
        self.x_w = nn.Parameter(1.0 - position.pow(0.9 * ratio_1_to_almost0))
        self.x_k = nn.Parameter(1.0 - position.pow(0.7 * ratio_1_to_almost0))
        self.x_v = nn.Parameter(1.0 - position.pow(0.7 * ratio_1_to_almost0))
        self.x_a = nn.Parameter(1.0 - position.pow(0.9 * ratio_1_to_almost0))
        self.x_g = nn.Parameter(1.0 - position.pow(0.2 * ratio_1_to_almost0))

        linear = torch.arange(c, dtype=torch.float32) / (c - 1) - 0.5
        within_head = torch.arange(c, dtype=torch.float32).remainder(n)
        zigzag = (within_head - (n - 1) / 2) / ((n - 1) / 2)
        zigzag = zigzag * zigzag.abs()
        decay_curve = -6 + 6 * (
            torch.arange(c, dtype=torch.float32) / (c - 1)
        ).pow(1 + ratio_0_to_1**0.3)

        self.w1 = nn.Parameter(torch.zeros(c, config.resolved_decay_lora_dim))
        self.w2 = _orthogonal_parameter(config.resolved_decay_lora_dim, c, 0.1)
        self.w0 = nn.Parameter(decay_curve + 0.5 + zigzag * 2.5)
        self.a1 = nn.Parameter(torch.zeros(c, config.resolved_aaa_lora_dim))
        self.a2 = _orthogonal_parameter(config.resolved_aaa_lora_dim, c, 0.1)
        self.a0 = nn.Parameter(torch.full((c,), -0.19) + zigzag * 0.3 + linear * 0.4)
        self.v1 = nn.Parameter(torch.zeros(c, config.resolved_value_lora_dim))
        self.v2 = _orthogonal_parameter(config.resolved_value_lora_dim, c, 0.1)
        self.v0 = nn.Parameter(torch.full((c,), 0.73) - linear * 0.4)
        self.g1 = nn.Parameter(torch.zeros(c, config.resolved_gate_lora_dim))
        self.g2 = _orthogonal_parameter(config.resolved_gate_lora_dim, c, 0.1)
        self.k_k = nn.Parameter(torch.full((c,), 0.71) - linear * 0.1)
        self.k_a = nn.Parameter(torch.full((c,), 1.02))
        self.r_k = nn.Parameter(torch.full((h, n), -0.04))
        self.receptance = nn.Linear(c, c, bias=False)
        self.key = nn.Linear(c, c, bias=False)
        self.value = nn.Linear(c, c, bias=False)
        self.output = nn.Linear(c, c, bias=False)
        self.group_norm = nn.GroupNorm(h, c, eps=RWKV7_GROUP_NORM_EPS)
        bound = c**-0.5
        self.receptance.weight.data.uniform_(-0.5 * bound, 0.5 * bound)
        self.key.weight.data.uniform_(-0.05 * bound, 0.05 * bound)
        self.value.weight.data.uniform_(-0.5 * bound, 0.5 * bound)
        self.output.weight.data.zero_()

        self.centered = centered
        if decay_mode not in CC_DECAY_MODES:
            raise ValueError(f"unsupported counterfactual decay mode: {decay_mode}")
        self.decay_mode = decay_mode
        self.action_parameter_network = RWKV7ActionParameterNetwork(
            config.model_dim, action_dim, action_hidden_dim
        )
        self.action_erase_key = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.action_write_key = nn.Linear(config.model_dim, config.model_dim, bias=False)
        bound = config.model_dim**-0.5
        self.action_erase_key.weight.data.uniform_(-0.05 * bound, 0.05 * bound)
        self.action_write_key.weight.data.uniform_(-0.05 * bound, 0.05 * bound)

    def forward(
        self,
        x: Tensor,
        previous_x: Tensor,
        matrix: Tensor,
        v_first: Tensor | None,
        action: Tensor,
        reference_action: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        if x.ndim != 2 or previous_x.shape != x.shape:
            raise ValueError("x and previous_x must have shape [batch, model_dim]")
        batch, channels = x.shape
        h = self.config.num_heads
        n = self.config.head_dim
        if matrix.shape != (batch, h, n, n):
            raise ValueError("matrix has an incompatible shape")

        difference = previous_x - x
        xr = x + difference * self.x_r
        xw = x + difference * self.x_w
        xk = x + difference * self.x_k
        xv = x + difference * self.x_v
        xa = x + difference * self.x_a
        xg = x + difference * self.x_g

        receptance = self.receptance(xr)
        world_decay_logit = self.w0 + torch.tanh(xw @ self.w1) @ self.w2
        key = self.key(xk)
        value = self.value(xv)
        if self.config.layer_id == 0:
            v_first = value
        else:
            if v_first is None:
                raise ValueError("v_first from layer zero is required by higher RWKV-7 layers")
            value_residual = torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
            value = value + (v_first - value) * value_residual
        learning_rate = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        output_gate = torch.sigmoid(xg @ self.g1) @ self.g2

        normalized_key = key * self.k_k
        normalized_key = F.normalize(
            normalized_key.view(batch, h, n),
            dim=-1,
            p=2.0,
            eps=RWKV7_NORMALIZE_EPS,
        )
        key = key * (1 + (learning_rate - 1) * self.k_a)

        actual = self.action_parameter_network(xw, action)
        if self.centered:
            reference = self.action_parameter_network(xw, reference_action)
            action_decay_delta = actual["decay"] - reference["decay"]
            action_erase_delta = actual["erase"] - reference["erase"]
            action_write_delta = actual["write"] - reference["write"]
        else:
            action_decay_delta = actual["decay"]
            action_erase_delta = actual["erase"]
            action_write_delta = actual["write"]
        intervention_gate = actual["gate"]

        # The default interprets the centered action output as a log-hazard
        # residual.  It keeps the official world rate at zero intervention but
        # restores the full (0, 1) double-exp retention range under action.
        gated_decay_delta = intervention_gate * action_decay_delta
        decay, world_decay, counterfactual_decay_rate = counterfactual_rwkv7_decay(
            world_decay_logit,
            gated_decay_delta,
            mode=self.decay_mode,
        )

        receptance_h = receptance.view(batch, h, n)
        key_h = key.view(batch, h, n)
        value_h = value.view(batch, h, n)
        learning_rate_h = learning_rate.view(batch, h, n)
        decay_h = decay.view(batch, h, n)
        world_decay_h = world_decay.view(batch, h, n)
        action_erase_delta_h = (
            intervention_gate * action_erase_delta
        ).view(batch, h, n)
        action_write_delta_h = (
            intervention_gate * action_write_delta
        ).view(batch, h, n)
        action_erase_key_h = torch.tanh(self.action_erase_key(xw)).view(batch, h, n)
        action_write_key_h = torch.tanh(self.action_write_key(xw)).view(batch, h, n)

        # The helper's world result is evaluated with the counterfactual decay;
        # a separate official-only result is retained for interpretable norms.
        next_matrix, _, rank_two_action_update = counterfactual_rwkv7_matrix_step(
            matrix,
            decay_h,
            -normalized_key,
            normalized_key * learning_rate_h,
            value_h,
            key_h,
            action_erase_delta_h,
            action_erase_key_h,
            action_write_delta_h,
            action_write_key_h,
        )
        official_world_next = rwkv7_matrix_step(
            matrix.float(),
            world_decay_h,
            key_h.float(),
            value_h.float(),
            normalized_key.float(),
            learning_rate_h.float(),
        )
        full_action_update = next_matrix - official_world_next

        readout = torch.einsum("bhij,bhj->bhi", next_matrix, receptance_h.float())
        readout = self.group_norm(readout.to(x.dtype).reshape(batch, channels))
        bonus = (receptance_h * key_h * self.r_k).sum(dim=-1, keepdim=True) * value_h
        readout = readout + bonus.reshape(batch, channels)
        output = self.output(readout * output_gate)
        diagnostics = {
            "world_decay_logit": world_decay_logit,
            "action_decay_delta": action_decay_delta,
            "action_log_rate_delta": gated_decay_delta.clamp(-8.0, 5.0),
            "world_decay_rate": (
                RWKV7_DECAY_SCALE * torch.sigmoid(world_decay_logit.float())
            ),
            "counterfactual_decay_rate": counterfactual_decay_rate,
            "counterfactual_decay": decay,
            "world_erase": -normalized_key.reshape(batch, channels),
            "action_erase_delta": action_erase_delta,
            "world_write": value,
            "action_write_delta": action_write_delta,
            "intervention_gate": intervention_gate,
            "matrix_update_world_norm": (
                official_world_next - matrix
            ).flatten(1).norm(dim=-1),
            "matrix_update_action_norm": full_action_update.flatten(1).norm(dim=-1),
            "rank_two_action_norm": rank_two_action_update.flatten(1).norm(dim=-1),
        }
        return output, x, next_matrix, v_first, diagnostics


class CounterfactualCenteredRWKV7Cell(nn.Module):
    """Pre-LN RWKV-7 block with separated world and action update paths."""

    def __init__(
        self,
        config: RWKV7BlockConfig,
        *,
        action_dim: int,
        action_hidden_dim: int,
        centered: bool,
        decay_mode: str = CC_DECAY_LOG_HAZARD,
    ) -> None:
        super().__init__()
        self.config = config
        self.ln0 = nn.LayerNorm(config.model_dim) if config.layer_id == 0 else nn.Identity()
        self.ln1 = nn.LayerNorm(config.model_dim)
        self.ln2 = nn.LayerNorm(config.model_dim)
        self.time_mix = CounterfactualCenteredRWKV7TimeMix(
            config,
            action_dim=action_dim,
            action_hidden_dim=action_hidden_dim,
            centered=centered,
            decay_mode=decay_mode,
        )
        self.channel_mix = RWKV7ChannelMix(config)

    def forward(
        self,
        x: Tensor,
        previous_time_x: Tensor,
        previous_channel_x: Tensor,
        matrix: Tensor,
        v_first: Tensor | None,
        action: Tensor,
        reference_action: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        x = self.ln0(x)
        time_output, next_time_x, next_matrix, v_first, diagnostics = self.time_mix(
            self.ln1(x), previous_time_x, matrix, v_first, action, reference_action
        )
        x = x + time_output
        channel_output, next_channel_x = self.channel_mix(self.ln2(x), previous_channel_x)
        x = x + channel_output
        return x, next_time_x, next_channel_x, next_matrix, v_first, diagnostics


class RWKV7TimeMix(nn.Module):
    """Pure-PyTorch RWKV-7 x070 TimeMix, matching the official RNN equations."""

    def __init__(self, config: RWKV7BlockConfig) -> None:
        super().__init__()
        self.config = config
        c = config.model_dim
        h = config.num_heads
        n = config.head_dim
        layer_id = config.layer_id
        ratio_0_to_1 = layer_id / (config.num_layers - 1)
        ratio_1_to_almost0 = 1.0 - layer_id / config.num_layers

        position = torch.arange(c, dtype=torch.float32).reshape(1, c) / c
        self.x_r = nn.Parameter(1.0 - position.pow(0.2 * ratio_1_to_almost0))
        self.x_w = nn.Parameter(1.0 - position.pow(0.9 * ratio_1_to_almost0))
        self.x_k = nn.Parameter(1.0 - position.pow(0.7 * ratio_1_to_almost0))
        self.x_v = nn.Parameter(1.0 - position.pow(0.7 * ratio_1_to_almost0))
        self.x_a = nn.Parameter(1.0 - position.pow(0.9 * ratio_1_to_almost0))
        self.x_g = nn.Parameter(1.0 - position.pow(0.2 * ratio_1_to_almost0))

        linear = torch.arange(c, dtype=torch.float32) / (c - 1) - 0.5
        within_head = torch.arange(c, dtype=torch.float32).remainder(n)
        zigzag = (within_head - (n - 1) / 2) / ((n - 1) / 2)
        zigzag = zigzag * zigzag.abs()
        decay_curve = -6 + 6 * (
            torch.arange(c, dtype=torch.float32) / (c - 1)
        ).pow(1 + ratio_0_to_1**0.3)

        self.w1 = nn.Parameter(torch.zeros(c, config.resolved_decay_lora_dim))
        self.w2 = _orthogonal_parameter(config.resolved_decay_lora_dim, c, 0.1)
        self.w0 = nn.Parameter(decay_curve + 0.5 + zigzag * 2.5)

        self.a1 = nn.Parameter(torch.zeros(c, config.resolved_aaa_lora_dim))
        self.a2 = _orthogonal_parameter(config.resolved_aaa_lora_dim, c, 0.1)
        self.a0 = nn.Parameter(torch.full((c,), -0.19) + zigzag * 0.3 + linear * 0.4)

        self.v1 = nn.Parameter(torch.zeros(c, config.resolved_value_lora_dim))
        self.v2 = _orthogonal_parameter(config.resolved_value_lora_dim, c, 0.1)
        self.v0 = nn.Parameter(torch.full((c,), 0.73) - linear * 0.4)

        self.g1 = nn.Parameter(torch.zeros(c, config.resolved_gate_lora_dim))
        self.g2 = _orthogonal_parameter(config.resolved_gate_lora_dim, c, 0.1)

        self.k_k = nn.Parameter(torch.full((c,), 0.71) - linear * 0.1)
        self.k_a = nn.Parameter(torch.full((c,), 1.02))
        self.r_k = nn.Parameter(torch.full((h, n), -0.04))

        self.receptance = nn.Linear(c, c, bias=False)
        self.key = nn.Linear(c, c, bias=False)
        self.value = nn.Linear(c, c, bias=False)
        self.output = nn.Linear(c, c, bias=False)
        self.group_norm = nn.GroupNorm(h, c, eps=RWKV7_GROUP_NORM_EPS)

        bound = c**-0.5
        self.receptance.weight.data.uniform_(-0.5 * bound, 0.5 * bound)
        self.key.weight.data.uniform_(-0.05 * bound, 0.05 * bound)
        self.value.weight.data.uniform_(-0.5 * bound, 0.5 * bound)
        self.output.weight.data.zero_()

    def forward(
        self,
        x: Tensor,
        previous_x: Tensor,
        matrix: Tensor,
        v_first: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        if x.ndim != 2 or previous_x.shape != x.shape:
            raise ValueError("x and previous_x must have shape [batch, model_dim]")
        batch, channels = x.shape
        h = self.config.num_heads
        n = self.config.head_dim
        if matrix.shape != (batch, h, n, n):
            raise ValueError("matrix has an incompatible shape")

        difference = previous_x - x
        xr = x + difference * self.x_r
        xw = x + difference * self.x_w
        xk = x + difference * self.x_k
        xv = x + difference * self.x_v
        xa = x + difference * self.x_a
        xg = x + difference * self.x_g

        receptance = self.receptance(xr)
        decay_logit = self.w0 + torch.tanh(xw @ self.w1) @ self.w2
        decay = torch.exp(-RWKV7_DECAY_SCALE * torch.sigmoid(decay_logit.float()))
        key = self.key(xk)
        value = self.value(xv)
        if self.config.layer_id == 0:
            v_first = value
        else:
            if v_first is None:
                raise ValueError("v_first from layer zero is required by higher RWKV-7 layers")
            value_residual = torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
            value = value + (v_first - value) * value_residual
        learning_rate = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        gate = torch.sigmoid(xg @ self.g1) @ self.g2

        normalized_key = key * self.k_k
        normalized_key = F.normalize(
            normalized_key.view(batch, h, n),
            dim=-1,
            p=2.0,
            eps=RWKV7_NORMALIZE_EPS,
        )
        key = key * (1 + (learning_rate - 1) * self.k_a)

        receptance_h = receptance.view(batch, h, n)
        key_h = key.view(batch, h, n)
        value_h = value.view(batch, h, n)
        learning_rate_h = learning_rate.view(batch, h, n)
        decay_h = decay.view(batch, h, n)
        next_matrix = rwkv7_matrix_step(
            matrix.float(),
            decay_h,
            key_h.float(),
            value_h.float(),
            normalized_key.float(),
            learning_rate_h.float(),
        )
        readout = torch.einsum("bhij,bhj->bhi", next_matrix, receptance_h.float())
        readout = self.group_norm(readout.to(x.dtype).reshape(batch, channels))
        bonus = (
            receptance_h * key_h * self.r_k
        ).sum(dim=-1, keepdim=True) * value_h
        readout = readout + bonus.reshape(batch, channels)
        output = self.output(readout * gate)
        diagnostics = {
            "decay_logit": decay_logit,
            "decay": decay,
            "learning_rate": learning_rate,
            "normalized_key": normalized_key.reshape(batch, channels),
            "matrix_update_norm": (next_matrix - matrix).flatten(1).norm(dim=-1),
        }
        return output, x, next_matrix, v_first, diagnostics


class RWKV7ChannelMix(nn.Module):
    def __init__(self, config: RWKV7BlockConfig) -> None:
        super().__init__()
        c = config.model_dim
        ratio_1_to_almost0 = 1.0 - config.layer_id / config.num_layers
        position = torch.arange(c, dtype=torch.float32).reshape(1, c) / c
        self.x_k = nn.Parameter(1.0 - position.pow(ratio_1_to_almost0**4))
        self.key = nn.Linear(c, config.channel_mlp_dim, bias=False)
        self.value = nn.Linear(config.channel_mlp_dim, c, bias=False)
        self.key.weight.data.uniform_(-0.5 / math.sqrt(c), 0.5 / math.sqrt(c))
        self.value.weight.data.zero_()

    def forward(self, x: Tensor, previous_x: Tensor) -> tuple[Tensor, Tensor]:
        mixed = x + (previous_x - x) * self.x_k
        hidden = torch.relu(self.key(mixed)).square()
        return self.value(hidden), x


class VanillaRWKV7Cell(nn.Module):
    """One official-style Pre-LN RWKV-7 residual block in recurrent mode."""

    def __init__(self, config: RWKV7BlockConfig) -> None:
        super().__init__()
        self.config = config
        self.ln0 = nn.LayerNorm(config.model_dim) if config.layer_id == 0 else nn.Identity()
        self.ln1 = nn.LayerNorm(config.model_dim)
        self.ln2 = nn.LayerNorm(config.model_dim)
        self.time_mix = RWKV7TimeMix(config)
        self.channel_mix = RWKV7ChannelMix(config)

    def forward(
        self,
        x: Tensor,
        previous_time_x: Tensor,
        previous_channel_x: Tensor,
        matrix: Tensor,
        v_first: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        x = self.ln0(x)
        time_output, next_time_x, next_matrix, v_first, diagnostics = self.time_mix(
            self.ln1(x), previous_time_x, matrix, v_first
        )
        x = x + time_output
        channel_output, next_channel_x = self.channel_mix(self.ln2(x), previous_channel_x)
        x = x + channel_output
        return x, next_time_x, next_channel_x, next_matrix, v_first, diagnostics
