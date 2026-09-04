from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(slots=True)
class RWKVMatrixState:
    """Persistent RWKV-7 recurrent state.

    RWKV-7 keeps two previous-token vectors per layer: one for TimeMix and one
    for ChannelMix.  The matrix state is always accumulated in fp32, matching
    the official recurrent implementation even when the network runs in a
    lower precision.
    """

    matrix: Tensor
    time_shift: Tensor
    channel_shift: Tensor
    steps: Tensor

    def __post_init__(self) -> None:
        if self.matrix.ndim != 5:
            raise ValueError("matrix must have shape [batch, layers, heads, head_dim, head_dim]")
        if self.time_shift.ndim != 3 or self.channel_shift.ndim != 3:
            raise ValueError("shift states must have shape [batch, layers, model_dim]")
        if self.time_shift.shape != self.channel_shift.shape:
            raise ValueError("TimeMix and ChannelMix shift states must have matching shapes")
        batch, layers = self.matrix.shape[:2]
        if self.time_shift.shape[:2] != (batch, layers):
            raise ValueError("matrix and shift state batch/layer dimensions must match")
        if self.steps.shape != (batch,):
            raise ValueError("steps must have shape [batch]")
        if self.matrix.dtype != torch.float32:
            raise ValueError("RWKV matrix state must use float32 accumulation")
        devices = {
            self.matrix.device,
            self.time_shift.device,
            self.channel_shift.device,
            self.steps.device,
        }
        if len(devices) != 1:
            raise ValueError("all state tensors must be on the same device")

    @property
    def batch_size(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def num_layers(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def device(self) -> torch.device:
        return self.matrix.device

    @property
    def dtype(self) -> torch.dtype:
        return self.time_shift.dtype

    def clone(self) -> RWKVMatrixState:
        return RWKVMatrixState(
            matrix=self.matrix.clone(),
            time_shift=self.time_shift.clone(),
            channel_shift=self.channel_shift.clone(),
            steps=self.steps.clone(),
        )

    def clone_branches(self, branches: int) -> RWKVMatrixState:
        """Clone every batch item into independent contiguous branch states."""

        if branches <= 0:
            raise ValueError("branches must be positive")
        return RWKVMatrixState(
            matrix=self.matrix.repeat_interleave(branches, dim=0).clone(),
            time_shift=self.time_shift.repeat_interleave(branches, dim=0).clone(),
            channel_shift=self.channel_shift.repeat_interleave(branches, dim=0).clone(),
            steps=self.steps.repeat_interleave(branches, dim=0).clone(),
        )

    def detach(self) -> RWKVMatrixState:
        return RWKVMatrixState(
            matrix=self.matrix.detach(),
            time_shift=self.time_shift.detach(),
            channel_shift=self.channel_shift.detach(),
            steps=self.steps.detach(),
        )

    def index_select(self, indices: Tensor) -> RWKVMatrixState:
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("indices must be a rank-1 int64 tensor")
        indices = indices.to(self.device)
        return RWKVMatrixState(
            matrix=self.matrix.index_select(0, indices),
            time_shift=self.time_shift.index_select(0, indices),
            channel_shift=self.channel_shift.index_select(0, indices),
            steps=self.steps.index_select(0, indices),
        )

    def to(self, device: torch.device | str) -> RWKVMatrixState:
        return RWKVMatrixState(
            matrix=self.matrix.to(device=device, dtype=torch.float32),
            time_shift=self.time_shift.to(device=device),
            channel_shift=self.channel_shift.to(device=device),
            steps=self.steps.to(device=device),
        )

    def as_dict(self) -> dict[str, Tensor]:
        return {
            "matrix": self.matrix,
            "time_shift": self.time_shift,
            "channel_shift": self.channel_shift,
            "steps": self.steps,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RWKVMatrixState:
        required = {"matrix", "time_shift", "channel_shift", "steps"}
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"state payload is missing fields: {sorted(missing)}")
        return cls(**{name: payload[name] for name in required})
