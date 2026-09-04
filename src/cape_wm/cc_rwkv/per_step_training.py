from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .dwm import DWMOutputBaseline
from .state import RWKVMatrixState


@dataclass(slots=True)
class PerStepBatch:
    history_latents: Tensor
    history_actions: Tensor
    history_mask: Tensor
    factual_actions: Tensor
    factual_latents: Tensor
    pulse_noop_actions: Tensor
    pulse_noop_latents: Tensor
    pulse_noop_mask: Tensor
    effect_latents: Tensor
    sample_ids: list[str]

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], device: torch.device | str) -> "PerStepBatch":
        names = (
            "history_latents", "history_actions", "history_mask", "factual_actions",
            "factual_latents", "pulse_noop_actions", "pulse_noop_latents",
            "pulse_noop_mask", "effect_latents",
        )
        values = {
            name: payload[name].to(device).float()
            if payload[name].dtype.is_floating_point
            else payload[name].to(device)
            for name in names
        }
        values["sample_ids"] = [str(item) for item in payload.get("sample_id", [])]
        batch = cls(**values)
        batch.validate()
        return batch

    def validate(self) -> None:
        if self.history_latents.ndim != 3 or self.history_actions.ndim != 3:
            raise ValueError("history tensors must be [batch,time,feature]")
        if self.factual_actions.ndim != 3 or self.factual_latents.ndim != 3:
            raise ValueError("factual tensors must be rank three")
        if self.pulse_noop_actions.ndim != 4 or self.pulse_noop_latents.ndim != 4:
            raise ValueError("pulse tensors must be rank four")
        if self.pulse_noop_mask.shape != self.pulse_noop_actions.shape[:3]:
            raise ValueError("pulse mask shape is invalid")
        if self.effect_latents.shape != self.pulse_noop_latents.shape:
            raise ValueError("effect and pulse latent shapes must match")
        if self.factual_latents.shape[1] != self.factual_actions.shape[1] + 1:
            raise ValueError("factual latents must include the initial latent")

    @property
    def batch_size(self) -> int:
        return int(self.factual_actions.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.factual_actions.shape[1])


class InMemoryPerStepSplit:
    def __init__(
        self,
        path: str | Path,
        *,
        split: str,
        limit: int | None = None,
        load_effect: bool = True,
    ) -> None:
        codes = {"train": 0, "validation": 1, "test": 2}
        if split not in codes:
            raise ValueError(f"unknown split: {split}")
        with h5py.File(path, "r") as handle:
            samples = handle["samples"]
            indices = torch.from_numpy(
                np.flatnonzero(np.asarray(samples["split"]) == codes[split])
            )
            if limit is not None:
                if limit <= 0:
                    raise ValueError("limit must be positive")
                indices = indices[:limit]
            if len(indices) == 0:
                raise ValueError(f"per-step split is empty: {split}")
            index = indices.numpy()
            self.sample_ids = [
                item.decode() if isinstance(item, bytes) else str(item)
                for item in samples["sample_id"][index]
            ]
            tensor_names = [
                "history_latents", "history_actions_raw", "history_mask",
                "factual_actions", "factual_latents", "pulse_noop_actions",
                "pulse_noop_latents", "pulse_noop_mask",
            ]
            if load_effect:
                tensor_names.append("effect_latents")
            self.tensors = {
                name: torch.from_numpy(np.asarray(samples[name][index]).copy())
                for name in tensor_names
            }

    def __len__(self) -> int:
        return len(self.sample_ids)

    @property
    def horizon(self) -> int:
        return int(self.tensors["factual_actions"].shape[1])

    def batch(self, indices: Tensor, *, device: torch.device | str) -> PerStepBatch:
        payload = {name: tensor.index_select(0, indices) for name, tensor in self.tensors.items()}
        payload["history_actions"] = payload.pop("history_actions_raw")
        if "effect_latents" not in payload:
            payload["effect_latents"] = torch.zeros_like(payload["pulse_noop_latents"])
        payload["sample_id"] = [self.sample_ids[index] for index in indices.tolist()]
        return PerStepBatch.from_mapping(payload, device)

    def random_batch(self, batch_size: int, *, generator: torch.Generator, device: torch.device | str) -> PerStepBatch:
        indices = torch.randint(len(self), (batch_size,), generator=generator)
        return self.batch(indices, device=device)

    def batches(self, batch_size: int, *, device: torch.device | str) -> Iterable[PerStepBatch]:
        for start in range(0, len(self), batch_size):
            yield self.batch(torch.arange(start, min(start + batch_size, len(self))), device=device)


def _predictor(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "predictor", model)


def _zero_actions(batch: int, dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.zeros(batch, dim, device=device, dtype=dtype)


def _cat_states(states: list[RWKVMatrixState]) -> RWKVMatrixState:
    if not states:
        raise ValueError("cannot concatenate an empty state list")
    return RWKVMatrixState(
        matrix=torch.cat([state.matrix for state in states], dim=0),
        time_shift=torch.cat([state.time_shift for state in states], dim=0),
        channel_shift=torch.cat([state.channel_shift for state in states], dim=0),
        steps=torch.cat([state.steps for state in states], dim=0),
    )


def _split_state(state: RWKVMatrixState, batch_size: int) -> list[RWKVMatrixState]:
    if batch_size <= 0 or state.batch_size % batch_size:
        raise ValueError("state batch must be divisible by branch batch size")
    matrices = state.matrix.split(batch_size, dim=0)
    time_shifts = state.time_shift.split(batch_size, dim=0)
    channel_shifts = state.channel_shift.split(batch_size, dim=0)
    steps = state.steps.split(batch_size, dim=0)
    return [
        RWKVMatrixState(matrix=matrix, time_shift=time_shift, channel_shift=channel_shift, steps=step)
        for matrix, time_shift, channel_shift, step in zip(
            matrices, time_shifts, channel_shifts, steps
        )
    ]


def per_step_rollout(
    model: torch.nn.Module,
    batch: PerStepBatch,
    *,
    horizon: int | None = None,
    return_diagnostics: bool = False,
    teacher_forcing: bool = False,
    vectorized_branches: bool = True,
) -> dict[str, Any]:
    predictor = _predictor(model)
    h = min(int(horizon or batch.horizon), batch.horizon)
    state = predictor.consume_history(batch.history_latents, batch.history_actions, mask=batch.history_mask)
    latent = batch.factual_latents[:, 0].to(state.dtype)
    factual_predictions: list[Tensor] = []
    states_before: list[RWKVMatrixState] = []
    factual_inputs: list[Tensor] = []
    zero = _zero_actions(batch.batch_size, batch.factual_actions.shape[-1], latent.device, latent.dtype)
    factual_diags: list[dict[str, Tensor]] = []
    for position in range(h):
        states_before.append(state.clone())
        factual_inputs.append(latent)
        prediction, state, diagnostics = predictor.step(
            latent, batch.factual_actions[:, position].to(latent.dtype), state, zero,
            return_diagnostics=return_diagnostics,
        )
        factual_predictions.append(prediction)
        if return_diagnostics:
            factual_diags.append(diagnostics)
        latent = (
            batch.factual_latents[:, position + 1].to(state.dtype)
            if teacher_forcing
            else prediction.to(state.dtype)
        )

    factual_pred = torch.stack(factual_predictions, dim=1)
    pulse_predictions: list[Tensor] = []
    pulse_diags: list[dict[str, Tensor]] = []
    position_predictions: list[list[Tensor]] = [[] for _ in range(h)]
    if vectorized_branches and not return_diagnostics:
        branch_states = [state.clone() for state in states_before]
        branch_latents = list(factual_inputs)
        for offset in range(h):
            active = h - offset
            state_batch = _cat_states(branch_states[:active])
            latent_batch = torch.cat(branch_latents[:active], dim=0)
            if offset == 0:
                action_batch = zero.repeat(active, 1)
            else:
                action_batch = torch.cat(
                    [
                        batch.factual_actions[:, position + offset].to(latent_batch.dtype)
                        for position in range(active)
                    ],
                    dim=0,
                )
            zero_batch = _zero_actions(
                action_batch.shape[0], action_batch.shape[-1], latent_batch.device, latent_batch.dtype
            )
            prediction, state_batch, _ = predictor.step(
                latent_batch,
                action_batch,
                state_batch,
                zero_batch,
                return_diagnostics=False,
            )
            prediction_chunks = list(prediction.split(batch.batch_size, dim=0))
            state_chunks = _split_state(state_batch, batch.batch_size)
            for position in range(active):
                position_predictions[position].append(prediction_chunks[position])
                branch_latents[position] = prediction_chunks[position].to(state_chunks[position].dtype)
                branch_states[position] = state_chunks[position]
    else:
        for position in range(h):
            branch_state = states_before[position].clone()
            branch_latent = factual_inputs[position]
            for offset in range(h - position):
                absolute = position + offset
                action = zero if offset == 0 else batch.factual_actions[:, absolute].to(branch_latent.dtype)
                prediction, branch_state, diagnostics = predictor.step(
                    branch_latent, action, branch_state, zero, return_diagnostics=return_diagnostics
                )
                position_predictions[position].append(prediction)
                if return_diagnostics:
                    pulse_diags.append(diagnostics)
                branch_latent = prediction.to(branch_state.dtype)
    for position in range(h):
        padded = list(position_predictions[position])
        padded.extend(
            factual_pred.new_zeros((batch.batch_size, factual_pred.shape[-1]))
            for _ in range(position)
        )
        pulse_predictions.append(torch.stack(padded, dim=1))
    pulse_pred = torch.stack(pulse_predictions, dim=1)

    mask = batch.pulse_noop_mask[:, :h, :h]
    return {
        "factual_predicted": factual_pred,
        "pulse_predicted": pulse_pred,
        "factual_target": batch.factual_latents[:, 1 : h + 1].to(factual_pred.dtype),
        "pulse_target": batch.pulse_noop_latents[:, :h, :h].to(factual_pred.dtype),
        "effect_target": batch.effect_latents[:, :h, :h].to(factual_pred.dtype),
        "mask": mask,
        "factual_diags": factual_diags,
        "pulse_diags": pulse_diags,
    }


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    return (value * mask.to(value.dtype)).sum() / mask.sum().clamp_min(1)


def per_step_loss(
    model: torch.nn.Module,
    batch: PerStepBatch,
    *,
    horizon: int | None = None,
    effect_threshold: float = 0.0,
    allow_paired_loss: bool = True,
    effect_weight: float = 1.0,
    teacher_forcing: bool = False,
) -> tuple[Tensor, dict[str, Tensor]]:
    if effect_weight < 0:
        raise ValueError("effect_weight must be non-negative")
    output = per_step_rollout(
        model, batch, horizon=horizon, return_diagnostics=False, teacher_forcing=teacher_forcing
    )
    h = output["factual_predicted"].shape[1]
    factual_point = F.smooth_l1_loss(output["factual_predicted"], output["factual_target"], reduction="none").mean(-1)
    factual = factual_point.mean()
    mask = output["mask"]
    pulse_point = F.smooth_l1_loss(output["pulse_predicted"], output["pulse_target"], reduction="none").mean(-1)
    pulse = _masked_mean(pulse_point, mask)
    zero = factual.new_zeros(())
    effect = direction = magnitude = zero
    if allow_paired_loss:
        target_effect = output["effect_target"]
        predicted_effect = output["pulse_predicted"] - output["factual_predicted"][:, None, :, :]
        predicted_effect = predicted_effect[:, :, :h]
        # Align absolute factual target t+k+1 with each intervention position.
        aligned = predicted_effect.new_zeros(predicted_effect.shape)
        for position in range(h):
            length = h - position
            aligned[:, position, :length] = output["factual_predicted"][:, position : position + length]
        predicted_effect = output["pulse_predicted"] - aligned
        pair_mask = mask
        effect_point = F.smooth_l1_loss(predicted_effect, target_effect, reduction="none").mean(-1)
        effect = _masked_mean(effect_point, pair_mask)
        true_norm = target_effect.norm(dim=-1)
        selected = pair_mask & (true_norm > effect_threshold)
        direction_point = 1.0 - F.cosine_similarity(predicted_effect, target_effect, dim=-1, eps=1e-6)
        magnitude_point = ((predicted_effect.norm(dim=-1) + 1e-6).log() - (true_norm + 1e-6).log()).abs()
        direction = _masked_mean(direction_point, selected)
        magnitude = _masked_mean(magnitude_point, selected)
    total = factual + pulse + effect_weight * (effect + 0.1 * direction + 0.1 * magnitude)
    return total, {
        "factual_prediction": factual,
        "noop_prediction": pulse,
        "paired_effect": effect,
        "effect_direction": direction,
        "effect_magnitude": magnitude,
    }
