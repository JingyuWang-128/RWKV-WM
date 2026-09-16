from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .dwm import DWMOutputBaseline, dwm_auxiliary_losses
from .state import RWKVMatrixState

ROLLOUT_MASK_POLICY = "position_plus_offset_lt_horizon_v2"


def _read_hdf5_rows(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    """Read a split without h5py's expensive large point-selection path."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices):
        raise ValueError("HDF5 row indices must be a non-empty vector")
    if np.any(indices[1:] <= indices[:-1]):
        raise ValueError("HDF5 row indices must be strictly increasing")
    # For a large split, one contiguous read plus an in-memory gather avoids
    # h5py constructing a huge fancy-selection representation. Small splits
    # retain direct indexed reads so smoke tests do not load the full dataset.
    if len(indices) * 4 >= dataset.shape[0]:
        full = np.asarray(dataset)
        return np.asarray(full[indices])
    return np.asarray(dataset[indices])


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
    def from_mapping(cls, payload: dict[str, Any], device: torch.device | str) -> PerStepBatch:
        names = (
            "history_latents",
            "history_actions",
            "history_mask",
            "factual_actions",
            "factual_latents",
            "pulse_noop_actions",
            "pulse_noop_latents",
            "pulse_noop_mask",
            "effect_latents",
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
            indices = torch.from_numpy(np.flatnonzero(np.asarray(samples["split"]) == codes[split]))
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
                "history_latents",
                "history_actions_raw",
                "history_mask",
                "factual_actions",
                "factual_latents",
                "pulse_noop_actions",
                "pulse_noop_latents",
                "pulse_noop_mask",
            ]
            if load_effect:
                tensor_names.append("effect_latents")
            self.tensors = {
                name: torch.from_numpy(_read_hdf5_rows(samples[name], index))
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

    def random_batch(
        self, batch_size: int, *, generator: torch.Generator, device: torch.device | str
    ) -> PerStepBatch:
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
        RWKVMatrixState(
            matrix=matrix, time_shift=time_shift, channel_shift=channel_shift, steps=step
        )
        for matrix, time_shift, channel_shift, step in zip(
            matrices, time_shifts, channel_shifts, steps, strict=True
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
    state = predictor.consume_history(
        batch.history_latents, batch.history_actions, mask=batch.history_mask
    )
    latent = batch.factual_latents[:, 0].to(state.dtype)
    factual_predictions: list[Tensor] = []
    states_before: list[RWKVMatrixState] = []
    factual_inputs: list[Tensor] = []
    zero = _zero_actions(
        batch.batch_size, batch.factual_actions.shape[-1], latent.device, latent.dtype
    )
    factual_diags: list[dict[str, Tensor]] = []
    for position in range(h):
        states_before.append(state.clone())
        factual_inputs.append(latent)
        prediction, state, diagnostics = predictor.step(
            latent,
            batch.factual_actions[:, position].to(latent.dtype),
            state,
            zero,
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
                action_batch.shape[0],
                action_batch.shape[-1],
                latent_batch.device,
                latent_batch.dtype,
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
                branch_latents[position] = prediction_chunks[position].to(
                    state_chunks[position].dtype
                )
                branch_states[position] = state_chunks[position]
    else:
        for position in range(h):
            branch_state = states_before[position].clone()
            branch_latent = factual_inputs[position]
            for offset in range(h - position):
                absolute = position + offset
                action = (
                    zero
                    if offset == 0
                    else batch.factual_actions[:, absolute].to(branch_latent.dtype)
                )
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

    # Dataset masks describe the stored horizon, not the shorter curriculum
    # horizon. Positions beyond the generated suffix are padding, even when
    # the dataset has a real target there. Never supervise those placeholders.
    positions = torch.arange(h, device=factual_pred.device)
    generated = positions[:, None] + positions[None, :] < h
    mask = batch.pulse_noop_mask[:, :h, :h] & generated[None]
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


def _position_balanced_mean(value: Tensor, mask: Tensor) -> Tensor:
    """Average offsets within each intervention before averaging positions.

    ``value`` and ``mask`` use the triangular ``[batch, position, offset]``
    layout. A flat masked mean would give early positions more weight merely
    because they have longer valid suffixes. This reduction gives every
    sample/position pair with at least one valid offset equal weight.
    """

    if value.shape != mask.shape or value.ndim != 3:
        raise ValueError("position-balanced inputs must share [batch,position,offset]")
    valid = mask.to(value.dtype)
    counts = valid.sum(dim=-1)
    per_position = (value * valid).sum(dim=-1) / counts.clamp_min(1)
    valid_positions = counts > 0
    return (per_position * valid_positions.to(value.dtype)).sum() / valid_positions.sum().clamp_min(
        1
    )


def per_step_dwm_auxiliary_loss(
    model: DWMOutputBaseline,
    batch: PerStepBatch,
    *,
    horizon: int,
    teacher_forcing: bool,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    """Apply the training-only DWM objective at every factual position.

    Both views start from the same factual RWKV state and latent. The factual
    action advances the persistent state, while a deterministic cyclic batch
    permutation supplies the temporary alternative-action view. The latter
    is discarded, so it cannot contaminate the factual rollout state. A
    deterministic permutation keeps batch sampling identical across methods.
    """

    if batch.batch_size < 2:
        raise ValueError("B3 DWM loss requires batch_size >= 2")
    predictor = model.predictor
    state = predictor.consume_history(
        batch.history_latents, batch.history_actions, mask=batch.history_mask
    )
    latent = batch.factual_latents[:, 0].to(state.dtype)
    contrastive: list[Tensor] = []
    orthogonality: list[Tensor] = []
    for position in range(horizon):
        action = batch.factual_actions[:, position].to(latent.dtype)
        alternative = action.roll(shifts=1, dims=0)
        prediction, next_state, factual_info = predictor.step(
            latent, action, state, return_diagnostics=True
        )
        _, _, alternative_info = predictor.step(latent, alternative, state, return_diagnostics=True)
        factual_hidden = factual_info["predictor_hidden"][:, 0]
        alternative_hidden = alternative_info["predictor_hidden"][:, 0]
        world, alternative_world = model.world_views(factual_hidden, alternative_hidden)
        losses = dwm_auxiliary_losses(
            prediction,
            world,
            alternative_world,
            temperature=temperature,
        )
        contrastive.append(losses.world_contrastive)
        orthogonality.append(losses.orthogonality)
        state = next_state
        latent = (
            batch.factual_latents[:, position + 1].to(state.dtype)
            if teacher_forcing
            else prediction.to(state.dtype)
        )
    return torch.stack(contrastive).mean(), torch.stack(orthogonality).mean()


def per_step_loss(
    model: torch.nn.Module,
    batch: PerStepBatch,
    *,
    horizon: int | None = None,
    effect_threshold: float = 0.0,
    allow_paired_loss: bool = True,
    effect_weight: float = 1.0,
    dwm_contrastive_weight: float = 0.3,
    dwm_orthogonality_weight: float = 0.5,
    dwm_temperature: float = 0.07,
    teacher_forcing: bool = False,
) -> tuple[Tensor, dict[str, Tensor]]:
    if effect_weight < 0:
        raise ValueError("effect_weight must be non-negative")
    if min(dwm_contrastive_weight, dwm_orthogonality_weight) < 0:
        raise ValueError("DWM loss weights must be non-negative")
    if dwm_temperature <= 0:
        raise ValueError("DWM temperature must be positive")
    output = per_step_rollout(
        model, batch, horizon=horizon, return_diagnostics=False, teacher_forcing=teacher_forcing
    )
    h = output["factual_predicted"].shape[1]
    factual_point = F.smooth_l1_loss(
        output["factual_predicted"], output["factual_target"], reduction="none"
    ).mean(-1)
    factual = factual_point.mean()
    mask = output["mask"]
    pulse_point = F.smooth_l1_loss(
        output["pulse_predicted"], output["pulse_target"], reduction="none"
    ).mean(-1)
    pulse = _position_balanced_mean(pulse_point, mask)
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
            aligned[:, position, :length] = output["factual_predicted"][
                :, position : position + length
            ]
        predicted_effect = output["pulse_predicted"] - aligned
        pair_mask = mask
        effect_point = F.smooth_l1_loss(predicted_effect, target_effect, reduction="none").mean(-1)
        effect = _position_balanced_mean(effect_point, pair_mask)
        true_norm = target_effect.norm(dim=-1)
        selected = pair_mask & (true_norm > effect_threshold)
        direction_point = 1.0 - F.cosine_similarity(
            predicted_effect, target_effect, dim=-1, eps=1e-6
        )
        magnitude_point = (
            (predicted_effect.norm(dim=-1) + 1e-6).log() - (true_norm + 1e-6).log()
        ).abs()
        direction = _position_balanced_mean(direction_point, selected)
        magnitude = _position_balanced_mean(magnitude_point, selected)
    dwm_contrastive = dwm_orthogonality = zero
    if isinstance(model, DWMOutputBaseline):
        dwm_contrastive, dwm_orthogonality = per_step_dwm_auxiliary_loss(
            model,
            batch,
            horizon=h,
            teacher_forcing=teacher_forcing,
            temperature=dwm_temperature,
        )
    total = (
        factual
        + pulse
        + effect_weight * (effect + 0.1 * direction + 0.1 * magnitude)
        + dwm_contrastive_weight * dwm_contrastive
        + dwm_orthogonality_weight * dwm_orthogonality
    )
    return total, {
        "factual_prediction": factual,
        "noop_prediction": pulse,
        "paired_effect": effect,
        "effect_direction": direction,
        "effect_magnitude": magnitude,
        "dwm_world_contrastive": dwm_contrastive,
        "dwm_orthogonality": dwm_orthogonality,
    }
