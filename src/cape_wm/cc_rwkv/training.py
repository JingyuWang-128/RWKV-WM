from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .predictor import VanillaRWKV7WorldPredictor


@dataclass(frozen=True, slots=True)
class LossWeights:
    prediction: float = 1.0
    effect: float = 1.0
    direction: float = 0.1
    magnitude: float = 0.1
    world: float = 0.5
    gate: float = 0.001
    sigreg: float = 0.0


@dataclass(slots=True)
class BranchBatch:
    history_latents: Tensor
    history_actions: Tensor
    history_mask: Tensor
    branch_actions: Tensor
    branch_latents: Tensor
    branch_mask: Tensor
    sample_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], device: torch.device | str) -> BranchBatch:
        aliases = {
            "history_actions": "history_actions_raw",
            "branch_actions": "branch_actions_raw",
        }
        values: dict[str, Any] = {}
        for field_name in (
            "history_latents",
            "history_actions",
            "history_mask",
            "branch_actions",
            "branch_latents",
            "branch_mask",
        ):
            source = aliases.get(field_name, field_name)
            if source not in payload:
                raise ValueError(f"branch batch is missing {source}")
            values[field_name] = payload[source].to(device)
        raw_ids = payload.get("sample_id", [])
        values["sample_ids"] = [str(item) for item in raw_ids]
        batch = cls(**values)
        batch.validate()
        return batch

    def validate(self) -> None:
        if self.history_latents.ndim != 3 or self.history_actions.ndim != 3:
            raise ValueError("history tensors must be [batch,time,feature]")
        if self.history_latents.shape[:2] != self.history_actions.shape[:2]:
            raise ValueError("history latent/action dimensions must match")
        if self.history_mask.shape != self.history_latents.shape[:2]:
            raise ValueError("history mask shape is invalid")
        if self.branch_actions.ndim != 4 or self.branch_latents.ndim != 4:
            raise ValueError("branch tensors must be rank four")
        if self.branch_latents.shape[:2] != self.branch_actions.shape[:2]:
            raise ValueError("branch batch/count dimensions must match")
        if self.branch_latents.shape[2] != self.branch_actions.shape[2] + 1:
            raise ValueError("branch latents must include the initial latent")
        if self.branch_mask.shape != self.branch_latents.shape[:3]:
            raise ValueError("branch mask shape is invalid")

    @property
    def batch_size(self) -> int:
        return self.history_latents.shape[0]

    @property
    def horizon(self) -> int:
        return self.branch_actions.shape[2]

    def truncate(self, horizon: int) -> BranchBatch:
        if not 0 < horizon <= self.horizon:
            raise ValueError("requested horizon is unavailable")
        return BranchBatch(
            history_latents=self.history_latents,
            history_actions=self.history_actions,
            history_mask=self.history_mask,
            branch_actions=self.branch_actions[:, :, :horizon],
            branch_latents=self.branch_latents[:, :, : horizon + 1],
            branch_mask=self.branch_mask[:, :, : horizon + 1],
            sample_ids=self.sample_ids,
        )


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    predicted: Tensor
    target: Tensor
    target_initial: Tensor
    valid_mask: Tensor
    final_state: Any
    diagnostics: dict[str, Tensor]


def _inference_predictor(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "predictor", model)


def rollout_branch_batch(
    model: torch.nn.Module,
    batch: BranchBatch,
    *,
    return_diagnostics: bool = False,
) -> RolloutBatch:
    """Free-run all branches; no true future latent re-enters the predictor."""

    predictor = _inference_predictor(model)
    state = predictor.consume_history(
        batch.history_latents,
        batch.history_actions,
        mask=batch.history_mask,
    )
    branches = batch.branch_actions.shape[1]
    state = state.clone_branches(branches)
    latent = batch.branch_latents[:, :, 0].reshape(batch.batch_size * branches, -1)
    actions = batch.branch_actions.reshape(batch.batch_size * branches, batch.horizon, -1)
    reference = (
        batch.branch_actions[:, :1]
        .expand(-1, branches, -1, -1)
        .reshape(batch.batch_size * branches, batch.horizon, -1)
    )
    outputs: list[Tensor] = []
    collected: dict[str, list[Tensor]] = {}
    for time_index in range(batch.horizon):
        prediction, state, diagnostics = predictor.step(
            latent,
            actions[:, time_index],
            state,
            reference[:, time_index],
            return_diagnostics=return_diagnostics,
        )
        outputs.append(prediction)
        # Autocast may emit a bf16 prediction while the persistent shift state
        # intentionally remains fp32.  Re-enter the recurrence in the state
        # dtype; the cast stays differentiable and the reported prediction
        # retains its mixed-precision dtype.
        latent = prediction.to(state.dtype)
        if return_diagnostics:
            for name, value in diagnostics.items():
                collected.setdefault(name, []).append(value)
    predicted = torch.stack(outputs, dim=1).reshape(batch.batch_size, branches, batch.horizon, -1)
    diagnostics = {
        name: torch.stack(values, dim=1).reshape(
            batch.batch_size, branches, batch.horizon, *values[0].shape[1:]
        )
        for name, values in collected.items()
    }
    return RolloutBatch(
        predicted=predicted,
        target=batch.branch_latents[:, :, 1:],
        target_initial=batch.branch_latents[:, :, 0],
        valid_mask=batch.branch_mask[:, :, 1:],
        final_state=state,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True, slots=True)
class M4LossOutput:
    total: Tensor
    components: dict[str, Tensor]


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    return (value * mask.to(value.dtype)).sum() / mask.sum().clamp_min(1)


def effect_norm_threshold(
    branch_latents: Tensor,
    *,
    quantile: float = 0.1,
    pairs: tuple[tuple[int, int], ...] = ((1, 2), (3, 1)),
) -> float:
    if not 0 <= quantile < 1:
        raise ValueError("effect threshold quantile must be in [0,1)")
    effects = torch.cat(
        [
            (branch_latents[:, factual, 1:] - branch_latents[:, reference, 1:])
            .norm(dim=-1)
            .flatten()
            for factual, reference in pairs
        ]
    )
    finite = effects[torch.isfinite(effects)]
    if finite.numel() == 0:
        raise ValueError("no finite effect norms")
    return float(torch.quantile(finite.float(), quantile))


def m4_counterfactual_loss(
    rollout: RolloutBatch,
    *,
    weights: LossWeights,
    effect_threshold: float,
    allow_paired_loss: bool,
    paired_horizon: int | None = None,
    sigreg: Tensor | None = None,
    pairs: tuple[tuple[int, int], ...] = ((1, 2), (3, 1)),
    eps: float = 1e-6,
) -> M4LossOutput:
    paired_weight = weights.effect + weights.direction + weights.magnitude
    if not allow_paired_loss and paired_weight != 0:
        raise ValueError("this method is not allowed to consume paired effect labels")
    if paired_horizon is not None and paired_horizon <= 0:
        raise ValueError("paired_horizon must be positive when provided")
    pointwise = F.smooth_l1_loss(rollout.predicted, rollout.target, reduction="none").mean(dim=-1)
    prediction = _masked_mean(pointwise, rollout.valid_mask)
    world_mask = rollout.valid_mask[:, (0, 2)]
    world = _masked_mean(pointwise[:, (0, 2)], world_mask)

    zero = prediction.new_zeros(())
    effect = direction = magnitude = zero
    if allow_paired_loss:
        predicted_effects = []
        true_effects = []
        pair_masks = []
        for factual, reference in pairs:
            predicted_effects.append(
                rollout.predicted[:, factual] - rollout.predicted[:, reference]
            )
            true_effects.append(rollout.target[:, factual] - rollout.target[:, reference])
            pair_masks.append(rollout.valid_mask[:, factual] & rollout.valid_mask[:, reference])
        predicted_effect = torch.stack(predicted_effects, dim=1)
        true_effect = torch.stack(true_effects, dim=1)
        pair_mask = torch.stack(pair_masks, dim=1)
        if paired_horizon is not None:
            used_paired_horizon = min(paired_horizon, predicted_effect.shape[2])
            predicted_effect = predicted_effect[:, :, :used_paired_horizon]
            true_effect = true_effect[:, :, :used_paired_horizon]
            pair_mask = pair_mask[:, :, :used_paired_horizon]
        effect_point = F.smooth_l1_loss(predicted_effect, true_effect, reduction="none").mean(
            dim=-1
        )
        effect = _masked_mean(effect_point, pair_mask)
        true_norm = true_effect.norm(dim=-1)
        selected = pair_mask & (true_norm > effect_threshold)
        direction_point = 1.0 - F.cosine_similarity(predicted_effect, true_effect, dim=-1, eps=eps)
        magnitude_point = (
            (predicted_effect.norm(dim=-1) + eps).log() - (true_norm + eps).log()
        ).abs()
        direction = _masked_mean(direction_point, selected)
        magnitude = _masked_mean(magnitude_point, selected)

    gate = zero
    if "intervention_gate" in rollout.diagnostics:
        gate_value = rollout.diagnostics["intervention_gate"]
        anti_saturation = (
            torch.relu(0.02 - gate_value).square() + torch.relu(gate_value - 0.98).square()
        ).mean()
        gate = gate_value.mean() + 0.1 * anti_saturation
    sigreg_value = zero if sigreg is None else sigreg
    components = {
        "prediction": prediction,
        "effect": effect,
        "direction": direction,
        "magnitude": magnitude,
        "world": world,
        "gate": gate,
        "sigreg": sigreg_value,
    }
    total = (
        weights.prediction * prediction
        + weights.effect * effect
        + weights.direction * direction
        + weights.magnitude * magnitude
        + weights.world * world
        + weights.gate * gate
        + weights.sigreg * sigreg_value
    )
    return M4LossOutput(total=total, components=components)


def weights_for_method(method: str, base: LossWeights | None = None) -> LossWeights:
    base = base or LossWeights()
    if method in {"b2", "b3", "b4"}:
        return LossWeights(
            prediction=base.prediction,
            effect=0.0,
            direction=0.0,
            magnitude=0.0,
            world=0.0,
            gate=base.gate if method == "b4" else 0.0,
            sigreg=base.sigreg,
        )
    if method == "b6":
        return base
    raise ValueError(f"unsupported M4 method: {method}")


@dataclass(slots=True)
class HorizonCurriculum:
    levels: tuple[int, ...]
    minimum_steps: int = 5000
    maximum_steps: int = 10000
    plateau_evaluations: int = 3
    relative_improvement: float = 0.005
    short_horizon_fraction: float = 0.25
    level_index: int = 0
    steps_at_level: int = 0
    best_score: float = float("inf")
    plateau_count: int = 0

    def __post_init__(self) -> None:
        if not self.levels or tuple(sorted(set(self.levels))) != self.levels:
            raise ValueError("curriculum levels must be unique and increasing")
        if (
            self.minimum_steps <= 0
            or self.maximum_steps < self.minimum_steps
            or self.plateau_evaluations <= 0
        ):
            raise ValueError("curriculum budgets must be positive")
        if not 0 <= self.short_horizon_fraction < 1:
            raise ValueError("short_horizon_fraction must be in [0,1)")

    @property
    def current_horizon(self) -> int:
        return self.levels[self.level_index]

    def sample_horizon(self, generator: torch.Generator | None = None) -> int:
        if self.level_index == 0:
            return self.current_horizon
        draw = torch.rand((), generator=generator).item()
        if draw >= self.short_horizon_fraction:
            return self.current_horizon
        index = int(torch.randint(self.level_index, (), generator=generator).item())
        return self.levels[index]

    def step(self) -> None:
        self.steps_at_level += 1

    def observe(self, score: float) -> bool:
        if not math.isfinite(score):
            return False
        improvement = (self.best_score - score) / max(abs(self.best_score), 1e-12)
        if not math.isfinite(self.best_score) or score < self.best_score:
            self.best_score = score
        self.plateau_count = (
            0 if improvement >= self.relative_improvement else self.plateau_count + 1
        )
        eligible_plateau = (
            self.steps_at_level >= self.minimum_steps
            and self.plateau_count >= self.plateau_evaluations
        )
        exhausted = self.steps_at_level >= self.maximum_steps
        can_advance = (eligible_plateau or exhausted) and self.level_index + 1 < len(self.levels)
        if can_advance:
            self.level_index += 1
            self.steps_at_level = 0
            self.best_score = float("inf")
            self.plateau_count = 0
        return can_advance

    def state_dict(self) -> dict[str, Any]:
        return {
            "level_index": self.level_index,
            "steps_at_level": self.steps_at_level,
            "best_score": self.best_score,
            "plateau_count": self.plateau_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name in ("level_index", "steps_at_level", "best_score", "plateau_count"):
            setattr(self, name, state[name])


@dataclass(slots=True)
class EarlyStopping:
    patience: int = 10
    relative_improvement: float = 0.005
    best: float = float("inf")
    bad_evaluations: int = 0

    def observe(self, score: float) -> bool:
        improved = score < self.best * (1.0 - self.relative_improvement)
        if improved or not math.isfinite(self.best):
            self.best = score
            self.bad_evaluations = 0
        else:
            self.bad_evaluations += 1
        return self.bad_evaluations >= self.patience


def make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be in [0,total_steps)")

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def teacher_forced_b2_loss(
    predictor: VanillaRWKV7WorldPredictor,
    latents: Tensor,
    actions: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """One-step teacher-forced Smooth-L1 loss for latent sequences."""

    if latents.ndim != 3 or latents.shape[1] != actions.shape[1] + 1:
        raise ValueError("latents must be [B,T+1,D] and actions [B,T,A]")
    if mask is None:
        mask = torch.ones(actions.shape[:2], device=actions.device, dtype=torch.bool)
    predictions, _, _ = predictor.forward_sequence(latents[:, :-1], actions, mask=mask)
    pointwise = F.smooth_l1_loss(predictions, latents[:, 1:], reduction="none").mean(dim=-1)
    valid = mask.to(pointwise.dtype)
    return (pointwise * valid).sum() / valid.sum().clamp_min(1)


def free_running_b2_loss(
    predictor: VanillaRWKV7WorldPredictor,
    latents: Tensor,
    actions: Tensor,
    *,
    initial_state_steps: int = 0,
) -> Tensor:
    """Fully autoregressive loss; true future latents never re-enter rollout."""

    if latents.ndim != 3 or latents.shape[1] != actions.shape[1] + 1:
        raise ValueError("latents must be [B,T+1,D] and actions [B,T,A]")
    if not 0 <= initial_state_steps < actions.shape[1]:
        raise ValueError("initial_state_steps must be within the action sequence")
    batch = latents.shape[0]
    state = predictor.init_state(batch, device=latents.device, dtype=latents.dtype)
    if initial_state_steps:
        state = predictor.consume_history(
            latents[:, :initial_state_steps], actions[:, :initial_state_steps]
        )
    start = initial_state_steps
    rollout = predictor.rollout(latents[:, start], actions[:, None, start:], state)["latents"][:, 0]
    return F.smooth_l1_loss(rollout, latents[:, start + 1 :])


def make_adamw(
    predictor: VanillaRWKV7WorldPredictor,
    *,
    learning_rate: float = 5e-5,
    weight_decay: float = 1e-3,
) -> torch.optim.AdamW:
    from .predictor import rwkv7_optimizer_groups

    groups = rwkv7_optimizer_groups(predictor, weight_decay=weight_decay)
    optimizer_groups = []
    for group in groups:
        lr_scale = float(group.pop("lr_scale"))
        optimizer_groups.append({**group, "lr": learning_rate * lr_scale})
    return torch.optim.AdamW(optimizer_groups, lr=learning_rate)
