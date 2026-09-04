from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .models import (
    ContinuousReachabilityDistribution,
    DirectedReachabilityDistribution,
    DurationConditionedMacroPredictor,
    ExecutabilityRiskHead,
    LossWeights,
    MacroActionEncoder,
    TrajectoryReachabilityMetric,
    continuous_reachability_loss,
    cross_scale_consistency_loss,
    macro_kl_loss,
    reachability_distribution_loss,
    reachability_semigroup_loss,
    risk_head_loss,
)


def set_reproducible_seed(seed: int, deterministic: bool = False) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
    }


def train_macro_epoch(
    action_encoder: MacroActionEncoder,
    predictor: DurationConditionedMacroPredictor,
    batches: Iterable[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    weights: LossWeights | None = None,
    grad_clip: float = 10.0,
) -> dict[str, float]:
    """One epoch with direct endpoint and within-segment composition supervision."""

    action_encoder.train()
    predictor.train()
    device = torch.device(device)
    weights = weights or LossWeights()
    totals = {"loss": 0.0, "prediction": 0.0, "cross_scale": 0.0, "kl": 0.0}
    count = 0
    for raw_batch in batches:
        batch = _move(raw_batch, device)
        current = batch["current"].float()
        target = batch["target"].float()
        actions = batch["actions"].float()
        duration = batch["duration"].long()
        mean, log_std = action_encoder(actions, duration)
        macro = action_encoder.sample(mean, log_std)
        prediction = predictor(current, macro, duration)
        prediction_loss = F.smooth_l1_loss(prediction, target)
        kl = macro_kl_loss(mean, log_std)

        # Compose two halves from the same true action sequence. This is applied
        # only where both halves have at least one step.
        half = torch.div(duration, 2, rounding_mode="floor").clamp_min(1)
        max_first = int(half.max().item())
        second_lengths = duration - half
        max_second = int(second_lengths.max().item())
        first_actions = actions[:, :max_first]
        second_actions = actions.new_zeros(actions.shape[0], max_second, actions.shape[-1])
        for index, (length, split) in enumerate(zip(duration.tolist(), half.tolist(), strict=True)):
            tail = actions[index, split:length]
            second_actions[index, : len(tail)] = tail
        mean_1, _ = action_encoder(first_actions, half)
        mean_2, _ = action_encoder(second_actions, second_lengths)
        middle = predictor(current, mean_1, half)
        composed = predictor(middle, mean_2, second_lengths)
        consistency = cross_scale_consistency_loss(prediction, composed)

        loss = (
            weights.prediction * prediction_loss
            + weights.cross_scale * consistency
            + weights.macro_kl * kl
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(action_encoder.parameters()) + list(predictor.parameters()), grad_clip
        )
        optimizer.step()
        totals["loss"] += float(loss.item())
        totals["prediction"] += float(prediction_loss.item())
        totals["cross_scale"] += float(consistency.item())
        totals["kl"] += float(kl.item())
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def train_risk_epoch(
    model: ExecutabilityRiskHead,
    batches: Iterable[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    grad_clip: float = 10.0,
) -> float:
    model.train()
    device = torch.device(device)
    total = 0.0
    count = 0
    for raw_batch in batches:
        batch = _move(raw_batch, device)
        outputs = model(batch["current"], batch["subgoal"], batch["duration"])
        loss = risk_head_loss(
            outputs,
            batch["observed_miss"],
            batch["success"],
            batch["residual_scale"],
            batch.get("residual_mask"),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.item())
        count += 1
    return total / max(count, 1)


def train_trm_epoch(
    model: TrajectoryReachabilityMetric,
    batches: Iterable[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    grad_clip: float = 10.0,
) -> float:
    """Regress temporal separation and rank within-batch shuffled negatives."""

    model.train()
    device = torch.device(device)
    total = 0.0
    count = 0
    for raw_batch in batches:
        batch = _move(raw_batch, device)
        prediction = model(batch["source"], batch["goal"], batch["horizon"])
        loss = F.smooth_l1_loss(prediction, batch["temporal_distance"].float())
        if len(prediction) > 1:
            permutation = torch.roll(torch.arange(len(prediction), device=device), 1)
            valid_negative = batch["trajectory_index"] != batch["trajectory_index"][permutation]
            if torch.any(valid_negative):
                negative = model(batch["source"], batch["goal"][permutation], batch["horizon"])
                ranking = F.relu(0.1 + prediction - negative)[valid_negative].mean()
                loss = loss + 0.5 * ranking
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.item())
        count += 1
    return total / max(count, 1)


def train_reachability_epoch(
    model: DirectedReachabilityDistribution | ContinuousReachabilityDistribution,
    batches: Iterable[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    negative_weight: float = 0.5,
    semigroup_weight: float = 0.25,
    mean_weight: float = 1.0,
    label_scale: float = 100.0,
    grad_clip: float = 10.0,
) -> dict[str, float]:
    """Train directed hitting time with explicitly labelled negatives.

    Optional ``successor`` and ``elapsed`` fields activate semigroup
    consistency. Unknown cross-trajectory pairs are never assumed unreachable;
    a dataset must provide ``negative_goal`` and ``negative_reachable`` when it
    has valid negative labels.
    """

    model.train()
    device = torch.device(device)
    totals = {"loss": 0.0, "positive": 0.0, "negative": 0.0, "semigroup": 0.0}
    count = 0
    for raw_batch in batches:
        batch = _move(raw_batch, device)
        source = batch["source"].float()
        goal = batch["goal"].float()
        separation = batch["separation"].float()
        if isinstance(model, ContinuousReachabilityDistribution):
            positive = continuous_reachability_loss(
                model,
                source,
                goal,
                separation,
                mean_weight=mean_weight,
                label_scale=label_scale,
            )
        else:
            positive = reachability_distribution_loss(model, source, goal, separation)

        negative = positive.new_zeros(())
        if (
            isinstance(model, DirectedReachabilityDistribution)
            and "negative_goal" in batch
            and "negative_reachable" in batch
        ):
            negative = reachability_distribution_loss(
                model,
                source,
                batch["negative_goal"].float(),
                batch.get(
                    "negative_separation",
                    torch.full_like(separation, model.overflow_horizon),
                ).float(),
                reachable=batch["negative_reachable"],
            )

        semigroup = positive.new_zeros(())
        if "successor" in batch and "elapsed" in batch:
            current_expected, _ = model.expected_and_std(source, goal)
            successor_expected, _ = model.expected_and_std(batch["successor"].float(), goal)
            semigroup = reachability_semigroup_loss(
                current_expected,
                successor_expected,
                batch["elapsed"],
                batch.get("semigroup_mask"),
            )

        loss = positive + negative_weight * negative + semigroup_weight * semigroup
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        totals["loss"] += float(loss.item())
        totals["positive"] += float(positive.item())
        totals["negative"] += float(negative.item())
        totals["semigroup"] += float(semigroup.item())
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def save_checkpoint(
    path: str | Path,
    modules: dict[str, torch.nn.Module],
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": "cape_wm_checkpoint_v1",
        "config": config,
        "modules": {name: module.state_dict() for name, module in modules.items()},
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, target)
