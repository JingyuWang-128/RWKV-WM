from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

DEFAULT_EFFECT_PAIRS = ((1, 2), (3, 1))


def _validate_rollouts(predicted: Tensor, target: Tensor) -> None:
    if predicted.ndim != 4 or target.shape != predicted.shape:
        raise ValueError("predicted and target must match [batch, branches, time, latent]")
    if predicted.shape[2] <= 0:
        raise ValueError("rollout horizon must be positive")


def paired_effects(
    predicted: Tensor,
    target: Tensor,
    *,
    pairs: tuple[tuple[int, int], ...] = DEFAULT_EFFECT_PAIRS,
) -> tuple[Tensor, Tensor]:
    """Return predicted/true factual-minus-reference effects as [B,P,T,D]."""

    _validate_rollouts(predicted, target)
    if not pairs:
        raise ValueError("at least one effect pair is required")
    branches = predicted.shape[1]
    for factual, reference in pairs:
        if not 0 <= factual < branches or not 0 <= reference < branches:
            raise ValueError("effect pair index is out of range")
    predicted_effect = torch.stack(
        [predicted[:, factual] - predicted[:, reference] for factual, reference in pairs],
        dim=1,
    )
    true_effect = torch.stack(
        [target[:, factual] - target[:, reference] for factual, reference in pairs],
        dim=1,
    )
    return predicted_effect, true_effect


def normalized_curve_auc(curve: Tensor) -> Tensor:
    """Trapezoidal AUC divided by its horizon span (one-step returns itself)."""

    if curve.ndim != 1 or curve.numel() == 0:
        raise ValueError("curve must be a non-empty vector")
    if curve.numel() == 1:
        return curve[0]
    return torch.trapezoid(curve, dx=1.0) / (curve.numel() - 1)


@dataclass(frozen=True, slots=True)
class CounterfactualCurves:
    cee: Tensor
    ced: Tensor
    cer: Tensor
    rollout_rmse: Tensor
    normalized_rollout_rmse: Tensor
    valid_effect_count: Tensor

    def summary(self) -> dict[str, float | list[float] | list[int]]:
        return {
            "cee": self.cee.detach().cpu().tolist(),
            "ced": self.ced.detach().cpu().tolist(),
            "cer": self.cer.detach().cpu().tolist(),
            "rollout_rmse": self.rollout_rmse.detach().cpu().tolist(),
            "normalized_rollout_rmse": self.normalized_rollout_rmse.detach().cpu().tolist(),
            "valid_effect_count": self.valid_effect_count.detach().cpu().tolist(),
            "cee_auc": float(normalized_curve_auc(self.cee)),
            "trajectory_auc": float(normalized_curve_auc(self.rollout_rmse)),
            "normalized_trajectory_auc": float(normalized_curve_auc(self.normalized_rollout_rmse)),
        }


def counterfactual_curves(
    predicted: Tensor,
    target: Tensor,
    target_initial: Tensor,
    *,
    effect_threshold: float = 0.0,
    pairs: tuple[tuple[int, int], ...] = DEFAULT_EFFECT_PAIRS,
    eps: float = 1e-6,
) -> CounterfactualCurves:
    """Compute CEE, CED, CER and ordinary trajectory curves.

    CEE is latent-dimension-normalized effect RMSE, CED is one minus cosine
    direction similarity, and CER is predicted/true effect norm.  CED/CER use
    only true effects above the frozen training threshold.
    """

    _validate_rollouts(predicted, target)
    if target_initial.shape != target.shape[:2] + target.shape[3:]:
        raise ValueError("target_initial must have shape [batch, branches, latent]")
    predicted_effect, true_effect = paired_effects(predicted, target, pairs=pairs)
    effect_error = (predicted_effect - true_effect).square().mean(dim=-1).sqrt()
    true_norm = true_effect.norm(dim=-1)
    predicted_norm = predicted_effect.norm(dim=-1)
    valid = true_norm > effect_threshold
    ced_point = 1.0 - F.cosine_similarity(predicted_effect, true_effect, dim=-1, eps=eps)
    cer_point = predicted_norm / true_norm.clamp_min(eps)

    cee = effect_error.mean(dim=(0, 1))
    counts = valid.sum(dim=(0, 1))
    denominator = counts.clamp_min(1).to(predicted.dtype)
    ced = (ced_point * valid).sum(dim=(0, 1)) / denominator
    cer = (cer_point * valid).sum(dim=(0, 1)) / denominator

    rollout_point = (predicted - target).square().mean(dim=-1).sqrt()
    rollout_rmse = rollout_point.mean(dim=(0, 1))
    # Normalize by the held-out target latent scale per horizon.  Per-sample
    # displacement is not a valid denominator because a legitimate reference
    # branch can have exactly zero physical motion.
    target_scale = target.float().std(dim=(0, 1, 3)).clamp_min(eps)
    normalized_rollout_rmse = rollout_rmse / target_scale
    return CounterfactualCurves(
        cee=cee,
        ced=ced,
        cer=cer,
        rollout_rmse=rollout_rmse,
        normalized_rollout_rmse=normalized_rollout_rmse,
        valid_effect_count=counts,
    )


def validation_score(curves: CounterfactualCurves) -> Tensor:
    return normalized_curve_auc(curves.cee) + 0.5 * normalized_curve_auc(
        curves.normalized_rollout_rmse
    )


def select_effect_weight(
    candidates: list[dict[str, float]],
    *,
    b2_one_step_error: float,
    maximum_degradation: float = 0.03,
) -> dict[str, float]:
    """Select validation score after rejecting >3% one-step degradation."""

    if not candidates or b2_one_step_error <= 0 or maximum_degradation < 0:
        raise ValueError("invalid effect-weight selection inputs")
    required = {"effect_weight", "validation_score", "one_step_error"}
    eligible = []
    for candidate in candidates:
        if not required <= candidate.keys():
            raise ValueError("effect candidate is missing required metrics")
        if candidate["one_step_error"] <= b2_one_step_error * (1 + maximum_degradation):
            eligible.append(candidate)
    if not eligible:
        raise ValueError("all effect weights violate the one-step degradation guard")
    return min(eligible, key=lambda item: item["validation_score"])


@dataclass(slots=True)
class RidgeProbe:
    weight: Tensor
    bias: Tensor
    regularization: float

    def predict(self, features: Tensor) -> Tensor:
        return features @ self.weight + self.bias


def fit_ridge_probe(
    features: Tensor,
    targets: Tensor,
    *,
    regularization: float = 1e-3,
) -> RidgeProbe:
    if features.ndim != 2 or targets.ndim != 2 or features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must be matching matrices")
    if features.shape[0] < 2 or regularization < 0:
        raise ValueError("probe requires at least two samples and non-negative regularization")
    x_mean = features.mean(dim=0, keepdim=True)
    y_mean = targets.mean(dim=0, keepdim=True)
    x = (features - x_mean).double()
    y = (targets - y_mean).double()
    if x.shape[1] <= x.shape[0]:
        identity = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
        weight = torch.linalg.solve(x.T @ x + regularization * identity, x.T @ y)
    else:
        # Matrix-state probes are normally very wide (L*H*Dh*Dh features).
        # The dual ridge form avoids constructing an impractical D x D matrix.
        identity = torch.eye(x.shape[0], device=x.device, dtype=x.dtype)
        weight = x.T @ torch.linalg.solve(x @ x.T + regularization * identity, y)
    bias = y_mean.double().squeeze(0) - x_mean.double().squeeze(0) @ weight
    return RidgeProbe(
        weight=weight.to(features.dtype),
        bias=bias.to(features.dtype),
        regularization=float(regularization),
    )


def probe_r2(probe: RidgeProbe, features: Tensor, targets: Tensor, eps: float = 1e-12) -> Tensor:
    prediction = probe.predict(features)
    residual = (targets - prediction).square().sum(dim=0)
    total = (targets - targets.mean(dim=0, keepdim=True)).square().sum(dim=0)
    valid = total > eps
    if not valid.any():
        return total.new_tensor(float("nan"))
    return (1.0 - residual[valid] / total[valid]).mean()


def nearest_centroid_accuracy(
    train_features: Tensor,
    train_labels: Tensor,
    test_features: Tensor,
    test_labels: Tensor,
) -> Tensor:
    if train_features.ndim != 2 or test_features.ndim != 2:
        raise ValueError("probe features must be matrices")
    labels = torch.unique(train_labels, sorted=True)
    if labels.numel() < 2:
        raise ValueError("classification probe requires at least two classes")
    centroids = torch.stack([train_features[train_labels == label].mean(dim=0) for label in labels])
    prediction = labels[torch.cdist(test_features.float(), centroids.float()).argmin(dim=1)]
    return (prediction == test_labels).float().mean()
