from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _duration_feature(duration: Tensor, max_duration: int) -> Tensor:
    value = duration.float().reshape(-1, 1) / float(max_duration)
    frequencies = torch.arange(1, 5, device=value.device, dtype=value.dtype).reshape(1, -1)
    phase = math.pi * value * frequencies
    return torch.cat((value, torch.sin(phase), torch.cos(phase)), dim=-1)


class MacroActionEncoder(nn.Module):
    """Encode a padded primitive-action chunk into one Gaussian macro action."""

    def __init__(
        self,
        action_dim: int,
        macro_dim: int = 32,
        model_dim: int = 128,
        depth: int = 2,
        heads: int = 4,
        max_duration: int = 40,
    ) -> None:
        super().__init__()
        self.max_duration = max_duration
        self.action_projection = nn.Linear(action_dim, model_dim)
        self.position = nn.Parameter(torch.randn(max_duration, model_dim) * 0.01)
        block = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=4 * model_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=depth)
        self.duration_projection = nn.Linear(9, model_dim)
        self.output = nn.Linear(model_dim, 2 * macro_dim)

    def forward(self, actions: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        if actions.ndim != 3:
            raise ValueError("actions must have shape [batch, time, action_dim]")
        batch, steps, _ = actions.shape
        if steps > self.max_duration:
            raise ValueError("action chunk exceeds configured maximum duration")
        positions = torch.arange(steps, device=actions.device)
        padding_mask = positions.unsqueeze(0) >= lengths.reshape(batch, 1)
        tokens = self.action_projection(actions) + self.position[:steps]
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        pooled = pooled + self.duration_projection(_duration_feature(lengths, self.max_duration))
        mean, log_std = self.output(pooled).chunk(2, dim=-1)
        return mean, log_std.clamp(-5.0, 2.0)

    @staticmethod
    def sample(mean: Tensor, log_std: Tensor) -> Tensor:
        return mean + torch.randn_like(mean) * log_std.exp()


class DurationConditionedMacroPredictor(nn.Module):
    """Predict a temporally extended latent transition from a macro action."""

    def __init__(
        self,
        latent_dim: int,
        macro_dim: int = 32,
        hidden_dim: int = 512,
        depth: int = 3,
        max_duration: int = 40,
    ) -> None:
        super().__init__()
        self.max_duration = max_duration
        input_dim = 2 * latent_dim + macro_dim + 9
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(depth - 1):
            layers.extend((nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU()))
        self.trunk = nn.Sequential(*layers)
        self.delta = nn.Linear(hidden_dim, latent_dim)
        self.gate = nn.Linear(hidden_dim, latent_dim)

    def forward(self, latent: Tensor, macro: Tensor, duration: Tensor) -> Tensor:
        duration_features = _duration_feature(duration, self.max_duration)
        features = torch.cat((latent, latent.square(), macro, duration_features), dim=-1)
        hidden = self.trunk(features)
        return latent + torch.sigmoid(self.gate(hidden)) * self.delta(hidden)


class ExecutabilityRiskHead(nn.Module):
    """Predict endpoint miss, success probability, and per-step residual scale."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 512,
        depth: int = 3,
        max_duration: int = 40,
    ) -> None:
        super().__init__()
        self.max_duration = max_duration
        input_dim = 4 * latent_dim + 9
        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(depth):
            layers.extend((nn.Linear(width, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()))
            width = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.miss = nn.Linear(hidden_dim, 1)
        self.success = nn.Linear(hidden_dim, 1)
        self.scale = nn.Linear(hidden_dim, max_duration)

    def forward(
        self, current: Tensor, subgoal: Tensor, duration: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        features = torch.cat(
            (
                current,
                subgoal,
                subgoal - current,
                (subgoal - current).abs(),
                _duration_feature(duration, self.max_duration),
            ),
            dim=-1,
        )
        hidden = self.trunk(features)
        miss = F.softplus(self.miss(hidden)).squeeze(-1)
        success = torch.sigmoid(self.success(hidden)).squeeze(-1)
        scale = F.softplus(self.scale(hidden)) + 1e-6
        return miss, success, scale


class TrajectoryReachabilityMetric(nn.Module):
    """Horizon-conditioned asymmetric terminal cost trained from trajectory order."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        max_horizon: int = 100,
    ) -> None:
        super().__init__()
        self.max_horizon = max_horizon
        self.network = nn.Sequential(
            nn.Linear(4 * latent_dim + 9, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, source: Tensor, goal: Tensor, horizon: Tensor) -> Tensor:
        features = torch.cat(
            (
                source,
                goal,
                goal - source,
                (goal - source).abs(),
                _duration_feature(horizon, self.max_horizon),
            ),
            dim=-1,
        )
        return F.softplus(self.network(features)).squeeze(-1)


class DirectedReachabilityDistribution(nn.Module):
    """Predict a discrete first-hitting-time distribution in a frozen latent space.

    The final class represents a goal that is not reached within the largest
    registered horizon.  Producing all horizons from one categorical
    distribution makes reachability monotone by construction and gives the
    planner one shared quantity for goal ranking, duration selection, and
    uncertainty estimation.
    """

    def __init__(
        self,
        latent_dim: int,
        horizon_bins: tuple[int, ...] = (5, 10, 20, 40, 80, 120),
        hidden_dim: int = 256,
        depth: int = 3,
        overflow_horizon: int | None = None,
    ) -> None:
        super().__init__()
        if not horizon_bins or any(value <= 0 for value in horizon_bins):
            raise ValueError("horizon bins must be positive")
        if tuple(sorted(set(horizon_bins))) != tuple(horizon_bins):
            raise ValueError("horizon bins must be strictly increasing")
        overflow = int(overflow_horizon or 2 * horizon_bins[-1])
        if overflow <= horizon_bins[-1]:
            raise ValueError("overflow horizon must exceed the largest horizon bin")
        self.latent_dim = int(latent_dim)
        self.horizon_bins = tuple(map(int, horizon_bins))
        self.overflow_horizon = overflow
        self.register_buffer(
            "support",
            torch.tensor((*self.horizon_bins, self.overflow_horizon), dtype=torch.float32),
        )
        input_dim = 4 * latent_dim
        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(depth):
            layers.extend((nn.Linear(width, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()))
            width = hidden_dim
        layers.append(nn.Linear(width, len(self.horizon_bins) + 1))
        self.network = nn.Sequential(*layers)

    def forward(self, source: Tensor, goal: Tensor) -> Tensor:
        if source.shape != goal.shape or source.shape[-1] != self.latent_dim:
            raise ValueError("source and goal must have matching latent shapes")
        delta = goal - source
        features = torch.cat((source, goal, delta, delta.abs()), dim=-1)
        return self.network(features)

    def probabilities(self, source: Tensor, goal: Tensor) -> Tensor:
        return self(source, goal).softmax(dim=-1)

    def cdf(self, source: Tensor, goal: Tensor) -> Tensor:
        """Return P(T <= h) for every registered finite horizon."""

        return self.probabilities(source, goal)[..., :-1].cumsum(dim=-1)

    def expected_and_std(self, source: Tensor, goal: Tensor) -> tuple[Tensor, Tensor]:
        probabilities = self.probabilities(source, goal)
        support = self.support.to(dtype=probabilities.dtype)
        mean = (probabilities * support).sum(dim=-1)
        variance = (probabilities * (support - mean.unsqueeze(-1)).square()).sum(dim=-1)
        return mean, variance.clamp_min(1e-8).sqrt()

    def target_class(self, separation: Tensor, reachable: Tensor | None = None) -> Tensor:
        """Map observed temporal separations to hitting-time classes."""

        bins = self.support[:-1].to(device=separation.device, dtype=separation.dtype)
        target = torch.bucketize(separation, bins, right=False)
        if reachable is not None:
            target = torch.where(
                reachable.bool(), target, torch.full_like(target, len(self.horizon_bins))
            )
        return target.clamp_max(len(self.horizon_bins)).long()


class ContinuousReachabilityDistribution(nn.Module):
    """Directed log-normal first-hitting-time model with arbitrary-horizon CDFs.

    Unlike a coarse categorical head, this distribution retains exact temporal
    separation for local candidate ranking.  Its CDF still supplies monotone
    multi-horizon reachability and its standard deviation supplies the scale
    used by conformal progress calibration.  All three planner quantities are
    therefore derived from one probabilistic prediction rather than separate
    distance, scale, and horizon heads.
    """

    def __init__(
        self,
        latent_dim: int,
        horizon_bins: tuple[int, ...] = (5, 10, 20, 40, 80, 100),
        hidden_dim: int = 256,
        depth: int = 3,
        min_log_scale: float = -3.0,
        max_log_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if not horizon_bins or any(value <= 0 for value in horizon_bins):
            raise ValueError("horizon bins must be positive")
        if tuple(sorted(set(horizon_bins))) != tuple(horizon_bins):
            raise ValueError("horizon bins must be strictly increasing")
        if min_log_scale >= max_log_scale:
            raise ValueError("min_log_scale must be smaller than max_log_scale")
        self.latent_dim = int(latent_dim)
        self.horizon_bins = tuple(map(int, horizon_bins))
        self.min_log_scale = float(min_log_scale)
        self.max_log_scale = float(max_log_scale)
        self.register_buffer(
            "support", torch.tensor(self.horizon_bins, dtype=torch.float32)
        )
        input_dim = 4 * latent_dim
        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(depth):
            layers.extend(
                (nn.Linear(width, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
            )
            width = hidden_dim
        output = nn.Linear(width, 2)
        nn.init.zeros_(output.weight)
        with torch.no_grad():
            output.bias.copy_(
                torch.tensor(
                    [math.log(float(horizon_bins[len(horizon_bins) // 2])), -0.5]
                )
            )
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def parameters_of_distribution(
        self, source: Tensor, goal: Tensor
    ) -> tuple[Tensor, Tensor]:
        if source.shape != goal.shape or source.shape[-1] != self.latent_dim:
            raise ValueError("source and goal must have matching latent shapes")
        delta = goal - source
        raw = self.network(torch.cat((source, goal, delta, delta.abs()), dim=-1))
        location = raw[..., 0]
        log_scale = raw[..., 1].clamp(self.min_log_scale, self.max_log_scale)
        return location, log_scale

    def distribution(self, source: Tensor, goal: Tensor) -> torch.distributions.LogNormal:
        location, log_scale = self.parameters_of_distribution(source, goal)
        return torch.distributions.LogNormal(location, log_scale.exp())

    def expected_and_std(self, source: Tensor, goal: Tensor) -> tuple[Tensor, Tensor]:
        distribution = self.distribution(source, goal)
        return distribution.mean, distribution.stddev.clamp_min(1e-6)

    def cdf(self, source: Tensor, goal: Tensor) -> Tensor:
        """Return monotone ``P(T <= h)`` at every configured horizon."""

        location, log_scale = self.parameters_of_distribution(source, goal)
        support = self.support.to(device=source.device, dtype=source.dtype)
        standard_normal = torch.distributions.Normal(
            torch.zeros_like(location[..., None]),
            torch.ones_like(location[..., None]),
        )
        standardized = (
            support.log() - location[..., None]
        ) / log_scale.exp()[..., None]
        return standard_normal.cdf(standardized)

    def bin_probabilities(self, source: Tensor, goal: Tensor) -> Tensor:
        cdf = self.cdf(source, goal)
        first = cdf[..., :1]
        middle = (cdf[..., 1:] - cdf[..., :-1]).clamp_min(0.0)
        tail = (1.0 - cdf[..., -1:]).clamp_min(0.0)
        return torch.cat((first, middle, tail), dim=-1)

    def target_class(self, separation: Tensor) -> Tensor:
        bins = self.support.to(device=separation.device, dtype=separation.dtype)
        return torch.bucketize(separation, bins, right=False).long()


def continuous_reachability_loss(
    model: ContinuousReachabilityDistribution,
    source: Tensor,
    goal: Tensor,
    separation: Tensor,
    *,
    mean_weight: float = 1.0,
    label_scale: float = 100.0,
) -> Tensor:
    """Exact-time likelihood plus a mean-calibration term on one distribution."""

    target = separation.float().clamp_min(1e-6)
    distribution = model.distribution(source, goal)
    nll = -distribution.log_prob(target).mean()
    mean = distribution.mean
    regression = F.smooth_l1_loss(mean / label_scale, target / label_scale)
    return nll + float(mean_weight) * regression


def reachability_distribution_loss(
    model: DirectedReachabilityDistribution,
    source: Tensor,
    goal: Tensor,
    separation: Tensor,
    reachable: Tensor | None = None,
) -> Tensor:
    target = model.target_class(separation, reachable)
    return F.cross_entropy(model(source, goal), target)


def reachability_semigroup_loss(
    current_expected: Tensor,
    successor_expected: Tensor,
    elapsed: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Enforce D(z_t,g) = d + D(z_{t+d},g) on observed goal-reaching segments."""

    error = F.smooth_l1_loss(
        current_expected,
        elapsed.to(current_expected.dtype) + successor_expected,
        reduction="none",
    )
    if mask is None:
        return error.mean()
    weights = mask.to(error.dtype)
    return (error * weights).sum() / weights.sum().clamp_min(1.0)


@dataclass(frozen=True)
class LossWeights:
    prediction: float = 1.0
    cross_scale: float = 0.25
    macro_kl: float = 1e-4


def macro_kl_loss(mean: Tensor, log_std: Tensor) -> Tensor:
    return -0.5 * (1.0 + 2.0 * log_std - mean.square() - (2.0 * log_std).exp()).mean()


def cross_scale_consistency_loss(long_prediction: Tensor, composed_prediction: Tensor) -> Tensor:
    return F.smooth_l1_loss(long_prediction, composed_prediction)


def risk_head_loss(
    outputs: tuple[Tensor, Tensor, Tensor],
    observed_miss: Tensor,
    success: Tensor,
    observed_residual_scale: Tensor,
    residual_mask: Tensor | None = None,
) -> Tensor:
    predicted_miss, success_probability, predicted_scale = outputs
    miss_loss = F.smooth_l1_loss(predicted_miss, observed_miss)
    success_loss = F.binary_cross_entropy(success_probability, success.float())
    scale_error = F.smooth_l1_loss(
        predicted_scale.log(),
        observed_residual_scale.clamp_min(1e-6).log(),
        reduction="none",
    )
    if residual_mask is None:
        scale_loss = scale_error.mean()
    else:
        mask = residual_mask.to(scale_error.dtype)
        scale_loss = (scale_error * mask).sum() / mask.sum().clamp_min(1.0)
    return miss_loss + success_loss + 0.25 * scale_loss
