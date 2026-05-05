"""
SIGReg: Sketch Isotropic Gaussian Regularizer

Ported from LeWorldModel (le-wm-main/module.py)
Used to prevent representation collapse in JEPA by enforcing
isotropic Gaussian distribution on feature embeddings.
"""

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """
    Sketch Isotropic Gaussian Regularizer (single-GPU implementation).

    Uses the Epps-Pulley statistical test to measure deviation from
    an isotropic Gaussian distribution N(0, I). Features that collapse
    (become constant or low-rank) will have high SIGReg loss.

    Args:
        knots: Number of evaluation points for the characteristic function (default: 17)
        num_proj: Number of random projections for the sketch (default: 1024)

    Input:
        proj: Tensor of shape (T, B, D) where T=time, B=batch, D=embedding dim
              Or (B, T, D) if transpose_input=True is passed to forward

    Returns:
        Scalar loss measuring deviation from isotropic Gaussian
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj

        # Evaluation points for characteristic function [0, 3]
        t = torch.linspace(0, 3, knots, dtype=torch.float32)

        # Simpson's rule integration weights
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # Endpoint weights

        # Gaussian characteristic function: phi(t) = exp(-t^2/2)
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor, transpose_input: bool = False) -> torch.Tensor:
        """
        Compute the SIGReg loss.

        Args:
            proj: Feature tensor of shape (T, B, D) or (B, T, D) if transpose_input=True
            transpose_input: If True, input is (B, T, D) and will be transposed

        Returns:
            Scalar loss value
        """
        if transpose_input:
            proj = proj.transpose(0, 1)  # (B, T, D) -> (T, B, D)

        # Sample random unit projections: A in R^{D x num_proj}
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))  # Normalize to unit vectors

        # Project features and scale by evaluation points
        # proj @ A: (T, B, num_proj)
        # x_t: (T, B, num_proj, knots)
        x_t = (proj @ A).unsqueeze(-1) * self.t

        # Compute Epps-Pulley statistic
        # Compare empirical characteristic function to Gaussian
        # E[exp(i*t*X)] = cos(t*X) + i*sin(t*X)
        # For N(0,1): phi(t) = exp(-t^2/2)
        cos_term = (x_t.cos().mean(dim=-3) - self.phi).square()  # Mean over batch
        sin_term = x_t.sin().mean(dim=-3).square()  # Should be 0 for symmetric dist
        err = cos_term + sin_term

        # Integrate using Simpson's rule, scale by batch size
        statistic = (err @ self.weights) * proj.size(-2)

        # Average over projections and time
        return statistic.mean()


class DecoderWeightScheduler:
    """
    Scheduler for decoder reconstruction loss weight decay.

    Allows gradual reduction of reconstruction loss importance
    during training, transitioning from pixel-grounded to abstract
    feature prediction.

    Args:
        init_weight: Initial weight for reconstruction loss
        min_weight: Minimum weight (usually 0.0)
        decay_type: 'linear', 'exponential', or 'cosine'
        decay_steps: Number of steps over which to decay
    """

    def __init__(
        self,
        init_weight: float = 1.0,
        min_weight: float = 0.0,
        decay_type: str = 'linear',
        decay_steps: int = 50000
    ):
        self.init_weight = init_weight
        self.min_weight = min_weight
        self.decay_type = decay_type
        self.decay_steps = decay_steps
        self._current_weight = init_weight

    def get_weight(self, step: int) -> float:
        """Get the decoder weight for the current step."""
        if step >= self.decay_steps:
            return self.min_weight

        t = step / self.decay_steps

        if self.decay_type == 'linear':
            weight = self.init_weight * (1 - t) + self.min_weight * t
        elif self.decay_type == 'exponential':
            # Exponential decay from init to min
            ratio = max(self.min_weight / self.init_weight, 1e-8)
            weight = self.init_weight * (ratio ** t)
        elif self.decay_type == 'cosine':
            import math
            # Cosine annealing
            weight = self.min_weight + (self.init_weight - self.min_weight) * \
                     (1 + math.cos(math.pi * t)) / 2
        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")

        self._current_weight = weight
        return weight

    @property
    def current_weight(self) -> float:
        """Get the last computed weight."""
        return self._current_weight
