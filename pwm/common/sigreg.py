"""
SIGReg (Sketch Isotropic Gaussian Regularizer) for preventing representation collapse.

Ported from le-wm-main/module.py (lines 10-36).
Original implementation by Lucas Maes, Quentin Le Lidec et al.

SIGReg uses the Epps-Pulley statistic to test if projections follow a standard Gaussian
distribution, which prevents the representation from collapsing to a single point.
"""

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """
    Sketch Isotropic Gaussian Regularizer (single-GPU!)

    Uses the Epps-Pulley statistic to enforce that the learned representations
    follow an isotropic Gaussian distribution, preventing representation collapse.

    Args:
        knots: Number of knots for numerical integration (default: 17)
        num_proj: Number of random projections (default: 1024)

    Input:
        proj: Tensor of shape (T, B, D) where T is sequence length, B is batch size, D is dimension
              OR shape (B, T, D) - will be automatically transposed

    Returns:
        Scalar loss value representing deviation from standard Gaussian
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj

        # Create time points for numerical integration
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)

        # Trapezoidal rule weights
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # Edge points get half weight

        # Gaussian window (characteristic function of standard Gaussian)
        window = torch.exp(-t.square() / 2.0)

        # Register as buffers (not parameters, but should be moved to device)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Compute the SIGReg loss.

        Args:
            proj: Embeddings of shape (T, B, D) or (B, T, D)
                  If shape is (B, T, D), it will be transposed to (T, B, D)

        Returns:
            Scalar loss value
        """
        # Handle (B, T, D) input by transposing to (T, B, D)
        if proj.dim() == 3 and proj.size(0) > proj.size(1):
            # Likely (B, T, D) format, transpose to (T, B, D)
            proj = proj.transpose(0, 1)

        # Sample random projections - each column is a unit vector
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))  # Normalize columns to unit length

        # Compute the Epps-Pulley statistic
        # Project embeddings onto random directions and scale by time points
        # x_t shape: (T, B, num_proj, knots)
        x_t = (proj @ A).unsqueeze(-1) * self.t

        # Compute characteristic function error
        # For standard Gaussian, E[cos(tx)] = exp(-t^2/2) and E[sin(tx)] = 0
        # err shape: (T, num_proj, knots)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()

        # Weighted sum over knots (numerical integration)
        # statistic shape: (T, num_proj)
        statistic = (err @ self.weights) * proj.size(-2)  # Scale by batch size

        # Average over projections and time
        return statistic.mean()


if __name__ == "__main__":
    # Simple test
    sigreg = SIGReg(knots=17, num_proj=1024)

    # Test with random Gaussian data (should have low loss)
    gaussian_data = torch.randn(10, 32, 256)  # (T, B, D)
    loss_gaussian = sigreg(gaussian_data)
    print(f"SIGReg loss for Gaussian data: {loss_gaussian.item():.4f}")

    # Test with collapsed data (should have high loss)
    collapsed_data = torch.ones(10, 32, 256) * 0.1
    loss_collapsed = sigreg(collapsed_data)
    print(f"SIGReg loss for collapsed data: {loss_collapsed.item():.4f}")

    # Test with (B, T, D) format
    btd_data = torch.randn(32, 10, 256)  # (B, T, D)
    loss_btd = sigreg(btd_data)
    print(f"SIGReg loss for (B,T,D) format: {loss_btd.item():.4f}")
