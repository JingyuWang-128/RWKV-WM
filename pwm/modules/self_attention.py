"""
Self-Attention Module for JEPA World Model

Optional self-attention layer to be added after CNN encoder
to expand receptive field and capture global dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SelfAttentionBlock(nn.Module):
    """
    Self-attention block to add after CNN encoder.

    Addresses CNN's limited receptive field by allowing global
    attention across spatial/temporal features.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_flash_attn: Whether to use flash attention (if available)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_flash_attn: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention with residual connection.

        Args:
            x: Input tensor (B, T, D) or (B*T, D)

        Returns:
            Output tensor with same shape as input
        """
        # Handle flattened batch*time input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)
            squeeze_output = True
        else:
            squeeze_output = False

        # Pre-norm
        x_norm = self.norm(x)

        # Self-attention
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        attn_out = self.proj(attn_out)
        attn_out = self.dropout(attn_out)

        # Residual connection
        output = x + attn_out

        if squeeze_output:
            output = output.squeeze(1)

        return output


class TemporalSelfAttention(nn.Module):
    """
    Temporal self-attention specifically for sequence modeling.

    Applies causal (or bidirectional) attention across time steps.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        causal: Whether to use causal masking
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        causal: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal

        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal self-attention.

        Args:
            x: Input tensor (B, T, D)

        Returns:
            Output tensor (B, T, D)
        """
        B, T, D = x.shape

        # Pre-norm
        x_norm = self.norm(x)

        # Create causal mask if needed
        if self.causal:
            mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1
            )
        else:
            mask = None

        # Self-attention
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=mask,
            is_causal=self.causal if hasattr(self.attn, 'is_causal') else False
        )
        attn_out = self.dropout(attn_out)

        # Residual connection
        return x + attn_out


class SpatialSelfAttention(nn.Module):
    """
    Spatial self-attention for feature maps.

    Applies attention across spatial positions of CNN feature maps,
    useful when processing visual observations.

    Args:
        channels: Number of channels in feature map
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial self-attention on feature maps.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, C, H, W)
        """
        B, C, H, W = x.shape
        head_dim = C // self.num_heads

        # Normalize
        x_norm = self.norm(x)

        # Compute Q, K, V
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(B, 3, self.num_heads, head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (B, heads, head_dim, HW)

        # Attention
        scale = head_dim ** -0.5
        attn = torch.einsum('bhdi,bhdj->bhij', q, k) * scale  # (B, heads, HW, HW)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.einsum('bhij,bhdj->bhdi', attn, v)  # (B, heads, head_dim, HW)
        out = out.reshape(B, C, H, W)
        out = self.proj(out)
        out = self.dropout(out)

        # Residual
        return x + out


class EncoderWithSelfAttention(nn.Module):
    """
    Wrapper that adds self-attention to an existing encoder.

    Args:
        encoder: Base encoder module (e.g., CNN encoder)
        use_self_attention: Whether to use self-attention
        num_heads: Number of attention heads
        dropout: Dropout rate
        attention_type: 'temporal' or 'spatial'
    """

    def __init__(
        self,
        encoder: nn.Module,
        use_self_attention: bool = True,
        num_heads: int = 4,
        dropout: float = 0.0,
        attention_type: str = 'temporal'
    ):
        super().__init__()
        self.encoder = encoder
        self.use_self_attention = use_self_attention

        # Get embed dimension from encoder
        if hasattr(encoder, 'embed'):
            embed_dim = encoder.embed
        elif hasattr(encoder, 'embed_dim'):
            embed_dim = encoder.embed_dim
        else:
            raise ValueError("Encoder must have 'embed' or 'embed_dim' attribute")

        self.embed = embed_dim

        if use_self_attention:
            if attention_type == 'temporal':
                self.self_attn = TemporalSelfAttention(
                    embed_dim, num_heads, dropout, causal=False
                )
            else:
                self.self_attn = SelfAttentionBlock(
                    embed_dim, num_heads, dropout
                )
        else:
            self.self_attn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder with optional self-attention.

        Args:
            x: Input tensor (B, T, C, H, W) for images

        Returns:
            Encoded features (B, T, D)
        """
        # Pass through base encoder
        feat = self.encoder(x)  # (B, T, D)

        # Apply self-attention if enabled
        if self.self_attn is not None:
            feat = self.self_attn(feat)

        return feat
