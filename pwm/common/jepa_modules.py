"""
JEPA-related modules for world model.

Includes:
- SelfAttentionBlock: Self-attention layer to add after CNN encoder (optional)
- LightweightDecoder: Simplified decoder for optional reconstruction loss
- TargetEncoderEMA: EMA target encoder management (optional)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SelfAttentionBlock(nn.Module):
    """
    Self-attention block to add global receptive field after CNN encoder.

    This helps the CNN encoder capture long-range dependencies that
    local convolutions might miss.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        qkv_bias: Whether to use bias in QKV projections
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.norm = nn.LayerNorm(dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, N, D) where N is spatial positions

        Returns:
            Output tensor of shape (B, N, D)
        """
        B, N, D = x.shape

        # Pre-norm
        x_norm = self.norm(x)

        # QKV projection
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0
        )

        # Reshape and project
        attn = attn.transpose(1, 2).reshape(B, N, D)
        out = self.proj(attn)
        out = self.dropout(out)

        # Residual connection
        return x + out


class LightweightDecoder(nn.Module):
    """
    Lightweight decoder for optional image reconstruction.

    Simpler than full VAE decoder - just 2-3 transposed conv layers.
    Used to enforce physical consistency through reconstruction loss.

    Args:
        embed_dim: Input embedding dimension
        out_ch: Number of encoder output channels (before flattening)
        img_size: Output image size
        stem_ch: Base channel count
        min_res: Minimum resolution (encoder feature map size)
        act: Activation function class
    """

    def __init__(
        self,
        embed_dim: int,
        out_ch: int,
        img_size: int,
        stem_ch: int = 32,
        min_res: int = 4,
        act: type = nn.SiLU,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.min_res = min_res

        # Calculate number of upsampling layers needed
        self.num_layers = int(torch.log2(torch.tensor(img_size // min_res)))

        # Initial projection to feature map
        self.initial_proj = nn.Linear(embed_dim, out_ch * min_res * min_res)

        # Build decoder layers
        layers = []
        in_ch = out_ch
        for i in range(self.num_layers):
            out_ch_layer = stem_ch * (2 ** (self.num_layers - i - 1))
            if i == self.num_layers - 1:
                out_ch_layer = 3  # RGB output

            layers.extend([
                nn.ConvTranspose2d(
                    in_ch, out_ch_layer,
                    kernel_size=4, stride=2, padding=1
                ),
                act() if i < self.num_layers - 1 else nn.Identity(),
            ])
            in_ch = out_ch_layer

        self.decoder = nn.Sequential(*layers)

        # Final activation for image output
        self.final_act = nn.Sigmoid()

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embed: Embedding tensor of shape (B, T, D) or (B, D)

        Returns:
            Reconstructed images of shape (B, T, C, H, W) or (B, C, H, W)
        """
        has_time = embed.dim() == 3

        if has_time:
            B, T, D = embed.shape
            embed = embed.reshape(B * T, D)
        else:
            B = embed.shape[0]

        # Project to feature map
        x = self.initial_proj(embed)
        x = x.view(-1, x.shape[-1] // (self.min_res * self.min_res), self.min_res, self.min_res)

        # Upsample
        x = self.decoder(x)

        # Final activation
        x = self.final_act(x)

        if has_time:
            x = x.view(B, T, *x.shape[1:])

        return x


class TargetEncoderEMA(nn.Module):
    """
    EMA (Exponential Moving Average) target encoder for JEPA.

    Maintains an exponential moving average of the online encoder parameters.
    This provides stable learning targets for the prediction loss.

    Args:
        encoder: Online encoder to track
        decay: EMA decay rate (default: 0.99)
                Higher values = slower update = more stable targets
    """

    def __init__(self, encoder: nn.Module, decay: float = 0.99):
        super().__init__()
        self.decay = decay

        # Create a copy of the encoder
        self.target_encoder = copy.deepcopy(encoder)

        # Freeze target encoder parameters
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, online_encoder: nn.Module):
        """
        Update target encoder with EMA of online encoder.

        Args:
            online_encoder: Online encoder whose parameters to track
        """
        for target_param, online_param in zip(
            self.target_encoder.parameters(),
            online_encoder.parameters()
        ):
            target_param.data.mul_(self.decay).add_(
                online_param.data, alpha=1 - self.decay
            )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through target encoder.

        Args and returns match the underlying encoder.
        """
        return self.target_encoder(*args, **kwargs)


class JEPAEncoder(nn.Module):
    """
    JEPA-style encoder with optional self-attention.

    Wraps a CNN encoder and optionally adds self-attention for global context.

    Args:
        cnn_encoder: Base CNN encoder module
        use_self_attention: Whether to add self-attention after CNN
        num_attention_heads: Number of attention heads (if using self-attention)
        attention_dropout: Dropout rate for attention
    """

    def __init__(
        self,
        cnn_encoder: nn.Module,
        use_self_attention: bool = False,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.cnn_encoder = cnn_encoder
        self.use_self_attention = use_self_attention

        # Get embed dimension from CNN encoder
        self.embed_dim = cnn_encoder.embed

        if use_self_attention:
            self.self_attention = SelfAttentionBlock(
                dim=self.embed_dim,
                num_heads=num_attention_heads,
                dropout=attention_dropout,
            )

            # Projection to flatten spatial dimensions
            self.spatial_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images of shape (B, T, C, H, W) or (B, C, H, W)

        Returns:
            Embeddings of shape (B, T, D) or (B, D)
        """
        has_time = x.dim() == 5

        if has_time:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)

        # CNN encoding
        embed = self.cnn_encoder(x)  # (B*T, D) or (B*T, D, h, w) depending on encoder

        if self.use_self_attention:
            # If embed has spatial dimensions, apply self-attention
            if embed.dim() == 4:
                B_T, D, h, w = embed.shape
                # Reshape to (B*T, h*w, D) for attention
                embed = embed.flatten(2).transpose(1, 2)  # (B*T, h*w, D)
                embed = self.self_attention(embed)
                # Global average pool or take first token
                embed = embed.mean(dim=1)  # (B*T, D)
            else:
                # If embed is already flattened (B*T, D), reshape for attention
                embed = embed.unsqueeze(1)  # (B*T, 1, D)
                embed = self.self_attention(embed)
                embed = embed.squeeze(1)  # (B*T, D)

        if has_time:
            embed = embed.reshape(B, T, -1)

        return embed

    @property
    def embed(self) -> int:
        """Return embedding dimension."""
        return self.embed_dim


class DecayScheduler:
    """
    Scheduler for decoder loss weight decay.

    Implements exponential decay: weight(t) = initial * decay_rate^t

    Args:
        initial_weight: Initial weight value
        decay_rate: Decay rate per step (e.g., 0.9999)
        min_weight: Minimum weight value (default: 0)
    """

    def __init__(
        self,
        initial_weight: float = 1.0,
        decay_rate: float = 0.9999,
        min_weight: float = 0.0,
    ):
        self.initial_weight = initial_weight
        self.decay_rate = decay_rate
        self.min_weight = min_weight

    def get_weight(self, step: int) -> float:
        """
        Get weight at given step.

        Args:
            step: Current training step

        Returns:
            Weight value
        """
        weight = self.initial_weight * (self.decay_rate ** step)
        return max(weight, self.min_weight)


if __name__ == "__main__":
    # Test SelfAttentionBlock
    print("Testing SelfAttentionBlock...")
    attn = SelfAttentionBlock(dim=256, num_heads=8)
    x = torch.randn(4, 16, 256)
    out = attn(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")

    # Test LightweightDecoder
    print("\nTesting LightweightDecoder...")
    decoder = LightweightDecoder(
        embed_dim=512,
        out_ch=256,
        img_size=64,
        stem_ch=32,
        min_res=4,
    )
    embed = torch.randn(4, 10, 512)
    recon = decoder(embed)
    print(f"  Input: {embed.shape}, Output: {recon.shape}")

    # Test DecayScheduler
    print("\nTesting DecayScheduler...")
    scheduler = DecayScheduler(initial_weight=1.0, decay_rate=0.9999)
    for step in [0, 10000, 50000, 100000]:
        weight = scheduler.get_weight(step)
        print(f"  Step {step}: weight = {weight:.6f}")

    # Test TargetEncoderEMA
    print("\nTesting TargetEncoderEMA...")
    online = nn.Linear(256, 128)
    ema_encoder = TargetEncoderEMA(online, decay=0.99)

    # Modify online encoder
    with torch.no_grad():
        online.weight.fill_(1.0)

    # Update EMA
    ema_encoder.update(online)

    print(f"  Online weight mean: {online.weight.mean().item():.4f}")
    print(f"  Target weight mean: {ema_encoder.target_encoder.weight.mean().item():.4f}")
