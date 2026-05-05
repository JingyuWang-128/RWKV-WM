"""
RWKV-6 Predictor for World Model

RWKV-6 features data-dependent time decay, providing stronger expression capability.
Supports both parallel (training) and recurrent (inference) modes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List


def odd_even_parallel_scan(inputs: List[torch.Tensor], operator):
    """
    Odd/Even Parallel Scan: O(2 log2 N) complexity
    Recursive implementation for parallel sequence processing.

    Args:
        inputs: List of tensors, each of shape (L, B, D)
        operator: Binary associative operator

    Returns:
        List of scanned tensors
    """
    Length = inputs[0].shape[0]

    if Length < 2:
        return inputs

    # Reduce: combine adjacent pairs
    reduced_inputs = operator(
        [inp[:-1][0::2] for inp in inputs],
        [inp[1::2] for inp in inputs]
    )
    odd_inputs = odd_even_parallel_scan(reduced_inputs, operator)

    # Expand: compute even positions
    if Length % 2 == 0:
        even_inputs = operator(
            [inp[:-1] for inp in odd_inputs],
            [inp[2::2] for inp in inputs]
        )
    else:
        even_inputs = operator(
            [inp for inp in odd_inputs],
            [inp[2::2] for inp in inputs]
        )

    # Prepend first element
    even_inputs = [
        torch.cat((inp[0:1], even_inp), dim=0)
        for inp, even_inp in zip(inputs, even_inputs)
    ]

    # Interleave odd and even
    def interleave(odd, even):
        padded_odd = torch.cat((odd, torch.zeros_like(odd[-1:])), dim=0)
        outputs = torch.stack((even, padded_odd[:even.shape[0]]), dim=1)
        outputs = outputs.flatten(0, 1)[:(odd.shape[0] + even.shape[0])]
        return outputs

    outputs = [
        interleave(odd_inp, even_inp)
        for even_inp, odd_inp in zip(even_inputs, odd_inputs)
    ]
    return outputs


def wkv_binary_operator(left: List[torch.Tensor], right: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Binary operator for WKV parallel scan.

    State representation: (a, b, c) where:
        a: accumulated weighted value (numerator component)
        b: accumulated weight (denominator component)
        c: decay coefficient

    Combination rule:
        (a1, b1, c1) * (a2, b2, c2) = (a1 * c2 + a2, b1 * c2 + b2, c1 * c2)
    """
    a1, b1, c1 = left
    a2, b2, c2 = right

    return [
        a1 * c2 + a2,  # accumulated weighted value
        b1 * c2 + b2,  # accumulated weight
        c1 * c2,       # accumulated decay
    ]


class RWKV6TimeMixing(nn.Module):
    """
    RWKV-6 Time Mixing layer with data-dependent decay.

    Key formula:
        w = w_base + tanh(x @ W_w)  (data-dependent decay)
        wkv = (exp(u+k)*v + sum(exp(-cumsum(w)+k_i)*v_i)) / normalizer
        output = sigmoid(r) * wkv

    Args:
        hidden: Hidden dimension
        n_heads: Number of attention heads
        head_size: Dimension per head (default: hidden // n_heads)
    """

    def __init__(self, hidden: int, n_heads: int = 8, head_size: Optional[int] = None):
        super().__init__()
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_size = head_size or (hidden // n_heads)

        assert hidden == n_heads * self.head_size, "hidden must be divisible by n_heads"

        # Time decay parameters (learnable base)
        self.time_decay = nn.Parameter(torch.zeros(n_heads, self.head_size))
        self.time_first = nn.Parameter(torch.zeros(n_heads, self.head_size))  # u (bonus for current)

        # Data-dependent decay projection
        self.time_decay_w = nn.Linear(hidden, n_heads * self.head_size, bias=False)

        # Linear projections for R, K, V
        self.receptance = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.output = nn.Linear(hidden, hidden, bias=False)

        # Time mixing weights (lerp between current and previous token)
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, hidden) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, hidden) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, hidden) * 0.5)
        self.time_mix_w = nn.Parameter(torch.ones(1, 1, hidden) * 0.5)

        # Group normalization for output
        self.ln_x = nn.GroupNorm(n_heads, hidden, eps=1e-5)

        self._init_weights()

    def _init_weights(self):
        # Initialize time decay to reasonable values
        nn.init.uniform_(self.time_decay, -5, -4)
        nn.init.uniform_(self.time_first, -1, 1)

        # Initialize projections
        for module in [self.receptance, self.key, self.value, self.output, self.time_decay_w]:
            if hasattr(module, 'weight'):
                nn.init.orthogonal_(module.weight, gain=0.5)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        parallel: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, T, D) for parallel or (B, D) for recurrent
            state: Previous state for recurrent mode
            parallel: Whether to use parallel (training) or recurrent (inference) mode

        Returns:
            output: Output tensor
            new_state: Updated state (only for recurrent mode)
        """
        if parallel:
            return self._parallel_forward(x), None
        else:
            return self._recurrent_forward(x, state)

    def _parallel_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel forward for training."""
        B, T, D = x.shape
        H, S = self.n_heads, self.head_size

        # Time shift: previous token for mixing
        xx = F.pad(x, (0, 0, 1, -1))  # Shift right by 1

        # Time mixing
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xw = x * self.time_mix_w + xx * (1 - self.time_mix_w)

        # Compute R, K, V
        r = self.receptance(xr).view(B, T, H, S)  # (B, T, H, S)
        k = self.key(xk).view(B, T, H, S)
        v = self.value(xv).view(B, T, H, S)

        # Data-dependent decay: w = w_base + tanh(xw @ W_w)
        w = self.time_decay + torch.tanh(self.time_decay_w(xw)).view(B, T, H, S)
        w = -torch.exp(w)  # Convert to decay factor (negative for exponential decay)

        # Current token bonus
        u = self.time_first  # (H, S)

        # Compute WKV using parallel scan
        wkv = self._parallel_wkv(r, k, v, w, u)  # (B, T, H, S)

        # Apply receptance gate and reshape
        out = torch.sigmoid(r) * wkv
        out = out.view(B, T, D)

        # Group norm and output projection
        out = self.ln_x(out.transpose(1, 2)).transpose(1, 2)
        out = self.output(out)

        return out

    def _parallel_wkv(
        self,
        r: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor
    ) -> torch.Tensor:
        """
        Parallel WKV computation using associative scan.

        Args:
            r: Receptance (B, T, H, S)
            k: Key (B, T, H, S)
            v: Value (B, T, H, S)
            w: Data-dependent decay (B, T, H, S), already negative
            u: Current token bonus (H, S)

        Returns:
            wkv: Weighted key-value (B, T, H, S)
        """
        B, T, H, S = k.shape

        # For numerical stability, work in log space where possible
        # exp(k) * v for numerator, exp(k) for denominator
        ek = torch.exp(k)  # (B, T, H, S)
        ekv = ek * v       # (B, T, H, S)

        # Decay coefficient: exp(w) since w is already negative
        decay = torch.exp(w)  # (B, T, H, S)

        # Reshape for parallel scan: (T, B*H*S)
        ekv_flat = ekv.permute(1, 0, 2, 3).reshape(T, -1)      # (T, B*H*S)
        ek_flat = ek.permute(1, 0, 2, 3).reshape(T, -1)        # (T, B*H*S)
        decay_flat = decay.permute(1, 0, 2, 3).reshape(T, -1)  # (T, B*H*S)

        # Parallel scan to compute cumulative sums with decay
        # State: (accumulated ekv, accumulated ek, accumulated decay)
        scanned = odd_even_parallel_scan(
            [ekv_flat, ek_flat, decay_flat],
            wkv_binary_operator
        )

        # Unpack scanned results
        cum_ekv = scanned[0].reshape(T, B, H, S).permute(1, 0, 2, 3)  # (B, T, H, S)
        cum_ek = scanned[1].reshape(T, B, H, S).permute(1, 0, 2, 3)   # (B, T, H, S)

        # Current token contribution with bonus u
        eu = torch.exp(u)  # (H, S)
        euk = torch.exp(k + u)  # (B, T, H, S)

        # WKV formula: (exp(u+k)*v + past_sum) / (exp(u+k) + past_denom)
        # Shift cumulative sums to get "past" values
        past_ekv = F.pad(cum_ekv[:, :-1], (0, 0, 0, 0, 1, 0))  # (B, T, H, S)
        past_ek = F.pad(cum_ek[:, :-1], (0, 0, 0, 0, 1, 0))    # (B, T, H, S)

        # Apply decay to past values
        past_ekv = past_ekv * decay
        past_ek = past_ek * decay

        numerator = euk * v + past_ekv
        denominator = euk + past_ek + 1e-8  # Add epsilon for numerical stability

        wkv = numerator / denominator

        return wkv

    def _recurrent_forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Recurrent forward for inference (single step)."""
        B, D = x.shape
        H, S = self.n_heads, self.head_size

        # Get previous token from state
        if state is None:
            xx = torch.zeros_like(x)
            state_num = torch.zeros(B, H, S, device=x.device, dtype=x.dtype)
            state_den = torch.zeros(B, H, S, device=x.device, dtype=x.dtype)
        else:
            xx = state.get('xx', torch.zeros_like(x))
            state_num = state.get('num', torch.zeros(B, H, S, device=x.device, dtype=x.dtype))
            state_den = state.get('den', torch.zeros(B, H, S, device=x.device, dtype=x.dtype))

        # Time mixing
        xk = x * self.time_mix_k.squeeze(1) + xx * (1 - self.time_mix_k.squeeze(1))
        xv = x * self.time_mix_v.squeeze(1) + xx * (1 - self.time_mix_v.squeeze(1))
        xr = x * self.time_mix_r.squeeze(1) + xx * (1 - self.time_mix_r.squeeze(1))
        xw = x * self.time_mix_w.squeeze(1) + xx * (1 - self.time_mix_w.squeeze(1))

        # Compute R, K, V
        r = self.receptance(xr).view(B, H, S)
        k = self.key(xk).view(B, H, S)
        v = self.value(xv).view(B, H, S)

        # Data-dependent decay
        w = self.time_decay + torch.tanh(self.time_decay_w(xw)).view(B, H, S)
        decay = torch.exp(-torch.exp(w))  # (B, H, S)

        # Current token with bonus
        u = self.time_first  # (H, S)
        euk = torch.exp(k + u)  # (B, H, S)
        ek = torch.exp(k)       # (B, H, S)

        # Apply decay to past state
        past_num = state_num * decay
        past_den = state_den * decay

        # WKV computation
        numerator = euk * v + past_num
        denominator = euk + past_den + 1e-8
        wkv = numerator / denominator  # (B, H, S)

        # Update state
        new_state_num = past_num + ek * v
        new_state_den = past_den + ek

        # Apply receptance gate
        out = torch.sigmoid(r) * wkv
        out = out.view(B, D)

        # Group norm and output projection
        out = self.ln_x(out.unsqueeze(-1)).squeeze(-1)
        out = self.output(out)

        new_state = {
            'xx': x,
            'num': new_state_num,
            'den': new_state_den,
        }

        return out, new_state


class RWKV6ChannelMixing(nn.Module):
    """
    RWKV-6 Channel Mixing layer (FFN with gating).

    Formula:
        k = x * time_mix_k + prev_x * (1 - time_mix_k)
        r = x * time_mix_r + prev_x * (1 - time_mix_r)
        output = sigmoid(r) * (relu(k)^2 @ W_v)
    """

    def __init__(self, hidden: int, ffn_mult: int = 4):
        super().__init__()
        self.hidden = hidden
        ffn_dim = hidden * ffn_mult

        self.time_mix_k = nn.Parameter(torch.ones(1, 1, hidden) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, hidden) * 0.5)

        self.key = nn.Linear(hidden, ffn_dim, bias=False)
        self.receptance = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(ffn_dim, hidden, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.key.weight, gain=1.0)
        nn.init.orthogonal_(self.receptance.weight, gain=0.5)
        nn.init.zeros_(self.value.weight)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        parallel: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input of shape (B, T, D) for parallel or (B, D) for recurrent
            state: Previous x for recurrent mode
            parallel: Whether to use parallel mode

        Returns:
            output: Output tensor
            new_state: Updated state (previous x)
        """
        if parallel:
            return self._parallel_forward(x), None
        else:
            return self._recurrent_forward(x, state)

    def _parallel_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel forward for training."""
        # Time shift
        xx = F.pad(x, (0, 0, 1, -1))

        # Time mixing
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        # Gated FFN
        k = self.key(xk)
        k = torch.relu(k) ** 2  # Squared ReLU
        r = self.receptance(xr)

        out = torch.sigmoid(r) * self.value(k)
        return out

    def _recurrent_forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recurrent forward for inference."""
        if state is None:
            xx = torch.zeros_like(x)
        else:
            xx = state

        # Time mixing
        xk = x * self.time_mix_k.squeeze(1) + xx * (1 - self.time_mix_k.squeeze(1))
        xr = x * self.time_mix_r.squeeze(1) + xx * (1 - self.time_mix_r.squeeze(1))

        # Gated FFN
        k = self.key(xk)
        k = torch.relu(k) ** 2
        r = self.receptance(xr)

        out = torch.sigmoid(r) * self.value(k)

        return out, x  # Return current x as new state


class RWKV6Block(nn.Module):
    """
    RWKV-6 Block: LayerNorm + TimeMixing + LayerNorm + ChannelMixing

    Args:
        hidden: Hidden dimension
        n_heads: Number of heads for time mixing
        ffn_mult: FFN expansion factor
    """

    def __init__(self, hidden: int, n_heads: int = 8, ffn_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.time_mixing = RWKV6TimeMixing(hidden, n_heads)
        self.channel_mixing = RWKV6ChannelMixing(hidden, ffn_mult)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        parallel: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, T, D) for parallel or (B, D) for recurrent
            state: Previous state for recurrent mode
            parallel: Whether to use parallel mode

        Returns:
            output: Output tensor
            new_state: Updated state
        """
        if state is None:
            tm_state = None
            cm_state = None
        else:
            tm_state = state.get('tm')
            cm_state = state.get('cm')

        # Time mixing with residual
        tm_out, new_tm_state = self.time_mixing(self.ln1(x), tm_state, parallel)
        x = x + tm_out

        # Channel mixing with residual
        cm_out, new_cm_state = self.channel_mixing(self.ln2(x), cm_state, parallel)
        x = x + cm_out

        if parallel:
            return x, None
        else:
            new_state = {
                'tm': new_tm_state,
                'cm': new_cm_state,
            }
            return x, new_state


class RWKV6Predictor(nn.Module):
    """
    RWKV-6 based dynamics predictor for world model.

    Replaces PSSM in the original architecture. Takes embeddings and actions
    as input and predicts next-step embeddings.

    Args:
        embed_dim: Embedding dimension (encoder output)
        hidden: Hidden dimension for RWKV layers
        action_dim: Action dimension
        num_layers: Number of RWKV blocks
        n_heads: Number of attention heads
        ffn_mult: FFN expansion factor
    """

    def __init__(
        self,
        embed_dim: int,
        hidden: int,
        action_dim: int,
        num_layers: int = 4,
        n_heads: int = 8,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.action_dim = action_dim
        self.num_layers = num_layers

        # Input projection (embed + action -> hidden)
        self.inp_proj = nn.Sequential(
            nn.Linear(embed_dim + action_dim, hidden),
            nn.LayerNorm(hidden),
        )

        # RWKV-6 layers
        self.blocks = nn.ModuleList([
            RWKV6Block(hidden, n_heads, ffn_mult)
            for _ in range(num_layers)
        ])

        # Output projection (hidden -> embed_dim)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, embed_dim),
        )

    def forward(
        self,
        embed: torch.Tensor,
        action: torch.Tensor,
        state: Optional[List[Dict[str, torch.Tensor]]] = None,
        parallel: bool = True
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]:
        """
        Forward pass.

        Args:
            embed: Encoder embeddings of shape (B, T, embed_dim) for parallel
                   or (B, embed_dim) for recurrent
            action: Actions of shape (B, T, action_dim) for parallel
                    or (B, action_dim) for recurrent
            state: List of states for each layer (recurrent mode only)
            parallel: Whether to use parallel mode

        Returns:
            pred: Predicted next embeddings of shape (B, T, embed_dim)
            new_state: Updated states for each layer
        """
        # Concatenate embedding and action
        x = torch.cat([embed, action], dim=-1)
        x = self.inp_proj(x)

        # Initialize states if needed
        if state is None:
            state = [None] * self.num_layers

        # Pass through RWKV blocks
        new_states = []
        for i, block in enumerate(self.blocks):
            x, new_state = block(x, state[i], parallel)
            new_states.append(new_state)

        # Project to embedding space
        pred = self.out_proj(x)

        if parallel:
            return pred, None
        else:
            return pred, new_states

    def initial_state(self, batch_size: int, device: torch.device) -> List[Dict[str, torch.Tensor]]:
        """
        Create initial state for recurrent mode.

        Args:
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            List of initial states for each layer
        """
        return [None] * self.num_layers


if __name__ == "__main__":
    # Test RWKV6Predictor
    batch_size = 4
    seq_len = 16
    embed_dim = 256
    hidden = 512
    action_dim = 6

    predictor = RWKV6Predictor(
        embed_dim=embed_dim,
        hidden=hidden,
        action_dim=action_dim,
        num_layers=4,
        n_heads=8,
    )

    # Test parallel mode
    embed = torch.randn(batch_size, seq_len, embed_dim)
    action = torch.randn(batch_size, seq_len, action_dim)

    pred, _ = predictor(embed, action, parallel=True)
    print(f"Parallel mode - Input shape: {embed.shape}, Output shape: {pred.shape}")

    # Test recurrent mode
    state = predictor.initial_state(batch_size, embed.device)
    outputs = []
    for t in range(seq_len):
        out, state = predictor(
            embed[:, t],
            action[:, t],
            state=state,
            parallel=False
        )
        outputs.append(out)

    recurrent_pred = torch.stack(outputs, dim=1)
    print(f"Recurrent mode - Output shape: {recurrent_pred.shape}")

    # Check consistency (should be approximately equal)
    diff = (pred - recurrent_pred).abs().mean()
    print(f"Difference between parallel and recurrent: {diff.item():.6f}")
