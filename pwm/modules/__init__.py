"""
Shared modules for JEPA-RWKV World Model MBRL.

This package contains:
- SIGReg: Sketch Isotropic Gaussian Regularizer for preventing feature collapse
- RWKVPredictor: RWKV-based temporal predictor for world model
- SelfAttention: Optional self-attention modules for encoder enhancement
"""

from .sigreg import SIGReg, DecoderWeightScheduler
from .rwkv_predictor import (
    RWKVPredictor,
    RWKVBlock,
    RWKVTimeMixing,
    RWKVChannelMixing,
    DiscreteActionEmbedding,
    ContinuousActionEncoder
)
from .self_attention import (
    SelfAttentionBlock,
    TemporalSelfAttention,
    SpatialSelfAttention,
    EncoderWithSelfAttention
)

__all__ = [
    # SIGReg
    'SIGReg',
    'DecoderWeightScheduler',
    # RWKV
    'RWKVPredictor',
    'RWKVBlock',
    'RWKVTimeMixing',
    'RWKVChannelMixing',
    'DiscreteActionEmbedding',
    'ContinuousActionEncoder',
    # Self-Attention
    'SelfAttentionBlock',
    'TemporalSelfAttention',
    'SpatialSelfAttention',
    'EncoderWithSelfAttention',
]
