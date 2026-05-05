"""
Common modules for JEPA+RWKV World Model

Includes:
- SIGReg: Sketch Isotropic Gaussian Regularizer for preventing representation collapse
- RWKV6: RWKV-6 predictor with data-dependent decay
- JEPA modules: Self-attention block, lightweight decoder, EMA target encoder
"""

from .sigreg import SIGReg
from .rwkv6 import RWKV6Block, RWKV6Predictor
from .jepa_modules import SelfAttentionBlock, LightweightDecoder, TargetEncoderEMA

__all__ = [
    'SIGReg',
    'RWKV6Block',
    'RWKV6Predictor',
    'SelfAttentionBlock',
    'LightweightDecoder',
    'TargetEncoderEMA',
]
