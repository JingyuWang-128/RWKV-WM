"""Counterfactual-centered RWKV world-model research utilities."""

from .b0 import open_loop_rollout_latents
from .branch_dataset import CounterfactualBranchDataset
from .branches import (
    ActionDelayTwoRoomAdapter,
    BranchType,
    TwoRoomBranchAdapter,
)
from .cell import (
    CC_DECAY_LOG_HAZARD,
    CC_DECAY_OFFICIAL_LOGIT,
    CounterfactualCenteredRWKV7Cell,
    RWKV7BlockConfig,
    VanillaRWKV7Cell,
    counterfactual_rwkv7_decay,
    counterfactual_rwkv7_matrix_step,
    rwkv7_matrix_step,
)
from .counterfactual import CounterfactualRWKV7Config, CounterfactualRWKV7WorldPredictor
from .dwm import DWMOutputBaseline, DWMWorldHead
from .fairness import FairnessViolation, RunRecord, audit_run_matrix
from .m5 import M5Run, aggregate_gate_ab, select_effect_weight_across_seeds
from .metrics import CounterfactualCurves, counterfactual_curves
from .predictor import RWKV7WorldModelConfig, VanillaRWKV7WorldPredictor
from .protocol import (
    HorizonSpec,
    build_tworoom_open_loop_arrays,
    file_sha256,
    load_frozen_test_episode_ids,
)
from .state import RWKVMatrixState
from .trainer import M4Trainer, M4TrainerConfig
from .training import BranchBatch, HorizonCurriculum, LossWeights

__all__ = [
    "HorizonSpec",
    "ActionDelayTwoRoomAdapter",
    "BranchBatch",
    "BranchType",
    "CounterfactualBranchDataset",
    "CC_DECAY_LOG_HAZARD",
    "CC_DECAY_OFFICIAL_LOGIT",
    "CounterfactualCenteredRWKV7Cell",
    "CounterfactualCurves",
    "CounterfactualRWKV7Config",
    "CounterfactualRWKV7WorldPredictor",
    "DWMOutputBaseline",
    "DWMWorldHead",
    "FairnessViolation",
    "HorizonCurriculum",
    "LossWeights",
    "M4Trainer",
    "M4TrainerConfig",
    "M5Run",
    "RWKV7BlockConfig",
    "RWKV7WorldModelConfig",
    "RWKVMatrixState",
    "RunRecord",
    "VanillaRWKV7Cell",
    "VanillaRWKV7WorldPredictor",
    "TwoRoomBranchAdapter",
    "audit_run_matrix",
    "aggregate_gate_ab",
    "build_tworoom_open_loop_arrays",
    "counterfactual_rwkv7_matrix_step",
    "counterfactual_rwkv7_decay",
    "counterfactual_curves",
    "file_sha256",
    "load_frozen_test_episode_ids",
    "open_loop_rollout_latents",
    "rwkv7_matrix_step",
    "select_effect_weight_across_seeds",
]
