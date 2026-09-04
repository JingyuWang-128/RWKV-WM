from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .checkpoint import (
    load_m4_checkpoint,
    restore_m4_training_state,
    save_m4_checkpoint,
)
from .dwm import DWMOutputBaseline, dwm_auxiliary_losses
from .metrics import counterfactual_curves, validation_score
from .training import (
    BranchBatch,
    EarlyStopping,
    HorizonCurriculum,
    LossWeights,
    M4LossOutput,
    m4_counterfactual_loss,
    make_adamw,
    make_warmup_cosine_scheduler,
    rollout_branch_batch,
    weights_for_method,
)


@dataclass(frozen=True, slots=True)
class M4TrainerConfig:
    method: str
    max_steps: int
    curriculum_levels: tuple[int, ...]
    minimum_horizon_steps: int = 5000
    maximum_horizon_steps: int = 10000
    short_horizon_fraction: float = 0.25
    learning_rate: float = 5e-5
    weight_decay: float = 1e-3
    warmup_steps: int = 2000
    gradient_clip: float = 1.0
    dwm_contrastive_weight: float = 0.3
    dwm_orthogonality_weight: float = 0.5
    dwm_temperature: float = 0.07
    stage_b_freeze_steps: int = 1000
    early_stopping_patience: int = 10
    seed: int = 0
    precision: str = "float32"

    def __post_init__(self) -> None:
        if self.method not in {"b2", "b3", "b4", "b6"}:
            raise ValueError("unsupported M4 method")
        if self.max_steps <= 0 or not self.curriculum_levels:
            raise ValueError("training budget and curriculum must be non-empty")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        if self.maximum_horizon_steps < self.minimum_horizon_steps:
            raise ValueError("maximum_horizon_steps must be >= minimum_horizon_steps")
        if not 0 <= self.short_horizon_fraction < 1:
            raise ValueError("short_horizon_fraction must be in [0,1)")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16")


def set_b6_stage_b_freeze(model: nn.Module, freeze_world: bool) -> None:
    """Freeze B6 world/readout parameters while retaining all new action paths."""

    for name, parameter in model.named_parameters():
        action_parameter = "action_" in name
        parameter.requires_grad_(action_parameter if freeze_world else True)


def matrix_action_probe_features(
    model: nn.Module,
    batch: BranchBatch,
    *,
    factual_branch: int = 1,
    reference_branch: int = 2,
) -> tuple[Tensor, Tensor]:
    predictor = getattr(model, "predictor", model)
    state = predictor.consume_history(
        batch.history_latents,
        batch.history_actions,
        mask=batch.history_mask,
    )
    initial = batch.branch_latents[:, factual_branch, 0]
    factual_action = batch.branch_actions[:, factual_branch, 0]
    reference_action = batch.branch_actions[:, reference_branch, 0]
    _, factual_state, _ = predictor.step(initial, factual_action, state, reference_action)
    _, reference_state, _ = predictor.step(initial, reference_action, state, reference_action)
    features = (factual_state.matrix - reference_state.matrix).flatten(1)
    targets = factual_action - reference_action
    return features, targets


class M4Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: M4TrainerConfig,
        *,
        loss_weights: LossWeights | None = None,
        effect_threshold: float,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.config = config
        self.loss_weights = weights_for_method(config.method, loss_weights)
        self.effect_threshold = float(effect_threshold)
        self.optimizer = make_adamw(
            self.model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        warmup = min(config.warmup_steps, max(config.max_steps - 1, 0))
        self.scheduler = make_warmup_cosine_scheduler(
            self.optimizer,
            warmup_steps=warmup,
            total_steps=config.max_steps,
        )
        self.curriculum = HorizonCurriculum(
            config.curriculum_levels,
            minimum_steps=config.minimum_horizon_steps,
            maximum_steps=config.maximum_horizon_steps,
            short_horizon_fraction=config.short_horizon_fraction,
        )
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)
        self.generator = torch.Generator().manual_seed(config.seed + 1701)
        self.global_step = 0
        self.consecutive_nonfinite = 0
        self.history: list[dict[str, Any]] = []
        if config.method == "b6" and config.stage_b_freeze_steps > 0:
            set_b6_stage_b_freeze(self.model, True)

    def _dwm_loss(self, batch: BranchBatch) -> tuple[Tensor, dict[str, Tensor]]:
        if not isinstance(self.model, DWMOutputBaseline):
            zero = batch.history_latents.new_zeros(())
            return zero, {"dwm_world_contrastive": zero, "dwm_orthogonality": zero}
        predictor = self.model.predictor
        state = predictor.consume_history(
            batch.history_latents,
            batch.history_actions,
            mask=batch.history_mask,
        )
        initial = batch.branch_latents[:, 1, 0]
        action = batch.branch_actions[:, 1, 0]
        permutation = torch.randperm(action.shape[0], generator=self.generator).to(action.device)
        alternative = action.index_select(0, permutation)
        prediction, _, actual_info = predictor.step(initial, action, state, return_diagnostics=True)
        _, _, alternative_info = predictor.step(
            initial, alternative, state, return_diagnostics=True
        )
        actual_hidden = actual_info["predictor_hidden"][:, 0]
        alternative_hidden = alternative_info["predictor_hidden"][:, 0]
        world, alternative_world = self.model.world_views(actual_hidden, alternative_hidden)
        losses = dwm_auxiliary_losses(
            prediction,
            world,
            alternative_world,
            temperature=self.config.dwm_temperature,
        )
        total = (
            self.config.dwm_contrastive_weight * losses.world_contrastive
            + self.config.dwm_orthogonality_weight * losses.orthogonality
        )
        return total, {
            "dwm_world_contrastive": losses.world_contrastive,
            "dwm_orthogonality": losses.orthogonality,
        }

    def compute_loss(self, batch: BranchBatch, *, horizon: int) -> M4LossOutput:
        batch = batch.truncate(horizon)
        needs_diagnostics = self.config.method in {"b4", "b6"}
        rollout = rollout_branch_batch(self.model, batch, return_diagnostics=needs_diagnostics)
        base = m4_counterfactual_loss(
            rollout,
            weights=self.loss_weights,
            effect_threshold=self.effect_threshold,
            allow_paired_loss=self.config.method == "b6",
            paired_horizon=1 if self.config.method == "b6" else None,
        )
        dwm_total, dwm_components = self._dwm_loss(batch)
        components = {**base.components, **dwm_components}
        return M4LossOutput(total=base.total + dwm_total, components=components)

    def train_step(self, batch: BranchBatch) -> dict[str, float]:
        if self.config.method == "b6" and self.global_step == self.config.stage_b_freeze_steps:
            set_b6_stage_b_freeze(self.model, False)
            # The optimizer already owns all parameters; unfreezing activates them.
        self.model.train()
        available = batch.horizon
        horizon = min(self.curriculum.sample_horizon(self.generator), available)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.precision == "bfloat16",
        ):
            output = self.compute_loss(batch, horizon=horizon)
        if not torch.isfinite(output.total):
            self.consecutive_nonfinite += 1
            self.optimizer.zero_grad(set_to_none=True)
            if self.consecutive_nonfinite >= 3:
                raise FloatingPointError("three consecutive non-finite M4 optimizer steps")
            return {"skipped_nonfinite": 1.0, "horizon": float(horizon)}
        self.consecutive_nonfinite = 0
        self.optimizer.zero_grad(set_to_none=True)
        output.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip
        )
        if not torch.isfinite(gradient_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("non-finite M4 gradient norm")
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        self.curriculum.step()
        record = {
            "global_step": float(self.global_step),
            "horizon": float(horizon),
            "total": float(output.total.detach()),
            "gradient_norm": float(gradient_norm),
            **{name: float(value.detach()) for name, value in output.components.items()},
        }
        self.history.append(record)
        return record

    @torch.no_grad()
    def evaluate(self, batches: Iterable[BranchBatch], *, horizon: int) -> dict[str, Any]:
        self.model.eval()
        predicted: list[Tensor] = []
        target: list[Tensor] = []
        initial: list[Tensor] = []
        gates: list[Tensor] = []
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.precision == "bfloat16",
        ):
            for batch in batches:
                rollout = rollout_branch_batch(
                    self.model,
                    batch.truncate(min(horizon, batch.horizon)),
                    return_diagnostics=self.config.method in {"b4", "b6"},
                )
                predicted.append(rollout.predicted.float())
                target.append(rollout.target.float())
                initial.append(rollout.target_initial.float())
                if "intervention_gate" in rollout.diagnostics:
                    gates.append(rollout.diagnostics["intervention_gate"].float().flatten())
        curves = counterfactual_curves(
            torch.cat(predicted),
            torch.cat(target),
            torch.cat(initial),
            effect_threshold=self.effect_threshold,
        )
        summary = curves.summary()
        summary["validation_score"] = float(validation_score(curves))
        summary["one_step_rmse"] = float(curves.rollout_rmse[0])
        if gates:
            gate = torch.cat(gates)
            summary["gate"] = {
                "mean": float(gate.mean()),
                "std": float(gate.std()),
                "minimum": float(gate.min()),
                "maximum": float(gate.max()),
                "saturated_fraction": float(((gate < 0.01) | (gate > 0.99)).float().mean()),
            }
        return summary

    def observe_validation(self, score: float) -> tuple[bool, bool]:
        advanced = self.curriculum.observe(score)
        if advanced:
            self.early_stopping = EarlyStopping(patience=self.config.early_stopping_patience)
            return True, False
        should_stop = self.early_stopping.observe(score)
        return advanced, should_stop

    def save(
        self,
        path: str | Path,
        *,
        data_provenance: dict[str, Any],
        encoder_provenance: dict[str, Any],
    ) -> None:
        save_m4_checkpoint(
            path,
            self.model,
            method=self.config.method,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            curriculum_state=self.curriculum.state_dict(),
            early_stopping_state=asdict(self.early_stopping),
            global_step=self.global_step,
            best_metric=self.early_stopping.best,
            training_config={
                "trainer": asdict(self.config),
                "loss_weights": asdict(self.loss_weights),
                "effect_threshold": self.effect_threshold,
            },
            data_provenance=data_provenance,
            encoder_provenance=encoder_provenance,
            history=self.history,
            trainer_rng_state=self.generator.get_state(),
        )

    @classmethod
    def resume(
        cls,
        path: str | Path,
        *,
        device: torch.device | str,
        effect_threshold: float,
        expected_provenance: dict[str, Any],
    ) -> M4Trainer:
        model, payload = load_m4_checkpoint(
            path,
            map_location=device,
            expected_provenance=expected_provenance,
        )
        stored_training = payload["training_config"]
        raw_config = stored_training.get("trainer", stored_training)
        raw_config["curriculum_levels"] = tuple(raw_config["curriculum_levels"])
        config = M4TrainerConfig(**raw_config)
        raw_weights = stored_training.get("loss_weights")
        trainer = cls(
            model,
            config,
            loss_weights=LossWeights(**raw_weights) if raw_weights is not None else None,
            effect_threshold=effect_threshold,
            device=device,
        )
        restore_m4_training_state(payload, optimizer=trainer.optimizer, scheduler=trainer.scheduler)
        trainer.curriculum.load_state_dict(payload["curriculum"])
        for name, value in payload["early_stopping"].items():
            setattr(trainer.early_stopping, name, value)
        trainer.global_step = int(payload["global_step"])
        trainer.history = list(payload["history"])
        trainer_rng_state = payload.get("rng_states", {}).get("trainer")
        if trainer_rng_state is not None:
            trainer.generator.set_state(trainer_rng_state.cpu())
        if config.method == "b6" and trainer.global_step >= config.stage_b_freeze_steps:
            set_b6_stage_b_freeze(trainer.model, False)
        return trainer
