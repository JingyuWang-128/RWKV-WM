from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import CalibrationRecord, save_calibration_records
from .policy import StableWorldModelCAPEPolicy
from .tworoom_audit import LatentPositionIndex


class TwoRoomCandidateCollectionPolicy(StableWorldModelCAPEPolicy):
    """Label actual CAPE macro candidates on isolated train/calibration states."""

    def __init__(
        self,
        *args: Any,
        position_index: LatentPositionIndex,
        action_scaler: Any,
        action_block: int,
        record_prefix: str,
        success_threshold: float = 16.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.position_index = position_index
        self.action_scaler = action_scaler
        self.action_block = int(action_block)
        self.record_prefix = str(record_prefix)
        self.success_threshold = float(success_threshold)
        self.candidate_records: list[CalibrationRecord] = []
        self.candidate_metadata: list[dict[str, Any]] = []
        self._decision_ids = np.zeros(len(self.planners), dtype=np.int64)

    def reset(self) -> None:
        super().reset()
        self.candidate_records = []
        self.candidate_metadata = []
        self._decision_ids.fill(0)

    @staticmethod
    def _simulate_boundaries(
        env: Any,
        start: torch.Tensor,
        actions: np.ndarray,
        action_block: int,
    ) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
        position = start.clone()
        speed = float(env.variation_space["agent"]["speed"].value.item())
        boundaries = []
        for index, action in enumerate(actions):
            action_tensor = torch.as_tensor(action, dtype=torch.float32).clamp(-1.0, 1.0)
            position = env._apply_collisions(position, position + action_tensor * speed)
            if (index + 1) % action_block == 0 or index + 1 == len(actions):
                boundaries.append((index, position.clone()))
        return position, boundaries

    def get_action(self, info: dict[str, Any]) -> np.ndarray:
        actions = super().get_action(info)
        for env_index, diagnostic in enumerate(self.last_diagnostics):
            if diagnostic is None or diagnostic.candidate_count == 0:
                continue
            planner = self.planners[env_index]
            current = planner.last_current_latent
            candidates = planner.last_candidates
            if current is None or len(candidates) != diagnostic.candidate_count:
                raise RuntimeError("candidate collection planner snapshot mismatch")
            subgoal_positions = self.position_index.nearest_positions(
                np.stack([candidate.subgoal for candidate in candidates])
            )
            env = self.env.envs[env_index].unwrapped
            original_position = env.agent_position.clone()
            goal_position = np.asarray(env.target_position.detach().cpu(), dtype=np.float32)
            start_position = np.asarray(original_position.detach().cpu(), dtype=np.float32)
            initial_goal_distance = float(np.linalg.norm(start_position - goal_position))
            pending: list[dict[str, Any]] = []
            rendered: list[np.ndarray] = []
            try:
                for candidate_index, (candidate, subgoal_position) in enumerate(
                    zip(candidates, subgoal_positions, strict=True)
                ):
                    raw_actions = self.action_scaler.inverse_transform(
                        np.asarray(candidate.actions, dtype=np.float32)
                    )
                    endpoint, boundaries = self._simulate_boundaries(
                        env, original_position, raw_actions, self.action_block
                    )
                    image_start = len(rendered)
                    for _, position in boundaries:
                        env.agent_position = position.clone()
                        rendered.append(np.asarray(env.render()).copy())
                    endpoint_np = np.asarray(endpoint.detach().cpu(), dtype=np.float32)
                    miss = float(np.linalg.norm(endpoint_np - subgoal_position))
                    pending.append(
                        {
                            "candidate": candidate,
                            "candidate_index": candidate_index,
                            "subgoal_position": subgoal_position,
                            "endpoint": endpoint_np,
                            "miss": miss,
                            "boundaries": boundaries,
                            "image_slice": slice(image_start, len(rendered)),
                            "true_goal_progress": initial_goal_distance
                            - float(np.linalg.norm(endpoint_np - goal_position)),
                        }
                    )
            finally:
                env.agent_position = original_position
            actual_latents = planner.model.batch_encode(rendered)
            decision_id = int(self._decision_ids[env_index])
            selected_index = planner.last_selected_index
            for item in pending:
                candidate = item["candidate"]
                predicted = np.stack(
                    [
                        candidate.predicted_path[
                            min(step_index, len(candidate.predicted_path) - 1)
                        ]
                        for step_index, _ in item["boundaries"]
                    ]
                )
                actual = actual_latents[item["image_slice"]]
                residuals = np.linalg.norm(actual - predicted, axis=1).astype(np.float32)
                record_id = (
                    f"{self.record_prefix}:cape-candidate:env{env_index}:decision{decision_id}:"
                    f"candidate{item['candidate_index']}:duration{candidate.duration}"
                )
                self.candidate_records.append(
                    CalibrationRecord(
                        record_id=record_id,
                        current_latent=np.asarray(current, dtype=np.float32),
                        subgoal_latent=np.asarray(candidate.subgoal, dtype=np.float32),
                        duration=int(candidate.duration),
                        observed_miss=float(item["miss"]),
                        success=bool(item["miss"] <= self.success_threshold),
                        step_residuals=residuals,
                    )
                )
                self.candidate_metadata.append(
                    {
                        "record_id": record_id,
                        "environment_index": env_index,
                        "decision_index": decision_id,
                        "candidate_index": item["candidate_index"],
                        "selected": item["candidate_index"] == selected_index,
                        "duration": int(candidate.duration),
                        "original_feasible": bool(candidate.feasible),
                        "original_score": float(candidate.score),
                        "model_progress": float(candidate.progress),
                        "observed_miss": float(item["miss"]),
                        "success": bool(item["miss"] <= self.success_threshold),
                        "true_goal_progress": float(item["true_goal_progress"]),
                        "endpoint_position": item["endpoint"].tolist(),
                        "subgoal_position": np.asarray(
                            item["subgoal_position"], dtype=np.float32
                        ).tolist(),
                    }
                )
            self._decision_ids[env_index] += 1
        return actions

    def save_candidates(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_calibration_records(path, self.candidate_records)
        metadata_path = path.with_suffix(".metadata.jsonl")
        metadata_path.write_text(
            "".join(json.dumps(item) + "\n" for item in self.candidate_metadata)
        )
