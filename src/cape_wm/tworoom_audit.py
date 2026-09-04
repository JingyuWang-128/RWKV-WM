from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from .policy import StableWorldModelCAPEPolicy


class LatentPositionIndex:
    """Map frozen LeWM subgoals to evaluator-only dataset positions."""

    def __init__(self, cache: Path, data: Path, device: torch.device) -> None:
        rows = np.load(cache / "latent_rows.npy")
        latent_values = np.load(cache / "latents.npy", mmap_mode="r")
        with h5py.File(data, "r") as handle:
            positions = np.asarray(handle["pos_agent"][rows], dtype=np.float32)
        self.latents = torch.as_tensor(np.asarray(latent_values).copy(), device=device)
        self.norms = self.latents.square().sum(dim=1)
        self.positions = positions
        self.device = device

    @torch.inference_mode()
    def nearest_positions(self, queries: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(queries, dtype=torch.float32, device=self.device)
        distances = (
            query.square().sum(dim=1, keepdim=True)
            + self.norms[None]
            - 2.0 * query @ self.latents.T
        )
        indices = distances.argmin(dim=1).detach().cpu().numpy()
        return self.positions[indices]


class TwoRoomSameCandidateAuditPolicy(StableWorldModelCAPEPolicy):
    """Evaluate every generated candidate from the exact current simulator state."""

    def __init__(
        self,
        *args: Any,
        position_index: LatentPositionIndex,
        action_scaler: Any,
        success_threshold: float = 16.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.position_index = position_index
        self.action_scaler = action_scaler
        self.success_threshold = float(success_threshold)
        self.audit_records: list[dict[str, Any]] = []

    def reset(self) -> None:
        super().reset()
        self.audit_records = []

    @staticmethod
    def _simulate(env: Any, start: np.ndarray, actions: np.ndarray) -> np.ndarray:
        position = torch.as_tensor(start, dtype=torch.float32)
        speed = float(env.variation_space["agent"]["speed"].value.item())
        for action in actions:
            action_tensor = torch.as_tensor(action, dtype=torch.float32).clamp(-1.0, 1.0)
            position = env._apply_collisions(position, position + action_tensor * speed)
        return position.detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _oracle_path_length(env: Any, start: np.ndarray, target: np.ndarray) -> float:
        """Shortest geometric path through a valid door in the frozen room."""

        wall_axis = int(env.wall_axis)
        crossing_axis = 0 if wall_axis == 1 else 1
        wall_center = float(env.WALL_CENTER)
        if (start[crossing_axis] < wall_center) == (
            target[crossing_axis] < wall_center
        ):
            return float(np.linalg.norm(start - target))

        agent_radius = float(env.variation_space["agent"]["radius"].value.item())
        path_lengths = []
        for index in range(int(env.num_doors)):
            if float(env.door_sizes[index]) < 1.1 * agent_radius:
                continue
            door_coordinate = float(env.door_positions[index])
            door = (
                np.asarray([wall_center, door_coordinate], dtype=np.float32)
                if wall_axis == 1
                else np.asarray([door_coordinate, wall_center], dtype=np.float32)
            )
            path_lengths.append(
                float(np.linalg.norm(start - door) + np.linalg.norm(door - target))
            )
        return min(path_lengths, default=float("inf"))

    def get_action(self, info: dict[str, Any]) -> np.ndarray:
        actions = super().get_action(info)
        for index, diagnostic in enumerate(self.last_diagnostics):
            if diagnostic is None or diagnostic.candidate_count == 0:
                continue
            planner = self.planners[index]
            candidates = planner.last_candidates
            if len(candidates) != diagnostic.candidate_count:
                raise RuntimeError("planner audit candidate snapshot mismatch")
            subgoal_positions = self.position_index.nearest_positions(
                np.stack([candidate.subgoal for candidate in candidates])
            )
            env = self.env.envs[index].unwrapped
            start = np.asarray(env.agent_position.detach().cpu(), dtype=np.float32)
            goal = np.asarray(env.target_position.detach().cpu(), dtype=np.float32)
            initial_goal_distance = float(np.linalg.norm(start - goal))
            audited = []
            execution_succeeded = []
            for candidate, subgoal_position in zip(
                candidates, subgoal_positions, strict=True
            ):
                raw_actions = self.action_scaler.inverse_transform(
                    np.asarray(candidate.actions, dtype=np.float32)
                )
                endpoint = self._simulate(env, start, raw_actions)
                low_level_reaches_subgoal = bool(
                    np.linalg.norm(endpoint - subgoal_position) <= self.success_threshold
                )
                path_length = self._oracle_path_length(env, start, subgoal_position)
                speed = float(env.variation_space["agent"]["speed"].value.item())
                oracle_executable = bool(
                    path_length
                    <= float(candidate.duration) * speed + self.success_threshold
                )
                progress = initial_goal_distance - float(
                    np.linalg.norm(subgoal_position - goal)
                )
                audited.append(
                    {
                        "oracle_executable": oracle_executable,
                        "oracle_goal_progress": progress,
                        "duration": int(candidate.duration),
                        "predicted_miss": float(candidate.predicted_miss),
                        "miss_upper_bound": float(candidate.miss_upper_bound),
                        "success_probability": float(candidate.success_probability),
                        "model_progress": float(candidate.progress),
                        "feasible": bool(candidate.feasible),
                        "score": float(candidate.score),
                    }
                )
                execution_succeeded.append(low_level_reaches_subgoal)
            selected = planner.last_selected_index
            self.audit_records.append(
                {
                    "candidates": audited,
                    "selected_index": selected,
                    "execution_succeeded": bool(
                        selected is not None and execution_succeeded[selected]
                    ),
                }
            )
        return actions

    def save_audit(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.audit_records, indent=2) + "\n")
