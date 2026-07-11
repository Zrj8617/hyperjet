from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

import config


CLEAN_MOVEMENT_UAV_FEATURE_DIM = 6


@dataclass(slots=True)
class CleanMovementObservation:
    uav_ids: list[int]
    uav_features: np.ndarray
    boundary_action_mask: np.ndarray
    ready_task_indices: list[int]
    pending_task_indices: list[int]
    ready_count_normalized: float
    pending_count_normalized: float


if torch is not None:

    class CleanMovementActor(nn.Module):
        """Shared clean movement actor with UAV-specific single-head cross-attention."""

        def __init__(
            self,
            task_embedding_dim: int,
            hidden_dim: int = 64,
            attention_dim: int | None = None,
        ) -> None:
            super().__init__()
            if task_embedding_dim <= 0:
                raise ValueError("task_embedding_dim must be positive.")
            self.task_embedding_dim = int(task_embedding_dim)
            self.attention_dim = int(attention_dim or task_embedding_dim)
            self.action_dim = int(config.CLEAN_MOVEMENT_ACTION_DIM)

            self.uav_proj = nn.Sequential(
                nn.Linear(CLEAN_MOVEMENT_UAV_FEATURE_DIM, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.attention_dim),
                nn.ReLU(),
            )
            # Ready and pending contexts share W_q/W_k/W_v in the first clean version.
            self.query_proj = nn.Linear(self.attention_dim, self.attention_dim)
            self.key_proj = nn.Linear(self.task_embedding_dim, self.attention_dim)
            self.value_proj = nn.Linear(self.task_embedding_dim, self.attention_dim)
            actor_input_dim = self.attention_dim * 3 + 2
            self.policy_head = nn.Sequential(
                nn.Linear(actor_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.action_dim),
            )

        def forward(
            self,
            uav_features: "torch.Tensor",
            task_embeddings: "torch.Tensor",
            ready_task_indices: list[int] | "torch.Tensor",
            pending_task_indices: list[int] | "torch.Tensor",
            ready_count_normalized: float | "torch.Tensor",
            pending_count_normalized: float | "torch.Tensor",
            boundary_action_mask: "torch.Tensor" | None = None,
        ) -> "torch.Tensor":
            if uav_features.dim() != 2:
                raise ValueError("uav_features must be a 2D tensor.")
            if uav_features.shape[1] != CLEAN_MOVEMENT_UAV_FEATURE_DIM:
                raise ValueError("uav_features has unexpected feature dimension.")
            if task_embeddings.dim() != 2:
                raise ValueError("task_embeddings must be a 2D tensor.")
            if task_embeddings.shape[1] != self.task_embedding_dim:
                raise ValueError("task_embeddings has unexpected embedding dimension.")

            uav_hidden = self.uav_proj(uav_features)
            ready_context = self._attend(uav_hidden, task_embeddings, ready_task_indices)
            pending_context = self._attend(uav_hidden, task_embeddings, pending_task_indices)
            counts = self._count_tensor(
                ready_count_normalized,
                pending_count_normalized,
                batch_size=uav_features.shape[0],
                device=uav_features.device,
                dtype=uav_features.dtype,
            )
            actor_input = torch.cat([uav_hidden, ready_context, pending_context, counts], dim=-1)
            logits = self.policy_head(actor_input)
            if boundary_action_mask is not None:
                mask = boundary_action_mask.to(device=logits.device, dtype=torch.bool)
                if mask.shape != logits.shape:
                    raise ValueError("boundary_action_mask shape must match movement logits.")
                logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            return logits

        def _attend(
            self,
            uav_hidden: "torch.Tensor",
            task_embeddings: "torch.Tensor",
            task_indices: list[int] | "torch.Tensor",
        ) -> "torch.Tensor":
            if isinstance(task_indices, torch.Tensor):
                indices = task_indices.to(device=task_embeddings.device, dtype=torch.long)
            else:
                indices = torch.as_tensor(task_indices, device=task_embeddings.device, dtype=torch.long)
            if indices.numel() == 0:
                return uav_hidden.new_zeros((uav_hidden.shape[0], self.attention_dim))

            selected_embeddings = task_embeddings.index_select(0, indices)
            queries = self.query_proj(uav_hidden)
            keys = self.key_proj(selected_embeddings)
            values = self.value_proj(selected_embeddings)
            scores = queries.matmul(keys.transpose(0, 1)) / math.sqrt(float(self.attention_dim))
            weights = torch.softmax(scores, dim=-1)
            return weights.matmul(values)

        def _count_tensor(
            self,
            ready_count_normalized: float | "torch.Tensor",
            pending_count_normalized: float | "torch.Tensor",
            batch_size: int,
            device: "torch.device",
            dtype: "torch.dtype",
        ) -> "torch.Tensor":
            ready = torch.as_tensor(ready_count_normalized, device=device, dtype=dtype).reshape(1)
            pending = torch.as_tensor(pending_count_normalized, device=device, dtype=dtype).reshape(1)
            return torch.stack([ready, pending], dim=-1).expand(batch_size, 2)

else:

    class CleanMovementActor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ModuleNotFoundError("torch is required for CleanMovementActor")


def build_clean_movement_observation(
    *,
    uavs: list[Any],
    executor: Any | None,
    graph_snapshot: Any,
    pre_move_positions: dict[int, Any] | None = None,
    current_time_seconds: float = 0.0,
) -> CleanMovementObservation:
    uav_ids = [int(uav.id) for uav in uavs]
    uav_features = _build_uav_movement_features(
        uavs=uavs,
        executor=executor,
        pre_move_positions=pre_move_positions,
        current_time_seconds=float(current_time_seconds),
    )
    boundary_action_mask = build_boundary_action_mask(uavs=uavs, pre_move_positions=pre_move_positions)
    active_count = max(len(getattr(graph_snapshot, "active_task_ids", getattr(graph_snapshot, "task_ids", []))), 1)
    ready_task_indices = _task_indices(getattr(graph_snapshot, "ready_task_ids", []), graph_snapshot)
    pending_task_indices = _task_indices(getattr(graph_snapshot, "pending_task_ids", []), graph_snapshot)
    return CleanMovementObservation(
        uav_ids=uav_ids,
        uav_features=uav_features,
        boundary_action_mask=boundary_action_mask,
        ready_task_indices=ready_task_indices,
        pending_task_indices=pending_task_indices,
        ready_count_normalized=float(np.clip(len(ready_task_indices) / active_count, 0.0, 1.0)),
        pending_count_normalized=float(np.clip(len(pending_task_indices) / active_count, 0.0, 1.0)),
    )


def build_boundary_action_mask(
    *,
    uavs: list[Any],
    pre_move_positions: dict[int, Any] | None = None,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for uav in uavs:
        uav_id = int(uav.id)
        pre_pos = _pre_move_position(uav, pre_move_positions, uav_id)
        legal = []
        for action_idx, action_name in enumerate(config.CLEAN_MOVEMENT_ACTIONS):
            if action_idx == 0 or action_name == config.CLEAN_MOVEMENT_HOVER_ACTION:
                legal.append(True)
                continue
            candidate = pre_pos + _movement_delta(action_name)
            legal.append(_inside_map(candidate))
        rows.append(np.asarray(legal, dtype=bool))
    if not rows:
        return np.zeros((0, int(config.CLEAN_MOVEMENT_ACTION_DIM)), dtype=bool)
    return np.stack(rows, axis=0)


def _build_uav_movement_features(
    *,
    uavs: list[Any],
    executor: Any | None,
    pre_move_positions: dict[int, Any] | None,
    current_time_seconds: float,
) -> np.ndarray:
    max_queue = max(float(config.CLEAN_MAX_QUEUE_PER_UAV), 1.0)
    max_available_time = max(float(config.EPISODE_LENGTH) * float(config.TIME_SLOT_DURATION), 1.0)
    max_workload = max(
        float(config.UAV_COMPUTE_RATE_OPS_PER_SEC) * float(config.TIME_SLOT_DURATION) * max_queue,
        1.0,
    )
    queues = getattr(executor, "uav_queues", {}) if executor is not None else {}
    available_times = getattr(executor, "uav_available_time", {}) if executor is not None else {}
    task_records = getattr(executor, "task_records", {}) if executor is not None else {}

    rows: list[np.ndarray] = []
    for uav in uavs:
        uav_id = int(uav.id)
        pre_pos = _pre_move_position(uav, pre_move_positions, uav_id)
        queue = list(queues.get(uav_id, []))
        queue_length = len(queue)
        remaining_slots = max(float(config.CLEAN_MAX_QUEUE_PER_UAV - queue_length), 0.0)
        available_time = max(float(available_times.get(uav_id, current_time_seconds)) - current_time_seconds, 0.0)
        queued_workload = 0.0
        for task_id in queue:
            record = task_records.get(str(task_id))
            if record is not None:
                queued_workload += max(float(getattr(record, "compute_time", 0.0)), 0.0) * float(
                    config.UAV_COMPUTE_RATE_OPS_PER_SEC
                )
        rows.append(
            np.asarray(
                [
                    np.clip(float(pre_pos[0]) / float(config.AREA_WIDTH), 0.0, 1.0),
                    np.clip(float(pre_pos[1]) / float(config.AREA_HEIGHT), 0.0, 1.0),
                    np.clip(float(queue_length) / max_queue, 0.0, 1.0),
                    np.clip(remaining_slots / max_queue, 0.0, 1.0),
                    np.clip(available_time / max_available_time, 0.0, 1.0),
                    np.clip(queued_workload / max_workload, 0.0, 1.0),
                ],
                dtype=np.float32,
            )
        )
    if not rows:
        return np.zeros((0, CLEAN_MOVEMENT_UAV_FEATURE_DIM), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32)


def _task_indices(task_ids: list[str], graph_snapshot: Any) -> list[int]:
    mapping = getattr(graph_snapshot, "task_id_to_idx", {})
    indices = [int(mapping[task_id]) for task_id in task_ids if task_id in mapping]
    return sorted(indices)


def _pre_move_position(uav: Any, pre_move_positions: dict[int, Any] | None, uav_id: int) -> np.ndarray:
    if pre_move_positions is not None and uav_id in pre_move_positions:
        return np.asarray(pre_move_positions[uav_id], dtype=np.float32).reshape(-1)[:2].copy()
    return np.asarray(getattr(uav, "pos"), dtype=np.float32).reshape(-1)[:2].copy()


def _movement_delta(action_name: str) -> np.ndarray:
    step_distance = float(config.CLEAN_UAV_MOVEMENT_SPEED) * float(config.TIME_SLOT_DURATION)
    if action_name == "+x":
        return np.array([step_distance, 0.0], dtype=np.float32)
    if action_name == "-x":
        return np.array([-step_distance, 0.0], dtype=np.float32)
    if action_name == "+y":
        return np.array([0.0, step_distance], dtype=np.float32)
    if action_name == "-y":
        return np.array([0.0, -step_distance], dtype=np.float32)
    return np.zeros((2,), dtype=np.float32)


def _inside_map(position: np.ndarray) -> bool:
    return (
        0.0 <= float(position[0]) <= float(config.AREA_WIDTH)
        and 0.0 <= float(position[1]) <= float(config.AREA_HEIGHT)
    )
