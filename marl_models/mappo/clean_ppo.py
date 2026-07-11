from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

import config

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None
    Categorical = None


CLEAN_CRITIC_UAV_FIELD_DIM = 6
CLEAN_CRITIC_SUMMARY_DIM = 5


@dataclass(slots=True)
class CleanMovementActionRecord:
    uav_id: int
    movement_observation: Any
    movement_mask: np.ndarray
    selected_action: int
    old_log_probability: float
    entropy: float


@dataclass(slots=True)
class CleanSlotRolloutRecord:
    graph_snapshot: Any
    critic_global_input: np.ndarray
    value: float
    reward: float
    terminated: bool
    truncated: bool
    next_value: float | None = None
    bootstrap_value: float | None = None
    movement_records: list[CleanMovementActionRecord] = field(default_factory=list)
    offloading_records: list[Any] = field(default_factory=list)


@dataclass(slots=True)
class CleanPPOBatchStats:
    movement_action_count: int
    offloading_action_count: int
    offloading_effective_slot_count: int
    slot_count: int


def build_clean_critic_global_input(
    *,
    task_embeddings: np.ndarray,
    graph_snapshot: Any,
    uavs: list[Any],
    executor: Any | None,
    pre_move_positions: dict[int, Any] | None = None,
    current_time_seconds: float = 0.0,
) -> np.ndarray:
    task_embeddings = np.asarray(task_embeddings, dtype=np.float32)
    if task_embeddings.ndim != 2:
        raise ValueError("task_embeddings must be a 2D array.")
    embedding_dim = int(task_embeddings.shape[1])
    active_mean = (
        task_embeddings.mean(axis=0).astype(np.float32)
        if task_embeddings.shape[0] > 0
        else np.zeros((embedding_dim,), dtype=np.float32)
    )
    non_graph = build_clean_critic_non_graph_input(
        graph_snapshot=graph_snapshot,
        uavs=uavs,
        executor=executor,
        pre_move_positions=pre_move_positions,
        current_time_seconds=float(current_time_seconds),
    )
    return np.concatenate([active_mean, non_graph]).astype(np.float32)


def build_clean_critic_non_graph_input(
    *,
    graph_snapshot: Any,
    uavs: list[Any],
    executor: Any | None,
    pre_move_positions: dict[int, Any] | None = None,
    current_time_seconds: float = 0.0,
) -> np.ndarray:
    """Build the critic input slice that does not depend on HGNN parameters."""
    u_global = _critic_uav_global(
        uavs=uavs,
        executor=executor,
        pre_move_positions=pre_move_positions,
        current_time_seconds=float(current_time_seconds),
    )
    active_count = len(getattr(graph_snapshot, "active_task_ids", getattr(graph_snapshot, "task_ids", [])))
    ready_count = len(getattr(graph_snapshot, "ready_task_ids", []))
    pending_count = len(getattr(graph_snapshot, "pending_task_ids", []))
    max_tasks = max(float(getattr(config, "DAG_MAX_TASKS", 1) * max(len(getattr(config, "BASE_UPLOAD_BANDWIDTH_MBPS", [1])), 1)), 1.0)
    counts = np.asarray(
        [
            np.clip(float(active_count) / max_tasks, 0.0, 1.0),
            np.clip(float(ready_count) / max(float(active_count), 1.0), 0.0, 1.0),
            np.clip(float(pending_count) / max(float(active_count), 1.0), 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    q_summary = _critic_queue_summary(
        uavs=uavs,
        executor=executor,
        current_time_seconds=float(current_time_seconds),
    )
    return np.concatenate([u_global, counts, q_summary]).astype(np.float32)


def assemble_clean_critic_global_input(
    *,
    task_embeddings: np.ndarray,
    critic_non_graph_input: np.ndarray,
) -> np.ndarray:
    """Combine current HGNN task embedding pool with historical non-graph critic input."""
    task_embeddings = np.asarray(task_embeddings, dtype=np.float32)
    if task_embeddings.ndim != 2:
        raise ValueError("task_embeddings must be a 2D array.")
    active_mean = (
        task_embeddings.mean(axis=0).astype(np.float32)
        if task_embeddings.shape[0] > 0
        else np.zeros((int(task_embeddings.shape[1]),), dtype=np.float32)
    )
    return np.concatenate([active_mean, np.asarray(critic_non_graph_input, dtype=np.float32).reshape(-1)]).astype(np.float32)


def clean_critic_input_dim(task_embedding_dim: int, num_uavs: int | None = None) -> int:
    return int(task_embedding_dim) + CLEAN_CRITIC_UAV_FIELD_DIM * int(num_uavs or config.NUM_UAVS) + 3 + CLEAN_CRITIC_SUMMARY_DIM


def compute_gae_numpy(
    records: list[CleanSlotRolloutRecord],
    next_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros((len(records),), dtype=np.float32)
    returns = np.zeros((len(records),), dtype=np.float32)
    gae = 0.0
    next_v = float(next_value)
    for idx in range(len(records) - 1, -1, -1):
        record = records[idx]
        mask = 0.0 if bool(record.terminated) else 1.0
        delta = float(record.reward) + float(gamma) * mask * next_v - float(record.value)
        gae = delta + float(gamma) * float(gae_lambda) * mask * gae
        advantages[idx] = float(gae)
        returns[idx] = float(gae + float(record.value))
        next_v = float(record.value)
    return returns, advantages


def summarize_ppo_records(records: list[CleanSlotRolloutRecord]) -> CleanPPOBatchStats:
    offloading_effective_slot_count = sum(1 for record in records if len(record.offloading_records) > 0)
    return CleanPPOBatchStats(
        movement_action_count=sum(len(record.movement_records) for record in records),
        offloading_action_count=sum(len(record.offloading_records) for record in records),
        offloading_effective_slot_count=offloading_effective_slot_count,
        slot_count=len(records),
    )


if torch is not None:

    class CleanCentralizedCritic(nn.Module):
        """Single slot-level centralized critic V(s_t)."""

        def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(int(input_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), 1),
            )

        def forward(self, critic_input: "torch.Tensor") -> "torch.Tensor":
            if critic_input.dim() == 1:
                critic_input = critic_input.unsqueeze(0)
            return self.net(critic_input).squeeze(-1)


    def ppo_clipped_loss_from_log_probs(
        new_log_probs: "torch.Tensor",
        old_log_probs: "torch.Tensor",
        advantages: "torch.Tensor",
        clip_epsilon: float = 0.2,
    ) -> "torch.Tensor":
        ratios = torch.exp(new_log_probs - old_log_probs)
        unclipped = ratios * advantages
        clipped = torch.clamp(ratios, 1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon)) * advantages
        return -torch.minimum(unclipped, clipped)


    def movement_ppo_loss(
        *,
        movement_actor: Any,
        hgnn: Any,
        records: list[CleanSlotRolloutRecord],
        advantages: "torch.Tensor",
        device: str | "torch.device" = "cpu",
        clip_epsilon: float = 0.2,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        per_slot_losses: list[torch.Tensor] = []
        per_slot_entropies: list[torch.Tensor] = []
        for slot_idx, record in enumerate(records):
            if not record.movement_records:
                continue
            task_embeddings = hgnn(record.graph_snapshot.task_features, record.graph_snapshot.incidence_matrix)
            action_losses: list[torch.Tensor] = []
            action_entropies: list[torch.Tensor] = []
            for movement_record in record.movement_records:
                obs = movement_record.movement_observation
                uav_index = list(obs.uav_ids).index(int(movement_record.uav_id))
                logits = movement_actor(
                    uav_features=torch.as_tensor(obs.uav_features, dtype=torch.float32, device=device),
                    task_embeddings=task_embeddings.to(device),
                    ready_task_indices=obs.ready_task_indices,
                    pending_task_indices=obs.pending_task_indices,
                    ready_count_normalized=obs.ready_count_normalized,
                    pending_count_normalized=obs.pending_count_normalized,
                    boundary_action_mask=torch.as_tensor(obs.boundary_action_mask, dtype=torch.bool, device=device),
                )
                dist = Categorical(logits=logits[uav_index])
                action = torch.as_tensor(int(movement_record.selected_action), dtype=torch.long, device=device)
                old_log_prob = torch.as_tensor(float(movement_record.old_log_probability), dtype=torch.float32, device=device)
                action_losses.append(
                    ppo_clipped_loss_from_log_probs(
                        dist.log_prob(action),
                        old_log_prob,
                        advantages[slot_idx].to(device),
                        clip_epsilon=clip_epsilon,
                    )
                )
                action_entropies.append(dist.entropy())
            per_slot_losses.append(torch.stack(action_losses).mean())
            per_slot_entropies.append(torch.stack(action_entropies).mean())
        if not per_slot_losses:
            zero = torch.zeros((), dtype=torch.float32, device=device)
            return zero, zero
        return torch.stack(per_slot_losses).mean(), torch.stack(per_slot_entropies).mean()


    def offloading_ppo_loss(
        *,
        offloading_scorer: Any,
        records: list[CleanSlotRolloutRecord],
        advantages: "torch.Tensor",
        device: str | "torch.device" = "cpu",
        clip_epsilon: float = 0.2,
    ) -> tuple["torch.Tensor", "torch.Tensor", int]:
        per_slot_losses: list[torch.Tensor] = []
        per_slot_entropies: list[torch.Tensor] = []
        for slot_idx, record in enumerate(records):
            if not record.offloading_records:
                continue
            action_losses: list[torch.Tensor] = []
            action_entropies: list[torch.Tensor] = []
            for offloading_record in record.offloading_records:
                features = offloading_record.candidate_features.to(device=device, dtype=torch.float32)
                mask = offloading_record.candidate_mask.to(device=device, dtype=torch.bool)
                logits = offloading_scorer(features).masked_fill(~mask, torch.finfo(torch.float32).min)
                dist = Categorical(logits=logits)
                action = torch.as_tensor(int(offloading_record.selected_action), dtype=torch.long, device=device)
                old_log_prob = torch.as_tensor(float(offloading_record.old_log_prob), dtype=torch.float32, device=device)
                action_losses.append(
                    ppo_clipped_loss_from_log_probs(
                        dist.log_prob(action),
                        old_log_prob,
                        advantages[slot_idx].to(device),
                        clip_epsilon=clip_epsilon,
                    )
                )
                action_entropies.append(dist.entropy())
            per_slot_losses.append(torch.stack(action_losses).mean())
            per_slot_entropies.append(torch.stack(action_entropies).mean())
        if not per_slot_losses:
            zero = torch.zeros((), dtype=torch.float32, device=device)
            return zero, zero, 0
        return torch.stack(per_slot_losses).mean(), torch.stack(per_slot_entropies).mean(), len(per_slot_losses)

else:

    class CleanCentralizedCritic:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ModuleNotFoundError("torch is required for CleanCentralizedCritic")


    def ppo_clipped_loss_from_log_probs(*args: Any, **kwargs: Any) -> Any:
        raise ModuleNotFoundError("torch is required for PPO loss computation")


    def movement_ppo_loss(*args: Any, **kwargs: Any) -> Any:
        raise ModuleNotFoundError("torch is required for movement PPO loss")


    def offloading_ppo_loss(*args: Any, **kwargs: Any) -> Any:
        raise ModuleNotFoundError("torch is required for offloading PPO loss")


def _critic_uav_global(
    *,
    uavs: list[Any],
    executor: Any | None,
    pre_move_positions: dict[int, Any] | None,
    current_time_seconds: float,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    queues = getattr(executor, "uav_queues", {}) if executor is not None else {}
    available_times = getattr(executor, "uav_available_time", {}) if executor is not None else {}
    task_records = getattr(executor, "task_records", {}) if executor is not None else {}
    max_queue = max(float(config.CLEAN_MAX_QUEUE_PER_UAV), 1.0)
    max_available_time = max(float(config.EPISODE_LENGTH) * float(config.TIME_SLOT_DURATION), 1.0)
    max_workload = max(float(config.UAV_COMPUTE_RATE_OPS_PER_SEC) * float(config.TIME_SLOT_DURATION) * max_queue, 1.0)
    for uav in sorted(uavs, key=lambda item: int(item.id)):
        uav_id = int(uav.id)
        if pre_move_positions is not None and uav_id in pre_move_positions:
            pos = np.asarray(pre_move_positions[uav_id], dtype=np.float32).reshape(-1)[:2]
        else:
            pos = np.asarray(getattr(uav, "pos"), dtype=np.float32).reshape(-1)[:2]
        queue = list(queues.get(uav_id, []))
        queue_length = float(len(queue))
        remaining_capacity = max(float(config.CLEAN_MAX_QUEUE_PER_UAV) - queue_length, 0.0)
        available_delta = max(float(available_times.get(uav_id, current_time_seconds)) - float(current_time_seconds), 0.0)
        workload = 0.0
        for task_id in queue:
            record = task_records.get(str(task_id))
            if record is not None:
                workload += max(float(getattr(record, "compute_time", 0.0)), 0.0) * float(config.UAV_COMPUTE_RATE_OPS_PER_SEC)
        rows.append(
            np.asarray(
                [
                    np.clip(float(pos[0]) / float(config.AREA_WIDTH), 0.0, 1.0),
                    np.clip(float(pos[1]) / float(config.AREA_HEIGHT), 0.0, 1.0),
                    np.clip(queue_length / max_queue, 0.0, 1.0),
                    np.clip(remaining_capacity / max_queue, 0.0, 1.0),
                    np.clip(available_delta / max_available_time, 0.0, 1.0),
                    np.clip(workload / max_workload, 0.0, 1.0),
                ],
                dtype=np.float32,
            )
        )
    if not rows:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(rows).astype(np.float32)


def _critic_queue_summary(
    *,
    uavs: list[Any],
    executor: Any | None,
    current_time_seconds: float,
) -> np.ndarray:
    queues = getattr(executor, "uav_queues", {}) if executor is not None else {}
    available_times = getattr(executor, "uav_available_time", {}) if executor is not None else {}
    task_records = getattr(executor, "task_records", {}) if executor is not None else {}
    queue_lengths: list[float] = []
    available_deltas: list[float] = []
    workloads: list[float] = []
    for uav in sorted(uavs, key=lambda item: int(item.id)):
        uav_id = int(uav.id)
        queue = list(queues.get(uav_id, []))
        queue_lengths.append(float(len(queue)))
        available_deltas.append(max(float(available_times.get(uav_id, current_time_seconds)) - float(current_time_seconds), 0.0))
        workload = 0.0
        for task_id in queue:
            record = task_records.get(str(task_id))
            if record is not None:
                workload += max(float(getattr(record, "compute_time", 0.0)), 0.0) * float(config.UAV_COMPUTE_RATE_OPS_PER_SEC)
        workloads.append(workload)
    max_queue = max(float(config.CLEAN_MAX_QUEUE_PER_UAV), 1.0)
    max_available_time = max(float(config.EPISODE_LENGTH) * float(config.TIME_SLOT_DURATION), 1.0)
    max_workload = max(float(config.UAV_COMPUTE_RATE_OPS_PER_SEC) * float(config.TIME_SLOT_DURATION) * max_queue * max(len(uavs), 1), 1.0)
    queue_arr = np.asarray(queue_lengths, dtype=np.float32)
    available_arr = np.asarray(available_deltas, dtype=np.float32)
    workload_arr = np.asarray(workloads, dtype=np.float32)
    return np.asarray(
        [
            np.clip(float(queue_arr.mean()) / max_queue if queue_arr.size else 0.0, 0.0, 1.0),
            np.clip(float(queue_arr.max()) / max_queue if queue_arr.size else 0.0, 0.0, 1.0),
            np.clip(float(available_arr.mean()) / max_available_time if available_arr.size else 0.0, 0.0, 1.0),
            np.clip(float(available_arr.max()) / max_available_time if available_arr.size else 0.0, 0.0, 1.0),
            np.clip(float(workload_arr.sum()) / max_workload if workload_arr.size else 0.0, 0.0, 1.0),
        ],
        dtype=np.float32,
    )
