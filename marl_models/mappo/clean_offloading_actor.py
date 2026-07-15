from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from environment.assignment import (
    CLEAN_OFFLOADING_PAIR_FEATURE_DIM,
    CLEAN_OFFLOADING_UAV_FEATURE_DIM,
    CleanAssignmentBuffer,
    TemporaryReservationState,
    build_offloading_candidate_components,
)


@dataclass(slots=True)
class CleanOffloadingActionRecord:
    task_id: str
    dag_id: str
    task_local_index: int
    decision_order: int
    candidate_uav_ids: list[int]
    candidate_features: torch.Tensor
    dynamic_uav_features: torch.Tensor
    pair_features: torch.Tensor
    candidate_mask: torch.Tensor
    selected_action: int
    selected_uav_id: int
    old_log_prob: float
    entropy: float
    selected_estimated_finish_time: float
    selected_estimated_incremental_delay: float


class SharedOffloadingCandidateScorer(nn.Module):
    """Shared f_off scorer applied to each task-UAV candidate row."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        if candidate_features.dim() != 2:
            raise ValueError("candidate_features must be a 2D tensor.")
        return self.net(candidate_features).squeeze(-1)


class CleanOffloadingActor(nn.Module):
    """Sequential clean offloading actor skeleton.

    The actor consumes frozen ready tasks in order and updates only temporary
    reservation state until the caller commits the resulting assignment buffer.
    """

    def __init__(self, task_embedding_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.task_embedding_dim = int(task_embedding_dim)
        self.candidate_feature_dim = (
            self.task_embedding_dim + CLEAN_OFFLOADING_UAV_FEATURE_DIM + CLEAN_OFFLOADING_PAIR_FEATURE_DIM
        )
        self.scorer = SharedOffloadingCandidateScorer(self.candidate_feature_dim, hidden_dim=hidden_dim)
        self.latest_records: list[CleanOffloadingActionRecord] = []

    @torch.no_grad()
    def act(
        self,
        *,
        frozen_ready_tasks: list[Any],
        task_embeddings: torch.Tensor | np.ndarray,
        graph_snapshot: Any,
        task_manager: Any,
        uavs: list[Any],
        executor: Any,
        current_time_seconds: float,
        uav_service_positions: dict[int, Any] | None = None,
        ue_service_positions: dict[int, Any] | None = None,
        ues: list[Any] | None = None,
        deterministic: bool = False,
    ) -> CleanAssignmentBuffer:
        device = next(self.parameters()).device
        task_embeddings_tensor = torch.as_tensor(task_embeddings, dtype=torch.float32, device=device)
        if task_embeddings_tensor.dim() != 2 or task_embeddings_tensor.shape[1] != self.task_embedding_dim:
            raise ValueError("task_embeddings shape does not match task_embedding_dim.")

        reservation = TemporaryReservationState.from_executor(uavs, executor)
        assignments = CleanAssignmentBuffer()
        records: list[CleanOffloadingActionRecord] = []
        for decision_order, task in enumerate(frozen_ready_tasks):
            task_idx = graph_snapshot.task_id_to_idx.get(task.task_id)
            if task_idx is None:
                continue
            dynamic_features_np, pair_features_np, candidate_mask_np, candidate_uav_ids, estimates = build_offloading_candidate_components(
                task=task,
                uavs=uavs,
                task_manager=task_manager,
                executor=executor,
                state_view=reservation,
                current_time_seconds=float(current_time_seconds),
                uav_service_positions=uav_service_positions,
                ue_service_positions=ue_service_positions,
                ues=ues,
            )
            if dynamic_features_np.shape[0] == 0:
                continue
            task_embedding_np = task_embeddings_tensor[int(task_idx)].detach().cpu().numpy().reshape(1, -1)
            candidate_features_np = np.concatenate(
                [
                    np.repeat(task_embedding_np, dynamic_features_np.shape[0], axis=0),
                    dynamic_features_np,
                    pair_features_np,
                ],
                axis=1,
            ).astype(np.float32)
            if candidate_features_np.shape[0] == 0 or not bool(candidate_mask_np.any()):
                continue

            candidate_features = torch.as_tensor(candidate_features_np, dtype=torch.float32, device=device)
            dynamic_features = torch.as_tensor(dynamic_features_np, dtype=torch.float32, device=device)
            pair_features = torch.as_tensor(pair_features_np, dtype=torch.float32, device=device)
            candidate_mask = torch.as_tensor(candidate_mask_np, dtype=torch.bool, device=device)
            logits = self.scorer(candidate_features)
            masked_logits = logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)
            dist = Categorical(logits=masked_logits)
            selected = torch.argmax(masked_logits) if deterministic else dist.sample()
            selected_action = int(selected.item())
            selected_uav_id = int(candidate_uav_ids[selected_action])
            selected_estimate = estimates[selected_action]
            assignments.append(task.task_id, selected_uav_id, decision_order)
            reservation.reserve(
                task.task_id,
                selected_uav_id,
                estimated_available_time=selected_estimate.estimated_finish_time,
                estimated_queued_workload=selected_estimate.estimated_queued_workload,
            )
            records.append(
                CleanOffloadingActionRecord(
                    task_id=str(task.task_id),
                    dag_id=str(task.dag_id),
                    task_local_index=int(task_idx),
                    decision_order=int(decision_order),
                    candidate_uav_ids=list(candidate_uav_ids),
                    candidate_features=candidate_features.detach().cpu().clone(),
                    dynamic_uav_features=dynamic_features.detach().cpu().clone(),
                    pair_features=pair_features.detach().cpu().clone(),
                    candidate_mask=candidate_mask.detach().cpu().clone(),
                    selected_action=selected_action,
                    selected_uav_id=selected_uav_id,
                    old_log_prob=float(dist.log_prob(selected).item()),
                    entropy=float(dist.entropy().item()),
                    selected_estimated_finish_time=float(selected_estimate.estimated_finish_time),
                    selected_estimated_incremental_delay=max(
                        float(selected_estimate.estimated_finish_time)
                        - float(current_time_seconds),
                        0.0,
                    ),
                )
            )

        self.latest_records = records
        return assignments
