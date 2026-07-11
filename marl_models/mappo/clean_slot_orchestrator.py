from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from environment.graph_builder import CleanGraphBuilder, CleanGraphSnapshot
from marl_models.mappo.clean_movement_actor import CleanMovementObservation, build_clean_movement_observation
from marl_models.mappo.clean_ppo import (
    assemble_clean_critic_global_input,
    build_clean_critic_non_graph_input,
)

try:
    import torch
except ModuleNotFoundError:
    torch = None


@dataclass(slots=True)
class CleanPreparedSlotState:
    """Parameter-independent clean decision state prepared from x_t^- to s_t."""

    slot_index: int
    time_step: int
    current_time_seconds: float
    previous_internal_state: str
    decision_state: str
    created_dags: int
    new_dag_arrived: bool
    dag_arrival_version: int
    frozen_ready_task_ids: list[str]
    ue_service_positions: dict[int, np.ndarray]
    uav_pre_move_positions: dict[int, np.ndarray]
    graph_snapshot: CleanGraphSnapshot
    critic_non_graph_input: np.ndarray


@dataclass(slots=True)
class CleanEncodedSlotState:
    """Parameter-dependent encoding for a prepared clean slot."""

    prepared_state: CleanPreparedSlotState
    task_embeddings: Any
    critic_global_input: Any
    value: float | None
    movement_observation: CleanMovementObservation
    movement_logits: Any | None = None


@dataclass(slots=True)
class CleanMovementRolloutRecord:
    uav_id: int
    uav_index: int
    uav_features: np.ndarray
    ready_task_indices: list[int]
    pending_task_indices: list[int]
    ready_count_normalized: float
    pending_count_normalized: float
    movement_mask: np.ndarray
    selected_action: int
    old_log_probability: float
    entropy: float


@dataclass(slots=True)
class CleanOffloadingRolloutRecord:
    task_id: str
    task_local_index: int
    decision_order: int
    candidate_uav_ids: list[int]
    dynamic_uav_features: np.ndarray
    pair_features: np.ndarray
    candidate_mask: np.ndarray
    selected_action: int
    selected_uav_id: int
    old_log_probability: float
    entropy: float


@dataclass(slots=True)
class CleanSlotRolloutRecord:
    slot_index: int
    graph_snapshot: CleanGraphSnapshot
    critic_non_graph_input: np.ndarray
    value: float
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    next_value: float | None = None
    bootstrap_value: float | None = None
    next_prepared_state: CleanPreparedSlotState | None = None
    movement_records: list[CleanMovementRolloutRecord] = field(default_factory=list)
    offloading_records: list[CleanOffloadingRolloutRecord] = field(default_factory=list)


class CleanSlotRolloutBuffer:
    """Slot-centric rollout container; PPO update is intentionally T12."""

    def __init__(self) -> None:
        self.records: list[CleanSlotRolloutRecord] = []
        self.closed: bool = False

    def append(self, record: CleanSlotRolloutRecord) -> None:
        if self.closed:
            raise RuntimeError("Cannot append to a closed clean rollout buffer.")
        self.records.append(record)

    def close(
        self,
        *,
        bootstrap_value: float | None = None,
        next_prepared_state: CleanPreparedSlotState | None = None,
    ) -> None:
        self.closed = True
        if self.records and bootstrap_value is not None:
            self.records[-1].bootstrap_value = float(bootstrap_value)
            self.records[-1].next_value = float(bootstrap_value)
        if self.records and next_prepared_state is not None:
            self.records[-1].next_prepared_state = next_prepared_state

    @property
    def checkpoint_safe(self) -> bool:
        return self.closed

    def clear(self) -> None:
        self.records.clear()
        self.closed = False

    def __len__(self) -> int:
        return len(self.records)


def prepare_slot_state(
    *,
    env: Any,
    graph_builder: CleanGraphBuilder,
) -> CleanPreparedSlotState:
    """Prepare s_t once, freeze R_t, build task-only graph, and copy non-graph critic input."""
    context = env.prepare_slot_state()
    graph_snapshot = graph_builder.build(
        task_manager=env.task_manager,
        uavs=env.uavs,
        current_time_step=env.time_step,
        executor=env.executor,
        frozen_ready_task_ids=context["frozen_ready_task_ids"],
        new_dag_arrived=bool(context["new_dag_arrived"]),
        dag_arrival_version=int(context["dag_arrival_version"]),
    )
    critic_non_graph_input = build_clean_critic_non_graph_input(
        graph_snapshot=graph_snapshot,
        uavs=env.uavs,
        executor=env.executor,
        pre_move_positions=context["uav_pre_move_positions"],
        current_time_seconds=env.current_time_seconds,
    )
    return CleanPreparedSlotState(
        slot_index=int(context["slot_index"]),
        time_step=int(context["time_step"]),
        current_time_seconds=float(env.current_time_seconds),
        previous_internal_state=str(context["previous_internal_state"]),
        decision_state=str(context["decision_state"]),
        created_dags=int(context["created_dags"]),
        new_dag_arrived=bool(context["new_dag_arrived"]),
        dag_arrival_version=int(context["dag_arrival_version"]),
        frozen_ready_task_ids=list(context["frozen_ready_task_ids"]),
        ue_service_positions=_copy_position_map(context["ue_service_positions"]),
        uav_pre_move_positions=_copy_position_map(context["uav_pre_move_positions"]),
        graph_snapshot=copy_clean_graph_snapshot(graph_snapshot),
        critic_non_graph_input=np.asarray(critic_non_graph_input, dtype=np.float32).copy(),
    )


def encode_prepared_slot(
    *,
    prepared_state: CleanPreparedSlotState,
    env: Any,
    hgnn: Any | None = None,
    critic: Any | None = None,
    movement_actor: Any | None = None,
    device: str | Any = "cpu",
    fallback_embedding_dim: int | None = None,
) -> CleanEncodedSlotState:
    """Encode an already prepared slot with the current model parameters."""
    snapshot = prepared_state.graph_snapshot
    task_embeddings = _encode_task_embeddings(snapshot, hgnn=hgnn, device=device, fallback_dim=fallback_embedding_dim)
    critic_global_input = _assemble_critic_input(task_embeddings, prepared_state.critic_non_graph_input, device=device)
    value = _critic_value(critic, critic_global_input, device=device)
    movement_observation = build_clean_movement_observation(
        uavs=env.uavs,
        executor=env.executor,
        graph_snapshot=snapshot,
        pre_move_positions=prepared_state.uav_pre_move_positions,
        current_time_seconds=prepared_state.current_time_seconds,
    )
    movement_logits = None
    if movement_actor is not None:
        if torch is None:
            raise ModuleNotFoundError("torch is required to run CleanMovementActor forward")
        movement_logits = movement_actor(
            uav_features=torch.as_tensor(movement_observation.uav_features, dtype=torch.float32, device=device),
            task_embeddings=task_embeddings.to(device) if hasattr(task_embeddings, "to") else torch.as_tensor(task_embeddings, dtype=torch.float32, device=device),
            ready_task_indices=movement_observation.ready_task_indices,
            pending_task_indices=movement_observation.pending_task_indices,
            ready_count_normalized=movement_observation.ready_count_normalized,
            pending_count_normalized=movement_observation.pending_count_normalized,
            boundary_action_mask=torch.as_tensor(movement_observation.boundary_action_mask, dtype=torch.bool, device=device),
        )
    return CleanEncodedSlotState(
        prepared_state=prepared_state,
        task_embeddings=task_embeddings,
        critic_global_input=critic_global_input,
        value=value,
        movement_observation=movement_observation,
        movement_logits=movement_logits,
    )


def copy_clean_graph_snapshot(snapshot: CleanGraphSnapshot) -> CleanGraphSnapshot:
    """Create an independent historical GraphSnapshot input copy for rollout storage."""
    task_features = np.asarray(snapshot.task_features, dtype=np.float32).copy()
    incidence_matrix = np.asarray(snapshot.incidence_matrix, dtype=np.float32).copy()
    task_features.setflags(write=False)
    incidence_matrix.setflags(write=False)
    return CleanGraphSnapshot(
        current_time_step=int(snapshot.current_time_step),
        active_task_ids=list(snapshot.active_task_ids),
        ready_task_ids=list(snapshot.ready_task_ids),
        pending_task_ids=list(snapshot.pending_task_ids),
        task_id_to_idx=dict(snapshot.task_id_to_idx),
        idx_to_task_id=dict(snapshot.idx_to_task_id),
        task_features=task_features,
        dag_hyperedges=[list(edge) for edge in snapshot.dag_hyperedges],
        khop_hyperedges=[list(edge) for edge in snapshot.khop_hyperedges],
        attribute_hyperedges=[list(edge) for edge in snapshot.attribute_hyperedges],
        partition_hyperedges=[list(edge) for edge in snapshot.partition_hyperedges],
        incidence_matrix=incidence_matrix,
        partition_status=str(getattr(snapshot, "partition_status", "disabled")),
    )


def make_slot_rollout_record(
    *,
    encoded_state: CleanEncodedSlotState,
    value: float | None = None,
) -> CleanSlotRolloutRecord:
    prepared = encoded_state.prepared_state
    encoded_value = encoded_state.value if value is None else value
    return CleanSlotRolloutRecord(
        slot_index=prepared.slot_index,
        graph_snapshot=copy_clean_graph_snapshot(prepared.graph_snapshot),
        critic_non_graph_input=np.asarray(prepared.critic_non_graph_input, dtype=np.float32).copy(),
        value=0.0 if encoded_value is None else float(encoded_value),
    )


def assert_graph_snapshot_task_only(snapshot: Any) -> None:
    forbidden_fields = {
        "uav_features",
        "pair_features",
        "candidate_mask",
        "reward",
        "metrics",
        "profiling",
        "critic_global_input",
    }
    present = sorted(field for field in forbidden_fields if hasattr(snapshot, field))
    if present:
        raise AssertionError(f"GraphSnapshot contains non-task fields: {present}")


def _encode_task_embeddings(
    snapshot: CleanGraphSnapshot,
    *,
    hgnn: Any | None,
    device: str | Any,
    fallback_dim: int | None,
) -> Any:
    if hgnn is None:
        task_features = np.asarray(snapshot.task_features, dtype=np.float32)
        if fallback_dim is None or int(fallback_dim) == task_features.shape[1]:
            return task_features.copy()
        if task_features.shape[0] == 0:
            return np.zeros((0, int(fallback_dim)), dtype=np.float32)
        output = np.zeros((task_features.shape[0], int(fallback_dim)), dtype=np.float32)
        width = min(task_features.shape[1], int(fallback_dim))
        output[:, :width] = task_features[:, :width]
        return output
    if torch is None:
        raise ModuleNotFoundError("torch is required to run HGNN forward")
    task_features_np = np.asarray(snapshot.task_features, dtype=np.float32).copy()
    incidence_np = np.asarray(snapshot.incidence_matrix, dtype=np.float32).copy()
    task_features = torch.as_tensor(task_features_np, dtype=torch.float32, device=device)
    incidence = torch.as_tensor(incidence_np, dtype=torch.float32, device=device)
    return hgnn(task_features, incidence)


def _assemble_critic_input(task_embeddings: Any, critic_non_graph_input: np.ndarray, *, device: str | Any) -> Any:
    if torch is not None and isinstance(task_embeddings, torch.Tensor):
        if task_embeddings.shape[0] > 0:
            active_mean = task_embeddings.mean(dim=0)
        else:
            active_mean = task_embeddings.new_zeros((task_embeddings.shape[1],))
        non_graph = torch.as_tensor(critic_non_graph_input, dtype=task_embeddings.dtype, device=task_embeddings.device)
        return torch.cat([active_mean, non_graph], dim=0)
    return assemble_clean_critic_global_input(
        task_embeddings=np.asarray(task_embeddings, dtype=np.float32),
        critic_non_graph_input=critic_non_graph_input,
    )


def _critic_value(critic: Any | None, critic_global_input: Any, *, device: str | Any) -> float | None:
    if critic is None:
        return None
    if torch is None:
        raise ModuleNotFoundError("torch is required to run centralized critic forward")
    critic_input = (
        critic_global_input.to(device=device)
        if hasattr(critic_global_input, "to")
        else torch.as_tensor(critic_global_input, dtype=torch.float32, device=device)
    )
    value = critic(critic_input)
    return float(value.detach().reshape(-1)[0].cpu().item())


def _copy_position_map(positions: dict[int, Any]) -> dict[int, np.ndarray]:
    return {int(item_id): np.asarray(position, dtype=np.float32).copy() for item_id, position in positions.items()}
