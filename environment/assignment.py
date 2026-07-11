from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import config
import numpy as np
from environment import comm_model
from environment.dag_tasks import DAGTaskManager, TaskNode


CLEAN_OFFLOADING_UAV_FEATURE_DIM = 7
CLEAN_OFFLOADING_PAIR_FEATURE_DIM = 8


@dataclass(slots=True)
class CleanAssignmentEntry:
    task_id: str
    uav_id: int
    decision_order: int


@dataclass(slots=True)
class CleanAssignmentBuffer:
    entries: list[CleanAssignmentEntry] = field(default_factory=list)

    def append(self, task_id: str, uav_id: int, decision_order: int) -> None:
        self.entries.append(
            CleanAssignmentEntry(task_id=str(task_id), uav_id=int(uav_id), decision_order=int(decision_order))
        )

    def to_assignment_dict(self) -> dict[str, int]:
        return {entry.task_id: entry.uav_id for entry in self.entries}

    @property
    def entry_count(self) -> int:
        return len(self.entries)


@dataclass(slots=True)
class TemporaryReservationState:
    queue_lengths: dict[int, int]
    available_times: dict[int, float] = field(default_factory=dict)
    queued_workloads: dict[int, float] = field(default_factory=dict)
    slot_assigned_counts: dict[int, int] = field(default_factory=dict)
    reserved_task_ids: set[str] = field(default_factory=set)

    @classmethod
    def from_executor(cls, uavs: list[Any], executor: Any) -> "TemporaryReservationState":
        queues = getattr(executor, "uav_queues", {})
        records = getattr(executor, "task_records", {})
        queued_workloads: dict[int, float] = {}
        for uav in uavs:
            uav_id = int(uav.id)
            workload = 0.0
            for task_id in queues.get(uav_id, []):
                record = records.get(str(task_id))
                if record is not None:
                    workload += max(float(getattr(record, "compute_time", 0.0)), 0.0) * float(
                        config.UAV_COMPUTE_RATE_OPS_PER_SEC
                    )
            queued_workloads[uav_id] = workload
        return cls(
            queue_lengths={int(uav.id): len(queues.get(int(uav.id), [])) for uav in uavs},
            available_times={
                int(uav.id): float(getattr(executor, "uav_available_time", {}).get(int(uav.id), 0.0))
                for uav in uavs
            },
            queued_workloads=queued_workloads,
            slot_assigned_counts={int(uav.id): 0 for uav in uavs},
        )

    def remaining_slots(self, uav_id: int) -> int:
        used = int(self.queue_lengths.get(int(uav_id), 0))
        return max(int(config.CLEAN_MAX_QUEUE_PER_UAV) - used, 0)

    def reserve(
        self,
        task_id: str,
        uav_id: int,
        estimated_available_time: float | None = None,
        estimated_queued_workload: float = 0.0,
    ) -> None:
        self.reserved_task_ids.add(str(task_id))
        key = int(uav_id)
        self.queue_lengths[key] = self.queue_lengths.get(key, 0) + 1
        self.slot_assigned_counts[key] = self.slot_assigned_counts.get(key, 0) + 1
        self.queued_workloads[key] = self.queued_workloads.get(key, 0.0) + max(float(estimated_queued_workload), 0.0)
        if estimated_available_time is not None:
            self.available_times[key] = max(
                float(self.available_times.get(key, 0.0)),
                float(estimated_available_time),
            )


@dataclass(slots=True)
class OffloadingCandidateEstimate:
    task_id: str
    uav_id: int
    legal: bool
    dynamic_uav_features: np.ndarray
    pair_features: np.ndarray
    estimated_finish_time: float = 0.0
    estimated_queued_workload: float = 0.0


def freeze_ready_tasks(task_manager: DAGTaskManager) -> list[TaskNode]:
    return sorted(task_manager.get_ready_tasks(), key=lambda task: _ready_sort_key(task, task_manager))


def is_assignment_legal(
    task: TaskNode | None,
    uav_id: int,
    state_view: TemporaryReservationState,
    valid_uav_ids: set[int],
    executor: Any,
    service_positions: dict[int, Any] | None = None,
) -> bool:
    """Minimal shared clean assignment legality.

    T7 should extend this helper with communication reachability, precise capacity,
    and pair-feature/executor estimator consistency.
    """
    del service_positions
    if task is None or not task.is_ready:
        return False
    if int(uav_id) not in valid_uav_ids:
        return False
    if str(task.task_id) in state_view.reserved_task_ids:
        return False
    if hasattr(executor, "is_task_scheduled") and executor.is_task_scheduled(task.task_id):
        return False
    return state_view.remaining_slots(int(uav_id)) > 0


def legal_candidate_uav_ids(
    task: TaskNode,
    uav_ids: list[int],
    state_view: TemporaryReservationState,
    executor: Any,
    service_positions: dict[int, Any] | None = None,
) -> list[int]:
    valid_uav_ids = set(int(uav_id) for uav_id in uav_ids)
    return [
        int(uav_id)
        for uav_id in uav_ids
        if is_assignment_legal(task, int(uav_id), state_view, valid_uav_ids, executor, service_positions)
    ]


def build_offloading_candidate_batch(
    *,
    task: TaskNode,
    task_embedding: np.ndarray,
    uavs: list[Any],
    task_manager: DAGTaskManager,
    executor: Any,
    state_view: TemporaryReservationState,
    current_time_seconds: float,
    uav_service_positions: dict[int, Any] | None = None,
    ue_service_positions: dict[int, Any] | None = None,
    ues: list[Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], list[OffloadingCandidateEstimate]]:
    dynamic_uav_features, pair_features, candidate_mask, candidate_uav_ids, estimates = build_offloading_candidate_components(
        task=task,
        uavs=uavs,
        task_manager=task_manager,
        executor=executor,
        state_view=state_view,
        current_time_seconds=current_time_seconds,
        uav_service_positions=uav_service_positions,
        ue_service_positions=ue_service_positions,
        ues=ues,
    )
    task_embedding = np.asarray(task_embedding, dtype=np.float32).reshape(1, -1)
    if dynamic_uav_features.shape[0] == 0:
        feature_dim = int(task_embedding.shape[1]) + CLEAN_OFFLOADING_UAV_FEATURE_DIM + CLEAN_OFFLOADING_PAIR_FEATURE_DIM
        return np.zeros((0, feature_dim), dtype=np.float32), candidate_mask, candidate_uav_ids, estimates
    repeated_task_embeddings = np.repeat(task_embedding, dynamic_uav_features.shape[0], axis=0)
    return (
        np.concatenate([repeated_task_embeddings, dynamic_uav_features, pair_features], axis=1).astype(np.float32),
        candidate_mask,
        candidate_uav_ids,
        estimates,
    )


def build_offloading_candidate_components(
    *,
    task: TaskNode,
    uavs: list[Any],
    task_manager: DAGTaskManager,
    executor: Any,
    state_view: TemporaryReservationState,
    current_time_seconds: float,
    uav_service_positions: dict[int, Any] | None = None,
    ue_service_positions: dict[int, Any] | None = None,
    ues: list[Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[OffloadingCandidateEstimate]]:
    """Build candidate dynamic non-graph inputs without task embeddings."""
    ordered_uavs = sorted(uavs, key=lambda item: int(item.id))
    uav_map = {int(uav.id): uav for uav in ordered_uavs}
    valid_uav_ids = set(uav_map)
    dynamic_rows: list[np.ndarray] = []
    pair_rows: list[np.ndarray] = []
    mask_values: list[bool] = []
    candidate_uav_ids: list[int] = []
    estimates: list[OffloadingCandidateEstimate] = []
    for uav in ordered_uavs:
        uav_id = int(uav.id)
        legal = is_assignment_legal(
            task=task,
            uav_id=uav_id,
            state_view=state_view,
            valid_uav_ids=valid_uav_ids,
            executor=executor,
            service_positions=uav_service_positions,
        )
        estimate = estimate_offloading_candidate(
            task=task,
            uav_id=uav_id,
            uav_map=uav_map,
            task_manager=task_manager,
            executor=executor,
            state_view=state_view,
            current_time_seconds=current_time_seconds,
            uav_service_positions=uav_service_positions,
            ue_service_positions=ue_service_positions,
            ues=ues,
            legal=legal,
        )
        dynamic_rows.append(np.asarray(estimate.dynamic_uav_features, dtype=np.float32).reshape(-1))
        pair_rows.append(np.asarray(estimate.pair_features, dtype=np.float32).reshape(-1))
        mask_values.append(bool(estimate.legal))
        candidate_uav_ids.append(uav_id)
        estimates.append(estimate)
    if not dynamic_rows:
        return (
            np.zeros((0, CLEAN_OFFLOADING_UAV_FEATURE_DIM), dtype=np.float32),
            np.zeros((0, CLEAN_OFFLOADING_PAIR_FEATURE_DIM), dtype=np.float32),
            np.zeros((0,), dtype=bool),
            [],
            [],
        )
    return (
        np.stack(dynamic_rows, axis=0).astype(np.float32),
        np.stack(pair_rows, axis=0).astype(np.float32),
        np.asarray(mask_values, dtype=bool),
        candidate_uav_ids,
        estimates,
    )


def estimate_offloading_candidate(
    *,
    task: TaskNode,
    uav_id: int,
    uav_map: dict[int, Any],
    task_manager: DAGTaskManager,
    executor: Any,
    state_view: TemporaryReservationState,
    current_time_seconds: float,
    uav_service_positions: dict[int, Any] | None = None,
    ue_service_positions: dict[int, Any] | None = None,
    ues: list[Any] | None = None,
    legal: bool = True,
) -> OffloadingCandidateEstimate:
    uav = uav_map.get(int(uav_id))
    dynamic_features = _dynamic_uav_features(
        uav=uav,
        uav_id=int(uav_id),
        state_view=state_view,
        current_time_seconds=current_time_seconds,
        uav_service_positions=uav_service_positions,
    )
    zero_pair = np.zeros((CLEAN_OFFLOADING_PAIR_FEATURE_DIM,), dtype=np.float32)
    if not legal or uav is None:
        return OffloadingCandidateEstimate(str(task.task_id), int(uav_id), False, dynamic_features, zero_pair)

    job = task_manager.get_job(task.dag_id)
    if job is None:
        return OffloadingCandidateEstimate(str(task.task_id), int(uav_id), False, dynamic_features, zero_pair)

    transfer_time = 0.0
    communication_energy = 0.0
    predecessor_ready_time = float(current_time_seconds)
    target_pos = _service_position(uav_service_positions, int(uav_id), getattr(uav, "pos"))

    if not task.predecessors:
        source_pos = _ue_service_position(job, ues, ue_service_positions)
        distance = comm_model.clean_distance_2d(source_pos, target_pos)
        transfer_time = _clean_tx_seconds(task.input_data_size_mb, job.base_upload_bandwidth_mbps, distance)
        communication_energy = transfer_time * float(config.P_UE_TX)
    else:
        parent_finish_times: list[float] = []
        for parent_id in task.predecessors:
            parent = task_manager.get_task(parent_id)
            if parent is None or parent.finish_time is None or parent.assigned_uav is None or parent.is_ready:
                return OffloadingCandidateEstimate(str(task.task_id), int(uav_id), False, dynamic_features, zero_pair)
            parent_finish_times.append(float(parent.finish_time))
            if int(parent.assigned_uav) == int(uav_id):
                continue
            parent_uav = uav_map.get(int(parent.assigned_uav))
            if parent_uav is None:
                return OffloadingCandidateEstimate(str(task.task_id), int(uav_id), False, dynamic_features, zero_pair)
            parent_pos = _service_position(
                uav_service_positions,
                int(parent.assigned_uav),
                getattr(parent_uav, "pos"),
            )
            distance = comm_model.clean_distance_2d(parent_pos, target_pos)
            parent_transfer_time = _clean_tx_seconds(
                parent.output_data_size_mb,
                job.base_upload_bandwidth_mbps,
                distance,
            )
            transfer_time += parent_transfer_time
            communication_energy += parent_transfer_time * float(config.P_UAV_TX)
        predecessor_ready_time = max(parent_finish_times) if parent_finish_times else float(current_time_seconds)

    available_time = float(state_view.available_times.get(int(uav_id), getattr(executor, "uav_available_time", {}).get(int(uav_id), 0.0)))
    queue_waiting_time = max(available_time, predecessor_ready_time, float(current_time_seconds)) - float(current_time_seconds)
    transfer_ready_time = max(float(current_time_seconds), available_time, predecessor_ready_time) + transfer_time
    compute_time = float(task.num_operation) / float(config.UAV_COMPUTE_RATE_OPS_PER_SEC)
    compute_energy = compute_time * float(config.P_UAV_COMPUTE)
    compute_finish_time = transfer_ready_time + compute_time

    return_time = 0.0
    return_energy = 0.0
    if task.task_id in set(job.sink_task_ids):
        ue_pos = _ue_service_position(job, ues, ue_service_positions)
        return_distance = comm_model.clean_distance_2d(target_pos, ue_pos)
        return_time = _clean_tx_seconds(task.output_data_size_mb, job.base_download_bandwidth_mbps, return_distance)
        return_energy = return_time * float(config.P_UAV_TX)

    estimated_finish_time = compute_finish_time + return_time
    incremental_delay = max(estimated_finish_time - float(current_time_seconds), 0.0)
    pair_features = _normalize_pair_features(
        [
            transfer_time,
            communication_energy,
            queue_waiting_time,
            compute_time,
            compute_energy,
            incremental_delay,
            return_time,
            return_energy,
        ]
    )
    return OffloadingCandidateEstimate(
        task_id=str(task.task_id),
        uav_id=int(uav_id),
        legal=True,
        dynamic_uav_features=dynamic_features,
        pair_features=pair_features,
        estimated_finish_time=estimated_finish_time,
        estimated_queued_workload=float(task.num_operation),
    )


def _ready_sort_key(task: TaskNode, task_manager: DAGTaskManager) -> tuple[float, str, int, str]:
    job = task_manager.get_job(task.dag_id)
    dag_arrival_time = float(job.arrival_time if job is not None else task.arrival_time)
    # Spec ready-sort key: (dag_arrival_time, dag_id, topological_index, task_id).
    # topological_index is now a real static field on TaskNode; fall back to the task_id
    # numeric suffix only for legacy nodes that predate the field. Use an explicit None
    # check so a valid topological_index == 0 (entry task) is not discarded.
    topological_index_value = getattr(task, "topological_index", None)
    topological_index = (
        int(topological_index_value)
        if topological_index_value is not None
        else _task_numeric_suffix(task.task_id)
    )
    return dag_arrival_time, str(task.dag_id), topological_index, str(task.task_id)


def _dynamic_uav_features(
    *,
    uav: Any | None,
    uav_id: int,
    state_view: TemporaryReservationState,
    current_time_seconds: float,
    uav_service_positions: dict[int, Any] | None,
) -> np.ndarray:
    if uav is None:
        service_pos = np.zeros((2,), dtype=np.float32)
    else:
        service_pos = np.asarray(_service_position(uav_service_positions, uav_id, getattr(uav, "pos")), dtype=np.float32)
    max_queue = max(float(config.CLEAN_MAX_QUEUE_PER_UAV), 1.0)
    max_available_time = max(float(config.EPISODE_LENGTH) * float(config.TIME_SLOT_DURATION), 1.0)
    max_workload = max(
        float(config.UAV_COMPUTE_RATE_OPS_PER_SEC) * float(config.TIME_SLOT_DURATION) * max_queue,
        1.0,
    )
    queue_length = float(state_view.queue_lengths.get(uav_id, 0))
    remaining_slots = float(state_view.remaining_slots(uav_id))
    available_delta = max(float(state_view.available_times.get(uav_id, current_time_seconds)) - float(current_time_seconds), 0.0)
    queued_workload = float(state_view.queued_workloads.get(uav_id, 0.0))
    slot_assigned = float(state_view.slot_assigned_counts.get(uav_id, 0))
    return np.asarray(
        [
            np.clip(float(service_pos.reshape(-1)[0]) / float(config.AREA_WIDTH), 0.0, 1.0),
            np.clip(float(service_pos.reshape(-1)[1]) / float(config.AREA_HEIGHT), 0.0, 1.0),
            np.clip(queue_length / max_queue, 0.0, 1.0),
            np.clip(remaining_slots / max_queue, 0.0, 1.0),
            np.clip(available_delta / max_available_time, 0.0, 1.0),
            np.clip(queued_workload / max_workload, 0.0, 1.0),
            np.clip(slot_assigned / max_queue, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def _normalize_pair_features(values: list[float]) -> np.ndarray:
    max_time = max(float(config.TIME_SLOT_DURATION) * float(config.EPISODE_LENGTH), 1.0)
    max_energy = max(float(config.P_UAV_COMPUTE) * float(config.TIME_SLOT_DURATION), 1.0)
    scales = np.asarray(
        [max_time, max_energy, max_time, max_time, max_energy, max_time, max_time, max_energy],
        dtype=np.float32,
    )
    raw = np.asarray(values, dtype=np.float32)
    return np.clip(raw / scales, 0.0, 1.0).astype(np.float32)


def _clean_tx_seconds(data_size_mb: float, base_bandwidth_mbps: float, distance_m: float) -> float:
    return float(comm_model.clean_transmission_time_seconds(data_size_mb, base_bandwidth_mbps, distance_m))


def _service_position(position_map: dict[int, Any] | None, entity_id: int, fallback: Any) -> Any:
    if position_map is None:
        return fallback
    return position_map.get(int(entity_id), fallback)


def _ue_service_position(job: Any, ues: list[Any] | None, ue_service_positions: dict[int, Any] | None) -> Any:
    if ue_service_positions is not None and int(job.ue_id) in ue_service_positions:
        return ue_service_positions[int(job.ue_id)]
    if ues is not None:
        for ue in ues:
            if int(ue.id) == int(job.ue_id):
                return getattr(ue, "pos")
    return job.source_pos


def _task_numeric_suffix(task_id: str) -> int:
    suffix = str(task_id).rsplit("_", 1)[-1]
    if suffix.isdigit():
        return int(suffix)
    return sum(ord(char) for char in str(task_id))
