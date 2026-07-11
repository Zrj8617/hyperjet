from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import config
from environment.dag_tasks import DAGTaskManager, TaskNode


@dataclass(slots=True)
class CleanGraphSnapshot:
    current_time_step: int
    active_task_ids: list[str]
    ready_task_ids: list[str]
    pending_task_ids: list[str]
    task_id_to_idx: dict[str, int]
    idx_to_task_id: dict[int, str]
    task_features: np.ndarray
    dag_hyperedges: list[list[int]]
    khop_hyperedges: list[list[int]]
    attribute_hyperedges: list[list[int]]
    partition_hyperedges: list[list[int]]
    incidence_matrix: np.ndarray
    # KaHyPar engineering-degrade state for this slot. One of:
    #   "disabled"          -> ENABLE_KAHYPAR_PARTITION_HYPEREDGES off or < 2 active tasks
    #   "success"           -> KaHyPar re-partitioned this slot; cache updated
    #   "cache_interval"    -> not attempted this slot (interval); reused global cache
    #   "no_base_hyperedges"-> active tasks exist but no valid base hyperedges; cache kept
    #   "degraded_cache"    -> attempt failed/unavailable; reused existing cache
    #   "degraded_no_cache" -> attempt failed/unavailable AND no cache; no partition edges
    partition_status: str = "disabled"

    @property
    def task_ids(self) -> list[str]:
        return list(self.active_task_ids)

    @property
    def hyperedges(self) -> list[list[int]]:
        return [
            *self.dag_hyperedges,
            *self.khop_hyperedges,
            *self.attribute_hyperedges,
            *self.partition_hyperedges,
        ]


class CleanGraphBuilder:
    """Clean mainline graph snapshot builder for zrj_3."""

    def __init__(self) -> None:
        self._cached_attribute_groups_global: list[list[str]] = []
        self._last_attribute_update_step: int | None = None
        self._last_attribute_task_ids: tuple[str, ...] = ()
        self._cached_partition_groups_global: list[list[str]] = []
        self._last_partition_update_step: int | None = None
        self._last_partition_attempt_step: int | None = None
        self._last_partition_status: str = "disabled"
        self._last_seen_dag_arrival_version: int = 0

    def reset(self) -> None:
        self._cached_attribute_groups_global = []
        self._last_attribute_update_step = None
        self._last_attribute_task_ids = ()
        self._cached_partition_groups_global = []
        self._last_partition_update_step = None
        self._last_partition_attempt_step = None
        self._last_partition_status = "disabled"
        self._last_seen_dag_arrival_version = 0

    @property
    def last_attribute_update_step(self) -> int | None:
        return self._last_attribute_update_step

    @property
    def last_partition_update_step(self) -> int | None:
        return self._last_partition_update_step

    @property
    def last_partition_attempt_step(self) -> int | None:
        return self._last_partition_attempt_step

    @property
    def last_partition_status(self) -> str:
        return self._last_partition_status

    def build(
        self,
        task_manager: DAGTaskManager,
        uavs: list[Any] | None,
        current_time_step: int,
        executor: Any | None = None,
        frozen_ready_task_ids: list[str] | None = None,
        new_dag_arrived: bool = False,
        dag_arrival_version: int | None = None,
        force_hypergraph_update: bool = False,
    ) -> CleanGraphSnapshot:
        del uavs, executor
        current_arrival_version = (
            int(task_manager.dag_arrival_version)
            if dag_arrival_version is None
            else int(dag_arrival_version)
        )
        version_changed = current_arrival_version > self._last_seen_dag_arrival_version
        force_update = bool(force_hypergraph_update or new_dag_arrived or version_changed)
        active_tasks = sorted(task_manager.get_active_tasks(), key=lambda task: _stable_task_key(task.task_id))
        active_task_ids = [task.task_id for task in active_tasks]
        task_id_to_idx = {task_id: idx for idx, task_id in enumerate(active_task_ids)}
        idx_to_task_id = {idx: task_id for task_id, idx in task_id_to_idx.items()}
        active_id_set = set(active_task_ids)
        if frozen_ready_task_ids is None:
            ready_task_ids = [
                task.task_id
                for task in sorted(task_manager.get_ready_tasks(), key=lambda task: _stable_task_key(task.task_id))
                if task.task_id in active_id_set
            ]
        else:
            ready_task_ids = [task_id for task_id in frozen_ready_task_ids if task_id in active_id_set]
        ready_id_set = set(ready_task_ids)
        pending_task_ids = [task_id for task_id in active_task_ids if task_id not in ready_id_set]

        task_features = self._build_task_features(active_tasks, task_manager, ready_id_set)
        dag_hyperedges = self._build_dag_hyperedges(active_tasks, task_id_to_idx)
        khop_hyperedges = self._build_khop_hyperedges(task_manager, task_id_to_idx)
        attribute_hyperedges, attribute_updated = self._build_attribute_hyperedges(
            active_tasks=active_tasks,
            task_manager=task_manager,
            task_id_to_idx=task_id_to_idx,
            current_time_step=int(current_time_step),
            force_update=force_update,
        )
        partition_hyperedges = self._build_partition_hyperedges(
            active_task_ids=active_task_ids,
            task_id_to_idx=task_id_to_idx,
            idx_to_task_id=idx_to_task_id,
            base_hyperedges=[*khop_hyperedges, *attribute_hyperedges],
            current_time_step=int(current_time_step),
            force_update=attribute_updated,
        )
        self._last_seen_dag_arrival_version = max(self._last_seen_dag_arrival_version, current_arrival_version)
        incidence_matrix = self._build_incidence_matrix(
            node_count=len(active_task_ids),
            hyperedges=[*dag_hyperedges, *khop_hyperedges, *attribute_hyperedges, *partition_hyperedges],
        )

        return CleanGraphSnapshot(
            current_time_step=int(current_time_step),
            active_task_ids=active_task_ids,
            ready_task_ids=ready_task_ids,
            pending_task_ids=pending_task_ids,
            task_id_to_idx=task_id_to_idx,
            idx_to_task_id=idx_to_task_id,
            task_features=task_features,
            dag_hyperedges=dag_hyperedges,
            khop_hyperedges=khop_hyperedges,
            attribute_hyperedges=attribute_hyperedges,
            partition_hyperedges=partition_hyperedges,
            incidence_matrix=incidence_matrix,
            partition_status=str(self._last_partition_status),
        )

    def _build_task_features(
        self,
        active_tasks: list[TaskNode],
        task_manager: DAGTaskManager,
        ready_task_ids: set[str],
    ) -> np.ndarray:
        if not active_tasks:
            return np.zeros((0, 12), dtype=np.float32)

        max_upload = max(float(max(config.BASE_UPLOAD_BANDWIDTH_MBPS)), 1.0)
        max_download = max(float(max(config.BASE_DOWNLOAD_BANDWIDTH_MBPS)), 1.0)
        max_input = max(float(config.INPUT_DATA_SIZE_MB_RANGE[1]), 1.0)
        max_output = max(float(config.OUTPUT_DATA_SIZE_MB_RANGE[1]), 1.0)
        max_operation = max(max(task.num_operation for task in active_tasks), 1.0)
        max_level = max(float(config.DAG_MAX_LEVELS - 1), 1.0)
        max_parent_count = max(float(config.DAG_MAX_PARENTS), 1.0)
        max_child_count = max(float(config.DAG_MAX_TASKS), 1.0)

        rows: list[np.ndarray] = []
        for task in active_tasks:
            job = task_manager.get_job(task.dag_id)
            base_upload = 0.0 if job is None else float(job.base_upload_bandwidth_mbps)
            base_download = 0.0 if job is None else float(job.base_download_bandwidth_mbps)
            is_ready = task.task_id in ready_task_ids
            rows.append(
                np.array(
                    [
                        np.clip(base_upload / max_upload, 0.0, 1.0),
                        np.clip(base_download / max_download, 0.0, 1.0),
                        np.clip(task.input_data_size_mb / max_input, 0.0, 1.0),
                        np.clip(task.output_data_size_mb / max_output, 0.0, 1.0),
                        np.clip(task.num_operation / max_operation, 0.0, 1.0),
                        np.clip(task.level / max_level, 0.0, 1.0),
                        1.0 if not task.predecessors else 0.0,
                        1.0 if not task.successors else 0.0,
                        np.clip(float(len(task.predecessors)) / max_parent_count, 0.0, 1.0),
                        np.clip(float(len(task.successors)) / max_child_count, 0.0, 1.0),
                        1.0 if is_ready else 0.0,
                        0.0 if is_ready else 1.0,
                    ],
                    dtype=np.float32,
                )
            )
        return np.stack(rows, axis=0).astype(np.float32)

    def _build_uav_features(self, uavs: list[Any], executor: Any | None) -> np.ndarray:
        if not uavs:
            return np.zeros((0, 5), dtype=np.float32)
        max_queue = max(float(getattr(config, "CLEAN_MAX_QUEUE_PER_UAV", 1)), 1.0)
        compute_rate = float(config.UAV_COMPUTE_RATE_OPS_PER_SEC)
        rows: list[np.ndarray] = []
        for uav in uavs:
            queue_length = 0
            if executor is not None and hasattr(executor, "uav_queues"):
                queue_length = len(executor.uav_queues.get(int(uav.id), []))
            remaining_energy_ratio = float(getattr(uav, "remaining_energy_ratio", 1.0))
            rows.append(
                np.array(
                    [
                        np.clip(float(uav.pos[0]) / float(config.AREA_WIDTH), 0.0, 1.0),
                        np.clip(float(uav.pos[1]) / float(config.AREA_HEIGHT), 0.0, 1.0),
                        np.clip(remaining_energy_ratio, 0.0, 1.0),
                        np.clip(float(queue_length) / max_queue, 0.0, 1.0),
                        1.0 if compute_rate > 0.0 else 0.0,
                    ],
                    dtype=np.float32,
                )
            )
        return np.stack(rows, axis=0).astype(np.float32)

    def _build_dag_hyperedges(
        self,
        active_tasks: list[TaskNode],
        task_id_to_idx: dict[str, int],
    ) -> list[list[int]]:
        if not config.ENABLE_DAG_DEPENDENCY_EDGES:
            return []
        hyperedges: list[list[int]] = []
        for task in active_tasks:
            parent_idx = task_id_to_idx[task.task_id]
            for child_id in sorted(task.successors, key=_stable_task_key):
                if child_id in task_id_to_idx:
                    hyperedges.append(sorted([parent_idx, task_id_to_idx[child_id]]))
        return hyperedges

    def _build_khop_hyperedges(
        self,
        task_manager: DAGTaskManager,
        task_id_to_idx: dict[str, int],
    ) -> list[list[int]]:
        if not config.ENABLE_KHOP_DEPENDENCY_HYPEREDGES:
            return []
        dedup: set[tuple[int, ...]] = set()
        output: list[list[int]] = []
        for job in task_manager.jobs.values():
            for group_global in job.khop_hyperedges_global:
                group = tuple(sorted(task_id_to_idx[task_id] for task_id in group_global if task_id in task_id_to_idx))
                if len(group) < 2 or group in dedup:
                    continue
                dedup.add(group)
                output.append(list(group))
        return output

    def _build_attribute_hyperedges(
        self,
        active_tasks: list[TaskNode],
        task_manager: DAGTaskManager,
        task_id_to_idx: dict[str, int],
        current_time_step: int,
        force_update: bool,
    ) -> tuple[list[list[int]], bool]:
        if not config.ENABLE_ATTRIBUTE_HYPEREDGES or len(active_tasks) < 2:
            self._cached_attribute_groups_global = []
            self._last_attribute_task_ids = tuple(task.task_id for task in active_tasks)
            return [], True

        task_ids = tuple(task.task_id for task in active_tasks)
        previous_task_ids = set(self._last_attribute_task_ids)
        has_new_active_task = any(task_id not in previous_task_ids for task_id in task_ids)
        interval = max(int(config.ATTRIBUTE_HYPEREDGE_UPDATE_INTERVAL), 1)
        should_update = (
            force_update
            or self._last_attribute_update_step is None
            or current_time_step % interval == 0
            or has_new_active_task
        )
        if should_update:
            vectors = self._build_attribute_vectors(active_tasks, task_manager)
            cluster_num = min(int(config.ATTRIBUTE_HYPEREDGE_CLUSTER_NUM), len(active_tasks))
            labels = _deterministic_kmeans(_normalize_columns(vectors), cluster_num)
            groups: list[list[str]] = []
            for cluster_id in range(cluster_num):
                group = [task.task_id for task, label in zip(active_tasks, labels) if int(label) == cluster_id]
                if len(group) >= 2:
                    groups.append(group)
            self._cached_attribute_groups_global = groups
            self._last_attribute_update_step = current_time_step
            self._last_attribute_task_ids = task_ids

        hyperedges: list[list[int]] = []
        for group in self._cached_attribute_groups_global:
            indices = sorted(task_id_to_idx[task_id] for task_id in group if task_id in task_id_to_idx)
            if len(indices) >= 2:
                hyperedges.append(indices)
        return hyperedges, should_update

    def _build_partition_hyperedges(
        self,
        active_task_ids: list[str],
        task_id_to_idx: dict[str, int],
        idx_to_task_id: dict[int, str],
        base_hyperedges: list[list[int]],
        current_time_step: int,
        force_update: bool,
    ) -> list[list[int]]:
        if not config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES or len(active_task_ids) < 2:
            self._last_partition_status = "disabled"
            return []

        valid_base_hyperedges = [
            sorted({int(idx) for idx in edge if 0 <= int(idx) < len(active_task_ids)})
            for edge in base_hyperedges
        ]
        valid_base_hyperedges = [edge for edge in valid_base_hyperedges if len(edge) >= 2]
        if not valid_base_hyperedges:
            # No information to partition on. Keep any previous cache untouched; do
            # not report KaHyPar success or clear cached groups with an empty result.
            self._last_partition_status = "no_base_hyperedges"
            return self._remap_global_groups(self._cached_partition_groups_global, task_id_to_idx)

        interval = max(int(config.KAHYPAR_PARTITION_UPDATE_INTERVAL), 1)
        should_update = (
            force_update
            or self._last_partition_update_step is None
            or current_time_step % interval == 0
        )
        if should_update:
            self._last_partition_attempt_step = current_time_step
            partition_groups = self._run_kahypar_partition(
                node_count=len(active_task_ids),
                base_hyperedges=valid_base_hyperedges,
            )
            if partition_groups is not None:
                self._cached_partition_groups_global = [
                    [idx_to_task_id[idx] for idx in group if idx in idx_to_task_id]
                    for group in partition_groups
                    if len(group) >= 2
                ]
                self._last_partition_update_step = current_time_step
                self._last_partition_status = "success"
            else:
                # Engineering degrade: KaHyPar unavailable or failed. Reuse cache if any,
                # otherwise emit no partition edges this slot (never silently "success").
                self._last_partition_status = (
                    "degraded_cache"
                    if self._cached_partition_groups_global
                    else "degraded_no_cache"
                )
        else:
            self._last_partition_status = "cache_interval"

        return self._remap_global_groups(self._cached_partition_groups_global, task_id_to_idx)

    def _run_kahypar_partition(
        self,
        node_count: int,
        base_hyperedges: list[list[int]],
    ) -> list[list[int]] | None:
        if node_count < 2 or not base_hyperedges:
            return None
        try:
            import kahypar  # type: ignore
        except Exception:
            return None

        try:
            cleaned_edges = [
                sorted({int(idx) for idx in edge if 0 <= int(idx) < node_count})
                for edge in base_hyperedges
            ]
            cleaned_edges = [edge for edge in cleaned_edges if len(edge) >= 2]
            if not cleaned_edges:
                return None

            hyperedge_indices = [0]
            pins: list[int] = []
            for edge in cleaned_edges:
                pins.extend(edge)
                hyperedge_indices.append(len(pins))

            partition_count = min(max(2, int(config.ATTRIBUTE_HYPEREDGE_CLUSTER_NUM)), node_count)
            context = kahypar.Context()
            if hasattr(context, "setK"):
                context.setK(partition_count)
            if hasattr(context, "setEpsilon"):
                context.setEpsilon(0.03)
            if hasattr(context, "suppressOutput"):
                context.suppressOutput(True)

            try:
                hypergraph = kahypar.Hypergraph(
                    node_count,
                    len(cleaned_edges),
                    hyperedge_indices,
                    pins,
                    partition_count,
                    [1] * len(cleaned_edges),
                    [1] * node_count,
                )
            except TypeError:
                hypergraph = kahypar.Hypergraph(
                    node_count,
                    len(cleaned_edges),
                    hyperedge_indices,
                    pins,
                    partition_count,
                )

            kahypar.partition(hypergraph, context)
            groups_by_block: dict[int, list[int]] = {}
            for node_idx in range(node_count):
                if hasattr(hypergraph, "blockID"):
                    block_id = int(hypergraph.blockID(node_idx))
                elif hasattr(hypergraph, "block_id"):
                    block_id = int(hypergraph.block_id(node_idx))
                else:
                    return None
                groups_by_block.setdefault(block_id, []).append(node_idx)
            return [group for group in groups_by_block.values() if len(group) >= 2]
        except Exception:
            return None

    def _remap_global_groups(
        self,
        groups_global: list[list[str]],
        task_id_to_idx: dict[str, int],
    ) -> list[list[int]]:
        dedup: set[tuple[int, ...]] = set()
        output: list[list[int]] = []
        for group_global in groups_global:
            group = tuple(sorted(task_id_to_idx[task_id] for task_id in group_global if task_id in task_id_to_idx))
            if len(group) < 2 or group in dedup:
                continue
            dedup.add(group)
            output.append(list(group))
        return output

    def _build_incidence_matrix(self, node_count: int, hyperedges: list[list[int]]) -> np.ndarray:
        if node_count <= 0 or not hyperedges:
            return np.zeros((max(node_count, 0), 0), dtype=np.float32)
        incidence = np.zeros((node_count, len(hyperedges)), dtype=np.float32)
        for edge_idx, members in enumerate(hyperedges):
            for node_idx in members:
                if 0 <= int(node_idx) < node_count:
                    incidence[int(node_idx), edge_idx] = 1.0
        return incidence

    def _build_attribute_vectors(
        self,
        active_tasks: list[TaskNode],
        task_manager: DAGTaskManager,
    ) -> np.ndarray:
        rows: list[list[float]] = []
        for task in active_tasks:
            job = task_manager.get_job(task.dag_id)
            rows.append(
                [
                    0.0 if job is None else float(job.base_upload_bandwidth_mbps),
                    0.0 if job is None else float(job.base_download_bandwidth_mbps),
                    float(task.input_data_size_mb),
                    float(task.output_data_size_mb),
                    float(task.num_operation),
                ]
            )
        return np.asarray(rows, dtype=np.float64)

    def _build_candidate_pairs(
        self,
        active_tasks: list[TaskNode],
        uav_ids: list[int],
        task_id_to_idx: dict[str, int],
        uav_id_to_idx: dict[int, int],
        executor: Any | None,
    ) -> np.ndarray:
        pairs: list[tuple[int, int]] = []
        for task in active_tasks:
            if not task.is_ready:
                continue
            if executor is not None and hasattr(executor, "is_task_scheduled") and executor.is_task_scheduled(task.task_id):
                continue
            task_idx = task_id_to_idx[task.task_id]
            for uav_id in uav_ids:
                pairs.append((task_idx, uav_id_to_idx[uav_id]))
        if not pairs:
            return np.zeros((2, 0), dtype=np.int64)
        return np.asarray(pairs, dtype=np.int64).T


def _normalize_columns(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64)
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    denom = np.where(maxs > mins, maxs - mins, 1.0)
    return (values - mins) / denom


def _deterministic_kmeans(values: np.ndarray, cluster_num: int, max_iter: int = 25) -> np.ndarray:
    n = int(values.shape[0])
    k = min(max(int(cluster_num), 1), n)
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    if k == 1:
        return np.zeros((n,), dtype=np.int64)

    initial_indices = np.linspace(0, n - 1, k, dtype=int)
    centers = values[initial_indices].copy()
    labels = np.zeros((n,), dtype=np.int64)
    for _ in range(max_iter):
        distances = np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_id in range(k):
            members = values[labels == cluster_id]
            if len(members) > 0:
                centers[cluster_id] = members.mean(axis=0)
    return labels


def _stable_task_key(task_id: str) -> tuple[int, str]:
    suffix = task_id.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return int(suffix), task_id
    return 0, task_id


HeteroGraphSnapshot = CleanGraphSnapshot
HeteroGraphBuilder = CleanGraphBuilder
