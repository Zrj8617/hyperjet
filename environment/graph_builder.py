from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Any

import numpy as np

import config
from environment.dag_tasks import DAGTaskManager, TaskNode


@dataclass(slots=True)
class CleanGraphSnapshot:
    current_time_step: int
    task_ids: list[str]
    uav_ids: list[int]
    task_id_to_idx: dict[str, int]
    uav_id_to_idx: dict[int, int]
    task_features: np.ndarray
    uav_features: np.ndarray
    dag_dependency_edges: np.ndarray
    khop_dependency_hyperedges: list[list[int]]
    attribute_hyperedges: list[list[int]]
    ready_task_to_uav_candidate_pairs: np.ndarray


class CleanGraphBuilder:
    """Clean mainline graph snapshot builder for zrj_3."""

    def __init__(self) -> None:
        self._cached_attribute_groups: list[list[str]] = []
        self._last_attribute_update_step: int | None = None
        self._last_attribute_task_ids: tuple[str, ...] = ()

    def reset(self) -> None:
        self._cached_attribute_groups = []
        self._last_attribute_update_step = None
        self._last_attribute_task_ids = ()

    def build(
        self,
        task_manager: DAGTaskManager,
        uavs: list[Any],
        current_time_step: int,
        executor: Any | None = None,
    ) -> CleanGraphSnapshot:
        active_tasks = sorted(task_manager.get_active_tasks(), key=lambda task: _stable_task_key(task.task_id))
        task_ids = [task.task_id for task in active_tasks]
        uav_ids = [int(uav.id) for uav in uavs]
        task_id_to_idx = {task_id: idx for idx, task_id in enumerate(task_ids)}
        uav_id_to_idx = {uav_id: idx for idx, uav_id in enumerate(uav_ids)}

        task_features = self._build_task_features(active_tasks, task_manager)
        uav_features = self._build_uav_features(uavs, executor)
        dag_dependency_edges = self._build_dependency_edges(active_tasks, task_id_to_idx)
        khop_dependency_hyperedges = self._build_khop_hyperedges(active_tasks, task_id_to_idx)
        attribute_hyperedges = self._build_attribute_hyperedges(
            active_tasks=active_tasks,
            task_manager=task_manager,
            task_id_to_idx=task_id_to_idx,
            current_time_step=int(current_time_step),
        )
        candidate_pairs = self._build_candidate_pairs(active_tasks, uav_ids, task_id_to_idx, uav_id_to_idx)

        return CleanGraphSnapshot(
            current_time_step=int(current_time_step),
            task_ids=task_ids,
            uav_ids=uav_ids,
            task_id_to_idx=task_id_to_idx,
            uav_id_to_idx=uav_id_to_idx,
            task_features=task_features,
            uav_features=uav_features,
            dag_dependency_edges=dag_dependency_edges,
            khop_dependency_hyperedges=khop_dependency_hyperedges,
            attribute_hyperedges=attribute_hyperedges,
            ready_task_to_uav_candidate_pairs=candidate_pairs,
        )

    def _build_task_features(
        self,
        active_tasks: list[TaskNode],
        task_manager: DAGTaskManager,
    ) -> np.ndarray:
        if not active_tasks:
            return np.zeros((0, 10), dtype=np.float32)

        max_upload = max(float(max(config.BASE_UPLOAD_BANDWIDTH_MBPS)), 1.0)
        max_download = max(float(max(config.BASE_DOWNLOAD_BANDWIDTH_MBPS)), 1.0)
        max_input = max(float(config.INPUT_DATA_SIZE_MB_RANGE[1]), 1.0)
        max_output = max(float(config.OUTPUT_DATA_SIZE_MB_RANGE[1]), 1.0)
        max_operation = max(max(task.num_operation for task in active_tasks), 1.0)
        max_level = max(float(config.DAG_MAX_LEVELS - 1), 1.0)

        rows: list[np.ndarray] = []
        for task in active_tasks:
            job = task_manager.get_job(task.dag_id)
            base_upload = 0.0 if job is None else float(job.base_upload_bandwidth_mbps)
            base_download = 0.0 if job is None else float(job.base_download_bandwidth_mbps)
            rows.append(
                np.array(
                    [
                        np.clip(base_upload / max_upload, 0.0, 1.0),
                        np.clip(base_download / max_download, 0.0, 1.0),
                        np.clip(task.input_data_size_mb / max_input, 0.0, 1.0),
                        np.clip(task.output_data_size_mb / max_output, 0.0, 1.0),
                        np.clip(task.num_operation / max_operation, 0.0, 1.0),
                        np.clip(task.level / max_level, 0.0, 1.0),
                        1.0 if task.is_ready else 0.0,
                        1.0 if task.is_critical_path else 0.0,
                        np.clip(task.source_pos[0] / float(config.AREA_WIDTH), 0.0, 1.0),
                        np.clip(task.source_pos[1] / float(config.AREA_HEIGHT), 0.0, 1.0),
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

    def _build_dependency_edges(
        self,
        active_tasks: list[TaskNode],
        task_id_to_idx: dict[str, int],
    ) -> np.ndarray:
        if not config.ENABLE_DAG_DEPENDENCY_EDGES:
            return np.zeros((2, 0), dtype=np.int64)
        edges: list[tuple[int, int]] = []
        for task in active_tasks:
            parent_idx = task_id_to_idx[task.task_id]
            for child_id in sorted(task.successors, key=_stable_task_key):
                if child_id in task_id_to_idx:
                    edges.append((parent_idx, task_id_to_idx[child_id]))
        if not edges:
            return np.zeros((2, 0), dtype=np.int64)
        return np.asarray(edges, dtype=np.int64).T

    def _build_khop_hyperedges(
        self,
        active_tasks: list[TaskNode],
        task_id_to_idx: dict[str, int],
    ) -> list[list[int]]:
        if not config.ENABLE_KHOP_DEPENDENCY_HYPEREDGES:
            return []
        task_map = {task.task_id: task for task in active_tasks}
        max_hops = max(int(config.KHOP_K), 0)
        dedup: set[tuple[int, ...]] = set()
        output: list[list[int]] = []
        for task in active_tasks:
            reachable: set[int] = {task_id_to_idx[task.task_id]}
            queue: deque[tuple[str, int]] = deque((child_id, 1) for child_id in task.successors if child_id in task_map)
            while queue:
                task_id, depth = queue.popleft()
                if depth > max_hops or task_id not in task_id_to_idx:
                    continue
                reachable.add(task_id_to_idx[task_id])
                if depth == max_hops:
                    continue
                for child_id in task_map[task_id].successors:
                    if child_id in task_map:
                        queue.append((child_id, depth + 1))
            if len(reachable) < 2:
                continue
            group = tuple(sorted(reachable))
            if group in dedup:
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
    ) -> list[list[int]]:
        if not config.ENABLE_ATTRIBUTE_HYPEREDGES or len(active_tasks) < 2:
            self._cached_attribute_groups = []
            self._last_attribute_task_ids = tuple(task.task_id for task in active_tasks)
            return []

        task_ids = tuple(task.task_id for task in active_tasks)
        interval = max(int(config.ATTRIBUTE_HYPEREDGE_UPDATE_INTERVAL), 1)
        should_update = (
            self._last_attribute_update_step is None
            or current_time_step % interval == 0
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
            self._cached_attribute_groups = groups
            self._last_attribute_update_step = current_time_step
            self._last_attribute_task_ids = task_ids

        hyperedges: list[list[int]] = []
        for group in self._cached_attribute_groups:
            indices = sorted(task_id_to_idx[task_id] for task_id in group if task_id in task_id_to_idx)
            if len(indices) >= 2:
                hyperedges.append(indices)
        return hyperedges

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
    ) -> np.ndarray:
        pairs: list[tuple[int, int]] = []
        for task in active_tasks:
            if not task.is_ready:
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
