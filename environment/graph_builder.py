from __future__ import annotations

from dataclasses import dataclass
import config
import numpy as np

from environment import comm_model as comms
from environment.dag_tasks import DAGTaskManager, TaskNode
from environment.task_execution import PhaseOneTaskExecutor


@dataclass(slots=True)
class HeteroGraphSnapshot:
    task_ids: list[str]
    uav_ids: list[int]
    task_features: np.ndarray
    uav_features: np.ndarray
    dependency_edges: list[tuple[str, str]]
    task_uav_edges: list[tuple[str, int]]
    task_uav_edge_features: np.ndarray
    uav_uav_edges: list[tuple[int, int]]
    collaborative_hyperedges: list[tuple[list[str], list[int]]]
    critical_hyperedges: list[list[str]]
    attribute_hyperedges: list[list[str]]


class HeteroGraphBuilder:
    """Builds a lightweight heterogeneous graph snapshot for phase one."""

    def build(
        self,
        task_manager: DAGTaskManager,
        uavs: list,
        current_time_step: float,
        executor: PhaseOneTaskExecutor | None = None,
    ) -> HeteroGraphSnapshot:
        active_tasks: list[TaskNode] = task_manager.get_active_tasks()
        task_ids = [task.task_id for task in active_tasks]
        uav_ids = [uav.id for uav in uavs]

        task_features_dict = task_manager.build_task_features(current_time_step)
        task_features = (
            np.stack([task_features_dict[task_id] for task_id in task_ids], axis=0)
            if task_ids
            else np.zeros((0, config.DAG_TASK_FEATURE_DIM), dtype=np.float32)
        )
        uav_features = np.stack([self._build_uav_feature(uav, uavs, executor) for uav in uavs], axis=0)

        dependency_edges: list[tuple[str, str]] = []
        for task in active_tasks:
            for parent_id in task.predecessors:
                if parent_id in task_manager.tasks and not task_manager.tasks[parent_id].is_terminal:
                    dependency_edges.append((parent_id, task.task_id))

        task_uav_edges = self._build_task_uav_edges(active_tasks, task_manager, uavs, current_time_step, executor)
        task_uav_edge_features = self._build_task_uav_edge_features(task_uav_edges, task_manager, uavs, current_time_step, executor)
        uav_uav_edges = self._build_uav_uav_edges(uavs)
        collaborative_hyperedges = (
            self._build_collaborative_hyperedges(task_manager, active_tasks, uavs, task_uav_edges)
            if config.USE_PHASE_ONE_HYPEREDGES and config.USE_COLLABORATIVE_HYPEREDGES
            else []
        )
        critical_hyperedges = (
            self._build_critical_hyperedges(active_tasks, current_time_step)
            if config.USE_PHASE_ONE_HYPEREDGES and config.USE_CRITICAL_HYPEREDGES
            else []
        )
        attribute_hyperedges = (
            self._build_attribute_hyperedges(active_tasks, current_time_step)
            if config.USE_PHASE_ONE_HYPEREDGES and config.USE_ATTRIBUTE_HYPEREDGES
            else []
        )

        return HeteroGraphSnapshot(
            task_ids=task_ids,
            uav_ids=uav_ids,
            task_features=task_features,
            uav_features=uav_features,
            dependency_edges=dependency_edges,
            task_uav_edges=task_uav_edges,
            task_uav_edge_features=task_uav_edge_features,
            uav_uav_edges=uav_uav_edges,
            collaborative_hyperedges=collaborative_hyperedges,
            critical_hyperedges=critical_hyperedges,
            attribute_hyperedges=attribute_hyperedges,
        )

    def _build_uav_feature(self, uav, uavs: list, executor: PhaseOneTaskExecutor | None) -> np.ndarray:
        queue_length = executor.get_queue_length(uav.id) if executor is not None else 0
        is_busy = 1.0 if executor is not None and executor.is_uav_busy(uav.id) else 0.0
        neighbor_count = sum(
            1
            for other_uav in uavs
            if other_uav.id != uav.id and float(np.linalg.norm(uav.pos - other_uav.pos)) <= config.UAV_SENSING_RANGE
        )
        local_energy_load = uav.energy / max(config.POWER_MOVE + config.POWER_HOVER, config.EPSILON)
        return np.array(
            [
                uav.pos[0] / float(config.AREA_WIDTH),
                uav.pos[1] / float(config.AREA_HEIGHT),
                min(queue_length / float(max(config.DAG_MAX_QUEUE_PER_UAV, 1)), 1.0),
                is_busy,
                min(local_energy_load, 1.0),
                neighbor_count / float(max(config.NUM_UAVS - 1, 1)),
                config.UAV_COMPUTING_CAPACITY[uav.id] / float(np.max(config.UAV_COMPUTING_CAPACITY)),
            ],
            dtype=np.float32,
        )

    def _build_task_uav_edges(
        self,
        active_tasks: list[TaskNode],
        task_manager: DAGTaskManager,
        uavs: list,
        current_time_step: float,
        executor: PhaseOneTaskExecutor | None,
    ) -> list[tuple[str, int]]:
        edges: list[tuple[str, int]] = []
        for task in active_tasks:
            if not task.is_ready:
                continue
            for uav in uavs:
                if self._is_task_uav_feasible(task, task_manager, uav, uavs, current_time_step, executor):
                    edges.append((task.task_id, uav.id))
        return edges

    def _is_task_uav_feasible(
        self,
        task: TaskNode,
        task_manager: DAGTaskManager,
        uav,
        uavs: list,
        current_time_step: float,
        executor: PhaseOneTaskExecutor | None,
    ) -> bool:
        distance = float(np.linalg.norm(task.source_pos - uav.pos[:2]))
        if distance > config.DAG_TASK_UAV_MAX_DISTANCE:
            return False
        if not task.is_ready:
            return False
        upload_rate = self._estimate_ue_uav_rate(task.source_pos, uav.pos)
        if upload_rate <= 0.0:
            return False
        if executor is not None and executor.get_queue_length(uav.id) >= config.DAG_MAX_QUEUE_PER_UAV:
            return False

        predecessor_ready_time = float(current_time_step)
        predecessor_transfer_time = 0.0
        for parent_id in task.predecessors:
            parent_task = task_manager.tasks[parent_id]
            if parent_task.finish_time is None or parent_task.assigned_uav is None:
                return False
            predecessor_ready_time = max(predecessor_ready_time, float(parent_task.finish_time))
            if parent_task.assigned_uav != uav.id:
                parent_uav = uavs[parent_task.assigned_uav]
                transfer_time = self._estimate_uav_uav_transfer_time(parent_task.output_size, parent_uav.pos, uav.pos)
                if not np.isfinite(transfer_time):
                    return False
                predecessor_transfer_time = max(predecessor_transfer_time, transfer_time)

        upload_time = task.input_size / upload_rate
        compute_time = task.cpu_cycles / float(config.UAV_COMPUTING_CAPACITY[uav.id])
        available_time = executor.get_available_time(uav.id) if executor is not None else float(current_time_step)
        earliest_start = max(float(current_time_step), available_time, predecessor_ready_time + predecessor_transfer_time, float(current_time_step) + upload_time)
        planned_finish = earliest_start + compute_time
        if planned_finish > task.deadline + config.DAG_MAX_DEADLINE_TOLERANCE:
            return False
        return True

    def _build_uav_uav_edges(self, uavs: list) -> list[tuple[int, int]]:
        edges: list[tuple[int, int]] = []
        for i, uav_i in enumerate(uavs):
            for j in range(i + 1, len(uavs)):
                uav_j = uavs[j]
                distance = float(np.linalg.norm(uav_i.pos - uav_j.pos))
                if distance <= config.UAV_SENSING_RANGE:
                    edges.append((uav_i.id, uav_j.id))
        return edges

    def _build_task_uav_edge_features(
        self,
        task_uav_edges: list[tuple[str, int]],
        task_manager: DAGTaskManager,
        uavs: list,
        current_time_step: float,
        executor: PhaseOneTaskExecutor | None,
    ) -> np.ndarray:
        if not task_uav_edges:
            return np.zeros((0, 9), dtype=np.float32)
        if not config.USE_TASK_UAV_PAIR_FEATURES:
            return np.zeros((len(task_uav_edges), 9), dtype=np.float32)

        uav_map = {uav.id: uav for uav in uavs}
        features: list[np.ndarray] = []
        max_deadline = float(max(config.DAG_MAX_DEADLINE_OFFSET + config.DAG_MAX_DEADLINE_TOLERANCE, 1))
        max_compute = float(max(np.max(config.UAV_COMPUTING_CAPACITY), 1))
        max_distance = float(max(config.DAG_TASK_UAV_MAX_DISTANCE, 1.0))
        for task_id, uav_id in task_uav_edges:
            task = task_manager.tasks[task_id]
            uav = uav_map[uav_id]
            upload_rate = self._estimate_ue_uav_rate(task.source_pos, uav.pos)
            upload_time = task.input_size / max(upload_rate, config.EPSILON)

            predecessor_ready_time = float(current_time_step)
            predecessor_transfer_time = 0.0
            cross_uav_count = 0
            for parent_id in task.predecessors:
                parent_task = task_manager.tasks[parent_id]
                if parent_task.finish_time is None or parent_task.assigned_uav is None:
                    continue
                predecessor_ready_time = max(predecessor_ready_time, float(parent_task.finish_time))
                if parent_task.assigned_uav != uav.id:
                    cross_uav_count += 1
                    parent_uav = uav_map[parent_task.assigned_uav]
                    transfer_time = self._estimate_uav_uav_transfer_time(parent_task.output_size, parent_uav.pos, uav.pos)
                    if np.isfinite(transfer_time):
                        predecessor_transfer_time = max(predecessor_transfer_time, transfer_time)

            compute_time = task.cpu_cycles / float(config.UAV_COMPUTING_CAPACITY[uav.id])
            available_time = executor.get_available_time(uav.id) if executor is not None else float(current_time_step)
            earliest_start = max(
                float(current_time_step),
                available_time,
                predecessor_ready_time + predecessor_transfer_time,
                float(current_time_step) + upload_time,
            )
            planned_finish = earliest_start + compute_time
            distance = float(np.linalg.norm(task.source_pos - uav.pos[:2]))
            queue_length = executor.get_queue_length(uav.id) if executor is not None else 0
            parent_ratio = cross_uav_count / float(max(len(task.predecessors), 1))
            feature_vec = np.array(
                [
                    upload_time / max_deadline,
                    predecessor_transfer_time / max_deadline,
                    compute_time / max_deadline,
                    max(available_time - float(current_time_step), 0.0) / max_deadline,
                    max(planned_finish - float(current_time_step), 0.0) / max_deadline,
                    (task.deadline - planned_finish) / max_deadline,
                    parent_ratio,
                    queue_length / float(max(config.DAG_MAX_QUEUE_PER_UAV, 1)),
                    distance / max_distance,
                ],
                dtype=np.float32,
            )
            features.append(feature_vec)
        return np.stack(features, axis=0).astype(np.float32)

    def _build_collaborative_hyperedges(
        self,
        task_manager: DAGTaskManager,
        active_tasks: list[TaskNode],
        uavs: list,
        task_uav_edges: list[tuple[str, int]],
    ) -> list[tuple[list[str], list[int]]]:
        edge_map: dict[str, list[int]] = {}
        for task_id, uav_id in task_uav_edges:
            edge_map.setdefault(task_id, []).append(uav_id)

        ready_tasks = sorted(task_manager.get_ready_tasks(), key=lambda task: (task.deadline, task.level))
        if not ready_tasks:
            return []

        hyperedges: list[tuple[list[str], list[int]]] = []
        anchor_tasks = ready_tasks[: config.DAG_COLLAB_TOP_M_TASKS]
        for anchor_task in anchor_tasks:
            local_task_objs = [
                task
                for task in ready_tasks
                if float(np.linalg.norm(task.source_pos - anchor_task.source_pos)) <= config.UAV_COVERAGE_RADIUS
            ]
            local_task_objs.sort(key=lambda task: (task.deadline, task.level))
            local_tasks = [task.task_id for task in local_task_objs[: config.DAG_COLLAB_TOP_M_TASKS]]
            if not local_tasks:
                continue

            candidate_scores: list[tuple[int, int]] = []
            for uav in uavs:
                covered_local_tasks = sum(1 for task_id in local_tasks if uav.id in edge_map.get(task_id, []))
                if covered_local_tasks > 0:
                    candidate_scores.append((covered_local_tasks, uav.id))
            candidate_scores.sort(key=lambda item: (-item[0], item[1]))
            candidate_uavs = [uav_id for _, uav_id in candidate_scores[: config.DAG_COLLAB_TOP_K_UAVS]]
            if candidate_uavs:
                hyperedges.append((local_tasks, candidate_uavs))

        deduped: list[tuple[list[str], list[int]]] = []
        seen: set[tuple[tuple[str, ...], tuple[int, ...]]] = set()
        for task_ids, uav_ids in hyperedges:
            key = (tuple(sorted(task_ids)), tuple(sorted(uav_ids)))
            if key in seen:
                continue
            seen.add(key)
            deduped.append((list(key[0]), list(key[1])))
        return deduped

    def _build_critical_hyperedges(self, active_tasks: list[TaskNode], current_time_step: float) -> list[list[str]]:
        urgent_tasks = [
            task.task_id
            for task in active_tasks
            if task.remaining_slack(current_time_step) <= config.DAG_CRITICAL_SLACK_THRESHOLD
        ]
        return [urgent_tasks] if len(urgent_tasks) >= 2 else []

    def _build_attribute_hyperedges(self, active_tasks: list[TaskNode], current_time_step: float) -> list[list[str]]:
        if len(active_tasks) < 2:
            return []

        task_features: dict[str, np.ndarray] = {}
        max_input = float(max(config.DAG_MAX_INPUT_SIZE, 1))
        max_cycles = float(max(config.DAG_MAX_CPU_CYCLES, 1))
        max_slack = float(max(config.DAG_MAX_DEADLINE_OFFSET, 1))
        max_level = float(max(config.DAG_MAX_TASK_LEVELS - 1, 1))
        for task in active_tasks:
            task_features[task.task_id] = np.array(
                [
                    task.input_size / max_input,
                    task.cpu_cycles / max_cycles,
                    max(task.remaining_slack(current_time_step), 0.0) / max_slack,
                    task.level / max_level,
                    1.0 if task.task_type == config.TASK_TYPE_PREPROCESS else 0.0,
                    1.0 if task.task_type == config.TASK_TYPE_COMPUTE else 0.0,
                    1.0 if task.task_type == config.TASK_TYPE_AGGREGATION else 0.0,
                ],
                dtype=np.float32,
            )

        anchors = sorted(
            active_tasks,
            key=lambda task: (
                0 if task.is_ready else 1,
                task.deadline,
                -task.cpu_cycles,
            ),
        )[: config.DAG_ATTRIBUTE_MAX_GROUPS]

        hyperedges: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for anchor in anchors:
            anchor_vec = task_features[anchor.task_id]
            neighbor_scores: list[tuple[float, str]] = []
            for task in active_tasks:
                candidate_vec = task_features[task.task_id]
                distance = float(np.linalg.norm(anchor_vec - candidate_vec))
                neighbor_scores.append((distance, task.task_id))
            neighbor_scores.sort(key=lambda item: (item[0], item[1]))
            members = [task_id for _, task_id in neighbor_scores[: config.DAG_ATTRIBUTE_TOP_M_TASKS]]
            if len(members) < 2:
                continue
            key = tuple(sorted(members))
            if key in seen:
                continue
            seen.add(key)
            hyperedges.append(list(key))

        return hyperedges

    def _estimate_ue_uav_rate(self, source_pos: np.ndarray, uav_pos: np.ndarray) -> float:
        return comms.calculate_g2a_rate(source_pos, uav_pos, 1)

    def _estimate_uav_uav_transfer_time(self, data_size: float, pos_a: np.ndarray, pos_b: np.ndarray) -> float:
        return comms.calculate_a2a_transfer_time(data_size, pos_a, pos_b)
