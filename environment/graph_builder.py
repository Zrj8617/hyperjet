from __future__ import annotations

from dataclasses import dataclass
import config
import numpy as np

from environment import comm_model as comms
from environment.dag_tasks import DAGTaskManager, TASK_STATE_FINISHED, TASK_STATE_WAITING, TaskNode
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
    service_domain_hyperedges: list[tuple[list[str], list[int]]]
    resource_competition_hyperedges: list[tuple[list[str], list[int]]]
    collaborative_hyperedges: list[tuple[list[str], list[int]]]
    critical_hyperedges: list[list[str]]
    critical_support_hyperedges: list[tuple[list[str], list[int]]]
    compute_attribute_hyperedges: list[list[str]]
    communication_attribute_hyperedges: list[list[str]]
    candidate_scarce_attribute_hyperedges: list[list[str]]
    task_type_attribute_hyperedges: list[list[str]]
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
        service_domain_hyperedges = (
            self._build_service_domain_hyperedges(task_manager, uavs, task_uav_edges)
            if config.USE_PHASE_ONE_HYPEREDGES
            and config.USE_COLLABORATIVE_HYPEREDGES
            and config.USE_SERVICE_DOMAIN_HYPEREDGES
            else []
        )
        resource_competition_hyperedges = (
            self._build_resource_competition_hyperedges(
                task_manager,
                task_uav_edges,
                task_uav_edge_features,
                executor,
                current_time_step,
            )
            if config.USE_PHASE_ONE_HYPEREDGES
            and config.USE_COLLABORATIVE_HYPEREDGES
            and config.USE_RESOURCE_COMPETITION_HYPEREDGES
            else []
        )
        collaborative_hyperedges = service_domain_hyperedges + resource_competition_hyperedges
        critical_support_hyperedges = (
            self._build_critical_support_hyperedges(
                active_tasks,
                task_manager,
                uavs,
                task_uav_edges,
                executor,
                current_time_step,
            )
            if config.USE_PHASE_ONE_HYPEREDGES
            and config.USE_CRITICAL_HYPEREDGES
            and config.USE_CRITICAL_SUPPORT_HYPEREDGES
            else []
        )
        critical_hyperedges = (
            self._build_critical_hyperedges(
                active_tasks,
                task_manager,
                task_uav_edges,
                current_time_step,
            )
            if config.USE_PHASE_ONE_HYPEREDGES and config.USE_CRITICAL_HYPEREDGES
            else []
        )
        ready_tasks = task_manager.get_ready_tasks()
        (
            compute_attribute_hyperedges,
            communication_attribute_hyperedges,
            candidate_scarce_attribute_hyperedges,
            task_type_attribute_hyperedges,
        ) = (
            self._build_attribute_hyperedges(ready_tasks, task_uav_edges, current_time_step)
            if config.USE_PHASE_ONE_HYPEREDGES and config.USE_ATTRIBUTE_HYPEREDGES
            else ([], [], [], [])
        )
        attribute_hyperedges = (
            compute_attribute_hyperedges
            + communication_attribute_hyperedges
            + candidate_scarce_attribute_hyperedges
            + task_type_attribute_hyperedges
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
            service_domain_hyperedges=service_domain_hyperedges,
            resource_competition_hyperedges=resource_competition_hyperedges,
            collaborative_hyperedges=collaborative_hyperedges,
            critical_hyperedges=critical_hyperedges,
            critical_support_hyperedges=critical_support_hyperedges,
            compute_attribute_hyperedges=compute_attribute_hyperedges,
            communication_attribute_hyperedges=communication_attribute_hyperedges,
            candidate_scarce_attribute_hyperedges=candidate_scarce_attribute_hyperedges,
            task_type_attribute_hyperedges=task_type_attribute_hyperedges,
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
        return np.array(
            [
                np.clip(uav.pos[0] / float(config.AREA_WIDTH), 0.0, 1.0),
                np.clip(uav.pos[1] / float(config.AREA_HEIGHT), 0.0, 1.0),
                min(queue_length / float(max(config.DAG_MAX_QUEUE_PER_UAV, 1)), 1.0),
                is_busy,
                uav.remaining_energy_ratio,
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
            return np.zeros((0, config.BASE_TASK_UAV_PAIR_FEATURE_DIM), dtype=np.float32)
        pair_feature_mode = getattr(config, "TASK_UAV_PAIR_FEATURE_MODE", "full")
        if not config.USE_TASK_UAV_PAIR_FEATURES or pair_feature_mode == "none":
            return np.zeros((len(task_uav_edges), config.BASE_TASK_UAV_PAIR_FEATURE_DIM), dtype=np.float32)

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
            if pair_feature_mode == "limited":
                # Keep low-leakage descriptors; remove direct EFT/deadline-answer features.
                feature_vec[[1, 3, 4, 5]] = 0.0
            features.append(feature_vec)
        return np.stack(features, axis=0).astype(np.float32)

    def _build_service_domain_hyperedges(
        self,
        task_manager: DAGTaskManager,
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

    def _build_resource_competition_hyperedges(
        self,
        task_manager: DAGTaskManager,
        task_uav_edges: list[tuple[str, int]],
        task_uav_edge_features: np.ndarray,
        executor: PhaseOneTaskExecutor | None,
        current_time_step: float,
    ) -> list[tuple[list[str], list[int]]]:
        edge_map: dict[str, set[int]] = {}
        for task_id, uav_id in task_uav_edges:
            edge_map.setdefault(task_id, set()).add(uav_id)

        ready_task_map = {task.task_id: task for task in task_manager.get_ready_tasks() if task.task_id in edge_map}
        if len(ready_task_map) < 2:
            return []

        task_candidate_scores: dict[str, list[tuple[tuple[float, float, float, float, int], int]]] = {}
        for edge_idx, (task_id, uav_id) in enumerate(task_uav_edges):
            if task_id not in ready_task_map:
                continue
            if task_uav_edge_features.size > 0 and edge_idx < len(task_uav_edge_features):
                feature = task_uav_edge_features[edge_idx]
                planned_finish = float(feature[4])
                deadline_margin = float(feature[5])
                queue_length = float(feature[7])
                distance = float(feature[8])
            else:
                planned_finish = 0.0
                deadline_margin = 0.0
                queue_length = 0.0
                distance = 0.0
            rank_key = (planned_finish, -deadline_margin, queue_length, distance, uav_id)
            task_candidate_scores.setdefault(task_id, []).append((rank_key, uav_id))

        uav_to_top_tasks: dict[int, list[str]] = {}
        for task_id, ranked_candidates in task_candidate_scores.items():
            ranked_candidates.sort(key=lambda item: item[0])
            top_candidates = ranked_candidates[: config.DAG_RESOURCE_COMPETITION_TOP_K_UAVS]
            for _, uav_id in top_candidates:
                uav_to_top_tasks.setdefault(uav_id, []).append(task_id)

        hyperedges: list[tuple[list[str], list[int]]] = []
        for anchor_uav_id, task_ids in sorted(uav_to_top_tasks.items()):
            candidate_tasks = [ready_task_map[task_id] for task_id in task_ids if task_id in ready_task_map]
            if len(candidate_tasks) < 2:
                continue
            has_queue_pressure = executor is not None and executor.get_queue_length(anchor_uav_id) > 0
            has_task_density_pressure = len(candidate_tasks) >= 3
            has_deadline_pressure = any(
                task.remaining_slack(current_time_step) <= config.DAG_CRITICAL_SLACK_THRESHOLD
                for task in candidate_tasks
            )
            if not (has_queue_pressure or has_task_density_pressure or has_deadline_pressure):
                continue
            candidate_tasks.sort(
                key=lambda task: (
                task.deadline,
                len(edge_map.get(task.task_id, set())),
                task.remaining_slack(current_time_step),
                task.task_id,
                )
            )
            selected_tasks = candidate_tasks[: config.DAG_COLLAB_TOP_M_TASKS]
            selected_task_ids = [task.task_id for task in selected_tasks]

            if len(selected_task_ids) >= 2:
                hyperedges.append((selected_task_ids, [anchor_uav_id]))

        deduped: list[tuple[list[str], list[int]]] = []
        seen: set[tuple[tuple[str, ...], tuple[int, ...]]] = set()
        for task_ids, uav_ids in hyperedges:
            key = (tuple(sorted(task_ids)), tuple(sorted(uav_ids)))
            if key in seen:
                continue
            seen.add(key)
            deduped.append((list(key[0]), list(key[1])))
        return deduped

    def _build_critical_hyperedges(
        self,
        active_tasks: list[TaskNode],
        task_manager: DAGTaskManager,
        task_uav_edges: list[tuple[str, int]],
        current_time_step: float,
    ) -> list[list[str]]:
        task_candidates: dict[str, set[int]] = {}
        for task_id, uav_id in task_uav_edges:
            task_candidates.setdefault(task_id, set()).add(uav_id)

        dag_groups: dict[str, list[tuple[float, TaskNode]]] = {}
        for task in active_tasks:
            job_tasks = task_manager.get_job_tasks(task.dag_id)
            if not job_tasks:
                continue
            max_level = max(job_task.level for job_task in job_tasks)
            on_or_near_critical_path = task_manager.is_critical_path_task(task.task_id) or task.level >= max_level - 1
            successor_unlock_value = len(task.successors) + task_manager.get_descendant_count(task.task_id)
            is_dag_tail = task.task_type == config.TASK_TYPE_AGGREGATION or not task.successors
            is_tight = task.remaining_slack(current_time_step) <= config.DAG_CRITICAL_SLACK_THRESHOLD
            if not (on_or_near_critical_path or successor_unlock_value > 0 or is_dag_tail or is_tight):
                continue
            if task.is_ready:
                if not task_candidates.get(task.task_id):
                    continue
                predecessor_completion_ratio = 1.0
            elif task.state == TASK_STATE_WAITING and task.predecessors:
                finished_parents = sum(
                    1
                    for parent_id in task.predecessors
                    if task_manager.tasks[parent_id].state == TASK_STATE_FINISHED
                )
                predecessor_completion_ratio = finished_parents / float(max(len(task.predecessors), 1))
                if predecessor_completion_ratio < 0.5:
                    continue
            else:
                continue
            score = (
                2.0 * float(on_or_near_critical_path)
                + 1.0 * float(is_dag_tail)
                + 0.4 * float(successor_unlock_value)
                + 0.5 * predecessor_completion_ratio
                - 0.05 * task.remaining_slack(current_time_step)
            )
            dag_groups.setdefault(task.dag_id, []).append((score, task))

        hyperedges: list[list[str]] = []
        for scored_tasks in dag_groups.values():
            if len(scored_tasks) < 2:
                continue
            scored_tasks.sort(
                key=lambda item: (
                    -item[0],
                    item[1].level,
                    item[1].deadline,
                    item[1].task_id,
                )
            )
            group = [
                task.task_id
                for _, task in scored_tasks[: max(2, min(config.DAG_CRITICAL_MAX_SIZE, 4))]
            ]
            if len(group) >= 2:
                hyperedges.append(group)
        return hyperedges

    def _build_critical_support_hyperedges(
        self,
        active_tasks: list[TaskNode],
        task_manager: DAGTaskManager,
        uavs: list,
        task_uav_edges: list[tuple[str, int]],
        executor: PhaseOneTaskExecutor | None,
        current_time_step: float,
    ) -> list[tuple[list[str], list[int]]]:
        task_candidates: dict[str, set[int]] = {}
        for task_id, uav_id in task_uav_edges:
            task_candidates.setdefault(task_id, set()).add(uav_id)

        uav_map = {uav.id: uav for uav in uavs}
        max_compute = float(max(np.max(config.UAV_COMPUTING_CAPACITY), 1))

        def resource_score(uav_id: int) -> float:
            queue_length = executor.get_queue_length(uav_id) if executor is not None else 0
            queue_score = 1.0 - min(queue_length / float(max(config.DAG_MAX_QUEUE_PER_UAV, 1)), 1.0)
            energy_score = float(getattr(uav_map[uav_id], "remaining_energy_ratio", 1.0))
            compute_score = float(config.UAV_COMPUTING_CAPACITY[uav_id] / max_compute)
            return 0.5 * queue_score + 0.3 * energy_score + 0.2 * compute_score

        def link_score(uav_id: int, reference_uav_ids: set[int]) -> float:
            if not reference_uav_ids:
                return 1.0
            scores: list[float] = []
            for ref_id in reference_uav_ids:
                if ref_id == uav_id or ref_id not in uav_map:
                    scores.append(1.0)
                    continue
                distance = float(np.linalg.norm(uav_map[uav_id].pos - uav_map[ref_id].pos))
                if distance > config.A2A_MAX_RANGE:
                    scores.append(0.0)
                else:
                    scores.append(1.0 - distance / float(max(config.A2A_MAX_RANGE, config.EPSILON)))
            return float(np.mean(scores)) if scores else 1.0

        dag_candidates: list[tuple[float, str, list[TaskNode], set[int]]] = []
        for dag_id, job in task_manager.jobs.items():
            job_tasks = [task_manager.tasks[task_id] for task_id in job.task_ids if task_id in task_manager.tasks]
            active_job_tasks = [task for task in job_tasks if not task.is_terminal]
            if not active_job_tasks:
                continue
            ready_tasks = [
                task
                for task in active_job_tasks
                if task.is_ready
                and task.remaining_slack(current_time_step) <= config.DAG_CRITICAL_SLACK_THRESHOLD
                and task_candidates.get(task.task_id)
            ]
            if not ready_tasks:
                continue

            min_slack = min(task.remaining_slack(current_time_step) for task in active_job_tasks)
            urgency = 1.0 - float(np.clip(min_slack / float(max(config.DAG_MAX_DEADLINE_OFFSET, 1)), 0.0, 1.0))
            remaining_workload = float(
                np.clip(
                    sum(task.cpu_cycles for task in active_job_tasks)
                    / float(max(len(job_tasks), 1) * max(config.DAG_MAX_CPU_CYCLES, 1)),
                    0.0,
                    1.0,
                )
            )
            candidate_counts = [len(task_candidates.get(task.task_id, set())) for task in ready_tasks]
            avg_candidates = float(np.mean(candidate_counts)) if candidate_counts else 0.0
            bottleneck = 1.0 - float(np.clip(avg_candidates / float(max(config.NUM_UAVS, 1)), 0.0, 1.0))
            finished_count = sum(1 for task in job_tasks if task.state == TASK_STATE_FINISHED)
            progress_value = finished_count / float(max(len(job_tasks), 1))
            risk = 0.4 * urgency + 0.3 * remaining_workload + 0.2 * bottleneck + 0.1 * progress_value
            dag_candidates.append((risk, dag_id, ready_tasks, set()))

        dag_candidates.sort(key=lambda item: (-item[0], item[1]))
        hyperedges: list[tuple[list[str], list[int]]] = []
        for _, _, ready_tasks, _ in dag_candidates[: config.DAG_CRITICAL_SUPPORT_TOP_DAGS]:
            ready_tasks.sort(
                key=lambda task: (
                    task.remaining_slack(current_time_step),
                    -len(task.successors),
                    len(task_candidates.get(task.task_id, set())),
                    task.deadline,
                    task.task_id,
                )
            )
            selected_tasks = ready_tasks[: min(config.DAG_CRITICAL_SUPPORT_TOP_TASKS, config.DAG_CRITICAL_MAX_SIZE)]
            selected_task_ids = [task.task_id for task in selected_tasks]
            if not selected_task_ids:
                continue

            parent_result_uavs: set[int] = set()
            total_predecessors = 0
            parent_count_by_uav: dict[int, int] = {}
            for task in selected_tasks:
                for parent_id in task.predecessors:
                    parent_task = task_manager.tasks.get(parent_id)
                    if parent_task is None:
                        continue
                    total_predecessors += 1
                    if parent_task.state == TASK_STATE_FINISHED and parent_task.assigned_uav is not None:
                        parent_result_uavs.add(parent_task.assigned_uav)
                        parent_count_by_uav[parent_task.assigned_uav] = parent_count_by_uav.get(parent_task.assigned_uav, 0) + 1

            candidate_uavs = sorted(
                {
                    uav_id
                    for task in selected_tasks
                    for uav_id in task_candidates.get(task.task_id, set())
                    if uav_id in uav_map
                }
            )
            if not candidate_uavs:
                continue

            service_count_by_uav = {
                uav_id: sum(1 for task in selected_tasks if uav_id in task_candidates.get(task.task_id, set()))
                for uav_id in candidate_uavs
            }

            scored_uavs: list[tuple[float, int]] = []
            for uav_id in candidate_uavs:
                parent_score = parent_count_by_uav.get(uav_id, 0) / float(max(total_predecessors, 1))
                service_score = service_count_by_uav[uav_id] / float(max(len(selected_tasks), 1))
                score = (
                    0.35 * parent_score
                    + 0.25 * resource_score(uav_id)
                    + 0.20 * link_score(uav_id, parent_result_uavs)
                    + 0.20 * service_score
                )
                scored_uavs.append((score, uav_id))
            scored_uavs.sort(key=lambda item: (-item[0], item[1]))
            anchor_uav = scored_uavs[0][1]

            support_candidates: list[tuple[float, int]] = []
            for uav_id in candidate_uavs:
                if uav_id == anchor_uav:
                    continue
                if executor is not None and executor.get_queue_length(uav_id) >= config.DAG_MAX_QUEUE_PER_UAV:
                    continue
                anchor_distance = float(np.linalg.norm(uav_map[uav_id].pos - uav_map[anchor_uav].pos))
                if anchor_distance > config.A2A_MAX_RANGE:
                    continue
                service_score = service_count_by_uav[uav_id] / float(max(len(selected_tasks), 1))
                support_score = (
                    0.45 * link_score(uav_id, {anchor_uav})
                    + 0.35 * resource_score(uav_id)
                    + 0.20 * service_score
                )
                support_candidates.append((support_score, uav_id))
            support_candidates.sort(key=lambda item: (-item[0], item[1]))
            support_uavs = [
                uav_id
                for _, uav_id in support_candidates[: config.DAG_CRITICAL_SUPPORT_MAX_NEIGHBORS]
            ]
            selected_uavs = [anchor_uav] + support_uavs
            selected_uavs = selected_uavs[: config.DAG_CRITICAL_SUPPORT_MAX_UAVS]
            if selected_uavs:
                hyperedges.append((selected_task_ids, selected_uavs))

        deduped: list[tuple[list[str], list[int]]] = []
        seen: set[tuple[tuple[str, ...], tuple[int, ...]]] = set()
        for task_ids, uav_ids in hyperedges:
            key = (tuple(sorted(task_ids)), tuple(sorted(uav_ids)))
            if key in seen:
                continue
            seen.add(key)
            deduped.append((list(key[0]), list(key[1])))
        return deduped

    def _build_attribute_hyperedges(
        self,
        ready_tasks: list[TaskNode],
        task_uav_edges: list[tuple[str, int]],
        current_time_step: float,
    ) -> tuple[list[list[str]], list[list[str]], list[list[str]], list[list[str]]]:
        if len(ready_tasks) < 2:
            return [], [], [], []

        candidate_count_by_task: dict[str, int] = {}
        for task_id, _ in task_uav_edges:
            candidate_count_by_task[task_id] = candidate_count_by_task.get(task_id, 0) + 1

        max_input = float(max(config.DAG_MAX_INPUT_SIZE, 1))
        max_output = float(max(config.DAG_MAX_OUTPUT_SIZE, 1))
        max_cycles = float(max(config.DAG_MAX_CPU_CYCLES, 1))

        seen: set[tuple[str, ...]] = set()

        def build_group(tasks: list[TaskNode]) -> list[list[str]]:
            members = [task.task_id for task in tasks[: config.DAG_ATTRIBUTE_TOP_M_TASKS]]
            if len(members) < 2:
                return []
            key = tuple(sorted(members))
            if key in seen:
                return []
            seen.add(key)
            return [list(key)]

        compute_heavy_tasks = sorted(
            ready_tasks,
            key=lambda task: (
                -float(np.clip(task.cpu_cycles / max_cycles, 0.0, 1.0)),
                task.deadline,
                task.task_id,
            ),
        )
        compute_hyperedges = (
            build_group(compute_heavy_tasks)
            if config.USE_COMPUTE_ATTRIBUTE_HYPEREDGES
            else []
        )

        communication_heavy_tasks = sorted(
            ready_tasks,
            key=lambda task: (
                -(
                    float(np.clip(task.input_size / max_input, 0.0, 1.0))
                    + float(np.clip(task.output_size / max_output, 0.0, 1.0))
                ),
                task.deadline,
                task.task_id,
            ),
        )
        communication_hyperedges = (
            build_group(communication_heavy_tasks)
            if config.USE_COMMUNICATION_ATTRIBUTE_HYPEREDGES
            else []
        )

        candidate_scarce_tasks = sorted(
            [task for task in ready_tasks if candidate_count_by_task.get(task.task_id, 0) > 0],
            key=lambda task: (
                candidate_count_by_task.get(task.task_id, 0),
                task.deadline,
                -task.cpu_cycles,
                task.task_id,
            ),
        )
        candidate_scarce_hyperedges = (
            build_group(candidate_scarce_tasks)
            if config.USE_CANDIDATE_SCARCE_ATTRIBUTE_HYPEREDGES
            else []
        )

        task_type_hyperedges: list[list[str]] = []
        if config.USE_TASK_TYPE_ATTRIBUTE_HYPEREDGES:
            for task_type in (
                config.TASK_TYPE_PREPROCESS,
                config.TASK_TYPE_COMPUTE,
                config.TASK_TYPE_AGGREGATION,
            ):
                typed_tasks = sorted(
                    [task for task in ready_tasks if task.task_type == task_type],
                    key=lambda task: (task.deadline, task.task_id),
                )
                task_type_hyperedges.extend(build_group(typed_tasks))

        return compute_hyperedges, communication_hyperedges, candidate_scarce_hyperedges, task_type_hyperedges

    def _estimate_ue_uav_rate(self, source_pos: np.ndarray, uav_pos: np.ndarray) -> float:
        return comms.calculate_g2a_rate(source_pos, uav_pos, 1)

    def _estimate_uav_uav_transfer_time(self, data_size: float, pos_a: np.ndarray, pos_b: np.ndarray) -> float:
        return comms.calculate_a2a_transfer_time(data_size, pos_a, pos_b)
