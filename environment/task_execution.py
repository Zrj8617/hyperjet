from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Callable
import math
import numpy as np

import config
from environment import comm_model as comms
from environment.dag_tasks import DAGTaskManager, TaskNode


@dataclass(slots=True)
class ScheduledTask:
    task_id: str
    uav_id: int
    planned_start: float
    planned_finish: float
    transmission_time: float
    execution_time: float
    total_energy: float


@dataclass(slots=True)
class PhaseOneStepStats:
    completed_tasks: int = 0
    on_time_completed_tasks: int = 0
    deadline_violations: int = 0
    invalid_actions: int = 0
    newly_assigned_tasks: int = 0
    score_selected_assignments: int = 0
    fallback_selected_assignments: int = 0
    score_heuristic_disagreements: int = 0
    score_raw_disagreements: int = 0
    score_guard_fallback_assignments: int = 0
    agreement_guard_rejections: int = 0
    bounded_guard_rejections: int = 0
    bounded_guard_clamps: int = 0
    step_delay: float = 0.0
    step_energy: float = 0.0
    completed_tasks_by_uav: dict[int, int] = field(default_factory=dict)
    on_time_completed_tasks_by_uav: dict[int, int] = field(default_factory=dict)
    progress_by_uav: dict[int, float] = field(default_factory=dict)


@dataclass(slots=True)
class TaskSupervisionTarget:
    task_id: str
    feasible_uav_ids: list[int]
    heuristic_best_uav: int
    heuristic_eft_by_uav: dict[int, float]
    heuristic_score_by_uav: dict[int, float]


@dataclass(slots=True)
class AssignmentCandidateRecord:
    uav_id: int
    planned_start: float
    planned_finish: float
    transmission_time: float
    execution_time: float
    total_energy: float
    score: float | None
    queue_length: int
    available_time: float


@dataclass(slots=True)
class AssignmentDecisionRecord:
    task_id: str
    task_state: str
    task_level: int
    task_type: int
    task_input_size: float
    task_cpu_cycles: float
    task_arrival_time: float
    task_deadline: float
    task_slack: float
    num_predecessors: int
    num_successors: int
    heuristic_uav: int | None
    score_uav: int | None
    selected_uav: int | None
    selection_mode: str | None
    disagrees_with_heuristic: bool
    candidates: list[AssignmentCandidateRecord]
    raw_score_uav: int | None = None
    raw_disagrees_with_heuristic: bool = False
    guard_reason: str | None = None


class PhaseOneTaskExecutor:
    """Assignment and execution engine for phase-one DAG scheduling."""

    def __init__(self) -> None:
        self._uav_available_time: dict[int, float] = {}
        self._queued: dict[int, list[tuple[float, str]]] = {}
        self._running: dict[int, ScheduledTask | None] = {}
        self._scheduled: dict[str, ScheduledTask] = {}
        self._finished_count: int = 0
        self._on_time_finished_count: int = 0
        self._deadline_violations: int = 0
        self._score_selected_assignments: int = 0
        self._fallback_selected_assignments: int = 0
        self._score_heuristic_disagreements: int = 0
        self._latest_assignment_records: list[AssignmentDecisionRecord] = []
        self._latest_stats: PhaseOneStepStats = PhaseOneStepStats()

    @property
    def latest_stats(self) -> PhaseOneStepStats:
        return self._latest_stats

    @property
    def latest_assignment_records(self) -> list[AssignmentDecisionRecord]:
        return self._latest_assignment_records

    def get_queue_length(self, uav_id: int) -> int:
        return len(self._queued.get(uav_id, []))

    def is_uav_busy(self, uav_id: int) -> bool:
        return self._running.get(uav_id) is not None

    def get_available_time(self, uav_id: int) -> float:
        return float(self._uav_available_time.get(uav_id, 0.0))

    def reset(self, uavs: list) -> None:
        self._uav_available_time = {uav.id: 0.0 for uav in uavs}
        self._queued = {uav.id: [] for uav in uavs}
        self._running = {uav.id: None for uav in uavs}
        self._scheduled.clear()
        self._finished_count = 0
        self._on_time_finished_count = 0
        self._deadline_violations = 0
        self._score_selected_assignments = 0
        self._fallback_selected_assignments = 0
        self._score_heuristic_disagreements = 0
        self._latest_assignment_records = []
        self._latest_stats = PhaseOneStepStats()

    def assign_ready_tasks(
        self,
        task_manager: DAGTaskManager,
        uavs: list,
        current_time_step: float,
        allowed_edges: set[tuple[str, int]] | None = None,
        edge_scores: dict[tuple[str, int], float] | None = None,
        score_provider: Callable[[TaskNode], tuple[set[tuple[str, int]] | None, dict[tuple[str, int], float] | None]] | None = None,
    ) -> None:
        stats = PhaseOneStepStats()
        self._latest_assignment_records = []
        ready_tasks = sorted(task_manager.get_ready_tasks(), key=lambda task: (task.deadline, task.level))
        for task in ready_tasks:
            if task.task_id in self._scheduled:
                continue
            task_allowed_edges = allowed_edges
            task_edge_scores = edge_scores
            if score_provider is not None:
                task_allowed_edges, task_edge_scores = score_provider(task)
            best_schedule, selection_mode, disagrees_with_heuristic, decision_record = self._select_best_uav(
                task,
                task_manager,
                uavs,
                current_time_step,
                task_allowed_edges,
                task_edge_scores,
            )
            if decision_record is not None:
                self._latest_assignment_records.append(decision_record)
            if best_schedule is None:
                stats.invalid_actions += 1
                continue
            self._scheduled[task.task_id] = best_schedule
            task_manager.mark_task_queued(task.task_id, best_schedule.uav_id, current_time_step)
            heapq.heappush(self._queued[best_schedule.uav_id], (best_schedule.planned_start, task.task_id))
            self._uav_available_time[best_schedule.uav_id] = best_schedule.planned_finish
            stats.newly_assigned_tasks += 1
            if selection_mode == "score":
                self._score_selected_assignments += 1
                stats.score_selected_assignments += 1
            elif selection_mode == "fallback":
                self._fallback_selected_assignments += 1
                stats.fallback_selected_assignments += 1
            elif selection_mode == "guard_fallback":
                self._fallback_selected_assignments += 1
                stats.fallback_selected_assignments += 1
                stats.score_guard_fallback_assignments += 1
                if decision_record is not None:
                    if decision_record.guard_reason == "agreement_only":
                        stats.agreement_guard_rejections += 1
                    elif decision_record.guard_reason == "runtime_bounded_guard":
                        stats.bounded_guard_rejections += 1
            if decision_record is not None and decision_record.guard_reason == "runtime_bounded_guard_clamp":
                stats.bounded_guard_clamps += 1
            if disagrees_with_heuristic:
                self._score_heuristic_disagreements += 1
                stats.score_heuristic_disagreements += 1
            if decision_record is not None and decision_record.raw_disagrees_with_heuristic:
                stats.score_raw_disagreements += 1
        self._latest_stats = stats

    def advance_one_slot(self, task_manager: DAGTaskManager, uavs: list, current_time_step: float) -> PhaseOneStepStats:
        stats = self._latest_stats
        step_start = float(current_time_step)
        step_end = float(current_time_step + config.TIME_SLOT_DURATION)

        for uav in uavs:
            slot_cursor = step_start
            while slot_cursor < step_end - config.EPSILON:
                running = self._running[uav.id]
                if running is None:
                    self._start_due_task(task_manager, uav.id, step_end)
                    running = self._running[uav.id]
                    if running is None:
                        break

                active_start = max(slot_cursor, running.planned_start)
                active_end = min(step_end, running.planned_finish)
                active_duration = max(0.0, active_end - active_start)
                if active_duration > 0:
                    energy_share = running.total_energy * (
                        active_duration / max(running.planned_finish - running.planned_start, config.EPSILON)
                    )
                    uav._energy_current_slot += energy_share
                    stats.step_energy += energy_share
                    stats.progress_by_uav[uav.id] = (
                        stats.progress_by_uav.get(uav.id, 0.0)
                        + active_duration / max(config.TIME_SLOT_DURATION, config.EPSILON)
                    )
                    slot_cursor = active_end
                else:
                    slot_cursor = max(slot_cursor, running.planned_start)

                if running.planned_finish <= step_end + config.EPSILON:
                    self._finish_running_task(task_manager, running, uav, running.planned_finish, stats)
                    self._running[uav.id] = None
                    continue
                break

        stats.deadline_violations = self._deadline_violations
        self._latest_stats = stats
        return stats

    def get_summary(self) -> dict[str, float]:
        completion_ratio = self._on_time_finished_count / max(self._finished_count, 1)
        return {
            "finished_count": float(self._finished_count),
            "on_time_ratio": float(completion_ratio),
            "deadline_violations": float(self._deadline_violations),
            "score_selected_assignments": float(self._score_selected_assignments),
            "fallback_selected_assignments": float(self._fallback_selected_assignments),
            "score_heuristic_disagreements": float(self._score_heuristic_disagreements),
        }

    def build_supervision_targets(
        self,
        task_manager: DAGTaskManager,
        uavs: list,
        current_time_step: float,
        allowed_edges: set[tuple[str, int]] | None = None,
    ) -> list[TaskSupervisionTarget]:
        targets: list[TaskSupervisionTarget] = []
        ready_tasks = sorted(task_manager.get_ready_tasks(), key=lambda task: (task.deadline, task.level))
        for task in ready_tasks:
            candidates: list[ScheduledTask] = []
            for uav in uavs:
                if allowed_edges is not None and (task.task_id, uav.id) not in allowed_edges:
                    continue
                schedule = self._estimate_schedule(task, task_manager, uav, uavs, current_time_step)
                if schedule is not None:
                    candidates.append(schedule)
            if not candidates:
                continue
            teacher_scores = {
                schedule.uav_id: self._compute_teacher_score(
                    schedule,
                    task,
                    task_manager,
                    uavs[schedule.uav_id],
                    current_time_step,
                )
                for schedule in candidates
            }
            heuristic_best = min(candidates, key=lambda schedule: teacher_scores[schedule.uav_id])
            targets.append(
                TaskSupervisionTarget(
                    task_id=task.task_id,
                    feasible_uav_ids=[schedule.uav_id for schedule in candidates],
                    heuristic_best_uav=heuristic_best.uav_id,
                    heuristic_eft_by_uav={schedule.uav_id: float(schedule.planned_finish) for schedule in candidates},
                    heuristic_score_by_uav={uav_id: float(score) for uav_id, score in teacher_scores.items()},
                )
            )
        return targets

    def _compute_teacher_score(
        self,
        schedule: ScheduledTask,
        task: TaskNode,
        task_manager: DAGTaskManager,
        uav,
        current_time_step: float,
    ) -> float:
        if not config.USE_DAG_AWARE_TEACHER_SCORE:
            return float(schedule.planned_finish)

        max_compute = float(max(np.max(config.UAV_COMPUTING_CAPACITY), 1.0))
        compute_ratio = float(config.UAV_COMPUTING_CAPACITY[schedule.uav_id] / max_compute)
        remaining_energy_ratio = float(getattr(uav, "remaining_energy_ratio", 1.0))
        resource_quality = 0.7 * compute_ratio + 0.3 * remaining_energy_ratio

        immediate_successors = len(task.successors)
        descendant_count = task_manager.get_descendant_count(task.task_id)
        unlock_bonus = (
            config.DAG_TEACHER_SUCCESSOR_COMPUTE_BONUS
            * config.TIME_SLOT_DURATION
            * float(immediate_successors + 0.25 * descendant_count)
            * resource_quality
        )

        critical_bonus = 0.0
        if task_manager.is_critical_path_task(task.task_id):
            critical_bonus = (
                config.DAG_TEACHER_CRITICAL_COMPUTE_BONUS
                * config.TIME_SLOT_DURATION
                * resource_quality
            )

        same_parent_count = 0
        for parent_id in task.predecessors:
            parent_task = task_manager.tasks[parent_id]
            if parent_task.assigned_uav == schedule.uav_id:
                same_parent_count += 1
        parent_locality_bonus = (
            config.DAG_TEACHER_PARENT_LOCALITY_BONUS
            * config.TIME_SLOT_DURATION
            * float(same_parent_count)
        )

        dag_slack = task_manager.get_dag_remaining_slack(task.dag_id, current_time_step)
        urgency = max(0.0, config.DAG_CRITICAL_SLACK_THRESHOLD - dag_slack) / float(max(config.DAG_CRITICAL_SLACK_THRESHOLD, 1))
        completion_ratio = task_manager.get_dag_completion_ratio(task.dag_id)
        deadline_margin = max(float(task.deadline) - float(schedule.planned_finish), 0.0)
        margin_bonus = (
            config.DAG_TEACHER_URGENCY_WEIGHT
            * config.TIME_SLOT_DURATION
            * urgency
            * min(deadline_margin / float(max(config.DAG_CRITICAL_SLACK_THRESHOLD, 1)), 1.0)
        )
        completion_bonus = (
            config.DAG_TEACHER_COMPLETION_WEIGHT
            * config.TIME_SLOT_DURATION
            * completion_ratio
            * resource_quality
        )

        return float(
            schedule.planned_finish
            - unlock_bonus
            - critical_bonus
            - parent_locality_bonus
            - margin_bonus
            - completion_bonus
        )

    def _start_due_task(self, task_manager: DAGTaskManager, uav_id: int, slot_end: float) -> None:
        queue = self._queued[uav_id]
        while queue:
            planned_start, task_id = queue[0]
            task = task_manager.tasks.get(task_id)
            if task is None or task.state in {"finished", "dropped"}:
                heapq.heappop(queue)
                self._scheduled.pop(task_id, None)
                continue
            if planned_start > slot_end + config.EPSILON:
                return
            if task.state not in {"ready", "queued"}:
                heapq.heappop(queue)
                continue
            heapq.heappop(queue)
            task_manager.mark_task_running(task_id, planned_start)
            self._running[uav_id] = self._scheduled[task_id]
            return

    def _finish_running_task(
        self,
        task_manager: DAGTaskManager,
        running: ScheduledTask,
        uav,
        finish_time: float,
        stats: PhaseOneStepStats,
    ) -> None:
        task_manager.mark_task_finished(running.task_id, finish_time)
        task = task_manager.tasks[running.task_id]
        self._finished_count += 1
        stats.completed_tasks += 1
        stats.completed_tasks_by_uav[running.uav_id] = stats.completed_tasks_by_uav.get(running.uav_id, 0) + 1
        delay = max(0.0, running.planned_finish - task.arrival_time)
        stats.step_delay += delay
        if running.planned_finish <= task.deadline:
            self._on_time_finished_count += 1
            stats.on_time_completed_tasks += 1
            stats.on_time_completed_tasks_by_uav[running.uav_id] = (
                stats.on_time_completed_tasks_by_uav.get(running.uav_id, 0) + 1
            )
        else:
            self._deadline_violations += 1

    def _select_best_uav(
        self,
        task: TaskNode,
        task_manager: DAGTaskManager,
        uavs: list,
        current_time_step: float,
        allowed_edges: set[tuple[str, int]] | None = None,
        edge_scores: dict[tuple[str, int], float] | None = None,
    ) -> tuple[ScheduledTask | None, str | None, bool, AssignmentDecisionRecord | None]:
        candidates: list[tuple[ScheduledTask, float | None]] = []
        for uav in uavs:
            if allowed_edges is not None and (task.task_id, uav.id) not in allowed_edges:
                continue
            schedule = self._estimate_schedule(task, task_manager, uav, uavs, current_time_step)
            if schedule is None:
                continue
            edge_score = edge_scores.get((task.task_id, uav.id)) if edge_scores is not None else None
            candidates.append((schedule, edge_score))

        if not candidates:
            record = self._build_assignment_record(task, current_time_step, [], None, None, None, None, False)
            return None, None, False, record

        heuristic_best_schedule = min(candidates, key=lambda item: item[0].planned_finish)[0]
        scored_candidates = [(schedule, score) for schedule, score in candidates if score is not None]
        if scored_candidates and edge_scores is not None:
            raw_score_best_schedule = max(scored_candidates, key=lambda item: (float(item[1]), -item[0].planned_finish))[0]
            raw_disagrees = raw_score_best_schedule.uav_id != heuristic_best_schedule.uav_id
            fallback_reason = None
            if config.USE_SCORE_AGREEMENT_ONLY and raw_disagrees:
                fallback_reason = "agreement_only"
            score_best_schedule = raw_score_best_schedule
            if fallback_reason is None and config.USE_SCORE_RUNTIME_BOUNDED_GUARD:
                finish_limit = heuristic_best_schedule.planned_finish + float(config.SCORE_RUNTIME_FINISH_TOLERANCE)
                safe_scored_candidates = [
                    (schedule, score)
                    for schedule, score in scored_candidates
                    if schedule.planned_finish <= finish_limit
                ]
                if safe_scored_candidates:
                    score_best_schedule = max(
                        safe_scored_candidates,
                        key=lambda item: (float(item[1]), -item[0].planned_finish),
                    )[0]
                else:
                    fallback_reason = "runtime_bounded_guard"
            if fallback_reason is not None:
                record = self._build_assignment_record(
                    task,
                    current_time_step,
                    candidates,
                    heuristic_best_schedule,
                    raw_score_best_schedule,
                    heuristic_best_schedule,
                    "guard_fallback",
                    False,
                    raw_score_best_schedule,
                    raw_disagrees,
                    fallback_reason,
                )
                return heuristic_best_schedule, "guard_fallback", False, record
            disagrees = score_best_schedule.uav_id != heuristic_best_schedule.uav_id
            guard_reason = (
                "runtime_bounded_guard_clamp"
                if (
                    config.USE_SCORE_RUNTIME_BOUNDED_GUARD
                    and score_best_schedule.uav_id != raw_score_best_schedule.uav_id
                )
                else None
            )
            record = self._build_assignment_record(
                task,
                current_time_step,
                candidates,
                heuristic_best_schedule,
                score_best_schedule,
                score_best_schedule,
                "score",
                disagrees,
                raw_score_best_schedule,
                raw_disagrees,
                guard_reason,
            )
            return score_best_schedule, "score", disagrees, record

        if edge_scores is not None and not config.SCORE_FALLBACK_TO_HEURISTIC:
            record = self._build_assignment_record(
                task,
                current_time_step,
                candidates,
                heuristic_best_schedule,
                None,
                None,
                None,
                False,
            )
            return None, None, False, record
        record = self._build_assignment_record(
            task,
            current_time_step,
            candidates,
            heuristic_best_schedule,
            None,
            heuristic_best_schedule,
            "fallback",
            False,
        )
        return heuristic_best_schedule, "fallback", False, record

    def _build_assignment_record(
        self,
        task: TaskNode,
        current_time_step: float,
        candidates: list[tuple[ScheduledTask, float | None]],
        heuristic_best_schedule: ScheduledTask | None,
        score_best_schedule: ScheduledTask | None,
        selected_schedule: ScheduledTask | None,
        selection_mode: str | None,
        disagrees_with_heuristic: bool,
        raw_score_schedule: ScheduledTask | None = None,
        raw_disagrees_with_heuristic: bool = False,
        guard_reason: str | None = None,
    ) -> AssignmentDecisionRecord:
        candidate_records = [
            AssignmentCandidateRecord(
                uav_id=schedule.uav_id,
                planned_start=float(schedule.planned_start),
                planned_finish=float(schedule.planned_finish),
                transmission_time=float(schedule.transmission_time),
                execution_time=float(schedule.execution_time),
                total_energy=float(schedule.total_energy),
                score=None if score is None else float(score),
                queue_length=self.get_queue_length(schedule.uav_id),
                available_time=self.get_available_time(schedule.uav_id),
            )
            for schedule, score in candidates
        ]
        return AssignmentDecisionRecord(
            task_id=task.task_id,
            task_state=task.state,
            task_level=int(task.level),
            task_type=int(task.task_type),
            task_input_size=float(task.input_size),
            task_cpu_cycles=float(task.cpu_cycles),
            task_arrival_time=float(task.arrival_time),
            task_deadline=float(task.deadline),
            task_slack=float(task.deadline - current_time_step),
            num_predecessors=len(task.predecessors),
            num_successors=len(task.successors),
            heuristic_uav=None if heuristic_best_schedule is None else heuristic_best_schedule.uav_id,
            score_uav=None if score_best_schedule is None else score_best_schedule.uav_id,
            selected_uav=None if selected_schedule is None else selected_schedule.uav_id,
            selection_mode=selection_mode,
            disagrees_with_heuristic=disagrees_with_heuristic,
            candidates=candidate_records,
            raw_score_uav=None if raw_score_schedule is None else raw_score_schedule.uav_id,
            raw_disagrees_with_heuristic=raw_disagrees_with_heuristic,
            guard_reason=guard_reason,
        )

    def _estimate_schedule(
        self,
        task: TaskNode,
        task_manager: DAGTaskManager,
        uav,
        uavs: list,
        current_time_step: float,
    ) -> ScheduledTask | None:
        distance_to_source = float(np.linalg.norm(task.source_pos - uav.pos[:2]))
        if distance_to_source > config.DAG_TASK_UAV_MAX_DISTANCE:
            return None
        if self.get_queue_length(uav.id) >= config.DAG_MAX_QUEUE_PER_UAV:
            return None

        upload_rate = self._estimate_ue_uav_rate(task.source_pos, uav.pos)
        if upload_rate <= 0.0:
            return None
        upload_time = task.input_size / upload_rate

        predecessor_ready_time = float(current_time_step)
        predecessor_transfer_time = 0.0
        for parent_id in task.predecessors:
            parent_task = task_manager.tasks[parent_id]
            if parent_task.finish_time is None or parent_task.assigned_uav is None:
                return None
            parent_finish = float(parent_task.finish_time)
            predecessor_ready_time = max(predecessor_ready_time, parent_finish)
            if parent_task.assigned_uav != uav.id:
                parent_uav = uavs[parent_task.assigned_uav]
                transfer_time = self._estimate_uav_uav_transfer_time(parent_task.output_size, parent_uav.pos, uav.pos)
                if not np.isfinite(transfer_time):
                    return None
                predecessor_transfer_time = max(predecessor_transfer_time, transfer_time)

        compute_time = task.cpu_cycles / float(config.UAV_COMPUTING_CAPACITY[uav.id])
        earliest_start = max(float(current_time_step), self._uav_available_time[uav.id], predecessor_ready_time + predecessor_transfer_time, current_time_step + upload_time)
        planned_finish = earliest_start + compute_time
        if planned_finish > task.deadline + config.DAG_MAX_DEADLINE_TOLERANCE:
            return None

        compute_energy = config.K_CPU * task.cpu_cycles * (float(config.UAV_COMPUTING_CAPACITY[uav.id]) ** 2)
        tx_energy = config.TRANSMIT_POWER * (upload_time + predecessor_transfer_time)
        return ScheduledTask(
            task_id=task.task_id,
            uav_id=uav.id,
            planned_start=earliest_start,
            planned_finish=planned_finish,
            transmission_time=upload_time + predecessor_transfer_time,
            execution_time=compute_time,
            total_energy=compute_energy + tx_energy,
        )

    def _estimate_ue_uav_rate(self, source_pos: np.ndarray, uav_pos: np.ndarray) -> float:
        return comms.calculate_g2a_rate(source_pos, uav_pos, 1)

    def _estimate_uav_uav_transfer_time(self, data_size: float, pos_a: np.ndarray, pos_b: np.ndarray) -> float:
        return comms.calculate_a2a_transfer_time(data_size, pos_a, pos_b)
