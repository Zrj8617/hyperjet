from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import config
from environment import comm_model
from environment.dag_tasks import DAGTaskManager, TASK_STATE_READY, TaskNode


@dataclass(slots=True)
class CleanScheduledTask:
    task_id: str
    uav_id: int
    assignment_time: float
    start_time: float
    finish_time: float
    upload_time: float
    inter_transfer_time: float
    compute_time: float
    return_time: float
    communication_energy: float
    compute_energy: float
    return_energy: float
    total_energy: float
    inter_transfer_energy: float = 0.0
    compute_finish_time: float = 0.0
    return_started: bool = False
    completed: bool = False


@dataclass(slots=True)
class CleanExecutionStepStats:
    newly_assigned_tasks: int = 0
    invalid_assignments: int = 0
    completed_tasks: int = 0
    completed_dags: int = 0
    step_task_energy: float = 0.0
    step_compute_energy: float = 0.0
    step_communication_energy: float = 0.0
    step_return_energy: float = 0.0
    completed_dag_ids: list[str] = field(default_factory=list)


class CleanTaskExecutor:
    """Strict clean-mainline task executor.

    The executor schedules exactly the task-UAV pairs it is given and does not
    substitute a different UAV.
    """

    def __init__(self) -> None:
        self.uav_available_time: dict[int, float] = {}
        self.uav_queues: dict[int, list[str]] = {}
        self.task_records: dict[str, CleanScheduledTask] = {}
        self.latest_stats: CleanExecutionStepStats = CleanExecutionStepStats()

    def reset(self, uavs: list[Any]) -> None:
        self.uav_available_time = {int(uav.id): 0.0 for uav in uavs}
        self.uav_queues = {int(uav.id): [] for uav in uavs}
        self.task_records.clear()
        self.latest_stats = CleanExecutionStepStats()

    def assign_tasks(
        self,
        assignments: dict[str, int] | None,
        task_manager: DAGTaskManager,
        uavs: list[Any],
        ues: list[Any],
        current_time_step: float,
    ) -> CleanExecutionStepStats:
        self.latest_stats = CleanExecutionStepStats()
        if not assignments:
            return self.latest_stats

        uav_map = {int(uav.id): uav for uav in uavs}
        for task_id, raw_uav_id in assignments.items():
            try:
                uav_id = int(raw_uav_id)
            except (TypeError, ValueError):
                self.latest_stats.invalid_assignments += 1
                continue
            task = task_manager.get_task(str(task_id))
            if task is None or uav_id not in uav_map:
                self.latest_stats.invalid_assignments += 1
                continue
            if task.state != TASK_STATE_READY or task.task_id in self.task_records:
                self.latest_stats.invalid_assignments += 1
                continue
            if len(self.uav_queues.setdefault(uav_id, [])) >= int(config.CLEAN_MAX_QUEUE_PER_UAV):
                self.latest_stats.invalid_assignments += 1
                continue

            record = self._build_schedule_record(
                task=task,
                task_manager=task_manager,
                uav=uav_map[uav_id],
                uav_map=uav_map,
                ues=ues,
                assignment_time=float(current_time_step),
            )
            if record is None:
                self.latest_stats.invalid_assignments += 1
                continue

            task_manager.mark_task_queued(task.task_id, uav_id, current_time_step)
            task_manager.mark_task_running(task.task_id, record.start_time)
            self.task_records[task.task_id] = record
            self.uav_queues[uav_id].append(task.task_id)
            self.uav_available_time[uav_id] = record.finish_time
            self.latest_stats.newly_assigned_tasks += 1
        return self.latest_stats

    def advance_one_slot(
        self,
        task_manager: DAGTaskManager,
        uavs: list[Any],
        ues: list[Any],
        current_time_step: float,
    ) -> CleanExecutionStepStats:
        step_end = float(current_time_step) + float(config.TIME_SLOT_DURATION)
        uav_map = {int(uav.id): uav for uav in uavs}

        for record in list(self.task_records.values()):
            if record.completed:
                continue
            task = task_manager.get_task(record.task_id)
            if task is None:
                continue
            job = task_manager.get_job(task.dag_id)
            if job is None:
                continue

            if task.task_id in set(job.sink_task_ids):
                self._maybe_start_return(record, task, job, uav_map, ues, step_end)
                if record.return_started and record.finish_time <= step_end:
                    self._complete_record(record, task, task_manager)
                continue

            if record.compute_finish_time <= step_end:
                self._complete_record(record, task, task_manager)

        return self.latest_stats

    def _build_schedule_record(
        self,
        task: TaskNode,
        task_manager: DAGTaskManager,
        uav: Any,
        uav_map: dict[int, Any],
        ues: list[Any],
        assignment_time: float,
    ) -> CleanScheduledTask | None:
        job = task_manager.get_job(task.dag_id)
        if job is None:
            return None

        upload_time = 0.0
        inter_transfer_time = 0.0
        communication_energy = 0.0
        inter_transfer_energy = 0.0
        predecessor_ready_time = assignment_time

        if not task.predecessors:
            distance = comm_model.clean_distance_2d(job.source_pos, getattr(uav, "pos"))
            upload_time = comm_model.clean_transmission_time_seconds(
                task.input_data_size_mb,
                job.base_upload_bandwidth_mbps,
                distance,
            )
            communication_energy += upload_time * float(config.P_UE_TX)
        else:
            parent_finish_times: list[float] = []
            for parent_id in task.predecessors:
                parent = task_manager.get_task(parent_id)
                if parent is None or parent.finish_time is None or parent.assigned_uav is None:
                    return None
                if parent.state == TASK_STATE_READY:
                    return None
                parent_finish_times.append(float(parent.finish_time))
                if int(parent.assigned_uav) == int(uav.id):
                    continue
                parent_uav = uav_map.get(int(parent.assigned_uav))
                if parent_uav is None:
                    return None
                distance = comm_model.clean_distance_2d(getattr(parent_uav, "pos"), getattr(uav, "pos"))
                transfer_time = comm_model.clean_transmission_time_seconds(
                    parent.output_data_size_mb,
                    job.base_upload_bandwidth_mbps,
                    distance,
                )
                inter_transfer_time += transfer_time
                inter_transfer_energy += transfer_time * float(config.P_UAV_TX)
            predecessor_ready_time = max(parent_finish_times) if parent_finish_times else assignment_time
            communication_energy += inter_transfer_energy

        transfer_ready_time = (
            max(
                assignment_time,
                float(self.uav_available_time.get(int(uav.id), 0.0)),
                predecessor_ready_time,
            )
            + upload_time
            + inter_transfer_time
        )
        compute_time = float(task.num_operation) / float(config.UAV_COMPUTE_RATE_OPS_PER_SEC)
        compute_finish_time = transfer_ready_time + compute_time
        compute_energy = compute_time * float(config.P_UAV_COMPUTE)

        is_sink = task.task_id in set(job.sink_task_ids)
        finish_time = compute_finish_time if not is_sink else compute_finish_time
        return_started = False
        record = CleanScheduledTask(
            task_id=task.task_id,
            uav_id=int(uav.id),
            assignment_time=assignment_time,
            start_time=transfer_ready_time,
            finish_time=finish_time,
            upload_time=upload_time,
            inter_transfer_time=inter_transfer_time,
            compute_time=compute_time,
            return_time=0.0,
            communication_energy=communication_energy,
            compute_energy=compute_energy,
            return_energy=0.0,
            total_energy=communication_energy + compute_energy,
            inter_transfer_energy=inter_transfer_energy,
            compute_finish_time=compute_finish_time,
            return_started=return_started,
        )
        if not is_sink:
            self.uav_available_time[int(uav.id)] = finish_time
        return record

    def _maybe_start_return(
        self,
        record: CleanScheduledTask,
        task: TaskNode,
        job: Any,
        uav_map: dict[int, Any],
        ues: list[Any],
        step_end: float,
    ) -> None:
        if record.return_started or record.compute_finish_time > step_end:
            return
        uav = uav_map.get(record.uav_id)
        if uav is None:
            return
        ue = next((item for item in ues if int(item.id) == int(job.ue_id)), None)
        if ue is None:
            return
        distance = comm_model.clean_distance_2d(getattr(uav, "pos"), getattr(ue, "pos"))
        record.return_time = comm_model.clean_transmission_time_seconds(
            task.output_data_size_mb,
            job.base_download_bandwidth_mbps,
            distance,
        )
        record.return_energy = record.return_time * float(config.P_UAV_TX)
        record.total_energy = record.communication_energy + record.compute_energy + record.return_energy
        record.finish_time = record.compute_finish_time + record.return_time
        record.return_started = True
        self.uav_available_time[record.uav_id] = max(
            float(self.uav_available_time.get(record.uav_id, 0.0)),
            record.finish_time,
        )

    def _complete_record(
        self,
        record: CleanScheduledTask,
        task: TaskNode,
        task_manager: DAGTaskManager,
    ) -> None:
        if record.completed:
            return
        task.assigned_uav = record.uav_id
        task.compute_energy = record.compute_energy
        task.communication_energy = record.communication_energy
        task.return_energy = record.return_energy
        task.total_energy = record.total_energy
        job = task_manager.get_job(task.dag_id)
        was_completed = bool(job is not None and job.completed)
        if record.return_started:
            task_manager.mark_task_returned(task.task_id, record.finish_time)
        else:
            task_manager.mark_task_finished(task.task_id, record.compute_finish_time)
        record.completed = True
        queue = self.uav_queues.get(record.uav_id, [])
        if task.task_id in queue:
            queue.remove(task.task_id)

        self.latest_stats.completed_tasks += 1
        self.latest_stats.step_task_energy += record.total_energy
        self.latest_stats.step_compute_energy += record.compute_energy
        self.latest_stats.step_communication_energy += record.communication_energy
        self.latest_stats.step_return_energy += record.return_energy

        job = task_manager.get_job(task.dag_id)
        became_completed = bool(job is not None and job.completed and not was_completed)
        if became_completed or task_manager.mark_dag_completed_if_ready(task.dag_id, task.finish_time):
            self.latest_stats.completed_dags += 1
            self.latest_stats.completed_dag_ids.append(task.dag_id)


# Compatibility alias for old imports. Clean Env uses CleanTaskExecutor directly.
PhaseOneTaskExecutor = CleanTaskExecutor
