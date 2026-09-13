from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

import config
from environment.dag_tasks import TASK_STATE_COMPLETED


@dataclass(slots=True)
class ForecastDelta:
    total: float
    target: float
    cross: float
    affected_dags: int
    before_phi: float
    after_phi: float
    wall_seconds: float


@dataclass(slots=True)
class _DAGStaticArrays:
    job: Any
    dag_id: str
    task_ids: tuple[str, ...]
    predecessors: tuple[np.ndarray, ...]
    sinks: np.ndarray
    compute_seconds: np.ndarray


@dataclass(slots=True)
class _DAGArrays:
    static: _DAGStaticArrays
    entry_upload_seconds: np.ndarray
    edge_transfer_seconds: dict[tuple[int, int], np.ndarray]
    fixed_uav: np.ndarray
    fixed_compute_finish: np.ndarray

    @property
    def dag_id(self) -> str:
        return self.static.dag_id

    @property
    def task_ids(self) -> tuple[str, ...]:
        return self.static.task_ids

    @property
    def predecessors(self) -> tuple[np.ndarray, ...]:
        return self.static.predecessors

    @property
    def sinks(self) -> np.ndarray:
        return self.static.sinks

    @property
    def compute_seconds(self) -> np.ndarray:
        return self.static.compute_seconds


class ForecastContext:
    """Compute-only optimistic DAG forecast for one frozen decision slot.

    Topology and all communication matrices are built once.  Reforecasting
    after a reservation performs only NumPy small-matrix DP; it never calls the
    communication model or mutates the executor/environment.
    """

    def __init__(
        self,
        *,
        task_manager: Any,
        executor: Any,
        uavs: list[Any],
        ues: list[Any],
        current_time_seconds: float,
        uav_service_positions: dict[int, Any] | None = None,
        ue_service_positions: dict[int, Any] | None = None,
    ) -> None:
        self.task_manager = task_manager
        self.executor = executor
        self.current_time_seconds = float(current_time_seconds)
        ordered_uavs = sorted(uavs, key=lambda item: int(item.id))
        self.uav_ids = np.asarray([int(item.id) for item in ordered_uavs], dtype=np.int64)
        self._uav_index = {int(value): idx for idx, value in enumerate(self.uav_ids.tolist())}
        self._uav_positions = np.stack(
            [
                np.asarray(
                    (uav_service_positions or {}).get(int(uav.id), uav.pos),
                    dtype=np.float64,
                ).reshape(-1)[:2]
                for uav in ordered_uavs
            ],
            axis=0,
        )
        self._ue_positions = {
            int(ue.id): np.asarray(
                (ue_service_positions or {}).get(int(ue.id), ue.pos), dtype=np.float64
            ).reshape(-1)[:2]
            for ue in ues
        }
        # The shadow queue is initialized from actual scheduled compute plans,
        # deliberately excluding sink return time.
        self.queue_clock = np.full(len(ordered_uavs), self.current_time_seconds, dtype=np.float64)
        for record in getattr(executor, "task_records", {}).values():
            if bool(getattr(record, "completed", False)):
                continue
            uidx = self._uav_index.get(int(record.uav_id))
            if uidx is not None:
                self.queue_clock[uidx] = max(
                    self.queue_clock[uidx], float(record.compute_finish_time)
                )
        self._dags = self._build_arrays()
        self.last_times = self.forecast_times()

    def reserve_compute(self, uav_id: int, compute_finish_time: float) -> None:
        idx = self._uav_index[int(uav_id)]
        self.queue_clock[idx] = max(self.queue_clock[idx], float(compute_finish_time))

    def delta_for_reservation(
        self,
        *,
        dag_id: str,
        uav_id: int,
        compute_finish_time: float,
    ) -> ForecastDelta:
        started = perf_counter()
        before = dict(self.last_times)
        before_phi = float(sum(before.values()))
        self.reserve_compute(int(uav_id), float(compute_finish_time))
        after = self.forecast_times()
        after_phi = float(sum(after.values()))
        changes = {key: float(after[key] - before[key]) for key in before}
        target = float(changes.get(str(dag_id), 0.0))
        total = float(after_phi - before_phi)
        self.last_times = after
        return ForecastDelta(
            total=total,
            target=target,
            cross=float(total - target),
            affected_dags=sum(abs(value) > 1e-9 for value in changes.values()),
            before_phi=before_phi,
            after_phi=after_phi,
            wall_seconds=float(perf_counter() - started),
        )

    def forecast_times(self) -> dict[str, float]:
        return {dag.dag_id: self._forecast_dag(dag) for dag in self._dags}

    def _forecast_dag(self, dag: _DAGArrays) -> float:
        task_count = len(dag.task_ids)
        uav_count = len(self.uav_ids)
        finish = np.full((task_count, uav_count), np.inf, dtype=np.float64)
        local_queue = self.queue_clock.copy()
        for tidx in range(task_count):
            fixed_idx = int(dag.fixed_uav[tidx])
            fixed_finish = float(dag.fixed_compute_finish[tidx])
            if np.isfinite(fixed_finish):
                if fixed_idx >= 0:
                    finish[tidx, fixed_idx] = fixed_finish
                else:
                    finish[tidx, :] = fixed_finish
                continue

            parents = dag.predecessors[tidx]
            ready = np.full(uav_count, self.current_time_seconds, dtype=np.float64)
            if parents.size:
                for pidx in parents.tolist():
                    transfer = dag.edge_transfer_seconds[(int(pidx), tidx)]
                    # One vectorized UxU parent-location -> child-location minimum.
                    parent_arrival = np.min(finish[int(pidx), :, None] + transfer, axis=0)
                    ready = np.maximum(ready, parent_arrival)
            else:
                ready = np.maximum(
                    ready, self.current_time_seconds + dag.entry_upload_seconds[tidx, :]
                )
            candidates = np.maximum(ready, local_queue) + dag.compute_seconds[tidx]
            finish[tidx, :] = candidates
            chosen = int(np.argmin(candidates))
            local_queue[chosen] = candidates[chosen]

        sink_finish = finish[dag.sinks, :]
        return float(np.max(np.min(sink_finish, axis=1)))

    def _build_arrays(self) -> tuple[_DAGArrays, ...]:
        output: list[_DAGArrays] = []
        uav_count = len(self.uav_ids)
        static_cache: dict[str, _DAGStaticArrays] = getattr(
            self.task_manager, "_forecast_static_topology_cache", {}
        )
        if not hasattr(self.task_manager, "_forecast_static_topology_cache"):
            setattr(self.task_manager, "_forecast_static_topology_cache", static_cache)
        distances = np.linalg.norm(
            self._uav_positions[:, None, :] - self._uav_positions[None, :, :], axis=2
        )
        for job in self.task_manager.jobs.values():
            if bool(job.completed):
                continue
            static = static_cache.get(str(job.dag_id))
            if static is None or static.job is not job:
                tasks = sorted(
                    self.task_manager.get_job_tasks(job.dag_id),
                    key=lambda item: (int(item.topological_index), str(item.task_id)),
                )
                index = {str(task.task_id): idx for idx, task in enumerate(tasks)}
                static = _DAGStaticArrays(
                    job=job,
                    dag_id=str(job.dag_id),
                    task_ids=tuple(str(task.task_id) for task in tasks),
                    predecessors=tuple(
                        np.asarray(
                            [index[str(pid)] for pid in task.predecessors], dtype=np.int64
                        )
                        for task in tasks
                    ),
                    sinks=np.asarray(
                        [index[str(task_id)] for task_id in job.sink_task_ids],
                        dtype=np.int64,
                    ),
                    compute_seconds=np.asarray(
                        [
                            float(task.num_operation)
                            / float(config.UAV_COMPUTE_RATE_OPS_PER_SEC)
                            for task in tasks
                        ],
                        dtype=np.float64,
                    ),
                )
                static_cache[str(job.dag_id)] = static
            tasks = [self.task_manager.get_task(task_id) for task_id in static.task_ids]
            entry = np.zeros((len(tasks), uav_count), dtype=np.float64)
            ue_pos = self._ue_positions.get(int(job.ue_id), np.asarray(job.source_pos, dtype=np.float64)[:2])
            ue_distances = np.linalg.norm(self._uav_positions - ue_pos[None, :], axis=1)
            for tidx, task in enumerate(tasks):
                assert task is not None
                if not task.predecessors:
                    entry[tidx, :] = self._tx_vector(
                        float(task.input_data_size_mb), float(job.base_upload_bandwidth_mbps), ue_distances
                    )
            edge_transfer: dict[tuple[int, int], np.ndarray] = {}
            for child_idx, task in enumerate(tasks):
                assert task is not None
                for parent_idx in static.predecessors[child_idx].tolist():
                    parent_idx = int(parent_idx)
                    parent = tasks[parent_idx]
                    assert parent is not None
                    edge_transfer[(parent_idx, child_idx)] = self._tx_vector(
                        float(parent.output_data_size_mb),
                        float(job.base_upload_bandwidth_mbps),
                        distances,
                    )
            fixed_uav = np.full(len(tasks), -1, dtype=np.int64)
            fixed_finish = np.full(len(tasks), np.nan, dtype=np.float64)
            for tidx, task in enumerate(tasks):
                assert task is not None
                record = getattr(self.executor, "task_records", {}).get(str(task.task_id))
                if record is not None:
                    fixed_uav[tidx] = self._uav_index.get(int(record.uav_id), -1)
                    fixed_finish[tidx] = float(record.compute_finish_time)
                elif task.state == TASK_STATE_COMPLETED and task.compute_finish_time is not None:
                    fixed_uav[tidx] = self._uav_index.get(int(task.assigned_uav), -1)
                    fixed_finish[tidx] = float(task.compute_finish_time)
            output.append(
                _DAGArrays(
                    static=static,
                    entry_upload_seconds=entry,
                    edge_transfer_seconds=edge_transfer,
                    fixed_uav=fixed_uav,
                    fixed_compute_finish=fixed_finish,
                )
            )
        return tuple(output)

    @staticmethod
    def _tx_vector(data_mb: float, bandwidth_mbps: float, distance: np.ndarray) -> np.ndarray:
        distances = np.asarray(distance, dtype=np.float64)
        return (
            float(data_mb)
            * 8.0
            * (1.0 + np.square(distances / 100.0))
            / float(bandwidth_mbps)
        )
