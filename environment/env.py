from __future__ import annotations

from typing import Any

import config
from environment.dag_tasks import DAGTaskManager
from environment.task_execution import CleanTaskExecutor
from environment.user_equipments import UE
from environment.uavs import UAV
import numpy as np


class Env:
    """zrj_3 clean mainline environment.

    This hard-cleanup environment intentionally exposes only the clean reset,
    UE mobility, hotspot-based DAG arrival, and minimal observation loop. Legacy
    graph construction, task execution, score-head scheduling, and request/cache
    paths are no longer part of this main environment entrypoint.
    """

    def __init__(self) -> None:
        self._time_step: int = 0
        self.hotspot_center: np.ndarray | None = None
        self.hotspot_radius: float = float(config.HOTSPOT_RADIUS)
        self._ues: list[UE] = []
        self._uavs: list[UAV] = []
        self._task_manager: DAGTaskManager = DAGTaskManager()
        self._executor: CleanTaskExecutor = CleanTaskExecutor()
        self._latest_info: dict[str, Any] = {}

    @property
    def uavs(self) -> list[UAV]:
        return self._uavs

    @property
    def ues(self) -> list[UE]:
        return self._ues

    @property
    def task_manager(self) -> DAGTaskManager:
        return self._task_manager

    @property
    def executor(self) -> CleanTaskExecutor:
        return self._executor

    @property
    def time_step(self) -> int:
        return self._time_step

    @property
    def latest_info(self) -> dict[str, Any]:
        return dict(self._latest_info)

    def reset(self) -> list[np.ndarray]:
        self._time_step = 0
        self.hotspot_radius = float(config.HOTSPOT_RADIUS)
        self.hotspot_center = self._sample_episode_hotspot()
        self._uavs = self._init_uavs_uniform()
        self._ues = self._init_ues_uniform()
        self._task_manager.reset()
        self._executor.reset(self._uavs)
        self._latest_info = {
            "time_step": self._time_step,
            "created_dags": 0,
            "service_waiting_ues": 0,
        }
        return self._get_obs()

    def step(self, actions: dict[str, int] | None = None) -> tuple[list[np.ndarray], list[float], bool, dict[str, Any]]:
        self._time_step += 1
        for ue in self._ues:
            ue.update_position()

        created_dags = self._process_clean_dag_arrivals()
        self._task_manager.refresh_ready_states()
        assignments = actions if isinstance(actions, dict) else {}
        self._executor.assign_tasks(
            assignments=assignments,
            task_manager=self._task_manager,
            uavs=self._uavs,
            ues=self._ues,
            current_time_step=self._time_step,
        )
        execution_stats = self._executor.advance_one_slot(
            task_manager=self._task_manager,
            uavs=self._uavs,
            ues=self._ues,
            current_time_step=self._time_step,
        )
        for dag_id in execution_stats.completed_dag_ids:
            self.release_ue_after_dag_completed(dag_id)

        info = {
            "time_step": self._time_step,
            "created_dags": created_dags,
            "newly_assigned_tasks": execution_stats.newly_assigned_tasks,
            "invalid_assignments": execution_stats.invalid_assignments,
            "completed_tasks": execution_stats.completed_tasks,
            "completed_dags": execution_stats.completed_dags,
            "step_task_energy": execution_stats.step_task_energy,
            "step_compute_energy": execution_stats.step_compute_energy,
            "step_communication_energy": execution_stats.step_communication_energy,
            "step_return_energy": execution_stats.step_return_energy,
            "active_dags": sum(1 for job in self._task_manager.jobs.values() if not job.completed),
            "service_waiting_ues": sum(1 for ue in self._ues if ue.service_waiting),
            "hotspot_center": None if self.hotspot_center is None else self.hotspot_center.copy(),
            "hotspot_radius": self.hotspot_radius,
        }
        self._latest_info = info
        rewards = [0.0 for _ in self._uavs]
        done = self._time_step >= int(config.EPISODE_LENGTH)
        return self._get_obs(), rewards, done, dict(info)

    def _sample_episode_hotspot(self) -> np.ndarray:
        radius = float(config.HOTSPOT_RADIUS)
        width = float(config.AREA_WIDTH)
        height = float(config.AREA_HEIGHT)
        if radius <= 0.0:
            raise ValueError("HOTSPOT_RADIUS must be positive.")
        if 2.0 * radius > width or 2.0 * radius > height:
            raise ValueError("HOTSPOT_RADIUS is too large for the configured map.")
        return np.array(
            [
                np.random.uniform(radius, width - radius),
                np.random.uniform(radius, height - radius),
            ],
            dtype=np.float32,
        )

    def _init_uavs_uniform(self) -> list[UAV]:
        return [UAV(i) for i in range(config.NUM_UAVS)]

    def _init_ues_uniform(self) -> list[UE]:
        ues = [UE(i) for i in range(config.NUM_UES)]
        for ue in ues:
            ue.reset_episode_state()
        return ues

    def _process_clean_dag_arrivals(self) -> int:
        created_count = 0
        for ue in self._ues:
            if ue.active_dag_id is not None:
                continue
            if self._task_manager.get_active_job_for_ue(ue.id) is not None:
                continue
            arrival_prob = ue.get_arrival_probability(self.hotspot_center, self.hotspot_radius)
            if np.random.random() >= arrival_prob:
                continue
            job = self._task_manager.create_dag_for_ue(
                ue_id=ue.id,
                source_pos=ue.pos[:2].copy(),
                current_time_step=self._time_step,
            )
            ue.enter_service_waiting(job.dag_id)
            created_count += 1
        return created_count

    def release_ue_after_dag_completed(self, dag_id: str) -> None:
        for ue in self._ues:
            if ue.active_dag_id == dag_id:
                ue.release_service_waiting(dag_id)
                return

    def _get_obs(self) -> list[np.ndarray]:
        hotspot = (
            np.zeros((2,), dtype=np.float32)
            if self.hotspot_center is None
            else self.hotspot_center.astype(np.float32, copy=False)
        )
        hotspot_norm = hotspot / np.array([config.AREA_WIDTH, config.AREA_HEIGHT], dtype=np.float32)
        obs: list[np.ndarray] = []
        for uav in self._uavs:
            pos_norm = np.clip(
                uav.pos[:2] / np.array([config.AREA_WIDTH, config.AREA_HEIGHT], dtype=np.float32),
                0.0,
                1.0,
            )
            obs.append(np.concatenate([pos_norm, hotspot_norm]).astype(np.float32, copy=False))
        return obs
