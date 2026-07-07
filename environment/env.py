from __future__ import annotations

from typing import Any

import config
from environment.assignment import (
    CleanAssignmentBuffer,
    TemporaryReservationState,
    freeze_ready_tasks,
    is_assignment_legal,
    legal_candidate_uav_ids,
)
from environment.dag_tasks import DAGTaskManager
from environment.metrics import CleanMetricsTracker
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
        self._metrics: CleanMetricsTracker = CleanMetricsTracker()
        self._latest_info: dict[str, Any] = {}
        self._ue_service_positions: dict[int, np.ndarray] = {}
        self._uav_pre_move_positions: dict[int, np.ndarray] = {}
        self._uav_service_positions: dict[int, np.ndarray] = {}
        self._latest_slot_boundary: dict[str, Any] = {}
        self._slot_service_positions_frozen: bool = False
        self._frozen_ready_task_ids: list[str] = []
        self._last_assignment_buffer: CleanAssignmentBuffer = CleanAssignmentBuffer()
        self._last_new_dag_arrived: bool = False
        self._latest_dag_arrival_version: int = 0
        self._last_movement_energy_by_uav: dict[int, float] = {}
        self._last_movement_energy_total: float = 0.0

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
    def metrics(self) -> CleanMetricsTracker:
        return self._metrics

    @property
    def time_step(self) -> int:
        return self._time_step

    @property
    def latest_info(self) -> dict[str, Any]:
        return dict(self._latest_info)

    @property
    def ue_service_positions(self) -> dict[int, np.ndarray]:
        return {ue_id: pos.copy() for ue_id, pos in self._ue_service_positions.items()}

    @property
    def uav_service_positions(self) -> dict[int, np.ndarray]:
        return {uav_id: pos.copy() for uav_id, pos in self._uav_service_positions.items()}

    @property
    def latest_slot_boundary(self) -> dict[str, Any]:
        return dict(self._latest_slot_boundary)

    @property
    def frozen_ready_task_ids(self) -> list[str]:
        return list(self._frozen_ready_task_ids)

    @property
    def last_assignment_buffer(self) -> CleanAssignmentBuffer:
        return CleanAssignmentBuffer(entries=list(self._last_assignment_buffer.entries))

    @property
    def new_dag_arrived(self) -> bool:
        return bool(self._last_new_dag_arrived)

    @property
    def dag_arrival_version(self) -> int:
        return int(self._latest_dag_arrival_version)

    def reset(self) -> list[np.ndarray]:
        self._time_step = 0
        self.hotspot_radius = float(config.HOTSPOT_RADIUS)
        self.hotspot_center = self._sample_episode_hotspot()
        self._uavs = self._init_uavs_uniform()
        self._ues = self._init_ues_uniform()
        self._task_manager.reset()
        self._executor.reset(self._uavs)
        self._metrics.reset([uav.id for uav in self._uavs])
        self._ue_service_positions = {int(ue.id): ue.pos[:2].copy() for ue in self._ues}
        self._uav_pre_move_positions = {int(uav.id): uav.pos[:2].copy() for uav in self._uavs}
        self._uav_service_positions = {int(uav.id): uav.pos[:2].copy() for uav in self._uavs}
        self._slot_service_positions_frozen = False
        self._frozen_ready_task_ids = []
        self._last_assignment_buffer = CleanAssignmentBuffer()
        self._last_new_dag_arrived = False
        self._latest_dag_arrival_version = self._task_manager.dag_arrival_version
        self._last_movement_energy_by_uav = {int(uav.id): 0.0 for uav in self._uavs}
        self._last_movement_energy_total = 0.0
        self._latest_slot_boundary = {
            "internal_state": "x_0^-",
            "decision_state": None,
            "post_execution_state": "x_0^-",
        }
        self._latest_info = {
            "time_step": self._time_step,
            "created_dags": 0,
            "service_waiting_ues": 0,
            "step_reward": 0.0,
            "episode_reward": 0.0,
        }
        return self._get_obs()

    def step(self, actions: Any | None = None) -> tuple[list[np.ndarray], list[float], bool, dict[str, Any]]:
        # Slot starts from the previous post-execution internal state x_t^-.
        slot_index = self._time_step
        previous_internal_state = f"x_{slot_index}^-"
        self._time_step += 1
        for ue in self._ues:
            ue.update_position()
        self._ue_service_positions = {int(ue.id): ue.pos[:2].copy() for ue in self._ues}
        self._slot_service_positions_frozen = True

        created_dags = self._process_clean_dag_arrivals()
        self._task_manager.refresh_ready_states()
        # After UE movement, DAG arrivals, and ready refresh, the clean decision
        # state s_t is formed. Graph/HGNN construction is deliberately left to T4.
        decision_state = f"s_{slot_index}"
        parsed_actions = self._parse_clean_actions(actions)
        assignments = parsed_actions["assignments"]
        self._apply_clean_movement(parsed_actions["movement_actions"])
        ready_tasks = freeze_ready_tasks(self._task_manager)
        self._frozen_ready_task_ids = [task.task_id for task in ready_tasks]
        assignment_buffer, offloading_skip_count = self._build_assignment_buffer(ready_tasks, assignments)
        self._last_assignment_buffer = assignment_buffer
        self._executor.assign_tasks(
            assignments=assignment_buffer,
            task_manager=self._task_manager,
            uavs=self._uavs,
            ues=self._ues,
            current_time_step=self._time_step,
            uav_service_positions=self._uav_service_positions,
            ue_service_positions=self._ue_service_positions,
        )
        execution_stats = self._executor.advance_one_slot(
            task_manager=self._task_manager,
            uavs=self._uavs,
            ues=self._ues,
            current_time_step=self._time_step,
            uav_service_positions=self._uav_service_positions,
            ue_service_positions=self._ue_service_positions,
        )
        for dag_id in execution_stats.completed_dag_ids:
            self.release_ue_after_dag_completed(dag_id)

        post_execution_state = f"x_{slot_index + 1}^-"
        self._latest_slot_boundary = {
            "slot": slot_index,
            "started_from": previous_internal_state,
            "decision_state": decision_state,
            "post_execution_state": post_execution_state,
            "next_decision_state": None,
        }
        step_reward = self._metrics.calculate_step_reward(
            self._task_manager,
            execution_stats,
            movement_energy_slot=self._last_movement_energy_total,
        )
        queue_lengths = [len(self._executor.uav_queues.get(int(uav.id), [])) for uav in self._uavs]
        self._metrics.update(
            task_manager=self._task_manager,
            execution_stats=execution_stats,
            step_reward=step_reward,
            created_dags=created_dags,
            action_count=assignment_buffer.entry_count,
            queue_lengths=queue_lengths,
            elapsed_steps=self._time_step,
            movement_energy_by_uav=self._last_movement_energy_by_uav,
        )
        metric_info = self._metrics.to_info(self._time_step)
        metric_info["generated_dag_count"] = float(len(self._task_manager.jobs))
        info = {
            "time_step": self._time_step,
            "created_dags": created_dags,
            "newly_assigned_tasks": execution_stats.newly_assigned_tasks,
            "invalid_assignments": execution_stats.invalid_assignments,
            "completed_tasks": execution_stats.completed_tasks,
            "completed_dags": execution_stats.completed_dags,
            "step_reward": step_reward.reward_total,
            "step_time_penalty": step_reward.time_penalty,
            "step_energy_penalty": step_reward.energy_penalty,
            "step_task_energy_penalty": step_reward.task_energy_penalty,
            "step_movement_energy_penalty": step_reward.movement_energy_penalty,
            "step_completed_dag_bonus": step_reward.completed_dag_bonus,
            "step_task_energy": execution_stats.step_task_energy,
            "step_movement_energy": self._last_movement_energy_total,
            "step_compute_energy": execution_stats.step_compute_energy,
            "step_communication_energy": execution_stats.step_communication_energy,
            "step_return_energy": execution_stats.step_return_energy,
            "active_dags": sum(1 for job in self._task_manager.jobs.values() if not job.completed),
            "service_waiting_ues": sum(1 for ue in self._ues if ue.service_waiting),
            "hotspot_center": None if self.hotspot_center is None else self.hotspot_center.copy(),
            "hotspot_radius": self.hotspot_radius,
            "slot_started_from": previous_internal_state,
            "slot_decision_state": decision_state,
            "slot_post_execution_state": post_execution_state,
            "frozen_ready_task_count": len(self._frozen_ready_task_ids),
            "assignment_buffer_entry_count": assignment_buffer.entry_count,
            "offloading_skipped_no_candidate": offloading_skip_count,
        }
        info.update(metric_info)
        self._latest_info = info
        # Use a global clean reward averaged across UAV agents to keep scale stable.
        reward_per_uav = step_reward.reward_total / float(max(len(self._uavs), 1))
        rewards = [reward_per_uav for _ in self._uavs]
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
        if not self._slot_service_positions_frozen:
            self._ue_service_positions = {int(ue.id): ue.pos[:2].copy() for ue in self._ues}
        version_before = self._task_manager.dag_arrival_version
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
                source_pos=self._ue_service_positions.get(int(ue.id), ue.pos[:2]).copy(),
                current_time_step=self._time_step,
            )
            ue.enter_service_waiting(job.dag_id)
            created_count += 1
        version_after = self._task_manager.dag_arrival_version
        self._last_new_dag_arrived = version_after > version_before or created_count > 0
        self._latest_dag_arrival_version = version_after
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

    def _parse_clean_actions(self, actions: Any | None) -> dict[str, Any]:
        if not isinstance(actions, dict):
            return {"assignments": {}, "movement_actions": {}}
        if "assignments" in actions or "offloading_assignments" in actions or "movement_actions" in actions:
            assignments = actions.get("assignments", actions.get("offloading_assignments", {}))
            movement_actions = actions.get("movement_actions", {})
            return {
                "assignments": assignments if isinstance(assignments, dict) else {},
                "movement_actions": movement_actions if isinstance(movement_actions, dict) else {},
            }
        return {"assignments": actions, "movement_actions": {}}

    def _build_assignment_buffer(
        self,
        ready_tasks: list[Any],
        requested_assignments: dict[str, int],
    ) -> tuple[CleanAssignmentBuffer, int]:
        reservation = TemporaryReservationState.from_executor(self._uavs, self._executor)
        valid_uav_ids = {int(uav.id) for uav in self._uavs}
        ordered_uav_ids = sorted(valid_uav_ids)
        buffer = CleanAssignmentBuffer()
        skipped_no_candidate = 0
        for decision_order, task in enumerate(ready_tasks):
            legal_candidates = legal_candidate_uav_ids(
                task=task,
                uav_ids=ordered_uav_ids,
                state_view=reservation,
                executor=self._executor,
                service_positions=self._uav_service_positions,
            )
            if not legal_candidates:
                skipped_no_candidate += 1
                continue
            if task.task_id not in requested_assignments:
                continue
            try:
                selected_uav_id = int(requested_assignments[task.task_id])
            except (TypeError, ValueError):
                continue
            if not is_assignment_legal(
                task=task,
                uav_id=selected_uav_id,
                state_view=reservation,
                valid_uav_ids=valid_uav_ids,
                executor=self._executor,
                service_positions=self._uav_service_positions,
            ):
                continue
            buffer.append(task.task_id, selected_uav_id, decision_order)
            reservation.reserve(task.task_id, selected_uav_id)
        return buffer, skipped_no_candidate

    def _apply_clean_movement(self, movement_actions: dict[int, int | str]) -> None:
        self._uav_pre_move_positions = {int(uav.id): uav.pos[:2].copy() for uav in self._uavs}
        self._uav_service_positions = {}
        self._last_movement_energy_by_uav = {}
        self._last_movement_energy_total = 0.0
        for uav in self._uavs:
            uav_id = int(uav.id)
            pre_pos = self._uav_pre_move_positions[uav_id]
            action = movement_actions.get(uav_id, movement_actions.get(str(uav_id), "hover"))
            delta = self._movement_delta(action)
            candidate = pre_pos + delta
            if not self._inside_map(candidate):
                candidate = pre_pos.copy()
            self._uav_service_positions[uav_id] = candidate.astype(np.float32, copy=True)
            move_distance = float(np.linalg.norm(candidate - pre_pos))
            move_energy = float(config.CLEAN_POWER_MOVE) * float(config.TIME_SLOT_DURATION) if move_distance > 0.0 else 0.0
            self._last_movement_energy_by_uav[uav_id] = move_energy
            self._last_movement_energy_total += move_energy
            uav.update_position(candidate)

    def _movement_delta(self, action: int | str) -> np.ndarray:
        step_distance = float(config.CLEAN_UAV_MOVEMENT_SPEED) * float(config.TIME_SLOT_DURATION)
        if isinstance(action, str):
            action_name = action
        else:
            action_names = tuple(getattr(config, "CLEAN_MOVEMENT_ACTIONS", ("hover", "+x", "-x", "+y", "-y")))
            action_name = action_names[int(action)] if 0 <= int(action) < len(action_names) else "hover"
        if action_name == "+x":
            return np.array([step_distance, 0.0], dtype=np.float32)
        if action_name == "-x":
            return np.array([-step_distance, 0.0], dtype=np.float32)
        if action_name == "+y":
            return np.array([0.0, step_distance], dtype=np.float32)
        if action_name == "-y":
            return np.array([0.0, -step_distance], dtype=np.float32)
        return np.zeros((2,), dtype=np.float32)

    def _inside_map(self, position: np.ndarray) -> bool:
        return (
            0.0 <= float(position[0]) <= float(config.AREA_WIDTH)
            and 0.0 <= float(position[1]) <= float(config.AREA_HEIGHT)
        )
