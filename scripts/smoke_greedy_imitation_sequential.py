from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.assignment import TemporaryReservationState
from environment.env import Env
from environment.graph_builder import CleanGraphBuilder
from marl_models.mappo.clean_slot_orchestrator import prepare_slot_state
from scripts import train_greedy_imitation_gate as gate


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    original_arrival_prob = config.DAG_BASE_ARRIVAL_PROB
    config.DAG_BASE_ARRIVAL_PROB = 0.0
    graph_builder = CleanGraphBuilder()
    try:
        env = Env(freeze_ue_mobility=True)
        env.reset()
        _create_manual_dags_until_ready(env, min_ready=2)
        prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
        env.apply_movement({})
        ready_tasks = [
            env.task_manager.get_task(task_id)
            for task_id in prepared.frozen_ready_task_ids
        ]
        ready_tasks = [task for task in ready_tasks if task is not None and task.is_ready]
        _assert(len(ready_tasks) >= 2, "sequential smoke requires at least two frozen ready tasks.")

        first_task = ready_tasks[0]
        second_task = ready_tasks[1]
        reservation = TemporaryReservationState.from_executor(env.uavs, env.executor)
        first_sample = gate._build_decision_sample(
            trajectory_policy="greedy_eft",
            prepared=prepared,
            task=first_task,
            decision_order=0,
            reservation=reservation,
            env=env,
            environment_seed=42,
            episode=0,
            slot=0,
        )
        _assert(first_sample is not None, "first ready task should have a legal candidate.")

        fresh_second = gate._build_decision_sample(
            trajectory_policy="greedy_eft",
            prepared=prepared,
            task=second_task,
            decision_order=1,
            reservation=TemporaryReservationState.from_executor(env.uavs, env.executor),
            env=env,
            environment_seed=42,
            episode=0,
            slot=0,
        )
        _assert(fresh_second is not None, "second ready task should have a legal candidate before reservation.")

        selected = int(first_sample["behavior_idx"])
        selected_uav_id = int(first_sample["candidate_uav_ids"][selected])
        reservation.reserve(
            str(first_task.task_id),
            selected_uav_id,
            estimated_available_time=float(first_sample["estimated_finish_times"][selected]),
            estimated_queued_workload=float(first_sample["estimated_queued_workloads"][selected]),
        )
        after_second = gate._build_decision_sample(
            trajectory_policy="greedy_eft",
            prepared=prepared,
            task=second_task,
            decision_order=1,
            reservation=reservation,
            env=env,
            environment_seed=42,
            episode=0,
            slot=0,
        )
        _assert(after_second is not None, "second ready task should remain sampleable after one reservation.")

        selected_row = after_second["candidate_uav_ids"].index(selected_uav_id)
        before_dynamic = np.asarray(fresh_second["dynamic_uav_features"], dtype=np.float32)[selected_row]
        after_dynamic = np.asarray(after_second["dynamic_uav_features"], dtype=np.float32)[selected_row]
        _assert(
            not np.allclose(before_dynamic, after_dynamic),
            "second decision dynamic_uav_features must reflect the first task temporary reservation.",
        )
        _assert(
            after_dynamic[2] >= before_dynamic[2] and after_dynamic[3] <= before_dynamic[3],
            "queue length should not decrease and remaining queue slots should not increase after reservation.",
        )

        repeat_after = gate._build_decision_sample(
            trajectory_policy="greedy_eft",
            prepared=prepared,
            task=second_task,
            decision_order=1,
            reservation=reservation,
            env=env,
            environment_seed=42,
            episode=0,
            slot=0,
        )
        _assert(repeat_after is not None, "repeat sample should still be valid.")
        for key in ("candidate_mask", "pair_features", "dynamic_uav_features", "estimated_finish_times"):
            _assert(
                np.allclose(np.asarray(after_second[key]), np.asarray(repeat_after[key])),
                f"{key} should be deterministic for a fixed reservation state.",
            )
        _assert(
            int(after_second["greedy_label_idx"]) == int(repeat_after["greedy_label_idx"]),
            "greedy label should be deterministic for the same reservation state.",
        )

        full_reservation = TemporaryReservationState.from_executor(env.uavs, env.executor)
        for uav in env.uavs:
            full_reservation.queue_lengths[int(uav.id)] = int(config.CLEAN_MAX_QUEUE_PER_UAV)
        no_candidate = gate._build_decision_sample(
            trajectory_policy="greedy_eft",
            prepared=prepared,
            task=first_task,
            decision_order=0,
            reservation=full_reservation,
            env=env,
            environment_seed=42,
            episode=0,
            slot=0,
        )
        _assert(no_candidate is None, "no legal UAV candidate should not produce a training sample.")

        _assert(
            "hyperedge_type_ids" in first_sample,
            "sample must save hyperedge_type_ids as raw graph input.",
        )
        incidence = np.asarray(first_sample["incidence_matrix"])
        type_ids = np.asarray(first_sample["hyperedge_type_ids"])
        _assert(
            int(incidence.shape[1]) == int(type_ids.shape[0]),
            "sample hyperedge_type_ids length must align with incidence columns.",
        )
        for key in ("task_features", "task_id_to_idx", "idx_to_task_id", "active_task_ids", "ready_task_ids", "pending_task_ids", "task_local_index"):
            _assert(key in first_sample, f"sample missing raw graph field {key}.")

        frozen_before = set(prepared.frozen_ready_task_ids)
        child_ready_from_parent = _complete_frozen_parent_if_possible(env, prepared.frozen_ready_task_ids)
        if not child_ready_from_parent:
            _create_one_more_dag_if_possible(env)
        env.task_manager.refresh_ready_states()
        live_ready = {task.task_id for task in env.task_manager.get_ready_tasks()}
        _assert(
            set(prepared.frozen_ready_task_ids) == frozen_before,
            "prepared frozen ready ids must not change after live task manager refresh.",
        )
        _assert(
            prepared.frozen_ready_task_ids == list(prepared.graph_snapshot.ready_task_ids),
            "graph snapshot ready ids must match the frozen ready sequence used by imitation.",
        )
        newly_ready_after_freeze = live_ready - frozen_before
        if child_ready_from_parent:
            _assert(newly_ready_after_freeze, "completing a frozen parent should create a newly ready child in this smoke.")
        if newly_ready_after_freeze:
            _assert(
                not newly_ready_after_freeze.intersection(prepared.frozen_ready_task_ids),
                "newly ready tasks after freeze must not enter the current imitation sequence.",
            )

        for forbidden in (
            "clean_mappo",
            "clean_assignment_policy",
            "train_clean_assignment_mappo",
        ):
            _assert(forbidden not in sys.modules, f"smoke must not import legacy module {forbidden}.")
    finally:
        graph_builder.close()
        config.DAG_BASE_ARRIVAL_PROB = original_arrival_prob

    print("smoke_greedy_imitation_sequential passed")
    return 0


def _create_manual_dags_until_ready(env: Env, *, min_ready: int) -> None:
    for ue in env.ues:
        if len(env.task_manager.get_ready_tasks()) >= int(min_ready):
            return
        if not env.task_manager.can_accept_dag_for_ue(int(ue.id)):
            continue
        job = env.task_manager.create_dag_for_ue(
            ue_id=int(ue.id),
            source_pos=ue.pos[:2].copy(),
            current_time_step=float(env.current_time_seconds),
        )
        if hasattr(ue, "enter_service_waiting"):
            ue.enter_service_waiting(job.dag_id)
        env.task_manager.refresh_ready_states()
    if len(env.task_manager.get_ready_tasks()) < int(min_ready):
        raise AssertionError("could not create enough ready tasks for sequential smoke.")


def _create_one_more_dag_if_possible(env: Env) -> None:
    for ue in env.ues:
        if not env.task_manager.can_accept_dag_for_ue(int(ue.id)):
            continue
        job = env.task_manager.create_dag_for_ue(
            ue_id=int(ue.id),
            source_pos=ue.pos[:2].copy(),
            current_time_step=float(env.current_time_seconds),
        )
        if hasattr(ue, "enter_service_waiting"):
            ue.enter_service_waiting(job.dag_id)
        return


def _complete_frozen_parent_if_possible(env: Env, frozen_ready_task_ids: list[str]) -> bool:
    frozen_set = set(str(task_id) for task_id in frozen_ready_task_ids)
    for task_id in frozen_ready_task_ids:
        task = env.task_manager.get_task(task_id)
        if task is None or not task.successors:
            continue
        for child_id in task.successors:
            child = env.task_manager.get_task(child_id)
            if child is None:
                continue
            if not set(str(parent_id) for parent_id in child.predecessors).issubset(frozen_set):
                continue
            for parent_id in child.predecessors:
                parent = env.task_manager.get_task(parent_id)
                if parent is not None and parent.is_ready:
                    env.task_manager.mark_task_finished(str(parent.task_id), float(env.current_time_seconds))
            env.task_manager.refresh_ready_states()
            refreshed_child = env.task_manager.get_task(child_id)
            if refreshed_child is not None and refreshed_child.is_ready:
                return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
