from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.assignment import (
    CLEAN_OFFLOADING_PAIR_FEATURE_DIM,
    CLEAN_OFFLOADING_UAV_FEATURE_DIM,
    TemporaryReservationState,
    build_offloading_candidate_batch,
)
from environment.env import Env
from environment.graph_builder import CleanGraphBuilder


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    np.random.seed(29)
    original_arrival_prob = config.DAG_BASE_ARRIVAL_PROB
    config.DAG_BASE_ARRIVAL_PROB = 0.0
    try:
        env = Env()
        env.reset()
        center = np.array([250.0, 250.0], dtype=np.float32)
        for idx, uav in enumerate(env.uavs):
            uav.pos[:2] = center + np.array([float(idx), 0.0], dtype=np.float32)
        ue = env.ues[0]
        ue.pos[:2] = center.copy()
        job = env.task_manager.create_dag_for_ue(
            ue_id=ue.id,
            source_pos=ue.pos[:2].copy(),
            current_time_step=env.current_time_seconds,
        )
        ue.enter_service_waiting(job.dag_id)
        env.task_manager.refresh_ready_states()
        snapshot = CleanGraphBuilder().build(env.task_manager, env.uavs, env.time_step, executor=env.executor)
        ready_tasks = env.task_manager.get_ready_tasks()
        _assert(ready_tasks, "DAG should expose at least one ready task.")

        task_embedding_dim = 8
        task_embeddings = np.random.normal(size=(len(snapshot.task_ids), task_embedding_dim)).astype(np.float32)
        task = ready_tasks[0]
        reservation = TemporaryReservationState.from_executor(env.uavs, env.executor)
        candidate_features, candidate_mask, candidate_uav_ids, estimates = build_offloading_candidate_batch(
            task=task,
            task_embedding=task_embeddings[snapshot.task_id_to_idx[task.task_id]],
            uavs=env.uavs,
            task_manager=env.task_manager,
            executor=env.executor,
            state_view=reservation,
            current_time_seconds=env.current_time_seconds,
            uav_service_positions={int(uav.id): uav.pos[:2].copy() for uav in env.uavs},
            ue_service_positions={int(ue.id): ue.pos[:2].copy() for ue in env.ues},
            ues=env.ues,
        )
        expected_dim = task_embedding_dim + CLEAN_OFFLOADING_UAV_FEATURE_DIM + CLEAN_OFFLOADING_PAIR_FEATURE_DIM
        _assert(candidate_features.shape == (config.NUM_UAVS, expected_dim), "candidate feature shape mismatch.")
        _assert(candidate_mask.shape == (config.NUM_UAVS,), "candidate mask shape mismatch.")
        _assert(candidate_mask.any(), "ready task should have at least one legal candidate.")
        _assert(candidate_uav_ids == sorted(candidate_uav_ids), "candidate UAV ids should be stable-sorted.")
        _assert(len(estimates) == config.NUM_UAVS, "estimate count should match candidates.")
        _assert(np.all(np.isfinite(candidate_features)), "candidate features should be finite.")
        _assert(not hasattr(snapshot, "candidate_mask"), "GraphSnapshot must not contain candidate masks.")
        _assert(not hasattr(snapshot, "candidate_features"), "GraphSnapshot must not contain candidate features.")

        selected_uav_id = candidate_uav_ids[int(np.flatnonzero(candidate_mask)[0])]
        selected_estimate = estimates[candidate_uav_ids.index(selected_uav_id)]
        reservation.reserve(
            task.task_id,
            selected_uav_id,
            estimated_available_time=selected_estimate.estimated_finish_time,
            estimated_queued_workload=selected_estimate.estimated_queued_workload,
        )
        _, reserved_mask, _, _ = build_offloading_candidate_batch(
            task=task,
            task_embedding=task_embeddings[snapshot.task_id_to_idx[task.task_id]],
            uavs=env.uavs,
            task_manager=env.task_manager,
            executor=env.executor,
            state_view=reservation,
            current_time_seconds=env.current_time_seconds,
            uav_service_positions={int(uav.id): uav.pos[:2].copy() for uav in env.uavs},
            ue_service_positions={int(ue.id): ue.pos[:2].copy() for ue in env.ues},
            ues=env.ues,
        )
        _assert(not reserved_mask.any(), "reserved task should have no legal candidates in the same sequential pass.")

        try:
            import torch
            from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                print("smoke_clean_offloading_actor feature/mask passed; actor skipped: torch is not installed")
                return
            raise

        torch.manual_seed(29)
        actor = CleanOffloadingActor(task_embedding_dim=task_embedding_dim, hidden_dim=16)
        assignment_buffer = actor.act(
            frozen_ready_tasks=ready_tasks,
            task_embeddings=task_embeddings,
            graph_snapshot=snapshot,
            task_manager=env.task_manager,
            uavs=env.uavs,
            executor=env.executor,
            current_time_seconds=env.current_time_seconds,
            uav_service_positions={int(uav.id): uav.pos[:2].copy() for uav in env.uavs},
            ue_service_positions={int(ue.id): ue.pos[:2].copy() for ue in env.ues},
            ues=env.ues,
            deterministic=True,
        )
        _assert(assignment_buffer.entry_count == len(actor.latest_records), "action record count should match buffer entries.")
        _assert(actor.latest_records, "actor should emit records for legal ready decisions.")
        first_record = actor.latest_records[0]
        _assert(first_record.task_id in {task.task_id for task in ready_tasks}, "record task id should come from frozen ready set.")
        _assert(first_record.candidate_features.shape[1] == expected_dim, "record candidate tensor dim mismatch.")
        _assert(first_record.candidate_mask.shape[0] == config.NUM_UAVS, "record candidate mask dim mismatch.")
        _assert(first_record.selected_uav_id in first_record.candidate_uav_ids, "selected UAV should be in candidate mapping.")
    finally:
        config.DAG_BASE_ARRIVAL_PROB = original_arrival_prob

    print("smoke_clean_offloading_actor passed")


if __name__ == "__main__":
    main()
