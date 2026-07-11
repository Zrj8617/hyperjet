from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.env import Env
from environment.graph_builder import CleanGraphBuilder


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    try:
        import torch
        from marl_models.mappo.clean_movement_actor import (
            CLEAN_MOVEMENT_UAV_FEATURE_DIM,
            CleanMovementActor,
            build_clean_movement_observation,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            print("smoke_clean_movement_actor skipped: torch is not installed in this Python runtime")
            return
        raise

    np.random.seed(23)
    torch.manual_seed(23)
    original_arrival_prob = config.DAG_BASE_ARRIVAL_PROB
    config.DAG_BASE_ARRIVAL_PROB = 0.0
    try:
        env = Env()
        env.reset()
        env.uavs[0].pos[:2] = np.array([0.0, 0.0], dtype=np.float32)
        ue = env.ues[0]
        job = env.task_manager.create_dag_for_ue(
            ue_id=ue.id,
            source_pos=ue.pos[:2].copy(),
            current_time_step=env.current_time_seconds,
        )
        ue.enter_service_waiting(job.dag_id)
        env.task_manager.refresh_ready_states()

        snapshot = CleanGraphBuilder().build(env.task_manager, env.uavs, env.time_step, executor=env.executor)
        observation = build_clean_movement_observation(
            uavs=env.uavs,
            executor=env.executor,
            graph_snapshot=snapshot,
            pre_move_positions={int(uav.id): uav.pos[:2].copy() for uav in env.uavs},
            current_time_seconds=env.current_time_seconds,
        )
        _assert(
            observation.uav_features.shape == (config.NUM_UAVS, CLEAN_MOVEMENT_UAV_FEATURE_DIM),
            "movement UAV feature shape mismatch.",
        )
        _assert(
            observation.boundary_action_mask.shape == (config.NUM_UAVS, config.CLEAN_MOVEMENT_ACTION_DIM),
            "movement boundary mask shape mismatch.",
        )
        _assert(bool(observation.boundary_action_mask[0, 0]), "hover should always be legal.")
        _assert(not bool(observation.boundary_action_mask[0, 2]), "-x should be illegal at x=0.")
        _assert(not bool(observation.boundary_action_mask[0, 4]), "-y should be illegal at y=0.")

        task_embedding_dim = 8
        task_embeddings = torch.randn((len(snapshot.task_ids), task_embedding_dim), dtype=torch.float32)
        actor = CleanMovementActor(task_embedding_dim=task_embedding_dim, hidden_dim=16)
        logits = actor(
            uav_features=torch.as_tensor(observation.uav_features, dtype=torch.float32),
            task_embeddings=task_embeddings,
            ready_task_indices=observation.ready_task_indices,
            pending_task_indices=observation.pending_task_indices,
            ready_count_normalized=observation.ready_count_normalized,
            pending_count_normalized=observation.pending_count_normalized,
            boundary_action_mask=torch.as_tensor(observation.boundary_action_mask, dtype=torch.bool),
        )
        _assert(logits.shape == (config.NUM_UAVS, config.CLEAN_MOVEMENT_ACTION_DIM), "movement logits shape mismatch.")
        boundary_mask = torch.as_tensor(observation.boundary_action_mask, dtype=torch.bool, device=logits.device)
        _assert(torch.isfinite(logits[boundary_mask]).all().item(), "legal logits should be finite.")
        _assert(
            torch.all(logits[~boundary_mask] < -1.0e20).item(),
            "illegal movement logits should be masked after network output.",
        )

        empty_env = Env()
        empty_env.reset()
        empty_snapshot = CleanGraphBuilder().build(empty_env.task_manager, empty_env.uavs, 0, executor=empty_env.executor)
        empty_observation = build_clean_movement_observation(
            uavs=empty_env.uavs,
            executor=empty_env.executor,
            graph_snapshot=empty_snapshot,
            current_time_seconds=0.0,
        )
        empty_logits = actor(
            uav_features=torch.as_tensor(empty_observation.uav_features, dtype=torch.float32),
            task_embeddings=torch.zeros((0, task_embedding_dim), dtype=torch.float32),
            ready_task_indices=empty_observation.ready_task_indices,
            pending_task_indices=empty_observation.pending_task_indices,
            ready_count_normalized=empty_observation.ready_count_normalized,
            pending_count_normalized=empty_observation.pending_count_normalized,
            boundary_action_mask=torch.as_tensor(empty_observation.boundary_action_mask, dtype=torch.bool),
        )
        _assert(empty_logits.shape == (config.NUM_UAVS, config.CLEAN_MOVEMENT_ACTION_DIM), "empty graph movement logits shape mismatch.")
    finally:
        config.DAG_BASE_ARRIVAL_PROB = original_arrival_prob

    print("smoke_clean_movement_actor passed")


if __name__ == "__main__":
    main()
