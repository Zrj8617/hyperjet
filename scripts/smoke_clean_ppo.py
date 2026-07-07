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
from marl_models.mappo.clean_ppo import (
    CleanMovementActionRecord,
    CleanSlotRolloutRecord,
    build_clean_critic_global_input,
    clean_critic_input_dim,
    compute_gae_numpy,
    summarize_ppo_records,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    np.random.seed(31)
    original_arrival_prob = config.DAG_BASE_ARRIVAL_PROB
    config.DAG_BASE_ARRIVAL_PROB = 0.0
    try:
        env = Env()
        env.reset()
        ue = env.ues[0]
        job = env.task_manager.create_dag_for_ue(
            ue_id=ue.id,
            source_pos=ue.pos[:2].copy(),
            current_time_step=env.time_step,
        )
        ue.enter_service_waiting(job.dag_id)
        env.task_manager.refresh_ready_states()
        snapshot = CleanGraphBuilder().build(env.task_manager, env.uavs, env.time_step, executor=env.executor)

        task_embedding_dim = 8
        task_embeddings = np.random.normal(size=(len(snapshot.task_ids), task_embedding_dim)).astype(np.float32)
        critic_input = build_clean_critic_global_input(
            task_embeddings=task_embeddings,
            graph_snapshot=snapshot,
            uavs=env.uavs,
            executor=env.executor,
            pre_move_positions={int(uav.id): uav.pos[:2].copy() for uav in env.uavs},
            current_time_step=env.time_step,
        )
        _assert(
            critic_input.shape == (clean_critic_input_dim(task_embedding_dim, config.NUM_UAVS),),
            "critic global input dimension mismatch.",
        )
        _assert(np.all(np.isfinite(critic_input)), "critic global input should be finite.")
        _assert(not hasattr(snapshot, "critic_global_input"), "GraphSnapshot must not store critic non-graph inputs.")

        empty_env = Env()
        empty_env.reset()
        empty_snapshot = CleanGraphBuilder().build(empty_env.task_manager, empty_env.uavs, 0, executor=empty_env.executor)
        empty_critic_input = build_clean_critic_global_input(
            task_embeddings=np.zeros((0, task_embedding_dim), dtype=np.float32),
            graph_snapshot=empty_snapshot,
            uavs=empty_env.uavs,
            executor=empty_env.executor,
            current_time_step=0,
        )
        _assert(empty_critic_input[:task_embedding_dim].sum() == 0.0, "empty active task mean should be zero.")

        movement_record = CleanMovementActionRecord(
            uav_id=0,
            movement_observation={"placeholder": "historical observation"},
            movement_mask=np.ones((config.CLEAN_MOVEMENT_ACTION_DIM,), dtype=bool),
            selected_action=0,
            old_log_probability=-0.1,
            entropy=0.2,
        )
        records = [
            CleanSlotRolloutRecord(
                graph_snapshot=snapshot,
                critic_global_input=critic_input,
                value=1.0,
                reward=1.0,
                terminated=False,
                truncated=True,
                movement_records=[movement_record],
                offloading_records=[],
            ),
            CleanSlotRolloutRecord(
                graph_snapshot=snapshot,
                critic_global_input=critic_input,
                value=2.0,
                reward=2.0,
                terminated=True,
                truncated=False,
                movement_records=[movement_record],
                offloading_records=[object()],
            ),
        ]
        returns, advantages = compute_gae_numpy(records, next_value=10.0, gamma=1.0, gae_lambda=1.0)
        _assert(np.isclose(advantages[1], 0.0), "terminated step should not bootstrap from next value.")
        _assert(np.isclose(advantages[0], 2.0), "truncated non-terminated step should continue bootstrapping through next value.")
        _assert(np.isclose(returns[0], 3.0), "return should equal value plus advantage.")
        stats = summarize_ppo_records(records)
        _assert(stats.movement_action_count == 2, "movement action count mismatch.")
        _assert(stats.offloading_action_count == 1, "offloading action count mismatch.")
        _assert(stats.offloading_effective_slot_count == 1, "M_t=0 slot should not enter offloading denominator.")

        try:
            import torch
            from marl_models.hgnn import CleanIncidenceHGNN
            from marl_models.mappo.clean_movement_actor import CleanMovementActor, build_clean_movement_observation
            from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
            from marl_models.mappo.clean_ppo import (
                CleanCentralizedCritic,
                movement_ppo_loss,
                offloading_ppo_loss,
            )
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                print("smoke_clean_ppo record/critic-input passed; torch PPO checks skipped")
                return
            raise

        torch.manual_seed(31)
        critic = CleanCentralizedCritic(input_dim=critic_input.shape[0], hidden_dim=16)
        value = critic(torch.as_tensor(critic_input, dtype=torch.float32))
        _assert(value.shape == (1,), "centralized critic should output slot-level V(s_t).")

        hgnn = CleanIncidenceHGNN(task_feature_dim=snapshot.task_features.shape[1], hidden_dim=16, output_dim=task_embedding_dim)
        movement_actor = CleanMovementActor(task_embedding_dim=task_embedding_dim, hidden_dim=16)
        movement_observation = build_clean_movement_observation(
            uavs=env.uavs,
            executor=env.executor,
            graph_snapshot=snapshot,
            current_time_step=env.time_step,
        )
        movement_records = [
            CleanMovementActionRecord(
                uav_id=uav_id,
                movement_observation=movement_observation,
                movement_mask=movement_observation.boundary_action_mask[idx].copy(),
                selected_action=0,
                old_log_probability=-1.0,
                entropy=0.0,
            )
            for idx, uav_id in enumerate(movement_observation.uav_ids)
        ]
        offloading_actor = CleanOffloadingActor(task_embedding_dim=task_embedding_dim, hidden_dim=16)
        ready_tasks = env.task_manager.get_ready_tasks()
        offloading_actor.act(
            frozen_ready_tasks=ready_tasks,
            task_embeddings=task_embeddings,
            graph_snapshot=snapshot,
            task_manager=env.task_manager,
            uavs=env.uavs,
            executor=env.executor,
            current_time_step=env.time_step,
            uav_service_positions={int(uav.id): uav.pos[:2].copy() for uav in env.uavs},
            ue_service_positions={int(ue.id): ue.pos[:2].copy() for ue in env.ues},
            ues=env.ues,
            deterministic=True,
        )
        torch_records = [
            CleanSlotRolloutRecord(
                graph_snapshot=snapshot,
                critic_global_input=critic_input,
                value=0.0,
                reward=1.0,
                terminated=False,
                truncated=False,
                movement_records=movement_records,
                offloading_records=list(offloading_actor.latest_records),
            ),
            CleanSlotRolloutRecord(
                graph_snapshot=snapshot,
                critic_global_input=critic_input,
                value=0.0,
                reward=0.0,
                terminated=False,
                truncated=True,
                movement_records=movement_records,
                offloading_records=[],
            ),
        ]
        advantages_t = torch.ones((len(torch_records),), dtype=torch.float32)
        move_loss, move_entropy = movement_ppo_loss(
            movement_actor=movement_actor,
            hgnn=hgnn,
            records=torch_records,
            advantages=advantages_t,
        )
        off_loss, off_entropy, off_slots = offloading_ppo_loss(
            offloading_scorer=offloading_actor.scorer,
            records=torch_records,
            advantages=advantages_t,
        )
        _assert(move_loss.dim() == 0 and move_entropy.dim() == 0, "movement PPO aggregation should return scalars.")
        _assert(off_loss.dim() == 0 and off_entropy.dim() == 0, "offloading PPO aggregation should return scalars.")
        _assert(off_slots == 1, "offloading loss should only count slots with M_t > 0.")
    finally:
        config.DAG_BASE_ARRIVAL_PROB = original_arrival_prob

    print("smoke_clean_ppo passed")


if __name__ == "__main__":
    main()
