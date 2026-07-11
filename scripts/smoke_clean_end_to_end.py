from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.assignment import CleanAssignmentBuffer, freeze_ready_tasks
from environment.dag_tasks import TASK_STATE_IN_SERVICE, TASK_STATE_READY_UNSCHEDULED
from environment.env import Env
from environment.graph_builder import CleanGraphBuilder
from marl_models.mappo.clean_ppo import (
    CleanMovementActionRecord,
    CleanSlotRolloutRecord,
    build_clean_critic_global_input,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _center_scene(env: Env) -> None:
    center = np.array([250.0, 250.0], dtype=np.float32)
    env.hotspot_center = center.copy()
    for idx, uav in enumerate(env.uavs):
        uav.pos[:2] = center + np.array([float(idx), 0.0], dtype=np.float32)
    for ue in env.ues:
        ue.pos[:2] = center.copy()


def main() -> None:
    np.random.seed(37)
    original_arrival_prob = config.DAG_BASE_ARRIVAL_PROB
    config.DAG_BASE_ARRIVAL_PROB = 0.0
    torch_skips: list[str] = []
    try:
        env = Env()
        obs = env.reset()
        _center_scene(env)
        _assert(len(obs) == config.NUM_UAVS, "Env reset should return one obs per UAV.")

        empty_builder = CleanGraphBuilder()
        empty_snapshot = empty_builder.build(env.task_manager, env.uavs, 0, executor=env.executor)
        _assert(empty_snapshot.active_task_ids == [], "empty active graph should be legal.")
        _assert(empty_snapshot.incidence_matrix.shape == (0, 0), "empty incidence shape mismatch.")

        version_before = env.task_manager.dag_arrival_version
        ue = env.ues[0]
        job = env.task_manager.create_dag_for_ue(
            ue_id=ue.id,
            source_pos=ue.pos[:2].copy(),
            current_time_step=env.current_time_seconds,
        )
        ue.enter_service_waiting(job.dag_id)
        env.task_manager.refresh_ready_states()
        new_dag_arrived = env.task_manager.dag_arrival_version > version_before
        builder = CleanGraphBuilder()
        snapshot = builder.build(
            env.task_manager,
            env.uavs,
            env.time_step,
            executor=env.executor,
            new_dag_arrived=new_dag_arrived,
            dag_arrival_version=env.task_manager.dag_arrival_version,
        )
        _assert(new_dag_arrived, "DAG arrival version should advance on new DAG.")
        _assert(builder.last_attribute_update_step == env.time_step, "new DAG event should reach GraphBuilder.")
        _assert(not hasattr(snapshot, "uav_features"), "GraphSnapshot must be task-only.")
        _assert(not hasattr(snapshot, "candidate_mask"), "GraphSnapshot must not contain candidate masks.")
        _assert(
            snapshot.incidence_matrix.shape == (len(snapshot.task_ids), len(snapshot.hyperedges)),
            "incidence matrix should be nodes x hyperedges.",
        )

        from marl_models.mappo.clean_movement_actor import build_boundary_action_mask

        env.uavs[0].pos[:2] = np.array([0.0, 0.0], dtype=np.float32)
        mask = build_boundary_action_mask(
            uavs=env.uavs,
            pre_move_positions={int(uav.id): uav.pos[:2].copy() for uav in env.uavs},
        )
        _assert(bool(mask[0, 0]), "hover should always be legal in boundary mask.")
        _assert(not bool(mask[0, 2]), "-x should be illegal at x=0.")

        ready_tasks = freeze_ready_tasks(env.task_manager)
        _assert(ready_tasks, "new DAG should expose frozen ready tasks.")
        ready_task = ready_tasks[0]
        saved_queues = {uav_id: list(queue) for uav_id, queue in env.executor.uav_queues.items()}
        try:
            for uav in env.uavs:
                env.executor.uav_queues[int(uav.id)] = [f"dummy_{uav.id}_{idx}" for idx in range(config.CLEAN_MAX_QUEUE_PER_UAV)]
            skipped_buffer, skipped_count = env._build_assignment_buffer(ready_tasks, {ready_task.task_id: int(env.uavs[0].id)})
            skipped_snapshot = builder.build(
                env.task_manager,
                env.uavs,
                env.time_step,
                executor=env.executor,
                frozen_ready_task_ids=[task.task_id for task in ready_tasks],
            )
            _assert(skipped_buffer.entry_count == 0, "no legal candidate should not enter assignment_buffer.")
            _assert(skipped_count >= 1, "no legal candidate should be counted as skipped.")
            _assert(ready_task.state == TASK_STATE_READY_UNSCHEDULED, "skipped ready task should remain ready.")
            _assert(ready_task.task_id in skipped_snapshot.ready_task_ids, "skipped frozen ready task should stay in R_t.")
            _assert(ready_task.task_id not in skipped_snapshot.pending_task_ids, "skipped ready task must not be traced to pending.")
        finally:
            env.executor.uav_queues = saved_queues

        commit_buffer = CleanAssignmentBuffer()
        commit_buffer.append(ready_task.task_id, int(env.uavs[0].id), decision_order=0)
        _assert(ready_task.state == TASK_STATE_READY_UNSCHEDULED, "task should stay READY before executor commit.")
        env.executor.assign_tasks(
            assignments=commit_buffer,
            task_manager=env.task_manager,
            uavs=env.uavs,
            ues=env.ues,
            current_time_seconds=env.current_time_seconds,
            uav_service_positions={int(uav.id): uav.pos[:2].copy() for uav in env.uavs},
            ue_service_positions={int(ue.id): ue.pos[:2].copy() for ue in env.ues},
        )
        _assert(ready_task.state == TASK_STATE_IN_SERVICE, "executor commit should move task to IN_SERVICE.")

        run_env = Env()
        run_env.reset()
        _center_scene(run_env)
        run_ue = run_env.ues[0]
        run_job = run_env.task_manager.create_dag_for_ue(
            ue_id=run_ue.id,
            source_pos=run_ue.pos[:2].copy(),
            current_time_step=run_env.current_time_seconds,
        )
        run_ue.enter_service_waiting(run_job.dag_id)
        run_env.task_manager.refresh_ready_states()
        final_info = {}
        for _ in range(600):
            assignments = {
                task.task_id: int(task.level % config.NUM_UAVS)
                for task in run_env.task_manager.get_ready_tasks()
            }
            _, _, _, final_info = run_env.step(assignments)
            if run_job.completed:
                break
        _assert(run_job.completed, "end-to-end DAG should complete in short smoke.")
        _assert(run_job.completion_reward_settled, "DAG completion reward should be settled once.")
        for task in run_env.task_manager.get_job_tasks(run_job.dag_id):
            _assert(task.reward_settled, f"task {task.task_id} reward should be settled once.")
        for key in [
            "generated_dag_count",
            "completed_dag_count",
            "dag_throughput",
            "invalid_assignment_rate",
            "action_executed_rate",
        ]:
            _assert(key in final_info, f"metrics should expose {key}.")
        _assert(final_info["dag_throughput"] <= final_info["completed_dag_count"] / float(config.TIME_SLOT_DURATION), "throughput denominator should use executed time.")
        _assert(0.0 <= final_info["invalid_assignment_rate"] <= 1.0, "invalid rate should be normalized.")
        _assert(0.0 <= final_info["action_executed_rate"] <= 1.0, "action executed rate should be normalized.")

        task_embedding_dim = 8
        task_embeddings = np.random.normal(size=(len(snapshot.task_ids), task_embedding_dim)).astype(np.float32)
        critic_input = build_clean_critic_global_input(
            task_embeddings=task_embeddings,
            graph_snapshot=snapshot,
            uavs=env.uavs,
            executor=env.executor,
            current_time_seconds=env.current_time_seconds,
        )
        movement_record = CleanMovementActionRecord(
            uav_id=int(env.uavs[0].id),
            movement_observation={"boundary_mask_checked": True},
            movement_mask=mask[0].copy(),
            selected_action=0,
            old_log_probability=-0.1,
            entropy=0.0,
        )
        slot_record = CleanSlotRolloutRecord(
            graph_snapshot=snapshot,
            critic_global_input=critic_input,
            value=0.0,
            reward=float(final_info.get("step_reward", 0.0)),
            terminated=False,
            truncated=False,
            movement_records=[movement_record],
            offloading_records=[{"task_id": ready_task.task_id, "decision_order": 0}],
        )
        _assert(slot_record.graph_snapshot is snapshot, "rollout record should keep historical GraphSnapshot object.")
        _assert(slot_record.movement_records and slot_record.offloading_records, "rollout record should keep action records.")

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
                torch_skips.append("torch is not installed; skipped HGNN/actor/critic/PPO forward checks")
            else:
                raise
        else:
            torch.manual_seed(37)
            hgnn = CleanIncidenceHGNN(task_feature_dim=snapshot.task_features.shape[1], hidden_dim=16, output_dim=task_embedding_dim)
            task_embeddings_t = hgnn(snapshot.task_features, snapshot.incidence_matrix)
            _assert(task_embeddings_t.shape == (len(snapshot.task_ids), task_embedding_dim), "HGNN forward shape mismatch.")
            movement_observation = build_clean_movement_observation(
                uavs=env.uavs,
                executor=env.executor,
                graph_snapshot=snapshot,
                current_time_seconds=env.current_time_seconds,
            )
            movement_actor = CleanMovementActor(task_embedding_dim=task_embedding_dim, hidden_dim=16)
            movement_logits = movement_actor(
                uav_features=torch.as_tensor(movement_observation.uav_features, dtype=torch.float32),
                task_embeddings=task_embeddings_t,
                ready_task_indices=movement_observation.ready_task_indices,
                pending_task_indices=movement_observation.pending_task_indices,
                ready_count_normalized=movement_observation.ready_count_normalized,
                pending_count_normalized=movement_observation.pending_count_normalized,
                boundary_action_mask=torch.as_tensor(movement_observation.boundary_action_mask, dtype=torch.bool),
            )
            _assert(movement_logits.shape == (config.NUM_UAVS, config.CLEAN_MOVEMENT_ACTION_DIM), "movement logits shape mismatch.")
            off_actor = CleanOffloadingActor(task_embedding_dim=task_embedding_dim, hidden_dim=16)
            ready_for_actor = env.task_manager.get_ready_tasks()
            off_actor.act(
                frozen_ready_tasks=ready_for_actor,
                task_embeddings=task_embeddings_t.detach().numpy(),
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
            if off_actor.latest_records:
                scorer_logits = off_actor.scorer(off_actor.latest_records[0].candidate_features)
                _assert(scorer_logits.ndim == 1, "offloading scorer logits should be 1D per candidate set.")
            critic = CleanCentralizedCritic(input_dim=critic_input.shape[0], hidden_dim=16)
            critic_value = critic(torch.as_tensor(critic_input, dtype=torch.float32))
            _assert(critic_value.shape == (1,), "critic forward should output slot value.")
            torch_slot = CleanSlotRolloutRecord(
                graph_snapshot=snapshot,
                critic_global_input=critic_input,
                value=0.0,
                reward=1.0,
                terminated=False,
                truncated=False,
                movement_records=[
                    CleanMovementActionRecord(
                        uav_id=uav_id,
                        movement_observation=movement_observation,
                        movement_mask=movement_observation.boundary_action_mask[idx].copy(),
                        selected_action=0,
                        old_log_probability=-1.0,
                        entropy=0.0,
                    )
                    for idx, uav_id in enumerate(movement_observation.uav_ids)
                ],
                offloading_records=list(off_actor.latest_records),
            )
            advantages = torch.ones((1,), dtype=torch.float32)
            move_loss, _ = movement_ppo_loss(
                movement_actor=movement_actor,
                hgnn=hgnn,
                records=[torch_slot],
                advantages=advantages,
            )
            off_loss, _, _ = offloading_ppo_loss(
                offloading_scorer=off_actor.scorer,
                records=[torch_slot],
                advantages=advantages,
            )
            _assert(move_loss.dim() == 0 and off_loss.dim() == 0, "PPO losses should be scalar.")
    finally:
        config.DAG_BASE_ARRIVAL_PROB = original_arrival_prob

    if torch_skips:
        for reason in torch_skips:
            print(f"smoke_clean_end_to_end torch checks skipped: {reason}")
    print("smoke_clean_end_to_end passed")


if __name__ == "__main__":
    main()
