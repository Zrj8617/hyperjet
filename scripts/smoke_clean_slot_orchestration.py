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
from marl_models.mappo.clean_slot_orchestrator import (
    CleanSlotRolloutBuffer,
    assert_graph_snapshot_task_only,
    copy_clean_graph_snapshot,
    encode_prepared_slot,
    make_slot_rollout_record,
    prepare_slot_state,
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
    np.random.seed(43)
    original_arrival_prob = config.DAG_BASE_ARRIVAL_PROB
    config.DAG_BASE_ARRIVAL_PROB = 0.0
    try:
        env = Env()
        env.reset()
        _center_scene(env)
        graph_builder = CleanGraphBuilder()

        empty_prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
        _assert(empty_prepared.graph_snapshot.active_task_ids == [], "empty active prepared graph should be legal.")
        empty_encoded = encode_prepared_slot(prepared_state=empty_prepared, env=env, fallback_embedding_dim=8)
        _assert(empty_encoded.task_embeddings.shape == (0, 8), "empty encode should return empty embeddings.")
        _assert(empty_encoded.movement_observation.boundary_action_mask.shape[0] == config.NUM_UAVS, "movement mask should still exist with empty graph.")
        env.apply_movement({})
        _, _, _, _ = env.commit_and_advance(assignments={})

        ue = env.ues[0]
        job = env.task_manager.create_dag_for_ue(
            ue_id=ue.id,
            source_pos=ue.pos[:2].copy(),
            current_time_step=env.time_step,
        )
        ue.enter_service_waiting(job.dag_id)
        env.task_manager.refresh_ready_states()
        prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
        _assert(prepared.frozen_ready_task_ids, "prepared state should freeze ready tasks before movement.")
        assert_graph_snapshot_task_only(prepared.graph_snapshot)

        encoded_a = encode_prepared_slot(prepared_state=prepared, env=env, fallback_embedding_dim=8)
        active_before = list(prepared.graph_snapshot.active_task_ids)
        features_before = prepared.graph_snapshot.task_features.copy()
        incidence_before = prepared.graph_snapshot.incidence_matrix.copy()
        encoded_b = encode_prepared_slot(prepared_state=prepared, env=env, fallback_embedding_dim=8)
        _assert(np.array_equal(encoded_a.task_embeddings, encoded_b.task_embeddings), "same prepared state should be encodable more than once.")
        _assert(prepared.graph_snapshot.active_task_ids == active_before, "encode must not mutate prepared graph ids.")
        _assert(np.array_equal(prepared.graph_snapshot.task_features, features_before), "encode must not mutate task features.")
        _assert(np.array_equal(prepared.graph_snapshot.incidence_matrix, incidence_before), "encode must not mutate incidence.")

        historical_copy = copy_clean_graph_snapshot(prepared.graph_snapshot)
        _assert(historical_copy is not prepared.graph_snapshot, "historical graph must be an independent object.")
        _assert(historical_copy.task_features is not prepared.graph_snapshot.task_features, "task_features must be copied.")
        _assert(historical_copy.incidence_matrix is not prepared.graph_snapshot.incidence_matrix, "incidence_matrix must be copied.")
        _assert(not historical_copy.task_features.flags.writeable, "historical task_features should be write-protected.")
        _assert(not historical_copy.incidence_matrix.flags.writeable, "historical incidence should be write-protected.")

        slot_record = make_slot_rollout_record(encoded_state=encoded_a)
        buffer = CleanSlotRolloutBuffer()
        buffer.append(slot_record)
        env.apply_movement({})
        _, _, _, _ = env.commit_and_advance(assignments={})

        _assert(slot_record.graph_snapshot.active_task_ids == historical_copy.active_task_ids, "rollout graph active ids changed after env advanced.")
        _assert(slot_record.graph_snapshot.ready_task_ids == historical_copy.ready_task_ids, "rollout graph ready ids changed after env advanced.")
        _assert(slot_record.graph_snapshot.pending_task_ids == historical_copy.pending_task_ids, "rollout graph pending ids changed after env advanced.")
        _assert(np.array_equal(slot_record.graph_snapshot.task_features, historical_copy.task_features), "rollout task features changed after env advanced.")
        _assert(np.array_equal(slot_record.graph_snapshot.incidence_matrix, historical_copy.incidence_matrix), "rollout incidence changed after env advanced.")

        next_prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
        next_encoded_old = encode_prepared_slot(prepared_state=next_prepared, env=env, fallback_embedding_dim=8)
        buffer.close(bootstrap_value=0.0 if next_encoded_old.value is None else float(next_encoded_old.value))
        _assert(buffer.closed, "rollout buffer should close at bootstrap boundary.")
        next_encoded_new = encode_prepared_slot(prepared_state=next_prepared, env=env, fallback_embedding_dim=8)
        _assert(next_prepared.slot_index == next_encoded_new.prepared_state.slot_index, "same next prepared state should be reused after update.")
        _assert(env.time_step == next_prepared.time_step, "re-encoding must not prepare or advance the environment.")

        try:
            env.prepare_slot_state()
        except RuntimeError:
            pass
        else:
            raise AssertionError("double prepare before commit should be rejected.")

        env.apply_movement({})
        _, _, _, _ = env.commit_and_advance(assignments={})

        try:
            import torch
            from marl_models.hgnn import CleanIncidenceHGNN
            from marl_models.mappo.clean_movement_actor import CleanMovementActor
            from marl_models.mappo.clean_ppo import CleanCentralizedCritic, clean_critic_input_dim
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                print("smoke_clean_slot_orchestration passed; torch encode checks skipped")
                return
            raise

        torch.manual_seed(43)
        torch_prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
        task_feature_dim = torch_prepared.graph_snapshot.task_features.shape[1]
        embedding_dim = 8
        hgnn = CleanIncidenceHGNN(task_feature_dim=task_feature_dim, hidden_dim=16, output_dim=embedding_dim)
        critic = CleanCentralizedCritic(input_dim=clean_critic_input_dim(embedding_dim, config.NUM_UAVS), hidden_dim=16)
        movement_actor = CleanMovementActor(task_embedding_dim=embedding_dim, hidden_dim=16)
        torch_encoded_a = encode_prepared_slot(
            prepared_state=torch_prepared,
            env=env,
            hgnn=hgnn,
            critic=critic,
            movement_actor=movement_actor,
        )
        torch_encoded_b = encode_prepared_slot(
            prepared_state=torch_prepared,
            env=env,
            hgnn=hgnn,
            critic=critic,
            movement_actor=movement_actor,
        )
        _assert(tuple(torch_encoded_a.task_embeddings.shape) == tuple(torch_encoded_b.task_embeddings.shape), "torch re-encode shape mismatch.")
        _assert(torch_encoded_a.movement_logits.shape == (config.NUM_UAVS, config.CLEAN_MOVEMENT_ACTION_DIM), "movement logits shape mismatch.")
        env.apply_movement({})
        _, _, _, _ = env.commit_and_advance(assignments={})
    finally:
        config.DAG_BASE_ARRIVAL_PROB = original_arrival_prob

    print("smoke_clean_slot_orchestration passed")


if __name__ == "__main__":
    main()
