from __future__ import annotations

from pathlib import Path
import json
import os
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.env import Env
from environment.graph_builder import CleanGraphBuilder
from marl_models.mappo.clean_slot_orchestrator import (
    CleanMovementRolloutRecord,
    CleanOffloadingRolloutRecord,
    CleanSlotRolloutBuffer,
    encode_prepared_slot,
    make_slot_rollout_record,
    prepare_slot_state,
)
from marl_models.mappo.clean_trainer import (
    CleanCheckpointManager,
    CleanJSONLLogger,
    CleanPPOUpdateConfig,
    CleanTrainingModules,
    build_single_optimizer,
    close_rollout_with_bootstrap,
    compute_slot_level_gae,
    reencode_prepared_after_update,
    write_clean_training_log,
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


def _non_torch_checks() -> None:
    dummy_records = [
        type("Record", (), {"reward": 1.0, "value": 0.5, "terminated": False})(),
        type("Record", (), {"reward": 1.0, "value": 0.25, "terminated": True})(),
    ]
    returns, advantages = compute_slot_level_gae(dummy_records, bootstrap_value=10.0, gamma=1.0, gae_lambda=1.0)
    _assert(np.isclose(advantages[1], 0.75), "terminated slot should not bootstrap from next value.")
    _assert(np.isclose(returns[1], 1.0), "terminated return should equal immediate reward.")
    _assert(np.isclose(returns[0], 2.0), "previous non-terminated slot should bootstrap from following slot value.")

    with _workspace_temp_dir("non_torch") as tmp_dir:
        logger = CleanJSONLLogger(tmp_dir)
        write_clean_training_log(
            logger,
            episode=0,
            global_slot=1,
            info={
                "step_reward": 1.0,
                "step_time_penalty": -0.1,
                "step_energy_penalty": -0.2,
                "generated_dag_count": 1,
                "completed_dag_count": 0,
                "dag_completion_rate": 0.0,
                "dag_throughput": 0.0,
                "invalid_assignment_count": 0,
                "invalid_assignment_rate": 0.0,
                "action_executed_rate": 1.0,
                "assignment_buffer_entry_count": 0,
                "truncated": True,
            },
            torch_skipped=True,
        )
        log_path = Path(tmp_dir) / "train_metrics.jsonl"
        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        _assert(len(rows) == 1, "JSONL logger should write one parseable row.")
        _assert(rows[0]["torch_model_checks_skipped"] is True, "JSONL should record torch skip status.")
        manager = CleanCheckpointManager(Path(tmp_dir) / "checkpoints")
        try:
            manager.save(
                modules=None,
                optimizer=None,
                episode=0,
                global_slot=0,
                update_step=0,
                config_snapshot={},
                safe_boundary=False,
            )
        except RuntimeError:
            pass
        except ModuleNotFoundError:
            raise AssertionError("unsafe checkpoint should fail before torch availability is checked.")
        else:
            raise AssertionError("checkpoint save must reject unsafe boundaries.")


def _torch_checks() -> None:
    import torch
    from torch.distributions import Categorical

    from environment.assignment import TemporaryReservationState, build_offloading_candidate_components
    from environment.dag_tasks import TASK_STATE_READY_UNSCHEDULED
    from marl_models.hgnn import CleanIncidenceHGNN
    from marl_models.mappo.clean_movement_actor import CleanMovementActor
    from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
    from marl_models.mappo.clean_ppo import CleanCentralizedCritic, clean_critic_input_dim
    from marl_models.mappo.clean_trainer import CleanPPOUpdater

    torch.manual_seed(53)
    np.random.seed(53)
    original_arrival_prob = config.DAG_BASE_ARRIVAL_PROB
    config.DAG_BASE_ARRIVAL_PROB = 0.0
    try:
        env = Env()
        env.reset()
        _center_scene(env)
        ue = env.ues[0]
        job = env.task_manager.create_dag_for_ue(
            ue_id=ue.id,
            source_pos=ue.pos[:2].copy(),
            current_time_step=env.current_time_seconds,
        )
        ue.enter_service_waiting(job.dag_id)
        env.task_manager.refresh_ready_states()
        graph_builder = CleanGraphBuilder()
        prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
        task_feature_dim = prepared.graph_snapshot.task_features.shape[1]
        embedding_dim = 8
        modules = CleanTrainingModules(
            hgnn=CleanIncidenceHGNN(task_feature_dim=task_feature_dim, hidden_dim=16, output_dim=embedding_dim),
            movement_actor=CleanMovementActor(task_embedding_dim=embedding_dim, hidden_dim=16),
            offloading_actor=CleanOffloadingActor(task_embedding_dim=embedding_dim, hidden_dim=16),
            critic=CleanCentralizedCritic(input_dim=clean_critic_input_dim(embedding_dim, config.NUM_UAVS), hidden_dim=16),
        )
        optimizer = build_single_optimizer(modules, lr=1e-3)
        encoded = encode_prepared_slot(
            prepared_state=prepared,
            env=env,
            hgnn=modules.hgnn,
            critic=modules.critic,
            movement_actor=modules.movement_actor,
        )
        original_time_step = env.time_step
        movement_obs = encoded.movement_observation
        movement_dist = Categorical(logits=encoded.movement_logits)
        selected_movement = movement_dist.sample()
        movement_records: list[CleanMovementRolloutRecord] = []
        movement_actions = {}
        for idx, uav_id in enumerate(movement_obs.uav_ids):
            action = int(selected_movement[idx].item())
            movement_actions[int(uav_id)] = action
            movement_records.append(
                CleanMovementRolloutRecord(
                    uav_id=int(uav_id),
                    uav_index=int(idx),
                    uav_features=movement_obs.uav_features[idx].copy(),
                    ready_task_indices=list(movement_obs.ready_task_indices),
                    pending_task_indices=list(movement_obs.pending_task_indices),
                    ready_count_normalized=float(movement_obs.ready_count_normalized),
                    pending_count_normalized=float(movement_obs.pending_count_normalized),
                    movement_mask=movement_obs.boundary_action_mask[idx].copy(),
                    selected_action=action,
                    old_log_probability=float(movement_dist.log_prob(selected_movement)[idx].detach().cpu().item()),
                    entropy=float(movement_dist.entropy()[idx].detach().cpu().item()),
                )
            )
        env.apply_movement(movement_actions)

        frozen_ready_tasks = [env.task_manager.get_task(task_id) for task_id in prepared.frozen_ready_task_ids]
        frozen_ready_tasks = [task for task in frozen_ready_tasks if task is not None and task.state == TASK_STATE_READY_UNSCHEDULED]
        reservation = TemporaryReservationState.from_executor(env.uavs, env.executor)
        assignment_buffer = modules.offloading_actor.act(
            frozen_ready_tasks=frozen_ready_tasks,
            task_embeddings=encoded.task_embeddings.detach(),
            graph_snapshot=prepared.graph_snapshot,
            task_manager=env.task_manager,
            uavs=env.uavs,
            executor=env.executor,
            current_time_seconds=env.current_time_seconds,
            uav_service_positions=env.uav_service_positions,
            ue_service_positions=env.ue_service_positions,
            ues=env.ues,
            deterministic=False,
        )
        offloading_records: list[CleanOffloadingRolloutRecord] = []
        for actor_record in modules.offloading_actor.latest_records:
            offloading_records.append(
                CleanOffloadingRolloutRecord(
                    task_id=actor_record.task_id,
                    task_local_index=int(actor_record.task_local_index),
                    decision_order=int(actor_record.decision_order),
                    candidate_uav_ids=list(actor_record.candidate_uav_ids),
                    dynamic_uav_features=actor_record.dynamic_uav_features.detach().cpu().numpy().copy(),
                    pair_features=actor_record.pair_features.detach().cpu().numpy().copy(),
                    candidate_mask=actor_record.candidate_mask.detach().cpu().numpy().astype(bool, copy=True),
                    selected_action=int(actor_record.selected_action),
                    selected_uav_id=int(actor_record.selected_uav_id),
                    old_log_probability=float(actor_record.old_log_prob),
                    entropy=float(actor_record.entropy),
                )
            )

        _, _, done, info = env.commit_and_advance(assignment_buffer=assignment_buffer)
        record = make_slot_rollout_record(encoded_state=encoded)
        record.movement_records = movement_records
        record.offloading_records = offloading_records
        record.reward = float(info["step_reward"])
        record.terminated = bool(done)
        record.truncated = False

        no_off_record = make_slot_rollout_record(encoded_state=encoded)
        no_off_record.movement_records = list(movement_records)
        no_off_record.offloading_records = []
        no_off_record.reward = 0.0
        no_off_record.terminated = False
        no_off_record.truncated = True

        buffer = CleanSlotRolloutBuffer()
        buffer.append(no_off_record)
        buffer.append(record)
        next_prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
        next_encoded_old = encode_prepared_slot(
            prepared_state=next_prepared,
            env=env,
            hgnn=modules.hgnn,
            critic=modules.critic,
            movement_actor=modules.movement_actor,
        )
        _assert(env.time_step == original_time_step + 1, "s_{t+1} should be prepared exactly once after slot close.")
        bootstrap_value = close_rollout_with_bootstrap(
            buffer=buffer,
            next_encoded_state=next_encoded_old,
            terminated=False,
        )
        _assert(buffer.closed, "rollout buffer should be closed before PPO update.")
        _assert(buffer.records[-1].next_prepared_state is next_prepared, "buffer should retain the single prepared next state.")
        _assert(np.isclose(buffer.records[-1].bootstrap_value, bootstrap_value), "V_old(s_{t+1}) should be stored for bootstrap.")

        updater = CleanPPOUpdater(
            modules=modules,
            optimizer=optimizer,
            config=CleanPPOUpdateConfig(ppo_epochs=1),
        )
        stats = updater.update(buffer)
        _assert(np.isfinite(stats.total_loss), "total loss should be finite.")
        _assert(np.isfinite(stats.grad_norm), "gradient norm should be finite.")
        _assert(stats.hgnn_grad_norm > 0.0, "actor/critic losses should send gradient back to HGNN.")
        _assert(stats.offloading_effective_slot_count == (1 if offloading_records else 0), "M_t=0 slot should not enter offloading denominator.")

        before_reencode_time = env.time_step
        next_encoded_new = reencode_prepared_after_update(
            prepared_state=next_prepared,
            env=env,
            modules=modules,
        )
        _assert(env.time_step == before_reencode_time, "re-encoding after update must not prepare another slot.")
        _assert(next_encoded_new.prepared_state is next_prepared, "re-encode should reuse the same prepared state object.")

        with _workspace_temp_dir("torch") as tmp_dir:
            logger = CleanJSONLLogger(tmp_dir)
            write_clean_training_log(logger, episode=0, global_slot=env.time_step, info=info, update_stats=stats)
            rows = [json.loads(line) for line in (Path(tmp_dir) / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            _assert(rows and "ppo_total_loss" in rows[-1], "JSONL should include PPO losses.")
            manager = CleanCheckpointManager(Path(tmp_dir) / "checkpoints")
            checkpoint_path = manager.save(
                modules=modules,
                optimizer=optimizer,
                episode=0,
                global_slot=env.time_step,
                update_step=stats.update_step,
                config_snapshot={"smoke": True},
                safe_boundary=buffer.checkpoint_safe,
            )
            _assert(checkpoint_path.exists(), "checkpoint should be saved at safe boundary.")
            payload = manager.load(modules=modules, optimizer=optimizer, path=checkpoint_path)
            _assert(payload["resume_semantics"] == "restart_from_new_episode_only", "checkpoint resume semantics should be explicit.")

        env.apply_movement({})
        env.commit_and_advance(assignments={})
    finally:
        config.DAG_BASE_ARRIVAL_PROB = original_arrival_prob


def main() -> None:
    _non_torch_checks()
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        print("smoke_clean_training_loop non-torch checks passed; torch backward/optimizer checks skipped")
        return
    _torch_checks()
    print("smoke_clean_training_loop passed")


class _workspace_temp_dir:
    def __init__(self, name: str) -> None:
        self.path = ROOT / ".codex_tmp_training_loop" / f"{name}_{os.getpid()}_{np.random.randint(0, 1_000_000)}"

    def __enter__(self) -> str:
        self.path.mkdir(parents=True, exist_ok=True)
        return str(self.path)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
        parent = ROOT / ".codex_tmp_training_loop"
        if parent.exists():
            try:
                if not any(parent.iterdir()):
                    parent.rmdir()
            except PermissionError:
                pass


if __name__ == "__main__":
    main()
