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
    CleanPPOUpdateConfig,
    CleanPPOUpdater,
    CleanTrainingModules,
    build_single_optimizer,
    close_rollout_with_bootstrap,
)


RUNBOOK = ROOT / "docs" / "hyperuav_clean_server_runbook.md"
MINIMAL_TRAINING_COMMAND = (
    "python scripts/train_clean_mainline.py --smoke --episodes 3 "
    "--max-steps-per-episode 20 --rollout-horizon 5 --run-name smoke"
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    _validate_runbook()
    try:
        import torch
    except ModuleNotFoundError:
        print("smoke_clean_server_torch non-torch checks passed; torch validation skipped because torch is not installed")
        return
    _torch_validation(torch)
    print("smoke_clean_server_torch passed")


def _validate_runbook() -> None:
    _assert(RUNBOOK.is_file(), "server runbook should exist.")
    text = RUNBOOK.read_text(encoding="utf-8")
    required = [
        "第一阶段",
        "第二阶段",
        "第三阶段",
        "python scripts/smoke_clean_server_torch.py",
        MINIMAL_TRAINING_COMMAND,
        "python scripts/train_clean_mainline.py --episodes 100",
        "git log -2 --oneline",
        "train_metrics.jsonl",
        "run_summary.json",
        "完整实验",
        "禁止",
        "train_clean_assignment_mappo.py",
        "clean_mappo.py",
        "clean_assignment_policy.py",
    ]
    for item in required:
        _assert(item in text, f"runbook missing required text: {item}")


def _torch_validation(torch) -> None:
    from torch.distributions import Categorical

    from environment.dag_tasks import TASK_STATE_READY_UNSCHEDULED
    from marl_models.hgnn import CleanIncidenceHGNN
    from marl_models.mappo.clean_movement_actor import CleanMovementActor
    from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
    from marl_models.mappo.clean_ppo import CleanCentralizedCritic, clean_critic_input_dim

    np.random.seed(67)
    torch.manual_seed(67)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        json.dumps(
            {
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "selected_device": str(device),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )

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
            current_time_step=env.time_step,
        )
        ue.enter_service_waiting(job.dag_id)
        env.task_manager.refresh_ready_states()

        graph_builder = CleanGraphBuilder()
        prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
        _assert(prepared.graph_snapshot.task_features.ndim == 2, "task_features should be 2D.")
        _assert(prepared.graph_snapshot.incidence_matrix.ndim == 2, "incidence_matrix should be 2D.")

        embedding_dim = 8
        modules = CleanTrainingModules(
            hgnn=CleanIncidenceHGNN(
                task_feature_dim=prepared.graph_snapshot.task_features.shape[1],
                hidden_dim=16,
                output_dim=embedding_dim,
            ).to(device),
            movement_actor=CleanMovementActor(task_embedding_dim=embedding_dim, hidden_dim=16).to(device),
            offloading_actor=CleanOffloadingActor(task_embedding_dim=embedding_dim, hidden_dim=16).to(device),
            critic=CleanCentralizedCritic(input_dim=clean_critic_input_dim(embedding_dim, config.NUM_UAVS), hidden_dim=16).to(device),
        )
        optimizer = build_single_optimizer(modules, lr=1e-3)
        encoded = encode_prepared_slot(
            prepared_state=prepared,
            env=env,
            hgnn=modules.hgnn,
            critic=modules.critic,
            movement_actor=modules.movement_actor,
            device=device,
        )
        _assert(encoded.task_embeddings.device == device, "HGNN task embeddings should be on selected device.")
        _assert(torch.isfinite(encoded.task_embeddings).all().item(), "HGNN embeddings should be finite.")
        _assert(torch.isfinite(encoded.movement_logits).all().item(), "movement logits should be finite after masking.")
        movement_mask = torch.as_tensor(encoded.movement_observation.boundary_action_mask, dtype=torch.bool, device=device)
        _assert(movement_mask.dtype == torch.bool, "movement boundary mask must be bool.")
        _assert(movement_mask.shape == encoded.movement_logits.shape, "movement mask shape must match logits.")
        _assert(encoded.value is not None and np.isfinite(encoded.value), "centralized critic value should be finite.")

        movement_dist = Categorical(logits=encoded.movement_logits)
        selected_movement = movement_dist.sample()
        movement_log_probs = movement_dist.log_prob(selected_movement)
        movement_entropy = movement_dist.entropy()
        movement_actions: dict[int, int] = {}
        movement_records: list[CleanMovementRolloutRecord] = []
        for idx, uav_id in enumerate(encoded.movement_observation.uav_ids):
            action = int(selected_movement[idx].item())
            movement_actions[int(uav_id)] = action
            movement_records.append(
                CleanMovementRolloutRecord(
                    uav_id=int(uav_id),
                    uav_index=int(idx),
                    uav_features=encoded.movement_observation.uav_features[idx].copy(),
                    ready_task_indices=list(encoded.movement_observation.ready_task_indices),
                    pending_task_indices=list(encoded.movement_observation.pending_task_indices),
                    ready_count_normalized=float(encoded.movement_observation.ready_count_normalized),
                    pending_count_normalized=float(encoded.movement_observation.pending_count_normalized),
                    movement_mask=encoded.movement_observation.boundary_action_mask[idx].copy(),
                    selected_action=action,
                    old_log_probability=float(movement_log_probs[idx].detach().cpu().item()),
                    entropy=float(movement_entropy[idx].detach().cpu().item()),
                )
            )
        env.apply_movement(movement_actions)

        ready_tasks = [env.task_manager.get_task(task_id) for task_id in prepared.frozen_ready_task_ids]
        ready_tasks = [task for task in ready_tasks if task is not None and task.state == TASK_STATE_READY_UNSCHEDULED]
        assignment_buffer = modules.offloading_actor.act(
            frozen_ready_tasks=ready_tasks,
            task_embeddings=encoded.task_embeddings.detach(),
            graph_snapshot=prepared.graph_snapshot,
            task_manager=env.task_manager,
            uavs=env.uavs,
            executor=env.executor,
            current_time_step=env.time_step,
            uav_service_positions=env.uav_service_positions,
            ue_service_positions=env.ue_service_positions,
            ues=env.ues,
            deterministic=False,
        )
        _assert(modules.offloading_actor.latest_records, "offloading scorer should produce at least one action record.")
        offloading_records: list[CleanOffloadingRolloutRecord] = []
        for record in modules.offloading_actor.latest_records:
            _assert(record.candidate_mask.dtype == torch.bool, "candidate mask tensor must be bool.")
            _assert(record.candidate_features.device.type == "cpu", "stored candidate_features should be CPU historical copy.")
            scorer_logits = modules.offloading_actor.scorer(record.candidate_features.to(device))
            _assert(torch.isfinite(scorer_logits).all().item(), "offloading scorer logits should be finite.")
            offloading_records.append(
                CleanOffloadingRolloutRecord(
                    task_id=record.task_id,
                    task_local_index=int(record.task_local_index),
                    decision_order=int(record.decision_order),
                    candidate_uav_ids=list(record.candidate_uav_ids),
                    dynamic_uav_features=record.dynamic_uav_features.detach().cpu().numpy().copy(),
                    pair_features=record.pair_features.detach().cpu().numpy().copy(),
                    candidate_mask=record.candidate_mask.detach().cpu().numpy().astype(bool, copy=True),
                    selected_action=int(record.selected_action),
                    selected_uav_id=int(record.selected_uav_id),
                    old_log_probability=float(record.old_log_prob),
                    entropy=float(record.entropy),
                )
            )

        _, _, done, info = env.commit_and_advance(assignment_buffer=assignment_buffer)
        slot_record = make_slot_rollout_record(encoded_state=encoded)
        slot_record.movement_records = movement_records
        slot_record.offloading_records = offloading_records
        slot_record.reward = float(info["step_reward"])
        slot_record.terminated = bool(done)
        slot_record.truncated = False

        buffer = CleanSlotRolloutBuffer()
        buffer.append(slot_record)
        next_prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
        next_encoded = encode_prepared_slot(
            prepared_state=next_prepared,
            env=env,
            hgnn=modules.hgnn,
            critic=modules.critic,
            movement_actor=modules.movement_actor,
            device=device,
        )
        bootstrap_value = close_rollout_with_bootstrap(buffer=buffer, next_encoded_state=next_encoded, terminated=False)
        _assert(np.isfinite(bootstrap_value), "bootstrap value should be finite.")

        updater = CleanPPOUpdater(
            modules=modules,
            optimizer=optimizer,
            config=CleanPPOUpdateConfig(ppo_epochs=1),
            device=device,
        )
        stats = updater.update(buffer)
        for value in [
            stats.movement_loss,
            stats.offloading_loss,
            stats.value_loss,
            stats.total_loss,
            stats.grad_norm,
            stats.hgnn_grad_norm,
        ]:
            _assert(np.isfinite(value), "PPO loss/gradient stats should be finite.")
        _assert(stats.hgnn_grad_norm > 0.0, "tiny backward should send gradient into HGNN.")
        _assert(stats.offloading_effective_slot_count == 1, "offloading loss denominator should include only M_t > 0 slot.")

        with _workspace_temp_dir("server_torch") as tmp_dir:
            manager = CleanCheckpointManager(Path(tmp_dir) / "checkpoints")
            checkpoint_path = manager.save(
                modules=modules,
                optimizer=optimizer,
                episode=0,
                global_slot=env.time_step,
                update_step=stats.update_step,
                config_snapshot={"server_torch_smoke": True},
                safe_boundary=buffer.checkpoint_safe,
            )
            _assert(checkpoint_path.exists(), "checkpoint save should create latest.pt.")
            payload = manager.load(modules=modules, optimizer=optimizer, path=checkpoint_path)
            _assert(payload["resume_semantics"] == "restart_from_new_episode_only", "checkpoint resume semantics should be explicit.")

        print(
            json.dumps(
                {
                    "task_features_shape": list(prepared.graph_snapshot.task_features.shape),
                    "incidence_shape": list(prepared.graph_snapshot.incidence_matrix.shape),
                    "movement_logits_shape": list(encoded.movement_logits.shape),
                    "movement_mask_dtype": str(movement_mask.dtype),
                    "offloading_action_count": len(offloading_records),
                    "ppo_total_loss": stats.total_loss,
                    "hgnn_grad_norm": stats.hgnn_grad_norm,
                    "device": str(device),
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
    finally:
        config.DAG_BASE_ARRIVAL_PROB = original_arrival_prob


def _center_scene(env: Env) -> None:
    center = np.array([250.0, 250.0], dtype=np.float32)
    env.hotspot_center = center.copy()
    for idx, uav in enumerate(env.uavs):
        uav.pos[:2] = center + np.array([float(idx), 0.0], dtype=np.float32)
    for ue in env.ues:
        ue.pos[:2] = center.copy()


class _workspace_temp_dir:
    def __init__(self, name: str) -> None:
        self.path = ROOT / ".codex_tmp_server_torch" / f"{name}_{os.getpid()}_{np.random.randint(0, 1_000_000)}"

    def __enter__(self) -> str:
        self.path.mkdir(parents=True, exist_ok=True)
        return str(self.path)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
        parent = ROOT / ".codex_tmp_server_torch"
        if parent.exists():
            try:
                if not any(parent.iterdir()):
                    parent.rmdir()
            except PermissionError:
                pass


if __name__ == "__main__":
    main()
