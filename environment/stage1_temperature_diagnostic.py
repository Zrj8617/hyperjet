from __future__ import annotations

from typing import Any

import config
import numpy as np

from environment.env import Env
from environment.stage1_temperature_tape import EPISODE_SLOTS, isolated_rng, instantiate_potential_template, potential_template_at, validate_scenario_shard
from environment.stage1_temperature_sampling import canonical_sha256, deterministic_masked_argmax, file_sha256, keyed_temperature_sample


FROZEN_CHECKPOINTS = {
    42: ("logs/decision_ppo_bandit/20260729_215923_stage1_formal_S1-B_seed42/checkpoints/checkpoint_update_0030.pt", "b4f0b84afa5ad443e901ceaa58bdd49f10fe4570655d7ca89f65812ead78c668"),
    86: ("logs/decision_ppo_bandit/20260729_220604_stage1_formal_S1-B_seed86/checkpoints/checkpoint_update_0030.pt", "abb7c3c09d61be51860c7020e2555339982b042b4b8de260546e112123a2d643"),
    1042: ("logs/decision_ppo_bandit/20260729_221421_stage1_formal_S1-B_seed1042/checkpoints/checkpoint_update_0030.pt", "95d7f1a13dbc671e214ef9162d89c9768a58ab4bc8df13e712817f5931804ac9"),
}

# Populated after long-v1 training: {seed: (relative_path, sha256, completed_update)}
FROZEN_CHECKPOINTS_LONG: dict[int, tuple[str, str, int]] = {
    42: ("logs/decision_ppo_bandit/20260807_124339_stage1_long_v1_S1-B_seed42/checkpoints/checkpoint_update_0300.pt", "d0e0dd573ef10c641e39d23bfdc176ddc1c7a5d58e68bba2130a3d460789b96a", 300),
    86: ("logs/decision_ppo_bandit/20260807_124341_stage1_long_v1_S1-B_seed86/checkpoints/checkpoint_update_0300.pt", "d838934161c7b3f7fbfec265af3fd25155489e8012a4b335653b65322235d155", 300),
    1042: ("logs/decision_ppo_bandit/20260807_124341_stage1_long_v1_S1-B_seed1042/checkpoints/checkpoint_update_0300.pt", "d69c190c027b9017984b6d81291906bfc44f75c201905ef817fe0c4ce467a4b1", 300),
}

# Populated after B1 training: {seed: (relative_path, sha256, completed_update)}
FROZEN_CHECKPOINTS_B1: dict[int, tuple[str, str, int]] = {
    42: ("logs/decision_ppo_bandit/20260808_115532_stage1_b1_softnorm_v1_S1-B_seed42/checkpoints/checkpoint_update_0300.pt", "303e0fde1b5af722420c7f1b3e006406fd6198f1994ec65d1580987bac610440", 300),
    86: ("logs/decision_ppo_bandit/20260808_115532_stage1_b1_softnorm_v1_S1-B_seed86/checkpoints/checkpoint_update_0300.pt", "0c1eabef827ef1144008edb1f09a5e777ca932bd28f152b14391e92745c3e22a", 300),
    1042: ("logs/decision_ppo_bandit/20260808_115532_stage1_b1_softnorm_v1_S1-B_seed1042/checkpoints/checkpoint_update_0300.pt", "88f41c93fd02c3685c5f2ffff70b213d59bb8a5dafaf4e0e143d423705d60c95", 300),
}

# B1 diagnostic sweep checkpoints: key=(training_seed, completed_update), value=(relative_path, sha256).
B1_SWEEP_CHECKPOINTS: dict[tuple[int, int], tuple[str, str]] = {
    (42, 0): ("logs/decision_ppo_bandit/20260808_115532_stage1_b1_softnorm_v1_S1-B_seed42/checkpoints/checkpoint_update_0000.pt", "731d22f99aa1033fbb0ea4675e1514314fd8d6ee2b19ffb520debffd63dc96e9"),
    (42, 30): ("logs/decision_ppo_bandit/20260808_115532_stage1_b1_softnorm_v1_S1-B_seed42/checkpoints/checkpoint_update_0030.pt", "deee4c9bb11eb36be3227306a92870308d3a5d351522fc5c1cffae4a486fad97"),
    (42, 100): ("logs/decision_ppo_bandit/20260808_115532_stage1_b1_softnorm_v1_S1-B_seed42/checkpoints/checkpoint_update_0100.pt", "84f611556e337063c3d46d0f1836357fc1c4eeccf664233476b8cab402d7d74c"),
    (42, 200): ("logs/decision_ppo_bandit/20260808_115532_stage1_b1_softnorm_v1_S1-B_seed42/checkpoints/checkpoint_update_0200.pt", "9f855c2e321df8bc3bcb9e2dc922281356e9663418189a80cc6391fc7905f377"),
    (86, 0): ("logs/decision_ppo_bandit/20260808_115532_stage1_b1_softnorm_v1_S1-B_seed86/checkpoints/checkpoint_update_0000.pt", "855e651bc45d0872eac345a8893f5283a42b5bb67b58316c3e1a3aaedf598b84"),
    (86, 30): ("logs/decision_ppo_bandit/20260808_115532_stage1_b1_softnorm_v1_S1-B_seed86/checkpoints/checkpoint_update_0030.pt", "526f567f8fdb177d114fe10febdc8ea11ab747e7b9884eebadb78343da3d917f"),
    (86, 100): ("logs/decision_ppo_bandit/20260808_115532_stage1_b1_softnorm_v1_S1-B_seed86/checkpoints/checkpoint_update_0100.pt", "a467747b10eef6dcfa2372185104083be876dab4bd76500840924d0ad78d65d9"),
    (86, 200): ("logs/decision_ppo_bandit/20260808_115532_stage1_b1_softnorm_v1_S1-B_seed86/checkpoints/checkpoint_update_0200.pt", "80f5b741b717fc40075dbedd0e28123137b086bddff2134e1410661d2a4d5d4d"),
    (1042, 0): ("logs/decision_ppo_bandit/20260808_115532_stage1_b1_softnorm_v1_S1-B_seed1042/checkpoints/checkpoint_update_0000.pt", "7b263c3a182b575356dc6499140b766aa731a6f19d0e3352fb544f6267f26b03"),
    (1042, 30): ("logs/decision_ppo_bandit/20260808_115532_stage1_b1_softnorm_v1_S1-B_seed1042/checkpoints/checkpoint_update_0030.pt", "28b504735dc57696326427417b0a0d217fd6d122315cfe75835fd755770999e9"),
    (1042, 100): ("logs/decision_ppo_bandit/20260808_115532_stage1_b1_softnorm_v1_S1-B_seed1042/checkpoints/checkpoint_update_0100.pt", "5f2c59a5613e900e3be04e45e10b9ebb0eb231142d981a76618ca289b72607f6"),
    (1042, 200): ("logs/decision_ppo_bandit/20260808_115532_stage1_b1_softnorm_v1_S1-B_seed1042/checkpoints/checkpoint_update_0200.pt", "68b8a7e8d93a33e00e6006391fc7ab7e0c511b6bf2d16ad0782081d3844d4a07"),
}

CHECKPOINT_SETS = {
    "formal_v1": FROZEN_CHECKPOINTS,
    "long_v1": FROZEN_CHECKPOINTS_LONG,
    "b1_v1": FROZEN_CHECKPOINTS_B1,
    "b1_sweep": B1_SWEEP_CHECKPOINTS,
}


class Stage1TemperatureDiagnosticEnv(Env):
    """Original Stage 1 A semantics driven only by frozen random material."""

    def __init__(self, *, scenario_shard: dict[str, Any]) -> None:
        validate_scenario_shard(scenario_shard)
        if int(config.CLEAN_MAX_QUEUE_PER_UAV) != 16:
            raise ValueError("Stage 1 A requires CLEAN_MAX_QUEUE_PER_UAV == 16")
        self.scenario_shard = scenario_shard
        self.evaluation_scenario_seed = int(scenario_shard["evaluation_scenario_seed"])
        self.generated_dag_count = 0
        super().__init__(completed_dag_weight=16.0, freeze_ue_mobility=False, max_active_dags_per_ue=1)

    def reset(self) -> list[np.ndarray]:
        with isolated_rng(self.evaluation_scenario_seed):
            super().reset()
        self._bind_initial_random_material()
        self.generated_dag_count = 0
        self._latest_info["generated_dag_count"] = 0
        self._latest_info["admitted_dag_count"] = 0
        return self._get_obs()

    def _bind_initial_random_material(self) -> None:
        radius = float(config.HOTSPOT_RADIUS)
        hotspot_uniforms = self.scenario_shard["hotspot_center_uniforms"]
        self.hotspot_center = np.asarray([
            radius + float(hotspot_uniforms[0]) * (float(config.AREA_WIDTH) - 2.0 * radius),
            radius + float(hotspot_uniforms[1]) * (float(config.AREA_HEIGHT) - 2.0 * radius),
        ], dtype=np.float32)
        speed_min = float(config.UE_GM_MIN_SPEED)
        speed_max = max(float(config.UE_WALK_SPEED_MEAN) * 2.0, speed_min)
        for ue, uniforms in zip(self._ues, self.scenario_shard["ue_initial_uniforms"], strict=True):
            ue.pos = np.asarray([float(uniforms[0]) * float(config.AREA_WIDTH), float(uniforms[1]) * float(config.AREA_HEIGHT), 0.0], dtype=np.float32)
            ue.speed = speed_min + float(uniforms[2]) * (speed_max - speed_min)
            ue.theta = float(uniforms[3]) * 2.0 * np.pi
            ue.velocity = ue._velocity_from_polar()
            ue.service_waiting = False
            ue.active_dag_id = None
        for uav, uniforms in zip(self._uavs, self.scenario_shard["uav_position_uniforms"], strict=True):
            uav.pos = np.asarray([float(uniforms[0]) * float(config.AREA_WIDTH), float(uniforms[1]) * float(config.AREA_HEIGHT), float(config.UAV_ALTITUDE)], dtype=np.float32)
        self._initial_hotspot_ue_count = sum(int(ue.is_inside_hotspot(self.hotspot_center, self.hotspot_radius)) for ue in self._ues)
        self._ue_service_positions = {int(ue.id): ue.pos[:2].copy() for ue in self._ues}
        self._uav_pre_move_positions = {int(uav.id): uav.pos[:2].copy() for uav in self._uavs}
        self._uav_service_positions = {int(uav.id): uav.pos[:2].copy() for uav in self._uavs}
        self._executor.reset(self._uavs)
        self._metrics.reset([uav.id for uav in self._uavs])
        self._latest_info["initial_hotspot_ue_count"] = int(self._initial_hotspot_ue_count)

    def _advance_ues_for_slot(self, slot_index: int) -> None:
        if not 0 <= int(slot_index) < EPISODE_SLOTS:
            raise IndexError("temperature diagnostic slot outside frozen 0..199 range")
        innovations = self.scenario_shard["ue_mobility_standard_normals"][int(slot_index)]
        for ue, pair in zip(self._ues, innovations, strict=True):
            ue.update_position(commit_position=True, speed_standard_normal=float(pair[0]), theta_standard_normal=float(pair[1]))

    def _process_clean_dag_arrivals(self) -> int:
        if not self._slot_service_positions_frozen:
            self._ue_service_positions = {int(ue.id): ue.pos[:2].copy() for ue in self._ues}
        slot_index = int(self._time_step) - 1
        if not 0 <= slot_index < EPISODE_SLOTS:
            raise IndexError("temperature diagnostic arrival slot outside frozen 0..199 range")
        version_before = self._task_manager.dag_arrival_version
        created_count = 0
        funnel = self._empty_arrival_funnel()
        arrival_uniforms = self.scenario_shard["arrival_uniforms"][slot_index]
        for ue in self._ues:
            ue_id = int(ue.id)
            funnel["arrival_attempt_count"] += 1
            if not self._task_manager.can_accept_dag_for_ue(ue_id):
                funnel["arrival_blocked_count"] += 1
                funnel["arrival_blocked_reasons"]["active_dag_cap"] += 1
                continue
            funnel["arrival_draw_count"] += 1
            arrival_probability = ue.get_arrival_probability(self.hotspot_center, self.hotspot_radius)
            if float(arrival_uniforms[ue_id]) >= float(arrival_probability):
                funnel["arrival_no_event_count"] += 1
                continue
            funnel["arrival_sampled_event_count"] += 1
            template = potential_template_at(self.scenario_shard, slot_index, ue_id)
            job = instantiate_potential_template(self._task_manager, template, source_pos=self._ue_service_positions.get(ue_id, ue.pos[:2]).copy(), arrival_time=self.current_time_seconds)
            ue.enter_service_waiting(job.dag_id)
            created_count += 1
            self.generated_dag_count += 1
            funnel["arrival_admitted_count"] += 1
        version_after = self._task_manager.dag_arrival_version
        self._latest_arrival_funnel = funnel
        self._accumulate_arrival_funnel(funnel)
        self._last_new_dag_arrived = version_after > version_before or created_count > 0
        self._latest_dag_arrival_version = version_after
        return created_count

    def arrival_identity_metrics(self) -> dict[str, int]:
        admitted = int(self._arrival_funnel_totals["arrival_admitted_count"])
        if admitted != int(self.generated_dag_count):
            raise AssertionError("admitted_dag_count alias must equal generated_dag_count")
        return {"generated_dag_count": int(self.generated_dag_count), "admitted_dag_count": admitted, "arrival_blocked_count": int(self._arrival_funnel_totals["arrival_blocked_count"])}


def load_frozen_checkpoint(
    checkpoint_path: Any,
    *,
    training_seed: int,
    device: str = "cuda",
    registry: Any = None,
    expected_completed_update: int = 30,
) -> tuple[Any, Any, dict[str, Any]]:
    """Strictly load one frozen MLP checkpoint after an isolated real-graph probe."""
    import random
    import torch
    from environment.graph_builder import CleanGraphBuilder
    from marl_models.hgnn import build_clean_task_encoder
    from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
    from marl_models.mappo.clean_slot_orchestrator import prepare_slot_state
    from scripts.train_decision_ppo_bandit_gate import CHECKPOINT_SCHEMA

    seed = int(training_seed)
    registry = FROZEN_CHECKPOINTS if registry is None else registry
    if seed not in registry:
        raise ValueError("checkpoint training seed is not frozen")
    entry = registry[seed]
    expected_relative, expected_hash = entry[0], entry[1]
    path = __import__("pathlib").Path(checkpoint_path).resolve()
    if path.as_posix().lower().endswith(expected_relative.lower()) is False:
        raise ValueError("checkpoint path does not match frozen seed mapping")
    actual_hash = file_sha256(path)
    if actual_hash != expected_hash:
        raise ValueError("checkpoint SHA-256 mismatch")
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    builder = CleanGraphBuilder()
    try:
        probe_env = Env(completed_dag_weight=16.0, freeze_ue_mobility=False, max_active_dags_per_ue=1)
        probe_env.reset(); builder.reset()
        probe = prepare_slot_state(env=probe_env, graph_builder=builder)
        graph_dim = int(probe.graph_snapshot.task_features.shape[1])
    finally:
        builder.close()
        random.setstate(python_state); np.random.set_state(numpy_state); torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != CHECKPOINT_SCHEMA or payload.get("group") != "S1-B" or int(payload.get("seed", -1)) != seed or int(payload.get("completed_update", -1)) != int(expected_completed_update):
        raise ValueError("checkpoint metadata mismatch")
    checkpoint_dim = int(payload["encoder_state_dict"]["input_proj.weight"].shape[1])
    if checkpoint_dim != graph_dim or graph_dim != 12:
        raise ValueError(f"checkpoint_dim={checkpoint_dim} graph_snapshot_dim={graph_dim}; expected both 12")
    cfg = payload.get("config", {})
    if str(cfg.get("encoder")) != "mlp":
        raise ValueError("frozen checkpoint must use MLP encoder")
    resolved_device = torch.device(device)
    encoder = build_clean_task_encoder(encoder_type="mlp", task_feature_dim=graph_dim, hidden_dim=int(cfg["hidden_dim"]), output_dim=int(cfg["task_embedding_dim"])).to(resolved_device)
    actor = CleanOffloadingActor(task_embedding_dim=int(cfg["task_embedding_dim"]), hidden_dim=int(cfg["hidden_dim"])).to(resolved_device)
    encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    actor.scorer.load_state_dict(payload["scorer_state_dict"], strict=True)
    encoder.eval(); actor.eval()
    return encoder, actor, {"checkpoint_path": str(path), "checkpoint_sha256": actual_hash, "training_seed": seed, "completed_update": int(expected_completed_update), "checkpoint_task_feature_dim": checkpoint_dim, "graph_snapshot_task_feature_dim": graph_dim, "resolved_task_feature_dim": graph_dim, "encoder_strict_load_pass": True, "scorer_strict_load_pass": True}


def act_with_temperature(*, actor: Any, frozen_ready_tasks: list[Any], task_embeddings: Any, graph_snapshot: Any, task_manager: Any, uavs: list[Any], executor: Any, current_time_seconds: float, uav_service_positions: dict[int, Any], ue_service_positions: dict[int, Any], ues: list[Any], temperature: float, checkpoint_sha256: str, evaluation_scenario_seed: int, slot_index: int, sampling_replicate: int, record_static: bool = False) -> tuple[Any, list[dict[str, Any]], int]:
    """Diagnostic-only copy of the frozen sequential actor loop with keyed sampling."""
    import torch
    from environment.assignment import CLEAN_OFFLOADING_UAV_FEATURE_DIM, CleanAssignmentBuffer, TemporaryReservationState, build_offloading_candidate_components
    if int(CLEAN_OFFLOADING_UAV_FEATURE_DIM) != 7:
        raise ValueError("seven-dimensional actor UAV input changed")
    device = next(actor.parameters()).device
    embedding_tensor = torch.as_tensor(task_embeddings, dtype=torch.float32, device=device)
    reservation = TemporaryReservationState.from_executor(uavs, executor)
    assignments, records, skips = CleanAssignmentBuffer(), [], 0
    for decision_order, task in enumerate(frozen_ready_tasks):
        task_idx = graph_snapshot.task_id_to_idx.get(task.task_id)
        if task_idx is None:
            skips += 1; continue
        dynamic, pair, mask, uav_ids, estimates = build_offloading_candidate_components(task=task, uavs=uavs, task_manager=task_manager, executor=executor, state_view=reservation, current_time_seconds=float(current_time_seconds), uav_service_positions=uav_service_positions, ue_service_positions=ue_service_positions, ues=ues)
        if dynamic.shape[0] == 0 or not bool(mask.any()):
            skips += 1; continue
        embedding = embedding_tensor[int(task_idx)].detach().cpu().numpy().reshape(1, -1)
        features = np.concatenate([np.repeat(embedding, dynamic.shape[0], axis=0), dynamic, pair], axis=1).astype(np.float32)
        with torch.no_grad():
            logits = actor.scorer(torch.as_tensor(features, dtype=torch.float32, device=device)).detach().cpu().numpy().astype(np.float64)
        sample = keyed_temperature_sample(logits=logits, candidate_mask=mask, candidate_uav_ids=uav_ids, temperature=temperature, checkpoint_sha256=checkpoint_sha256, evaluation_scenario_seed=evaluation_scenario_seed, slot_index=slot_index, stable_task_id=str(task.task_id), decision_order=decision_order, sampling_replicate=sampling_replicate)
        selected, estimate = int(sample.selected_index), estimates[int(sample.selected_index)]
        assignments.append(task.task_id, int(sample.selected_uav_id), decision_order)
        reservation.reserve(task.task_id, int(sample.selected_uav_id), estimated_available_time=estimate.estimated_finish_time, estimated_queued_workload=estimate.estimated_queued_workload)
        if record_static and int(np.count_nonzero(mask)) >= 2:
            eft = [float(value.estimated_finish_time) for value in estimates]
            deterministic_index = deterministic_masked_argmax(logits, mask, uav_ids)
            legal_indices = np.flatnonzero(mask); minimum_eft = float(np.min(np.asarray(eft)[legal_indices])); greedy_indices = [int(index) for index in legal_indices if float(eft[index]) == minimum_eft]; greedy_index = min(greedy_indices, key=lambda index: int(uav_ids[index]))
            records.append({"checkpoint_sha256": checkpoint_sha256, "evaluation_scenario_seed": int(evaluation_scenario_seed), "episode_index": int(evaluation_scenario_seed) - 424242, "slot_index": int(slot_index), "stable_task_id": str(task.task_id), "decision_order": int(decision_order), "sampling_replicate": int(sampling_replicate), "candidate_uav_ids": [int(value) for value in uav_ids], "candidate_mask": [bool(value) for value in mask], "raw_logits": [float(value) for value in logits], "eft": eft, "gumbels": [float(value) for value in sample.gumbels], "sampled_uav_id_t1": int(sample.selected_uav_id) if float(temperature) == 1.0 else None, "deterministic_actor_uav_id": int(uav_ids[deterministic_index]), "greedy_eft_uav_id": int(uav_ids[greedy_index]), "actor_input_sha256": canonical_sha256(features.tolist())})
    return assignments, records, skips
