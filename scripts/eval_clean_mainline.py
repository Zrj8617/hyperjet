from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.env import Env
from environment.graph_builder import CleanGraphBuilder
from marl_models.mappo.clean_slot_orchestrator import encode_prepared_slot, prepare_slot_state
from marl_models.mappo.clean_trainer import CleanTrainingModules
from scripts.offloading_policy_gate import (
    OFFLOADING_GATE_SCHEMA_VERSION,
    OFFLOADING_POLICIES,
    RANDOM_HASH_VERSION,
    finalize_decision_realizations,
    select_eval_offloading_actions,
    summarize_offloading_decisions,
)
from scripts.train_clean_mainline import checkpoint_experiment_controls, _movement_action_distribution


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic HyperUAV clean mainline evaluation.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--arrival-steps", type=int, default=200)
    parser.add_argument("--max-drain-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if _torch_cuda_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("logs") / "clean_eval")
    parser.add_argument("--run-name", type=str, default="eval")
    parser.add_argument(
        "--offloading-policy",
        choices=OFFLOADING_POLICIES,
        default="actor_argmax",
        help="Evaluation-only offloading selector; actor_argmax preserves the legacy default.",
    )
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no-render", action="store_true", default=True)
    parser.add_argument(
        "--freeze-movement",
        action="store_true",
        default=False,
        help="Force every UAV to hover throughout arrival and drain evaluation. "
        "Use this when evaluating checkpoints trained with --freeze-movement.",
    )
    parser.add_argument(
        "--freeze-ue-mobility",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Inherit UE mobility mode from the checkpoint; an explicit conflicting override is rejected.",
    )
    parser.add_argument("--task-embedding-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    return parser


def _resolve_eval_ue_mobility(
    args: argparse.Namespace,
    experiment_controls: dict[str, Any],
) -> bool:
    checkpoint_freeze = experiment_controls.get("freeze_ue_mobility", False)
    if not isinstance(checkpoint_freeze, bool):
        raise ValueError("checkpoint freeze_ue_mobility must be boolean")
    requested = getattr(args, "freeze_ue_mobility", None)
    if requested is not None and bool(requested) != checkpoint_freeze:
        raise ValueError(
            "evaluation checkpoint UE mobility mismatch: "
            f"requested freeze={bool(requested)}, checkpoint freeze={checkpoint_freeze}"
        )
    return checkpoint_freeze


def create_eval_run_directory(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(args.run_name)).strip("_")
    tag = clean_name or "eval"
    run_id = f"{timestamp}_{tag}_seed{int(args.seed)}"
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_eval_config(
    args: argparse.Namespace,
    experiment_controls: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if experiment_controls is None:
        controls = {
            "completed_dag_weight": float(config.REWARD_COMPLETED_DAG_WEIGHT),
            "detach_critic_hgnn": False,
            "freeze_ue_mobility": False,
            "offloading_counterfactual_coef": 0.0,
            "offloading_action_value_loss_coef": 0.0,
            "offloading_lagged_q_coef": 0.0,
            "offloading_lagged_q_loss_coef": 0.0,
            "offloading_lagged_q_scale_seconds": 200.0,
            "offloading_lagged_q_censor_weight": 0.25,
            "task_encoder": "hgnn",
        }
        requested = getattr(args, "freeze_ue_mobility", None)
        freeze_ue_mobility = False if requested is None else bool(requested)
        controls["freeze_ue_mobility"] = freeze_ue_mobility
    else:
        controls = experiment_controls
        freeze_ue_mobility = _resolve_eval_ue_mobility(args, controls)
    return {
        "cli": _namespace_to_dict(args),
        "checkpoint_experiment_controls": dict(controls),
        "protocol": {
            "arrival_phase": (
                "fixed episode-initial UE positions and DAG arrivals"
                if freeze_ue_mobility
                else "normal UE movement and DAG arrivals"
            ),
            "drain_phase": (
                "UE positions remain fixed; DAG arrivals disabled; executor continues"
                if freeze_ue_mobility
                else "UE movement continues; DAG arrivals disabled; executor continues"
            ),
            "throughput_denominator": "total_executed_slots * TIME_SLOT_DURATION",
            "default_action": "masked_argmax_deterministic",
            "offloading_policy": str(args.offloading_policy),
            "offloading_gate_schema_version": int(OFFLOADING_GATE_SCHEMA_VERSION),
            "random_hash_version": RANDOM_HASH_VERSION if str(args.offloading_policy) == "random_hash" else None,
            "estimator_error_convention": "realized_final_finish_time - selected_estimated_finish",
            "closed_loop_pairing": (
                "shared seeds pair initial random conditions only; different offloading choices may change "
                "later arrivals, queues, movement observations, and trajectories"
            ),
            "movement_mode": "forced_hover" if bool(args.freeze_movement) else "masked_argmax_deterministic",
            "ue_mobility_mode": "fixed" if freeze_ue_mobility else "moving",
        },
        "clean_scene": {
            "AREA_WIDTH": config.AREA_WIDTH,
            "AREA_HEIGHT": config.AREA_HEIGHT,
            "NUM_UAVS": config.NUM_UAVS,
            "NUM_UES": config.NUM_UES,
            "TIME_SLOT_DURATION": config.TIME_SLOT_DURATION,
            "HOTSPOT_RADIUS": config.HOTSPOT_RADIUS,
        },
        "kahypar": {
            "enabled": bool(config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES),
            "package_version_required": "1.3.7",
            "ini_relative_path": str(config.KAHYPAR_INI_RELATIVE_PATH),
            "seed": int(config.KAHYPAR_SEED),
            "epsilon": float(config.KAHYPAR_EPSILON),
            "worker_timeout_seconds": float(config.KAHYPAR_WORKER_TIMEOUT_SECONDS),
            "max_consecutive_failures": int(config.KAHYPAR_MAX_CONSECUTIVE_FAILURES),
        },
    }


def initialize_eval_files(
    run_dir: Path,
    args: argparse.Namespace,
    experiment_controls: dict[str, Any] | None = None,
) -> None:
    if experiment_controls is None:
        controls = {
            "completed_dag_weight": float(config.REWARD_COMPLETED_DAG_WEIGHT),
            "detach_critic_hgnn": False,
            "freeze_ue_mobility": False,
            "offloading_counterfactual_coef": 0.0,
            "offloading_action_value_loss_coef": 0.0,
            "offloading_lagged_q_coef": 0.0,
            "offloading_lagged_q_loss_coef": 0.0,
            "offloading_lagged_q_scale_seconds": 200.0,
            "offloading_lagged_q_censor_weight": 0.25,
        }
        requested = getattr(args, "freeze_ue_mobility", None)
        freeze_ue_mobility = False if requested is None else bool(requested)
        controls["freeze_ue_mobility"] = freeze_ue_mobility
    else:
        controls = experiment_controls
        freeze_ue_mobility = _resolve_eval_ue_mobility(args, controls)
    _write_json(run_dir / "config.json", build_eval_config(args, experiment_controls))
    (run_dir / "offloading_decisions.jsonl").touch(exist_ok=True)
    _write_json(
        run_dir / "eval_summary.json",
        {
            "status": "initialized",
            "run_dir": str(run_dir),
            "torch_required_for_evaluation": True,
            "deterministic": bool(args.deterministic),
            "arrival_steps": int(args.arrival_steps),
            "max_drain_steps": int(args.max_drain_steps),
            "completed_dag_weight": float(controls["completed_dag_weight"]),
            "detach_critic_hgnn": bool(controls["detach_critic_hgnn"]),
            "freeze_ue_mobility": freeze_ue_mobility,
            "offloading_counterfactual_coef": float(controls["offloading_counterfactual_coef"]),
            "offloading_action_value_loss_coef": float(controls["offloading_action_value_loss_coef"]),
            "offloading_lagged_q_coef": float(controls["offloading_lagged_q_coef"]),
            "offloading_lagged_q_loss_coef": float(controls["offloading_lagged_q_loss_coef"]),
            "offloading_lagged_q_scale_seconds": float(controls["offloading_lagged_q_scale_seconds"]),
            "offloading_lagged_q_censor_weight": float(controls["offloading_lagged_q_censor_weight"]),
            "offloading_policy": str(args.offloading_policy),
            "git_commit": _git_commit(),
        },
    )


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = create_eval_run_directory(args)
    # Preserve the entrypoint contract that even a failed checkpoint validation
    # leaves an initialized run record. A valid checkpoint rewrites these files
    # below with checkpoint-derived experiment controls.
    initialize_eval_files(run_dir, args)
    torch = _require_torch()
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for deterministic clean evaluation.")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not bool(args.deterministic):
        raise ValueError("Clean evaluation defaults to deterministic masked argmax; stochastic eval is not implemented in T15.")

    _set_seed(int(args.seed), torch=torch)
    device = torch.device(str(args.device))
    checkpoint_payload = _load_trusted_checkpoint(torch, Path(args.checkpoint))
    experiment_controls = checkpoint_experiment_controls(checkpoint_payload)
    checkpoint_model_seed = _checkpoint_model_seed(checkpoint_payload)
    freeze_ue_mobility = _resolve_eval_ue_mobility(args, experiment_controls)
    initialize_eval_files(run_dir, args, experiment_controls)
    module_dims = _module_dims_from_checkpoint(checkpoint_payload, args)
    modules = _build_modules(dims=module_dims, experiment_controls=experiment_controls, device=device)
    _load_module_state(modules, checkpoint_payload)
    _set_eval_mode(modules)

    metrics_rows: list[dict[str, Any]] = []
    aggregate = _empty_aggregate()
    graph_builder = CleanGraphBuilder()
    try:
        for episode in range(int(args.episodes)):
            episode_seed = int(args.seed) + episode
            _set_seed(episode_seed, torch=torch)
            env = Env(
                completed_dag_weight=float(experiment_controls["completed_dag_weight"]),
                freeze_ue_mobility=freeze_ue_mobility,
            )
            env.reset()
            graph_builder.reset()
            episode_result = _run_eval_episode(
                env=env,
                graph_builder=graph_builder,
                modules=modules,
                device=device,
                arrival_steps=int(args.arrival_steps),
                max_drain_steps=int(args.max_drain_steps),
                episode=episode,
                environment_seed=episode_seed,
                offloading_policy=str(args.offloading_policy),
                checkpoint_path=str(Path(args.checkpoint)),
                checkpoint_model_seed=checkpoint_model_seed,
                completed_dag_weight=float(experiment_controls["completed_dag_weight"]),
                freeze_movement=bool(args.freeze_movement),
                detach_critic_hgnn=bool(experiment_controls["detach_critic_hgnn"]),
                offloading_counterfactual_coef=float(
                    experiment_controls["offloading_counterfactual_coef"]
                ),
                offloading_action_value_loss_coef=float(
                    experiment_controls["offloading_action_value_loss_coef"]
                ),
                offloading_lagged_q_coef=float(experiment_controls["offloading_lagged_q_coef"]),
                offloading_lagged_q_loss_coef=float(
                    experiment_controls["offloading_lagged_q_loss_coef"]
                ),
            )
            episode_decisions = episode_result.pop("_offloading_decisions")
            metrics_rows.append(episode_result)
            _write_jsonl(run_dir / "eval_metrics.jsonl", episode_result)
            for decision in episode_decisions:
                _write_jsonl(run_dir / "offloading_decisions.jsonl", decision)
            _update_aggregate(aggregate, episode_result)
    finally:
        graph_builder.close()

    summary = _aggregate_summary(aggregate, episode_count=int(args.episodes))
    summary.update(
        {
            "status": "completed",
            "run_dir": str(run_dir),
            "checkpoint": str(args.checkpoint),
            "checkpoint_model_seed": checkpoint_model_seed,
            "deterministic": True,
            "offloading_policy": str(args.offloading_policy),
            "offloading_gate_schema_version": int(OFFLOADING_GATE_SCHEMA_VERSION),
            "git_commit": _git_commit(),
            "movement_frozen": bool(args.freeze_movement),
            "completed_dag_weight": float(experiment_controls["completed_dag_weight"]),
            "detach_critic_hgnn": bool(experiment_controls["detach_critic_hgnn"]),
            "freeze_ue_mobility": freeze_ue_mobility,
            "offloading_counterfactual_coef": float(
                experiment_controls["offloading_counterfactual_coef"]
            ),
            "offloading_action_value_loss_coef": float(
                experiment_controls["offloading_action_value_loss_coef"]
            ),
            "offloading_lagged_q_coef": float(experiment_controls["offloading_lagged_q_coef"]),
            "offloading_lagged_q_loss_coef": float(
                experiment_controls["offloading_lagged_q_loss_coef"]
            ),
            "offloading_lagged_q_scale_seconds": float(
                experiment_controls["offloading_lagged_q_scale_seconds"]
            ),
            "offloading_lagged_q_censor_weight": float(
                experiment_controls["offloading_lagged_q_censor_weight"]
            ),
            "episodes": int(args.episodes),
            "kahypar_circuit_open": bool(graph_builder.kahypar_circuit_open),
            "kahypar_last_failure_reason": graph_builder.kahypar_last_failure_reason,
            "kahypar_cleanup_failed": bool(graph_builder.kahypar_cleanup_failed),
            "kahypar_worker_alive_after_close": bool(graph_builder.kahypar_worker_alive),
        }
    )
    _write_json(run_dir / "eval_summary.json", summary)
    return summary


def _run_eval_episode(
    *,
    env: Env,
    graph_builder: CleanGraphBuilder,
    modules: CleanTrainingModules,
    device: Any,
    arrival_steps: int,
    max_drain_steps: int,
    episode: int,
    environment_seed: int,
    offloading_policy: str,
    checkpoint_path: str,
    checkpoint_model_seed: int | None,
    completed_dag_weight: float,
    freeze_movement: bool = False,
    detach_critic_hgnn: bool = False,
    offloading_counterfactual_coef: float = 0.0,
    offloading_action_value_loss_coef: float = 0.0,
    offloading_lagged_q_coef: float = 0.0,
    offloading_lagged_q_loss_coef: float = 0.0,
) -> dict[str, Any]:
    arrival_slots = 0
    drain_slots = 0
    movement_action_counts = {str(action): 0 for action in config.CLEAN_MOVEMENT_ACTIONS}
    offloading_action_count = 0
    last_info: dict[str, Any] = {}
    ready_task_counts: list[int] = []
    skipped_ready_no_candidate_total = 0
    assignment_buffer_entry_total = 0
    committed_assignment_total = 0
    kahypar_status_counts: dict[str, int] = {}
    kahypar_partition_hyperedge_total = 0
    kahypar_partition_nonzero_slot_count = 0
    offloading_decisions: list[dict[str, Any]] = []

    for _ in range(max(int(arrival_steps), 0)):
        done, info, movement_counts, off_count, slot_decisions = _eval_one_slot(
            env=env,
            graph_builder=graph_builder,
            modules=modules,
            device=device,
            allow_dag_arrivals=True,
            offloading_policy=offloading_policy,
            environment_seed=environment_seed,
            episode=episode,
            slot=arrival_slots,
            checkpoint_path=checkpoint_path,
            checkpoint_model_seed=checkpoint_model_seed,
            freeze_movement=freeze_movement,
            detach_critic_hgnn=detach_critic_hgnn,
        )
        arrival_slots += 1
        offloading_action_count += off_count
        offloading_decisions.extend(slot_decisions)
        _merge_counts(movement_action_counts, movement_counts)
        last_info = info
        ready_task_counts.append(int(info.get("frozen_ready_task_count", 0)))
        skipped_ready_no_candidate_total += int(info.get("offloading_skipped_no_candidate", 0))
        assignment_buffer_entry_total += int(info.get("assignment_buffer_entry_count", 0))
        committed_assignment_total += int(info.get("newly_assigned_tasks", 0))
        _count_kahypar_status(kahypar_status_counts, info)
        partition_edge_count = int(info.get("kahypar_partition_hyperedge_count", 0))
        kahypar_partition_hyperedge_total += partition_edge_count
        kahypar_partition_nonzero_slot_count += int(partition_edge_count > 0)
        if done:
            break

    arrival_snapshot = _snapshot_arrival_metrics(env=env, arrival_slots=arrival_slots)

    while drain_slots < max(int(max_drain_steps), 0) and _active_dag_count(env) > 0:
        done, info, movement_counts, off_count, slot_decisions = _eval_one_slot(
            env=env,
            graph_builder=graph_builder,
            modules=modules,
            device=device,
            allow_dag_arrivals=False,
            offloading_policy=offloading_policy,
            environment_seed=environment_seed,
            episode=episode,
            slot=arrival_slots + drain_slots,
            checkpoint_path=checkpoint_path,
            checkpoint_model_seed=checkpoint_model_seed,
            freeze_movement=freeze_movement,
            detach_critic_hgnn=detach_critic_hgnn,
        )
        drain_slots += 1
        offloading_action_count += off_count
        offloading_decisions.extend(slot_decisions)
        _merge_counts(movement_action_counts, movement_counts)
        last_info = info
        ready_task_counts.append(int(info.get("frozen_ready_task_count", 0)))
        skipped_ready_no_candidate_total += int(info.get("offloading_skipped_no_candidate", 0))
        assignment_buffer_entry_total += int(info.get("assignment_buffer_entry_count", 0))
        committed_assignment_total += int(info.get("newly_assigned_tasks", 0))
        _count_kahypar_status(kahypar_status_counts, info)
        partition_edge_count = int(info.get("kahypar_partition_hyperedge_count", 0))
        kahypar_partition_hyperedge_total += partition_edge_count
        kahypar_partition_nonzero_slot_count += int(partition_edge_count > 0)
        if done:
            break

    total_slots = int(arrival_slots + drain_slots)
    total_time = float(total_slots) * float(config.TIME_SLOT_DURATION)
    info_metrics = env.metrics.to_info(total_slots, total_time_seconds=total_time)
    completed = float(info_metrics.get("completed_dag_count", 0.0))
    generated = float(info_metrics.get("generated_dag_count", 0.0))
    assignment_entries = float(env.metrics.metrics.action_count)
    movement_distribution = _normalized_distribution(movement_action_counts)
    diagnostics = _evaluation_diagnostics(
        env=env,
        last_info=last_info,
        ready_task_counts=ready_task_counts,
        skipped_ready_no_candidate_total=skipped_ready_no_candidate_total,
        assignment_buffer_entry_total=assignment_buffer_entry_total,
        committed_assignment_total=committed_assignment_total,
        drain_slots=drain_slots,
        max_drain_steps=max_drain_steps,
    )
    average_flowtime = None if completed <= 0.0 else float(info_metrics.get("average_dag_flowtime", 0.0))
    energy_per_completed = None if completed <= 0.0 else float(info_metrics.get("energy_per_completed_dag", 0.0))
    finalize_decision_realizations(offloading_decisions, env=env)
    offloading_summary = summarize_offloading_decisions(offloading_decisions)
    dag_flowtime_samples = [float(value) for value in env.metrics.metrics.dag_flowtimes]
    return {
        "episode": int(episode),
        "environment_seed": int(environment_seed),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_model_seed": checkpoint_model_seed,
        "offloading_policy": str(offloading_policy),
        "offloading_gate_schema_version": int(OFFLOADING_GATE_SCHEMA_VERSION),
        "git_commit": _git_commit(),
        "completed_dag_weight": float(completed_dag_weight),
        "detach_critic_hgnn": bool(detach_critic_hgnn),
        "offloading_counterfactual_coef": float(offloading_counterfactual_coef),
        "offloading_action_value_loss_coef": float(offloading_action_value_loss_coef),
        "offloading_lagged_q_coef": float(offloading_lagged_q_coef),
        "offloading_lagged_q_loss_coef": float(offloading_lagged_q_loss_coef),
        "generated_DAG_count": generated,
        "completed_DAG_count": completed,
        "DAG_completion_rate": float(completed / max(generated, 1.0)),
        "Average_DAG_flowtime": average_flowtime,
        "Median_DAG_flowtime": _percentile(dag_flowtime_samples, 50.0),
        "P90_DAG_flowtime": _percentile(dag_flowtime_samples, 90.0),
        "DAG_throughput": float(completed / max(total_time, float(config.TIME_SLOT_DURATION))),
        "Average_critical_path_task_completion_delay": float(
            info_metrics.get("average_critical_path_task_completion_delay", 0.0)
        ),
        "Energy_per_completed_DAG": energy_per_completed,
        "total_executed_slots": total_slots,
        "arrival_slots_executed": int(arrival_slots),
        **arrival_snapshot,
        "drain_slots_executed": int(drain_slots),
        "invalid_assignment_count": float(info_metrics.get("invalid_assignment_count", 0.0)),
        "invalid_assignment_rate": None if assignment_entries <= 0.0 else float(env.metrics.metrics.invalid_assignment_count / assignment_entries),
        "action_executed_rate": None if assignment_entries <= 0.0 else float(env.metrics.metrics.executed_action_count / assignment_entries),
        "movement_action_distribution": movement_distribution,
        "movement_frozen": bool(freeze_movement),
        "freeze_ue_mobility": bool(env.freeze_ue_mobility),
        "initial_hotspot_ue_count": int(env.initial_hotspot_ue_count),
        "offloading_action_count": int(offloading_action_count),
        "hover_action_ratio": info_metrics.get("hover_action_ratio"),
        "mean_uav_displacement_per_slot": info_metrics.get("mean_uav_displacement_per_slot"),
        "kahypar_partition_status_counts": dict(kahypar_status_counts),
        "kahypar_degraded_slot_count": int(
            sum(count for status, count in kahypar_status_counts.items() if str(status).startswith("degraded"))
        ),
        "kahypar_success_slot_count": int(kahypar_status_counts.get("success", 0)),
        "kahypar_partition_hyperedge_total": int(kahypar_partition_hyperedge_total),
        "kahypar_partition_nonzero_slot_count": int(kahypar_partition_nonzero_slot_count),
        "kahypar_degraded_label": (
            str(config.KAHYPAR_DEGRADED_EXPERIMENT_LABEL)
            if any(str(status).startswith("degraded") for status in kahypar_status_counts)
            else None
        ),
        "total_evaluation_time": total_time,
        "active_dag_count_after_eval": int(_active_dag_count(env)),
        "arrival_backlog_DAG_count": int(arrival_snapshot["arrival_active_DAG_count"]),
        "arrival_backlog_task_count": int(arrival_snapshot["arrival_active_task_count"]),
        "final_backlog_DAG_count": int(_active_dag_count(env)),
        "final_backlog_task_count": int(_active_task_count(env)),
        **offloading_summary,
        "_dag_flowtime_samples": dag_flowtime_samples,
        "_offloading_decisions": offloading_decisions,
        **diagnostics,
        "last_info": _jsonable(last_info),
    }


def _eval_one_slot(
    *,
    env: Env,
    graph_builder: CleanGraphBuilder,
    modules: CleanTrainingModules,
    device: Any,
    allow_dag_arrivals: bool,
    offloading_policy: str,
    environment_seed: int,
    episode: int,
    slot: int,
    checkpoint_path: str,
    checkpoint_model_seed: int | None,
    freeze_movement: bool = False,
    detach_critic_hgnn: bool = False,
) -> tuple[bool, dict[str, Any], dict[str, int], int, list[dict[str, Any]]]:
    import torch

    with _dag_arrival_enabled(allow_dag_arrivals):
        prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
    with torch.no_grad():
        encoded = encode_prepared_slot(
            prepared_state=prepared,
            env=env,
            hgnn=modules.hgnn,
            critic=modules.critic,
            movement_actor=modules.movement_actor,
            device=device,
            detach_critic_hgnn=detach_critic_hgnn,
        )
    movement_actions = _select_deterministic_movement_actions(encoded, freeze_movement=freeze_movement)
    movement_records = [
        type("MovementEvalRecord", (), {"selected_action": action})()
        for action in movement_actions.values()
    ]
    env.apply_movement(movement_actions)
    ready_tasks = [env.task_manager.get_task(task_id) for task_id in prepared.frozen_ready_task_ids]
    ready_tasks = [task for task in ready_tasks if task is not None and task.is_ready]
    assignment_buffer, offloading_decisions = select_eval_offloading_actions(
        policy=offloading_policy,
        offloading_actor=modules.offloading_actor,
        frozen_ready_tasks=ready_tasks,
        task_embeddings=encoded.task_embeddings.detach(),
        graph_snapshot=prepared.graph_snapshot,
        task_manager=env.task_manager,
        uavs=env.uavs,
        executor=env.executor,
        current_time_seconds=env.current_time_seconds,
        environment_seed=environment_seed,
        episode=episode,
        slot=slot,
        checkpoint_path=checkpoint_path,
        checkpoint_model_seed=checkpoint_model_seed,
        uav_service_positions=env.uav_service_positions,
        ue_service_positions=env.ue_service_positions,
        ues=env.ues,
    )
    _, _, done, info = env.commit_and_advance(assignment_buffer=assignment_buffer)
    movement_distribution = _movement_action_distribution(movement_records)
    movement_counts = {
        action: int(round(float(ratio) * max(len(movement_records), 1)))
        for action, ratio in movement_distribution.items()
    }
    info["movement_action_distribution"] = movement_distribution
    info["movement_frozen"] = bool(freeze_movement)
    info["offloading_action_count"] = len(offloading_decisions)
    info["kahypar_partition_status"] = str(getattr(prepared.graph_snapshot, "partition_status", "disabled"))
    info["kahypar_partition_hyperedge_count"] = int(len(prepared.graph_snapshot.partition_hyperedges))
    return bool(done), info, movement_counts, len(offloading_decisions), offloading_decisions


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = run_evaluation(args)
    except ModuleNotFoundError as exc:
        if exc.name == "torch" or "torch" in str(exc).lower():
            print("clean deterministic evaluation requires torch and a clean checkpoint.", file=sys.stderr)
            return 2
        raise
    except (FileNotFoundError, ValueError) as exc:
        print(f"clean deterministic evaluation unavailable: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


def _build_modules(
    *,
    dims: dict[str, int],
    experiment_controls: dict[str, Any],
    device: Any,
) -> CleanTrainingModules:
    from marl_models.hgnn import build_clean_task_encoder
    from marl_models.mappo.clean_movement_actor import CleanMovementActor
    from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
    from marl_models.mappo.clean_offloading_action_value import CleanOffloadingActionValueCritic
    from marl_models.mappo.clean_lagged_residual_q import (
        build_rng_neutral_lagged_residual_q_critic,
    )
    from marl_models.mappo.clean_ppo import CleanCentralizedCritic, clean_critic_input_dim

    critic_input_dim = clean_critic_input_dim(int(dims["task_embedding_dim"]), config.NUM_UAVS)
    offloading_actor = CleanOffloadingActor(
        task_embedding_dim=int(dims["task_embedding_dim"]),
        hidden_dim=int(dims["hidden_dim"]),
    ).to(device)
    action_value_enabled = float(experiment_controls.get("offloading_counterfactual_coef", 0.0)) > 0.0
    lagged_q_enabled = float(experiment_controls.get("offloading_lagged_q_coef", 0.0)) > 0.0
    encoder_type = str(experiment_controls.get("task_encoder", "hgnn"))
    modules = CleanTrainingModules(
        hgnn=build_clean_task_encoder(
            encoder_type=encoder_type,
            task_feature_dim=int(dims["task_feature_dim"]),
            hidden_dim=int(dims["hidden_dim"]),
            output_dim=int(dims["task_embedding_dim"]),
        ).to(device),
        movement_actor=CleanMovementActor(
            task_embedding_dim=int(dims["task_embedding_dim"]),
            hidden_dim=int(dims["hidden_dim"]),
        ).to(device),
        offloading_actor=offloading_actor,
        critic=CleanCentralizedCritic(
            input_dim=critic_input_dim,
            hidden_dim=int(dims["hidden_dim"]),
        ).to(device),
        offloading_action_value_critic=(
            CleanOffloadingActionValueCritic(
                input_dim=int(offloading_actor.candidate_feature_dim) + int(critic_input_dim),
                hidden_dim=int(dims["hidden_dim"]),
            ).to(device)
            if action_value_enabled
            else None
        ),
        offloading_lagged_q_critic=(
            build_rng_neutral_lagged_residual_q_critic(
                input_dim=int(offloading_actor.candidate_feature_dim) + int(critic_input_dim),
                hidden_dim=int(dims["hidden_dim"]),
            ).to(device)
            if lagged_q_enabled
            else None
        ),
    )
    return modules


def _load_module_state(modules: CleanTrainingModules, payload: dict[str, Any]) -> None:
    modules.hgnn.load_state_dict(payload["hgnn"])
    modules.movement_actor.load_state_dict(payload["movement_actor"])
    modules.offloading_actor.load_state_dict(payload["offloading_actor"])
    modules.critic.load_state_dict(payload["critic"])
    action_value_state = payload.get("offloading_action_value_critic")
    lagged_q_state = payload.get("offloading_lagged_residual_q")
    if modules.offloading_action_value_critic is None and action_value_state is not None:
        raise ValueError("checkpoint action-value critic is enabled but eval controls disabled it")
    if modules.offloading_action_value_critic is not None and action_value_state is None:
        raise ValueError("enabled checkpoint is missing offloading action-value critic state")
    if modules.offloading_action_value_critic is not None:
        modules.offloading_action_value_critic.load_state_dict(action_value_state)
    if modules.offloading_lagged_q_critic is None and lagged_q_state is not None:
        raise ValueError("checkpoint lagged residual Q is enabled but eval controls disabled it")
    if modules.offloading_lagged_q_critic is not None and lagged_q_state is None:
        raise ValueError("enabled checkpoint is missing lagged residual Q state")
    if modules.offloading_lagged_q_critic is not None:
        modules.offloading_lagged_q_critic.load_state_dict(lagged_q_state)


def _load_trusted_checkpoint(torch: Any, checkpoint: Path) -> dict[str, Any]:
    """Load a project-generated clean checkpoint across PyTorch versions."""
    try:
        return torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(checkpoint, map_location="cpu")


def _module_dims_from_checkpoint(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, int]:
    cli_config = payload.get("config", {}).get("cli", {}) if isinstance(payload.get("config"), dict) else {}
    task_embedding_dim = args.task_embedding_dim or cli_config.get("task_embedding_dim") or _infer_hgnn_output_dim(payload)
    hidden_dim = args.hidden_dim or cli_config.get("hidden_dim") or _infer_hidden_dim(payload)
    return {
        "task_feature_dim": int(_infer_hgnn_task_feature_dim(payload)),
        "task_embedding_dim": int(task_embedding_dim),
        "hidden_dim": int(hidden_dim),
    }


def _infer_hgnn_task_feature_dim(payload: dict[str, Any]) -> int:
    state = payload.get("hgnn", {})
    weight = state.get("input_proj.weight")
    return int(weight.shape[1]) if hasattr(weight, "shape") else 12


def _infer_hgnn_output_dim(payload: dict[str, Any]) -> int:
    state = payload.get("hgnn", {})
    weight = state.get("output_proj.weight")
    return int(weight.shape[0]) if hasattr(weight, "shape") else 64


def _infer_hidden_dim(payload: dict[str, Any]) -> int:
    state = payload.get("hgnn", {})
    weight = state.get("input_proj.weight")
    return int(weight.shape[0]) if hasattr(weight, "shape") else 128


def _set_eval_mode(modules: CleanTrainingModules) -> None:
    modules.hgnn.eval()
    modules.movement_actor.eval()
    modules.offloading_actor.eval()
    modules.critic.eval()
    if modules.offloading_action_value_critic is not None:
        modules.offloading_action_value_critic.eval()
    if modules.offloading_lagged_q_critic is not None:
        modules.offloading_lagged_q_critic.eval()


def _select_deterministic_movement_actions(encoded: Any, *, freeze_movement: bool) -> dict[int, int]:
    uav_ids = encoded.movement_observation.uav_ids
    if freeze_movement:
        hover_action = int(config.CLEAN_MOVEMENT_ACTIONS.index(config.CLEAN_MOVEMENT_HOVER_ACTION))
        return {int(uav_id): hover_action for uav_id in uav_ids}

    import torch

    selected_movement = torch.argmax(encoded.movement_logits, dim=-1)
    return {
        int(uav_id): int(selected_movement[idx].detach().cpu().item())
        for idx, uav_id in enumerate(uav_ids)
    }


@contextmanager
def _dag_arrival_enabled(enabled: bool):
    original = config.DAG_BASE_ARRIVAL_PROB
    if not enabled:
        config.DAG_BASE_ARRIVAL_PROB = 0.0
    try:
        yield
    finally:
        config.DAG_BASE_ARRIVAL_PROB = original


def _active_dag_count(env: Env) -> int:
    return sum(1 for job in env.task_manager.jobs.values() if not job.completed)


def _active_task_count(env: Env) -> int:
    return sum(1 for task in env.task_manager.tasks.values() if getattr(task, "state", None) != "COMPLETED")


def _snapshot_arrival_metrics(*, env: Env, arrival_slots: int) -> dict[str, Any]:
    arrival_time = float(arrival_slots) * float(config.TIME_SLOT_DURATION)
    info_metrics = env.metrics.to_info(int(arrival_slots), total_time_seconds=arrival_time)
    generated = float(info_metrics.get("generated_dag_count", 0.0))
    completed = float(info_metrics.get("completed_dag_count", 0.0))
    return {
        "arrival_generated_DAG_count": generated,
        "arrival_completed_DAG_count": completed,
        "arrival_DAG_completion_rate": float(completed / max(generated, 1.0)),
        "arrival_active_DAG_count": int(_active_dag_count(env)),
        "arrival_active_task_count": int(_active_task_count(env)),
    }


def _evaluation_diagnostics(
    *,
    env: Env,
    last_info: dict[str, Any],
    ready_task_counts: list[int],
    skipped_ready_no_candidate_total: int,
    assignment_buffer_entry_total: int,
    committed_assignment_total: int,
    drain_slots: int,
    max_drain_steps: int,
) -> dict[str, Any]:
    task_manager = env.task_manager
    tasks = list(task_manager.tasks.values())
    jobs = list(task_manager.jobs.values())
    lifecycle_states = [
        "WAITING_DEPENDENCY",
        "READY_UNSCHEDULED",
        "IN_SERVICE",
        "RETURNING",
        "COMPLETED",
    ]
    service_phases = ["UPLOADING_OR_TRANSFERRING", "QUEUED", "COMPUTING"]
    lifecycle_counts = {state: 0 for state in lifecycle_states}
    service_phase_counts = {phase: 0 for phase in service_phases}
    for task in tasks:
        state = str(getattr(task, "state", ""))
        if state in lifecycle_counts:
            lifecycle_counts[state] += 1
        phase = getattr(task, "service_phase", None)
        if phase in service_phase_counts:
            service_phase_counts[phase] += 1

    ready_count = int(lifecycle_counts["READY_UNSCHEDULED"])
    queue_lengths = [
        len(env.executor.uav_queues.get(int(getattr(uav, "id")), []))
        for uav in env.uavs
    ]
    queued_workloads: list[float] = []
    for queue in env.executor.uav_queues.values():
        workload = 0.0
        for task_id in queue:
            task = task_manager.get_task(task_id)
            if task is not None:
                workload += float(getattr(task, "num_operation", 0.0))
        queued_workloads.append(workload)
    active_jobs = [job for job in jobs if not job.completed]
    sink_task_ids = {task_id for job in jobs for task_id in job.sink_task_ids}
    reward_completed_task_count = sum(1 for task in tasks if bool(getattr(task, "reward_settled", False)))
    completed_non_sink_task_count = sum(
        1
        for task in tasks
        if task.task_id not in sink_task_ids and getattr(task, "state", None) == "COMPLETED"
    )
    returning_sink_task_count = sum(
        1
        for task in tasks
        if task.task_id in sink_task_ids and getattr(task, "state", None) == "RETURNING"
    )
    completed_sink_task_count = sum(
        1
        for task in tasks
        if task.task_id in sink_task_ids and getattr(task, "state", None) == "COMPLETED"
    )
    progress_samples = []
    for job in active_jobs[:5]:
        job_tasks = task_manager.get_job_tasks(job.dag_id)
        job_sink_ids = set(job.sink_task_ids)
        progress_samples.append(
            {
                "dag_id": str(job.dag_id),
                "total_tasks": int(len(job_tasks)),
                "completed_tasks": int(sum(1 for task in job_tasks if getattr(task, "state", None) == "COMPLETED")),
                "sink_count": int(len(job_sink_ids)),
                "completed_sink_count": int(
                    sum(1 for task in job_tasks if task.task_id in job_sink_ids and getattr(task, "state", None) == "COMPLETED")
                ),
                "returning_sink_count": int(
                    sum(1 for task in job_tasks if task.task_id in job_sink_ids and getattr(task, "state", None) == "RETURNING")
                ),
            }
        )
    max_available = max((float(value) for value in env.executor.uav_available_time.values()), default=0.0)
    return {
        "final_active_DAG_count": int(len(active_jobs)),
        "final_active_task_count": int(len([task for task in tasks if getattr(task, "state", None) != "COMPLETED"])),
        "task_lifecycle_counts": lifecycle_counts,
        "service_phase_counts": service_phase_counts,
        "ready_task_count_mean": float(np.mean(ready_task_counts)) if ready_task_counts else float(ready_count),
        "ready_task_count_max": int(max(ready_task_counts)) if ready_task_counts else int(ready_count),
        "skipped_ready_due_to_no_legal_candidate_count": int(skipped_ready_no_candidate_total),
        "assignment_buffer_entry_count": int(assignment_buffer_entry_total),
        "successfully_committed_assignment_count": int(committed_assignment_total),
        "reward_completed_task_count": int(reward_completed_task_count),
        "completed_non_sink_task_count": int(completed_non_sink_task_count),
        "returning_sink_task_count": int(returning_sink_task_count),
        "completed_sink_task_count": int(completed_sink_task_count),
        "unfinished_DAG_progress_samples": progress_samples,
        "executor_queue_summary": {
            "mean_queue_length": float(np.mean(queue_lengths)) if queue_lengths else 0.0,
            "max_queue_length": int(max(queue_lengths)) if queue_lengths else 0,
            "total_queued_workload": float(sum(queued_workloads)),
            "max_available_time": float(max_available),
        },
        "drain_end_reason": (
            "all_completed"
            if len(active_jobs) == 0
            else "max_drain_steps_reached"
            if int(drain_slots) >= max(int(max_drain_steps), 0)
            else "episode_terminated"
        ),
    }


def _empty_aggregate() -> dict[str, Any]:
    return {
        "episodes": [],
        "movement_counts": {str(action): 0.0 for action in config.CLEAN_MOVEMENT_ACTIONS},
    }


def _update_aggregate(aggregate: dict[str, Any], row: dict[str, Any]) -> None:
    aggregate["episodes"].append(row)
    for action, ratio in row.get("movement_action_distribution", {}).items():
        aggregate["movement_counts"][action] = aggregate["movement_counts"].get(action, 0.0) + float(ratio)


def _aggregate_summary(aggregate: dict[str, Any], *, episode_count: int) -> dict[str, Any]:
    rows = aggregate["episodes"]
    generated_total = float(sum(row["generated_DAG_count"] for row in rows))
    completed_total = float(sum(row["completed_DAG_count"] for row in rows))
    arrival_generated_total = float(sum(row.get("arrival_generated_DAG_count", 0.0) for row in rows))
    arrival_completed_total = float(sum(row.get("arrival_completed_DAG_count", 0.0) for row in rows))
    flowtime_samples = _flatten_samples(rows, "_dag_flowtime_samples")
    estimator_errors = _flatten_samples(rows, "_estimator_error_samples")
    regret_samples = _flatten_samples(rows, "_selected_estimated_regret_samples")
    entropy_samples = _flatten_samples(rows, "_actor_entropy_samples")
    margin_samples = _flatten_samples(rows, "_actor_margin_samples")
    agreement_count = int(sum(row.get("actor_greedy_agreement_count", 0) for row in rows))
    agreement_total = int(sum(row.get("actor_greedy_comparison_count", 0) for row in rows))
    return {
        "generated_DAG_count": generated_total,
        "completed_DAG_count": completed_total,
        "DAG_completion_rate": _mean(row["DAG_completion_rate"] for row in rows),
        "Average_DAG_flowtime": None if completed_total <= 0.0 else _mean(row["Average_DAG_flowtime"] for row in rows),
        "Median_DAG_flowtime": _percentile(flowtime_samples, 50.0),
        "P90_DAG_flowtime": _percentile(flowtime_samples, 90.0),
        "DAG_throughput": _mean(row["DAG_throughput"] for row in rows),
        "Average_critical_path_task_completion_delay": _mean(row["Average_critical_path_task_completion_delay"] for row in rows),
        "Energy_per_completed_DAG": None if completed_total <= 0.0 else _mean(row["Energy_per_completed_DAG"] for row in rows),
        "total_executed_slots": int(sum(row["total_executed_slots"] for row in rows)),
        "arrival_slots_executed": int(sum(row["arrival_slots_executed"] for row in rows)),
        "arrival_generated_DAG_count": arrival_generated_total,
        "arrival_completed_DAG_count": arrival_completed_total,
        "arrival_DAG_completion_rate": float(arrival_completed_total / max(arrival_generated_total, 1.0)),
        "arrival_active_DAG_count": int(sum(row.get("arrival_active_DAG_count", 0) for row in rows)),
        "arrival_active_task_count": int(sum(row.get("arrival_active_task_count", 0) for row in rows)),
        "drain_slots_executed": int(sum(row["drain_slots_executed"] for row in rows)),
        "invalid_assignment_count": float(sum(row["invalid_assignment_count"] for row in rows)),
        "invalid_assignment_rate": _mean(row["invalid_assignment_rate"] for row in rows if row["invalid_assignment_rate"] is not None),
        "action_executed_rate": _mean(row["action_executed_rate"] for row in rows if row["action_executed_rate"] is not None),
        "movement_action_distribution": {
            action: float(total) / max(float(episode_count), 1.0)
            for action, total in aggregate["movement_counts"].items()
        },
        "movement_frozen": bool(rows) and all(bool(row.get("movement_frozen", False)) for row in rows),
        "freeze_ue_mobility": bool(rows) and all(bool(row.get("freeze_ue_mobility", False)) for row in rows),
        "initial_hotspot_ue_count_mean": _mean(row.get("initial_hotspot_ue_count") for row in rows),
        "offloading_action_count": int(sum(row["offloading_action_count"] for row in rows)),
        "offloading_policy": _common_value(row.get("offloading_policy") for row in rows),
        "offloading_gate_schema_version": int(OFFLOADING_GATE_SCHEMA_VERSION),
        "valid_candidate_count_distribution": _sum_count_dicts(
            row.get("valid_candidate_count_distribution", {}) for row in rows
        ),
        "actor_normalized_entropy_mean": _mean(entropy_samples),
        "actor_entropy_sample_count": int(len(entropy_samples)),
        "actor_top1_top2_margin_mean": _mean(margin_samples),
        "actor_margin_sample_count": int(len(margin_samples)),
        "actor_greedy_agreement_rate": (
            float(agreement_count / agreement_total) if agreement_total > 0 else None
        ),
        "actor_greedy_agreement_count": agreement_count,
        "actor_greedy_comparison_count": agreement_total,
        "selected_estimated_regret_mean": _mean(regret_samples),
        "selected_estimated_regret_p90": _percentile(regret_samples, 90.0),
        "estimator_calibration_count": int(len(estimator_errors)),
        "estimator_calibration_mae": _mean(abs(value) for value in estimator_errors),
        "estimator_calibration_bias": _mean(estimator_errors),
        "estimator_calibration_p90_abs_error": _percentile(
            [abs(value) for value in estimator_errors], 90.0
        ),
        "realized_cross_uav_transfer_time": float(
            sum(float(row.get("realized_cross_uav_transfer_time", 0.0)) for row in rows)
        ),
        "realized_queue_resource_wait": float(
            sum(float(row.get("realized_queue_resource_wait", 0.0)) for row in rows)
        ),
        "arrival_backlog_DAG_count": int(sum(row.get("arrival_backlog_DAG_count", 0) for row in rows)),
        "arrival_backlog_task_count": int(sum(row.get("arrival_backlog_task_count", 0) for row in rows)),
        "final_backlog_DAG_count": int(sum(row.get("final_backlog_DAG_count", 0) for row in rows)),
        "final_backlog_task_count": int(sum(row.get("final_backlog_task_count", 0) for row in rows)),
        "final_active_DAG_count": int(sum(row.get("final_active_DAG_count", 0) for row in rows)),
        "final_active_task_count": int(sum(row.get("final_active_task_count", 0) for row in rows)),
        "task_lifecycle_counts": _sum_count_dicts(row.get("task_lifecycle_counts", {}) for row in rows),
        "service_phase_counts": _sum_count_dicts(row.get("service_phase_counts", {}) for row in rows),
        "ready_task_count_mean": _mean(row.get("ready_task_count_mean") for row in rows),
        "ready_task_count_max": int(max((row.get("ready_task_count_max", 0) for row in rows), default=0)),
        "skipped_ready_due_to_no_legal_candidate_count": int(
            sum(row.get("skipped_ready_due_to_no_legal_candidate_count", 0) for row in rows)
        ),
        "assignment_buffer_entry_count": int(sum(row.get("assignment_buffer_entry_count", 0) for row in rows)),
        "successfully_committed_assignment_count": int(
            sum(row.get("successfully_committed_assignment_count", 0) for row in rows)
        ),
        "reward_completed_task_count": int(sum(row.get("reward_completed_task_count", 0) for row in rows)),
        "completed_non_sink_task_count": int(sum(row.get("completed_non_sink_task_count", 0) for row in rows)),
        "returning_sink_task_count": int(sum(row.get("returning_sink_task_count", 0) for row in rows)),
        "completed_sink_task_count": int(sum(row.get("completed_sink_task_count", 0) for row in rows)),
        "unfinished_DAG_progress_samples": [
            sample
            for row in rows
            for sample in row.get("unfinished_DAG_progress_samples", [])
        ][:10],
        "executor_queue_summary": _aggregate_queue_summaries(row.get("executor_queue_summary", {}) for row in rows),
        "drain_end_reason": _aggregate_drain_end_reason(row.get("drain_end_reason") for row in rows),
        "hover_action_ratio": _mean(row.get("hover_action_ratio") for row in rows),
        "mean_uav_displacement_per_slot": _mean(row.get("mean_uav_displacement_per_slot") for row in rows),
        "kahypar_partition_status_counts": _sum_count_dicts(
            row.get("kahypar_partition_status_counts", {}) for row in rows
        ),
        "kahypar_degraded_slot_count": int(sum(int(row.get("kahypar_degraded_slot_count", 0)) for row in rows)),
        "kahypar_success_slot_count": int(sum(int(row.get("kahypar_success_slot_count", 0)) for row in rows)),
        "kahypar_partition_hyperedge_total": int(
            sum(int(row.get("kahypar_partition_hyperedge_total", 0)) for row in rows)
        ),
        "kahypar_partition_nonzero_slot_count": int(
            sum(int(row.get("kahypar_partition_nonzero_slot_count", 0)) for row in rows)
        ),
        "kahypar_degraded_label": (
            str(config.KAHYPAR_DEGRADED_EXPERIMENT_LABEL)
            if sum(int(row.get("kahypar_degraded_slot_count", 0)) for row in rows) > 0
            else None
        ),
    }


def _mean(values: Any) -> float | None:
    values_list = [float(value) for value in values if value is not None]
    return float(np.mean(values_list)) if values_list else None


def _percentile(values: Any, percentile: float) -> float | None:
    values_list = [float(value) for value in values if value is not None]
    return float(np.percentile(np.asarray(values_list, dtype=np.float64), percentile)) if values_list else None


def _flatten_samples(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(value) for row in rows for value in row.get(key, [])]


def _common_value(values: Any) -> Any:
    resolved = [value for value in values if value is not None]
    if not resolved:
        return None
    return resolved[0] if all(value == resolved[0] for value in resolved) else "mixed"


def _sum_count_dicts(dicts: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in dicts:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            result[str(key)] = int(result.get(str(key), 0)) + int(value)
    return result


def _aggregate_queue_summaries(summaries: Any) -> dict[str, float]:
    items = [summary for summary in summaries if isinstance(summary, dict)]
    return {
        "mean_queue_length": _mean(item.get("mean_queue_length") for item in items),
        "max_queue_length": float(max((item.get("max_queue_length", 0.0) for item in items), default=0.0)),
        "total_queued_workload": float(sum(float(item.get("total_queued_workload", 0.0)) for item in items)),
        "max_available_time": float(max((item.get("max_available_time", 0.0) for item in items), default=0.0)),
    }


def _aggregate_drain_end_reason(reasons: Any) -> str | None:
    reason_list = [str(reason) for reason in reasons if reason is not None]
    if not reason_list:
        return None
    if all(reason == "all_completed" for reason in reason_list):
        return "all_completed"
    if any(reason == "max_drain_steps_reached" for reason in reason_list):
        return "max_drain_steps_reached"
    return "mixed"


def _merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = int(target.get(key, 0)) + int(value)


def _count_kahypar_status(target: dict[str, int], info: dict[str, Any]) -> None:
    status = str(info.get("kahypar_partition_status", "disabled"))
    target[status] = int(target.get(status, 0)) + 1


def _normalized_distribution(counts: dict[str, int]) -> dict[str, float]:
    total = max(float(sum(counts.values())), 1.0)
    return {str(key): float(value) / total for key, value in counts.items()}


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise ModuleNotFoundError("torch is required for clean deterministic evaluation") from exc
        raise
    return torch


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return bool(torch.cuda.is_available())


def _set_seed(seed: int, *, torch: Any) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _namespace_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _checkpoint_model_seed(payload: dict[str, Any]) -> int | None:
    checkpoint_config = payload.get("config", {})
    cli = checkpoint_config.get("cli", {}) if isinstance(checkpoint_config, dict) else {}
    seed = cli.get("seed") if isinstance(cli, dict) else None
    try:
        return None if seed is None else int(seed)
    except (TypeError, ValueError):
        return None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), ensure_ascii=True, sort_keys=True) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
