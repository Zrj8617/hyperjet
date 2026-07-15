from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

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
    CleanPPOUpdateStats,
    CleanTrainingModules,
    build_single_optimizer,
    close_rollout_with_bootstrap,
    reencode_prepared_after_update,
    write_clean_training_log,
)


def _validated_completed_dag_weight(value: str | float) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError("completed-DAG weight must be finite and non-negative")
    return resolved


def _nonnegative_finite_float(value: str | float) -> float:
    try:
        return _validated_completed_dag_weight(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _resolved_completed_dag_weight(args: argparse.Namespace) -> float:
    return _validated_completed_dag_weight(
        getattr(args, "completed_dag_weight", config.REWARD_COMPLETED_DAG_WEIGHT)
    )


def _validated_offloading_action_value_controls(
    counterfactual_coef: str | float,
    action_value_loss_coef: str | float,
) -> tuple[float, float]:
    beta = _validated_completed_dag_weight(counterfactual_coef)
    eta = _validated_completed_dag_weight(action_value_loss_coef)
    if (beta > 0.0) != (eta > 0.0):
        raise ValueError(
            "offloading counterfactual and action-value loss coefficients must be enabled together"
        )
    return beta, eta


def _resolved_offloading_action_value_controls(args: argparse.Namespace) -> tuple[float, float]:
    return _validated_offloading_action_value_controls(
        getattr(args, "offloading_counterfactual_coef", 0.0),
        getattr(args, "offloading_action_value_loss_coef", 0.0),
    )


def _validated_offloading_lagged_q_controls(
    lagged_q_coef: str | float,
    lagged_q_loss_coef: str | float,
    scale_seconds: str | float,
    censor_weight: str | float,
) -> tuple[float, float, float, float]:
    beta = _validated_completed_dag_weight(lagged_q_coef)
    eta = _validated_completed_dag_weight(lagged_q_loss_coef)
    scale = float(scale_seconds)
    censor = float(censor_weight)
    if (beta > 0.0) != (eta > 0.0):
        raise ValueError("offloading lagged-Q advantage and loss coefficients must be enabled together")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("offloading lagged-Q scale seconds must be finite and positive")
    if not math.isfinite(censor) or not 0.0 <= censor <= 1.0:
        raise ValueError("offloading lagged-Q censor weight must be finite and in [0, 1]")
    return beta, eta, scale, censor


def _resolved_offloading_lagged_q_controls(args: argparse.Namespace) -> tuple[float, float, float, float]:
    controls = _validated_offloading_lagged_q_controls(
        getattr(args, "offloading_lagged_q_coef", 0.0),
        getattr(args, "offloading_lagged_q_loss_coef", 0.0),
        getattr(args, "offloading_lagged_q_scale_seconds", 200.0),
        getattr(args, "offloading_lagged_q_censor_weight", 0.25),
    )
    v1_beta, _ = _resolved_offloading_action_value_controls(args)
    if v1_beta > 0.0 and controls[0] > 0.0:
        raise ValueError("offloading counterfactual v1 and lagged-Q v2 are mutually exclusive")
    return controls


def checkpoint_experiment_controls(payload: dict[str, Any]) -> dict[str, Any]:
    """Resolve experiment controls from new or legacy clean checkpoints."""
    checkpoint_config = payload.get("config", {})
    cli = checkpoint_config.get("cli", {}) if isinstance(checkpoint_config, dict) else {}
    if not isinstance(cli, dict):
        cli = {}
    detach_critic_hgnn = cli.get("detach_critic_hgnn", False)
    if not isinstance(detach_critic_hgnn, bool):
        raise ValueError("checkpoint detach_critic_hgnn must be boolean")
    freeze_ue_mobility = cli.get("freeze_ue_mobility", False)
    if not isinstance(freeze_ue_mobility, bool):
        raise ValueError("checkpoint freeze_ue_mobility must be boolean")
    counterfactual_coef, action_value_loss_coef = _validated_offloading_action_value_controls(
        cli.get("offloading_counterfactual_coef", 0.0),
        cli.get("offloading_action_value_loss_coef", 0.0),
    )
    lagged_q_coef, lagged_q_loss_coef, lagged_q_scale_seconds, lagged_q_censor_weight = (
        _validated_offloading_lagged_q_controls(
            cli.get("offloading_lagged_q_coef", 0.0),
            cli.get("offloading_lagged_q_loss_coef", 0.0),
            cli.get("offloading_lagged_q_scale_seconds", 200.0),
            cli.get("offloading_lagged_q_censor_weight", 0.25),
        )
    )
    if counterfactual_coef > 0.0 and lagged_q_coef > 0.0:
        raise ValueError("checkpoint offloading counterfactual v1 and lagged-Q v2 cannot both be enabled")
    return {
        "completed_dag_weight": _validated_completed_dag_weight(
            cli.get("completed_dag_weight", config.REWARD_COMPLETED_DAG_WEIGHT)
        ),
        "detach_critic_hgnn": detach_critic_hgnn,
        "freeze_ue_mobility": freeze_ue_mobility,
        "offloading_counterfactual_coef": counterfactual_coef,
        "offloading_action_value_loss_coef": action_value_loss_coef,
        "offloading_lagged_q_coef": lagged_q_coef,
        "offloading_lagged_q_loss_coef": lagged_q_loss_coef,
        "offloading_lagged_q_scale_seconds": lagged_q_scale_seconds,
        "offloading_lagged_q_censor_weight": lagged_q_censor_weight,
    }


def validate_resume_experiment_controls(
    args: argparse.Namespace,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Reject a resume that would silently change its reward objective."""
    saved = checkpoint_experiment_controls(payload)
    requested_weight = _resolved_completed_dag_weight(args)
    if not math.isclose(
        requested_weight,
        float(saved["completed_dag_weight"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "resume checkpoint completed-DAG weight mismatch: "
            f"requested {requested_weight}, checkpoint {saved['completed_dag_weight']}"
        )
    requested_detach = bool(getattr(args, "detach_critic_hgnn", False))
    if requested_detach != bool(saved["detach_critic_hgnn"]):
        raise ValueError(
            "resume checkpoint critic-HGNN detach mismatch: "
            f"requested {requested_detach}, checkpoint {saved['detach_critic_hgnn']}"
        )
    requested_freeze_ue = bool(getattr(args, "freeze_ue_mobility", False))
    if requested_freeze_ue != bool(saved["freeze_ue_mobility"]):
        raise ValueError(
            "resume checkpoint UE mobility mismatch: "
            f"requested freeze={requested_freeze_ue}, checkpoint freeze={saved['freeze_ue_mobility']}"
        )
    requested_beta, requested_eta = _resolved_offloading_action_value_controls(args)
    for key, requested in (
        ("offloading_counterfactual_coef", requested_beta),
        ("offloading_action_value_loss_coef", requested_eta),
    ):
        if not math.isclose(
            requested,
            float(saved[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"resume checkpoint {key.replace('_', ' ')} mismatch: "
                f"requested {requested}, checkpoint {saved[key]}"
            )
    requested_lagged = _resolved_offloading_lagged_q_controls(args)
    for key, requested in zip(
        (
            "offloading_lagged_q_coef",
            "offloading_lagged_q_loss_coef",
            "offloading_lagged_q_scale_seconds",
            "offloading_lagged_q_censor_weight",
        ),
        requested_lagged,
    ):
        if not math.isclose(requested, float(saved[key]), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"resume checkpoint {key.replace('_', ' ')} mismatch: "
                f"requested {requested}, checkpoint {saved[key]}"
            )
    return saved


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the HyperUAV clean mainline joint PPO skeleton.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps-per-episode", type=int, default=int(config.EPISODE_LENGTH))
    parser.add_argument("--rollout-horizon", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument(
        "--completed-dag-weight",
        type=_nonnegative_finite_float,
        default=float(config.REWARD_COMPLETED_DAG_WEIGHT),
        help="Run-level completed-DAG reward weight; the clean baseline remains 2.0.",
    )
    parser.add_argument(
        "--detach-critic-hgnn",
        action="store_true",
        default=False,
        help="Block value-loss gradients from the critic into the shared HGNN.",
    )
    parser.add_argument(
        "--freeze-ue-mobility",
        action="store_true",
        default=False,
        help="Keep each UE at its episode-initial position while consuming the normal mobility RNG draws.",
    )
    parser.add_argument(
        "--offloading-counterfactual-coef",
        type=_nonnegative_finite_float,
        default=0.0,
        help="Weight beta for the detached action-conditioned counterfactual offloading advantage.",
    )
    parser.add_argument(
        "--offloading-action-value-loss-coef",
        type=_nonnegative_finite_float,
        default=0.0,
        help="Weight eta for the selected-action value regression loss.",
    )
    parser.add_argument(
        "--offloading-lagged-q-coef",
        type=_nonnegative_finite_float,
        default=0.0,
        help="Weight beta_lq for the detached lagged DAG-outcome Q correction.",
    )
    parser.add_argument(
        "--offloading-lagged-q-loss-coef",
        type=_nonnegative_finite_float,
        default=0.0,
        help="Weight eta_lq for lagged DAG-outcome residual-Q regression.",
    )
    parser.add_argument("--offloading-lagged-q-scale-seconds", type=float, default=200.0)
    parser.add_argument("--offloading-lagged-q-censor-weight", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=Path("logs") / "clean_mainline")
    parser.add_argument("--run-name", type=str, default="clean")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--freeze-movement",
        action="store_true",
        default=False,
        help="Phase 4 P1 A/B switch: force all UAV movement actions to hover and "
        "skip movement rollout records, so the movement head contributes no "
        "policy loss/entropy. Offloading actor, critic, and HGNN train normally. "
        "Off by default; pass explicitly for the frozen-movement baseline runs.",
    )
    parser.add_argument("--device", type=str, default="cuda" if _torch_cuda_available() else "cpu")
    parser.add_argument("--task-embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    return parser


def apply_smoke_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if not bool(args.smoke):
        return args
    args.episodes = min(int(args.episodes), 2)
    args.max_steps_per_episode = min(int(args.max_steps_per_episode), 30)
    args.rollout_horizon = min(int(args.rollout_horizon), 8)
    args.checkpoint_interval = 1
    args.ppo_epochs = min(int(args.ppo_epochs), 1)
    return args


def create_run_directory(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(args.run_name)).strip("_")
    tag = clean_name or "clean"
    if bool(args.smoke) and not tag.startswith("smoke"):
        tag = f"smoke_{tag}"
    run_id = f"{timestamp}_{tag}_seed{int(args.seed)}"
    run_dir = Path(args.output_dir) / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    return run_dir


def build_config_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    completed_dag_weight = _resolved_completed_dag_weight(args)
    counterfactual_coef, action_value_loss_coef = _resolved_offloading_action_value_controls(args)
    lagged_q_coef, lagged_q_loss_coef, lagged_q_scale_seconds, lagged_q_censor_weight = (
        _resolved_offloading_lagged_q_controls(args)
    )
    return {
        "cli": _namespace_to_dict(args),
        "experiment_controls": {
            "completed_dag_weight": completed_dag_weight,
            "detach_critic_hgnn": bool(getattr(args, "detach_critic_hgnn", False)),
            "freeze_ue_mobility": bool(getattr(args, "freeze_ue_mobility", False)),
            "offloading_counterfactual_coef": counterfactual_coef,
            "offloading_action_value_loss_coef": action_value_loss_coef,
            "offloading_lagged_q_coef": lagged_q_coef,
            "offloading_lagged_q_loss_coef": lagged_q_loss_coef,
            "offloading_lagged_q_scale_seconds": lagged_q_scale_seconds,
            "offloading_lagged_q_censor_weight": lagged_q_censor_weight,
            "lagged_q_resume_pending_policy": "discard_restarted_episode",
        },
        "clean_scene": {
            "AREA_WIDTH": config.AREA_WIDTH,
            "AREA_HEIGHT": config.AREA_HEIGHT,
            "NUM_UAVS": config.NUM_UAVS,
            "NUM_UES": config.NUM_UES,
            "TIME_SLOT_DURATION": config.TIME_SLOT_DURATION,
            "HOTSPOT_RADIUS": config.HOTSPOT_RADIUS,
            "EPISODE_LENGTH": config.EPISODE_LENGTH,
            "DAG_BASE_ARRIVAL_PROB": config.DAG_BASE_ARRIVAL_PROB,
            "DAG_HOTSPOT_ARRIVAL_MULTIPLIER": config.DAG_HOTSPOT_ARRIVAL_MULTIPLIER,
            "INPUT_DATA_SIZE_MB_RANGE": list(config.INPUT_DATA_SIZE_MB_RANGE),
            "OUTPUT_DATA_SIZE_MB_RANGE": list(config.OUTPUT_DATA_SIZE_MB_RANGE),
            "TASK_CONSTANT_RANGE": list(config.TASK_CONSTANT_RANGE),
            "TASK_COMPLEXITY_PROBS": dict(config.TASK_COMPLEXITY_PROBS),
            "BASE_UPLOAD_BANDWIDTH_MBPS": list(config.BASE_UPLOAD_BANDWIDTH_MBPS),
            "BASE_DOWNLOAD_BANDWIDTH_MBPS": list(config.BASE_DOWNLOAD_BANDWIDTH_MBPS),
            "BANDWIDTH_LEVEL_PROBS": list(config.BANDWIDTH_LEVEL_PROBS),
            "UAV_COMPUTE_RATE_OPS_PER_SEC": config.UAV_COMPUTE_RATE_OPS_PER_SEC,
            "CLEAN_MAX_QUEUE_PER_UAV": config.CLEAN_MAX_QUEUE_PER_UAV,
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
        "resume_semantics": "checkpoint restore starts from a new episode; mid-episode exact resume is unsupported",
    }


def initialize_run_files(run_dir: Path, args: argparse.Namespace) -> None:
    counterfactual_coef, action_value_loss_coef = _resolved_offloading_action_value_controls(args)
    lagged_q_coef, lagged_q_loss_coef, lagged_q_scale_seconds, lagged_q_censor_weight = (
        _resolved_offloading_lagged_q_controls(args)
    )
    _write_json(run_dir / "config.json", build_config_snapshot(args))
    _write_json(
        run_dir / "run_summary.json",
        {
            "status": "initialized",
            "run_dir": str(run_dir),
            "torch_required_for_training": True,
            "completed_dag_weight": _resolved_completed_dag_weight(args),
            "detach_critic_hgnn": bool(getattr(args, "detach_critic_hgnn", False)),
            "freeze_ue_mobility": bool(getattr(args, "freeze_ue_mobility", False)),
            "offloading_counterfactual_coef": counterfactual_coef,
            "offloading_action_value_loss_coef": action_value_loss_coef,
            "offloading_lagged_q_coef": lagged_q_coef,
            "offloading_lagged_q_loss_coef": lagged_q_loss_coef,
            "offloading_lagged_q_scale_seconds": lagged_q_scale_seconds,
            "offloading_lagged_q_censor_weight": lagged_q_censor_weight,
            "lagged_q_resume_pending_policy": "discard_restarted_episode",
            "resume_semantics": "restart_from_new_episode_only",
        },
    )


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    args = apply_smoke_overrides(args)
    args.completed_dag_weight = _resolved_completed_dag_weight(args)
    (
        args.offloading_counterfactual_coef,
        args.offloading_action_value_loss_coef,
    ) = _resolved_offloading_action_value_controls(args)
    (
        args.offloading_lagged_q_coef,
        args.offloading_lagged_q_loss_coef,
        args.offloading_lagged_q_scale_seconds,
        args.offloading_lagged_q_censor_weight,
    ) = _resolved_offloading_lagged_q_controls(args)
    run_dir = create_run_directory(args)
    initialize_run_files(run_dir, args)

    torch, Categorical = _require_torch()
    _set_seed(int(args.seed), torch=torch)

    from environment.dag_tasks import TASK_STATE_READY_UNSCHEDULED
    from marl_models.hgnn import CleanIncidenceHGNN
    from marl_models.mappo.clean_movement_actor import CleanMovementActor
    from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
    from marl_models.mappo.clean_offloading_action_value import CleanOffloadingActionValueCritic
    from marl_models.mappo.clean_lagged_residual_q import (
        CleanLaggedOutcomeTracker,
        CleanLaggedResidualQCritic,
    )
    from marl_models.mappo.clean_ppo import CleanCentralizedCritic, clean_critic_input_dim
    from marl_models.mappo.clean_trainer import CleanPPOUpdater

    env = Env(
        completed_dag_weight=float(args.completed_dag_weight),
        freeze_ue_mobility=bool(args.freeze_ue_mobility),
    )
    graph_builder = CleanGraphBuilder()
    env.reset()
    graph_builder.reset()
    initial_prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
    task_feature_dim = int(initial_prepared.graph_snapshot.task_features.shape[1])
    critic_input_dim = clean_critic_input_dim(int(args.task_embedding_dim), config.NUM_UAVS)
    offloading_actor = CleanOffloadingActor(
        task_embedding_dim=int(args.task_embedding_dim),
        hidden_dim=int(args.hidden_dim),
    )
    action_value_enabled = float(args.offloading_counterfactual_coef) > 0.0
    offloading_action_value_critic = (
        CleanOffloadingActionValueCritic(
            input_dim=int(offloading_actor.candidate_feature_dim) + int(critic_input_dim),
            hidden_dim=int(args.hidden_dim),
        )
        if action_value_enabled
        else None
    )
    lagged_q_enabled = float(args.offloading_lagged_q_coef) > 0.0
    offloading_lagged_q_critic = (
        CleanLaggedResidualQCritic(
            input_dim=int(offloading_actor.candidate_feature_dim) + int(critic_input_dim),
            hidden_dim=int(args.hidden_dim),
        )
        if lagged_q_enabled
        else None
    )
    modules = CleanTrainingModules(
        hgnn=CleanIncidenceHGNN(
            task_feature_dim=task_feature_dim,
            hidden_dim=int(args.hidden_dim),
            output_dim=int(args.task_embedding_dim),
        ),
        movement_actor=CleanMovementActor(task_embedding_dim=int(args.task_embedding_dim), hidden_dim=int(args.hidden_dim)),
        offloading_actor=offloading_actor,
        critic=CleanCentralizedCritic(
            input_dim=critic_input_dim,
            hidden_dim=int(args.hidden_dim),
        ),
        offloading_action_value_critic=offloading_action_value_critic,
        offloading_lagged_q_critic=offloading_lagged_q_critic,
    )
    device = torch.device(str(args.device))
    _move_modules_to_device(modules, device)
    optimizer = build_single_optimizer(modules, lr=float(args.lr))
    updater = CleanPPOUpdater(
        modules=modules,
        optimizer=optimizer,
        config=CleanPPOUpdateConfig(
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            clip_epsilon=float(args.clip_ratio),
            ppo_epochs=int(args.ppo_epochs),
            value_coef=float(args.value_coef),
            movement_entropy_coef=float(args.entropy_coef),
            offloading_entropy_coef=float(args.entropy_coef),
            max_grad_norm=float(args.max_grad_norm),
            detach_critic_hgnn=bool(args.detach_critic_hgnn),
            offloading_counterfactual_coef=float(args.offloading_counterfactual_coef),
            offloading_action_value_loss_coef=float(args.offloading_action_value_loss_coef),
            offloading_lagged_q_coef=float(args.offloading_lagged_q_coef),
            offloading_lagged_q_loss_coef=float(args.offloading_lagged_q_loss_coef),
        ),
        device=device,
    )
    checkpoint_manager = CleanCheckpointManager(run_dir / "checkpoints")
    logger = CleanJSONLLogger(run_dir)
    lagged_q_tracker = (
        CleanLaggedOutcomeTracker(
            scale_seconds=float(args.offloading_lagged_q_scale_seconds),
            censor_weight=float(args.offloading_lagged_q_censor_weight),
        )
        if lagged_q_enabled
        else None
    )
    start_episode = 0
    global_slot = 0
    latest_update_stats: CleanPPOUpdateStats | None = None

    if args.resume_checkpoint is not None:
        payload = checkpoint_manager.read(args.resume_checkpoint)
        validate_resume_experiment_controls(args, payload)
        checkpoint_manager.restore(modules=modules, optimizer=optimizer, payload=payload)
        start_episode = int(payload.get("episode", -1)) + 1
        global_slot = int(payload.get("global_slot", 0))
        updater.update_step = int(payload.get("update_step", 0))

    progress = _make_progress_bar(total=max(int(args.episodes) - start_episode, 0) * int(args.max_steps_per_episode))
    try:
        for episode in range(start_episode, int(args.episodes)):
            env.reset()
            if lagged_q_tracker is not None:
                lagged_q_tracker.start_episode(episode)
            graph_builder.reset()
            current_prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
            current_encoded = encode_prepared_slot(
                prepared_state=current_prepared,
                env=env,
                hgnn=modules.hgnn,
                critic=modules.critic,
                movement_actor=modules.movement_actor,
                device=device,
                detach_critic_hgnn=bool(args.detach_critic_hgnn),
            )
            buffer = CleanSlotRolloutBuffer()
            episode_reward = 0.0
            episode_component_totals: dict[str, float] = {
                "reward": 0.0,
                "time_penalty": 0.0,
                "dag_bonus": 0.0,
                "task_energy_penalty": 0.0,
                "movement_energy_penalty": 0.0,
            }
            last_info: dict[str, Any] = {}

            for episode_step in range(int(args.max_steps_per_episode)):
                slot_record, done, info = _collect_clean_slot(
                    env=env,
                    modules=modules,
                    encoded_state=current_encoded,
                    categorical_cls=Categorical,
                    device=device,
                    task_state_ready=TASK_STATE_READY_UNSCHEDULED,
                    freeze_movement=bool(args.freeze_movement),
                    lagged_q_enabled=bool(lagged_q_enabled),
                )
                global_slot += 1
                episode_reward += float(slot_record.reward)
                episode_component_totals["reward"] += float(info.get("step_reward", 0.0))
                episode_component_totals["time_penalty"] += float(info.get("step_time_penalty", 0.0))
                episode_component_totals["dag_bonus"] += float(info.get("step_completed_dag_bonus", 0.0))
                episode_component_totals["task_energy_penalty"] += float(info.get("step_task_energy_penalty", 0.0))
                episode_component_totals["movement_energy_penalty"] += float(info.get("step_movement_energy_penalty", 0.0))
                truncated = bool((episode_step + 1) >= int(args.max_steps_per_episode) and not done)
                slot_record.terminated = bool(done)
                slot_record.truncated = truncated
                info["terminated"] = bool(done)
                info["truncated"] = truncated
                buffer.append(slot_record)
                if lagged_q_tracker is not None:
                    lagged_q_tracker.register_rollout_actions(slot_record=slot_record, env=env)
                    lagged_q_tracker.resolve_completed(env=env)

                next_prepared = None
                next_encoded_old = None
                if not done:
                    next_prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
                    next_encoded_old = encode_prepared_slot(
                        prepared_state=next_prepared,
                        env=env,
                        hgnn=modules.hgnn,
                        critic=modules.critic,
                        movement_actor=modules.movement_actor,
                        device=device,
                        detach_critic_hgnn=bool(args.detach_critic_hgnn),
                    )

                should_update = len(buffer) >= int(args.rollout_horizon) or bool(done) or truncated
                if should_update:
                    lagged_tracker_summary: dict[str, int] | None = None
                    if lagged_q_tracker is not None and bool(done or truncated):
                        lagged_q_tracker.resolve_completed(env=env)
                        lagged_q_tracker.finalize_censored(
                            episode_end_time=float(env.current_time_seconds)
                        )
                    lagged_q_samples = (
                        lagged_q_tracker.pop_finalized()
                        if lagged_q_tracker is not None
                        else []
                    )
                    close_rollout_with_bootstrap(buffer=buffer, next_encoded_state=next_encoded_old, terminated=bool(done))
                    latest_update_stats = updater.update(
                        buffer,
                        lagged_q_samples=lagged_q_samples,
                        lagged_q_pending_count=(
                            lagged_q_tracker.pending_count if lagged_q_tracker is not None else 0
                        ),
                    )
                    if lagged_q_tracker is not None and bool(done or truncated):
                        lagged_tracker_summary = lagged_q_tracker.finish_episode()
                    write_clean_training_log(
                        logger,
                        episode=episode,
                        global_slot=global_slot,
                        info=info,
                        update_stats=latest_update_stats,
                        extra=_episode_diagnostics_payload(
                            episode_component_totals=episode_component_totals,
                            terminal=bool(done or truncated),
                            env=env,
                            completed_dag_weight=float(args.completed_dag_weight),
                            detach_critic_hgnn=bool(args.detach_critic_hgnn),
                            freeze_ue_mobility=bool(args.freeze_ue_mobility),
                            offloading_counterfactual_coef=float(args.offloading_counterfactual_coef),
                            offloading_action_value_loss_coef=float(args.offloading_action_value_loss_coef),
                            offloading_lagged_q_coef=float(args.offloading_lagged_q_coef),
                            offloading_lagged_q_loss_coef=float(args.offloading_lagged_q_loss_coef),
                            lagged_tracker_summary=lagged_tracker_summary,
                        ),
                    )
                    checkpoint_manager.save(
                        modules=modules,
                        optimizer=optimizer,
                        episode=episode,
                        global_slot=global_slot,
                        update_step=updater.update_step,
                        config_snapshot=build_config_snapshot(args),
                        safe_boundary=buffer.checkpoint_safe,
                        filename="latest.pt",
                    )
                    if not done and not truncated and next_prepared is not None:
                        current_prepared = next_prepared
                        current_encoded = reencode_prepared_after_update(
                            prepared_state=current_prepared,
                            env=env,
                            modules=modules,
                            device=device,
                            detach_critic_hgnn=bool(args.detach_critic_hgnn),
                        )
                        buffer = CleanSlotRolloutBuffer()
                    else:
                        last_info = info
                        break
                else:
                    assert next_prepared is not None and next_encoded_old is not None
                    current_prepared = next_prepared
                    current_encoded = next_encoded_old

                last_info = info
                _update_progress(progress, episode, global_slot, episode_reward, info, latest_update_stats)
                progress.update(1)

            if int(args.checkpoint_interval) > 0 and (episode + 1) % int(args.checkpoint_interval) == 0:
                checkpoint_manager.save(
                    modules=modules,
                    optimizer=optimizer,
                    episode=episode,
                    global_slot=global_slot,
                    update_step=updater.update_step,
                    config_snapshot=build_config_snapshot(args),
                    safe_boundary=True,
                    filename=f"checkpoint_ep_{episode + 1:04d}.pt",
                )
            _write_json(
                run_dir / "run_summary.json",
                {
                    "status": "running" if episode + 1 < int(args.episodes) else "completed",
                    "episode": episode,
                    "global_slot": global_slot,
                    "episode_reward": episode_reward,
                    "latest_info": _jsonable(last_info),
                    "latest_update": None if latest_update_stats is None else asdict(latest_update_stats),
                    "completed_dag_weight": float(args.completed_dag_weight),
                    "detach_critic_hgnn": bool(args.detach_critic_hgnn),
                    "freeze_ue_mobility": bool(args.freeze_ue_mobility),
                    "offloading_counterfactual_coef": float(args.offloading_counterfactual_coef),
                    "offloading_action_value_loss_coef": float(args.offloading_action_value_loss_coef),
                    "offloading_lagged_q_coef": float(args.offloading_lagged_q_coef),
                    "offloading_lagged_q_loss_coef": float(args.offloading_lagged_q_loss_coef),
                    "offloading_lagged_q_scale_seconds": float(args.offloading_lagged_q_scale_seconds),
                    "offloading_lagged_q_censor_weight": float(args.offloading_lagged_q_censor_weight),
                    "initial_hotspot_ue_count": int(env.initial_hotspot_ue_count),
                    "resume_semantics": "restart_from_new_episode_only",
                },
            )
    finally:
        progress.close()
        graph_builder.close()

    return {
        "run_dir": str(run_dir),
        "global_slot": global_slot,
        "latest_update": None if latest_update_stats is None else asdict(latest_update_stats),
        "completed_dag_weight": float(args.completed_dag_weight),
        "detach_critic_hgnn": bool(args.detach_critic_hgnn),
        "freeze_ue_mobility": bool(args.freeze_ue_mobility),
        "offloading_counterfactual_coef": float(args.offloading_counterfactual_coef),
        "offloading_action_value_loss_coef": float(args.offloading_action_value_loss_coef),
        "offloading_lagged_q_coef": float(args.offloading_lagged_q_coef),
        "offloading_lagged_q_loss_coef": float(args.offloading_lagged_q_loss_coef),
        "offloading_lagged_q_scale_seconds": float(args.offloading_lagged_q_scale_seconds),
        "offloading_lagged_q_censor_weight": float(args.offloading_lagged_q_censor_weight),
        "kahypar_circuit_open": bool(graph_builder.kahypar_circuit_open),
        "kahypar_last_failure_reason": graph_builder.kahypar_last_failure_reason,
        "kahypar_cleanup_failed": bool(graph_builder.kahypar_cleanup_failed),
        "kahypar_worker_alive_after_close": bool(graph_builder.kahypar_worker_alive),
    }


def _collect_clean_slot(
    *,
    env: Env,
    modules: CleanTrainingModules,
    encoded_state: Any,
    categorical_cls: Any,
    device: Any,
    task_state_ready: str,
    freeze_movement: bool = False,
    lagged_q_enabled: bool = False,
) -> tuple[Any, bool, dict[str, Any]]:
    import torch

    movement_obs = encoded_state.movement_observation
    movement_actions: dict[int, int] = {}
    movement_records: list[CleanMovementRolloutRecord] = []
    if freeze_movement:
        # Phase 4 P1 frozen-movement baseline: force hover (action index 0),
        # record NO movement rollout entries so the PPO movement loss/entropy
        # terms see zero actions (trainer already averages only over slots with
        # movement records). Offloading/critic/HGNN training is unaffected.
        #
        # RNG alignment (Phase 4 Commit 2): draw and DISCARD the same movement
        # sample the learned path would consume, so the torch RNG stream feeding
        # the subsequent offloading sampling stays aligned between frozen and
        # learned runs of the same seed. This aligns sampling-stream consumption
        # only; trajectories still diverge through the movement treatment itself.
        movement_dist = categorical_cls(logits=encoded_state.movement_logits)
        _ = movement_dist.sample()
        hover_action = 0
        assert config.CLEAN_MOVEMENT_ACTIONS[hover_action] == config.CLEAN_MOVEMENT_HOVER_ACTION
        for uav_id in movement_obs.uav_ids:
            movement_actions[int(uav_id)] = hover_action
        return _finish_collect_clean_slot(
            env=env,
            modules=modules,
            encoded_state=encoded_state,
            task_state_ready=task_state_ready,
            movement_actions=movement_actions,
            movement_records=movement_records,
            movement_frozen=True,
            lagged_q_enabled=bool(lagged_q_enabled),
        )

    movement_dist = categorical_cls(logits=encoded_state.movement_logits)
    selected_movement = movement_dist.sample()
    movement_log_probs = movement_dist.log_prob(selected_movement)
    movement_entropy = movement_dist.entropy()
    for idx, uav_id in enumerate(movement_obs.uav_ids):
        action = int(selected_movement[idx].item())
        movement_actions[int(uav_id)] = action
        movement_records.append(
            CleanMovementRolloutRecord(
                uav_id=int(uav_id),
                uav_index=int(idx),
                uav_features=np.asarray(movement_obs.uav_features[idx], dtype=np.float32).copy(),
                ready_task_indices=list(movement_obs.ready_task_indices),
                pending_task_indices=list(movement_obs.pending_task_indices),
                ready_count_normalized=float(movement_obs.ready_count_normalized),
                pending_count_normalized=float(movement_obs.pending_count_normalized),
                movement_mask=np.asarray(movement_obs.boundary_action_mask[idx], dtype=bool).copy(),
                selected_action=action,
                old_log_probability=float(movement_log_probs[idx].detach().cpu().item()),
                entropy=float(movement_entropy[idx].detach().cpu().item()),
            )
        )

    return _finish_collect_clean_slot(
        env=env,
        modules=modules,
        encoded_state=encoded_state,
        task_state_ready=task_state_ready,
        movement_actions=movement_actions,
        movement_records=movement_records,
        movement_frozen=False,
        lagged_q_enabled=bool(lagged_q_enabled),
    )


def _finish_collect_clean_slot(
    *,
    env: Env,
    modules: CleanTrainingModules,
    encoded_state: Any,
    task_state_ready: str,
    movement_actions: dict[int, int],
    movement_records: list[CleanMovementRolloutRecord],
    movement_frozen: bool,
    lagged_q_enabled: bool = False,
) -> tuple[Any, bool, dict[str, Any]]:
    assignment_time_seconds = float(env.current_time_seconds)
    env.apply_movement(movement_actions)
    frozen_ready_tasks = [env.task_manager.get_task(task_id) for task_id in encoded_state.prepared_state.frozen_ready_task_ids]
    frozen_ready_tasks = [task for task in frozen_ready_tasks if task is not None and task.state == task_state_ready]
    assignment_buffer = modules.offloading_actor.act(
        frozen_ready_tasks=frozen_ready_tasks,
        task_embeddings=encoded_state.task_embeddings.detach() if hasattr(encoded_state.task_embeddings, "detach") else encoded_state.task_embeddings,
        graph_snapshot=encoded_state.prepared_state.graph_snapshot,
        task_manager=env.task_manager,
        uavs=env.uavs,
        executor=env.executor,
        current_time_seconds=env.current_time_seconds,
        uav_service_positions=env.uav_service_positions,
        ue_service_positions=env.ue_service_positions,
        ues=env.ues,
        deterministic=False,
    )
    offloading_records = [
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
            dag_id=str(record.dag_id) if lagged_q_enabled else None,
            assignment_time_seconds=assignment_time_seconds if lagged_q_enabled else None,
            candidate_features=(
                record.candidate_features.detach().cpu().numpy().copy()
                if lagged_q_enabled
                else None
            ),
            critic_global_context=(
                encoded_state.critic_global_input.detach().cpu().numpy().copy()
                if lagged_q_enabled and hasattr(encoded_state.critic_global_input, "detach")
                else (
                    np.asarray(encoded_state.critic_global_input, dtype=np.float32).copy()
                    if lagged_q_enabled
                    else None
                )
            ),
            selected_estimated_finish_time=(
                float(record.selected_estimated_finish_time) if lagged_q_enabled else None
            ),
            selected_estimated_incremental_delay=(
                float(record.selected_estimated_incremental_delay) if lagged_q_enabled else None
            ),
        )
        for record in modules.offloading_actor.latest_records
    ]

    _, _, done, info = env.commit_and_advance(assignment_buffer=assignment_buffer)
    slot_record = make_slot_rollout_record(encoded_state=encoded_state)
    slot_record.movement_records = movement_records
    slot_record.offloading_records = offloading_records
    slot_record.reward = float(info["step_reward"])
    if movement_frozen:
        info["movement_action_distribution"] = {
            str(action): (1.0 if str(action) == str(config.CLEAN_MOVEMENT_HOVER_ACTION) else 0.0)
            for action in config.CLEAN_MOVEMENT_ACTIONS
        }
    else:
        info["movement_action_distribution"] = _movement_action_distribution(movement_records)
    info["movement_frozen"] = bool(movement_frozen)
    info["offloading_action_count"] = len(offloading_records)
    graph_snapshot = encoded_state.prepared_state.graph_snapshot
    partition_status = str(getattr(graph_snapshot, "partition_status", "disabled"))
    info["kahypar_partition_status"] = partition_status
    info["kahypar_partition_hyperedge_count"] = int(len(graph_snapshot.partition_hyperedges))
    if partition_status.startswith("degraded"):
        info["kahypar_degraded_label"] = str(config.KAHYPAR_DEGRADED_EXPERIMENT_LABEL)
    return slot_record, bool(done), info


def _episode_diagnostics_payload(
    *,
    episode_component_totals: dict[str, float],
    terminal: bool,
    env: Env,
    completed_dag_weight: float,
    detach_critic_hgnn: bool,
    freeze_ue_mobility: bool,
    offloading_counterfactual_coef: float,
    offloading_action_value_loss_coef: float,
    offloading_lagged_q_coef: float,
    offloading_lagged_q_loss_coef: float,
    lagged_tracker_summary: dict[str, int] | None,
) -> dict[str, Any]:
    """Episode-level reward component accumulation (diagnostics only).

    Mid-episode update boundaries (e.g. slot 128) report `episode_*_so_far`;
    the terminal boundary (done/truncated) additionally reports
    `episode_*_total` plus the counterfactual unsettled backlog estimate.
    """
    payload: dict[str, Any] = {
        f"episode_{key}_so_far": float(value) for key, value in episode_component_totals.items()
    }
    payload["episode_terminal_record"] = bool(terminal)
    payload["completed_dag_weight"] = float(completed_dag_weight)
    payload["detach_critic_hgnn"] = bool(detach_critic_hgnn)
    payload["freeze_ue_mobility"] = bool(freeze_ue_mobility)
    payload["offloading_counterfactual_coef"] = float(offloading_counterfactual_coef)
    payload["offloading_action_value_loss_coef"] = float(offloading_action_value_loss_coef)
    payload["offloading_lagged_q_coef"] = float(offloading_lagged_q_coef)
    payload["offloading_lagged_q_loss_coef"] = float(offloading_lagged_q_loss_coef)
    payload["initial_hotspot_ue_count"] = int(env.initial_hotspot_ue_count)
    if terminal:
        payload.update(
            {f"episode_{key}_total": float(value) for key, value in episode_component_totals.items()}
        )
        payload.update(_unsettled_backlog_estimate(env))
        if lagged_tracker_summary is not None:
            payload.update(
                {f"lagged_q_episode_{key}": int(value) for key, value in lagged_tracker_summary.items()}
            )
    return payload


def _unsettled_backlog_estimate(env: Env) -> dict[str, Any]:
    """Counterfactual estimate of reward-relevant backlog at horizon end.

    For every active unfinished (not reward-settled) task, estimate the delay
    it WOULD contribute if it reward-completed right now (now - ready/arrival
    time), plus the same clipped/weighted norm-time cost the reward would
    charge. These are estimates for diagnostics; they never enter the reward.
    """
    now = float(env.current_time_seconds)
    time_ref = max(float(config.CLEAN_REWARD_TIME_REF), 1.0)
    time_clip = float(getattr(config, "CLEAN_REWARD_TIME_CLIP", float("inf")))
    unsettled_tasks = 0
    delay_sum = 0.0
    norm_cost = 0.0
    for task in env.task_manager.get_active_tasks():
        if bool(getattr(task, "reward_settled", False)):
            continue
        unsettled_tasks += 1
        reference = task.ready_time if task.ready_time is not None else task.arrival_time
        delay = max(now - float(reference), 0.0)
        delay_sum += delay
        weight = (
            float(config.CRITICAL_TASK_WEIGHT)
            if bool(getattr(task, "is_critical_path", False))
            else float(config.NONCRITICAL_TASK_WEIGHT)
        )
        norm_cost += weight * min(delay / time_ref, time_clip)
    unsettled_dags = sum(1 for job in env.task_manager.jobs.values() if not job.completed)
    return {
        "unsettled_task_count": int(unsettled_tasks),
        "unsettled_dag_count": int(unsettled_dags),
        "unsettled_delay_seconds_sum_estimate": round(delay_sum, 2),
        "unsettled_norm_time_cost_estimate": round(norm_cost, 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = apply_smoke_overrides(parser.parse_args(argv))
    try:
        result = run_training(args)
    except ModuleNotFoundError as exc:
        if exc.name == "torch" or "torch" in str(exc).lower():
            print("clean mainline training requires torch; install/use a torch environment before running training.", file=sys.stderr)
            return 2
        raise
    except RuntimeError as exc:
        if "torch" in str(exc).lower():
            print(f"clean mainline training requires torch: {exc}", file=sys.stderr)
            return 2
        raise
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch.distributions import Categorical
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise ModuleNotFoundError("torch is required for clean mainline training") from exc
        raise
    return torch, Categorical


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


def _move_modules_to_device(modules: CleanTrainingModules, device: Any) -> None:
    modules.hgnn.to(device)
    modules.movement_actor.to(device)
    modules.offloading_actor.to(device)
    modules.critic.to(device)
    if modules.offloading_action_value_critic is not None:
        modules.offloading_action_value_critic.to(device)
    if modules.offloading_lagged_q_critic is not None:
        modules.offloading_lagged_q_critic.to(device)


def _make_progress_bar(total: int) -> Any:
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        return _PlainProgress(total)
    return tqdm(total=int(total), desc="clean-mainline", unit="slot")


def _update_progress(
    progress: Any,
    episode: int,
    global_slot: int,
    episode_reward: float,
    info: dict[str, Any],
    stats: CleanPPOUpdateStats | None,
) -> None:
    postfix = {
        "episode": int(episode),
        "global_slot": int(global_slot),
        "recent_reward": round(float(info.get("step_reward", 0.0)), 4),
        "avg_reward": round(float(episode_reward) / max(int(info.get("time_step", 1)), 1), 4),
        "completion_rate": info.get("dag_completion_rate"),
        "throughput": info.get("dag_throughput"),
        "invalid_rate": info.get("invalid_assignment_rate"),
        "hover_rate": _hover_rate(info.get("movement_action_distribution", {})),
    }
    if stats is not None:
        postfix["policy_loss"] = round(float(stats.movement_loss + stats.offloading_loss), 4)
        postfix["value_loss"] = round(float(stats.value_loss), 4)
    if hasattr(progress, "set_postfix"):
        progress.set_postfix(postfix)


class _PlainProgress:
    def __init__(self, total: int) -> None:
        self.total = int(total)
        self.current = 0

    def update(self, amount: int = 1) -> None:
        self.current += int(amount)
        if self.current == self.total or self.current % 20 == 0:
            pct = 100.0 * float(self.current) / max(float(self.total), 1.0)
            print(f"clean-mainline {self.current}/{self.total} slots ({pct:.1f}%)")

    def set_postfix(self, values: dict[str, Any]) -> None:
        del values

    def close(self) -> None:
        pass


def _movement_action_distribution(records: list[CleanMovementRolloutRecord]) -> dict[str, float]:
    counts = {str(action): 0 for action in config.CLEAN_MOVEMENT_ACTIONS}
    for record in records:
        action_name = str(config.CLEAN_MOVEMENT_ACTIONS[int(record.selected_action)])
        counts[action_name] = counts.get(action_name, 0) + 1
    total = max(float(len(records)), 1.0)
    return {action: float(count) / total for action, count in counts.items()}


def _hover_rate(distribution: Any) -> Any:
    if not isinstance(distribution, dict):
        return None
    return distribution.get(config.CLEAN_MOVEMENT_HOVER_ACTION)


def _namespace_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in vars(args).items():
        output[key] = str(value) if isinstance(value, Path) else value
    return output


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


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
