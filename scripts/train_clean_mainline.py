from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import random
import sys
import time
import traceback
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.env import Env
from environment.graph_builder import CleanGraphBuilder
from marl_models.mappo.clean_ppo import (
    CLEAN_CRITIC_TASK_POOLING_CHOICES,
    normalize_clean_critic_task_pooling,
)
from marl_models.mappo.clean_slot_orchestrator import (
    CleanMovementRolloutRecord,
    CleanOffloadingRolloutRecord,
    CleanSlotRolloutBuffer,
    encode_prepared_slot,
    make_slot_rollout_record,
    prepare_slot_state,
)
from marl_models.mappo.clean_trainer import (
    CLEAN_COUNTERFACTUAL_BETA,
    CLEAN_COUNTERFACTUAL_CREDIT_MODE,
    CLEAN_COUNTERFACTUAL_GRADIENT_CLIPPING,
    CLEAN_COUNTERFACTUAL_Q_LOSS_COEF,
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

TASK_ENCODER_LEGACY_HGNN = "hgnn"
TASK_ENCODER_MLP = "mlp"
TASK_ENCODER_CURRENT_MEAN_HGNN = "current_mean_hgnn"
TASK_ENCODER_STANDARD_WEIGHTED_HGNN = "standard_weighted_hgnn"
TASK_ENCODER_TYPED_GATED_HGNN = "typed_gated_hgnn"
BANDIT_CHECKPOINT_SCHEMA = "greedy_eft_contextual_bandit_checkpoint_v1"
TASK_ENCODER_CHOICES = (
    TASK_ENCODER_LEGACY_HGNN,
    TASK_ENCODER_MLP,
    TASK_ENCODER_CURRENT_MEAN_HGNN,
    TASK_ENCODER_STANDARD_WEIGHTED_HGNN,
    TASK_ENCODER_TYPED_GATED_HGNN,
)


def _normalize_task_encoder_for_comparison(task_encoder: str) -> str:
    value = str(task_encoder)
    if value == TASK_ENCODER_LEGACY_HGNN:
        return TASK_ENCODER_CURRENT_MEAN_HGNN
    if value not in TASK_ENCODER_CHOICES:
        raise ValueError(f"unsupported task encoder: {value}")
    return value


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


def _positive_int(value: str | int) -> int:
    resolved = int(value)
    if resolved <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return resolved


def _nonnegative_int(value: str | int) -> int:
    resolved = int(value)
    if resolved < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return resolved


def _positive_finite_float(value: str | float) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return resolved


def _parse_update_counts(value: str) -> set[int]:
    text = str(value).strip()
    if not text:
        return set()
    counts: set[int] = set()
    for token in text.split(","):
        count = int(token.strip())
        if count < 0:
            raise ValueError("checkpoint update counts must be non-negative")
        counts.add(count)
    return counts


def _module_state_sha256(module: Any) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _initialize_offloading_policy(
    *,
    args: argparse.Namespace,
    modules: CleanTrainingModules,
    torch: Any,
) -> dict[str, Any]:
    checkpoint = getattr(args, "offloading_init_bandit_checkpoint", None)
    expected_checksum = getattr(
        args, "offloading_init_bandit_dataset_checksum", None
    )
    if checkpoint is None:
        if expected_checksum is not None:
            raise ValueError(
                "bandit dataset checksum requires --offloading-init-bandit-checkpoint"
            )
        return {
            "mode": "random",
            "training_seed": int(args.seed),
            "task_encoder_state_sha256": _module_state_sha256(modules.hgnn),
            "candidate_scorer_state_sha256": _module_state_sha256(
                modules.offloading_actor.scorer
            ),
        }
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"bandit initialization checkpoint is missing: {checkpoint_path}"
        )
    if str(args.task_encoder) != "mlp":
        raise ValueError("bandit initialization requires the MLP task encoder")
    if expected_checksum is None or not str(expected_checksum):
        raise ValueError(
            "strict bandit initialization requires a dataset checksum"
        )
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    expected = {
        "schema": BANDIT_CHECKPOINT_SCHEMA,
        "stage": "trained",
        "encoder": "mlp",
        "training_seed": int(args.seed),
        "dataset_checksum": str(expected_checksum),
        "task_embedding_dim": int(args.task_embedding_dim),
        "hidden_dim": int(args.hidden_dim),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"bandit initialization {key} mismatch: "
                f"expected {value!r}, got {payload.get(key)!r}"
            )
    modules.hgnn.load_state_dict(payload["task_encoder_state_dict"], strict=True)
    modules.offloading_actor.scorer.load_state_dict(
        payload["candidate_scorer_state_dict"], strict=True
    )
    return {
        "mode": "bandit_checkpoint",
        "training_seed": int(args.seed),
        "dataset_checksum": str(expected_checksum),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "task_encoder_state_sha256": _module_state_sha256(modules.hgnn),
        "candidate_scorer_state_sha256": _module_state_sha256(
            modules.offloading_actor.scorer
        ),
        "optimizer_state_loaded": False,
    }


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


def clean_counterfactual_experiment_controls(args: argparse.Namespace) -> dict[str, Any]:
    enabled = bool(getattr(args, "clean_counterfactual_credit", False))
    return {
        "clean_counterfactual_credit": enabled,
        "clean_counterfactual_credit_mode": (
            CLEAN_COUNTERFACTUAL_CREDIT_MODE if enabled else "disabled"
        ),
        "counterfactual_beta": CLEAN_COUNTERFACTUAL_BETA if enabled else 0.0,
        "counterfactual_q_loss_coef": (
            CLEAN_COUNTERFACTUAL_Q_LOSS_COEF if enabled else 0.0
        ),
        "gradient_clipping": (
            CLEAN_COUNTERFACTUAL_GRADIENT_CLIPPING if enabled else "legacy_global"
        ),
    }


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


def validate_clean_counterfactual_credit_controls(args: argparse.Namespace) -> None:
    if not bool(getattr(args, "clean_counterfactual_credit", False)):
        return
    legacy_beta, legacy_eta = _resolved_offloading_action_value_controls(args)
    if legacy_beta > 0.0 or legacy_eta > 0.0:
        raise ValueError(
            "--clean-counterfactual-credit cannot combine with the legacy "
            "counterfactual coefficient path"
        )
    lagged_beta, lagged_eta, _, _ = _resolved_offloading_lagged_q_controls(args)
    if lagged_beta > 0.0 or lagged_eta > 0.0:
        raise ValueError("--clean-counterfactual-credit cannot combine with lagged-Q")
    if bool(getattr(args, "decision_critic", False)):
        raise ValueError("--clean-counterfactual-credit cannot combine with decision critic")
    if bool(getattr(args, "offloading_eft_advantage", False)):
        raise ValueError("--clean-counterfactual-credit cannot combine with EFT advantage")
    if float(getattr(args, "eft_auxiliary_lambda_initial", 0.0)) > 0.0:
        raise ValueError("--clean-counterfactual-credit cannot combine with EFT auxiliary loss")
    if getattr(args, "offloading_init_bandit_checkpoint", None) is not None:
        raise ValueError(
            "--clean-counterfactual-credit cannot combine with bandit initialization"
        )


def checkpoint_experiment_controls(payload: dict[str, Any]) -> dict[str, Any]:
    """Resolve experiment controls from new or legacy clean checkpoints."""
    checkpoint_config = payload.get("config", {})
    cli = checkpoint_config.get("cli", {}) if isinstance(checkpoint_config, dict) else {}
    experiment_controls = (
        checkpoint_config.get("experiment_controls", {})
        if isinstance(checkpoint_config, dict)
        else {}
    )
    if not isinstance(cli, dict):
        cli = {}
    if not isinstance(experiment_controls, dict):
        experiment_controls = {}
    clean_counterfactual_credit = cli.get(
        "clean_counterfactual_credit",
        experiment_controls.get("clean_counterfactual_credit", False),
    )
    if not isinstance(clean_counterfactual_credit, bool):
        raise ValueError("checkpoint clean_counterfactual_credit must be boolean")
    expected_clean_controls = clean_counterfactual_experiment_controls(
        argparse.Namespace(clean_counterfactual_credit=clean_counterfactual_credit)
    )
    for key in (
        "clean_counterfactual_credit_mode",
        "counterfactual_beta",
        "counterfactual_q_loss_coef",
        "gradient_clipping",
    ):
        if key in experiment_controls and experiment_controls[key] != expected_clean_controls[key]:
            raise ValueError(
                f"checkpoint {key} is inconsistent with clean counterfactual mode"
            )
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
    if clean_counterfactual_credit and (counterfactual_coef > 0.0 or lagged_q_coef > 0.0):
        raise ValueError(
            "checkpoint clean counterfactual credit cannot combine with legacy counterfactual or lagged-Q"
        )
    normalize_value_targets = cli.get("normalize_value_targets", False)
    if not isinstance(normalize_value_targets, bool):
        raise ValueError("checkpoint normalize_value_targets must be boolean")
    value_clip_epsilon = float(cli.get("value_clip_epsilon", 0.0))
    if not math.isfinite(value_clip_epsilon) or value_clip_epsilon < 0.0:
        raise ValueError("checkpoint value_clip_epsilon must be finite and non-negative")
    task_encoder = str(cli.get("task_encoder", "hgnn"))
    _normalize_task_encoder_for_comparison(task_encoder)
    critic_task_pooling = normalize_clean_critic_task_pooling(
        cli.get(
            "critic_task_pooling",
            experiment_controls.get("critic_task_pooling", "mean"),
        )
    )
    num_envs = int(cli.get("num_envs", 1))
    if num_envs <= 0:
        raise ValueError("checkpoint num_envs must be positive")
    sampler_backend = str(cli.get("sampler_backend", "synchronous"))
    if sampler_backend not in {"synchronous", "process"}:
        raise ValueError(f"unsupported checkpoint sampler backend: {sampler_backend}")
    eft_auxiliary_lambda_initial = float(
        cli.get("eft_auxiliary_lambda_initial", 0.0)
    )
    eft_auxiliary_regret_scale = float(
        cli.get("eft_auxiliary_regret_scale", 1.0)
    )
    offloading_eft_advantage = cli.get("offloading_eft_advantage", False)
    if not isinstance(offloading_eft_advantage, bool):
        raise ValueError("checkpoint offloading_eft_advantage must be boolean")
    offloading_lr_scale = float(cli.get("offloading_lr_scale", 1.0))
    if not math.isfinite(offloading_lr_scale) or offloading_lr_scale <= 0.0:
        raise ValueError("checkpoint offloading_lr_scale must be finite and positive")
    movement_position_advantage = cli.get("movement_position_advantage", False)
    if not isinstance(movement_position_advantage, bool):
        raise ValueError("checkpoint movement_position_advantage must be boolean")
    movement_lr_scale = float(cli.get("movement_lr_scale", 1.0))
    if not math.isfinite(movement_lr_scale) or movement_lr_scale <= 0.0:
        raise ValueError("checkpoint movement_lr_scale must be finite and positive")
    decision_critic_enabled = cli.get("decision_critic_enabled", False)
    if not isinstance(decision_critic_enabled, bool):
        raise ValueError("checkpoint decision_critic_enabled must be boolean")
    decision_critic_coef = float(cli.get("decision_critic_coef", 0.5))
    decision_critic_discount = float(cli.get("decision_critic_discount", 0.99))
    if not math.isfinite(decision_critic_coef) or decision_critic_coef < 0.0:
        raise ValueError("checkpoint decision_critic_coef must be finite and non-negative")
    if not math.isfinite(decision_critic_discount) or decision_critic_discount < 0.0:
        raise ValueError("checkpoint decision_critic_discount must be finite and non-negative")
    if not math.isfinite(eft_auxiliary_lambda_initial) or eft_auxiliary_lambda_initial < 0.0:
        raise ValueError("checkpoint EFT auxiliary lambda must be finite and non-negative")
    if not math.isfinite(eft_auxiliary_regret_scale) or eft_auxiliary_regret_scale <= 0.0:
        raise ValueError("checkpoint EFT auxiliary scale must be finite and positive")
    freeze_movement = cli.get("freeze_movement", False)
    if not isinstance(freeze_movement, bool):
        raise ValueError("checkpoint freeze_movement must be boolean")
    offloading_initialization = experiment_controls.get(
        "offloading_initialization", {"mode": "random"}
    )
    if not isinstance(offloading_initialization, dict):
        raise ValueError("checkpoint offloading initialization identity must be a mapping")
    if clean_counterfactual_credit:
        if decision_critic_enabled:
            raise ValueError(
                "checkpoint clean counterfactual credit cannot combine with decision critic"
            )
        if offloading_eft_advantage or eft_auxiliary_lambda_initial > 0.0:
            raise ValueError(
                "checkpoint clean counterfactual credit cannot combine with EFT guidance"
            )
        if str(offloading_initialization.get("mode", "random")) != "random":
            raise ValueError(
                "checkpoint clean counterfactual credit cannot use bandit initialization"
            )
    return {
        **expected_clean_controls,
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
        "normalize_value_targets": normalize_value_targets,
        "value_clip_epsilon": value_clip_epsilon,
        "task_encoder": task_encoder,
        "critic_task_pooling": critic_task_pooling,
        "num_envs": num_envs,
        "sampler_backend": sampler_backend,
        "freeze_movement": freeze_movement,
        "eft_auxiliary_lambda_initial": eft_auxiliary_lambda_initial,
        "eft_auxiliary_regret_scale": eft_auxiliary_regret_scale,
        "offloading_eft_advantage": offloading_eft_advantage,
        "offloading_lr_scale": offloading_lr_scale,
        "movement_position_advantage": movement_position_advantage,
        "movement_lr_scale": movement_lr_scale,
        "decision_critic_enabled": decision_critic_enabled,
        "decision_critic_coef": decision_critic_coef,
        "decision_critic_discount": decision_critic_discount,
        "offloading_initialization": dict(offloading_initialization),
    }


def validate_resume_experiment_controls(
    args: argparse.Namespace,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Reject a resume that would silently change its reward objective."""
    saved = checkpoint_experiment_controls(payload)
    requested_clean_counterfactual = bool(
        getattr(args, "clean_counterfactual_credit", False)
    )
    if requested_clean_counterfactual != bool(saved["clean_counterfactual_credit"]):
        raise ValueError(
            "resume checkpoint clean counterfactual credit mismatch: "
            f"requested {requested_clean_counterfactual}, "
            f"checkpoint {saved['clean_counterfactual_credit']}"
        )
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
    requested_normalization = bool(getattr(args, "normalize_value_targets", True))
    if requested_normalization != bool(saved["normalize_value_targets"]):
        raise ValueError(
            "resume checkpoint value-target normalization mismatch: "
            f"requested {requested_normalization}, checkpoint {saved['normalize_value_targets']}"
        )
    requested_value_clip = float(getattr(args, "value_clip_epsilon", 0.2))
    if not math.isclose(
        requested_value_clip,
        float(saved["value_clip_epsilon"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "resume checkpoint value clip mismatch: "
            f"requested {requested_value_clip}, checkpoint {saved['value_clip_epsilon']}"
        )
    requested_task_encoder = str(getattr(args, "task_encoder", "hgnn"))
    if _normalize_task_encoder_for_comparison(requested_task_encoder) != _normalize_task_encoder_for_comparison(
        str(saved["task_encoder"])
    ):
        raise ValueError(
            "resume checkpoint task encoder mismatch: "
            f"requested {requested_task_encoder}, checkpoint {saved['task_encoder']}"
        )
    requested_critic_task_pooling = normalize_clean_critic_task_pooling(
        getattr(args, "critic_task_pooling", "mean")
    )
    if requested_critic_task_pooling != str(saved["critic_task_pooling"]):
        raise ValueError(
            "resume checkpoint critic task pooling mismatch: "
            f"requested {requested_critic_task_pooling}, checkpoint {saved['critic_task_pooling']}"
        )
    requested_num_envs = int(getattr(args, "num_envs", 1))
    if requested_num_envs != int(saved["num_envs"]):
        raise ValueError(
            "resume checkpoint sampler environment count mismatch: "
            f"requested {requested_num_envs}, checkpoint {saved['num_envs']}"
        )
    requested_sampler_backend = str(getattr(args, "sampler_backend", "synchronous"))
    if requested_sampler_backend != str(saved["sampler_backend"]):
        raise ValueError(
            "resume checkpoint sampler backend mismatch: "
            f"requested {requested_sampler_backend}, checkpoint {saved['sampler_backend']}"
        )
    requested_freeze_movement = bool(getattr(args, "freeze_movement", False))
    if requested_freeze_movement != bool(saved["freeze_movement"]):
        raise ValueError(
            "resume checkpoint forced-movement mode mismatch: "
            f"requested freeze={requested_freeze_movement}, "
            f"checkpoint freeze={saved['freeze_movement']}"
        )
    for key, requested in (
        (
            "eft_auxiliary_lambda_initial",
            float(getattr(args, "eft_auxiliary_lambda_initial", 0.0)),
        ),
        (
            "eft_auxiliary_regret_scale",
            float(getattr(args, "eft_auxiliary_regret_scale", 1.0)),
        ),
        (
            "offloading_eft_advantage",
            bool(getattr(args, "offloading_eft_advantage", False)),
        ),
        (
            "offloading_lr_scale",
            float(getattr(args, "offloading_lr_scale", 1.0)),
        ),
        (
            "movement_position_advantage",
            bool(getattr(args, "movement_position_advantage", False)),
        ),
        (
            "movement_lr_scale",
            float(getattr(args, "movement_lr_scale", 1.0)),
        ),
        (
            "decision_critic_enabled",
            bool(getattr(args, "decision_critic", False)),
        ),
        (
            "decision_critic_coef",
            float(getattr(args, "decision_critic_coef", 0.5)),
        ),
        (
            "decision_critic_discount",
            float(getattr(args, "decision_critic_discount", 0.99)),
        ),
    ):
        if isinstance(requested, bool):
            if requested != bool(saved[key]):
                raise ValueError(
                    f"resume checkpoint {key.replace('_', ' ')} mismatch: "
                    f"requested {requested}, checkpoint {bool(saved[key])}"
                )
        elif not math.isclose(requested, float(saved[key]), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"resume checkpoint {key.replace('_', ' ')} mismatch: "
                f"requested {requested}, checkpoint {saved[key]}"
            )
    requested_identity = getattr(args, "_offloading_initialization_identity", {})
    saved_identity = dict(saved["offloading_initialization"])
    for key in (
        "mode",
        "training_seed",
        "task_encoder_state_sha256",
        "candidate_scorer_state_sha256",
        "checkpoint_sha256",
        "dataset_checksum",
    ):
        if key in requested_identity or key in saved_identity:
            if requested_identity.get(key) != saved_identity.get(key):
                raise ValueError(
                    f"resume checkpoint offloading initialization {key} mismatch"
                )
    return saved


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the HyperUAV clean mainline joint PPO skeleton.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps-per-episode", type=int, default=int(config.EPISODE_LENGTH))
    parser.add_argument("--rollout-horizon", type=int, default=128)
    parser.add_argument(
        "--num-envs",
        type=_positive_int,
        default=1,
        help="Number of synchronous independent environment sampler lanes.",
    )
    parser.add_argument(
        "--sampler-backend",
        choices=("synchronous", "process"),
        default="synchronous",
        help="Sampling backend. 'process' uses persistent independent worker processes.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument(
        "--normalize-value-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardize returns and value predictions together inside each rollout value loss.",
    )
    parser.add_argument(
        "--value-clip-epsilon",
        type=_nonnegative_finite_float,
        default=0.2,
        help="PPO value clipping radius in normalized value units; zero disables clipping.",
    )
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
        "--critic-task-pooling",
        choices=CLEAN_CRITIC_TASK_POOLING_CHOICES,
        default="mean",
        help="Active-task embedding aggregation used only by the centralized critic.",
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
        "--clean-counterfactual-credit",
        action="store_true",
        default=False,
        help=(
            "Phase3-B clean action-conditioned credit with fixed beta=0.25 and "
            "Q loss coefficient=0.5; mutually exclusive with legacy guidance paths."
        ),
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
        "--enable-kahypar",
        action="store_true",
        default=False,
        help="Enable KaHyPar partition hyperedges for this run; Linux formal runs only.",
    )
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
    parser.add_argument(
        "--task-encoder",
        choices=TASK_ENCODER_CHOICES,
        default="hgnn",
        help=(
            "Task encoder. 'hgnn' is the legacy alias for current_mean_hgnn; "
            "new skeleton modes are standard_weighted_hgnn and typed_gated_hgnn."
        ),
    )
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--max-updates",
        type=_nonnegative_int,
        default=None,
        help="Optional exact outer PPO update limit for short diagnostics.",
    )
    parser.add_argument(
        "--checkpoint-update-counts",
        type=str,
        default="",
        help="Comma-separated completed-update counts to checkpoint.",
    )
    parser.add_argument(
        "--eft-auxiliary-lambda-initial",
        type=_nonnegative_finite_float,
        default=0.0,
        help="Diagnostic per-decision EFT auxiliary coefficient; zero preserves baseline MAPPO.",
    )
    parser.add_argument(
        "--eft-auxiliary-regret-scale",
        type=_positive_finite_float,
        default=1.0,
        help="Frozen train-split RMS raw-EFT regret scale from the validated bandit gate.",
    )
    parser.add_argument(
        "--eft-auxiliary-sampling-seed",
        type=int,
        default=0,
        help="Independent auxiliary-action RNG seed.",
    )
    parser.add_argument(
        "--offloading-eft-advantage",
        action="store_true",
        default=False,
        help=(
            "Per-decision offloading PPO advantage gate: replace the shared "
            "slot-level GAE advantage for the offloading head with each "
            "decision's own detached decision-time EFT-regret advantage "
            "(batch-standardized). Recommended with --freeze-movement for the "
            "offloading-only diagnostic; off preserves the MAPPO baseline."
        ),
    )
    parser.add_argument(
        "--offloading-lr-scale",
        type=_positive_finite_float,
        default=1.0,
        help=(
            "Multiplier applied to the shared learning rate for the offloading "
            "scorer parameters only (other modules keep the base --lr). Values "
            "above 1.0 accelerate the offloading head when its per-decision "
            "PPO gradients are small relative to the critic."
        ),
    )
    parser.add_argument(
        "--movement-position-advantage",
        action="store_true",
        default=False,
        help=(
            "Per-UAV movement PPO advantage gate: replace the shared slot-level "
            "GAE advantage for the movement head with each UAV's own detached "
            "post-move ready-task coverage signal (batch-standardized). Use "
            "without --freeze-movement; off preserves the MAPPO baseline."
        ),
    )
    parser.add_argument(
        "--movement-lr-scale",
        type=_positive_finite_float,
        default=1.0,
        help=(
            "Multiplier applied to the shared learning rate for the movement "
            "actor parameters only (other modules keep the base --lr). Values "
            "above 1.0 accelerate the movement head when its per-UAV PPO "
            "gradients are small."
        ),
    )
    parser.add_argument(
        "--decision-critic",
        action="store_true",
        default=False,
        help=(
            "Train and use a decision-level value baseline: each offloading "
            "decision gets A = r + gamma*V(s_next) - V(s) over its slot's "
            "decision stream, and each UAV movement decision gets A = r - V(s). "
            "This makes the critic genuinely participate in learning at decision "
            "granularity. Off preserves the current per-decision advantage path."
        ),
    )
    parser.add_argument(
        "--decision-critic-coef",
        type=_nonnegative_finite_float,
        default=0.5,
        help="Weight of the decision-level critic regression losses.",
    )
    parser.add_argument(
        "--decision-critic-discount",
        type=_nonnegative_finite_float,
        default=0.99,
        help="Discount used when bootstrapping across decisions in a slot.",
    )
    parser.add_argument(
        "--offloading-init-bandit-checkpoint",
        type=Path,
        default=None,
        help="Strictly import matching-seed MLP encoder/scorer weights only.",
    )
    parser.add_argument(
        "--offloading-init-bandit-dataset-checksum",
        type=str,
        default=None,
        help="Required frozen dataset checksum for bandit initialization.",
    )
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
            **clean_counterfactual_experiment_controls(args),
            "completed_dag_weight": completed_dag_weight,
            "enable_kahypar": bool(getattr(args, "enable_kahypar", False)),
            "detach_critic_hgnn": bool(getattr(args, "detach_critic_hgnn", False)),
            "critic_task_pooling": normalize_clean_critic_task_pooling(
                getattr(args, "critic_task_pooling", "mean")
            ),
            "freeze_ue_mobility": bool(getattr(args, "freeze_ue_mobility", False)),
            "offloading_counterfactual_coef": counterfactual_coef,
            "offloading_action_value_loss_coef": action_value_loss_coef,
            "offloading_lagged_q_coef": lagged_q_coef,
            "offloading_lagged_q_loss_coef": lagged_q_loss_coef,
            "offloading_lagged_q_scale_seconds": lagged_q_scale_seconds,
            "offloading_lagged_q_censor_weight": lagged_q_censor_weight,
            "eft_auxiliary_lambda_initial": float(
                getattr(args, "eft_auxiliary_lambda_initial", 0.0)
            ),
            "eft_auxiliary_regret_scale": float(
                getattr(args, "eft_auxiliary_regret_scale", 1.0)
            ),
            "offloading_eft_advantage": bool(
                getattr(args, "offloading_eft_advantage", False)
            ),
            "offloading_lr_scale": float(
                getattr(args, "offloading_lr_scale", 1.0)
            ),
            "movement_position_advantage": bool(
                getattr(args, "movement_position_advantage", False)
            ),
            "movement_lr_scale": float(
                getattr(args, "movement_lr_scale", 1.0)
            ),
            "decision_critic_enabled": bool(
                getattr(args, "decision_critic", False)
            ),
            "decision_critic_coef": float(
                getattr(args, "decision_critic_coef", 0.5)
            ),
            "decision_critic_discount": float(
                getattr(args, "decision_critic_discount", 0.99)
            ),
            "eft_auxiliary_schedule": {
                "constant_through_update_index": 8,
                "zero_at_update_index": 20,
            },
            "eft_auxiliary_sampling_seed": int(
                getattr(args, "eft_auxiliary_sampling_seed", 0)
            ),
            "offloading_initialization": getattr(
                args, "_offloading_initialization_identity", {"mode": "random"}
            ),
            "normalize_value_targets": bool(args.normalize_value_targets),
            "value_clip_epsilon": float(args.value_clip_epsilon),
            "task_encoder": str(args.task_encoder),
            "num_envs": int(args.num_envs),
            "sampler_backend": str(args.sampler_backend),
            "environment_seeds": [
                _derive_environment_seed(int(args.seed), lane_index)
                for lane_index in range(int(args.num_envs))
            ],
            "episode_count_semantics": "total_across_all_environments",
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
            "task_encoder": str(getattr(args, "task_encoder", "hgnn")),
            "enable_kahypar": bool(getattr(args, "enable_kahypar", False)),
            "num_envs": int(getattr(args, "num_envs", 1)),
            "sampler_backend": str(getattr(args, "sampler_backend", "synchronous")),
            "multisample_label": "multisample",
            "detach_critic_hgnn": bool(getattr(args, "detach_critic_hgnn", False)),
            "critic_task_pooling": normalize_clean_critic_task_pooling(
                getattr(args, "critic_task_pooling", "mean")
            ),
            "freeze_ue_mobility": bool(getattr(args, "freeze_ue_mobility", False)),
            **clean_counterfactual_experiment_controls(args),
            "offloading_counterfactual_coef": counterfactual_coef,
            "offloading_action_value_loss_coef": action_value_loss_coef,
            "offloading_lagged_q_coef": lagged_q_coef,
            "offloading_lagged_q_loss_coef": lagged_q_loss_coef,
            "offloading_lagged_q_scale_seconds": lagged_q_scale_seconds,
            "offloading_lagged_q_censor_weight": lagged_q_censor_weight,
            "eft_auxiliary_lambda_initial": float(
                getattr(args, "eft_auxiliary_lambda_initial", 0.0)
            ),
            "eft_auxiliary_regret_scale": float(
                getattr(args, "eft_auxiliary_regret_scale", 1.0)
            ),
            "offloading_eft_advantage": bool(
                getattr(args, "offloading_eft_advantage", False)
            ),
            "offloading_lr_scale": float(
                getattr(args, "offloading_lr_scale", 1.0)
            ),
            "movement_position_advantage": bool(
                getattr(args, "movement_position_advantage", False)
            ),
            "movement_lr_scale": float(
                getattr(args, "movement_lr_scale", 1.0)
            ),
            "decision_critic_enabled": bool(
                getattr(args, "decision_critic", False)
            ),
            "decision_critic_coef": float(
                getattr(args, "decision_critic_coef", 0.5)
            ),
            "decision_critic_discount": float(
                getattr(args, "decision_critic_discount", 0.99)
            ),
            "offloading_initialization": getattr(
                args, "_offloading_initialization_identity", {"mode": "pending"}
            ),
            "lagged_q_resume_pending_policy": "discard_restarted_episode",
            "resume_semantics": "restart_from_new_episode_only",
        },
    )


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    args = apply_smoke_overrides(args)
    args.critic_task_pooling = normalize_clean_critic_task_pooling(
        getattr(args, "critic_task_pooling", "mean")
    )
    config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES = bool(args.enable_kahypar)
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
    validate_clean_counterfactual_credit_controls(args)
    if getattr(args, "max_updates", None) is not None and int(args.num_envs) != 1:
        raise ValueError("--max-updates diagnostic control currently requires --num-envs 1")
    if float(args.eft_auxiliary_lambda_initial) > 0.0 and str(args.task_encoder) != "mlp":
        raise ValueError("EFT auxiliary diagnostic requires --task-encoder mlp")
    if float(args.eft_auxiliary_lambda_initial) > 0.0 and not bool(args.freeze_movement):
        raise ValueError("EFT auxiliary diagnostic requires --freeze-movement")
    if (
        float(args.eft_auxiliary_lambda_initial) > 0.0
        and (
            float(args.offloading_counterfactual_coef) > 0.0
            or float(args.offloading_action_value_loss_coef) > 0.0
            or float(args.offloading_lagged_q_coef) > 0.0
            or float(args.offloading_lagged_q_loss_coef) > 0.0
        )
    ):
        raise ValueError(
            "EFT auxiliary diagnostic cannot combine with counterfactual or lagged-Q controls"
        )
    run_dir = create_run_directory(args)

    torch, Categorical = _require_torch()
    _set_seed(int(args.seed), torch=torch)

    from environment.dag_tasks import TASK_STATE_READY_UNSCHEDULED
    from marl_models.hgnn import build_clean_task_encoder
    from marl_models.mappo.clean_movement_actor import (
        CLEAN_MOVEMENT_UAV_FEATURE_DIM,
        CleanMovementActor,
    )
    from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
    from marl_models.mappo.clean_offloading_action_value import (
        CleanOffloadingActionValueCritic,
        build_rng_neutral_clean_counterfactual_q,
    )
    from marl_models.mappo.clean_lagged_residual_q import (
        CleanLaggedOutcomeTracker,
        build_rng_neutral_lagged_residual_q_critic,
    )
    from marl_models.mappo.clean_ppo import CleanCentralizedCritic, clean_critic_input_dim
    from marl_models.mappo.clean_ppo import CleanDecisionCritic
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
    critic_input_dim = clean_critic_input_dim(
        int(args.task_embedding_dim),
        config.NUM_UAVS,
        task_pooling=str(args.critic_task_pooling),
    )
    offloading_actor = CleanOffloadingActor(
        task_embedding_dim=int(args.task_embedding_dim),
        hidden_dim=int(args.hidden_dim),
    )
    clean_counterfactual_enabled = bool(args.clean_counterfactual_credit)
    legacy_action_value_enabled = float(args.offloading_counterfactual_coef) > 0.0
    offloading_action_value_critic = (
        build_rng_neutral_clean_counterfactual_q(
            input_dim=int(offloading_actor.candidate_feature_dim) + int(critic_input_dim),
            hidden_dim=int(args.hidden_dim),
        )
        if clean_counterfactual_enabled
        else CleanOffloadingActionValueCritic(
            input_dim=int(offloading_actor.candidate_feature_dim) + int(critic_input_dim),
            hidden_dim=int(args.hidden_dim),
        )
        if legacy_action_value_enabled
        else None
    )
    lagged_q_enabled = float(args.offloading_lagged_q_coef) > 0.0
    offloading_lagged_q_critic = (
        build_rng_neutral_lagged_residual_q_critic(
            input_dim=int(offloading_actor.candidate_feature_dim) + int(critic_input_dim),
            hidden_dim=int(args.hidden_dim),
        )
        if lagged_q_enabled
        else None
    )
    task_encoder = build_clean_task_encoder(
        encoder_type=str(args.task_encoder),
        task_feature_dim=task_feature_dim,
        hidden_dim=int(args.hidden_dim),
        output_dim=int(args.task_embedding_dim),
    )
    decision_critic_enabled = bool(args.decision_critic)
    offloading_decision_critic = (
        CleanDecisionCritic(
            input_dim=int(offloading_actor.candidate_feature_dim),
            hidden_dim=int(args.hidden_dim),
        )
        if decision_critic_enabled
        else None
    )
    movement_decision_critic = (
        CleanDecisionCritic(
            input_dim=(
                int(CLEAN_MOVEMENT_UAV_FEATURE_DIM)
                + 2
                + 2
                + 2 * int(args.task_embedding_dim)
            ),
            hidden_dim=int(args.hidden_dim),
        )
        if decision_critic_enabled
        else None
    )
    modules = CleanTrainingModules(
        hgnn=task_encoder,
        movement_actor=CleanMovementActor(task_embedding_dim=int(args.task_embedding_dim), hidden_dim=int(args.hidden_dim)),
        offloading_actor=offloading_actor,
        critic=CleanCentralizedCritic(
            input_dim=critic_input_dim,
            hidden_dim=int(args.hidden_dim),
            task_pooling=str(args.critic_task_pooling),
        ),
        offloading_action_value_critic=offloading_action_value_critic,
        offloading_lagged_q_critic=offloading_lagged_q_critic,
        offloading_decision_critic=offloading_decision_critic,
        movement_decision_critic=movement_decision_critic,
    )
    args._offloading_initialization_identity = _initialize_offloading_policy(
        args=args,
        modules=modules,
        torch=torch,
    )
    if bool(args.freeze_movement):
        for parameter in modules.movement_actor.parameters():
            parameter.requires_grad_(False)
    initialize_run_files(run_dir, args)
    device = torch.device(str(args.device))
    _move_modules_to_device(modules, device)
    optimizer = build_single_optimizer(
        modules,
        lr=float(args.lr),
        offloading_lr_scale=float(args.offloading_lr_scale),
        movement_lr_scale=float(args.movement_lr_scale),
    )
    updater = CleanPPOUpdater(
        modules=modules,
        optimizer=optimizer,
        config=CleanPPOUpdateConfig(
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            clip_epsilon=float(args.clip_ratio),
            ppo_epochs=int(args.ppo_epochs),
            value_coef=float(args.value_coef),
            normalize_value_targets=bool(args.normalize_value_targets),
            value_clip_epsilon=float(args.value_clip_epsilon),
            movement_entropy_coef=float(args.entropy_coef),
            offloading_entropy_coef=float(args.entropy_coef),
            max_grad_norm=float(args.max_grad_norm),
            detach_critic_hgnn=bool(args.detach_critic_hgnn),
            clean_counterfactual_credit=clean_counterfactual_enabled,
            offloading_eft_advantage=bool(args.offloading_eft_advantage),
            movement_position_advantage=bool(args.movement_position_advantage),
            decision_critic_enabled=bool(args.decision_critic),
            decision_critic_coef=float(args.decision_critic_coef),
            decision_critic_discount=float(args.decision_critic_discount),
            offloading_counterfactual_coef=float(args.offloading_counterfactual_coef),
            offloading_action_value_loss_coef=float(args.offloading_action_value_loss_coef),
            offloading_lagged_q_coef=float(args.offloading_lagged_q_coef),
            offloading_lagged_q_loss_coef=float(args.offloading_lagged_q_loss_coef),
            eft_auxiliary_lambda_initial=float(args.eft_auxiliary_lambda_initial),
            eft_auxiliary_regret_scale=float(args.eft_auxiliary_regret_scale),
            eft_auxiliary_constant_through_update=8,
            eft_auxiliary_zero_at_update=20,
            eft_auxiliary_sampling_seed=int(args.eft_auxiliary_sampling_seed),
        ),
        device=device,
    )
    checkpoint_manager = CleanCheckpointManager(run_dir / "checkpoints")
    logger = CleanJSONLLogger(run_dir)
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

    checkpoint_update_counts = _parse_update_counts(
        str(args.checkpoint_update_counts)
    )
    if int(updater.update_step) == 0:
        checkpoint_manager.save(
            modules=modules,
            optimizer=optimizer,
            episode=-1,
            global_slot=global_slot,
            update_step=0,
            config_snapshot=build_config_snapshot(args),
            safe_boundary=True,
            filename="checkpoint_update_0000.pt",
        )
    if args.max_updates is not None and int(args.max_updates) == 0:
        _write_json(
            run_dir / "run_summary.json",
            {
                "status": "completed",
                "completion_reason": "max_updates",
                "global_slot": int(global_slot),
                "completed_update_count": int(updater.update_step),
                "offloading_initialization": args._offloading_initialization_identity,
                **clean_counterfactual_experiment_controls(args),
            },
        )
        graph_builder.close()
        return {
            "run_dir": str(run_dir),
            "global_slot": int(global_slot),
            "latest_update": None,
            "completed_update_count": int(updater.update_step),
            "offloading_initialization": args._offloading_initialization_identity,
        }

    if str(args.sampler_backend) == "process":
        if clean_counterfactual_enabled or float(args.offloading_counterfactual_coef) > 0.0 or float(
            args.offloading_lagged_q_coef
        ) > 0.0:
            graph_builder.close()
            raise ValueError(
                "process sampler backend currently supports only the formal baseline "
                "with counterfactual and lagged residual-Q disabled"
            )
        graph_builder.close()
        return _run_process_sampler_training_loop(
            args=args,
            task_feature_dim=task_feature_dim,
            modules=modules,
            updater=updater,
            checkpoint_manager=checkpoint_manager,
            logger=logger,
            task_state_ready=TASK_STATE_READY_UNSCHEDULED,
            start_episode=start_episode,
            global_slot=global_slot,
        )

    if int(args.num_envs) > 1:
        return _run_multisample_training_loop(
            args=args,
            initial_env=env,
            initial_graph_builder=graph_builder,
            modules=modules,
            updater=updater,
            checkpoint_manager=checkpoint_manager,
            logger=logger,
            categorical_cls=Categorical,
            device=device,
            task_state_ready=TASK_STATE_READY_UNSCHEDULED,
            lagged_tracker_cls=CleanLaggedOutcomeTracker,
            lagged_q_enabled=lagged_q_enabled,
            start_episode=start_episode,
            global_slot=global_slot,
        )

    lagged_q_tracker = (
        CleanLaggedOutcomeTracker(
            scale_seconds=float(args.offloading_lagged_q_scale_seconds),
            censor_weight=float(args.offloading_lagged_q_censor_weight),
        )
        if lagged_q_enabled
        else None
    )
    progress = _make_progress_bar(total=max(int(args.episodes) - start_episode, 0) * int(args.max_steps_per_episode))
    max_updates_reached = False
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
                    if int(updater.update_step) in checkpoint_update_counts:
                        checkpoint_manager.save(
                            modules=modules,
                            optimizer=optimizer,
                            episode=episode,
                            global_slot=global_slot,
                            update_step=updater.update_step,
                            config_snapshot=build_config_snapshot(args),
                            safe_boundary=buffer.checkpoint_safe,
                            filename=f"checkpoint_update_{int(updater.update_step):04d}.pt",
                        )
                    max_updates_reached = (
                        args.max_updates is not None
                        and int(updater.update_step) >= int(args.max_updates)
                    )
                    if max_updates_reached:
                        last_info = info
                        break
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
                    "status": (
                        "completed"
                        if max_updates_reached or episode + 1 >= int(args.episodes)
                        else "running"
                    ),
                    "completion_reason": (
                        "max_updates" if max_updates_reached else "episodes"
                    ),
                    "completed_update_count": int(updater.update_step),
                    "episode": episode,
                    "global_slot": global_slot,
                    "episode_reward": episode_reward,
                    "latest_info": _jsonable(last_info),
                    "latest_update": None if latest_update_stats is None else asdict(latest_update_stats),
                    "completed_dag_weight": float(args.completed_dag_weight),
                    "task_encoder": str(args.task_encoder),
                    "detach_critic_hgnn": bool(args.detach_critic_hgnn),
                    "critic_task_pooling": str(args.critic_task_pooling),
                    **clean_counterfactual_experiment_controls(args),
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
            if max_updates_reached:
                break
    finally:
        progress.close()
        graph_builder.close()

    return {
        "run_dir": str(run_dir),
        "global_slot": global_slot,
        "completed_update_count": int(updater.update_step),
        "latest_update": None if latest_update_stats is None else asdict(latest_update_stats),
        "completed_dag_weight": float(args.completed_dag_weight),
        "detach_critic_hgnn": bool(args.detach_critic_hgnn),
        "critic_task_pooling": str(args.critic_task_pooling),
        **clean_counterfactual_experiment_controls(args),
        "freeze_ue_mobility": bool(args.freeze_ue_mobility),
        "offloading_counterfactual_coef": float(args.offloading_counterfactual_coef),
        "offloading_action_value_loss_coef": float(args.offloading_action_value_loss_coef),
        "offloading_lagged_q_coef": float(args.offloading_lagged_q_coef),
        "offloading_lagged_q_loss_coef": float(args.offloading_lagged_q_loss_coef),
        "offloading_lagged_q_scale_seconds": float(args.offloading_lagged_q_scale_seconds),
        "offloading_lagged_q_censor_weight": float(args.offloading_lagged_q_censor_weight),
        "eft_auxiliary_lambda_initial": float(args.eft_auxiliary_lambda_initial),
        "eft_auxiliary_regret_scale": float(args.eft_auxiliary_regret_scale),
        "offloading_initialization": args._offloading_initialization_identity,
        "kahypar_circuit_open": bool(graph_builder.kahypar_circuit_open),
        "kahypar_last_failure_reason": graph_builder.kahypar_last_failure_reason,
        "kahypar_cleanup_failed": bool(graph_builder.kahypar_cleanup_failed),
        "kahypar_worker_alive_after_close": bool(graph_builder.kahypar_worker_alive),
    }


@dataclass
class _EnvironmentRNGState:
    python_state: object
    numpy_state: tuple[Any, ...]


@dataclass
class _SamplerLane:
    lane_index: int
    environment_seed: int
    env: Env
    graph_builder: CleanGraphBuilder
    rng_state: _EnvironmentRNGState
    lagged_q_tracker: Any | None = None
    episode: int = -1
    episode_step: int = 0
    buffer: CleanSlotRolloutBuffer = field(default_factory=CleanSlotRolloutBuffer)
    current_prepared: Any | None = None
    current_encoded: Any | None = None
    next_prepared: Any | None = None
    next_encoded_old: Any | None = None
    episode_reward: float = 0.0
    episode_component_totals: dict[str, float] = field(default_factory=dict)
    last_info: dict[str, Any] = field(default_factory=dict)
    done: bool = False
    truncated: bool = False

    @property
    def terminal(self) -> bool:
        return bool(self.done or self.truncated)


def _derive_environment_seed(training_seed: int, lane_index: int) -> int:
    modulus = (2**32) - 1
    return int((int(training_seed) + 1_000_003 * int(lane_index)) % modulus)


def _capture_environment_rng_state() -> _EnvironmentRNGState:
    return _EnvironmentRNGState(
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
    )


@contextmanager
def _activate_lane_rng(lane: _SamplerLane):
    outer_state = _capture_environment_rng_state()
    random.setstate(lane.rng_state.python_state)
    np.random.set_state(lane.rng_state.numpy_state)
    try:
        yield
    finally:
        lane.rng_state = _capture_environment_rng_state()
        random.setstate(outer_state.python_state)
        np.random.set_state(outer_state.numpy_state)


def _new_episode_component_totals() -> dict[str, float]:
    return {
        "reward": 0.0,
        "time_penalty": 0.0,
        "dag_bonus": 0.0,
        "task_energy_penalty": 0.0,
        "movement_energy_penalty": 0.0,
    }


def _make_sampler_lanes(
    *,
    args: argparse.Namespace,
    initial_env: Env,
    initial_graph_builder: CleanGraphBuilder,
    lagged_tracker_cls: Any,
    lagged_q_enabled: bool,
) -> list[_SamplerLane]:
    lane_zero_state = _capture_environment_rng_state()
    lanes = [
        _SamplerLane(
            lane_index=0,
            environment_seed=_derive_environment_seed(int(args.seed), 0),
            env=initial_env,
            graph_builder=initial_graph_builder,
            rng_state=lane_zero_state,
        )
    ]
    for lane_index in range(1, int(args.num_envs)):
        random.seed(_derive_environment_seed(int(args.seed), lane_index))
        np.random.seed(_derive_environment_seed(int(args.seed), lane_index))
        env = Env(
            completed_dag_weight=float(args.completed_dag_weight),
            freeze_ue_mobility=bool(args.freeze_ue_mobility),
        )
        graph_builder = CleanGraphBuilder()
        env.reset()
        graph_builder.reset()
        prepare_slot_state(env=env, graph_builder=graph_builder)
        lanes.append(
            _SamplerLane(
                lane_index=lane_index,
                environment_seed=_derive_environment_seed(int(args.seed), lane_index),
                env=env,
                graph_builder=graph_builder,
                rng_state=_capture_environment_rng_state(),
            )
        )
    random.setstate(lane_zero_state.python_state)
    np.random.set_state(lane_zero_state.numpy_state)
    if lagged_q_enabled:
        for lane in lanes:
            lane.lagged_q_tracker = lagged_tracker_cls(
                scale_seconds=float(args.offloading_lagged_q_scale_seconds),
                censor_weight=float(args.offloading_lagged_q_censor_weight),
            )
    return lanes


def _start_sampler_lane_episode(
    *,
    lane: _SamplerLane,
    episode: int,
    args: argparse.Namespace,
    modules: CleanTrainingModules,
    device: Any,
) -> None:
    lane.episode = int(episode)
    lane.episode_step = 0
    lane.buffer = CleanSlotRolloutBuffer()
    lane.episode_reward = 0.0
    lane.episode_component_totals = _new_episode_component_totals()
    lane.last_info = {}
    lane.done = False
    lane.truncated = False
    lane.next_prepared = None
    lane.next_encoded_old = None
    with _activate_lane_rng(lane):
        lane.env.reset()
        lane.graph_builder.reset()
        if lane.lagged_q_tracker is not None:
            lane.lagged_q_tracker.start_episode(int(episode))
        lane.current_prepared = prepare_slot_state(
            env=lane.env,
            graph_builder=lane.graph_builder,
        )
    lane.current_encoded = encode_prepared_slot(
        prepared_state=lane.current_prepared,
        env=lane.env,
        hgnn=modules.hgnn,
        critic=modules.critic,
        movement_actor=modules.movement_actor,
        device=device,
        detach_critic_hgnn=bool(args.detach_critic_hgnn),
    )


def _collect_sampler_lane_step(
    *,
    lane: _SamplerLane,
    args: argparse.Namespace,
    modules: CleanTrainingModules,
    categorical_cls: Any,
    device: Any,
    task_state_ready: str,
    lagged_q_enabled: bool,
) -> None:
    with _activate_lane_rng(lane):
        slot_record, done, info = _collect_clean_slot(
            env=lane.env,
            modules=modules,
            encoded_state=lane.current_encoded,
            categorical_cls=categorical_cls,
            device=device,
            task_state_ready=task_state_ready,
            freeze_movement=bool(args.freeze_movement),
            lagged_q_enabled=bool(lagged_q_enabled),
        )
        if lane.lagged_q_tracker is not None:
            lane.lagged_q_tracker.register_rollout_actions(
                slot_record=slot_record,
                env=lane.env,
            )
            lane.lagged_q_tracker.resolve_completed(env=lane.env)
        next_prepared = None
        if not done:
            next_prepared = prepare_slot_state(
                env=lane.env,
                graph_builder=lane.graph_builder,
            )
    next_encoded_old = None
    if next_prepared is not None:
        next_encoded_old = encode_prepared_slot(
            prepared_state=next_prepared,
            env=lane.env,
            hgnn=modules.hgnn,
            critic=modules.critic,
            movement_actor=modules.movement_actor,
            device=device,
            detach_critic_hgnn=bool(args.detach_critic_hgnn),
        )

    lane.episode_step += 1
    lane.episode_reward += float(slot_record.reward)
    lane.episode_component_totals["reward"] += float(info.get("step_reward", 0.0))
    lane.episode_component_totals["time_penalty"] += float(
        info.get("step_time_penalty", 0.0)
    )
    lane.episode_component_totals["dag_bonus"] += float(
        info.get("step_completed_dag_bonus", 0.0)
    )
    lane.episode_component_totals["task_energy_penalty"] += float(
        info.get("step_task_energy_penalty", 0.0)
    )
    lane.episode_component_totals["movement_energy_penalty"] += float(
        info.get("step_movement_energy_penalty", 0.0)
    )
    truncated = bool(
        lane.episode_step >= int(args.max_steps_per_episode) and not bool(done)
    )
    slot_record.terminated = bool(done)
    slot_record.truncated = truncated
    info["terminated"] = bool(done)
    info["truncated"] = truncated
    lane.buffer.append(slot_record)
    lane.done = bool(done)
    lane.truncated = truncated
    lane.last_info = info
    lane.next_prepared = next_prepared
    lane.next_encoded_old = next_encoded_old
    # Refresh the lane's current encoding for the next slot. Without this the
    # synchronous multisample path reuses the episode-start encoding for every
    # slot in the rollout, so the offloading head never sees newly arrived
    # tasks and skips all assignments (DAG completion stays 0).
    if next_encoded_old is not None:
        lane.current_encoded = next_encoded_old
        lane.current_prepared = next_prepared


def _close_sampler_lane_rollout(lane: _SamplerLane) -> tuple[list[Any], dict[str, int] | None]:
    lagged_summary: dict[str, int] | None = None
    if lane.lagged_q_tracker is not None and lane.terminal:
        with _activate_lane_rng(lane):
            lane.lagged_q_tracker.resolve_completed(env=lane.env)
            lane.lagged_q_tracker.finalize_censored(
                episode_end_time=float(lane.env.current_time_seconds)
            )
    lagged_samples = (
        lane.lagged_q_tracker.pop_finalized()
        if lane.lagged_q_tracker is not None
        else []
    )
    close_rollout_with_bootstrap(
        buffer=lane.buffer,
        next_encoded_state=lane.next_encoded_old,
        terminated=bool(lane.done),
    )
    if lane.lagged_q_tracker is not None and lane.terminal:
        lagged_summary = lane.lagged_q_tracker.finish_episode()
    return lagged_samples, lagged_summary


def _run_multisample_training_loop(
    *,
    args: argparse.Namespace,
    initial_env: Env,
    initial_graph_builder: CleanGraphBuilder,
    modules: CleanTrainingModules,
    updater: Any,
    checkpoint_manager: CleanCheckpointManager,
    logger: CleanJSONLLogger,
    categorical_cls: Any,
    device: Any,
    task_state_ready: str,
    lagged_tracker_cls: Any,
    lagged_q_enabled: bool,
    start_episode: int,
    global_slot: int,
) -> dict[str, Any]:
    run_dir = logger.run_dir
    lanes = _make_sampler_lanes(
        args=args,
        initial_env=initial_env,
        initial_graph_builder=initial_graph_builder,
        lagged_tracker_cls=lagged_tracker_cls,
        lagged_q_enabled=lagged_q_enabled,
    )
    latest_update_stats: CleanPPOUpdateStats | None = None
    progress = _make_progress_bar(
        total=max(int(args.episodes) - int(start_episode), 0)
        * int(args.max_steps_per_episode)
    )
    next_episode = int(start_episode)
    initial_global_slot = int(global_slot)
    started_at = time.perf_counter()
    graph_health: list[dict[str, Any]] = []
    try:
        while next_episode < int(args.episodes):
            active_lanes = lanes[
                : min(int(args.num_envs), int(args.episodes) - next_episode)
            ]
            for lane in active_lanes:
                _start_sampler_lane_episode(
                    lane=lane,
                    episode=next_episode,
                    args=args,
                    modules=modules,
                    device=device,
                )
                next_episode += 1

            while any(not lane.terminal for lane in active_lanes):
                for lane in active_lanes:
                    if lane.terminal or len(lane.buffer) >= int(args.rollout_horizon):
                        continue
                    _collect_sampler_lane_step(
                        lane=lane,
                        args=args,
                        modules=modules,
                        categorical_cls=categorical_cls,
                        device=device,
                        task_state_ready=task_state_ready,
                        lagged_q_enabled=lagged_q_enabled,
                    )
                    global_slot += 1
                    _update_progress(
                        progress,
                        lane.episode,
                        global_slot,
                        lane.episode_reward,
                        lane.last_info,
                        latest_update_stats,
                    )
                    progress.update(1)

                update_lanes = [
                    lane
                    for lane in active_lanes
                    if len(lane.buffer) > 0
                    and (
                        len(lane.buffer) >= int(args.rollout_horizon)
                        or lane.terminal
                    )
                ]
                waiting_lanes = [
                    lane
                    for lane in active_lanes
                    if not lane.terminal
                    and 0 < len(lane.buffer) < int(args.rollout_horizon)
                ]
                if waiting_lanes:
                    continue
                if not update_lanes:
                    continue

                lagged_samples: list[Any] = []
                lagged_summaries: dict[int, dict[str, int] | None] = {}
                for lane in update_lanes:
                    lane_samples, lane_summary = _close_sampler_lane_rollout(lane)
                    lagged_samples.extend(lane_samples)
                    lagged_summaries[lane.lane_index] = lane_summary
                latest_update_stats = updater.update_many(
                    [lane.buffer for lane in update_lanes],
                    lagged_q_samples=lagged_samples,
                    lagged_q_pending_count=sum(
                        lane.lagged_q_tracker.pending_count
                        if lane.lagged_q_tracker is not None
                        else 0
                        for lane in active_lanes
                    ),
                )
                elapsed = max(time.perf_counter() - started_at, 1e-9)
                for log_index, lane in enumerate(update_lanes):
                    extra = _episode_diagnostics_payload(
                        episode_component_totals=lane.episode_component_totals,
                        terminal=lane.terminal,
                        env=lane.env,
                        completed_dag_weight=float(args.completed_dag_weight),
                        detach_critic_hgnn=bool(args.detach_critic_hgnn),
                        freeze_ue_mobility=bool(args.freeze_ue_mobility),
                        offloading_counterfactual_coef=float(
                            args.offloading_counterfactual_coef
                        ),
                        offloading_action_value_loss_coef=float(
                            args.offloading_action_value_loss_coef
                        ),
                        offloading_lagged_q_coef=float(args.offloading_lagged_q_coef),
                        offloading_lagged_q_loss_coef=float(
                            args.offloading_lagged_q_loss_coef
                        ),
                        lagged_tracker_summary=lagged_summaries[lane.lane_index],
                    )
                    extra.update(
                        {
                            "multisample_label": "multisample",
                            "num_envs": int(args.num_envs),
                            "active_env_count": int(len(active_lanes)),
                            "environment_index": int(lane.lane_index),
                            "environment_seed": int(lane.environment_seed),
                            "environment_slots_this_run": int(
                                global_slot - initial_global_slot
                            ),
                            "environment_slots_per_second": float(
                                (global_slot - initial_global_slot) / elapsed
                            ),
                        }
                    )
                    write_clean_training_log(
                        logger,
                        episode=lane.episode,
                        global_slot=global_slot,
                        info=lane.last_info,
                        update_stats=latest_update_stats if log_index == 0 else None,
                        extra=extra,
                    )

                checkpoint_manager.save(
                    modules=modules,
                    optimizer=updater.optimizer,
                    episode=max(lane.episode for lane in active_lanes),
                    global_slot=global_slot,
                    update_step=updater.update_step,
                    config_snapshot=build_config_snapshot(args),
                    safe_boundary=all(lane.buffer.checkpoint_safe for lane in update_lanes),
                    filename="latest.pt",
                )
                for lane in update_lanes:
                    if lane.terminal:
                        lane.buffer = CleanSlotRolloutBuffer()
                        continue
                    lane.current_prepared = lane.next_prepared
                    lane.current_encoded = reencode_prepared_after_update(
                        prepared_state=lane.current_prepared,
                        env=lane.env,
                        modules=modules,
                        device=device,
                        detach_critic_hgnn=bool(args.detach_critic_hgnn),
                    )
                    lane.buffer = CleanSlotRolloutBuffer()

            finished_episode = max(lane.episode for lane in active_lanes)
            for lane in active_lanes:
                if (
                    int(args.checkpoint_interval) > 0
                    and (lane.episode + 1) % int(args.checkpoint_interval) == 0
                ):
                    checkpoint_manager.save(
                        modules=modules,
                        optimizer=updater.optimizer,
                        episode=lane.episode,
                        global_slot=global_slot,
                        update_step=updater.update_step,
                        config_snapshot=build_config_snapshot(args),
                        safe_boundary=True,
                        filename=f"checkpoint_ep_{lane.episode + 1:04d}.pt",
                    )
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            _write_json(
                run_dir / "run_summary.json",
                {
                    "status": (
                        "running"
                        if finished_episode + 1 < int(args.episodes)
                        else "completed"
                    ),
                    "episode": int(finished_episode),
                    "global_slot": int(global_slot),
                    "latest_update": (
                        None
                        if latest_update_stats is None
                        else asdict(latest_update_stats)
                    ),
                    "num_envs": int(args.num_envs),
                    "multisample_label": "multisample",
                    "environment_seeds": [
                        int(lane.environment_seed) for lane in lanes
                    ],
                    "environment_slots_this_run": int(
                        global_slot - initial_global_slot
                    ),
                    "environment_slots_per_second": float(
                        (global_slot - initial_global_slot) / elapsed
                    ),
                    "elapsed_seconds": float(elapsed),
                    "completed_dag_weight": float(args.completed_dag_weight),
                    "task_encoder": str(args.task_encoder),
                    "detach_critic_hgnn": bool(args.detach_critic_hgnn),
                    "critic_task_pooling": str(args.critic_task_pooling),
                    **clean_counterfactual_experiment_controls(args),
                    "freeze_ue_mobility": bool(args.freeze_ue_mobility),
                    "offloading_counterfactual_coef": float(
                        args.offloading_counterfactual_coef
                    ),
                    "offloading_action_value_loss_coef": float(
                        args.offloading_action_value_loss_coef
                    ),
                    "offloading_lagged_q_coef": float(
                        args.offloading_lagged_q_coef
                    ),
                    "offloading_lagged_q_loss_coef": float(
                        args.offloading_lagged_q_loss_coef
                    ),
                    "resume_semantics": "restart_from_new_episode_only",
                },
            )
    finally:
        progress.close()
        for lane in lanes:
            lane.graph_builder.close()
            graph_health.append(
                {
                    "lane_index": int(lane.lane_index),
                    "kahypar_circuit_open": bool(
                        lane.graph_builder.kahypar_circuit_open
                    ),
                    "kahypar_last_failure_reason": (
                        lane.graph_builder.kahypar_last_failure_reason
                    ),
                    "kahypar_cleanup_failed": bool(
                        lane.graph_builder.kahypar_cleanup_failed
                    ),
                    "kahypar_worker_alive_after_close": bool(
                        lane.graph_builder.kahypar_worker_alive
                    ),
                }
            )

    elapsed = max(time.perf_counter() - started_at, 1e-9)
    return {
        "run_dir": str(run_dir),
        "global_slot": int(global_slot),
        "latest_update": (
            None if latest_update_stats is None else asdict(latest_update_stats)
        ),
        "num_envs": int(args.num_envs),
        "multisample_label": "multisample",
        "environment_seeds": [int(lane.environment_seed) for lane in lanes],
        "environment_slots_this_run": int(global_slot - initial_global_slot),
        "environment_slots_per_second": float(
            (global_slot - initial_global_slot) / elapsed
        ),
        "elapsed_seconds": float(elapsed),
        "completed_dag_weight": float(args.completed_dag_weight),
        "detach_critic_hgnn": bool(args.detach_critic_hgnn),
        "critic_task_pooling": str(args.critic_task_pooling),
        **clean_counterfactual_experiment_controls(args),
        "freeze_ue_mobility": bool(args.freeze_ue_mobility),
        "offloading_counterfactual_coef": float(args.offloading_counterfactual_coef),
        "offloading_action_value_loss_coef": float(
            args.offloading_action_value_loss_coef
        ),
        "offloading_lagged_q_coef": float(args.offloading_lagged_q_coef),
        "offloading_lagged_q_loss_coef": float(
            args.offloading_lagged_q_loss_coef
        ),
        "kahypar_circuit_open": any(
            row["kahypar_circuit_open"] for row in graph_health
        ),
        "kahypar_last_failure_reason": next(
            (
                row["kahypar_last_failure_reason"]
                for row in graph_health
                if row["kahypar_last_failure_reason"] is not None
            ),
            None,
        ),
        "kahypar_cleanup_failed": any(
            row["kahypar_cleanup_failed"] for row in graph_health
        ),
        "kahypar_worker_alive_after_close": any(
            row["kahypar_worker_alive_after_close"] for row in graph_health
        ),
        "environment_health": graph_health,
    }


def _process_sampler_worker(
    connection: Any,
    *,
    lane_index: int,
    environment_seed: int,
    task_feature_dim: int,
    worker_config: dict[str, Any],
) -> None:
    graph_builder: CleanGraphBuilder | None = None
    try:
        torch, categorical_cls = _require_torch()
        torch.set_num_threads(1)
        _set_seed(int(environment_seed), torch=torch)
        env = Env(
            completed_dag_weight=float(worker_config["completed_dag_weight"]),
            freeze_ue_mobility=bool(worker_config["freeze_ue_mobility"]),
        )
        graph_builder = CleanGraphBuilder()
        env.reset()
        graph_builder.reset()
        probe = prepare_slot_state(env=env, graph_builder=graph_builder)
        observed_feature_dim = int(probe.graph_snapshot.task_features.shape[1])
        if observed_feature_dim != int(task_feature_dim):
            raise RuntimeError(
                "worker task feature dimension mismatch: "
                f"{observed_feature_dim} != {task_feature_dim}"
            )
        modules = _build_process_worker_modules(
            task_feature_dim=int(task_feature_dim),
            task_embedding_dim=int(worker_config["task_embedding_dim"]),
            hidden_dim=int(worker_config["hidden_dim"]),
            task_encoder=str(worker_config["task_encoder"]),
            critic_task_pooling=str(worker_config["critic_task_pooling"]),
            device=torch.device("cpu"),
        )
        connection.send(
            {
                "type": "ready",
                "lane_index": int(lane_index),
                "pid": int(multiprocessing.current_process().pid or -1),
            }
        )
        active_episode: int | None = None
        episode_step = 0
        episode_reward = 0.0
        component_totals = _new_episode_component_totals()
        current_prepared: Any | None = None
        current_encoded: Any | None = None

        while True:
            command = connection.recv()
            command_type = str(command.get("type", ""))
            if command_type == "shutdown":
                break
            if command_type != "collect":
                raise ValueError(f"unsupported process sampler command: {command_type}")
            _load_module_state_payload(modules, command["module_state"])
            requested_episode = command.get("episode")
            if requested_episode is not None:
                if active_episode is not None:
                    raise RuntimeError("worker received a new episode before finishing the active one")
                active_episode = int(requested_episode)
                episode_step = 0
                episode_reward = 0.0
                component_totals = _new_episode_component_totals()
                env.reset()
                graph_builder.reset()
                current_prepared = prepare_slot_state(
                    env=env,
                    graph_builder=graph_builder,
                )
            elif active_episode is None or current_prepared is None:
                raise RuntimeError("worker collect continuation has no active episode")

            current_encoded = encode_prepared_slot(
                prepared_state=current_prepared,
                env=env,
                hgnn=modules.hgnn,
                critic=modules.critic,
                movement_actor=modules.movement_actor,
                device=torch.device("cpu"),
                detach_critic_hgnn=bool(worker_config["detach_critic_hgnn"]),
            )
            buffer = CleanSlotRolloutBuffer()
            last_info: dict[str, Any] = {}
            done = False
            truncated = False
            collect_started = time.perf_counter()
            for _ in range(int(command["rollout_horizon"])):
                slot_record, done, info = _collect_clean_slot(
                    env=env,
                    modules=modules,
                    encoded_state=current_encoded,
                    categorical_cls=categorical_cls,
                    device=torch.device("cpu"),
                    task_state_ready=str(worker_config["task_state_ready"]),
                    freeze_movement=bool(worker_config["freeze_movement"]),
                    lagged_q_enabled=False,
                )
                episode_step += 1
                episode_reward += float(slot_record.reward)
                component_totals["reward"] += float(info.get("step_reward", 0.0))
                component_totals["time_penalty"] += float(
                    info.get("step_time_penalty", 0.0)
                )
                component_totals["dag_bonus"] += float(
                    info.get("step_completed_dag_bonus", 0.0)
                )
                component_totals["task_energy_penalty"] += float(
                    info.get("step_task_energy_penalty", 0.0)
                )
                component_totals["movement_energy_penalty"] += float(
                    info.get("step_movement_energy_penalty", 0.0)
                )
                truncated = bool(
                    episode_step >= int(worker_config["max_steps_per_episode"])
                    and not bool(done)
                )
                slot_record.terminated = bool(done)
                slot_record.truncated = bool(truncated)
                info["terminated"] = bool(done)
                info["truncated"] = bool(truncated)
                buffer.append(slot_record)
                last_info = info

                next_prepared = None
                next_encoded_old = None
                if not done:
                    next_prepared = prepare_slot_state(
                        env=env,
                        graph_builder=graph_builder,
                    )
                    next_encoded_old = encode_prepared_slot(
                        prepared_state=next_prepared,
                        env=env,
                        hgnn=modules.hgnn,
                        critic=modules.critic,
                        movement_actor=modules.movement_actor,
                        device=torch.device("cpu"),
                        detach_critic_hgnn=bool(
                            worker_config["detach_critic_hgnn"]
                        ),
                    )
                current_prepared = next_prepared
                current_encoded = next_encoded_old
                if bool(done or truncated):
                    break

            close_rollout_with_bootstrap(
                buffer=buffer,
                next_encoded_state=current_encoded,
                terminated=bool(done),
            )
            terminal = bool(done or truncated)
            episode_diagnostics = _episode_diagnostics_payload(
                episode_component_totals=component_totals,
                terminal=terminal,
                env=env,
                completed_dag_weight=float(worker_config["completed_dag_weight"]),
                detach_critic_hgnn=bool(worker_config["detach_critic_hgnn"]),
                freeze_ue_mobility=bool(worker_config["freeze_ue_mobility"]),
                offloading_counterfactual_coef=0.0,
                offloading_action_value_loss_coef=0.0,
                offloading_lagged_q_coef=0.0,
                offloading_lagged_q_loss_coef=0.0,
                lagged_tracker_summary=None,
            )
            response = {
                "type": "rollout",
                "lane_index": int(lane_index),
                "environment_seed": int(environment_seed),
                "episode": int(active_episode),
                "episode_step": int(episode_step),
                "episode_reward": float(episode_reward),
                "episode_component_totals": dict(component_totals),
                "episode_diagnostics": episode_diagnostics,
                "last_info": last_info,
                "terminal": terminal,
                "done": bool(done),
                "truncated": bool(truncated),
                "step_count": int(len(buffer)),
                "worker_collect_seconds": float(
                    time.perf_counter() - collect_started
                ),
                "initial_hotspot_ue_count": int(env.initial_hotspot_ue_count),
                "buffer": buffer,
            }
            connection.send(response)
            if terminal:
                active_episode = None
                current_prepared = None
                current_encoded = None
    except BaseException as exc:
        try:
            connection.send(
                {
                    "type": "error",
                    "lane_index": int(lane_index),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
        except BaseException:
            pass
    finally:
        health = {
            "lane_index": int(lane_index),
            "kahypar_circuit_open": False,
            "kahypar_last_failure_reason": None,
            "kahypar_cleanup_failed": False,
            "kahypar_worker_alive_after_close": False,
        }
        if graph_builder is not None:
            graph_builder.close()
            health.update(
                {
                    "kahypar_circuit_open": bool(
                        graph_builder.kahypar_circuit_open
                    ),
                    "kahypar_last_failure_reason": (
                        graph_builder.kahypar_last_failure_reason
                    ),
                    "kahypar_cleanup_failed": bool(
                        graph_builder.kahypar_cleanup_failed
                    ),
                    "kahypar_worker_alive_after_close": bool(
                        graph_builder.kahypar_worker_alive
                    ),
                }
            )
        try:
            connection.send({"type": "shutdown", "health": health})
        except BaseException:
            pass
        connection.close()


def _run_process_sampler_training_loop(
    *,
    args: argparse.Namespace,
    task_feature_dim: int,
    modules: CleanTrainingModules,
    updater: Any,
    checkpoint_manager: CleanCheckpointManager,
    logger: CleanJSONLLogger,
    task_state_ready: str,
    start_episode: int,
    global_slot: int,
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    workers: list[dict[str, Any]] = []
    worker_config = {
        "completed_dag_weight": float(args.completed_dag_weight),
        "freeze_ue_mobility": bool(args.freeze_ue_mobility),
        "freeze_movement": bool(args.freeze_movement),
        "detach_critic_hgnn": bool(args.detach_critic_hgnn),
        "task_embedding_dim": int(args.task_embedding_dim),
        "hidden_dim": int(args.hidden_dim),
        "task_encoder": str(args.task_encoder),
        "critic_task_pooling": str(args.critic_task_pooling),
        "max_steps_per_episode": int(args.max_steps_per_episode),
        "task_state_ready": str(task_state_ready),
    }
    for lane_index in range(int(args.num_envs)):
        parent_connection, child_connection = context.Pipe(duplex=True)
        process = context.Process(
            target=_process_sampler_worker,
            kwargs={
                "connection": child_connection,
                "lane_index": int(lane_index),
                "environment_seed": _derive_environment_seed(
                    int(args.seed), lane_index
                ),
                "task_feature_dim": int(task_feature_dim),
                "worker_config": worker_config,
            },
            name=f"multisample-env-{lane_index}",
        )
        process.start()
        child_connection.close()
        workers.append(
            {
                "lane_index": int(lane_index),
                "environment_seed": _derive_environment_seed(
                    int(args.seed), lane_index
                ),
                "connection": parent_connection,
                "process": process,
                "episode": None,
            }
        )

    progress = _make_progress_bar(
        total=max(int(args.episodes) - int(start_episode), 0)
        * int(args.max_steps_per_episode)
    )
    next_episode = int(start_episode)
    initial_global_slot = int(global_slot)
    latest_update_stats: CleanPPOUpdateStats | None = None
    started_at = time.perf_counter()
    worker_health: list[dict[str, Any]] = []
    completed_episodes = int(start_episode)
    failure: BaseException | None = None
    try:
        for worker in workers:
            connection = worker["connection"]
            if not connection.poll(120.0):
                raise TimeoutError(
                    f"process sampler worker {worker['lane_index']} startup timed out"
                )
            ready = connection.recv()
            if ready.get("type") != "ready":
                raise RuntimeError(f"process sampler worker startup failed: {ready}")

        while completed_episodes < int(args.episodes):
            round_workers: list[dict[str, Any]] = []
            module_state = _module_state_payload(modules)
            for worker in workers:
                if worker["episode"] is None:
                    if next_episode >= int(args.episodes):
                        continue
                    worker["episode"] = int(next_episode)
                    next_episode += 1
                    requested_episode: int | None = int(worker["episode"])
                else:
                    requested_episode = None
                worker["connection"].send(
                    {
                        "type": "collect",
                        "episode": requested_episode,
                        "rollout_horizon": int(args.rollout_horizon),
                        "module_state": module_state,
                    }
                )
                round_workers.append(worker)

            responses: list[dict[str, Any]] = []
            for worker in round_workers:
                connection = worker["connection"]
                if not connection.poll(600.0):
                    raise TimeoutError(
                        f"process sampler worker {worker['lane_index']} rollout timed out"
                    )
                response = connection.recv()
                if response.get("type") == "error":
                    raise RuntimeError(
                        "process sampler worker failed:\n"
                        + str(response.get("traceback", response.get("error")))
                    )
                if response.get("type") != "rollout":
                    raise RuntimeError(
                        f"unexpected process sampler response: {response}"
                    )
                responses.append(response)

            global_slot += sum(int(row["step_count"]) for row in responses)
            latest_update_stats = updater.update_many(
                [row["buffer"] for row in responses]
            )
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            for response_index, (worker, response) in enumerate(
                zip(round_workers, responses)
            ):
                terminal = bool(response["terminal"])
                extra = dict(response["episode_diagnostics"])
                extra.update(
                    {
                        "multisample_label": "multisample_process",
                        "sampler_backend": "process",
                        "num_envs": int(args.num_envs),
                        "active_env_count": int(len(round_workers)),
                        "environment_index": int(response["lane_index"]),
                        "environment_seed": int(response["environment_seed"]),
                        "worker_collect_seconds": float(
                            response["worker_collect_seconds"]
                        ),
                        "environment_slots_this_run": int(
                            global_slot - initial_global_slot
                        ),
                        "environment_slots_per_second": float(
                            (global_slot - initial_global_slot) / elapsed
                        ),
                    }
                )
                write_clean_training_log(
                    logger,
                    episode=int(response["episode"]),
                    global_slot=int(global_slot),
                    info=dict(response["last_info"]),
                    update_stats=(
                        latest_update_stats if response_index == 0 else None
                    ),
                    extra=extra,
                )
                progress.update(int(response["step_count"]))
                if terminal:
                    worker["episode"] = None
                    completed_episodes += 1

            checkpoint_manager.save(
                modules=modules,
                optimizer=updater.optimizer,
                episode=max(int(row["episode"]) for row in responses),
                global_slot=int(global_slot),
                update_step=int(updater.update_step),
                config_snapshot=build_config_snapshot(args),
                safe_boundary=True,
                filename="latest.pt",
            )
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            _write_json(
                logger.run_dir / "run_summary.json",
                {
                    "status": (
                        "completed"
                        if completed_episodes >= int(args.episodes)
                        else "running"
                    ),
                    "episode": int(completed_episodes - 1),
                    "completed_episode_count": int(completed_episodes),
                    "global_slot": int(global_slot),
                    "latest_update": asdict(latest_update_stats),
                    "num_envs": int(args.num_envs),
                    "sampler_backend": "process",
                    "multisample_label": "multisample_process",
                    "environment_seeds": [
                        int(worker["environment_seed"]) for worker in workers
                    ],
                    "environment_slots_this_run": int(
                        global_slot - initial_global_slot
                    ),
                    "environment_slots_per_second": float(
                        (global_slot - initial_global_slot) / elapsed
                    ),
                    "elapsed_seconds": float(elapsed),
                    "completed_dag_weight": float(args.completed_dag_weight),
                    "task_encoder": str(args.task_encoder),
                    "resume_semantics": "restart_from_new_episode_only",
                },
            )
    except BaseException as exc:
        failure = exc
    finally:
        progress.close()
        for worker in workers:
            connection = worker["connection"]
            process = worker["process"]
            if process.is_alive():
                try:
                    connection.send({"type": "shutdown"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for worker in workers:
            connection = worker["connection"]
            process = worker["process"]
            try:
                if connection.poll(15.0):
                    message = connection.recv()
                    if message.get("type") == "shutdown":
                        worker_health.append(dict(message["health"]))
            except (BrokenPipeError, EOFError, OSError):
                pass
            process.join(timeout=15.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
            connection.close()
            if not any(
                int(row["lane_index"]) == int(worker["lane_index"])
                for row in worker_health
            ):
                worker_health.append(
                    {
                        "lane_index": int(worker["lane_index"]),
                        "process_exitcode": process.exitcode,
                        "worker_health_missing": True,
                    }
                )
    if failure is not None:
        raise failure

    elapsed = max(time.perf_counter() - started_at, 1e-9)
    return {
        "run_dir": str(logger.run_dir),
        "global_slot": int(global_slot),
        "latest_update": (
            None if latest_update_stats is None else asdict(latest_update_stats)
        ),
        "num_envs": int(args.num_envs),
        "sampler_backend": "process",
        "multisample_label": "multisample_process",
        "environment_seeds": [
            _derive_environment_seed(int(args.seed), lane_index)
            for lane_index in range(int(args.num_envs))
        ],
        "environment_slots_this_run": int(global_slot - initial_global_slot),
        "environment_slots_per_second": float(
            (global_slot - initial_global_slot) / elapsed
        ),
        "elapsed_seconds": float(elapsed),
        "completed_dag_weight": float(args.completed_dag_weight),
        "detach_critic_hgnn": bool(args.detach_critic_hgnn),
        "critic_task_pooling": str(args.critic_task_pooling),
        **clean_counterfactual_experiment_controls(args),
        "freeze_ue_mobility": bool(args.freeze_ue_mobility),
        "offloading_counterfactual_coef": 0.0,
        "offloading_action_value_loss_coef": 0.0,
        "offloading_lagged_q_coef": 0.0,
        "offloading_lagged_q_loss_coef": 0.0,
        "kahypar_circuit_open": any(
            bool(row.get("kahypar_circuit_open", False))
            for row in worker_health
        ),
        "kahypar_last_failure_reason": next(
            (
                row.get("kahypar_last_failure_reason")
                for row in worker_health
                if row.get("kahypar_last_failure_reason") is not None
            ),
            None,
        ),
        "kahypar_cleanup_failed": any(
            bool(row.get("kahypar_cleanup_failed", False))
            for row in worker_health
        ),
        "kahypar_worker_alive_after_close": any(
            bool(row.get("kahypar_worker_alive_after_close", False))
            for row in worker_health
        ),
        "environment_health": worker_health,
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
    if movement_records:
        ready_task_ids_for_movement = list(
            encoded_state.prepared_state.frozen_ready_task_ids
        )
        env.apply_movement(movement_actions)
        service_centroid = env.compute_service_demand_centroid()
        post_move_positions = env.uav_service_positions
        area_diagonal = float(
            (config.AREA_WIDTH ** 2 + config.AREA_HEIGHT ** 2) ** 0.5
        )
        hover_action = config.CLEAN_MOVEMENT_ACTIONS.index(
            config.CLEAN_MOVEMENT_HOVER_ACTION
        )
        # Movement signal = -(post-move distance to the service-demand centroid
        # / map diagonal) minus a config-tunable movement-energy penalty for
        # non-hover actions. The centroid tracks users currently waiting for
        # service (hotspot-weighted), so UAVs learn to fly to the user-dense
        # area once and then hover to serve them; the energy term stops
        # constant chasing.
        movement_energy_penalty_signal = float(
            config.CLEAN_MOVEMENT_ENERGY_PENALTY_SIGNAL
        )
        centroid_normalized = (
            np.asarray(
                [
                    float(service_centroid[0]) / float(config.AREA_WIDTH),
                    float(service_centroid[1]) / float(config.AREA_HEIGHT),
                ],
                dtype=np.float32,
            )
            if service_centroid is not None
            else None
        )
        for movement_record in movement_records:
            uav_id = int(movement_record.uav_id)
            energy_penalty = (
                movement_energy_penalty_signal
                if int(movement_record.selected_action) != int(hover_action)
                else 0.0
            )
            distance_signal = 0.0
            if service_centroid is not None:
                uav_position = np.asarray(
                    post_move_positions.get(
                        uav_id, np.zeros((2,), dtype=np.float32)
                    ),
                    dtype=np.float32,
                ).reshape(-1)[:2]
                distance_signal = -float(
                    np.linalg.norm(uav_position - service_centroid)
                ) / area_diagonal
            movement_record.movement_position_signal = float(
                distance_signal - energy_penalty
            )
            if centroid_normalized is not None:
                movement_record.service_centroid_normalized = (
                    centroid_normalized.copy()
                )
    else:
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
            dynamic_uav_features=_immutable_numpy_copy(
                record.dynamic_uav_features, dtype=np.float32
            ),
            pair_features=_immutable_numpy_copy(
                record.pair_features, dtype=np.float32
            ),
            candidate_mask=_immutable_numpy_copy(
                record.candidate_mask, dtype=bool
            ),
            candidate_estimated_finish_times=_immutable_numpy_copy(
                record.candidate_estimated_finish_times, dtype=np.float32
            ),
            selected_action=int(record.selected_action),
            selected_uav_id=int(record.selected_uav_id),
            old_log_probability=float(record.old_log_prob),
            entropy=float(record.entropy),
            old_masked_probabilities=_immutable_numpy_copy(
                record.old_masked_probabilities,
                dtype=np.float32,
            ),
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


def _immutable_numpy_copy(value: Any, *, dtype: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    copied = np.asarray(value, dtype=dtype).copy()
    copied.setflags(write=False)
    return copied


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


def _build_process_worker_modules(
    *,
    task_feature_dim: int,
    task_embedding_dim: int,
    hidden_dim: int,
    task_encoder: str,
    critic_task_pooling: str,
    device: Any,
) -> CleanTrainingModules:
    from marl_models.hgnn import build_clean_task_encoder
    from marl_models.mappo.clean_movement_actor import CleanMovementActor
    from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
    from marl_models.mappo.clean_ppo import (
        CleanCentralizedCritic,
        clean_critic_input_dim,
    )

    critic_input_dim = clean_critic_input_dim(
        int(task_embedding_dim),
        config.NUM_UAVS,
        task_pooling=critic_task_pooling,
    )
    encoder = build_clean_task_encoder(
        encoder_type=str(task_encoder),
        task_feature_dim=int(task_feature_dim),
        hidden_dim=int(hidden_dim),
        output_dim=int(task_embedding_dim),
    )
    modules = CleanTrainingModules(
        hgnn=encoder,
        movement_actor=CleanMovementActor(
            task_embedding_dim=int(task_embedding_dim),
            hidden_dim=int(hidden_dim),
        ),
        offloading_actor=CleanOffloadingActor(
            task_embedding_dim=int(task_embedding_dim),
            hidden_dim=int(hidden_dim),
        ),
        critic=CleanCentralizedCritic(
            input_dim=int(critic_input_dim),
            hidden_dim=int(hidden_dim),
            task_pooling=critic_task_pooling,
        ),
    )
    _move_modules_to_device(modules, device)
    return modules


def _module_state_payload(modules: CleanTrainingModules) -> dict[str, dict[str, Any]]:
    return {
        name: {
            key: value.detach().cpu()
            for key, value in getattr(modules, name).state_dict().items()
        }
        for name in ("hgnn", "movement_actor", "offloading_actor", "critic")
    }


def _load_module_state_payload(
    modules: CleanTrainingModules,
    payload: dict[str, dict[str, Any]],
) -> None:
    for name in ("hgnn", "movement_actor", "offloading_actor", "critic"):
        getattr(modules, name).load_state_dict(payload[name])


def _move_modules_to_device(modules: CleanTrainingModules, device: Any) -> None:
    modules.hgnn.to(device)
    modules.movement_actor.to(device)
    modules.offloading_actor.to(device)
    modules.critic.to(device)
    if modules.offloading_action_value_critic is not None:
        modules.offloading_action_value_critic.to(device)
    if modules.offloading_lagged_q_critic is not None:
        modules.offloading_lagged_q_critic.to(device)
    if modules.offloading_decision_critic is not None:
        modules.offloading_decision_critic.to(device)
    if modules.movement_decision_critic is not None:
        modules.movement_decision_critic.to(device)


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
