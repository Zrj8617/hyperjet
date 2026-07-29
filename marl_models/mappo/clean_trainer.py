from __future__ import annotations

from dataclasses import dataclass, asdict, field
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from marl_models.mappo.clean_eft_auxiliary import (
    compute_eft_auxiliary_objective,
    eft_auxiliary_lambda,
    summarize_historical_eft_items,
)
from marl_models.mappo.clean_slot_orchestrator import (
    CleanEncodedSlotState,
    CleanSlotRolloutBuffer,
    CleanSlotRolloutRecord,
    encode_prepared_slot,
)

try:
    import torch
    from torch.distributions import Categorical
    from marl_models.mappo.clean_offloading_action_value import (
        masked_counterfactual_value,
        normalize_counterfactual_values,
    )
except ModuleNotFoundError:
    torch = None
    Categorical = None
    masked_counterfactual_value = None
    normalize_counterfactual_values = None


@dataclass(slots=True)
class CleanPPOUpdateConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    ppo_epochs: int = 1
    value_coef: float = 0.5
    # Keep critic predictions on the environment reward scale for GAE, but
    # standardize targets and predictions together inside the value loss.
    normalize_value_targets: bool = False
    # PPO-style clipping radius in the normalized value space. Zero disables
    # clipping and preserves the legacy squared-error objective.
    value_clip_epsilon: float = 0.0
    movement_entropy_coef: float = 0.01
    offloading_entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    # Experimental boundary: the critic head still trains, but its value loss
    # cannot update the shared HGNN when this is enabled.
    detach_critic_hgnn: bool = False
    offloading_counterfactual_coef: float = 0.0
    offloading_action_value_loss_coef: float = 0.0
    offloading_lagged_q_coef: float = 0.0
    offloading_lagged_q_loss_coef: float = 0.0
    # Phase-1 diagnostic only. Zero preserves the clean MAPPO baseline.
    eft_auxiliary_lambda_initial: float = 0.0
    eft_auxiliary_regret_scale: float = 1.0
    eft_auxiliary_constant_through_update: int = 8
    eft_auxiliary_zero_at_update: int = 20
    eft_auxiliary_sampling_seed: int = 0
    # Diagnostics-only: every N-th update, decompose the HGNN gradient into its
    # actor-loss and value-loss components via torch.autograd.grad (never
    # touches .grad). 0 disables the decomposition.
    hgnn_grad_decomposition_interval: int = 5


@dataclass(slots=True)
class CleanPPOUpdateStats:
    slot_count: int
    movement_action_count: int
    offloading_action_count: int
    offloading_effective_slot_count: int
    movement_loss: float
    offloading_loss: float
    value_loss: float
    movement_entropy: float
    offloading_entropy: float
    total_loss: float
    grad_norm: float
    hgnn_grad_norm: float
    update_step: int
    offloading_action_value_loss: float = 0.0
    offloading_lagged_q_loss: float = 0.0
    # Value/return scale diagnostics (Phase 4 P0): computed from the pre-update
    # GAE returns and value predictions of the rollout being consumed.
    returns_mean: float = 0.0
    returns_std: float = 0.0
    value_pred_mean: float = 0.0
    explained_variance: float = 0.0
    # Pure diagnostics (Phase 4 Commit 1): per-module pre/post-clip grad norms,
    # actual clip scale, HGNN actor/value grad decomposition (+cosine), and
    # rollout-time normalized entropies. Never used by the update itself.
    diagnostics: dict = field(default_factory=dict)


@dataclass(slots=True)
class CleanTrainingModules:
    hgnn: Any
    movement_actor: Any
    offloading_actor: Any
    critic: Any
    offloading_action_value_critic: Any | None = None
    offloading_lagged_q_critic: Any | None = None


class CleanJSONLLogger:
    """Small JSONL writer for clean training diagnostics only."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write(self, filename: str, payload: dict[str, Any]) -> None:
        path = self.run_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(payload), ensure_ascii=True, sort_keys=True) + "\n")


class CleanCheckpointManager:
    """Checkpoint helper restricted to T12 safe boundaries.

    First clean version restores model/optimizer/counters/random states and
    continues from a new episode. It does not claim mid-episode exact resume.
    """

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        modules: CleanTrainingModules,
        optimizer: Any,
        episode: int,
        global_slot: int,
        update_step: int,
        config_snapshot: dict[str, Any],
        safe_boundary: bool,
        filename: str = "latest.pt",
    ) -> Path:
        if not safe_boundary:
            raise RuntimeError("Clean checkpoint can only be saved after episode end or closed rollout update.")
        if torch is None:
            raise ModuleNotFoundError("torch is required to save clean model checkpoints")
        payload = {
            "hgnn": modules.hgnn.state_dict(),
            "movement_actor": modules.movement_actor.state_dict(),
            "offloading_actor": modules.offloading_actor.state_dict(),
            "critic": modules.critic.state_dict(),
            "optimizer": optimizer.state_dict(),
            "episode": int(episode),
            "global_slot": int(global_slot),
            "update_step": int(update_step),
            "config": dict(config_snapshot),
            "rng_state": _rng_state(),
            "resume_semantics": "restart_from_new_episode_only",
            "safe_boundary": True,
        }
        if modules.offloading_action_value_critic is not None:
            payload["offloading_action_value_critic"] = modules.offloading_action_value_critic.state_dict()
        if modules.offloading_lagged_q_critic is not None:
            payload["offloading_lagged_residual_q"] = modules.offloading_lagged_q_critic.state_dict()
        path = self.checkpoint_dir / filename
        torch.save(payload, path)
        return path

    def load(
        self,
        *,
        modules: CleanTrainingModules,
        optimizer: Any,
        path: str | Path,
    ) -> dict[str, Any]:
        payload = self.read(path)
        return self.restore(modules=modules, optimizer=optimizer, payload=payload)

    def read(self, path: str | Path) -> dict[str, Any]:
        """Read a trusted project checkpoint without mutating live training state."""
        if torch is None:
            raise ModuleNotFoundError("torch is required to load clean model checkpoints")
        try:
            return torch.load(Path(path), map_location="cpu", weights_only=False)
        except TypeError as exc:
            if "weights_only" not in str(exc):
                raise
            return torch.load(Path(path), map_location="cpu")

    def restore(
        self,
        *,
        modules: CleanTrainingModules,
        optimizer: Any,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Restore a previously validated checkpoint payload."""
        action_value_state = payload.get("offloading_action_value_critic")
        lagged_q_state = payload.get("offloading_lagged_residual_q")
        if modules.offloading_action_value_critic is None and action_value_state is not None:
            raise ValueError("checkpoint contains an enabled offloading action-value critic, but the live run disabled it")
        if modules.offloading_action_value_critic is not None and action_value_state is None:
            raise ValueError("enabled offloading action-value critic state is missing from checkpoint")
        modules.hgnn.load_state_dict(payload["hgnn"])
        modules.movement_actor.load_state_dict(payload["movement_actor"])
        modules.offloading_actor.load_state_dict(payload["offloading_actor"])
        modules.critic.load_state_dict(payload["critic"])
        if modules.offloading_action_value_critic is not None:
            modules.offloading_action_value_critic.load_state_dict(action_value_state)
        if modules.offloading_lagged_q_critic is None and lagged_q_state is not None:
            raise ValueError("checkpoint contains lagged residual Q, but the live run disabled it")
        if modules.offloading_lagged_q_critic is not None and lagged_q_state is None:
            raise ValueError("enabled lagged residual Q state is missing from checkpoint")
        if modules.offloading_lagged_q_critic is not None:
            modules.offloading_lagged_q_critic.load_state_dict(lagged_q_state)
        optimizer.load_state_dict(payload["optimizer"])
        _set_rng_state(payload.get("rng_state", {}))
        return payload


class CleanPPOUpdater:
    """Slot-level PPO updater for a closed clean rollout buffer."""

    def __init__(
        self,
        *,
        modules: CleanTrainingModules,
        optimizer: Any,
        config: CleanPPOUpdateConfig | None = None,
        device: str | Any = "cpu",
    ) -> None:
        if torch is None:
            raise ModuleNotFoundError("torch is required for clean PPO update")
        self.modules = modules
        self.optimizer = optimizer
        self.config = config or CleanPPOUpdateConfig()
        self.device = device
        self.update_step = 0
        _validate_action_value_configuration(self.config, self.modules)
        _validate_value_configuration(self.config)
        _validate_eft_auxiliary_configuration(self.config)

    def update(
        self,
        buffer: CleanSlotRolloutBuffer,
        *,
        lagged_q_samples: list[Any] | None = None,
        lagged_q_pending_count: int = 0,
    ) -> CleanPPOUpdateStats:
        return self.update_many(
            [buffer],
            lagged_q_samples=lagged_q_samples,
            lagged_q_pending_count=lagged_q_pending_count,
        )

    def update_many(
        self,
        buffers: list[CleanSlotRolloutBuffer],
        *,
        lagged_q_samples: list[Any] | None = None,
        lagged_q_pending_count: int = 0,
    ) -> CleanPPOUpdateStats:
        """Update once from independent closed rollout trajectories.

        GAE is evaluated separately for every buffer before the resulting arrays
        are concatenated. This prevents the last state of one environment from
        bootstrapping or discounting into the first state of another.
        """
        rollout_buffers = list(buffers)
        if not rollout_buffers:
            raise ValueError("Clean PPO multi-buffer update requires at least one buffer.")
        for buffer in rollout_buffers:
            if not buffer.closed:
                raise RuntimeError("Clean PPO update requires every rollout buffer to be closed.")
            if not buffer.records:
                raise ValueError("Cannot update from an empty rollout buffer.")
        records = [
            record
            for buffer in rollout_buffers
            for record in buffer.records
        ]
        lagged_samples = list(lagged_q_samples or [])
        frozen_lagged_corrections, lagged_correction_diagnostics = (
            self._precompute_lagged_q_corrections(records)
        )

        returns_np, advantages_np = compute_multi_trajectory_gae(
            rollout_buffers,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        # Scale diagnostics from the pre-update rollout: value predictions are
        # returns - advantages (GAE identity), explained variance uses raw
        # (un-normalized) advantages.
        values_np = returns_np - advantages_np
        returns_var = float(np.var(returns_np))
        explained_variance = (
            1.0 - float(np.var(advantages_np)) / returns_var if returns_var > 1e-12 else 0.0
        )
        scale_diags = {
            "returns_mean": float(np.mean(returns_np)),
            "returns_std": float(np.std(returns_np)),
            "value_pred_mean": float(np.mean(values_np)),
            "explained_variance": float(explained_variance),
        }
        returns = torch.as_tensor(returns_np, dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(values_np, dtype=torch.float32, device=self.device)
        if bool(self.config.normalize_value_targets):
            value_target_mean = returns.mean().detach()
            value_target_scale = returns.std(unbiased=False).detach().clamp_min(1e-8)
        else:
            value_target_mean = torch.zeros((), dtype=torch.float32, device=self.device)
            value_target_scale = torch.ones((), dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages_np, dtype=torch.float32, device=self.device)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)

        latest_stats: CleanPPOUpdateStats | None = None
        diagnostics: dict = _rollout_entropy_diagnostics(records)
        diagnostics.update(lagged_correction_diagnostics)
        diagnostics["rollout_environment_count"] = int(len(rollout_buffers))
        diagnostics["rollout_records_per_environment"] = [
            int(len(buffer.records)) for buffer in rollout_buffers
        ]
        diagnostics["offloading_lagged_q_pending_count"] = int(lagged_q_pending_count)
        diagnostics["critic_hgnn_detached"] = bool(self.config.detach_critic_hgnn)
        diagnostics.update(
            {
                "value_target_normalization_enabled": bool(self.config.normalize_value_targets),
                "value_target_normalization_mean": float(value_target_mean.cpu().item()),
                "value_target_normalization_scale": float(value_target_scale.cpu().item()),
                "value_clip_epsilon": float(self.config.value_clip_epsilon),
            }
        )
        decompose_interval = int(getattr(self.config, "hgnn_grad_decomposition_interval", 0))
        decompose_due = decompose_interval > 0 and (self.update_step % decompose_interval == 0)
        for epoch_index in range(max(int(self.config.ppo_epochs), 1)):
            auxiliary_generator = None
            if float(self.config.eft_auxiliary_lambda_initial) > 0.0:
                auxiliary_generator = torch.Generator(device=self.device)
                auxiliary_generator.manual_seed(
                    int(self.config.eft_auxiliary_sampling_seed)
                    + int(self.update_step) * max(int(self.config.ppo_epochs), 1)
                    + int(epoch_index)
                )
            loss_parts = self._loss(
                records=records,
                returns=returns,
                old_values=old_values,
                value_target_mean=value_target_mean,
                value_target_scale=value_target_scale,
                advantages=advantages,
                lagged_q_samples=lagged_samples,
                frozen_lagged_corrections=frozen_lagged_corrections,
                eft_auxiliary_generator=auxiliary_generator,
            )
            diagnostics.update(loss_parts.get("action_value_diagnostics", {}))
            diagnostics.update(loss_parts.get("lagged_q_diagnostics", {}))
            diagnostics.update(loss_parts.get("value_diagnostics", {}))
            diagnostics.update(loss_parts.get("eft_auxiliary_diagnostics", {}))
            if epoch_index == 0:
                target_parameters = _unique_parameters(
                    [self.modules.hgnn, self.modules.offloading_actor]
                )
                ppo_target_grad = _loss_to_parameter_grad_norm(
                    loss_parts["offloading_loss"], target_parameters
                )
                raw_eft_target_grad = _loss_to_parameter_grad_norm(
                    loss_parts["eft_auxiliary_loss"], target_parameters
                )
                weighted_eft_target_grad = _loss_to_parameter_grad_norm(
                    loss_parts["weighted_eft_auxiliary_loss"], target_parameters
                )
                diagnostics.update(
                    {
                        "eft_aux_target_grad_epoch": int(epoch_index),
                        "offloading_ppo_target_grad_norm": float(ppo_target_grad),
                        "eft_aux_raw_target_grad_norm": float(raw_eft_target_grad),
                        "eft_aux_weighted_target_grad_norm": float(weighted_eft_target_grad),
                        "eft_aux_weighted_to_ppo_grad_ratio": float(
                            weighted_eft_target_grad / max(ppo_target_grad, 1e-12)
                        ),
                        "eft_aux_direct_critic_grad_norm": _loss_to_module_grad_norm(
                            loss_parts["eft_auxiliary_loss"], self.modules.critic
                        ),
                        "eft_aux_direct_movement_grad_norm": _loss_to_module_grad_norm(
                            loss_parts["eft_auxiliary_loss"], self.modules.movement_actor
                        ),
                    }
                )
            if decompose_due and epoch_index == 0:
                # torch.autograd.grad reads the live graph without writing .grad,
                # so this is behavior-neutral for the optimizer step below.
                diagnostics.update(
                    _hgnn_grad_decomposition(
                        loss_parts=loss_parts,
                        hgnn=self.modules.hgnn,
                        config=self.config,
                    )
                )
            if epoch_index == 0 and self.modules.offloading_action_value_critic is not None:
                diagnostics["offloading_action_value_hgnn_grad_norm"] = _loss_to_module_grad_norm(
                    loss_parts["offloading_action_value_loss"],
                    self.modules.hgnn,
                )
            if epoch_index == 0 and self.modules.offloading_lagged_q_critic is not None:
                lagged_loss = loss_parts["offloading_lagged_q_loss"]
                diagnostics["offloading_lagged_q_direct_grad_q"] = _loss_to_module_grad_norm(
                    lagged_loss, self.modules.offloading_lagged_q_critic
                )
                diagnostics["offloading_lagged_q_direct_grad_hgnn"] = _loss_to_module_grad_norm(
                    lagged_loss, self.modules.hgnn
                )
                diagnostics["offloading_lagged_q_direct_grad_actor"] = _loss_to_module_grad_norm(
                    lagged_loss, self.modules.offloading_actor
                )
                diagnostics["offloading_lagged_q_direct_grad_critic"] = _loss_to_module_grad_norm(
                    lagged_loss, self.modules.critic
                )
            self.optimizer.zero_grad(set_to_none=True)
            loss_parts["total_loss"].backward()
            pre_clip = {
                "grad_pre_clip_movement": _module_grad_norm(self.modules.movement_actor),
                "grad_pre_clip_offloading": _module_grad_norm(self.modules.offloading_actor),
                "grad_pre_clip_critic": _module_grad_norm(self.modules.critic),
                "grad_pre_clip_hgnn": _module_grad_norm(self.modules.hgnn),
                "grad_pre_clip_offloading_action_value": _module_grad_norm(
                    self.modules.offloading_action_value_critic
                ),
                "grad_pre_clip_offloading_lagged_q": _module_grad_norm(
                    self.modules.offloading_lagged_q_critic
                ),
            }
            grad_norm = torch.nn.utils.clip_grad_norm_(
                _unique_parameters(
                    [
                        self.modules.hgnn,
                        self.modules.movement_actor,
                        self.modules.offloading_actor,
                        self.modules.critic,
                        self.modules.offloading_action_value_critic,
                        self.modules.offloading_lagged_q_critic,
                    ]
                ),
                float(self.config.max_grad_norm),
            )
            post_clip = {
                "grad_post_clip_movement": _module_grad_norm(self.modules.movement_actor),
                "grad_post_clip_offloading": _module_grad_norm(self.modules.offloading_actor),
                "grad_post_clip_critic": _module_grad_norm(self.modules.critic),
                "grad_post_clip_hgnn": _module_grad_norm(self.modules.hgnn),
                "grad_post_clip_offloading_action_value": _module_grad_norm(
                    self.modules.offloading_action_value_critic
                ),
                "grad_post_clip_offloading_lagged_q": _module_grad_norm(
                    self.modules.offloading_lagged_q_critic
                ),
            }
            diagnostics.update(pre_clip)
            diagnostics.update(post_clip)
            diagnostics["grad_pre_clip_global"] = float(grad_norm)
            # Pre/post-clip module norms always reflect the LAST completed PPO
            # epoch; the HGNN decomposition (if present) reflects epoch 0.
            diagnostics["grad_norms_epoch"] = int(epoch_index)
            diagnostics["grad_clip_scale"] = float(
                min(1.0, float(self.config.max_grad_norm) / max(float(grad_norm), 1e-12))
            )
            self.optimizer.step()
            latest_stats = self._stats_from_loss_parts(records, loss_parts, float(grad_norm))
            for key, value in scale_diags.items():
                setattr(latest_stats, key, float(value))
            latest_stats.diagnostics = dict(diagnostics)

        self.update_step += 1
        assert latest_stats is not None
        latest_stats.update_step = self.update_step
        return latest_stats

    def _precompute_lagged_q_corrections(
        self,
        records: list[CleanSlotRolloutRecord],
    ) -> tuple[list[Any], dict[str, Any]]:
        module = self.modules.offloading_lagged_q_critic
        diagnostics: dict[str, Any] = {
            "offloading_lagged_q_coef": float(self.config.offloading_lagged_q_coef),
            "offloading_lagged_q_loss_coef": float(self.config.offloading_lagged_q_loss_coef),
            "offloading_lagged_q_current_action_count": int(
                sum(len(record.offloading_records) for record in records)
            ),
            "offloading_lagged_q_legal_spread_mean": 0.0,
            "offloading_lagged_q_legal_spread_median": 0.0,
            "offloading_lagged_q_legal_spread_p90": 0.0,
            "offloading_lagged_q_correction_mean": 0.0,
            "offloading_lagged_q_correction_std": 0.0,
            "offloading_lagged_q_correction_min": 0.0,
            "offloading_lagged_q_correction_max": 0.0,
            "offloading_lagged_q_multi_candidate_fraction": 0.0,
            "offloading_lagged_q_clamp_fraction": 0.0,
        }
        if module is None:
            return [], diagnostics
        corrections: list[Any] = []
        spreads: list[float] = []
        raw_legal_count = 0
        clamped_count = 0
        multi_candidate_count = 0
        with torch.no_grad():
            for slot_record in records:
                for record in slot_record.offloading_records:
                    if record.candidate_features is None or record.critic_global_context is None:
                        raise ValueError("lagged Q current rollout record lacks candidate/context features")
                    features = torch.as_tensor(
                        record.candidate_features, dtype=torch.float32, device=self.device
                    )
                    context = torch.as_tensor(
                        record.critic_global_context, dtype=torch.float32, device=self.device
                    ).reshape(1, -1)
                    if features.dim() != 2:
                        raise ValueError("lagged Q current candidate features must be 2D")
                    candidate_count = int(features.shape[0])
                    q_inputs = torch.cat([features, context.expand(candidate_count, -1)], dim=1)
                    raw_q = module(q_inputs)
                    mask = torch.as_tensor(
                        record.candidate_mask, dtype=torch.bool, device=self.device
                    )
                    if raw_q.dim() != 1 or mask.shape != raw_q.shape or not bool(mask.any().item()):
                        raise ValueError("lagged Q current candidate output/mask mismatch")
                    if not bool(torch.isfinite(raw_q[mask]).all().item()):
                        raise FloatingPointError("lagged Q current legal values are non-finite")
                    raw_legal = raw_q[mask]
                    raw_legal_count += int(raw_legal.numel())
                    clamped_count += int(((raw_legal <= -1.0) | (raw_legal >= 1.0)).sum().item())
                    q_values = raw_q.clamp(-1.0, 1.0)
                    logits = self.modules.offloading_actor.scorer(features)
                    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
                    probabilities = torch.softmax(masked_logits, dim=0)
                    selected = int(record.selected_action)
                    if selected < 0 or selected >= candidate_count or not bool(mask[selected].item()):
                        raise ValueError("lagged Q selected action is not legal")
                    legal_count = int(mask.sum().item())
                    if legal_count > 1:
                        multi_candidate_count += 1
                        correction = q_values[selected] - torch.sum(probabilities * q_values)
                    else:
                        correction = q_values.new_zeros(())
                    corrections.append(correction.detach().reshape(()))
                    spreads.append(float((q_values[mask].max() - q_values[mask].min()).item()))
        if corrections:
            correction_values = np.asarray(
                [float(value.cpu().item()) for value in corrections], dtype=np.float64
            )
            spread_values = np.asarray(spreads, dtype=np.float64)
            diagnostics.update(
                {
                    "offloading_lagged_q_legal_spread_mean": float(spread_values.mean()),
                    "offloading_lagged_q_legal_spread_median": float(np.median(spread_values)),
                    "offloading_lagged_q_legal_spread_p90": float(np.percentile(spread_values, 90.0)),
                    "offloading_lagged_q_correction_mean": float(correction_values.mean()),
                    "offloading_lagged_q_correction_std": float(correction_values.std()),
                    "offloading_lagged_q_correction_min": float(correction_values.min()),
                    "offloading_lagged_q_correction_max": float(correction_values.max()),
                    "offloading_lagged_q_multi_candidate_fraction": float(
                        multi_candidate_count / len(corrections)
                    ),
                    "offloading_lagged_q_clamp_fraction": float(
                        clamped_count / max(raw_legal_count, 1)
                    ),
                }
            )
        return corrections, diagnostics

    def _loss(
        self,
        *,
        records: list[CleanSlotRolloutRecord],
        returns: Any,
        old_values: Any,
        value_target_mean: Any,
        value_target_scale: Any,
        advantages: Any,
        lagged_q_samples: list[Any] | None = None,
        frozen_lagged_corrections: list[Any] | None = None,
        eft_auxiliary_generator: Any | None = None,
    ) -> dict[str, Any]:
        per_slot_move_losses: list[Any] = []
        per_slot_move_entropies: list[Any] = []
        value_losses: list[Any] = []
        value_clip_indicators: list[Any] = []
        offloading_items: list[dict[str, Any]] = []
        eft_auxiliary_items: list[dict[str, Any]] = []
        action_value_losses_by_slot: dict[int, list[Any]] = {}
        action_value_targets: list[Any] = []
        selected_action_values: list[Any] = []
        legal_q_spreads: list[Any] = []
        raw_counterfactual_values: list[Any] = []
        action_value_enabled = self.modules.offloading_action_value_critic is not None
        lagged_q_enabled = self.modules.offloading_lagged_q_critic is not None
        lagged_corrections = list(frozen_lagged_corrections or [])

        for slot_idx, record in enumerate(records):
            task_features_np = np.asarray(record.graph_snapshot.task_features, dtype=np.float32).copy()
            incidence_np = np.asarray(record.graph_snapshot.incidence_matrix, dtype=np.float32).copy()
            hyperedge_type_ids_np = np.asarray(record.graph_snapshot.hyperedge_type_ids, dtype=np.int64).copy()
            task_features = torch.as_tensor(task_features_np, dtype=torch.float32, device=self.device)
            incidence = torch.as_tensor(incidence_np, dtype=torch.float32, device=self.device)
            hyperedge_type_ids = torch.as_tensor(hyperedge_type_ids_np, dtype=torch.long, device=self.device)
            task_embeddings = self.modules.hgnn(task_features, incidence, hyperedge_type_ids)
            critic_embeddings = (
                task_embeddings.detach()
                if bool(self.config.detach_critic_hgnn)
                else task_embeddings
            )
            critic_input = _critic_input_tensor(critic_embeddings, record.critic_non_graph_input)
            value = self.modules.critic(critic_input).reshape(-1)[0]
            value_loss, was_clipped = _normalized_clipped_value_loss(
                value=value,
                old_value=old_values[slot_idx],
                target=returns[slot_idx],
                target_mean=value_target_mean,
                target_scale=value_target_scale,
                clip_epsilon=float(self.config.value_clip_epsilon),
            )
            value_losses.append(value_loss)
            value_clip_indicators.append(was_clipped)

            move_losses: list[Any] = []
            move_entropies: list[Any] = []
            for movement_record in record.movement_records:
                logits = self.modules.movement_actor(
                    uav_features=torch.as_tensor(movement_record.uav_features, dtype=torch.float32, device=self.device).reshape(1, -1),
                    task_embeddings=task_embeddings,
                    ready_task_indices=list(movement_record.ready_task_indices),
                    pending_task_indices=list(movement_record.pending_task_indices),
                    ready_count_normalized=float(movement_record.ready_count_normalized),
                    pending_count_normalized=float(movement_record.pending_count_normalized),
                    boundary_action_mask=torch.as_tensor(movement_record.movement_mask, dtype=torch.bool, device=self.device).reshape(1, -1),
                )[0]
                dist = Categorical(logits=logits)
                action = torch.as_tensor(int(movement_record.selected_action), dtype=torch.long, device=self.device)
                old_log_prob = torch.as_tensor(float(movement_record.old_log_probability), dtype=torch.float32, device=self.device)
                move_losses.append(
                    _ppo_action_loss(
                        new_log_prob=dist.log_prob(action),
                        old_log_prob=old_log_prob,
                        advantage=advantages[slot_idx],
                        clip_epsilon=self.config.clip_epsilon,
                    )
                )
                move_entropies.append(dist.entropy())
            if move_losses:
                per_slot_move_losses.append(torch.stack(move_losses).mean())
                per_slot_move_entropies.append(torch.stack(move_entropies).mean())

            for offloading_record in record.offloading_records:
                task_idx = int(offloading_record.task_local_index)
                if task_idx < 0 or task_idx >= int(task_embeddings.shape[0]):
                    raise ValueError(
                        f"offloading task index {task_idx} is outside the slot embedding range "
                        f"[0, {int(task_embeddings.shape[0])})"
                    )
                task_embedding = task_embeddings[task_idx].reshape(1, -1)
                dynamic_features = torch.as_tensor(offloading_record.dynamic_uav_features, dtype=torch.float32, device=self.device)
                pair_features = torch.as_tensor(offloading_record.pair_features, dtype=torch.float32, device=self.device)
                if dynamic_features.dim() != 2 or pair_features.dim() != 2:
                    raise ValueError("offloading dynamic and pair features must be 2D")
                if dynamic_features.shape[0] != pair_features.shape[0]:
                    raise ValueError("offloading dynamic and pair candidate counts differ")
                candidate_count = int(dynamic_features.shape[0])
                if candidate_count <= 0:
                    raise ValueError("recorded offloading action has no candidates")
                features = torch.cat([task_embedding.expand(candidate_count, -1), dynamic_features, pair_features], dim=1)
                mask = torch.as_tensor(offloading_record.candidate_mask, dtype=torch.bool, device=self.device)
                if mask.dim() != 1 or int(mask.shape[0]) != candidate_count:
                    raise ValueError("offloading candidate mask shape is inconsistent with candidate features")
                if not bool(mask.any().item()):
                    raise ValueError("recorded offloading action has no legal candidate")
                logits = self.modules.offloading_actor.scorer(features).masked_fill(~mask, torch.finfo(torch.float32).min)
                dist = Categorical(logits=logits)
                action_index = int(offloading_record.selected_action)
                if action_index < 0 or action_index >= candidate_count or not bool(mask[action_index].item()):
                    raise ValueError("recorded offloading selected action is not a legal candidate")
                action = torch.as_tensor(action_index, dtype=torch.long, device=self.device)
                old_log_prob = torch.as_tensor(float(offloading_record.old_log_probability), dtype=torch.float32, device=self.device)
                item: dict[str, Any] = {
                    "slot_idx": int(slot_idx),
                    "dist": dist,
                    "action": action,
                    "old_log_prob": old_log_prob,
                    "entropy": dist.entropy(),
                }
                eft_values_np = offloading_record.candidate_estimated_finish_times
                if eft_values_np is None:
                    if float(self.config.eft_auxiliary_lambda_initial) > 0.0:
                        raise ValueError(
                            "offloading rollout record lacks decision-time candidate EFT vector"
                        )
                else:
                    eft_values = torch.as_tensor(
                        np.asarray(eft_values_np, dtype=np.float32).copy(),
                        dtype=torch.float32,
                        device=self.device,
                    ).detach()
                    if eft_values.shape != mask.shape:
                        raise ValueError(
                            "offloading candidate EFT vector shape is inconsistent"
                        )
                    eft_auxiliary_items.append(
                        {
                            "masked_logits": logits,
                            "candidate_mask": mask,
                            "candidate_estimated_finish_times": eft_values,
                            "rollout_action": int(action_index),
                            "decision_order": int(offloading_record.decision_order),
                        }
                    )
                if action_value_enabled:
                    global_context = critic_input.detach().reshape(1, -1).expand(candidate_count, -1)
                    action_value_input = torch.cat([features.detach(), global_context], dim=1)
                    action_values = self.modules.offloading_action_value_critic(action_value_input)
                    if action_values.dim() != 1 or int(action_values.shape[0]) != candidate_count:
                        raise ValueError("offloading action-value critic returned an inconsistent shape")
                    if not bool(torch.isfinite(action_values[mask]).all().item()):
                        raise FloatingPointError("offloading action-value critic returned non-finite legal values")
                    selected_value = action_values[action_index]
                    target = advantages[slot_idx].detach()
                    q_loss = 0.5 * (selected_value - target).pow(2)
                    counterfactual, q_spread = masked_counterfactual_value(
                        logits=logits,
                        action_values=action_values,
                        candidate_mask=mask,
                        selected_action=action_index,
                    )
                    if not bool(torch.isfinite(q_loss).item()):
                        raise FloatingPointError("offloading action-value loss is non-finite")
                    action_value_losses_by_slot.setdefault(int(slot_idx), []).append(q_loss)
                    action_value_targets.append(target)
                    selected_action_values.append(selected_value.detach())
                    legal_q_spreads.append(q_spread)
                    raw_counterfactual_values.append(counterfactual)
                offloading_items.append(item)

        normalized_counterfactual, counterfactual_diagnostics = normalize_counterfactual_values(
            raw_counterfactual_values
        ) if action_value_enabled else ([], {
            "mean": 0.0,
            "std": 0.0,
            "normalized_std": 0.0,
            "effective_count": 0,
        })

        if lagged_q_enabled and len(lagged_corrections) != len(offloading_items):
            raise ValueError(
                "frozen lagged-Q correction count must equal current PPO offloading action count"
            )

        off_losses_by_slot: dict[int, list[Any]] = {}
        off_entropies_by_slot: dict[int, list[Any]] = {}
        for item_index, item in enumerate(offloading_items):
            slot_idx = int(item["slot_idx"])
            offloading_advantage = advantages[slot_idx]
            if action_value_enabled:
                offloading_advantage = offloading_advantage + float(
                    self.config.offloading_counterfactual_coef
                ) * normalized_counterfactual[item_index]
            if lagged_q_enabled:
                offloading_advantage = offloading_advantage + float(
                    self.config.offloading_lagged_q_coef
                ) * lagged_corrections[item_index]
            off_loss = _ppo_action_loss(
                new_log_prob=item["dist"].log_prob(item["action"]),
                old_log_prob=item["old_log_prob"],
                advantage=offloading_advantage.detach(),
                clip_epsilon=self.config.clip_epsilon,
            )
            off_losses_by_slot.setdefault(slot_idx, []).append(off_loss)
            off_entropies_by_slot.setdefault(slot_idx, []).append(item["entropy"])

        per_slot_off_losses = [
            torch.stack(off_losses_by_slot[idx]).mean() for idx in sorted(off_losses_by_slot)
        ]
        per_slot_off_entropies = [
            torch.stack(off_entropies_by_slot[idx]).mean() for idx in sorted(off_entropies_by_slot)
        ]
        per_slot_action_value_losses = [
            torch.stack(action_value_losses_by_slot[idx]).mean()
            for idx in sorted(action_value_losses_by_slot)
        ]

        zero = torch.zeros((), dtype=torch.float32, device=self.device)
        movement_loss = torch.stack(per_slot_move_losses).mean() if per_slot_move_losses else zero
        movement_entropy = torch.stack(per_slot_move_entropies).mean() if per_slot_move_entropies else zero
        offloading_loss = torch.stack(per_slot_off_losses).mean() if per_slot_off_losses else zero
        offloading_entropy = torch.stack(per_slot_off_entropies).mean() if per_slot_off_entropies else zero
        value_loss = torch.stack(value_losses).mean() if value_losses else zero
        offloading_action_value_loss = (
            torch.stack(per_slot_action_value_losses).mean()
            if per_slot_action_value_losses
            else zero
        )
        offloading_lagged_q_loss, lagged_q_diagnostics = _lagged_q_regression_loss(
            module=self.modules.offloading_lagged_q_critic,
            samples=list(lagged_q_samples or []),
            device=self.device,
        )
        lagged_q_diagnostics["offloading_lagged_q_label_coverage"] = float(
            len(list(lagged_q_samples or [])) / max(len(offloading_items), 1)
        )
        eft_lambda = eft_auxiliary_lambda(
            self.update_step,
            lambda_initial=float(self.config.eft_auxiliary_lambda_initial),
            constant_through_update=int(
                self.config.eft_auxiliary_constant_through_update
            ),
            zero_at_update=int(self.config.eft_auxiliary_zero_at_update),
        )
        eft_rollout_diagnostics = summarize_historical_eft_items(eft_auxiliary_items)
        if (
            float(self.config.eft_auxiliary_lambda_initial) > 0.0
            and eft_auxiliary_items
        ):
            eft_auxiliary_loss, eft_auxiliary_diagnostics = (
                compute_eft_auxiliary_objective(
                    eft_auxiliary_items,
                    regret_scale=float(self.config.eft_auxiliary_regret_scale),
                    categorical_cls=Categorical,
                    generator=eft_auxiliary_generator,
                )
            )
        else:
            eft_auxiliary_loss = zero
            eft_auxiliary_diagnostics = {
                "eft_aux_effective_decision_count": 0,
                "eft_aux_excluded_zero_legal_count": 0,
                "eft_aux_excluded_single_legal_count": 0,
                "eft_aux_invalid_action_count": 0,
                "eft_aux_sampled_raw_regret_mean": 0.0,
                "eft_aux_sampled_raw_regret_p95": 0.0,
                "eft_aux_greedy_agreement": 0.0,
                "eft_aux_margin_ge_5s_accuracy": None,
                "eft_aux_margin_ge_20s_accuracy": None,
                "eft_aux_entropy_mean": 0.0,
            }
        weighted_eft_auxiliary_loss = float(eft_lambda) * eft_auxiliary_loss
        eft_auxiliary_diagnostics.update(eft_rollout_diagnostics)
        eft_auxiliary_diagnostics.update(
            {
                "eft_aux_lambda": float(eft_lambda),
                "eft_aux_lambda_initial": float(
                    self.config.eft_auxiliary_lambda_initial
                ),
                "eft_aux_regret_scale": float(
                    self.config.eft_auxiliary_regret_scale
                ),
                "eft_aux_raw_loss": float(
                    eft_auxiliary_loss.detach().cpu().item()
                ),
                "eft_aux_weighted_loss": float(
                    weighted_eft_auxiliary_loss.detach().cpu().item()
                ),
            }
        )
        total_loss = (
            movement_loss
            + offloading_loss
            + float(self.config.value_coef) * value_loss
            + float(self.config.offloading_action_value_loss_coef) * offloading_action_value_loss
            + float(self.config.offloading_lagged_q_loss_coef) * offloading_lagged_q_loss
            + weighted_eft_auxiliary_loss
            - float(self.config.movement_entropy_coef) * movement_entropy
            - float(self.config.offloading_entropy_coef) * offloading_entropy
        )
        for loss_name, loss_value in (
            ("movement_loss", movement_loss),
            ("offloading_loss", offloading_loss),
            ("value_loss", value_loss),
            ("offloading_action_value_loss", offloading_action_value_loss),
            ("offloading_lagged_q_loss", offloading_lagged_q_loss),
            ("eft_auxiliary_loss", eft_auxiliary_loss),
            ("weighted_eft_auxiliary_loss", weighted_eft_auxiliary_loss),
            ("total_loss", total_loss),
        ):
            if not bool(torch.isfinite(loss_value).item()):
                raise FloatingPointError(f"{loss_name} is non-finite")

        action_value_diagnostics = _action_value_diagnostics(
            targets=action_value_targets,
            selected_values=selected_action_values,
            legal_q_spreads=legal_q_spreads,
            counterfactual=counterfactual_diagnostics,
            config=self.config,
        )
        return {
            "movement_loss": movement_loss,
            "movement_entropy": movement_entropy,
            "offloading_loss": offloading_loss,
            "offloading_entropy": offloading_entropy,
            "value_loss": value_loss,
            "offloading_action_value_loss": offloading_action_value_loss,
            "offloading_lagged_q_loss": offloading_lagged_q_loss,
            "eft_auxiliary_loss": eft_auxiliary_loss,
            "weighted_eft_auxiliary_loss": weighted_eft_auxiliary_loss,
            "total_loss": total_loss,
            "action_value_diagnostics": action_value_diagnostics,
            "lagged_q_diagnostics": lagged_q_diagnostics,
            "eft_auxiliary_diagnostics": eft_auxiliary_diagnostics,
            "value_diagnostics": {
                "value_clip_fraction": float(
                    torch.stack(value_clip_indicators).mean().detach().cpu().item()
                ) if value_clip_indicators else 0.0,
                "value_normalized_target_mean": float(
                    ((returns - value_target_mean) / value_target_scale).mean().detach().cpu().item()
                ),
                "value_normalized_target_std": float(
                    ((returns - value_target_mean) / value_target_scale)
                    .std(unbiased=False).detach().cpu().item()
                ),
            },
        }

    def _stats_from_loss_parts(
        self,
        records: list[CleanSlotRolloutRecord],
        loss_parts: dict[str, Any],
        grad_norm: float,
    ) -> CleanPPOUpdateStats:
        return CleanPPOUpdateStats(
            slot_count=len(records),
            movement_action_count=sum(len(record.movement_records) for record in records),
            offloading_action_count=sum(len(record.offloading_records) for record in records),
            offloading_effective_slot_count=sum(1 for record in records if record.offloading_records),
            movement_loss=float(loss_parts["movement_loss"].detach().cpu().item()),
            offloading_loss=float(loss_parts["offloading_loss"].detach().cpu().item()),
            value_loss=float(loss_parts["value_loss"].detach().cpu().item()),
            movement_entropy=float(loss_parts["movement_entropy"].detach().cpu().item()),
            offloading_entropy=float(loss_parts["offloading_entropy"].detach().cpu().item()),
            total_loss=float(loss_parts["total_loss"].detach().cpu().item()),
            grad_norm=float(grad_norm),
            hgnn_grad_norm=_module_grad_norm(self.modules.hgnn),
            update_step=self.update_step,
            offloading_action_value_loss=float(
                loss_parts["offloading_action_value_loss"].detach().cpu().item()
            ),
            offloading_lagged_q_loss=float(
                loss_parts["offloading_lagged_q_loss"].detach().cpu().item()
            ),
        )


def _validate_action_value_configuration(
    config: CleanPPOUpdateConfig,
    modules: CleanTrainingModules,
) -> None:
    beta = float(config.offloading_counterfactual_coef)
    eta = float(config.offloading_action_value_loss_coef)
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("offloading counterfactual coefficient must be finite and non-negative")
    if not np.isfinite(eta) or eta < 0.0:
        raise ValueError("offloading action-value loss coefficient must be finite and non-negative")
    coefficient_enabled = beta > 0.0 and eta > 0.0
    if (beta > 0.0) != (eta > 0.0):
        raise ValueError("offloading counterfactual and action-value loss coefficients must be enabled together")
    module_enabled = modules.offloading_action_value_critic is not None
    if coefficient_enabled != module_enabled:
        raise ValueError(
            "offloading action-value critic presence must match the enabled coefficient pair"
        )

    lagged_beta = float(config.offloading_lagged_q_coef)
    lagged_eta = float(config.offloading_lagged_q_loss_coef)
    if not np.isfinite(lagged_beta) or lagged_beta < 0.0:
        raise ValueError("offloading lagged-Q coefficient must be finite and non-negative")
    if not np.isfinite(lagged_eta) or lagged_eta < 0.0:
        raise ValueError("offloading lagged-Q loss coefficient must be finite and non-negative")
    lagged_coefficients_enabled = lagged_beta > 0.0 and lagged_eta > 0.0
    if (lagged_beta > 0.0) != (lagged_eta > 0.0):
        raise ValueError("offloading lagged-Q advantage and loss coefficients must be enabled together")
    lagged_module_enabled = modules.offloading_lagged_q_critic is not None
    if lagged_coefficients_enabled != lagged_module_enabled:
        raise ValueError("lagged residual-Q module presence must match its enabled coefficient pair")
    if coefficient_enabled and lagged_coefficients_enabled:
        raise ValueError("offloading counterfactual v1 and lagged-Q v2 are mutually exclusive")


def _validate_value_configuration(config: CleanPPOUpdateConfig) -> None:
    if not isinstance(config.normalize_value_targets, bool):
        raise ValueError("normalize_value_targets must be boolean")
    clip_epsilon = float(config.value_clip_epsilon)
    if not np.isfinite(clip_epsilon) or clip_epsilon < 0.0:
        raise ValueError("value clip epsilon must be finite and non-negative")


def _validate_eft_auxiliary_configuration(config: CleanPPOUpdateConfig) -> None:
    initial = float(config.eft_auxiliary_lambda_initial)
    scale = float(config.eft_auxiliary_regret_scale)
    if not np.isfinite(initial) or initial < 0.0:
        raise ValueError("EFT auxiliary lambda must be finite and non-negative")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("EFT auxiliary regret scale must be finite and positive")
    # Validate the full schedule even when disabled, so saved controls are
    # unambiguous and resume checks can reject malformed configurations.
    eft_auxiliary_lambda(
        0,
        lambda_initial=initial,
        constant_through_update=int(config.eft_auxiliary_constant_through_update),
        zero_at_update=int(config.eft_auxiliary_zero_at_update),
    )
    if initial > 0.0 and (
        float(config.offloading_counterfactual_coef) > 0.0
        or float(config.offloading_action_value_loss_coef) > 0.0
        or float(config.offloading_lagged_q_coef) > 0.0
        or float(config.offloading_lagged_q_loss_coef) > 0.0
    ):
        raise ValueError(
            "EFT auxiliary diagnostic cannot be combined with counterfactual or lagged-Q paths"
        )


def _normalized_clipped_value_loss(
    *,
    value: Any,
    old_value: Any,
    target: Any,
    target_mean: Any,
    target_scale: Any,
    clip_epsilon: float,
) -> tuple[Any, Any]:
    """PPO value loss without changing the critic's raw reward-scale output."""
    normalized_value = (value - target_mean) / target_scale
    normalized_old_value = (old_value - target_mean) / target_scale
    normalized_target = (target - target_mean) / target_scale
    loss_unclipped = (normalized_value - normalized_target).pow(2)
    if float(clip_epsilon) <= 0.0:
        return 0.5 * loss_unclipped, torch.zeros((), dtype=torch.float32, device=value.device)
    delta = normalized_value - normalized_old_value
    clipped_value = normalized_old_value + delta.clamp(-float(clip_epsilon), float(clip_epsilon))
    loss_clipped = (clipped_value - normalized_target).pow(2)
    was_clipped = (delta.abs() > float(clip_epsilon)).to(dtype=torch.float32)
    return 0.5 * torch.maximum(loss_unclipped, loss_clipped), was_clipped


def _lagged_q_regression_loss(
    *,
    module: Any | None,
    samples: list[Any],
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    zero = torch.zeros((), dtype=torch.float32, device=device)
    diagnostics: dict[str, Any] = {
        "offloading_lagged_q_training_sample_count": int(len(samples)),
        "offloading_lagged_q_completed_sample_count": int(
            sum(not bool(sample.censored) for sample in samples)
        ),
        "offloading_lagged_q_censored_sample_count": int(
            sum(bool(sample.censored) for sample in samples)
        ),
        "offloading_lagged_q_effective_weight": 0.0,
        "offloading_lagged_q_censor_fraction": 0.0,
        "offloading_lagged_q_target_mean": 0.0,
        "offloading_lagged_q_target_std": 0.0,
        "offloading_lagged_q_target_min": 0.0,
        "offloading_lagged_q_target_max": 0.0,
        "offloading_lagged_q_selected_mean": 0.0,
        "offloading_lagged_q_selected_std": 0.0,
        "offloading_lagged_q_explained_variance": 0.0,
    }
    if module is None:
        if samples:
            raise ValueError("lagged Q samples were supplied while the module is disabled")
        return zero, diagnostics
    if not samples:
        return zero, diagnostics
    if any(hasattr(sample, "old_log_probability") or hasattr(sample, "old_log_prob") for sample in samples):
        raise ValueError("lagged Q regression samples must not contain PPO log probabilities")
    inputs = torch.as_tensor(
        np.stack([np.asarray(sample.selected_input, dtype=np.float32) for sample in samples]),
        dtype=torch.float32,
        device=device,
    )
    targets = torch.as_tensor(
        [float(sample.target) for sample in samples], dtype=torch.float32, device=device
    )
    weights = torch.as_tensor(
        [float(sample.weight) for sample in samples], dtype=torch.float32, device=device
    )
    if not bool(torch.isfinite(inputs).all().item()) or not bool(torch.isfinite(targets).all().item()):
        raise FloatingPointError("lagged Q regression input/target is non-finite")
    if not bool(torch.isfinite(weights).all().item()) or bool((weights < 0.0).any().item()):
        raise FloatingPointError("lagged Q regression weight is invalid")
    effective_weight = weights.sum()
    if float(effective_weight.item()) <= 0.0:
        return zero, diagnostics
    predictions = module(inputs)
    if predictions.shape != targets.shape or not bool(torch.isfinite(predictions).all().item()):
        raise FloatingPointError("lagged Q regression prediction is invalid")
    element_losses = torch.nn.functional.smooth_l1_loss(
        predictions, targets, reduction="none"
    )
    loss = torch.sum(element_losses * weights) / effective_weight
    target_variance = targets.var(unbiased=False)
    residual_variance = (targets - predictions.detach()).var(unbiased=False)
    explained_variance = (
        1.0 - float(residual_variance.item()) / float(target_variance.item())
        if float(target_variance.item()) > 1e-12
        else 0.0
    )
    diagnostics.update(
        {
            "offloading_lagged_q_effective_weight": float(effective_weight.item()),
            "offloading_lagged_q_censor_fraction": float(
                sum(bool(sample.censored) for sample in samples) / len(samples)
            ),
            "offloading_lagged_q_target_mean": float(targets.mean().item()),
            "offloading_lagged_q_target_std": float(targets.std(unbiased=False).item()),
            "offloading_lagged_q_target_min": float(targets.min().item()),
            "offloading_lagged_q_target_max": float(targets.max().item()),
            "offloading_lagged_q_selected_mean": float(predictions.detach().mean().item()),
            "offloading_lagged_q_selected_std": float(
                predictions.detach().std(unbiased=False).item()
            ),
            "offloading_lagged_q_explained_variance": float(explained_variance),
        }
    )
    return loss, diagnostics


def _action_value_diagnostics(
    *,
    targets: list[Any],
    selected_values: list[Any],
    legal_q_spreads: list[Any],
    counterfactual: dict[str, Any],
    config: CleanPPOUpdateConfig,
) -> dict[str, Any]:
    diagnostics = {
        "offloading_counterfactual_coef": float(config.offloading_counterfactual_coef),
        "offloading_action_value_loss_coef": float(config.offloading_action_value_loss_coef),
        "offloading_action_value_target_mean": 0.0,
        "offloading_action_value_target_std": 0.0,
        "offloading_action_value_selected_mean": 0.0,
        "offloading_action_value_selected_std": 0.0,
        "offloading_action_value_explained_variance": 0.0,
        "offloading_legal_q_spread_mean": 0.0,
        "offloading_counterfactual_advantage_mean": float(counterfactual.get("mean", 0.0)),
        "offloading_counterfactual_advantage_std": float(counterfactual.get("std", 0.0)),
        "offloading_counterfactual_advantage_normalized_std": float(
            counterfactual.get("normalized_std", 0.0)
        ),
        "offloading_counterfactual_effective_action_count": int(
            counterfactual.get("effective_count", 0)
        ),
    }
    if not targets:
        return diagnostics

    target_tensor = torch.stack([value.detach().reshape(()) for value in targets])
    selected_tensor = torch.stack([value.detach().reshape(()) for value in selected_values])
    if not bool(torch.isfinite(target_tensor).all().item()):
        raise FloatingPointError("offloading action-value targets contain non-finite values")
    if not bool(torch.isfinite(selected_tensor).all().item()):
        raise FloatingPointError("selected offloading action values contain non-finite values")
    target_variance = target_tensor.var(unbiased=False)
    residual_variance = (target_tensor - selected_tensor).var(unbiased=False)
    explained_variance = (
        1.0 - float(residual_variance.item()) / float(target_variance.item())
        if float(target_variance.item()) > 1e-12
        else 0.0
    )
    spread_mean = (
        float(torch.stack([value.detach().reshape(()) for value in legal_q_spreads]).mean().item())
        if legal_q_spreads
        else 0.0
    )
    diagnostics.update(
        {
            "offloading_action_value_target_mean": float(target_tensor.mean().item()),
            "offloading_action_value_target_std": float(target_tensor.std(unbiased=False).item()),
            "offloading_action_value_selected_mean": float(selected_tensor.mean().item()),
            "offloading_action_value_selected_std": float(selected_tensor.std(unbiased=False).item()),
            "offloading_action_value_explained_variance": float(explained_variance),
            "offloading_legal_q_spread_mean": spread_mean,
        }
    )
    return diagnostics


def _loss_to_module_grad_norm(loss: Any, module: Any) -> float:
    if module is None or not (
        isinstance(loss, torch.Tensor) and loss.requires_grad and loss.grad_fn is not None
    ):
        return 0.0
    params = [param for param in module.parameters() if param.requires_grad]
    if not params:
        return 0.0
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = sum(
        float(grad.detach().pow(2).sum().cpu().item())
        for grad in grads
        if grad is not None
    )
    return float(total ** 0.5)


def _loss_to_parameter_grad_norm(loss: Any, parameters: list[Any]) -> float:
    params = [param for param in parameters if param.requires_grad]
    if not params or not (
        isinstance(loss, torch.Tensor) and loss.requires_grad and loss.grad_fn is not None
    ):
        return 0.0
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = sum(
        float(grad.detach().pow(2).sum().cpu().item())
        for grad in grads
        if grad is not None
    )
    return float(total ** 0.5)


def build_single_optimizer(
    modules: CleanTrainingModules,
    *,
    lr: float = 3e-4,
    optimizer_cls: Any | None = None,
) -> Any:
    if torch is None:
        raise ModuleNotFoundError("torch is required to build clean PPO optimizer")
    optimizer_type = optimizer_cls or torch.optim.Adam
    return optimizer_type(
        _unique_parameters(
            [
                modules.hgnn,
                modules.movement_actor,
                modules.offloading_actor,
                modules.critic,
                modules.offloading_action_value_critic,
                modules.offloading_lagged_q_critic,
            ]
        ),
        lr=float(lr),
    )


def close_rollout_with_bootstrap(
    *,
    buffer: CleanSlotRolloutBuffer,
    next_encoded_state: CleanEncodedSlotState | None,
    terminated: bool,
) -> float:
    bootstrap = 0.0 if bool(terminated) or next_encoded_state is None or next_encoded_state.value is None else float(next_encoded_state.value)
    buffer.close(
        bootstrap_value=bootstrap,
        next_prepared_state=None if next_encoded_state is None else next_encoded_state.prepared_state,
    )
    return bootstrap


def reencode_prepared_after_update(
    *,
    prepared_state: Any,
    env: Any,
    modules: CleanTrainingModules,
    device: str | Any = "cpu",
    detach_critic_hgnn: bool = False,
) -> CleanEncodedSlotState:
    return encode_prepared_slot(
        prepared_state=prepared_state,
        env=env,
        hgnn=modules.hgnn,
        critic=modules.critic,
        movement_actor=modules.movement_actor,
        device=device,
        detach_critic_hgnn=detach_critic_hgnn,
    )


def compute_slot_level_gae(
    records: list[CleanSlotRolloutRecord],
    *,
    bootstrap_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros((len(records),), dtype=np.float32)
    returns = np.zeros((len(records),), dtype=np.float32)
    gae = 0.0
    next_value = float(bootstrap_value)
    for idx in range(len(records) - 1, -1, -1):
        record = records[idx]
        mask = 0.0 if bool(record.terminated) else 1.0
        delta = float(record.reward) + float(gamma) * mask * next_value - float(record.value)
        gae = delta + float(gamma) * float(gae_lambda) * mask * gae
        advantages[idx] = float(gae)
        returns[idx] = float(gae + float(record.value))
        next_value = float(record.value)
    return returns, advantages


def compute_multi_trajectory_gae(
    buffers: list[CleanSlotRolloutBuffer],
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute independent GAE recursions and concatenate their results."""
    rollout_buffers = list(buffers)
    if not rollout_buffers:
        raise ValueError("multi-trajectory GAE requires at least one rollout buffer")
    trajectory_results: list[tuple[np.ndarray, np.ndarray]] = []
    for buffer in rollout_buffers:
        records = list(buffer.records)
        if not buffer.closed:
            raise RuntimeError("multi-trajectory GAE requires closed rollout buffers")
        if not records:
            raise ValueError("multi-trajectory GAE does not accept empty rollout buffers")
        trajectory_results.append(
            compute_slot_level_gae(
                records,
                bootstrap_value=_buffer_bootstrap_value(records),
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
        )
    return (
        np.concatenate([returns for returns, _ in trajectory_results], axis=0),
        np.concatenate(
            [advantages for _, advantages in trajectory_results],
            axis=0,
        ),
    )


def write_clean_training_log(
    logger: CleanJSONLLogger,
    *,
    episode: int,
    global_slot: int,
    info: dict[str, Any],
    update_stats: CleanPPOUpdateStats | None = None,
    torch_skipped: bool = False,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "episode": int(episode),
        "global_slot": int(global_slot),
        "reward": float(info.get("step_reward", 0.0)),
        "reward_time_penalty": float(info.get("step_time_penalty", 0.0)),
        "reward_energy_penalty": float(info.get("step_energy_penalty", 0.0)),
        "reward_task_energy_penalty": float(info.get("step_task_energy_penalty", 0.0)),
        "reward_movement_energy_penalty": float(info.get("step_movement_energy_penalty", 0.0)),
        "reward_completed_dag_bonus": float(info.get("step_completed_dag_bonus", 0.0)),
        "generated_DAG_count": info.get("generated_dag_count"),
        "completed_DAG_count": info.get("completed_dag_count"),
        "DAG_completion_rate": info.get("dag_completion_rate"),
        "DAG_throughput": info.get("dag_throughput"),
        "average_dag_flowtime": info.get(
            "average_dag_flowtime", info.get("avg_dag_flowtime")
        ),
        "arrival_attempt_count": info.get("arrival_attempt_count"),
        "arrival_admitted_count": info.get("arrival_admitted_count"),
        "arrival_blocked_count": info.get("arrival_blocked_count"),
        "arrival_blocked_reasons": info.get("arrival_blocked_reasons", {}),
        "step_arrival_attempt_count": info.get("step_arrival_attempt_count"),
        "step_arrival_admitted_count": info.get("step_arrival_admitted_count"),
        "step_arrival_blocked_count": info.get("step_arrival_blocked_count"),
        "step_arrival_blocked_reasons": info.get(
            "step_arrival_blocked_reasons", {}
        ),
        "invalid_assignment_count": info.get("invalid_assignment_count"),
        "invalid_assignment_rate": info.get("invalid_assignment_rate"),
        "action_executed_rate": info.get("action_executed_rate"),
        "movement_action_distribution": info.get("movement_action_distribution", {}),
        "offloading_action_count": info.get("offloading_action_count", info.get("assignment_buffer_entry_count", 0)),
        "kahypar_partition_status": info.get("kahypar_partition_status"),
        "kahypar_degraded_label": info.get("kahypar_degraded_label"),
        "hover_action_ratio": info.get("hover_action_ratio"),
        "movement_frozen": bool(info.get("movement_frozen", False)),
        "avg_uav_queue_length": info.get("avg_uav_queue_length"),
        "active_dags": info.get("active_dags"),
        "frozen_ready_task_count": info.get("frozen_ready_task_count"),
        "offloading_skipped_no_candidate": info.get("offloading_skipped_no_candidate"),
        "mean_uav_displacement_per_slot": info.get("mean_uav_displacement_per_slot"),
        "terminated": bool(info.get("terminated", False)),
        "truncated": bool(info.get("truncated", False)),
        "torch_model_checks_skipped": bool(torch_skipped),
    }
    if update_stats is not None:
        payload.update({f"ppo_{key}": value for key, value in asdict(update_stats).items()})
    if extra:
        payload.update(extra)
    logger.write("train_metrics.jsonl", payload)


def _buffer_bootstrap_value(records: list[CleanSlotRolloutRecord]) -> float:
    if not records:
        return 0.0
    last = records[-1]
    if bool(last.terminated):
        return 0.0
    if last.bootstrap_value is not None:
        return float(last.bootstrap_value)
    if last.next_value is not None:
        return float(last.next_value)
    return 0.0


def _ppo_action_loss(*, new_log_prob: Any, old_log_prob: Any, advantage: Any, clip_epsilon: float) -> Any:
    ratio = torch.exp(new_log_prob - old_log_prob)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon)) * advantage
    return -torch.minimum(unclipped, clipped)


def _critic_input_tensor(task_embeddings: Any, critic_non_graph_input: np.ndarray) -> Any:
    if task_embeddings.shape[0] > 0:
        active_mean = task_embeddings.mean(dim=0)
    else:
        active_mean = task_embeddings.new_zeros((task_embeddings.shape[1],))
    non_graph = torch.as_tensor(critic_non_graph_input, dtype=task_embeddings.dtype, device=task_embeddings.device)
    return torch.cat([active_mean, non_graph], dim=0)


def _unique_parameters(modules: Iterable[Any]) -> list[Any]:
    params = []
    seen: set[int] = set()
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            marker = id(param)
            if marker not in seen:
                params.append(param)
                seen.add(marker)
    return params


def _module_grad_norm(module: Any) -> float:
    if torch is None or module is None:
        return 0.0
    total = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        total += float(param.grad.detach().pow(2).sum().cpu().item())
    return float(total ** 0.5)


def _rollout_entropy_diagnostics(records: list[CleanSlotRolloutRecord]) -> dict:
    """Normalized rollout-time entropies (diagnostics only).

    Offloading entropy is normalized by log(n_valid_candidates) per action;
    actions with n_valid <= 1 are excluded from the normalized mean (a single
    legal candidate has no meaningful entropy). Movement likewise uses the
    boundary-legal action count. Raw entropies stay in the main stats.
    """
    import math

    off_norm: list[float] = []
    off_valid: list[int] = []
    move_norm: list[float] = []
    move_valid: list[int] = []
    for record in records:
        for off in record.offloading_records:
            n_valid = int(np.asarray(off.candidate_mask, dtype=bool).sum())
            off_valid.append(n_valid)
            if n_valid >= 2:
                off_norm.append(float(off.entropy) / math.log(n_valid))
        for move in record.movement_records:
            n_valid = int(np.asarray(move.movement_mask, dtype=bool).sum())
            move_valid.append(n_valid)
            if n_valid >= 2:
                move_norm.append(float(move.entropy) / math.log(n_valid))
    return {
        "rollout_offloading_entropy_normalized_mean": float(np.mean(off_norm)) if off_norm else None,
        "rollout_offloading_valid_candidates_mean": float(np.mean(off_valid)) if off_valid else None,
        "rollout_movement_entropy_normalized_mean": float(np.mean(move_norm)) if move_norm else None,
        "rollout_movement_valid_actions_mean": float(np.mean(move_valid)) if move_valid else None,
    }


def _hgnn_grad_decomposition(*, loss_parts: dict, hgnn: Any, config: CleanPPOUpdateConfig) -> dict:
    """Decompose the HGNN gradient into actor-loss vs value-loss components.

    Uses torch.autograd.grad with retain_graph=True so .grad is never written
    and the subsequent total_loss.backward() sees an untouched graph. The actor
    term uses the ACTUAL weighted objective (policy losses minus entropy
    bonuses); the value term includes the actual value_coef.

    Epoch semantics: computed on PPO epoch 0 only, i.e. against the pre-update
    parameters of the rollout being consumed (reported as
    hgnn_decomposition_epoch=0). The per-module pre/post-clip norms elsewhere in
    diagnostics refer to the LAST PPO epoch (grad_norms_epoch).

    Frozen-movement rollouts can yield loss terms that are graph-free constants
    (e.g. zero movement loss, or zero offloading loss when M_t = 0 everywhere).
    Terms without a grad_fn are reported as norm 0.0 and the cosine as None.
    """
    params = [param for param in hgnn.parameters() if param.requires_grad]
    if not params:
        return {}
    actor_term = (
        loss_parts["movement_loss"]
        + loss_parts["offloading_loss"]
        - float(config.movement_entropy_coef) * loss_parts["movement_entropy"]
        - float(config.offloading_entropy_coef) * loss_parts["offloading_entropy"]
    )
    value_term = float(config.value_coef) * loss_parts["value_loss"]

    def _flat_or_none(term: Any) -> Any:
        if not (isinstance(term, torch.Tensor) and term.requires_grad and term.grad_fn is not None):
            return None
        grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        return torch.cat(
            [
                (grad if grad is not None else torch.zeros_like(param)).reshape(-1)
                for grad, param in zip(grads, params)
            ]
        )

    actor_flat = _flat_or_none(actor_term)
    value_flat = _flat_or_none(value_term)
    actor_norm = 0.0 if actor_flat is None else float(actor_flat.norm().item())
    value_norm = 0.0 if value_flat is None else float(value_flat.norm().item())
    if actor_flat is None or value_flat is None or actor_norm * value_norm < 1e-12:
        cosine = None
    else:
        cosine = float(torch.dot(actor_flat, value_flat).item() / (actor_norm * value_norm))
    return {
        "hgnn_actor_grad_norm": actor_norm,
        "hgnn_value_grad_norm": value_norm,
        "hgnn_actor_value_cosine": cosine,
        "hgnn_decomposition_epoch": 0,
    }


def _rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch is not None:
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if torch is not None and "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch is not None and torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value
