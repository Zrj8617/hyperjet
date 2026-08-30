from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Iterable

import numpy as np
import torch

import config
from marl_models.mappo.clean_decision_transitions import (
    CleanDecisionState,
    CleanDecisionTransition,
)
from marl_models.mappo.clean_ppo import (
    CleanDecisionCritic,
    normalized_clipped_value_loss,
)


DecisionKey = tuple[int, int, int, int, str]


def decision_state_key(state: CleanDecisionState) -> DecisionKey:
    return (
        int(state.episode_index),
        int(state.lane_index),
        int(state.slot_index),
        int(state.decision_order),
        str(state.task_id),
    )


def encode_decision_state(state: CleanDecisionState) -> np.ndarray:
    """Encode action-independent V(s_decision) input with legal-set pooling."""
    candidates = np.asarray(state.candidate_features, dtype=np.float32)
    mask = np.asarray(state.candidate_mask, dtype=bool).reshape(-1)
    context = np.asarray(state.critic_global_context, dtype=np.float32).reshape(-1)
    if candidates.ndim != 2 or candidates.shape[0] != mask.size:
        raise ValueError("decision candidate features/mask have inconsistent shapes")
    legal = candidates[mask]
    if legal.shape[0] == 0:
        raise ValueError("decision state has no legal candidates")
    legal_fraction = float(legal.shape[0]) / float(max(candidates.shape[0], 1))
    order_scale = max(float(config.CLEAN_NORM_ACTIVE_TASK_REF) - 1.0, 1.0)
    normalized_order = float(np.clip(float(state.decision_order) / order_scale, 0.0, 1.0))
    return np.concatenate(
        [
            legal.mean(axis=0),
            legal.max(axis=0),
            legal.std(axis=0, ddof=0),
            context,
            np.asarray([legal_fraction, normalized_order], dtype=np.float32),
        ]
    ).astype(np.float32)


@dataclass(frozen=True, slots=True)
class FrozenOffloadingDecisionCreditBatch:
    actor_advantages: dict[DecisionKey, float]
    critic_inputs: np.ndarray
    critic_targets: np.ndarray
    critic_old_predictions: np.ndarray
    transitions: tuple[CleanDecisionTransition, ...]
    diagnostics: dict[str, Any]


def compute_smdp_decision_gae(
    transitions: Iterable[CleanDecisionTransition],
    *,
    values: dict[DecisionKey, float],
    gamma: float,
    gae_lambda: float,
) -> tuple[dict[DecisionKey, float], dict[DecisionKey, float], dict[DecisionKey, float]]:
    """Compute frozen SMDP TD, primitive-slot GAE and value targets."""
    rows = list(transitions)
    by_key = {decision_state_key(row.state): row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("duplicate offloading decision transition key")
    td_by_key: dict[DecisionKey, float] = {}
    eligible: dict[DecisionKey, CleanDecisionTransition] = {}
    for row in rows:
        key = decision_state_key(row.state)
        if row.truncated or row.unresolved:
            continue
        if row.delta < 0 or not math.isfinite(float(row.rho)):
            raise ValueError("invalid SMDP transition rho/delta")
        current_value = float(values[key])
        if row.terminated:
            bootstrap = 0.0
        else:
            if row.next_state is None:
                raise ValueError("eligible nonterminal transition lacks next state")
            next_key = decision_state_key(row.next_state)
            if next_key not in values:
                raise ValueError("next decision value is missing")
            bootstrap = (float(gamma) ** int(row.delta)) * float(values[next_key])
        td_by_key[key] = float(row.rho) + bootstrap - current_value
        eligible[key] = row

    advantage_by_key: dict[DecisionKey, float] = {}
    visiting: set[DecisionKey] = set()

    def advantage(key: DecisionKey) -> float:
        if key in advantage_by_key:
            return advantage_by_key[key]
        if key in visiting:
            raise ValueError("decision transition chain contains a cycle")
        visiting.add(key)
        row = eligible[key]
        continuation = 0.0
        if not row.terminated and row.next_state is not None:
            next_key = decision_state_key(row.next_state)
            if next_key in eligible:
                continuation = (
                    (float(gamma) * float(gae_lambda)) ** int(row.delta)
                ) * advantage(next_key)
        result = float(td_by_key[key]) + continuation
        visiting.remove(key)
        advantage_by_key[key] = result
        return result

    for key in eligible:
        advantage(key)
    targets = {
        key: float(advantage_by_key[key]) + float(values[key])
        for key in eligible
    }
    return td_by_key, advantage_by_key, targets


class CleanOffloadingDecisionCredit:
    """Independent environment-return decision critic and frozen credit builder."""

    def __init__(
        self,
        *,
        critic: CleanDecisionCritic,
        optimizer: Any,
        gamma: float,
        gae_lambda: float,
        max_grad_norm: float,
        ppo_epochs: int,
        value_clip_epsilon: float,
        device: Any,
    ) -> None:
        self.critic = critic.to(device)
        self.optimizer = optimizer
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_epochs = max(int(ppo_epochs), 1)
        self.value_clip_epsilon = float(value_clip_epsilon)
        if not math.isfinite(self.value_clip_epsilon) or self.value_clip_epsilon < 0.0:
            raise ValueError("decision critic value_clip_epsilon must be finite and non-negative")
        self.device = device
        self.update_count = 0
        self.total_transition_count = 0
        self.total_eligible_count = 0
        self.total_unresolved_count = 0

    @classmethod
    def build_rng_neutral(
        cls,
        *,
        input_dim: int,
        hidden_dim: int,
        learning_rate: float,
        gamma: float,
        gae_lambda: float,
        max_grad_norm: float,
        device: Any,
        ppo_epochs: int = 1,
        value_clip_epsilon: float = 0.0,
    ) -> "CleanOffloadingDecisionCredit":
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            critic = CleanDecisionCritic(input_dim=int(input_dim), hidden_dim=int(hidden_dim))
            critic.to(device)
            optimizer = torch.optim.Adam(critic.parameters(), lr=float(learning_rate))
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        return cls(
            critic=critic,
            optimizer=optimizer,
            gamma=gamma,
            gae_lambda=gae_lambda,
            max_grad_norm=max_grad_norm,
            ppo_epochs=ppo_epochs,
            value_clip_epsilon=value_clip_epsilon,
            device=device,
        )

    def prepare_rollout(
        self, transitions: Iterable[CleanDecisionTransition]
    ) -> FrozenOffloadingDecisionCreditBatch:
        rows = list(transitions)
        all_states: dict[DecisionKey, CleanDecisionState] = {}
        for row in rows:
            all_states[decision_state_key(row.state)] = row.state
            if row.next_state is not None:
                all_states[decision_state_key(row.next_state)] = row.next_state
        encoded = {key: encode_decision_state(state) for key, state in all_states.items()}
        with torch.no_grad():
            values = {
                key: float(
                    self.critic(
                        torch.as_tensor(value, dtype=torch.float32, device=self.device)
                    ).reshape(-1)[0].cpu().item()
                )
                for key, value in encoded.items()
            }
        td, raw_advantages, targets = compute_smdp_decision_gae(
            rows,
            values=values,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        keys = list(raw_advantages)
        raw = np.asarray([raw_advantages[key] for key in keys], dtype=np.float32)
        if raw.size > 1:
            actor_values = (raw - raw.mean()) / max(float(raw.std(ddof=0)), 1e-8)
        else:
            actor_values = raw.copy()
        actor_advantages = {
            key: float(actor_values[index]) for index, key in enumerate(keys)
        }
        critic_inputs = (
            np.stack([encoded[key] for key in keys]).astype(np.float32)
            if keys
            else np.zeros((0, self.critic.net[0].in_features), dtype=np.float32)
        )
        critic_targets = np.asarray([targets[key] for key in keys], dtype=np.float32)
        critic_old_predictions = np.asarray([values[key] for key in keys], dtype=np.float32)
        deltas = np.asarray([row.delta for row in rows], dtype=np.float64)
        rhos = np.asarray([row.rho for row in rows], dtype=np.float64)
        same_slot = [
            row for row in rows
            if row.next_state is not None
            and row.state.episode_index == row.next_state.episode_index
            and row.state.lane_index == row.next_state.lane_index
            and row.state.slot_index == row.next_state.slot_index
        ]
        groups: dict[tuple[int, int, int], list[float]] = {}
        for key, value in raw_advantages.items():
            groups.setdefault(key[:3], []).append(float(value))
        within_slot = [float(np.std(values_, ddof=0)) for values_ in groups.values() if len(values_) > 1]
        unresolved_count = sum(bool(row.truncated or row.unresolved) for row in rows)
        diagnostics = {
            "decision_transition_count": int(len(rows)),
            "decision_eligible_count": int(len(keys)),
            "decision_unresolved_count": int(unresolved_count),
            "decision_same_slot_fraction": float(len(same_slot) / max(len(rows), 1)),
            "decision_delta_mean": float(deltas.mean()) if deltas.size else 0.0,
            "decision_delta_p50": float(np.percentile(deltas, 50)) if deltas.size else 0.0,
            "decision_delta_p90": float(np.percentile(deltas, 90)) if deltas.size else 0.0,
            "decision_delta_max": int(deltas.max()) if deltas.size else 0,
            "decision_rho_mean": float(rhos.mean()) if rhos.size else 0.0,
            "decision_rho_std": float(rhos.std(ddof=0)) if rhos.size else 0.0,
            "decision_td_residual_mean": float(np.mean(list(td.values()))) if td else 0.0,
            "decision_td_residual_std": float(np.std(list(td.values()), ddof=0)) if td else 0.0,
            "decision_advantage_mean": float(raw.mean()) if raw.size else 0.0,
            "decision_advantage_std": float(raw.std(ddof=0)) if raw.size else 0.0,
            "decision_within_slot_advantage_std": float(np.mean(within_slot)) if within_slot else 0.0,
            "decision_critic_raw_target_mean": float(critic_targets.mean()) if critic_targets.size else 0.0,
            "decision_critic_raw_target_std": float(critic_targets.std(ddof=0)) if critic_targets.size else 0.0,
        }
        self.total_transition_count += len(rows)
        self.total_eligible_count += len(keys)
        self.total_unresolved_count += unresolved_count
        return FrozenOffloadingDecisionCreditBatch(
            actor_advantages=actor_advantages,
            critic_inputs=critic_inputs,
            critic_targets=critic_targets,
            critic_old_predictions=critic_old_predictions,
            transitions=tuple(rows),
            diagnostics=diagnostics,
        )

    def train_frozen(self, batch: FrozenOffloadingDecisionCreditBatch) -> dict[str, Any]:
        diagnostics = dict(batch.diagnostics)
        if batch.critic_targets.size == 0:
            diagnostics.update(
                {
                    "decision_critic_loss": 0.0,
                    "decision_critic_ev": 0.0,
                    "decision_critic_grad_norm": 0.0,
                    "decision_critic_normalized_loss": 0.0,
                    "decision_critic_ev_pre_update": 0.0,
                    "decision_critic_ev_post_update": 0.0,
                    "decision_critic_preclip_grad_norm_mean": 0.0,
                    "decision_critic_preclip_grad_norm_max": 0.0,
                    "decision_critic_value_clip_fraction": 0.0,
                    "decision_critic_raw_target_mean": 0.0,
                    "decision_critic_raw_target_std": 0.0,
                    "decision_critic_target_normalization_mean": 0.0,
                    "decision_critic_target_normalization_scale": 1.0,
                }
            )
            return diagnostics
        inputs = torch.as_tensor(batch.critic_inputs, dtype=torch.float32, device=self.device)
        targets = torch.as_tensor(batch.critic_targets, dtype=torch.float32, device=self.device)
        old_predictions = torch.as_tensor(
            batch.critic_old_predictions, dtype=torch.float32, device=self.device
        )
        target_mean = targets.mean().detach()
        target_scale = targets.std(unbiased=False).detach().clamp_min(1e-8)

        def raw_explained_variance(predictions: Any) -> float:
            target_var = float(targets.var(unbiased=False).detach().cpu().item())
            if target_var <= 1e-12:
                return 0.0
            error_var = float(
                (targets - predictions).var(unbiased=False).detach().cpu().item()
            )
            return 1.0 - error_var / target_var

        ev_pre = raw_explained_variance(old_predictions)
        losses: list[float] = []
        grad_norms: list[float] = []
        clip_fractions: list[float] = []
        for _ in range(self.ppo_epochs):
            predictions = self.critic(inputs)
            per_sample_loss, was_clipped = normalized_clipped_value_loss(
                value=predictions,
                old_value=old_predictions,
                target=targets,
                target_mean=target_mean,
                target_scale=target_scale,
                clip_epsilon=self.value_clip_epsilon,
            )
            loss = per_sample_loss.mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.critic.parameters()), self.max_grad_norm
            )
            self.optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            grad_norms.append(float(grad_norm.detach().cpu().item()))
            clip_fractions.append(float(was_clipped.mean().detach().cpu().item()))
        self.update_count += 1
        with torch.no_grad():
            post_predictions = self.critic(inputs)
        ev_post = raw_explained_variance(post_predictions)
        normalized_loss = float(np.mean(losses))
        grad_norm_mean = float(np.mean(grad_norms))
        grad_norm_max = float(np.max(grad_norms))
        diagnostics.update(
            {
                "decision_critic_loss": normalized_loss,
                "decision_critic_ev": ev_post,
                "decision_critic_grad_norm": grad_norm_mean,
                "decision_critic_normalized_loss": normalized_loss,
                "decision_critic_ev_pre_update": ev_pre,
                "decision_critic_ev_post_update": ev_post,
                "decision_critic_preclip_grad_norm_mean": grad_norm_mean,
                "decision_critic_preclip_grad_norm_max": grad_norm_max,
                "decision_critic_value_clip_fraction": float(np.mean(clip_fractions)),
                "decision_critic_target_normalization_mean": float(target_mean.cpu().item()),
                "decision_critic_target_normalization_scale": float(target_scale.cpu().item()),
            }
        )
        return diagnostics

    def state_dict(self) -> dict[str, Any]:
        return {
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "max_grad_norm": self.max_grad_norm,
            "ppo_epochs": self.ppo_epochs,
            "value_clip_epsilon": self.value_clip_epsilon,
            "update_count": self.update_count,
            "total_transition_count": self.total_transition_count,
            "total_eligible_count": self.total_eligible_count,
            "total_unresolved_count": self.total_unresolved_count,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        for key, expected in (
            ("gamma", self.gamma),
            ("gae_lambda", self.gae_lambda),
            ("max_grad_norm", self.max_grad_norm),
            ("value_clip_epsilon", self.value_clip_epsilon),
        ):
            if not math.isclose(float(payload[key]), float(expected)):
                raise ValueError(f"offloading decision credit {key} mismatch")
        if int(payload["ppo_epochs"]) != self.ppo_epochs:
            raise ValueError("offloading decision credit ppo_epochs mismatch")
        self.critic.load_state_dict(payload["critic"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.update_count = int(payload["update_count"])
        self.total_transition_count = int(payload["total_transition_count"])
        self.total_eligible_count = int(payload["total_eligible_count"])
        self.total_unresolved_count = int(payload["total_unresolved_count"])
