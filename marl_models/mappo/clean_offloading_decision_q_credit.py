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
from marl_models.mappo.clean_offloading_decision_credit import (
    DecisionKey,
    decision_state_key,
)
from marl_models.mappo.clean_ppo import (
    CleanDecisionCritic,
    normalized_clipped_value_loss,
)


def encode_decision_candidate_rows(
    state: CleanDecisionState,
) -> tuple[np.ndarray, np.ndarray]:
    """Return original legal row indices and action-conditioned Q inputs."""
    candidates = np.asarray(state.candidate_features, dtype=np.float32)
    mask = np.asarray(state.candidate_mask, dtype=bool).reshape(-1)
    context = np.asarray(state.critic_global_context, dtype=np.float32).reshape(-1)
    if candidates.ndim != 2 or candidates.shape[0] != mask.size:
        raise ValueError("decision candidate features/mask have inconsistent shapes")
    legal_indices = np.flatnonzero(mask).astype(np.int64)
    if legal_indices.size == 0:
        raise ValueError("decision state has no legal candidates")
    legal_fraction = float(legal_indices.size) / float(max(candidates.shape[0], 1))
    order_scale = max(float(config.CLEAN_NORM_ACTIVE_TASK_REF) - 1.0, 1.0)
    normalized_order = float(np.clip(float(state.decision_order) / order_scale, 0.0, 1.0))
    suffix = np.concatenate(
        [context, np.asarray([legal_fraction, normalized_order], dtype=np.float32)]
    )
    rows = np.concatenate(
        [
            candidates[legal_indices],
            np.repeat(suffix.reshape(1, -1), legal_indices.size, axis=0),
        ],
        axis=1,
    ).astype(np.float32)
    return legal_indices, rows


def expected_behavior_q(
    *, legal_indices: np.ndarray, probabilities: np.ndarray, q_values: np.ndarray
) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    q_values = np.asarray(q_values, dtype=np.float64).reshape(-1)
    legal_probabilities = probabilities[np.asarray(legal_indices, dtype=np.int64)]
    if legal_probabilities.shape != q_values.shape:
        raise ValueError("behavior probabilities and legal Q values do not align")
    probability_sum = float(legal_probabilities.sum())
    if not math.isfinite(probability_sum) or probability_sum <= 0.0:
        raise ValueError("legal behavior probabilities have invalid mass")
    legal_probabilities = legal_probabilities / probability_sum
    return float(np.dot(legal_probabilities, q_values))


def selected_legal_row(state: CleanDecisionState, legal_indices: np.ndarray) -> int:
    matches = np.flatnonzero(
        np.asarray(legal_indices, dtype=np.int64) == int(state.selected_action)
    )
    if matches.size != 1:
        raise ValueError("selected action does not identify exactly one legal Q row")
    return int(matches[0])


@dataclass(frozen=True, slots=True)
class FrozenOffloadingDecisionQBatch:
    actor_advantages: dict[DecisionKey, float]
    critic_inputs: np.ndarray
    critic_targets: np.ndarray
    critic_old_predictions: np.ndarray
    transitions: tuple[CleanDecisionTransition, ...]
    diagnostics: dict[str, Any]


class CleanOffloadingDecisionQCredit:
    """Environment-return action-conditioned Q critic and frozen PPO credit."""

    def __init__(
        self,
        *,
        critic: CleanDecisionCritic,
        optimizer: Any,
        gamma: float,
        max_grad_norm: float,
        ppo_epochs: int,
        value_clip_epsilon: float,
        device: Any,
    ) -> None:
        self.critic = critic.to(device)
        self.optimizer = optimizer
        self.gamma = float(gamma)
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_epochs = max(int(ppo_epochs), 1)
        self.value_clip_epsilon = float(value_clip_epsilon)
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
        max_grad_norm: float,
        ppo_epochs: int,
        value_clip_epsilon: float,
        device: Any,
    ) -> "CleanOffloadingDecisionQCredit":
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
            max_grad_norm=max_grad_norm,
            ppo_epochs=ppo_epochs,
            value_clip_epsilon=value_clip_epsilon,
            device=device,
        )

    def _old_state_values(
        self, states: dict[DecisionKey, CleanDecisionState]
    ) -> dict[DecisionKey, tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        result = {}
        with torch.no_grad():
            for key, state in states.items():
                legal_indices, inputs = encode_decision_candidate_rows(state)
                q_values = (
                    self.critic(torch.as_tensor(inputs, dtype=torch.float32, device=self.device))
                    .reshape(-1)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                expected = expected_behavior_q(
                    legal_indices=legal_indices,
                    probabilities=state.old_masked_probabilities,
                    q_values=q_values,
                )
                result[key] = (legal_indices, inputs, q_values, expected)
        return result

    def prepare_rollout(
        self, transitions: Iterable[CleanDecisionTransition]
    ) -> FrozenOffloadingDecisionQBatch:
        rows = list(transitions)
        states: dict[DecisionKey, CleanDecisionState] = {}
        for row in rows:
            states[decision_state_key(row.state)] = row.state
            if row.next_state is not None:
                states[decision_state_key(row.next_state)] = row.next_state
        old = self._old_state_values(states)
        raw_advantages: dict[DecisionKey, float] = {}
        targets: dict[DecisionKey, float] = {}
        selected_inputs: dict[DecisionKey, np.ndarray] = {}
        selected_predictions: dict[DecisionKey, float] = {}
        spreads: list[float] = []
        eligible_rows: list[CleanDecisionTransition] = []
        for row in rows:
            if row.truncated or row.unresolved:
                continue
            key = decision_state_key(row.state)
            legal_indices, inputs, q_values, expected_current = old[key]
            selected_row = selected_legal_row(row.state, legal_indices)
            q_selected = float(q_values[selected_row])
            if row.terminated:
                bootstrap = 0.0
            else:
                if row.next_state is None:
                    raise ValueError("eligible nonterminal Q transition lacks next state")
                bootstrap = (self.gamma ** int(row.delta)) * old[
                    decision_state_key(row.next_state)
                ][3]
            target = float(row.rho) + float(bootstrap)
            raw_advantages[key] = q_selected - float(expected_current)
            targets[key] = target
            selected_inputs[key] = inputs[selected_row]
            selected_predictions[key] = q_selected
            spreads.append(float(np.max(q_values) - np.min(q_values)))
            eligible_rows.append(row)

        keys = list(raw_advantages)
        raw = np.asarray([raw_advantages[key] for key in keys], dtype=np.float32)
        input_dim = int(self.critic.net[0].in_features)
        critic_inputs = (
            np.stack([selected_inputs[key] for key in keys]).astype(np.float32)
            if keys else np.zeros((0, input_dim), dtype=np.float32)
        )
        critic_targets = np.asarray([targets[key] for key in keys], dtype=np.float32)
        target_std = float(critic_targets.std(ddof=0)) if keys else 0.0
        actor_advantage_scale = max(target_std, 1e-8)
        scaled_actor_advantages = raw / actor_advantage_scale
        actor_advantages = {
            key: float(scaled_actor_advantages[index])
            for index, key in enumerate(keys)
        }
        old_predictions = np.asarray(
            [selected_predictions[key] for key in keys], dtype=np.float32
        )
        td_errors = critic_targets - old_predictions
        groups: dict[tuple[int, int, int], list[float]] = {}
        for key, value in raw_advantages.items():
            groups.setdefault(key[:3], []).append(float(value))
        within_slot = [
            float(np.std(values, ddof=0))
            for values in groups.values() if len(values) > 1
        ]
        unresolved_count = sum(bool(row.truncated or row.unresolved) for row in rows)
        diagnostics = {
            "decision_q_transition_count": len(rows),
            "decision_q_eligible_count": len(keys),
            "decision_q_unresolved_count": unresolved_count,
            "decision_q_unresolved_fraction": unresolved_count / max(len(rows), 1),
            "decision_q_target_mean": float(critic_targets.mean()) if keys else 0.0,
            "decision_q_target_std": target_std,
            "decision_q_td_error_mean": float(td_errors.mean()) if keys else 0.0,
            "decision_q_td_error_std": float(td_errors.std(ddof=0)) if keys else 0.0,
            "decision_q_legal_action_spread_mean": float(np.mean(spreads)) if spreads else 0.0,
            "decision_q_legal_action_spread_std": float(np.std(spreads, ddof=0)) if spreads else 0.0,
            "decision_q_advantage_mean": float(raw.mean()) if keys else 0.0,
            "decision_q_advantage_std": float(raw.std(ddof=0)) if keys else 0.0,
            "decision_q_actor_advantage_scale": actor_advantage_scale,
            "decision_q_scaled_actor_advantage_std": (
                float(scaled_actor_advantages.std(ddof=0)) if keys else 0.0
            ),
            "decision_q_within_slot_advantage_std": float(np.mean(within_slot)) if within_slot else 0.0,
        }
        self.total_transition_count += len(rows)
        self.total_eligible_count += len(keys)
        self.total_unresolved_count += unresolved_count
        return FrozenOffloadingDecisionQBatch(
            actor_advantages=actor_advantages,
            critic_inputs=critic_inputs,
            critic_targets=critic_targets,
            critic_old_predictions=old_predictions,
            transitions=tuple(rows),
            diagnostics=diagnostics,
        )

    def train_frozen(self, batch: FrozenOffloadingDecisionQBatch) -> dict[str, Any]:
        diagnostics = dict(batch.diagnostics)
        if batch.critic_targets.size == 0:
            diagnostics.update({
                "decision_q_normalized_loss": 0.0,
                "decision_q_ev_pre_update": 0.0,
                "decision_q_ev_post_update": 0.0,
                "decision_q_preclip_grad_norm_mean": 0.0,
                "decision_q_preclip_grad_norm_max": 0.0,
                "decision_q_value_clip_fraction": 0.0,
            })
            return diagnostics
        inputs = torch.as_tensor(batch.critic_inputs, dtype=torch.float32, device=self.device)
        targets = torch.as_tensor(batch.critic_targets, dtype=torch.float32, device=self.device)
        old_predictions = torch.as_tensor(
            batch.critic_old_predictions, dtype=torch.float32, device=self.device
        )
        target_mean = targets.mean().detach()
        target_scale = targets.std(unbiased=False).detach().clamp_min(1e-8)

        def explained_variance(predictions: Any) -> float:
            target_var = float(targets.var(unbiased=False).detach().cpu().item())
            if target_var <= 1e-12:
                return 0.0
            error_var = float((targets - predictions).var(unbiased=False).detach().cpu().item())
            return 1.0 - error_var / target_var

        ev_pre = explained_variance(old_predictions)
        losses, grad_norms, clip_fractions = [], [], []
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
        diagnostics.update({
            "decision_q_loss": float(np.mean(losses)),
            "decision_q_normalized_loss": float(np.mean(losses)),
            "decision_q_ev_pre_update": ev_pre,
            "decision_q_ev_post_update": explained_variance(post_predictions),
            "decision_q_preclip_grad_norm_mean": float(np.mean(grad_norms)),
            "decision_q_preclip_grad_norm_max": float(np.max(grad_norms)),
            "decision_q_value_clip_fraction": float(np.mean(clip_fractions)),
            "decision_q_target_normalization_mean": float(target_mean.cpu().item()),
            "decision_q_target_normalization_scale": float(target_scale.cpu().item()),
        })
        return diagnostics

    def state_dict(self) -> dict[str, Any]:
        return {
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "gamma": self.gamma,
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
            ("max_grad_norm", self.max_grad_norm),
            ("value_clip_epsilon", self.value_clip_epsilon),
        ):
            if not math.isclose(float(payload[key]), float(expected)):
                raise ValueError(f"offloading decision Q credit {key} mismatch")
        if int(payload["ppo_epochs"]) != self.ppo_epochs:
            raise ValueError("offloading decision Q credit ppo_epochs mismatch")
        self.critic.load_state_dict(payload["critic"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.update_count = int(payload["update_count"])
        self.total_transition_count = int(payload["total_transition_count"])
        self.total_eligible_count = int(payload["total_eligible_count"])
        self.total_unresolved_count = int(payload["total_unresolved_count"])
