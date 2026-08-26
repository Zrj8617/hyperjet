from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np
import torch

from marl_models.mappo.clean_decision_transitions import (
    CleanDecisionState,
    CleanDecisionTransition,
)
from marl_models.mappo.clean_decision_td_dataset import CleanDecisionTDRawCapture
from marl_models.mappo.clean_offloading_action_value import (
    build_rng_neutral_clean_counterfactual_q,
)


PHASE4_TARGET_SYNC_MODE = "hard"
PHASE4_TARGET_SYNC_INTERVAL = 10
SlotKey = tuple[int, int, int]


@dataclass(slots=True)
class _SlotAdvantageLabel:
    value: float
    expected_decision_count: int
    matched_decision_count: int = 0


@dataclass(frozen=True, slots=True)
class _DiagnosticSample:
    transition: CleanDecisionTransition
    phase3b_slot_advantage_target: float


def decision_state_key(state: CleanDecisionState) -> SlotKey:
    return (int(state.episode_index), int(state.lane_index), int(state.slot_index))


def decision_q_input(state: CleanDecisionState) -> np.ndarray:
    candidate_features = np.asarray(state.candidate_features, dtype=np.float32)
    global_context = np.asarray(state.critic_global_context, dtype=np.float32).reshape(-1)
    if candidate_features.ndim != 2:
        raise ValueError("decision candidate features must be 2D")
    context = np.broadcast_to(global_context, (candidate_features.shape[0], global_context.size))
    combined = np.concatenate([candidate_features, context], axis=1).astype(
        np.float32, copy=False
    )
    if not bool(np.isfinite(combined).all()):
        raise FloatingPointError("decision Q input contains non-finite values")
    return combined


def validated_behavior_probabilities(state: CleanDecisionState) -> np.ndarray:
    mask = np.asarray(state.candidate_mask, dtype=bool).reshape(-1)
    probabilities = np.asarray(state.old_masked_probabilities, dtype=np.float32).reshape(-1)
    candidate_count = int(np.asarray(state.candidate_features).shape[0])
    if len(state.candidate_uav_ids) != candidate_count:
        raise ValueError("candidate UAV ordering does not match candidate features")
    if mask.size != candidate_count or probabilities.size != candidate_count:
        raise ValueError("behavior probabilities/mask do not match candidate count")
    if not bool(mask.any()) or not bool(np.isfinite(probabilities).all()):
        raise ValueError("behavior policy probabilities are invalid")
    if bool((probabilities < -1e-7).any()):
        raise ValueError("behavior policy probabilities must be non-negative")
    if bool((np.abs(probabilities[~mask]) > 1e-6).any()):
        raise ValueError("illegal candidates must have zero behavior probability")
    legal_sum = float(probabilities[mask].sum())
    if not math.isclose(legal_sum, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError("legal behavior probabilities must sum to one")
    return probabilities


def expected_sarsa_target(
    transition: CleanDecisionTransition,
    *,
    target_q: Any,
    gamma: float,
    device: Any,
) -> torch.Tensor:
    target, _ = expected_sarsa_target_components(
        transition,
        target_q=target_q,
        gamma=gamma,
        device=device,
    )
    return target


def expected_sarsa_target_components(
    transition: CleanDecisionTransition,
    *,
    target_q: Any,
    gamma: float,
    device: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    rho = torch.tensor(float(transition.rho), dtype=torch.float32, device=device)
    if bool(transition.terminated):
        return rho, rho.new_zeros(())
    if bool(transition.truncated or transition.unresolved):
        raise ValueError("unresolved transition is not an eligible TD target")
    if transition.next_state is None:
        raise ValueError("non-terminal TD transition is missing next state")
    probabilities_np = validated_behavior_probabilities(transition.next_state)
    next_input = torch.as_tensor(
        decision_q_input(transition.next_state), dtype=torch.float32, device=device
    )
    probabilities = torch.as_tensor(
        probabilities_np, dtype=torch.float32, device=device
    )
    with torch.no_grad():
        next_values = target_q(next_input)
        expected_next = torch.sum(probabilities * next_values)
        bootstrap = (float(gamma) ** int(transition.delta)) * expected_next
        target = rho + bootstrap
    return target.detach(), bootstrap.detach()


class CleanDecisionTDDiagnostic:
    """Shadow decision-TD learner isolated from PPO and actor optimization."""

    def __init__(
        self,
        *,
        online_q: Any,
        target_q: Any,
        optimizer: Any,
        gamma: float,
        max_grad_norm: float,
        device: Any,
        raw_capture: CleanDecisionTDRawCapture | None = None,
    ) -> None:
        self.online_q = online_q.to(device)
        self.target_q = target_q.to(device)
        self.target_q.load_state_dict(self.online_q.state_dict())
        self.target_q.requires_grad_(False)
        self.target_q.eval()
        self.optimizer = optimizer
        self.gamma = float(gamma)
        self.max_grad_norm = float(max_grad_norm)
        self.device = device
        self.raw_capture = raw_capture
        self.target_sync_mode = PHASE4_TARGET_SYNC_MODE
        self.target_sync_interval = PHASE4_TARGET_SYNC_INTERVAL
        self.shadow_update_count = 0
        self.target_sync_count = 0
        self.consumed_transition_count = 0
        self.training_eligible_transition_count = 0
        self.terminal_transition_count = 0
        self.unresolved_transition_count = 0
        self._slot_labels: dict[SlotKey, _SlotAdvantageLabel] = {}
        self._waiting_transitions: dict[SlotKey, list[CleanDecisionTransition]] = {}
        self._ready_samples: list[_DiagnosticSample] = []

    @property
    def pending_transition_count(self) -> int:
        return sum(len(rows) for rows in self._waiting_transitions.values())

    def ingest_transitions(self, transitions: Iterable[CleanDecisionTransition]) -> None:
        for transition in transitions:
            self.consumed_transition_count += 1
            if bool(transition.terminated):
                self.terminal_transition_count += 1
            if bool(transition.truncated or transition.unresolved):
                self.unresolved_transition_count += 1
            key = decision_state_key(transition.state)
            label = self._slot_labels.get(key)
            if label is None:
                self._waiting_transitions.setdefault(key, []).append(transition)
            else:
                self._match_transition(transition, label)

    def record_slot_advantages(
        self,
        *,
        slot_keys: list[SlotKey],
        normalized_advantages: np.ndarray,
        decision_counts: list[int],
    ) -> None:
        values = np.asarray(normalized_advantages, dtype=np.float32).reshape(-1)
        if len(slot_keys) != values.size or len(slot_keys) != len(decision_counts):
            raise ValueError("slot advantage observer inputs have inconsistent lengths")
        for key, value, decision_count in zip(slot_keys, values, decision_counts):
            count = int(decision_count)
            if count <= 0:
                continue
            if key in self._slot_labels:
                raise ValueError(f"duplicate Phase3-B slot advantage label: {key}")
            label = _SlotAdvantageLabel(
                value=float(value),
                expected_decision_count=count,
            )
            self._slot_labels[key] = label
            for transition in self._waiting_transitions.pop(key, []):
                self._match_transition(transition, label)
            self._drop_complete_label(key, label)

    def train_ready(self, *, update_step: int | None = None) -> dict[str, Any]:
        samples = list(self._ready_samples)
        self._ready_samples.clear()
        diagnostics = self._empty_diagnostics()
        if not samples:
            return diagnostics

        selected_predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        legal_spreads: list[float] = []
        phase3b_targets: list[float] = []
        selected_q_inputs: list[np.ndarray] = []
        bootstrap_values: list[torch.Tensor] = []
        transitions: list[CleanDecisionTransition] = []
        for sample in samples:
            transition = sample.transition
            validated_behavior_probabilities(transition.state)
            current_input_np = decision_q_input(transition.state)
            current_input = torch.as_tensor(
                current_input_np,
                dtype=torch.float32,
                device=self.device,
            )
            current_values = self.online_q(current_input)
            selected_action = int(transition.state.selected_action)
            mask = np.asarray(transition.state.candidate_mask, dtype=bool).reshape(-1)
            if selected_action < 0 or selected_action >= mask.size or not bool(mask[selected_action]):
                raise ValueError("selected action does not identify a legal candidate")
            legal_values = current_values[torch.as_tensor(mask, dtype=torch.bool, device=self.device)]
            selected_predictions.append(current_values[selected_action])
            selected_q_inputs.append(
                np.asarray(current_input_np[selected_action], dtype=np.float32).copy()
            )
            legal_spreads.append(float((legal_values.max() - legal_values.min()).detach().cpu().item()))
            target, bootstrap = expected_sarsa_target_components(
                transition,
                target_q=self.target_q,
                gamma=self.gamma,
                device=self.device,
            )
            targets.append(target)
            bootstrap_values.append(bootstrap)
            phase3b_targets.append(float(sample.phase3b_slot_advantage_target))
            transitions.append(transition)

        prediction_tensor = torch.stack(selected_predictions)
        target_tensor = torch.stack(targets).detach()
        if self.raw_capture is not None:
            if update_step is None:
                raise ValueError("raw diagnostic capture requires the current update step")
            self.raw_capture.write_batch(
                update_step=int(update_step),
                transitions=transitions,
                selected_q_inputs=np.stack(selected_q_inputs, axis=0),
                bootstrap_values=torch.stack(bootstrap_values).detach().cpu().numpy(),
                td_targets=target_tensor.cpu().numpy(),
                phase3b_targets=np.asarray(phase3b_targets, dtype=np.float32),
                online_selected_q_predictions=prediction_tensor.detach().cpu().numpy(),
            )
        loss = 0.5 * (prediction_tensor - target_tensor).pow(2).mean()
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("Phase4 shadow Q loss is non-finite")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            list(self.online_q.parameters()), self.max_grad_norm
        )
        self.optimizer.step()
        self.shadow_update_count += 1
        target_synced = False
        if self.shadow_update_count % self.target_sync_interval == 0:
            self.target_q.load_state_dict(self.online_q.state_dict())
            self.target_sync_count += 1
            target_synced = True

        diagnostics.update(
            self._batch_diagnostics(
                transitions=transitions,
                predictions=prediction_tensor.detach().cpu().numpy(),
                targets=target_tensor.cpu().numpy(),
                phase3b_targets=np.asarray(phase3b_targets, dtype=np.float32),
                legal_spreads=np.asarray(legal_spreads, dtype=np.float32),
            )
        )
        if self.raw_capture is not None:
            diagnostics.update(self.raw_capture.summary())
        diagnostics.update(
            {
                "phase4_shadow_q_loss": float(loss.detach().cpu().item()),
                "phase4_shadow_q_gradient_norm": float(gradient_norm),
                "phase4_shadow_optimizer_step": True,
                "phase4_target_synced": bool(target_synced),
                "phase4_shadow_update_count": int(self.shadow_update_count),
                "phase4_target_sync_count": int(self.target_sync_count),
            }
        )
        return diagnostics

    def state_dict(self) -> dict[str, Any]:
        return {
            "target_sync_mode": self.target_sync_mode,
            "target_sync_interval": int(self.target_sync_interval),
            "shadow_update_count": int(self.shadow_update_count),
            "target_sync_count": int(self.target_sync_count),
            "online_q": self.online_q.state_dict(),
            "target_q": self.target_q.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if str(payload.get("target_sync_mode")) != self.target_sync_mode:
            raise ValueError("Phase4 target sync mode mismatch")
        if int(payload.get("target_sync_interval", -1)) != self.target_sync_interval:
            raise ValueError("Phase4 target sync interval mismatch")
        self.online_q.load_state_dict(payload["online_q"])
        self.target_q.load_state_dict(payload["target_q"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.shadow_update_count = int(payload["shadow_update_count"])
        self.target_sync_count = int(payload["target_sync_count"])

    def _match_transition(
        self,
        transition: CleanDecisionTransition,
        label: _SlotAdvantageLabel,
    ) -> None:
        label.matched_decision_count += 1
        if label.matched_decision_count > label.expected_decision_count:
            raise ValueError("more transitions matched a slot than recorded decisions")
        if not bool(transition.truncated or transition.unresolved):
            if not transition.terminated and transition.next_state is None:
                raise ValueError("eligible non-terminal transition has no next state")
            self._ready_samples.append(
                _DiagnosticSample(
                    transition=transition,
                    phase3b_slot_advantage_target=float(label.value),
                )
            )
            self.training_eligible_transition_count += 1
        self._drop_complete_label(decision_state_key(transition.state), label)

    def _drop_complete_label(self, key: SlotKey, label: _SlotAdvantageLabel) -> None:
        if label.matched_decision_count == label.expected_decision_count:
            self._slot_labels.pop(key, None)

    def _empty_diagnostics(self) -> dict[str, Any]:
        coverage = (
            float(self.training_eligible_transition_count)
            / float(self.consumed_transition_count)
            if self.consumed_transition_count > 0
            else 0.0
        )
        return {
            "phase4_consumed_transition_count": int(self.consumed_transition_count),
            "phase4_training_eligible_transitions": int(self.training_eligible_transition_count),
            "phase4_terminal_transitions": int(self.terminal_transition_count),
            "phase4_truncated_unresolved_transitions": int(self.unresolved_transition_count),
            "phase4_pending_transition_count": int(self.pending_transition_count),
            "phase4_coverage_fraction": float(coverage),
            "phase4_shadow_q_loss": 0.0,
            "phase4_shadow_q_gradient_norm": 0.0,
            "phase4_shadow_optimizer_step": False,
            "phase4_target_synced": False,
            "phase4_shadow_update_count": int(self.shadow_update_count),
            "phase4_target_sync_count": int(self.target_sync_count),
            "phase4_target_sync_mode": self.target_sync_mode,
            "phase4_target_sync_interval": int(self.target_sync_interval),
            "phase4_scheme_b_oracle_correlation": None,
            "phase4_same_slot_transition_fraction": 0.0,
            "phase4_same_slot_global_context_identical_fraction": 0.0,
            "phase4_same_slot_candidate_feature_change_fraction": 0.0,
            "phase4_delta_mean": 0.0,
            "phase4_delta_p50": 0.0,
            "phase4_delta_p90": 0.0,
            "phase4_delta_max": 0,
            "phase4_rho_mean": 0.0,
            "phase4_rho_std": 0.0,
            "phase4_rho_min": 0.0,
            "phase4_rho_max": 0.0,
            "phase4_td_target_mean": 0.0,
            "phase4_td_target_std": 0.0,
            "phase4_td_target_min": 0.0,
            "phase4_td_target_max": 0.0,
            "phase4_td_target_nonzero_fraction": 0.0,
            "phase4_td_target_finite_fraction": 0.0,
            "phase4_online_q_selected_mean": 0.0,
            "phase4_online_q_selected_std": 0.0,
            "phase4_q_explained_variance": 0.0,
            "phase4_td_error_mean": 0.0,
            "phase4_td_error_std": 0.0,
            "phase4_td_error_abs_mean": 0.0,
            "phase4_legal_q_spread_mean": 0.0,
            "phase4_legal_q_spread_median": 0.0,
            "phase4_legal_q_spread_p90": 0.0,
            "phase4_multi_candidate_fraction": 0.0,
            "phase4_phase3b_target_mean": 0.0,
            "phase4_phase3b_target_std": 0.0,
            "phase4_target_mean": 0.0,
            "phase4_target_std": 0.0,
            "phase4_phase3b_td_pearson": None,
            "phase4_phase3b_td_spearman": None,
            "phase4_phase3b_td_sign_agreement": 0.0,
            "phase4_within_slot_phase3b_target_std_mean": 0.0,
            "phase4_within_slot_phase4_target_std_mean": 0.0,
            "phase4_within_slot_target_separation_fraction": 0.0,
        }

    def _batch_diagnostics(
        self,
        *,
        transitions: list[CleanDecisionTransition],
        predictions: np.ndarray,
        targets: np.ndarray,
        phase3b_targets: np.ndarray,
        legal_spreads: np.ndarray,
    ) -> dict[str, Any]:
        predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
        targets = np.asarray(targets, dtype=np.float64).reshape(-1)
        phase3b_targets = np.asarray(phase3b_targets, dtype=np.float64).reshape(-1)
        errors = targets - predictions
        target_variance = float(np.var(targets))
        q_ev = (
            1.0 - float(np.var(errors)) / target_variance
            if target_variance > 1e-12
            else 0.0
        )
        deltas = np.asarray([row.delta for row in transitions], dtype=np.float64)
        rhos = np.asarray([row.rho for row in transitions], dtype=np.float64)
        same_slot = [
            row for row in transitions
            if row.next_state is not None
            and decision_state_key(row.state) == decision_state_key(row.next_state)
        ]
        global_equal = [
            np.array_equal(row.state.critic_global_context, row.next_state.critic_global_context)
            for row in same_slot
        ]
        candidate_changed = [
            not np.array_equal(row.state.candidate_features, row.next_state.candidate_features)
            for row in same_slot
        ]
        groups: dict[SlotKey, list[int]] = {}
        for index, row in enumerate(transitions):
            groups.setdefault(decision_state_key(row.state), []).append(index)
        multi_groups = [indices for indices in groups.values() if len(indices) >= 2]
        phase3b_group_stds = [float(np.std(phase3b_targets[indices])) for indices in multi_groups]
        phase4_group_stds = [float(np.std(targets[indices])) for indices in multi_groups]
        separation = [value > 1e-8 for value in phase4_group_stds]
        multi_candidate = [
            int(np.asarray(row.state.candidate_mask, dtype=bool).sum()) > 1
            for row in transitions
        ]
        return {
            "phase4_same_slot_transition_fraction": _fraction(len(same_slot), len(transitions)),
            "phase4_same_slot_global_context_identical_fraction": _mean_bool(global_equal),
            "phase4_same_slot_candidate_feature_change_fraction": _mean_bool(candidate_changed),
            "phase4_delta_mean": float(np.mean(deltas)),
            "phase4_delta_p50": float(np.percentile(deltas, 50)),
            "phase4_delta_p90": float(np.percentile(deltas, 90)),
            "phase4_delta_max": int(np.max(deltas)),
            "phase4_rho_mean": float(np.mean(rhos)),
            "phase4_rho_std": float(np.std(rhos)),
            "phase4_rho_min": float(np.min(rhos)),
            "phase4_rho_max": float(np.max(rhos)),
            "phase4_td_target_mean": float(np.mean(targets)),
            "phase4_td_target_std": float(np.std(targets)),
            "phase4_td_target_min": float(np.min(targets)),
            "phase4_td_target_max": float(np.max(targets)),
            "phase4_td_target_nonzero_fraction": float(np.mean(np.abs(targets) > 1e-12)),
            "phase4_td_target_finite_fraction": float(np.mean(np.isfinite(targets))),
            "phase4_online_q_selected_mean": float(np.mean(predictions)),
            "phase4_online_q_selected_std": float(np.std(predictions)),
            "phase4_q_explained_variance": float(q_ev),
            "phase4_td_error_mean": float(np.mean(errors)),
            "phase4_td_error_std": float(np.std(errors)),
            "phase4_td_error_abs_mean": float(np.mean(np.abs(errors))),
            "phase4_legal_q_spread_mean": float(np.mean(legal_spreads)),
            "phase4_legal_q_spread_median": float(np.median(legal_spreads)),
            "phase4_legal_q_spread_p90": float(np.percentile(legal_spreads, 90)),
            "phase4_multi_candidate_fraction": _mean_bool(multi_candidate),
            "phase4_phase3b_target_mean": float(np.mean(phase3b_targets)),
            "phase4_phase3b_target_std": float(np.std(phase3b_targets)),
            "phase4_target_mean": float(np.mean(targets)),
            "phase4_target_std": float(np.std(targets)),
            "phase4_phase3b_td_pearson": _correlation(phase3b_targets, targets),
            "phase4_phase3b_td_spearman": _correlation(
                _rank_values(phase3b_targets), _rank_values(targets)
            ),
            "phase4_phase3b_td_sign_agreement": float(
                np.mean(np.sign(phase3b_targets) == np.sign(targets))
            ),
            "phase4_within_slot_phase3b_target_std_mean": _mean_or_zero(phase3b_group_stds),
            "phase4_within_slot_phase4_target_std_mean": _mean_or_zero(phase4_group_stds),
            "phase4_within_slot_target_separation_fraction": _mean_bool(separation),
        }


def build_clean_decision_td_diagnostic(
    *,
    input_dim: int,
    hidden_dim: int,
    learning_rate: float,
    gamma: float,
    max_grad_norm: float,
    device: Any,
    raw_capture: CleanDecisionTDRawCapture | None = None,
) -> CleanDecisionTDDiagnostic:
    online_q = build_rng_neutral_clean_counterfactual_q(
        input_dim=int(input_dim), hidden_dim=int(hidden_dim)
    )
    target_q = build_rng_neutral_clean_counterfactual_q(
        input_dim=int(input_dim), hidden_dim=int(hidden_dim)
    )
    optimizer = torch.optim.Adam(online_q.parameters(), lr=float(learning_rate))
    return CleanDecisionTDDiagnostic(
        online_q=online_q,
        target_q=target_q,
        optimizer=optimizer,
        gamma=float(gamma),
        max_grad_norm=float(max_grad_norm),
        device=device,
        raw_capture=raw_capture,
    )


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _mean_bool(values: Iterable[bool]) -> float:
    rows = list(values)
    return float(np.mean(rows)) if rows else 0.0


def _mean_or_zero(values: Iterable[float]) -> float:
    rows = list(values)
    return float(np.mean(rows)) if rows else 0.0


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _rank_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks
