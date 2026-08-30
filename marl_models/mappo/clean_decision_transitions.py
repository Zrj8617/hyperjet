from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np


def _frozen_array(value: Any, *, dtype: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class CleanDecisionState:
    """Minimal rollout-time state needed by a future decision-value estimator."""

    episode_index: int
    lane_index: int
    slot_index: int
    task_id: str
    task_local_index: int
    dag_id: str
    decision_order: int
    selected_action: int
    selected_uav_id: int
    candidate_uav_ids: tuple[int, ...]
    candidate_mask: np.ndarray
    candidate_features: np.ndarray
    critic_global_context: np.ndarray
    old_masked_probabilities: np.ndarray
    old_log_probability: float

    @classmethod
    def from_rollout_record(
        cls,
        record: Any,
        *,
        episode_index: int,
        lane_index: int,
        slot_index: int,
    ) -> "CleanDecisionState":
        if record.dag_id is None:
            raise ValueError("decision transition record is missing dag_id")
        if record.candidate_features is None or record.critic_global_context is None:
            raise ValueError("decision transition record is missing candidate/global features")
        if record.old_masked_probabilities is None:
            raise ValueError("decision transition record is missing rollout behavior probabilities")
        return cls(
            episode_index=int(episode_index),
            lane_index=int(lane_index),
            slot_index=int(slot_index),
            task_id=str(record.task_id),
            task_local_index=int(record.task_local_index),
            dag_id=str(record.dag_id),
            decision_order=int(record.decision_order),
            selected_action=int(record.selected_action),
            selected_uav_id=int(record.selected_uav_id),
            candidate_uav_ids=tuple(int(value) for value in record.candidate_uav_ids),
            candidate_mask=_frozen_array(record.candidate_mask, dtype=bool),
            candidate_features=_frozen_array(record.candidate_features, dtype=np.float32),
            critic_global_context=_frozen_array(
                record.critic_global_context, dtype=np.float32
            ).reshape(-1),
            old_masked_probabilities=_frozen_array(
                record.old_masked_probabilities, dtype=np.float32
            ).reshape(-1),
            old_log_probability=float(record.old_log_probability),
        )


@dataclass(frozen=True, slots=True)
class CleanDecisionTransition:
    state: CleanDecisionState
    rho: float
    delta: int
    next_state: CleanDecisionState | None
    terminated: bool
    truncated: bool
    unresolved: bool
    future_bootstrap: float | None


@dataclass(slots=True)
class _PendingDecisionTransition:
    state: CleanDecisionState
    rho: float = 0.0
    delta: int = 0


class CleanDecisionTransitionTracker:
    """Environment-local decision stream, independent of PPO rollout buffers."""

    def __init__(self, *, gamma: float, lane_index: int = 0) -> None:
        if not math.isfinite(float(gamma)) or not 0.0 <= float(gamma) <= 1.0:
            raise ValueError("decision transition gamma must be finite and in [0, 1]")
        self.gamma = float(gamma)
        self.lane_index = int(lane_index)
        self._episode_index: int | None = None
        self._pending: _PendingDecisionTransition | None = None
        self._completed: list[CleanDecisionTransition] = []
        self.registered_decision_count = 0
        self.completed_transition_count = 0
        self.terminal_transition_count = 0
        self.censored_transition_count = 0

    @property
    def pending(self) -> bool:
        return self._pending is not None

    @property
    def completed_transitions(self) -> tuple[CleanDecisionTransition, ...]:
        return tuple(self._completed)

    def start_episode(self, episode_index: int) -> None:
        if self._pending is not None:
            raise RuntimeError("decision transition tracker still has a pending decision")
        self._episode_index = int(episode_index)

    def record_decisions(self, *, slot_index: int, records: Iterable[Any]) -> int:
        if self._episode_index is None:
            raise RuntimeError("decision transition tracker episode is not initialized")
        count = 0
        for record in records:
            state = CleanDecisionState.from_rollout_record(
                record,
                episode_index=self._episode_index,
                lane_index=self.lane_index,
                slot_index=int(slot_index),
            )
            if self._pending is not None:
                self._close_pending(
                    next_state=state,
                    terminated=False,
                    truncated=False,
                    unresolved=False,
                    future_bootstrap=None,
                )
            self._pending = _PendingDecisionTransition(state=state)
            self.registered_decision_count += 1
            count += 1
        return count

    def record_slot_reward(self, reward: float) -> None:
        if not math.isfinite(float(reward)):
            raise ValueError("decision transition reward must be finite")
        if self._pending is None:
            return
        self._pending.rho += (self.gamma ** self._pending.delta) * float(reward)
        self._pending.delta += 1

    def close_terminated(self) -> None:
        if self._pending is not None:
            self._close_pending(
                next_state=None,
                terminated=True,
                truncated=False,
                unresolved=False,
                future_bootstrap=0.0,
            )
            self.terminal_transition_count += 1
        self._episode_index = None

    def close_truncated(self) -> None:
        if self._pending is not None:
            self._close_pending(
                next_state=None,
                terminated=False,
                truncated=True,
                unresolved=True,
                future_bootstrap=None,
            )
            self.censored_transition_count += 1
        self._episode_index = None

    def close_rollout_boundary(self) -> None:
        """Censor a pending decision without ending the live environment episode."""
        if self._pending is not None:
            self._close_pending(
                next_state=None,
                terminated=False,
                truncated=True,
                unresolved=True,
                future_bootstrap=None,
            )
            self.censored_transition_count += 1

    def pop_completed(self) -> list[CleanDecisionTransition]:
        completed = list(self._completed)
        self._completed.clear()
        return completed

    def summary(self) -> dict[str, int | bool]:
        return {
            "registered_decision_count": int(self.registered_decision_count),
            "completed_transition_count": int(self.completed_transition_count),
            "terminal_transition_count": int(self.terminal_transition_count),
            "censored_transition_count": int(self.censored_transition_count),
            "queued_transition_count": int(len(self._completed)),
            "pending": bool(self.pending),
        }

    def _close_pending(
        self,
        *,
        next_state: CleanDecisionState | None,
        terminated: bool,
        truncated: bool,
        unresolved: bool,
        future_bootstrap: float | None,
    ) -> None:
        if self._pending is None:
            raise RuntimeError("cannot close an empty decision transition")
        self._completed.append(
            CleanDecisionTransition(
                state=self._pending.state,
                rho=float(self._pending.rho),
                delta=int(self._pending.delta),
                next_state=next_state,
                terminated=bool(terminated),
                truncated=bool(truncated),
                unresolved=bool(unresolved),
                future_bootstrap=future_bootstrap,
            )
        )
        self._pending = None
        self.completed_transition_count += 1
