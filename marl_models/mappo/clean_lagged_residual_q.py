from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(slots=True)
class CleanLaggedPendingAction:
    episode_index: int
    slot_index: int
    task_id: str
    dag_id: str
    assignment_time: float
    selected_estimated_incremental_delay: float
    selected_input: np.ndarray


@dataclass(slots=True)
class CleanLaggedQSample:
    episode_index: int
    slot_index: int
    task_id: str
    dag_id: str
    selected_input: np.ndarray
    target: float
    weight: float
    censored: bool
    residual_seconds: float


def lagged_residual_target(
    *,
    assignment_time: float,
    outcome_time: float,
    estimated_incremental_delay: float,
    scale_seconds: float,
    censored: bool,
) -> tuple[float, float]:
    values = (assignment_time, outcome_time, estimated_incremental_delay, scale_seconds)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("lagged residual target inputs must be finite")
    if float(scale_seconds) <= 0.0:
        raise ValueError("lagged residual target scale must be positive")
    if float(outcome_time) < float(assignment_time):
        raise ValueError("lagged residual outcome precedes assignment")
    residual = (
        float(outcome_time) - float(assignment_time) - float(estimated_incremental_delay)
    )
    if bool(censored):
        residual = max(residual, 0.0)
    target = -math.tanh(residual / float(scale_seconds))
    if not math.isfinite(target) or target < -1.0 or target > 1.0:
        raise FloatingPointError("lagged residual target is invalid")
    return float(target), float(residual)


class CleanLaggedOutcomeTracker:
    """Episode-local delayed DAG-outcome tracker; never stores PPO log probabilities."""

    def __init__(self, *, scale_seconds: float, censor_weight: float) -> None:
        if not math.isfinite(float(scale_seconds)) or float(scale_seconds) <= 0.0:
            raise ValueError("lagged Q scale must be finite and positive")
        if not math.isfinite(float(censor_weight)) or not 0.0 <= float(censor_weight) <= 1.0:
            raise ValueError("lagged Q censor weight must be in [0, 1]")
        self.scale_seconds = float(scale_seconds)
        self.censor_weight = float(censor_weight)
        self._episode_index: int | None = None
        self._pending: dict[tuple[int, str], CleanLaggedPendingAction] = {}
        self._finalized: list[CleanLaggedQSample] = []
        self.registered_count = 0
        self.completed_count = 0
        self.censored_count = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def finalized_count(self) -> int:
        return len(self._finalized)

    def start_episode(self, episode_index: int) -> None:
        if self._pending or self._finalized:
            raise RuntimeError("lagged outcome tracker was not cleared before a new episode")
        self._episode_index = int(episode_index)
        self.registered_count = 0
        self.completed_count = 0
        self.censored_count = 0

    def register_rollout_actions(self, *, slot_record: Any, env: Any) -> None:
        if self._episode_index is None:
            raise RuntimeError("lagged outcome tracker episode is not initialized")
        for record in slot_record.offloading_records:
            if record.candidate_features is None or record.critic_global_context is None:
                raise ValueError("lagged Q action record is missing stored candidate/context features")
            if record.dag_id is None or record.assignment_time_seconds is None:
                raise ValueError("lagged Q action record is missing DAG/assignment provenance")
            if record.selected_estimated_incremental_delay is None:
                raise ValueError("lagged Q action record is missing selected EFT delay")
            features = np.asarray(record.candidate_features, dtype=np.float32)
            context = np.asarray(record.critic_global_context, dtype=np.float32).reshape(-1)
            selected = int(record.selected_action)
            if features.ndim != 2 or selected < 0 or selected >= int(features.shape[0]):
                raise ValueError("lagged Q candidate feature/selected action mismatch")
            selected_input = np.concatenate([features[selected], context], axis=0).astype(
                np.float32, copy=False
            )
            if not bool(np.isfinite(selected_input).all()):
                raise FloatingPointError("lagged Q selected input is non-finite")
            task_id = str(record.task_id)
            key = (int(self._episode_index), task_id)
            if key in self._pending:
                raise ValueError(f"duplicate lagged Q task registration: {key}")
            task = env.task_manager.get_task(task_id)
            if task is None or str(task.dag_id) != str(record.dag_id):
                raise ValueError(f"lagged Q task/DAG provenance mismatch: {task_id}")
            self._pending[key] = CleanLaggedPendingAction(
                episode_index=int(self._episode_index),
                slot_index=int(slot_record.slot_index),
                task_id=task_id,
                dag_id=str(record.dag_id),
                assignment_time=float(record.assignment_time_seconds),
                selected_estimated_incremental_delay=float(
                    record.selected_estimated_incremental_delay
                ),
                selected_input=selected_input.copy(),
            )
            self.registered_count += 1

    def resolve_completed(self, *, env: Any) -> int:
        resolved_keys: list[tuple[int, str]] = []
        for key, pending in self._pending.items():
            job = env.task_manager.get_job(pending.dag_id)
            if job is None or not bool(job.completed) or job.return_complete_time is None:
                continue
            self._finalized.append(
                self._finalize(
                    pending,
                    outcome_time=float(job.return_complete_time),
                    censored=False,
                )
            )
            resolved_keys.append(key)
            self.completed_count += 1
        for key in resolved_keys:
            del self._pending[key]
        return len(resolved_keys)

    def finalize_censored(self, *, episode_end_time: float) -> int:
        if not math.isfinite(float(episode_end_time)):
            raise ValueError("lagged Q episode end time must be finite")
        keys = list(self._pending)
        for key in keys:
            pending = self._pending.pop(key)
            self._finalized.append(
                self._finalize(
                    pending,
                    outcome_time=float(episode_end_time),
                    censored=True,
                )
            )
            self.censored_count += 1
        return len(keys)

    def pop_finalized(self) -> list[CleanLaggedQSample]:
        samples = list(self._finalized)
        self._finalized.clear()
        return samples

    def finish_episode(self) -> dict[str, int]:
        if self._pending or self._finalized:
            raise RuntimeError("lagged outcome tracker finished with unsettled samples")
        summary = {
            "registered": int(self.registered_count),
            "completed": int(self.completed_count),
            "censored": int(self.censored_count),
            "unresolved_before_censoring": int(self.censored_count),
            "pending_after_clear": 0,
        }
        self._episode_index = None
        return summary

    def _finalize(
        self,
        pending: CleanLaggedPendingAction,
        *,
        outcome_time: float,
        censored: bool,
    ) -> CleanLaggedQSample:
        target, residual = lagged_residual_target(
            assignment_time=pending.assignment_time,
            outcome_time=outcome_time,
            estimated_incremental_delay=pending.selected_estimated_incremental_delay,
            scale_seconds=self.scale_seconds,
            censored=censored,
        )
        return CleanLaggedQSample(
            episode_index=pending.episode_index,
            slot_index=pending.slot_index,
            task_id=pending.task_id,
            dag_id=pending.dag_id,
            selected_input=pending.selected_input.copy(),
            target=target,
            weight=self.censor_weight if censored else 1.0,
            censored=bool(censored),
            residual_seconds=residual,
        )


class CleanLaggedResidualQCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        if int(input_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("lagged residual Q dimensions must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        output = self.net[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() != 2 or int(inputs.shape[1]) != self.input_dim:
            raise ValueError("lagged residual Q input shape mismatch")
        return self.net(inputs).squeeze(-1)
