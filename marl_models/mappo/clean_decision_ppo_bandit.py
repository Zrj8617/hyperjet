from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


DECISION_BANDIT_REGRET_SCALE = 61.75621424202263


def _immutable_array(value: Any, *, dtype: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    copied = np.asarray(value, dtype=dtype).copy()
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True, slots=True)
class DecisionBanditSkipEvent:
    environment_id: int
    trajectory_id: str
    episode_id: int
    physical_slot: int
    task_id: str
    dag_id: str
    decision_order: int
    valid_candidate_count: int
    skip_reason: str


@dataclass(frozen=True, slots=True)
class DecisionBanditRecord:
    environment_id: int
    trajectory_id: str
    episode_id: int
    physical_slot: int
    decision_order: int
    task_id: str
    dag_id: str
    task_local_index: int
    task_id_to_idx: tuple[tuple[str, int], ...]
    idx_to_task_id: tuple[str, ...]
    task_features: np.ndarray
    dynamic_uav_features: np.ndarray
    pair_features: np.ndarray
    candidate_uav_ids: tuple[int, ...]
    candidate_mask: np.ndarray
    candidate_estimated_finish_times: np.ndarray
    valid_candidate_count: int
    executed_action: int
    executed_uav_id: int
    old_masked_probabilities: np.ndarray
    old_log_prob: float
    best_legal_eft: float
    selected_eft: float
    raw_eft_regret: float
    regret_scale: float
    old_policy_baseline: float
    advantage: float


@dataclass(slots=True)
class DecisionBanditRolloutBuffer:
    records: list[DecisionBanditRecord] = field(default_factory=list)
    skip_events: list[DecisionBanditSkipEvent] = field(default_factory=list)
    physical_slot_count: int = 0
    forced_decision_count: int = 0

    def append_record(self, record: DecisionBanditRecord) -> None:
        if int(record.valid_candidate_count) < 2:
            self.forced_decision_count += 1
            return
        self.records.append(record)

    def append_skip(self, event: DecisionBanditSkipEvent) -> None:
        if int(event.valid_candidate_count) != 0:
            raise ValueError("skip events must have valid_candidate_count=0")
        self.skip_events.append(event)

    @property
    def choice_decision_count(self) -> int:
        return len(self.records)

    @property
    def skipped_no_candidate(self) -> int:
        return len(self.skip_events)


@dataclass(frozen=True, slots=True)
class DecisionBanditUpdateConfig:
    clip_epsilon: float = 0.2
    ppo_epochs: int = 3
    max_grad_norm: float = 0.5
    chunk_decisions: int = 64
    entropy_coef: float = 0.0


@dataclass(slots=True)
class DecisionBanditUpdateStats:
    empty_actor_batch: bool
    effective_decision_count: int
    optimizer_step_count: int
    epochs: list[dict[str, Any]]


def build_decision_bandit_record(
    *,
    action_record: Any,
    graph_snapshot: Any,
    environment_id: int,
    trajectory_id: str,
    episode_id: int,
    physical_slot: int,
    regret_scale: float = DECISION_BANDIT_REGRET_SCALE,
    tolerance: float = 1e-5,
) -> DecisionBanditRecord:
    scale = float(regret_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("regret_scale must be finite and positive")
    mask = _immutable_array(action_record.candidate_mask, dtype=bool)
    eft = _immutable_array(action_record.candidate_estimated_finish_times, dtype=np.float32)
    probabilities = _immutable_array(action_record.old_masked_probabilities, dtype=np.float32)
    dynamic = _immutable_array(action_record.dynamic_uav_features, dtype=np.float32)
    pair = _immutable_array(action_record.pair_features, dtype=np.float32)
    task_features = _immutable_array(graph_snapshot.task_features, dtype=np.float32)
    candidate_uav_ids = tuple(int(value) for value in action_record.candidate_uav_ids)
    candidate_count = int(mask.shape[0])
    if (
        mask.ndim != 1
        or eft.shape != mask.shape
        or probabilities.shape != mask.shape
        or dynamic.ndim != 2
        or pair.ndim != 2
        or dynamic.shape[0] != candidate_count
        or pair.shape[0] != candidate_count
        or len(candidate_uav_ids) != candidate_count
    ):
        raise ValueError("decision-time candidate rows, mask, probabilities, and EFT are misaligned")
    legal_count = int(mask.sum())
    if legal_count <= 0:
        raise ValueError("a DecisionBanditRecord requires at least one legal candidate")
    if not np.isfinite(eft[mask]).all() or not np.isfinite(probabilities).all():
        raise FloatingPointError("decision-time EFT/probabilities must be finite")
    if not np.allclose(probabilities[~mask], 0.0, atol=tolerance, rtol=0.0):
        raise ValueError("illegal candidates must have zero old probability")
    if not math.isclose(float(probabilities[mask].sum()), 1.0, abs_tol=tolerance, rel_tol=tolerance):
        raise ValueError("legal old probabilities must sum to one")
    action = int(action_record.selected_action)
    if action < 0 or action >= candidate_count or not bool(mask[action]):
        raise ValueError("executed action is not legal in its historical mask")
    expected_log_prob = math.log(max(float(probabilities[action]), np.finfo(np.float32).tiny))
    if not math.isclose(
        float(action_record.old_log_prob),
        expected_log_prob,
        abs_tol=tolerance,
        rel_tol=tolerance,
    ):
        raise ValueError("old_log_prob is inconsistent with old_masked_probabilities")
    selected_eft = float(eft[action])
    if not math.isclose(
        float(action_record.selected_estimated_finish_time),
        selected_eft,
        abs_tol=tolerance,
        rel_tol=tolerance,
    ):
        raise ValueError("selected EFT is inconsistent with the historical EFT vector")
    legal_best = float(eft[mask].min())
    raw_regrets = np.zeros_like(eft, dtype=np.float32)
    raw_regrets[mask] = np.maximum(eft[mask] - legal_best, 0.0)
    rewards = -raw_regrets / scale
    baseline = float(np.sum(probabilities * rewards))
    advantage = float(rewards[action] - baseline)
    return DecisionBanditRecord(
        environment_id=int(environment_id),
        trajectory_id=str(trajectory_id),
        episode_id=int(episode_id),
        physical_slot=int(physical_slot),
        decision_order=int(action_record.decision_order),
        task_id=str(action_record.task_id),
        dag_id=str(action_record.dag_id),
        task_local_index=int(action_record.task_local_index),
        task_id_to_idx=tuple(
            (str(task_id), int(index))
            for task_id, index in sorted(
                graph_snapshot.task_id_to_idx.items(), key=lambda item: int(item[1])
            )
        ),
        idx_to_task_id=tuple(str(value) for value in graph_snapshot.idx_to_task_id),
        task_features=task_features,
        dynamic_uav_features=dynamic,
        pair_features=pair,
        candidate_uav_ids=candidate_uav_ids,
        candidate_mask=mask,
        candidate_estimated_finish_times=eft,
        valid_candidate_count=legal_count,
        executed_action=action,
        executed_uav_id=int(action_record.selected_uav_id),
        old_masked_probabilities=probabilities,
        old_log_prob=float(action_record.old_log_prob),
        best_legal_eft=legal_best,
        selected_eft=selected_eft,
        raw_eft_regret=float(raw_regrets[action]),
        regret_scale=scale,
        old_policy_baseline=baseline,
        advantage=advantage,
    )


def decision_ppo_surrogate(
    *,
    new_log_prob: Any,
    old_log_prob: Any,
    advantage: Any,
    clip_epsilon: float,
) -> tuple[Any, Any]:
    ratio = (new_log_prob - old_log_prob).exp()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon)) * advantage
    return -minimum(unclipped, clipped), ratio


def minimum(left: Any, right: Any) -> Any:
    import torch

    return torch.minimum(left, right)


def _module_grad_norm(module: Any) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().cpu().item())
    return math.sqrt(total)


class DecisionBanditPPOUpdater:
    def __init__(
        self,
        *,
        encoder: Any,
        scorer: Any,
        optimizer: Any,
        config: DecisionBanditUpdateConfig | None = None,
        device: Any = "cpu",
    ) -> None:
        import torch

        self.torch = torch
        self.encoder = encoder
        self.scorer = scorer
        self.optimizer = optimizer
        self.config = config or DecisionBanditUpdateConfig()
        self.device = device
        if float(self.config.entropy_coef) != 0.0:
            raise ValueError("Stage 1 entropy_coef must be exactly zero")
        if int(self.config.ppo_epochs) <= 0 or int(self.config.chunk_decisions) <= 0:
            raise ValueError("ppo_epochs and chunk_decisions must be positive")

    def _current_distribution(self, record: DecisionBanditRecord) -> Any:
        torch = self.torch
        task_features = torch.as_tensor(
            np.array(record.task_features, dtype=np.float32, copy=True),
            dtype=torch.float32,
            device=self.device,
        )
        task_embeddings = self.encoder(task_features)
        task_index = int(record.task_local_index)
        if task_index < 0 or task_index >= int(task_embeddings.shape[0]):
            raise ValueError("historical task_local_index is outside MLP output")
        dynamic = torch.as_tensor(
            np.array(record.dynamic_uav_features, dtype=np.float32, copy=True),
            dtype=torch.float32,
            device=self.device,
        )
        pair = torch.as_tensor(
            np.array(record.pair_features, dtype=np.float32, copy=True),
            dtype=torch.float32,
            device=self.device,
        )
        task = task_embeddings[task_index].reshape(1, -1).expand(dynamic.shape[0], -1)
        features = torch.cat([task, dynamic, pair], dim=1)
        mask = torch.as_tensor(
            np.array(record.candidate_mask, dtype=bool, copy=True),
            dtype=torch.bool,
            device=self.device,
        )
        logits = self.scorer(features)
        masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return torch.distributions.Categorical(logits=masked_logits)

    def update(self, buffer: DecisionBanditRolloutBuffer) -> DecisionBanditUpdateStats:
        torch = self.torch
        records = list(buffer.records)
        if not records:
            return DecisionBanditUpdateStats(
                empty_actor_batch=True,
                effective_decision_count=0,
                optimizer_step_count=0,
                epochs=[
                    {
                        "actor_loss": None,
                        "entropy": None,
                        "approx_kl": None,
                        "clip_fraction": None,
                        "ratio_mean": None,
                        "ratio_std": None,
                        "ratio_min": None,
                        "ratio_max": None,
                    }
                    for _ in range(int(self.config.ppo_epochs))
                ],
            )

        epoch_rows: list[dict[str, Any]] = []
        optimizer_steps = 0
        full_count = len(records)
        for epoch_index in range(int(self.config.ppo_epochs)):
            self.optimizer.zero_grad(set_to_none=True)
            losses: list[float] = []
            ratios: list[float] = []
            entropies: list[float] = []
            kls: list[float] = []
            clipped_flags: list[float] = []
            for start in range(0, full_count, int(self.config.chunk_decisions)):
                chunk = records[start : start + int(self.config.chunk_decisions)]
                chunk_loss_sum = None
                for record in chunk:
                    dist = self._current_distribution(record)
                    action = torch.as_tensor(
                        int(record.executed_action), dtype=torch.long, device=self.device
                    )
                    new_log_prob = dist.log_prob(action)
                    old_log_prob = torch.as_tensor(
                        float(record.old_log_prob), dtype=torch.float32, device=self.device
                    )
                    advantage = torch.as_tensor(
                        float(record.advantage), dtype=torch.float32, device=self.device
                    ).detach()
                    item_loss, ratio = decision_ppo_surrogate(
                        new_log_prob=new_log_prob,
                        old_log_prob=old_log_prob,
                        advantage=advantage,
                        clip_epsilon=float(self.config.clip_epsilon),
                    )
                    chunk_loss_sum = item_loss if chunk_loss_sum is None else chunk_loss_sum + item_loss
                    ratio_value = float(ratio.detach().cpu().item())
                    loss_value = float(item_loss.detach().cpu().item())
                    if not math.isfinite(ratio_value) or not math.isfinite(loss_value):
                        raise FloatingPointError("decision PPO ratio/loss is non-finite")
                    losses.append(loss_value)
                    ratios.append(ratio_value)
                    entropies.append(float(dist.entropy().detach().cpu().item()))
                    kls.append(float(old_log_prob.detach().cpu().item() - new_log_prob.detach().cpu().item()))
                    clipped_flags.append(
                        float(abs(ratio_value - 1.0) > float(self.config.clip_epsilon))
                    )
                assert chunk_loss_sum is not None
                (chunk_loss_sum / float(full_count)).backward()

            pre_encoder = _module_grad_norm(self.encoder)
            pre_scorer = _module_grad_norm(self.scorer)
            parameters = list(self.encoder.parameters()) + list(self.scorer.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, float(self.config.max_grad_norm)
            )
            post_encoder = _module_grad_norm(self.encoder)
            post_scorer = _module_grad_norm(self.scorer)
            if not math.isfinite(float(grad_norm.detach().cpu().item())):
                raise FloatingPointError("decision PPO gradient norm is non-finite")
            self.optimizer.step()
            optimizer_steps += 1
            ratio_array = np.asarray(ratios, dtype=np.float64)
            epoch_rows.append(
                {
                    "epoch_index": int(epoch_index),
                    "actor_loss": float(np.mean(losses)),
                    "entropy": float(np.mean(entropies)),
                    "approx_kl": float(np.mean(kls)),
                    "clip_fraction": float(np.mean(clipped_flags)),
                    "ratio_mean": float(ratio_array.mean()),
                    "ratio_std": float(ratio_array.std()),
                    "ratio_min": float(ratio_array.min()),
                    "ratio_max": float(ratio_array.max()),
                    "encoder_grad_pre_clip": pre_encoder,
                    "scorer_grad_pre_clip": pre_scorer,
                    "encoder_grad_post_clip": post_encoder,
                    "scorer_grad_post_clip": post_scorer,
                }
            )
        return DecisionBanditUpdateStats(
            empty_actor_batch=False,
            effective_decision_count=full_count,
            optimizer_step_count=optimizer_steps,
            epochs=epoch_rows,
        )
