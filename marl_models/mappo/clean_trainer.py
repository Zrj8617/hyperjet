from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from marl_models.mappo.clean_slot_orchestrator import (
    CleanEncodedSlotState,
    CleanSlotRolloutBuffer,
    CleanSlotRolloutRecord,
    encode_prepared_slot,
)

try:
    import torch
    from torch.distributions import Categorical
except ModuleNotFoundError:
    torch = None
    Categorical = None


@dataclass(slots=True)
class CleanPPOUpdateConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    ppo_epochs: int = 1
    value_coef: float = 0.5
    movement_entropy_coef: float = 0.01
    offloading_entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


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


@dataclass(slots=True)
class CleanTrainingModules:
    hgnn: Any
    movement_actor: Any
    offloading_actor: Any
    critic: Any


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
        if torch is None:
            raise ModuleNotFoundError("torch is required to load clean model checkpoints")
        payload = torch.load(Path(path), map_location="cpu")
        modules.hgnn.load_state_dict(payload["hgnn"])
        modules.movement_actor.load_state_dict(payload["movement_actor"])
        modules.offloading_actor.load_state_dict(payload["offloading_actor"])
        modules.critic.load_state_dict(payload["critic"])
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

    def update(self, buffer: CleanSlotRolloutBuffer) -> CleanPPOUpdateStats:
        if not buffer.closed:
            raise RuntimeError("Clean PPO update requires a closed rollout buffer.")
        records = list(buffer.records)
        if not records:
            raise ValueError("Cannot update from an empty rollout buffer.")

        returns_np, advantages_np = compute_slot_level_gae(
            records,
            bootstrap_value=_buffer_bootstrap_value(records),
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        returns = torch.as_tensor(returns_np, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages_np, dtype=torch.float32, device=self.device)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)

        latest_stats: CleanPPOUpdateStats | None = None
        for _ in range(max(int(self.config.ppo_epochs), 1)):
            loss_parts = self._loss(records=records, returns=returns, advantages=advantages)
            self.optimizer.zero_grad(set_to_none=True)
            loss_parts["total_loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                _unique_parameters(
                    [
                        self.modules.hgnn,
                        self.modules.movement_actor,
                        self.modules.offloading_actor,
                        self.modules.critic,
                    ]
                ),
                float(self.config.max_grad_norm),
            )
            self.optimizer.step()
            latest_stats = self._stats_from_loss_parts(records, loss_parts, float(grad_norm))

        self.update_step += 1
        assert latest_stats is not None
        latest_stats.update_step = self.update_step
        return latest_stats

    def _loss(
        self,
        *,
        records: list[CleanSlotRolloutRecord],
        returns: Any,
        advantages: Any,
    ) -> dict[str, Any]:
        per_slot_move_losses: list[Any] = []
        per_slot_move_entropies: list[Any] = []
        per_slot_off_losses: list[Any] = []
        per_slot_off_entropies: list[Any] = []
        value_losses: list[Any] = []

        for slot_idx, record in enumerate(records):
            task_features = torch.as_tensor(record.graph_snapshot.task_features, dtype=torch.float32, device=self.device)
            incidence = torch.as_tensor(record.graph_snapshot.incidence_matrix, dtype=torch.float32, device=self.device)
            task_embeddings = self.modules.hgnn(task_features, incidence)
            critic_input = _critic_input_tensor(task_embeddings, record.critic_non_graph_input)
            value = self.modules.critic(critic_input).reshape(-1)[0]
            value_losses.append(0.5 * (value - returns[slot_idx]).pow(2))

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

            off_losses: list[Any] = []
            off_entropies: list[Any] = []
            for offloading_record in record.offloading_records:
                task_idx = int(offloading_record.task_local_index)
                if task_idx < 0 or task_idx >= int(task_embeddings.shape[0]):
                    continue
                task_embedding = task_embeddings[task_idx].reshape(1, -1)
                dynamic_features = torch.as_tensor(offloading_record.dynamic_uav_features, dtype=torch.float32, device=self.device)
                pair_features = torch.as_tensor(offloading_record.pair_features, dtype=torch.float32, device=self.device)
                candidate_count = int(dynamic_features.shape[0])
                features = torch.cat([task_embedding.expand(candidate_count, -1), dynamic_features, pair_features], dim=1)
                mask = torch.as_tensor(offloading_record.candidate_mask, dtype=torch.bool, device=self.device)
                logits = self.modules.offloading_actor.scorer(features).masked_fill(~mask, torch.finfo(torch.float32).min)
                dist = Categorical(logits=logits)
                action = torch.as_tensor(int(offloading_record.selected_action), dtype=torch.long, device=self.device)
                old_log_prob = torch.as_tensor(float(offloading_record.old_log_probability), dtype=torch.float32, device=self.device)
                off_losses.append(
                    _ppo_action_loss(
                        new_log_prob=dist.log_prob(action),
                        old_log_prob=old_log_prob,
                        advantage=advantages[slot_idx],
                        clip_epsilon=self.config.clip_epsilon,
                    )
                )
                off_entropies.append(dist.entropy())
            if off_losses:
                per_slot_off_losses.append(torch.stack(off_losses).mean())
                per_slot_off_entropies.append(torch.stack(off_entropies).mean())

        zero = torch.zeros((), dtype=torch.float32, device=self.device)
        movement_loss = torch.stack(per_slot_move_losses).mean() if per_slot_move_losses else zero
        movement_entropy = torch.stack(per_slot_move_entropies).mean() if per_slot_move_entropies else zero
        offloading_loss = torch.stack(per_slot_off_losses).mean() if per_slot_off_losses else zero
        offloading_entropy = torch.stack(per_slot_off_entropies).mean() if per_slot_off_entropies else zero
        value_loss = torch.stack(value_losses).mean() if value_losses else zero
        total_loss = (
            movement_loss
            + offloading_loss
            + float(self.config.value_coef) * value_loss
            - float(self.config.movement_entropy_coef) * movement_entropy
            - float(self.config.offloading_entropy_coef) * offloading_entropy
        )
        return {
            "movement_loss": movement_loss,
            "movement_entropy": movement_entropy,
            "offloading_loss": offloading_loss,
            "offloading_entropy": offloading_entropy,
            "value_loss": value_loss,
            "total_loss": total_loss,
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
        )


def build_single_optimizer(
    modules: CleanTrainingModules,
    *,
    lr: float = 3e-4,
    optimizer_cls: Any | None = None,
) -> Any:
    if torch is None:
        raise ModuleNotFoundError("torch is required to build clean PPO optimizer")
    optimizer_type = optimizer_cls or torch.optim.Adam
    return optimizer_type(_unique_parameters([modules.hgnn, modules.movement_actor, modules.offloading_actor, modules.critic]), lr=float(lr))


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
) -> CleanEncodedSlotState:
    return encode_prepared_slot(
        prepared_state=prepared_state,
        env=env,
        hgnn=modules.hgnn,
        critic=modules.critic,
        movement_actor=modules.movement_actor,
        device=device,
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


def write_clean_training_log(
    logger: CleanJSONLLogger,
    *,
    episode: int,
    global_slot: int,
    info: dict[str, Any],
    update_stats: CleanPPOUpdateStats | None = None,
    torch_skipped: bool = False,
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
        "invalid_assignment_count": info.get("invalid_assignment_count"),
        "invalid_assignment_rate": info.get("invalid_assignment_rate"),
        "action_executed_rate": info.get("action_executed_rate"),
        "movement_action_distribution": info.get("movement_action_distribution", {}),
        "offloading_action_count": info.get("offloading_action_count", info.get("assignment_buffer_entry_count", 0)),
        "terminated": bool(info.get("terminated", False)),
        "truncated": bool(info.get("truncated", False)),
        "torch_model_checks_skipped": bool(torch_skipped),
    }
    if update_stats is not None:
        payload.update({f"ppo_{key}": value for key, value in asdict(update_stats).items()})
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
        for param in module.parameters():
            marker = id(param)
            if marker not in seen:
                params.append(param)
                seen.add(marker)
    return params


def _module_grad_norm(module: Any) -> float:
    if torch is None:
        return 0.0
    total = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        total += float(param.grad.detach().pow(2).sum().cpu().item())
    return float(total ** 0.5)


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
