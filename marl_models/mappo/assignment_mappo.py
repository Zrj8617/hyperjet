from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

import config
from environment.graph_builder import HeteroGraphSnapshot
from marl_models.hgnn.scheduler import PhaseOneGraphScheduler
from marl_models.mappo.assignment_agents import AssignmentActor, AssignmentCritic


@dataclass(slots=True)
class AssignmentUpdateMetrics:
    actor_loss: float = 0.0
    critic_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    rollout_steps: int = 0
    assignment_decisions: int = 0
    executed_decisions: int = 0


class AssignmentMAPPO:
    """Minimal centralized PPO policy for task-to-UAV assignment."""

    def __init__(self, device: str = "cpu", encoder_checkpoint: str = "") -> None:
        if not config.RL_ASSIGNMENT_USE_HGNN_ENCODER:
            raise NotImplementedError("Assignment RL v1 requires the HGNN encoder.")
        if config.RL_ASSIGNMENT_TRAIN_ENCODER:
            raise NotImplementedError("Assignment RL v1 stores detached actor inputs; encoder fine-tuning is not implemented.")
        self.device = torch.device(device)
        self.scheduler = PhaseOneGraphScheduler(device=device)
        if encoder_checkpoint and config.RL_ASSIGNMENT_LOAD_ENCODER_CHECKPOINT:
            self._load_encoder_checkpoint(encoder_checkpoint)
        self.scheduler.eval()
        for parameter in self.scheduler.encoder.parameters():
            parameter.requires_grad = False

        actor_input_dim = config.TASK_EMB_DIM + config.UAV_EMB_DIM + config.BASE_TASK_UAV_PAIR_FEATURE_DIM
        critic_input_dim = config.TASK_EMB_DIM + config.UAV_EMB_DIM
        self.actor = AssignmentActor(actor_input_dim, config.HGNN_HIDDEN_DIM).to(self.device)
        self.critic = AssignmentCritic(critic_input_dim, config.HGNN_HIDDEN_DIM).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.RL_ASSIGNMENT_LR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.RL_ASSIGNMENT_LR)
        self._rollout_steps: list[dict[str, Any]] = []
        self._latest_critic_input: torch.Tensor | None = None
        self._latest_shared_value: float = 0.0

    def _load_encoder_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"HGNN encoder checkpoint not found: {path}")
        state_dict = torch.load(path, map_location=self.device)
        encoder_state = {
            key.removeprefix("encoder."): value
            for key, value in state_dict.items()
            if key.startswith("encoder.")
        }
        if not encoder_state:
            raise RuntimeError(f"Checkpoint does not contain encoder parameters: {path}")
        self.scheduler.encoder.load_state_dict(encoder_state, strict=True)

    def _global_critic_input(self, task_embeddings: torch.Tensor, uav_embeddings: torch.Tensor) -> torch.Tensor:
        task_pool = (
            task_embeddings.mean(dim=0)
            if task_embeddings.shape[0] > 0
            else torch.zeros((config.TASK_EMB_DIM,), dtype=torch.float32, device=self.device)
        )
        uav_pool = (
            uav_embeddings.mean(dim=0)
            if uav_embeddings.shape[0] > 0
            else torch.zeros((config.UAV_EMB_DIM,), dtype=torch.float32, device=self.device)
        )
        return torch.cat([task_pool, uav_pool], dim=0)

    @torch.no_grad()
    def act(
        self,
        assignment_snapshot: HeteroGraphSnapshot,
        exploration: bool = True,
    ) -> tuple[dict[tuple[str, int], float], list[dict[str, Any]]]:
        encoded = self.scheduler.encode_graph(assignment_snapshot)
        global_input = self._global_critic_input(encoded.task_embeddings, encoded.uav_embeddings)
        shared_value = float(self.critic(global_input.unsqueeze(0)).item())
        self._latest_critic_input = global_input.detach().cpu()
        self._latest_shared_value = shared_value

        edge_rows_by_task: dict[str, list[tuple[int, int]]] = {}
        for edge_idx, (task_id, uav_id) in enumerate(assignment_snapshot.task_uav_edges):
            if task_id in encoded.task_index and uav_id in encoded.uav_index:
                edge_rows_by_task.setdefault(task_id, []).append((edge_idx, uav_id))

        edge_scores: dict[tuple[str, int], float] = {}
        records: list[dict[str, Any]] = []
        for task_id, edge_rows in edge_rows_by_task.items():
            task_embedding = encoded.task_embeddings[encoded.task_index[task_id]]
            candidate_uav_ids: list[int] = []
            actor_inputs: list[torch.Tensor] = []
            for edge_idx, uav_id in edge_rows:
                candidate_uav_ids.append(uav_id)
                actor_inputs.append(
                    torch.cat(
                        [
                            task_embedding,
                            encoded.uav_embeddings[encoded.uav_index[uav_id]],
                            encoded.edge_pair_features[edge_idx],
                        ],
                        dim=0,
                    )
                )
            candidate_inputs = torch.stack(actor_inputs, dim=0)
            logits = self.actor(candidate_inputs)
            dist = Categorical(logits=logits)
            selected_index = dist.sample() if exploration else torch.argmax(logits)
            selected_index_int = int(selected_index.item())
            selected_uav = int(candidate_uav_ids[selected_index_int])
            for uav_id in candidate_uav_ids:
                edge_scores[(task_id, uav_id)] = 1.0 if uav_id == selected_uav else 0.0
            records.append(
                {
                    "task_id": task_id,
                    "actor_called": True,
                    "candidate_uav_ids": list(candidate_uav_ids),
                    "selected_action_index": selected_index_int,
                    "actor_selected_uav": selected_uav,
                    "executor_selected_uav": None,
                    "action_executed": False,
                    "fallback_used": False,
                    "failure_reason": None,
                    "old_log_prob": float(dist.log_prob(selected_index).item()),
                    "entropy": float(dist.entropy().item()),
                    "shared_value": shared_value,
                    "candidate_actor_inputs": candidate_inputs.detach().cpu(),
                    "critic_input": global_input.detach().cpu(),
                }
            )
        return edge_scores, records

    @torch.no_grad()
    def act_for_task(
        self,
        assignment_snapshot: HeteroGraphSnapshot,
        task_id: str,
        candidate_uav_ids: list[int],
        exploration: bool = True,
    ) -> dict[str, Any]:
        """Samples one UAV for one task from the executor-provided candidate set."""
        encoded = self.scheduler.encode_graph(assignment_snapshot)
        global_input = self._global_critic_input(encoded.task_embeddings, encoded.uav_embeddings)
        shared_value = float(self.critic(global_input.unsqueeze(0)).item())
        self._latest_critic_input = global_input.detach().cpu()
        self._latest_shared_value = shared_value

        allowed_uav_ids = {int(uav_id) for uav_id in candidate_uav_ids}
        edge_rows = [
            (edge_idx, int(uav_id))
            for edge_idx, (edge_task_id, uav_id) in enumerate(assignment_snapshot.task_uav_edges)
            if edge_task_id == task_id
            and int(uav_id) in allowed_uav_ids
            and task_id in encoded.task_index
            and uav_id in encoded.uav_index
        ]
        edge_uav_ids = {uav_id for _, uav_id in edge_rows}
        if edge_uav_ids != allowed_uav_ids:
            raise RuntimeError(
                f"Executor candidates and graph edges disagree for task {task_id}: "
                f"executor={sorted(allowed_uav_ids)} graph={sorted(edge_uav_ids)}"
            )

        task_embedding = encoded.task_embeddings[encoded.task_index[task_id]]
        actor_inputs = [
            torch.cat(
                [
                    task_embedding,
                    encoded.uav_embeddings[encoded.uav_index[uav_id]],
                    encoded.edge_pair_features[edge_idx],
                ],
                dim=0,
            )
            for edge_idx, uav_id in edge_rows
        ]
        if not actor_inputs:
            raise RuntimeError(f"Actor called without feasible candidates for task {task_id}.")
        candidate_inputs = torch.stack(actor_inputs, dim=0)
        ordered_candidate_uav_ids = [uav_id for _, uav_id in edge_rows]
        logits = self.actor(candidate_inputs)
        dist = Categorical(logits=logits)
        selected_index = dist.sample() if exploration else torch.argmax(logits)
        selected_index_int = int(selected_index.item())
        selected_uav = int(ordered_candidate_uav_ids[selected_index_int])
        return {
            "task_id": task_id,
            "actor_called": True,
            "candidate_uav_ids": ordered_candidate_uav_ids,
            "selected_action_index": selected_index_int,
            "actor_selected_uav": selected_uav,
            "executor_selected_uav": None,
            "action_executed": False,
            "fallback_used": False,
            "failure_reason": None,
            "old_log_prob": float(dist.log_prob(selected_index).item()),
            "entropy": float(dist.entropy().item()),
            "shared_value": shared_value,
            "candidate_actor_inputs": candidate_inputs.detach().cpu(),
            "critic_input": global_input.detach().cpu(),
        }

    def store_step(
        self,
        *,
        env_step_id: int,
        rl_records: list[dict[str, Any]],
        shared_reward: float,
        done: bool,
    ) -> None:
        actor_records = [record for record in rl_records if record.get("actor_called", True)]
        decision_count = len(actor_records)
        records: list[dict[str, Any]] = []
        for record in rl_records:
            copied = dict(record)
            copied["env_step_id"] = int(env_step_id)
            copied["decision_count"] = int(decision_count)
            copied["shared_reward"] = float(shared_reward)
            copied["done"] = bool(done)
            records.append(copied)
        critic_record = next((record for record in records if "critic_input" in record), None)
        critic_input = critic_record["critic_input"] if critic_record is not None else self._latest_critic_input
        shared_value = float(critic_record["shared_value"]) if critic_record is not None else self._latest_shared_value
        self._rollout_steps.append(
            {
                "env_step_id": int(env_step_id),
                "reward": float(shared_reward),
                "done": bool(done),
                "value": shared_value,
                "critic_input": critic_input,
                "decisions": records,
            }
        )

    def _discounted_returns_and_advantages(self) -> tuple[list[float], list[float]]:
        returns = [0.0] * len(self._rollout_steps)
        advantages = [0.0] * len(self._rollout_steps)
        next_return = 0.0
        for idx in range(len(self._rollout_steps) - 1, -1, -1):
            step = self._rollout_steps[idx]
            mask = 0.0 if step["done"] else 1.0
            next_return = float(step["reward"]) + config.RL_ASSIGNMENT_GAMMA * next_return * mask
            returns[idx] = next_return
            advantages[idx] = next_return - float(step["value"])
        if advantages:
            advantage_tensor = torch.tensor(advantages, dtype=torch.float32)
            normalized = (advantage_tensor - advantage_tensor.mean()) / advantage_tensor.std(unbiased=False).clamp_min(1e-8)
            advantages = normalized.tolist()
        return returns, advantages

    def update(self) -> AssignmentUpdateMetrics:
        if not self._rollout_steps:
            return AssignmentUpdateMetrics()
        returns, advantages = self._discounted_returns_and_advantages()
        actor_samples: list[dict[str, Any]] = []
        critic_samples: list[dict[str, Any]] = []
        for step_idx, step in enumerate(self._rollout_steps):
            if step["critic_input"] is not None:
                critic_samples.append(
                    {
                        "critic_input": step["critic_input"],
                        "return": returns[step_idx],
                    }
                )
            for decision in step["decisions"]:
                if not decision.get("action_executed", False):
                    continue
                actor_samples.append(
                    {
                        "candidate_actor_inputs": decision["candidate_actor_inputs"],
                        "selected_action_index": int(decision["selected_action_index"]),
                        "old_log_prob": float(decision["old_log_prob"]),
                        "advantage": float(advantages[step_idx]),
                        "loss_weight": 1.0 / float(max(int(decision["decision_count"]), 1)),
                    }
                )

        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        approx_kls: list[float] = []
        clip_fractions: list[float] = []
        batch_size = max(int(config.RL_ASSIGNMENT_MINIBATCH_SIZE), 1)

        for _ in range(max(int(config.RL_ASSIGNMENT_UPDATE_EPOCHS), 1)):
            if critic_samples:
                for indices in torch.randperm(len(critic_samples)).split(batch_size):
                    batch = [critic_samples[int(idx)] for idx in indices]
                    critic_inputs = torch.stack([row["critic_input"] for row in batch]).to(self.device)
                    return_tensor = torch.tensor([row["return"] for row in batch], dtype=torch.float32, device=self.device)
                    values = self.critic(critic_inputs)
                    critic_loss = config.RL_ASSIGNMENT_VALUE_COEF * 0.5 * (values - return_tensor).pow(2).mean()
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), config.RL_ASSIGNMENT_MAX_GRAD_NORM)
                    self.critic_optimizer.step()
                    critic_losses.append(float(critic_loss.item()))

            if actor_samples:
                for indices in torch.randperm(len(actor_samples)).split(batch_size):
                    batch = [actor_samples[int(idx)] for idx in indices]
                    weighted_losses: list[torch.Tensor] = []
                    weighted_entropies: list[torch.Tensor] = []
                    batch_kls: list[torch.Tensor] = []
                    batch_clips: list[torch.Tensor] = []
                    weights: list[float] = []
                    for row in batch:
                        inputs = row["candidate_actor_inputs"].to(self.device)
                        dist = Categorical(logits=self.actor(inputs))
                        action = torch.tensor(row["selected_action_index"], dtype=torch.long, device=self.device)
                        old_log_prob = torch.tensor(row["old_log_prob"], dtype=torch.float32, device=self.device)
                        advantage = torch.tensor(row["advantage"], dtype=torch.float32, device=self.device)
                        new_log_prob = dist.log_prob(action)
                        ratio = torch.exp(new_log_prob - old_log_prob)
                        clipped_ratio = torch.clamp(
                            ratio,
                            1.0 - config.RL_ASSIGNMENT_CLIP_EPS,
                            1.0 + config.RL_ASSIGNMENT_CLIP_EPS,
                        )
                        weight = float(row["loss_weight"])
                        weighted_losses.append(-torch.minimum(ratio * advantage, clipped_ratio * advantage) * weight)
                        weighted_entropies.append(dist.entropy() * weight)
                        batch_kls.append((old_log_prob - new_log_prob).detach())
                        batch_clips.append((torch.abs(ratio - 1.0) > config.RL_ASSIGNMENT_CLIP_EPS).float().detach())
                        weights.append(weight)
                    normalizer = max(sum(weights), 1e-8)
                    policy_loss = torch.stack(weighted_losses).sum() / normalizer
                    entropy = torch.stack(weighted_entropies).sum() / normalizer
                    actor_loss = policy_loss - config.RL_ASSIGNMENT_ENTROPY_COEF * entropy
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), config.RL_ASSIGNMENT_MAX_GRAD_NORM)
                    self.actor_optimizer.step()
                    actor_losses.append(float(actor_loss.item()))
                    entropies.append(float(entropy.item()))
                    approx_kls.append(float(torch.stack(batch_kls).mean().item()))
                    clip_fractions.append(float(torch.stack(batch_clips).mean().item()))

        executed_decisions = sum(
            1
            for step in self._rollout_steps
            for decision in step["decisions"]
            if decision.get("action_executed", False)
        )
        metrics = AssignmentUpdateMetrics(
            actor_loss=float(np.mean(actor_losses)) if actor_losses else 0.0,
            critic_loss=float(np.mean(critic_losses)) if critic_losses else 0.0,
            entropy=float(np.mean(entropies)) if entropies else 0.0,
            approx_kl=float(np.mean(approx_kls)) if approx_kls else 0.0,
            clip_fraction=float(np.mean(clip_fractions)) if clip_fractions else 0.0,
            rollout_steps=len(self._rollout_steps),
            assignment_decisions=sum(
                1
                for step in self._rollout_steps
                for decision in step["decisions"]
                if decision.get("actor_called", True)
            ),
            executed_decisions=executed_decisions,
        )
        self._rollout_steps.clear()
        return metrics

    def save(self, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "encoder": self.scheduler.encoder.state_dict(),
            },
            path,
        )
