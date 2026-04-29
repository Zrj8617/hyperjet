from marl_models.base_model import MARLModel, ExperienceBatch
from marl_models.mappo.agents import ActorNetwork, CriticNetwork
from marl_models.buffer_and_helpers import get_state_dict, load_safe
import config
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from typing import cast


class MAPPO(MARLModel):
    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)
        self.state_dim: int = obs_dim * num_agents

        self.actor: ActorNetwork = ActorNetwork(obs_dim, action_dim).to(device)
        self.critic: CriticNetwork = CriticNetwork(self.state_dim, num_agents).to(device)

        self.actor = cast(ActorNetwork, torch.compile(self.actor))
        self.critic = cast(CriticNetwork, torch.compile(self.critic))

        self.actor_optimizer: torch.optim.Adam = torch.optim.Adam(self.actor.parameters(), lr=config.ACTOR_LR)
        self.critic_optimizer: torch.optim.Adam = torch.optim.Adam(self.critic.parameters(), lr=config.CRITIC_LR)

        self.eye = torch.eye(num_agents, device=device)  # For one-hot encoding agent IDs

    def select_actions(self, observations: np.ndarray, exploration: bool) -> np.ndarray:
        with torch.no_grad():
            obs_tensor: torch.Tensor = torch.from_numpy(observations).to(self.device)
            dist: Normal = self.actor(obs_tensor)
            actions: torch.Tensor = dist.sample() if exploration else dist.mean

        # For training, we clip in train.py but for testing we clip here itself
        return np.clip(actions.cpu().numpy(), -1.0, 1.0)

    def get_action_and_value(self, obs: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            obs_tensor: torch.Tensor = torch.from_numpy(obs).to(self.device)
            dist: Normal = self.actor(obs_tensor)
            actions: torch.Tensor = dist.sample()

            # Get the log probability of the sampled actions
            log_probs: torch.Tensor = dist.log_prob(actions).sum(dim=-1)

            state_tensor: torch.Tensor = torch.from_numpy(state).to(self.device)
            state_batch: torch.Tensor = state_tensor.unsqueeze(0).expand(self.num_agents, -1)  # (Num_Agents, State_Dim)
            critic_input: torch.Tensor = torch.cat([state_batch, self.eye], dim=-1)  # (Num_Agents, State_Dim + Num_Agents)
            values: torch.Tensor = self.critic(critic_input).squeeze(-1)  # (Num_Agents, 1) -> (Num_Agents,)

        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    def update(self, batch: ExperienceBatch) -> dict:
        assert isinstance(batch, dict), "MAPPO expects OnPolicyExperienceBatch (dict)"
        obs_batch: torch.Tensor = batch["obs"]
        actions_batch: torch.Tensor = batch["actions"]
        old_log_probs_batch: torch.Tensor = batch["old_log_probs"]
        advantages_batch: torch.Tensor = batch["advantages"]
        returns_batch: torch.Tensor = batch["returns"]
        states_batch: torch.Tensor = batch["states"]
        agent_ids_batch: torch.Tensor = batch["agent_ids"]
        old_values_batch: torch.Tensor = batch["old_values"]

        one_hots: torch.Tensor = F.one_hot(agent_ids_batch, num_classes=self.num_agents).float()
        critic_input: torch.Tensor = torch.cat([states_batch, one_hots], dim=-1)

        # Critic Update
        values: torch.Tensor = self.critic(critic_input).squeeze(-1)

        # Value clipping
        values_clipped: torch.Tensor = old_values_batch + torch.clamp(values - old_values_batch, -config.PPO_CLIP_EPS, config.PPO_CLIP_EPS)
        vf_loss1: torch.Tensor = (values - returns_batch).pow(2)
        vf_loss2: torch.Tensor = (values_clipped - returns_batch).pow(2)
        critic_loss: torch.Tensor = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), config.MAX_GRAD_NORM)
        self.critic_optimizer.step()

        # Actor Update
        dist: Normal = self.actor(obs_batch)
        new_log_probs: torch.Tensor = dist.log_prob(actions_batch).sum(dim=-1)
        ratio: torch.Tensor = torch.exp(new_log_probs - old_log_probs_batch)

        # PPO surrogate loss
        surr1: torch.Tensor = ratio * advantages_batch
        surr2: torch.Tensor = torch.clamp(ratio, 1.0 - config.PPO_CLIP_EPS, 1.0 + config.PPO_CLIP_EPS) * advantages_batch
        actor_loss: torch.Tensor = -torch.min(surr1, surr2).mean()

        # Adding entropy bonus for exploration
        entropy_loss: torch.Tensor = dist.entropy().mean()
        actor_loss -= config.PPO_ENTROPY_COEF * entropy_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), config.MAX_GRAD_NORM)
        self.actor_optimizer.step()

        # Return losses for logging
        return {
            "actor": float(actor_loss.item()),
            "critic": float(critic_loss.item()),
            "entropy": float(entropy_loss.item()),
        }

    def reset(self) -> None:
        pass  # Nothing to reset

    def save(self, directory: str) -> None:
        torch.save(
            {
                "actor": get_state_dict(self.actor),
                "critic": get_state_dict(self.critic),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            os.path.join(directory, "mappo.pth"),
        )

    def load(self, directory: str) -> None:
        path: str = os.path.join(directory, "mappo.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Model file not found: {path}")
        checkpoint: dict = torch.load(path, map_location=self.device, weights_only=True)
        load_safe(self.actor, checkpoint["actor"])
        load_safe(self.critic, checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
