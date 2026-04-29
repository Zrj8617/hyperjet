from marl_models.base_model import MARLModel, ExperienceBatch
from marl_models.maddpg.agents import ActorNetwork, CriticNetwork
from marl_models.buffer_and_helpers import soft_update, GaussianNoise, get_state_dict, load_safe
import config
import torch
import torch.nn.functional as F
import numpy as np
import os
from typing import cast


class MADDPG(MARLModel):
    def __init__(
        self,
        model_name: str,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        device: str,
    ) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)
        self.total_obs_dim: int = num_agents * obs_dim
        self.total_action_dim: int = num_agents * action_dim

        # Create networks for each agent
        self.actors: list[ActorNetwork] = [ActorNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self.critics: list[CriticNetwork] = [CriticNetwork(self.total_obs_dim, self.total_action_dim).to(device) for _ in range(num_agents)]

        self.target_actors: list[ActorNetwork] = [ActorNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self.target_critics: list[CriticNetwork] = [CriticNetwork(self.total_obs_dim, self.total_action_dim).to(device) for _ in range(num_agents)]
        self._init_target_networks()

        self.actors = [cast(ActorNetwork, torch.compile(actor)) for actor in self.actors]
        self.critics = [cast(CriticNetwork, torch.compile(critic)) for critic in self.critics]

        # Create optimizers
        self.actor_optimizers: list[torch.optim.Adam] = [torch.optim.Adam(actor.parameters(), lr=config.ACTOR_LR) for actor in self.actors]
        self.critic_optimizers: list[torch.optim.Adam] = [torch.optim.Adam(critic.parameters(), lr=config.CRITIC_LR) for critic in self.critics]

        # Exploration Noise
        self.noise: list[GaussianNoise] = [GaussianNoise() for _ in range(num_agents)]

    def select_actions(self, observations: np.ndarray, exploration: bool) -> np.ndarray:
        with torch.no_grad():
            obs_tensor: torch.Tensor = torch.from_numpy(observations).to(self.device)
            actions: np.ndarray = np.empty_like(observations[:, : config.ACTION_DIM])

            for i in range(self.num_agents):
                action: np.ndarray = self.actors[i](obs_tensor[i].unsqueeze(0)).squeeze(0).cpu().numpy()

                if exploration:
                    action += self.noise[i].sample()
                actions[i] = np.clip(action, -1.0, 1.0)

        return actions

    def update(self, batch: ExperienceBatch) -> dict:
        assert isinstance(batch, tuple) and len(batch) == 5, "MADDPG expects OffPolicyExperienceBatch (tuple of 5 elements)"
        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = batch

        obs_tensor: torch.Tensor = torch.from_numpy(obs_batch).to(self.device, non_blocking=True)
        actions_tensor: torch.Tensor = torch.from_numpy(actions_batch).to(self.device, non_blocking=True)
        rewards_tensor: torch.Tensor = torch.from_numpy(rewards_batch).to(self.device, non_blocking=True)
        next_obs_tensor: torch.Tensor = torch.from_numpy(next_obs_batch).to(self.device, non_blocking=True)
        dones_tensor: torch.Tensor = torch.from_numpy(dones_batch).to(self.device, non_blocking=True)

        batch_size: int = obs_tensor.shape[0]  # Get batch size from the data
        obs_flat: torch.Tensor = obs_tensor.reshape(batch_size, -1)
        next_obs_flat: torch.Tensor = next_obs_tensor.reshape(batch_size, -1)
        actions_flat: torch.Tensor = actions_tensor.reshape(batch_size, -1)

        agent_losses: list[float] = []
        agent_critic_losses: list[float] = []

        with torch.no_grad():
            next_actions_list: list[torch.Tensor] = [self.target_actors[i](next_obs_tensor[:, i, :]) for i in range(self.num_agents)]
            next_actions_tensor: torch.Tensor = torch.cat(next_actions_list, dim=1)

        for agent_idx in range(self.num_agents):
            # Update Critic
            with torch.no_grad():
                target_q_value: torch.Tensor = self.target_critics[agent_idx](next_obs_flat, next_actions_tensor)
                agent_reward: torch.Tensor = rewards_tensor[:, agent_idx].unsqueeze(1)
                agent_done: torch.Tensor = dones_tensor[:, agent_idx].unsqueeze(1)
                y: torch.Tensor = agent_reward + config.DISCOUNT_FACTOR * target_q_value * (1 - agent_done)

            self.critic_optimizers[agent_idx].zero_grad(set_to_none=True)
            current_q_value: torch.Tensor = self.critics[agent_idx](obs_flat, actions_flat)
            critic_loss: torch.Tensor = F.mse_loss(current_q_value, y)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent_idx].parameters(), config.MAX_GRAD_NORM)
            self.critic_optimizers[agent_idx].step()
            agent_critic_losses.append(float(critic_loss.item()))

            # Update Actor
            pred_actions_tensor: torch.Tensor = actions_tensor.detach().clone()
            pred_actions_tensor[:, agent_idx, :] = self.actors[agent_idx](obs_tensor[:, agent_idx, :])
            pred_actions_flat: torch.Tensor = pred_actions_tensor.reshape(batch_size, -1)

            self.actor_optimizers[agent_idx].zero_grad(set_to_none=True)
            actor_loss: torch.Tensor = -self.critics[agent_idx](obs_flat, pred_actions_flat).mean()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), config.MAX_GRAD_NORM)
            self.actor_optimizers[agent_idx].step()
            agent_losses.append(float(actor_loss.item()))

            # Soft update target networks
            soft_update(self.target_actors[agent_idx], self.actors[agent_idx], config.UPDATE_FACTOR)
            soft_update(self.target_critics[agent_idx], self.critics[agent_idx], config.UPDATE_FACTOR)

        for n in self.noise:
            n.decay()

        # Return averaged losses across all agents
        return {
            "actor": float(np.mean(agent_losses)),
            "critic": float(np.mean(agent_critic_losses)),
        }

    def _init_target_networks(self) -> None:
        for actor, target_actor in zip(self.actors, self.target_actors):
            target_actor.load_state_dict(actor.state_dict())
        for critic, target_critic in zip(self.critics, self.target_critics):
            target_critic.load_state_dict(critic.state_dict())

    def reset(self) -> None:
        for n in self.noise:
            n.reset()

    def save(self, directory: str) -> None:
        for i in range(self.num_agents):
            torch.save(
                {
                    "actor": get_state_dict(self.actors[i]),
                    "critic": get_state_dict(self.critics[i]),
                    "target_actor": self.target_actors[i].state_dict(),
                    "target_critic": self.target_critics[i].state_dict(),
                    "actor_optimizer": self.actor_optimizers[i].state_dict(),
                    "critic_optimizer": self.critic_optimizers[i].state_dict(),
                    "noise_scale": self.noise[i].scale,
                },
                os.path.join(directory, f"agent_{i}.pth"),
            )

    def load(self, directory: str) -> None:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"❌ Model directory not found: {directory}")

        for i in range(self.num_agents):
            agent_path: str = os.path.join(directory, f"agent_{i}.pth")
            if not os.path.exists(agent_path):
                raise FileNotFoundError(f"❌ Model file not found: {agent_path}")
            checkpoint: dict = torch.load(agent_path, map_location=self.device, weights_only=True)

            load_safe(self.actors[i], checkpoint["actor"])
            load_safe(self.critics[i], checkpoint["critic"])
            self.target_actors[i].load_state_dict(checkpoint["target_actor"])
            self.target_critics[i].load_state_dict(checkpoint["target_critic"])
            self.actor_optimizers[i].load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizers[i].load_state_dict(checkpoint["critic_optimizer"])
            self.noise[i].scale = checkpoint["noise_scale"]
