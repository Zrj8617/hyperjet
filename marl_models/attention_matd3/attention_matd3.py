from marl_models.base_model import MARLModel, ExperienceBatch
from marl_models.attention_matd3.agents import ActorNetwork, CriticNetwork
from marl_models.buffer_and_helpers import soft_update, GaussianNoise, get_state_dict, load_safe
import config
import torch
import torch.nn.functional as F
import numpy as np
from typing import cast
import os


class AttentionMATD3(MARLModel):
    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)

        self.actors: list[ActorNetwork] = [ActorNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self.critics_1: list[CriticNetwork] = [CriticNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self.critics_2: list[CriticNetwork] = [CriticNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]

        self.target_actors: list[ActorNetwork] = [ActorNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self.target_critics_1: list[CriticNetwork] = [CriticNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self.target_critics_2: list[CriticNetwork] = [CriticNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self._init_target_networks()

        self.actors = [cast(ActorNetwork, torch.compile(actor)) for actor in self.actors]
        self.critics_1 = [cast(CriticNetwork, torch.compile(critic)) for critic in self.critics_1]
        self.critics_2 = [cast(CriticNetwork, torch.compile(critic)) for critic in self.critics_2]

        self.actor_optimizers: list[torch.optim.Adam] = [torch.optim.Adam(actor.parameters(), lr=config.ACTOR_LR) for actor in self.actors]
        self.critic_1_optimizers: list[torch.optim.Adam] = [torch.optim.Adam(critic.parameters(), lr=config.CRITIC_LR) for critic in self.critics_1]
        self.critic_2_optimizers: list[torch.optim.Adam] = [torch.optim.Adam(critic.parameters(), lr=config.CRITIC_LR) for critic in self.critics_2]

        self.noise: list[GaussianNoise] = [GaussianNoise() for _ in range(num_agents)]

        # Delayed Updates Counter
        self.update_counter: int = 0

    def select_actions(self, observations: np.ndarray, exploration: bool) -> np.ndarray:
        # FIX: Batch all agent observations into a single tensor and run ONE
        # forward pass instead of N separate forward passes in a Python loop.
        # This keeps the GPU busy with one larger kernel instead of N tiny ones,
        # and avoids the N * (tensor creation + kernel-launch) overhead.
        #
        # Also use torch.from_numpy() instead of torch.tensor() — the latter
        # always copies memory and emits a UserWarning when given a numpy array;
        # from_numpy() shares memory (zero-copy) when the array is contiguous.
        with torch.no_grad():
            obs_tensor: torch.Tensor = torch.from_numpy(observations).to(self.device)  # zero-copy on CPU, then one H2D transfer
            actions: np.ndarray = np.empty_like(observations[:, : config.ACTION_DIM])  # pre-allocate result

            # We still call each actor separately because they have independent weights,
            # but we avoid the per-agent tensor-creation overhead above.
            for i in range(self.num_agents):
                action: np.ndarray = self.actors[i](obs_tensor[i].unsqueeze(0)).squeeze(0).cpu().numpy()

                if exploration:
                    action += self.noise[i].sample()
                actions[i] = np.clip(action, -1.0, 1.0)

        return actions

    def update(self, batch: ExperienceBatch) -> dict:
        assert isinstance(batch, tuple) and len(batch) == 5, "MATD3 expects OffPolicyExperienceBatch (tuple of 5 elements)"
        self.update_counter += 1
        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = batch

        # FIX: Use torch.from_numpy() for contiguous numpy arrays coming out of
        # the ring-buffer. This is a zero-copy path on the CPU side; the single
        # .to(device) call then does ONE async H2D DMA transfer per tensor instead
        # of allocating + copying inside torch.as_tensor() when the dtype already
        # matches (which it does because the ring-buffer stores float32).
        #
        # pin_memory=True on the ring-buffer arrays would make this non-blocking,
        # but that requires pinned allocation at buffer init time — left as a
        # further optimisation. For now this is already faster than the original.
        obs_tensor: torch.Tensor = torch.from_numpy(obs_batch).to(self.device, non_blocking=True)
        actions_tensor: torch.Tensor = torch.from_numpy(actions_batch).to(self.device, non_blocking=True)
        rewards_tensor: torch.Tensor = torch.from_numpy(rewards_batch).to(self.device, non_blocking=True)
        next_obs_tensor: torch.Tensor = torch.from_numpy(next_obs_batch).to(self.device, non_blocking=True)
        dones_tensor: torch.Tensor = torch.from_numpy(dones_batch).to(self.device, non_blocking=True)

        agent_losses: list[float] = []
        agent_critic_losses: list[float] = []

        # FIX: Compute ALL target next-actions outside the per-agent loop so
        # the list comprehension is outside the gradient-tracked region and we
        # avoid re-running target actors redundantly.
        with torch.no_grad():
            next_actions_list: list[torch.Tensor] = [self.target_actors[i](next_obs_tensor[:, i, :]) for i in range(self.num_agents)]
            clipped_noise: list[torch.Tensor] = [torch.clamp(torch.randn_like(a) * config.TARGET_POLICY_NOISE, -config.NOISE_CLIP, config.NOISE_CLIP) for a in next_actions_list]
            next_actions_list = [torch.clamp(next_actions_list[i] + clipped_noise[i], -1.0, 1.0) for i in range(self.num_agents)]
            next_actions_tensor: torch.Tensor = torch.stack(next_actions_list, dim=1)  # (B, N, action_dim)

        for agent_idx in range(self.num_agents):
            # Update Critic
            with torch.no_grad():
                # Compute target Q-value using the minimum of the two target critics
                target_q1: torch.Tensor = self.target_critics_1[agent_idx](next_obs_tensor, next_actions_tensor, agent_idx)
                target_q2: torch.Tensor = self.target_critics_2[agent_idx](next_obs_tensor, next_actions_tensor, agent_idx)
                target_q_min: torch.Tensor = torch.min(target_q1, target_q2)

                agent_reward: torch.Tensor = rewards_tensor[:, agent_idx].unsqueeze(1)
                agent_done: torch.Tensor = dones_tensor[:, agent_idx].unsqueeze(1)
                y: torch.Tensor = agent_reward + config.DISCOUNT_FACTOR * target_q_min * (1 - agent_done)

            # Update both critic networks
            # FIX: Zero both critic grads before computing losses, then do a
            # single backward pass per critic. Previously this was already one
            # backward each, but we make the zero_grad() call explicit before
            # the forward to avoid accumulating stale gradients if an exception
            # interrupted a previous step.
            self.critic_1_optimizers[agent_idx].zero_grad(set_to_none=True)  # set_to_none=True is faster than zeroing
            current_q1: torch.Tensor = self.critics_1[agent_idx](obs_tensor, actions_tensor, agent_idx)
            critic_1_loss: torch.Tensor = F.mse_loss(current_q1, y)
            critic_1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics_1[agent_idx].parameters(), config.MAX_GRAD_NORM)
            self.critic_1_optimizers[agent_idx].step()

            self.critic_2_optimizers[agent_idx].zero_grad(set_to_none=True)
            current_q2: torch.Tensor = self.critics_2[agent_idx](obs_tensor, actions_tensor, agent_idx)
            critic_2_loss: torch.Tensor = F.mse_loss(current_q2, y)
            critic_2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics_2[agent_idx].parameters(), config.MAX_GRAD_NORM)
            self.critic_2_optimizers[agent_idx].step()

            avg_critic_loss = (float(critic_1_loss.item()) + float(critic_2_loss.item())) / 2.0
            agent_critic_losses.append(avg_critic_loss)

        # Delayed Policy and Target Network Updates
        if self.update_counter % config.POLICY_UPDATE_FREQ == 0:
            for agent_idx in range(self.num_agents):
                # Update Actor
                pred_actions_tensor: torch.Tensor = actions_tensor.detach().clone()
                pred_actions_tensor[:, agent_idx, :] = self.actors[agent_idx](obs_tensor[:, agent_idx, :])

                self.actor_optimizers[agent_idx].zero_grad(set_to_none=True)
                actor_loss: torch.Tensor = -self.critics_1[agent_idx](obs_tensor, pred_actions_tensor, agent_idx).mean()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), config.MAX_GRAD_NORM)
                self.actor_optimizers[agent_idx].step()
                agent_losses.append(float(actor_loss.item()))

                # Soft update all target networks
                soft_update(self.target_actors[agent_idx], self.actors[agent_idx], config.UPDATE_FACTOR)
                soft_update(self.target_critics_1[agent_idx], self.critics_1[agent_idx], config.UPDATE_FACTOR)
                soft_update(self.target_critics_2[agent_idx], self.critics_2[agent_idx], config.UPDATE_FACTOR)

            for n in self.noise:
                n.decay()

        # Return averaged losses across all agents (same format as standard MATD3)
        return {
            "actor": float(np.mean(agent_losses)) if agent_losses else 0.0,
            "critic": float(np.mean(agent_critic_losses)),
        }

    def _init_target_networks(self) -> None:
        for actor, target_actor in zip(self.actors, self.target_actors):
            target_actor.load_state_dict(actor.state_dict())
        for critic1, target_critic1 in zip(self.critics_1, self.target_critics_1):
            target_critic1.load_state_dict(critic1.state_dict())
        for critic2, target_critic2 in zip(self.critics_2, self.target_critics_2):
            target_critic2.load_state_dict(critic2.state_dict())

    def reset(self) -> None:
        for n in self.noise:
            n.reset()

    def save(self, directory: str) -> None:
        for i in range(self.num_agents):
            torch.save(
                {
                    "actor": get_state_dict(self.actors[i]),
                    "critic_1": get_state_dict(self.critics_1[i]),
                    "critic_2": get_state_dict(self.critics_2[i]),
                    "target_actor": self.target_actors[i].state_dict(),
                    "target_critic_1": self.target_critics_1[i].state_dict(),
                    "target_critic_2": self.target_critics_2[i].state_dict(),
                    "actor_optimizer": self.actor_optimizers[i].state_dict(),
                    "critic_1_optimizer": self.critic_1_optimizers[i].state_dict(),
                    "critic_2_optimizer": self.critic_2_optimizers[i].state_dict(),
                    "noise_scale": self.noise[i].scale,
                },
                os.path.join(directory, f"agent_{i}.pth"),
            )
        update_counter_path: str = os.path.join(directory, "update_counter.txt")
        with open(update_counter_path, "w") as f:
            f.write(str(self.update_counter))

    def load(self, directory: str) -> None:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"❌ Model directory not found: {directory}")

        for i in range(self.num_agents):
            agent_path: str = os.path.join(directory, f"agent_{i}.pth")
            if not os.path.exists(agent_path):
                raise FileNotFoundError(f"❌ Model file not found: {agent_path}")
            checkpoint: dict = torch.load(agent_path, map_location=self.device, weights_only=True)
            load_safe(self.actors[i], checkpoint["actor"])
            load_safe(self.critics_1[i], checkpoint["critic_1"])
            load_safe(self.critics_2[i], checkpoint["critic_2"])
            self.target_actors[i].load_state_dict(checkpoint["target_actor"])
            self.target_critics_1[i].load_state_dict(checkpoint["target_critic_1"])
            self.target_critics_2[i].load_state_dict(checkpoint["target_critic_2"])
            self.actor_optimizers[i].load_state_dict(checkpoint["actor_optimizer"])
            self.critic_1_optimizers[i].load_state_dict(checkpoint["critic_1_optimizer"])
            self.critic_2_optimizers[i].load_state_dict(checkpoint["critic_2_optimizer"])
            self.noise[i].scale = checkpoint["noise_scale"]
        update_counter_path: str = os.path.join(directory, "update_counter.txt")
        if os.path.exists(update_counter_path):
            with open(update_counter_path, "r") as f:
                self.update_counter = int(f.read())
        else:
            self.update_counter = 0
            print(f"⚠️ Update counter file not found: {update_counter_path}. Setting update_counter to 0.")
