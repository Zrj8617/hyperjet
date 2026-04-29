from marl_models.base_model import OffPolicyExperienceBatch
import config
import torch
import torch.nn as nn
import numpy as np
from collections.abc import Generator


# FIX: Replace deque-based ReplayBuffer with numpy ring-buffer.
# The original deque caused O(n) random access: `self.buffer[i]` on a deque
# is O(n) per index, making each sample() call O(batch_size * buffer_size).
# A numpy ring-buffer gives O(1) indexing and enables vectorised batch slicing.
class ReplayBuffer:
    def __init__(self, max_size: int) -> None:
        self.max_size: int = max_size
        self.ptr: int = 0  # Points to next write position
        self.size: int = 0  # Current number of valid entries

        # Pre-allocate numpy arrays.
        self._obs: np.ndarray = np.zeros((self.max_size, config.NUM_UAVS, config.OBS_DIM_SINGLE), dtype=np.float32)  # Shape: (max_size, num_uavs, obs_dim)
        self._next_obs: np.ndarray = np.zeros((self.max_size, config.NUM_UAVS, config.OBS_DIM_SINGLE), dtype=np.float32)
        self._actions: np.ndarray = np.zeros((self.max_size, config.NUM_UAVS, config.ACTION_DIM), dtype=np.float32)
        self._rewards: np.ndarray = np.zeros((self.max_size, config.NUM_UAVS), dtype=np.float32)
        self._dones: np.ndarray = np.zeros((self.max_size, config.NUM_UAVS), dtype=np.float32)

    def add(self, obs_arr: np.ndarray, actions: np.ndarray, rewards: list[float], next_obs_arr: np.ndarray, done: bool) -> None:
        rewards_arr: np.ndarray = np.array(rewards, dtype=np.float32)
        dones_arr: np.ndarray = np.full(config.NUM_UAVS, float(done), dtype=np.float32)

        self._obs[self.ptr] = obs_arr
        self._actions[self.ptr] = actions
        self._rewards[self.ptr] = rewards_arr
        self._next_obs[self.ptr] = next_obs_arr
        self._dones[self.ptr] = dones_arr

        # Ring-buffer write pointer
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> OffPolicyExperienceBatch:
        """Sample a batch of experiences. O(1) random batch sampling via numpy fancy-indexing."""
        indices: np.ndarray = np.random.randint(0, self.size, size=batch_size)
        return (
            self._obs[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_obs[indices],
            self._dones[indices],
        )

    def __len__(self) -> int:
        return self.size


class RolloutBuffer:
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, buffer_size: int, device: str) -> None:
        self.num_agents: int = num_agents
        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim
        self.state_dim: int = obs_dim * num_agents
        self.buffer_size: int = buffer_size
        self.device: str = device

        # Initialize storage
        self.states: np.ndarray = np.zeros((buffer_size, self.state_dim), dtype=np.float32)
        self.observations: np.ndarray = np.zeros((buffer_size, num_agents, obs_dim), dtype=np.float32)
        self.actions: np.ndarray = np.zeros((buffer_size, num_agents, action_dim), dtype=np.float32)
        self.log_probs: np.ndarray = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.rewards: np.ndarray = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.dones: np.ndarray = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.values: np.ndarray = np.zeros((buffer_size, num_agents), dtype=np.float32)

        # For GAE calculation
        self.advantages: np.ndarray = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.returns: np.ndarray = np.zeros((buffer_size, num_agents), dtype=np.float32)

        self.step: int = 0

    def add(self, state: np.ndarray, obs: np.ndarray, actions: np.ndarray, log_probs: np.ndarray, rewards: list[float], done: bool, values: np.ndarray) -> None:
        if self.step >= self.buffer_size:
            raise ValueError("Rollout buffer overflow")
        self.states[self.step] = state
        self.observations[self.step] = obs
        self.actions[self.step] = actions
        self.log_probs[self.step] = log_probs
        self.rewards[self.step] = np.array(rewards, dtype=np.float32)
        self.dones[self.step] = np.full(config.NUM_UAVS, float(done), dtype=np.float32)
        self.values[self.step] = values

        self.step += 1

    def compute_returns_and_advantages(self, last_values: np.ndarray, gamma: float, gae_lambda: float) -> None:
        """Computes the advantages and returns for the collected trajectories using GAE."""
        last_gae_lam: float = 0.0
        for t in reversed(range(self.buffer_size)):
            next_values: np.ndarray = last_values if t == self.buffer_size - 1 else self.values[t + 1]
            delta: np.ndarray = self.rewards[t] + gamma * next_values * (1.0 - self.dones[t]) - self.values[t]
            self.advantages[t] = last_gae_lam = delta + gamma * gae_lambda * (1.0 - self.dones[t]) * last_gae_lam

        self.returns = self.advantages + self.values

        # Normalize advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_batches(self, batch_size: int) -> Generator[dict[str, torch.Tensor], None, None]:
        """A generator that yields mini-batches from the buffer."""
        num_samples: int = self.buffer_size * self.num_agents

        states: np.ndarray = np.repeat(self.states, self.num_agents, axis=0)
        agent_ids: np.ndarray = np.tile(np.arange(self.num_agents), self.buffer_size)
        obs: np.ndarray = self.observations.reshape(-1, self.obs_dim)
        actions: np.ndarray = self.actions.reshape(-1, self.action_dim)  # Reshape to (N, action_dim)
        log_probs: np.ndarray = self.log_probs.reshape(-1)
        advantages: np.ndarray = self.advantages.reshape(-1)
        returns: np.ndarray = self.returns.reshape(-1)
        values: np.ndarray = self.values.reshape(-1)

        # FIX: Convert entire epoch's data to GPU tensors ONCE before looping,
        # instead of calling torch.as_tensor(..., device=...) per mini-batch.
        # This eliminates repeated CPU→GPU transfers inside the hot path.
        t_states: torch.Tensor = torch.from_numpy(states).to(self.device)
        t_agent_ids: torch.Tensor = torch.from_numpy(agent_ids).to(self.device)
        t_obs: torch.Tensor = torch.from_numpy(obs).to(self.device)
        t_actions: torch.Tensor = torch.from_numpy(actions).to(self.device)
        t_log_probs: torch.Tensor = torch.from_numpy(log_probs).to(self.device)
        t_advantages: torch.Tensor = torch.from_numpy(advantages).to(self.device)
        t_returns: torch.Tensor = torch.from_numpy(returns).to(self.device)
        t_values: torch.Tensor = torch.from_numpy(values).to(self.device)

        indices: torch.Tensor = torch.randperm(num_samples, device=self.device)

        for start in range(0, num_samples, batch_size):
            idx: torch.Tensor = indices[start : start + batch_size]

            yield {
                "states": t_states[idx],
                "agent_ids": t_agent_ids[idx],
                "obs": t_obs[idx],
                "actions": t_actions[idx],
                "old_log_probs": t_log_probs[idx],
                "advantages": t_advantages[idx],
                "returns": t_returns[idx],
                "old_values": t_values[idx],
            }

    def clear(self) -> None:
        self.step = 0


class AttentionRolloutBuffer(RolloutBuffer):
    """Preserves (Batch, Num_Agents, Dim) structure required for Graph Attention."""

    def get_batches(self, batch_size: int):
        num_time_steps: int = self.buffer_size

        # FIX (same as above): upload all data to GPU once per epoch
        t_states: torch.Tensor = torch.from_numpy(self.states).to(self.device)
        t_obs: torch.Tensor = torch.from_numpy(self.observations).to(self.device)
        t_actions: torch.Tensor = torch.from_numpy(self.actions).to(self.device)
        t_log_probs: torch.Tensor = torch.from_numpy(self.log_probs).to(self.device)
        t_advantages: torch.Tensor = torch.from_numpy(self.advantages).to(self.device)
        t_returns: torch.Tensor = torch.from_numpy(self.returns).to(self.device)
        t_values: torch.Tensor = torch.from_numpy(self.values).to(self.device)

        indices: torch.Tensor = torch.randperm(num_time_steps, device=self.device)

        for start in range(0, num_time_steps, batch_size):
            idx: torch.Tensor = indices[start : start + batch_size]

            yield {
                "states": t_states[idx],
                "obs": t_obs[idx],
                "actions": t_actions[idx],
                "old_log_probs": t_log_probs[idx],
                "advantages": t_advantages[idx],
                "returns": t_returns[idx],
                "old_values": t_values[idx],
            }


def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float):
    """Performs a soft update of the target network's parameters."""
    with torch.no_grad():
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.copy_(tau * param + (1.0 - tau) * target_param)


class GaussianNoise:
    """Gaussian noise with decay for exploration."""

    def __init__(self) -> None:
        self.scale: float = config.INITIAL_NOISE_SCALE

    def sample(self) -> np.ndarray:
        return np.random.normal(0, self.scale, config.ACTION_DIM)

    def decay(self) -> None:
        self.scale = max(config.MIN_NOISE_SCALE, self.scale * config.NOISE_DECAY_RATE)

    def reset(self) -> None:
        self.scale = config.INITIAL_NOISE_SCALE


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Added orthogonal initialization for better training stability"""
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


def get_state_dict(model):  # Helper to strip the compile wrapper
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.state_dict()
    return model.state_dict()


def load_safe(model, state_dict):  # Helper to load into potentially compiled models
    if hasattr(model, "_orig_mod"):  # If compiled, try loading into _orig_mod first
        try:
            model._orig_mod.load_state_dict(state_dict)
            return
        except Exception:
            pass  # Fallback to loading directly
    model.load_state_dict(state_dict)
