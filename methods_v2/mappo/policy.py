"""S1 飞行 actor、预留的放置 actor 与集中式 critic。"""

import numpy as np
import torch
from torch import nn

from methods_v2.learning.distributions import Categorical, DiagGaussian, FixedNormal
from methods_v2.learning.networks import Critic, MLPBase


class FlightActor(nn.Module):
    """按配置共享飞行参数，只输出两维连续动作。"""

    def __init__(self, observation_dim, uav_count, args):
        super().__init__()
        self.mode = args.flight_actor
        self.uav_count = uav_count
        if self.mode == "per_uav":
            self.bases = nn.ModuleList([
                MLPBase(observation_dim, args) for _ in range(uav_count)
            ])
            self.flights = nn.ModuleList([
                DiagGaussian(args.hidden_size, 2, args.gain) for _ in range(uav_count)
            ])
        else:
            inputs = observation_dim + uav_count if self.mode == "shared_id" else observation_dim
            self.base = MLPBase(inputs, args)
            self.flight = DiagGaussian(args.hidden_size, 2, args.gain)

    def forward(self, observations):
        """返回五架 UAV 的联合批量对角高斯分布。"""
        if self.mode == "per_uav":
            distributions = [
                flight(base(observations[index::self.uav_count]))
                for index, (base, flight) in enumerate(zip(self.bases, self.flights))
            ]
            return FixedNormal(
                torch.stack([distribution.mean for distribution in distributions], dim=1).reshape(-1, 2),
                torch.stack([distribution.stddev for distribution in distributions], dim=1).reshape(-1, 2),
            )
        inputs = observations
        if self.mode == "shared_id":
            identity = torch.eye(
                self.uav_count, dtype=observations.dtype, device=observations.device,
            ).repeat(observations.shape[0] // self.uav_count, 1)
            inputs = torch.cat((observations, identity), dim=-1)
        return self.flight(self.base(inputs))

    def log_std_values(self):
        """以五架 UAV × 两分量的统一形状返回 log_std。"""
        if self.mode == "per_uav":
            return torch.stack([flight.log_std for flight in self.flights])
        return self.flight.log_std.unsqueeze(0).expand(self.uav_count, -1)


class PlacementActor(nn.Module):
    """为 S2 保留的独立局部主干和候选打分器。"""

    def __init__(self, observation_dim, task_dim, args):
        super().__init__()
        self.base = MLPBase(observation_dim, args)
        self.scorer = MLPBase(args.hidden_size + task_dim, args)
        self.placement = Categorical(args.hidden_size, args.gain)

    def forward(self, observations, task_features, owners, available_actions):
        """按任务所属 UAV 取局部编码并对候选执行节点打分。"""
        if len(owners) == 0:
            return None
        hidden = self.base(observations)
        task_hidden = hidden.index_select(0, owners).unsqueeze(1).expand(
            -1, task_features.shape[1], -1,
        )
        scores = self.scorer(torch.cat((task_hidden, task_features), dim=-1))
        return self.placement(scores, available_actions)


class Policy:
    """S1 只创建飞行 actor；critic 每个时隙计算一个团队价值。"""

    def __init__(self, builder, args):
        self.actor = FlightActor(builder.observation_dim, builder.settings.uav.count, args)
        self.critic = Critic(builder.state_dim, args)

    @torch.no_grad()
    def act(self, features, generator, deterministic=False):
        """返回环境飞行动作、flight-only PPO 样本和团队价值。"""
        sample = {
            "observations": features.observations,
            "state": features.state,
        }
        observations = torch.from_numpy(features.observations)
        distribution = self.actor(observations)
        flight = distribution.mode() if deterministic else distribution.sample(generator)
        sample.update(
            flight=flight.numpy(),
            log_probs=distribution.log_probs(flight).numpy(),
        )
        value = self.critic(torch.from_numpy(features.state).unsqueeze(0)).item()
        return np.clip(sample["flight"], -1, 1).astype(np.float64), sample, value

    def evaluate_actions(self, batch):
        """只用飞行动作重算 PPO 概率、熵和团队价值。"""
        distribution = self.actor(batch["observations"])
        log_probs = distribution.log_probs(batch["flight"])
        return self.critic(batch["states"]), log_probs, distribution.entropy().mean()
