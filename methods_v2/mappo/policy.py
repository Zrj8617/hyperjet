"""独立的飞行或放置 actor，与每时隙一个值的集中式 critic。"""

import numpy as np
import torch
from torch import nn

from methods_v2.ers import sort_placements
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
    """S2 独立局部主干和候选打分器。"""

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


def checkpoint_args(values):
    """在一个入口补齐旧 S1 checkpoint 已知缺失的配置。"""
    from types import SimpleNamespace

    return SimpleNamespace(**(dict(flight="actor", placement="heft", user_info="full") | values))


class Policy:
    """只创建当前学习的 actor；critic 每个时隙计算一个团队价值。"""

    def __init__(self, builder, args):
        self.flight = args.flight
        if (args.flight, args.placement) not in (
            ("actor", "heft"), ("actor", "heft_uav"), ("follow", "actor"),
        ):
            raise ValueError("本轮只支持 S1 actor+heft/heft_uav 和 S2 follow+actor")
        self.actor = (FlightActor(builder.observation_dim, builder.settings.uav.count, args)
                      if self.flight == "actor" else
                      PlacementActor(builder.observation_dim, builder.task_dim, args))
        self.critic = Critic(builder.state_dim, args)

    @torch.no_grad()
    def act(self, features, generator, deterministic=False):
        """返回当前 actor 的环境动作、未排序 PPO 样本和团队价值。"""
        sample = {"observations": features.observations, "state": features.state}
        observations = torch.from_numpy(features.observations)
        value = self.critic(torch.from_numpy(features.state).unsqueeze(0)).item()
        if self.flight == "actor":
            distribution = self.actor(observations)
            flight = distribution.mode() if deterministic else distribution.sample(generator)
            sample.update(flight=flight.numpy(), log_probs=distribution.log_probs(flight).numpy())
            return np.clip(sample["flight"], -1, 1).astype(np.float64), sample, value

        sample.update(task_features=features.task_features,
                      available_actions=features.available_actions, owners=features.owners)
        distribution = self.actor(observations, torch.from_numpy(features.task_features),
                                  torch.from_numpy(features.owners),
                                  torch.from_numpy(features.available_actions))
        if distribution is None:
            sample.update(placements=np.empty((0, 1), dtype=np.int64),
                          log_probs=np.empty((0, 1), dtype=np.float32))
        else:
            actions = distribution.mode() if deterministic else distribution.sample(generator)
            sample.update(placements=actions.numpy(), log_probs=distribution.log_probs(actions).numpy())
        placements = [(key, None if action == 0 else int(action - 1))
                      for key, action in zip(features.keys, sample["placements"].ravel())]
        return sort_placements(placements, features.ranks), sample, value

    def evaluate_actions(self, batch):
        """飞行按 UAV、放置按子任务重算概率；价值始终按时隙。"""
        if self.flight == "actor":
            distribution = self.actor(batch["observations"])
            actions = batch["flight"]
        else:
            distribution = self.actor(batch["observations"], batch["task_features"],
                                      batch["owners"], batch["available_actions"])
            actions = batch["placements"]
        return (self.critic(batch["states"]), distribution.log_probs(actions),
                distribution.entropy().mean())
