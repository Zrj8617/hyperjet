"""PPO 更新；来源 on-policy algorithms/r_mappo/r_mappo.py、utils/util.py，commit de66d7a，MIT 许可见 LICENSE。"""

import torch
from torch import nn
from .valuenorm import ValueNorm


def huber_loss(error, delta):
    """保留官方 Huber 损失公式。"""
    a = (abs(error) <= delta).float()
    b = (abs(error) > delta).float()
    return a * error ** 2 / 2 + b * delta * (abs(error) - delta / 2)


class MAPPO:
    """共享 actor、集中式 critic 的裁剪 PPO 更新器。"""

    def __init__(self, policy, args):
        self.policy, self.args = policy, args
        self.value_normalizer = ValueNorm(1)
        self.actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=args.lr,
                                                eps=args.opti_eps, weight_decay=args.weight_decay)
        self.critic_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=args.critic_lr,
                                                 eps=args.opti_eps, weight_decay=args.weight_decay)

    def cal_value_loss(self, values, predictions, returns):
        """每个 minibatch 更新 ValueNorm，在归一化量纲里裁剪价值。"""
        clipped = predictions + (values - predictions).clamp(-self.args.clip_param, self.args.clip_param)
        self.value_normalizer.update(returns)
        targets = self.value_normalizer.normalize(returns)
        return torch.max(huber_loss(targets - values, self.args.huber_delta),
                         huber_loss(targets - clipped, self.args.huber_delta)).mean()

    def ppo_update(self, batch):
        """按官方顺序更新 actor 和 critic，同时记录 KL 与裁剪比例。"""
        values, log_probs, entropy = self.policy.evaluate_actions(batch)
        log_ratio = log_probs - batch["log_probs"]
        ratios = torch.exp(log_ratio)
        surr1 = ratios * batch["advantages"]
        surr2 = ratios.clamp(1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * batch["advantages"]
        policy_loss = -torch.min(surr1, surr2).sum(dim=-1, keepdim=True).mean()
        self.actor_optimizer.zero_grad()
        (policy_loss - entropy * self.args.entropy_coef).backward()
        actor_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.args.max_grad_norm)
        self.actor_optimizer.step()
        value_loss = self.cal_value_loss(values, batch["value_preds"], batch["returns"])
        self.critic_optimizer.zero_grad()
        (value_loss * self.args.value_loss_coef).backward()
        critic_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.args.max_grad_norm)
        self.critic_optimizer.step()
        return dict(value_loss=value_loss.item(), policy_loss=policy_loss.item(), entropy=entropy.item(),
                    approx_kl=((ratios - 1) - log_ratio).mean().item(),
                    clip_fraction=((ratios - 1).abs() > self.args.clip_param).float().mean().item(),
                    actor_grad_norm=actor_norm.item(), critic_grad_norm=critic_norm.item())

    def train(self, buffer, rng):
        """归一化整批优势，按配置执行多轮 minibatch 更新。"""
        self.policy.actor.train()
        self.policy.critic.train()
        buffer.compute_returns(self.value_normalizer)
        advantages = buffer.returns - self.value_normalizer.denormalize(buffer.value_preds)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        totals = {}
        for _ in range(self.args.ppo_epoch):
            for batch in buffer.feed_forward_generator(advantages, rng):
                for key, value in self.ppo_update(batch).items():
                    totals[key] = totals.get(key, 0.0) + value
        return {key: value / (self.args.ppo_epoch * self.args.num_mini_batch) for key, value in totals.items()}
