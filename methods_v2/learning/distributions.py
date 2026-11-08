"""动作分布；来源 on-policy algorithms/utils/distributions.py，commit de66d7a，MIT 许可见 LICENSE。"""

import torch
from torch import nn
from .networks import init_linear


class FixedCategorical(torch.distributions.Categorical):
    """统一离散动作形状，并用方法层自己的生成器采样。"""

    def sample(self, generator):
        """为每条分布抽一个动作。"""
        return torch.multinomial(self.probs, 1, generator=generator)

    def log_probs(self, actions):
        """返回每条离散动作的对数概率。"""
        return super().log_prob(actions.squeeze(-1)).unsqueeze(-1)

    def mode(self):
        """返回概率最大的动作。"""
        return self.probs.argmax(dim=-1, keepdim=True)


class FixedNormal(torch.distributions.Normal):
    """对角高斯，各连续分量的对数概率和熵相加。"""

    @torch.no_grad()
    def sample(self, generator):
        """用方法层自己的生成器抽取未裁剪动作。"""
        return self.mean + self.stddev * torch.randn(self.mean.shape, generator=generator)

    def log_probs(self, actions):
        """对连续动作分量求和。"""
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        """对连续动作分量的熵求和。"""
        return super().entropy().sum(-1)

    def mode(self):
        """确定性动作为高斯均值。"""
        return self.mean


class Categorical(nn.Module):
    """共享标量头为各候选打分，屏蔽不可选的执行节点。"""

    def __init__(self, inputs, gain):
        super().__init__()
        self.linear = init_linear(inputs, 1, gain)

    def forward(self, x, available_actions):
        """将每个候选的特征映射成一个 logit。"""
        logits = self.linear(x).squeeze(-1)
        return FixedCategorical(logits=logits.masked_fill(~available_actions, -1e10))


class DiagGaussian(nn.Module):
    """线性均值头和初值为零的可学习 log_std。"""

    def __init__(self, inputs, outputs, gain):
        super().__init__()
        self.fc_mean = init_linear(inputs, outputs, gain)
        self.log_std = nn.Parameter(torch.zeros(outputs))

    def forward(self, x):
        """返回连续动作的对角高斯分布。"""
        return FixedNormal(self.fc_mean(x), self.log_std.exp())
