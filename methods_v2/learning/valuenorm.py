"""值归一化；来源 on-policy onpolicy/utils/valuenorm.py，commit de66d7a，MIT 许可见 LICENSE。"""

import torch
from torch import nn


class ValueNorm(nn.Module):
    """维护去偏均值和方差，对价值目标作归一化。"""

    def __init__(self, input_shape, norm_axes=1, beta=0.99999, epsilon=1e-5):
        super().__init__()
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def running_mean_var(self):
        """返回去偏均值和方差，保留官方方差下限。"""
        mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        return mean, (mean_sq - mean ** 2).clamp(min=1e-2)

    @torch.no_grad()
    def update(self, input_vector):
        """按官方批更新公式更新一阶、二阶矩。"""
        values = torch.as_tensor(input_vector, dtype=torch.float32)
        batch_mean = values.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (values ** 2).mean(dim=tuple(range(self.norm_axes)))
        self.running_mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.running_mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 - self.beta)

    def normalize(self, input_vector):
        """把原量纲目标变成归一化目标。"""
        values = torch.as_tensor(input_vector, dtype=torch.float32)
        mean, var = self.running_mean_var()
        return (values - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]

    def denormalize(self, input_vector):
        """恢复原量纲，并按官方接口返回 NumPy 数组。"""
        values = torch.as_tensor(input_vector, dtype=torch.float32)
        mean, var = self.running_mean_var()
        return (values * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]).numpy()
