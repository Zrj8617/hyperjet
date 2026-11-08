"""MLP 与价值头；来源 on-policy algorithms/utils/{mlp,util}.py、r_mappo/algorithm/r_actor_critic.py，commit de66d7a，MIT 许可见 LICENSE。"""

from torch import nn


def init_linear(inputs, outputs, gain=1.0):
    """按官方方式正交初始化线性层，偏置为零。"""
    layer = nn.Linear(inputs, outputs)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)
    return layer


class MLPLayer(nn.Module):
    """一层输入映射加 layer_N 层隐藏映射，每层使用 ReLU 和 LayerNorm。"""

    def __init__(self, input_dim, hidden_size, layer_N):
        super().__init__()
        gain = nn.init.calculate_gain("relu")
        self.fc1 = nn.Sequential(init_linear(input_dim, hidden_size, gain), nn.ReLU(), nn.LayerNorm(hidden_size))
        self.fc2 = nn.ModuleList([
            nn.Sequential(init_linear(hidden_size, hidden_size, gain), nn.ReLU(), nn.LayerNorm(hidden_size))
            for _ in range(layer_N)
        ])

    def forward(self, x):
        """依次应用输入层和隐藏层。"""
        x = self.fc1(x)
        for layer in self.fc2:
            x = layer(x)
        return x


class MLPBase(nn.Module):
    """输入先做 LayerNorm，再通过官方 MLP 主干。"""

    def __init__(self, input_dim, args):
        super().__init__()
        self.feature_norm = nn.LayerNorm(input_dim)
        self.mlp = MLPLayer(input_dim, args.hidden_size, args.layer_N)

    def forward(self, x):
        """编码一批特征。"""
        return self.mlp(self.feature_norm(x))


class Critic(nn.Module):
    """从全局状态输出一个归一化团队价值。"""

    def __init__(self, state_dim, args):
        super().__init__()
        self.base = MLPBase(state_dim, args)
        self.v_out = init_linear(args.hidden_size, 1)

    def forward(self, states):
        """每条全局状态只计算一个价值。"""
        return self.v_out(self.base(states))
