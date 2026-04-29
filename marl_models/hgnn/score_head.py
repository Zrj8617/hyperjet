from __future__ import annotations

import torch
from torch import nn


class TaskUAVScoreHead(nn.Module):
    """Produces a scalar score for each feasible task-UAV pair."""

    def __init__(self, task_dim: int, uav_dim: int, pair_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(task_dim + uav_dim + pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, task_embeddings: torch.Tensor, uav_embeddings: torch.Tensor, pair_embeddings: torch.Tensor) -> torch.Tensor:
        if task_embeddings.numel() == 0 or uav_embeddings.numel() == 0 or pair_embeddings.numel() == 0:
            return torch.zeros((0,), device=task_embeddings.device)
        combined_features = torch.cat([task_embeddings, uav_embeddings, pair_embeddings], dim=-1)
        return self.net(combined_features).squeeze(-1)
