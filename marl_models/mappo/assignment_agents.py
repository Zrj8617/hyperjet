from __future__ import annotations

import torch
from torch import nn


class AssignmentActor(nn.Module):
    """Scores one feasible task-UAV assignment candidate."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, candidate_inputs: torch.Tensor) -> torch.Tensor:
        if candidate_inputs.numel() == 0:
            return torch.zeros((0,), dtype=candidate_inputs.dtype, device=candidate_inputs.device)
        return self.net(candidate_inputs).squeeze(-1)


class AssignmentCritic(nn.Module):
    """Produces one shared step-level value from pooled graph embeddings."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_inputs: torch.Tensor) -> torch.Tensor:
        return self.net(global_inputs).squeeze(-1)
