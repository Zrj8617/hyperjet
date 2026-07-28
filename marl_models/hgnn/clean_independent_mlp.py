from __future__ import annotations

import numpy as np
import torch
from torch import nn


class CleanIndependentTaskMLP(nn.Module):
    """Encode each task independently without graph or hypergraph messages."""

    def __init__(
        self,
        task_feature_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        if int(task_feature_dim) <= 0:
            raise ValueError("task_feature_dim must be positive.")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive.")
        self.output_dim = int(output_dim if output_dim is not None else hidden_dim)
        self.input_proj = nn.Linear(int(task_feature_dim), int(hidden_dim))
        self.output_proj = nn.Linear(int(hidden_dim), self.output_dim)
        self.activation = nn.ReLU()

    def forward(
        self,
        task_features: torch.Tensor | np.ndarray,
        incidence_matrix: torch.Tensor | np.ndarray | None = None,
        hyperedge_type_ids: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        del incidence_matrix
        del hyperedge_type_ids
        if isinstance(task_features, torch.Tensor):
            features = task_features.float()
        else:
            features = torch.as_tensor(task_features, dtype=torch.float32)
        if features.dim() != 2:
            raise ValueError("task_features must be a 2D tensor or array.")
        if features.shape[0] == 0:
            return features.new_zeros((0, self.output_dim))
        return self.output_proj(self.activation(self.input_proj(features)))
