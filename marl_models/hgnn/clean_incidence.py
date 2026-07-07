from __future__ import annotations

import numpy as np
import torch
from torch import nn


class IncidenceHGNNLayer(nn.Module):
    """One incidence-matrix hypergraph message-passing layer."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.hyperedge_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, node_features: torch.Tensor, incidence_matrix: torch.Tensor) -> torch.Tensor:
        if node_features.numel() == 0 or incidence_matrix.numel() == 0:
            return node_features

        node_degrees = incidence_matrix.sum(dim=1, keepdim=True).clamp_min(1.0)
        edge_degrees = incidence_matrix.sum(dim=0, keepdim=True).transpose(0, 1).clamp_min(1.0)
        edge_features = incidence_matrix.transpose(0, 1).matmul(node_features) / edge_degrees
        node_messages = incidence_matrix.matmul(edge_features) / node_degrees
        updated = self.self_proj(node_features) + self.hyperedge_proj(node_messages)
        return self.activation(self.norm(updated))


class CleanIncidenceHGNN(nn.Module):
    """Minimal clean-mainline incidence-matrix HGNN.

    Inputs are task-only features and H[v, e]. UAV features, pair features,
    candidate masks, metrics, profiling, and rewards are intentionally outside
    this module.
    """

    def __init__(
        self,
        task_feature_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if task_feature_dim <= 0:
            raise ValueError("task_feature_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        layer_count = max(int(num_layers), 1)
        self.output_dim = int(output_dim if output_dim is not None else hidden_dim)
        self.input_proj = nn.Linear(int(task_feature_dim), int(hidden_dim))
        self.layers = nn.ModuleList(IncidenceHGNNLayer(int(hidden_dim)) for _ in range(layer_count))
        self.output_proj = nn.Linear(int(hidden_dim), self.output_dim)
        self.activation = nn.ReLU()

    def forward(
        self,
        task_features: torch.Tensor | np.ndarray,
        incidence_matrix: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        task_features_tensor = _as_float_tensor(task_features)
        if task_features_tensor.dim() != 2:
            raise ValueError("task_features must be a 2D tensor or array.")

        if task_features_tensor.shape[0] == 0:
            return task_features_tensor.new_zeros((0, self.output_dim))

        incidence_tensor = _as_float_tensor(
            incidence_matrix,
            device=task_features_tensor.device,
            dtype=task_features_tensor.dtype,
        )
        if incidence_tensor.dim() != 2:
            raise ValueError("incidence_matrix must be a 2D tensor or array.")
        if incidence_tensor.shape[0] != task_features_tensor.shape[0]:
            raise ValueError("incidence_matrix row count must match task_features row count.")

        hidden = self.activation(self.input_proj(task_features_tensor))
        if incidence_tensor.shape[1] == 0:
            return self.output_proj(hidden)

        for layer in self.layers:
            hidden = layer(hidden, incidence_tensor)
        return self.output_proj(hidden)


def _as_float_tensor(
    value: torch.Tensor | np.ndarray,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
        if device is not None or dtype is not None:
            tensor = tensor.to(device=device, dtype=dtype)
        return tensor.float() if dtype is None else tensor
    return torch.as_tensor(value, device=device, dtype=dtype or torch.float32)
