from __future__ import annotations

import torch
from torch import nn


class EdgeAggregationLayer(nn.Module):
    """A lightweight message-passing layer for homogeneous edge lists."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()

    def forward(
        self,
        dst_features: torch.Tensor,
        src_features: torch.Tensor,
        edges: list[tuple[int, int]],
    ) -> torch.Tensor:
        if dst_features.numel() == 0:
            return dst_features

        device = dst_features.device
        agg = torch.zeros_like(dst_features)
        counts = torch.zeros((dst_features.shape[0], 1), device=device, dtype=dst_features.dtype)

        if edges:
            src_idx = torch.tensor([src for src, _ in edges], device=device, dtype=torch.long)
            dst_idx = torch.tensor([dst for _, dst in edges], device=device, dtype=torch.long)
            agg.index_add_(0, dst_idx, src_features[src_idx])
            counts.index_add_(0, dst_idx, torch.ones((len(edges), 1), device=device, dtype=dst_features.dtype))

        mean_agg = agg / counts.clamp_min(1.0)
        updated = self.self_proj(dst_features) + self.msg_proj(mean_agg)
        return self.activation(self.norm(updated))


class HyperedgePoolingLayer(nn.Module):
    """Broadcasts pooled hyperedge features back to member nodes."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hyper_proj = nn.Linear(hidden_dim, hidden_dim)
        self.node_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, node_features: torch.Tensor, hyperedges: list[list[int]]) -> torch.Tensor:
        if node_features.numel() == 0 or not hyperedges:
            return node_features

        device = node_features.device
        agg = torch.zeros_like(node_features)
        counts = torch.zeros((node_features.shape[0], 1), device=device, dtype=node_features.dtype)

        for members in hyperedges:
            if not members:
                continue
            idx = torch.tensor(members, device=device, dtype=torch.long)
            pooled = node_features[idx].mean(dim=0, keepdim=True)
            agg.index_add_(0, idx, pooled.expand(len(members), -1))
            counts.index_add_(0, idx, torch.ones((len(members), 1), device=device, dtype=node_features.dtype))

        mean_agg = agg / counts.clamp_min(1.0)
        updated = self.node_proj(node_features) + self.hyper_proj(mean_agg)
        return self.activation(self.norm(updated))
