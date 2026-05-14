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


class MixedTaskUAVHyperedgePoolingLayer(nn.Module):
    """Pools mixed task-UAV hyperedges and broadcasts context to both node types."""

    def __init__(self, hidden_dim: int, pair_feature_dim: int) -> None:
        super().__init__()
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2 + pair_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.task_node_proj = nn.Linear(hidden_dim, hidden_dim)
        self.task_hyper_proj = nn.Linear(hidden_dim, hidden_dim)
        self.uav_node_proj = nn.Linear(hidden_dim, hidden_dim)
        self.uav_hyper_proj = nn.Linear(hidden_dim, hidden_dim)
        self.task_norm = nn.LayerNorm(hidden_dim)
        self.uav_norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()

    def forward(
        self,
        task_features: torch.Tensor,
        uav_features: torch.Tensor,
        hyperedges: list[tuple[list[int], list[int], torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if task_features.numel() == 0 or uav_features.numel() == 0 or not hyperedges:
            return task_features, uav_features

        device = task_features.device
        task_agg = torch.zeros_like(task_features)
        task_counts = torch.zeros((task_features.shape[0], 1), device=device, dtype=task_features.dtype)
        uav_agg = torch.zeros_like(uav_features)
        uav_counts = torch.zeros((uav_features.shape[0], 1), device=device, dtype=uav_features.dtype)

        for task_members, uav_members, pair_summary in hyperedges:
            if not task_members or not uav_members:
                continue
            task_idx = torch.tensor(task_members, device=device, dtype=torch.long)
            uav_idx = torch.tensor(uav_members, device=device, dtype=torch.long)
            pair_summary = pair_summary.to(device=device, dtype=task_features.dtype)
            if pair_summary.dim() == 1:
                pair_summary = pair_summary.unsqueeze(0)

            task_mean = task_features[task_idx].mean(dim=0, keepdim=True)
            uav_mean = uav_features[uav_idx].mean(dim=0, keepdim=True)
            context = self.context_proj(torch.cat([task_mean, uav_mean, pair_summary], dim=-1))

            task_agg.index_add_(0, task_idx, context.expand(len(task_members), -1))
            task_counts.index_add_(
                0,
                task_idx,
                torch.ones((len(task_members), 1), device=device, dtype=task_features.dtype),
            )
            uav_agg.index_add_(0, uav_idx, context.expand(len(uav_members), -1))
            uav_counts.index_add_(
                0,
                uav_idx,
                torch.ones((len(uav_members), 1), device=device, dtype=uav_features.dtype),
            )

        task_mean_agg = task_agg / task_counts.clamp_min(1.0)
        uav_mean_agg = uav_agg / uav_counts.clamp_min(1.0)
        task_updated = self.task_node_proj(task_features) + self.task_hyper_proj(task_mean_agg)
        uav_updated = self.uav_node_proj(uav_features) + self.uav_hyper_proj(uav_mean_agg)
        return self.activation(self.task_norm(task_updated)), self.activation(self.uav_norm(uav_updated))
