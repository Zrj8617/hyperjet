from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

import config
from marl_models.hgnn.layers import EdgeAggregationLayer, HyperedgePoolingLayer, MixedTaskUAVHyperedgePoolingLayer


@dataclass(slots=True)
class HGNNEncodingOutput:
    task_embeddings: torch.Tensor
    uav_embeddings: torch.Tensor


class PhaseOneHGNNEncoder(nn.Module):
    """Minimal heterogeneous graph encoder for phase-one scheduling."""

    def __init__(self, task_input_dim: int, uav_input_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.task_input = nn.Linear(task_input_dim, hidden_dim)
        self.uav_input = nn.Linear(uav_input_dim, hidden_dim)
        self.task_dep_layers = nn.ModuleList([EdgeAggregationLayer(hidden_dim) for _ in range(num_layers)])
        self.task_from_uav_layers = nn.ModuleList([EdgeAggregationLayer(hidden_dim) for _ in range(num_layers)])
        self.uav_from_task_layers = nn.ModuleList([EdgeAggregationLayer(hidden_dim) for _ in range(num_layers)])
        self.uav_peer_layers = nn.ModuleList([EdgeAggregationLayer(hidden_dim) for _ in range(num_layers)])
        self.service_domain_hyper_layers = nn.ModuleList(
            [MixedTaskUAVHyperedgePoolingLayer(hidden_dim, 9) for _ in range(num_layers)]
        )
        self.resource_competition_hyper_layers = nn.ModuleList(
            [MixedTaskUAVHyperedgePoolingLayer(hidden_dim, 9) for _ in range(num_layers)]
        )
        self.critical_support_hyper_layers = nn.ModuleList(
            [MixedTaskUAVHyperedgePoolingLayer(hidden_dim, 9) for _ in range(num_layers)]
        )
        self.critical_hyper_layers = nn.ModuleList([HyperedgePoolingLayer(hidden_dim) for _ in range(num_layers)])
        self.compute_attribute_hyper_layers = nn.ModuleList([HyperedgePoolingLayer(hidden_dim) for _ in range(num_layers)])
        self.communication_attribute_hyper_layers = nn.ModuleList([HyperedgePoolingLayer(hidden_dim) for _ in range(num_layers)])
        self.candidate_scarce_attribute_hyper_layers = nn.ModuleList([HyperedgePoolingLayer(hidden_dim) for _ in range(num_layers)])
        self.uav_hyper_layers = nn.ModuleList([HyperedgePoolingLayer(hidden_dim) for _ in range(num_layers)])
        self.task_out = nn.Linear(hidden_dim, config.TASK_EMB_DIM)
        self.uav_out = nn.Linear(hidden_dim, config.UAV_EMB_DIM)
        self.activation = nn.ReLU()

    def forward(
        self,
        task_features: torch.Tensor,
        uav_features: torch.Tensor,
        dependency_edges: list[tuple[int, int]],
        task_to_uav_edges: list[tuple[int, int]],
        uav_to_task_edges: list[tuple[int, int]],
        uav_uav_edges: list[tuple[int, int]],
        service_domain_hyperedges: list[tuple[list[int], list[int], torch.Tensor]],
        resource_competition_hyperedges: list[tuple[list[int], list[int], torch.Tensor]],
        critical_support_hyperedges: list[tuple[list[int], list[int], torch.Tensor]],
        critical_hyperedges: list[list[int]],
        compute_attribute_hyperedges: list[list[int]],
        communication_attribute_hyperedges: list[list[int]],
        candidate_scarce_attribute_hyperedges: list[list[int]],
        uav_hyperedges: list[list[int]],
    ) -> HGNNEncodingOutput:
        task_hidden = self.activation(self.task_input(task_features))
        uav_hidden = self.activation(self.uav_input(uav_features))

        undirected_uav_edges = uav_uav_edges + [(dst, src) for src, dst in uav_uav_edges]

        for layer_idx in range(len(self.task_dep_layers)):
            task_hidden = self.task_dep_layers[layer_idx](task_hidden, task_hidden, dependency_edges)
            task_hidden = self.task_from_uav_layers[layer_idx](task_hidden, uav_hidden, uav_to_task_edges)
            uav_hidden = self.uav_from_task_layers[layer_idx](uav_hidden, task_hidden, task_to_uav_edges)
            uav_hidden = self.uav_peer_layers[layer_idx](uav_hidden, uav_hidden, undirected_uav_edges)
            task_hidden, uav_hidden = self.service_domain_hyper_layers[layer_idx](
                task_hidden,
                uav_hidden,
                service_domain_hyperedges,
            )
            task_hidden, uav_hidden = self.resource_competition_hyper_layers[layer_idx](
                task_hidden,
                uav_hidden,
                resource_competition_hyperedges,
            )
            task_hidden, uav_hidden = self.critical_support_hyper_layers[layer_idx](
                task_hidden,
                uav_hidden,
                critical_support_hyperedges,
            )
            task_hidden = self.critical_hyper_layers[layer_idx](task_hidden, critical_hyperedges)
            task_hidden = self.compute_attribute_hyper_layers[layer_idx](task_hidden, compute_attribute_hyperedges)
            task_hidden = self.communication_attribute_hyper_layers[layer_idx](
                task_hidden,
                communication_attribute_hyperedges,
            )
            task_hidden = self.candidate_scarce_attribute_hyper_layers[layer_idx](
                task_hidden,
                candidate_scarce_attribute_hyperedges,
            )
            uav_hidden = self.uav_hyper_layers[layer_idx](uav_hidden, uav_hyperedges)

        return HGNNEncodingOutput(
            task_embeddings=self.task_out(task_hidden),
            uav_embeddings=self.uav_out(uav_hidden),
        )
