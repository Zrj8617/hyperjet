from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

import config
from environment.graph_builder import HeteroGraphSnapshot
from marl_models.hgnn.encoder import PhaseOneHGNNEncoder
from marl_models.hgnn.score_head import TaskUAVScoreHead


@dataclass(slots=True)
class GraphSchedulingOutput:
    task_embeddings: np.ndarray
    uav_embeddings: np.ndarray
    edge_scores: dict[tuple[str, int], float]


@dataclass(slots=True)
class GraphSchedulingTorchOutput:
    task_embeddings: torch.Tensor
    uav_embeddings: torch.Tensor
    edge_keys: list[tuple[str, int]]
    edge_scores: torch.Tensor


class PhaseOneGraphScheduler(nn.Module):
    """Encodes the current graph snapshot and scores feasible task-UAV edges."""

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.device = torch.device(device)
        self.pair_feature_dim = 9
        self.encoder = PhaseOneHGNNEncoder(
            task_input_dim=config.DAG_TASK_FEATURE_DIM,
            uav_input_dim=7,
            hidden_dim=config.HGNN_HIDDEN_DIM,
            num_layers=config.HGNN_NUM_LAYERS,
        )
        self.score_head = TaskUAVScoreHead(config.TASK_EMB_DIM, config.UAV_EMB_DIM, self.pair_feature_dim, config.HGNN_HIDDEN_DIM)
        self.to(self.device)
        self.eval()

    def _prepare_graph_inputs(
        self, snapshot: HeteroGraphSnapshot
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, int],
        dict[int, int],
        list[tuple[int, int]],
        list[tuple[int, int]],
        list[tuple[int, int]],
        list[tuple[int, int]],
        list[list[int]],
        list[list[int]],
    ]:
        task_features = torch.as_tensor(snapshot.task_features, dtype=torch.float32, device=self.device)
        uav_features = torch.as_tensor(snapshot.uav_features, dtype=torch.float32, device=self.device)
        edge_pair_features = torch.as_tensor(snapshot.task_uav_edge_features, dtype=torch.float32, device=self.device)

        task_index = {task_id: idx for idx, task_id in enumerate(snapshot.task_ids)}
        uav_index = {uav_id: idx for idx, uav_id in enumerate(snapshot.uav_ids)}

        dependency_edges = [(task_index[src], task_index[dst]) for src, dst in snapshot.dependency_edges if src in task_index and dst in task_index]
        task_to_uav_edges = [(task_index[task_id], uav_index[uav_id]) for task_id, uav_id in snapshot.task_uav_edges if task_id in task_index and uav_id in uav_index]
        uav_to_task_edges = [(uav_index[uav_id], task_index[task_id]) for task_id, uav_id in snapshot.task_uav_edges if task_id in task_index and uav_id in uav_index]
        uav_uav_edges = [(uav_index[src], uav_index[dst]) for src, dst in snapshot.uav_uav_edges if src in uav_index and dst in uav_index]

        task_hyperedges: list[list[int]] = []
        uav_hyperedges: list[list[int]] = []
        for task_ids, uav_ids in snapshot.collaborative_hyperedges:
            members = [task_index[task_id] for task_id in task_ids if task_id in task_index]
            if members:
                task_hyperedges.append(members)
            uav_members = [uav_index[uav_id] for uav_id in uav_ids if uav_id in uav_index]
            if uav_members:
                uav_hyperedges.append(uav_members)
        for task_ids in snapshot.critical_hyperedges:
            members = [task_index[task_id] for task_id in task_ids if task_id in task_index]
            if members:
                task_hyperedges.append(members)
        for task_ids in snapshot.attribute_hyperedges:
            members = [task_index[task_id] for task_id in task_ids if task_id in task_index]
            if members:
                task_hyperedges.append(members)

        return (
            task_features,
            uav_features,
            edge_pair_features,
            task_index,
            uav_index,
            dependency_edges,
            task_to_uav_edges,
            uav_to_task_edges,
            uav_uav_edges,
            task_hyperedges,
            uav_hyperedges,
        )

    def forward_graph(self, snapshot: HeteroGraphSnapshot) -> GraphSchedulingTorchOutput:
        (
            task_features,
            uav_features,
            edge_pair_features,
            task_index,
            uav_index,
            dependency_edges,
            task_to_uav_edges,
            uav_to_task_edges,
            uav_uav_edges,
            task_hyperedges,
            uav_hyperedges,
        ) = self._prepare_graph_inputs(snapshot)

        encoded = self.encoder(
            task_features=task_features,
            uav_features=uav_features,
            dependency_edges=dependency_edges,
            task_to_uav_edges=task_to_uav_edges,
            uav_to_task_edges=uav_to_task_edges,
            uav_uav_edges=uav_uav_edges,
            task_hyperedges=task_hyperedges,
            uav_hyperedges=uav_hyperedges,
        )

        edge_keys: list[tuple[str, int]] = []
        edge_scores_tensor = torch.zeros((0,), dtype=torch.float32, device=self.device)
        if snapshot.task_uav_edges:
            task_emb = []
            uav_emb = []
            pair_emb = []
            for edge_idx, (task_id, uav_id) in enumerate(snapshot.task_uav_edges):
                if task_id not in task_index or uav_id not in uav_index:
                    continue
                task_emb.append(encoded.task_embeddings[task_index[task_id]])
                uav_emb.append(encoded.uav_embeddings[uav_index[uav_id]])
                pair_emb.append(edge_pair_features[edge_idx])
                edge_keys.append((task_id, uav_id))
            if edge_keys:
                edge_scores_tensor = self.score_head(
                    torch.stack(task_emb, dim=0),
                    torch.stack(uav_emb, dim=0),
                    torch.stack(pair_emb, dim=0),
                )

        return GraphSchedulingTorchOutput(
            task_embeddings=encoded.task_embeddings,
            uav_embeddings=encoded.uav_embeddings,
            edge_keys=edge_keys,
            edge_scores=edge_scores_tensor,
        )

    @torch.no_grad()
    def score_graph(self, snapshot: HeteroGraphSnapshot) -> GraphSchedulingOutput:
        torch_output = self.forward_graph(snapshot)
        edge_scores = {
            edge_key: float(score)
            for edge_key, score in zip(torch_output.edge_keys, torch_output.edge_scores.detach().cpu().tolist())
        }

        return GraphSchedulingOutput(
            task_embeddings=torch_output.task_embeddings.detach().cpu().numpy(),
            uav_embeddings=torch_output.uav_embeddings.detach().cpu().numpy(),
            edge_scores=edge_scores,
        )
