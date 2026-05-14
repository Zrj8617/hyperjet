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
        self.pair_feature_dim = config.BASE_TASK_UAV_PAIR_FEATURE_DIM
        self.score_pair_feature_dim = self.pair_feature_dim + (
            config.PAIR_HYPEREDGE_SCORE_FEATURE_DIM
            if config.USE_PAIR_HYPEREDGE_SCORE_FEATURES
            else 0
        )
        self.encoder = PhaseOneHGNNEncoder(
            task_input_dim=config.DAG_TASK_FEATURE_DIM,
            uav_input_dim=7,
            hidden_dim=config.HGNN_HIDDEN_DIM,
            num_layers=config.HGNN_NUM_LAYERS,
        )
        self.score_head = TaskUAVScoreHead(
            config.TASK_EMB_DIM,
            config.UAV_EMB_DIM,
            self.score_pair_feature_dim,
            config.HGNN_HIDDEN_DIM,
        )
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
        list[tuple[list[int], list[int], torch.Tensor]],
        list[tuple[list[int], list[int], torch.Tensor]],
        list[tuple[list[int], list[int], torch.Tensor]],
        list[list[int]],
        list[list[int]],
        list[list[int]],
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

        service_domain_hyperedges: list[tuple[list[int], list[int], torch.Tensor]] = []
        resource_competition_hyperedges: list[tuple[list[int], list[int], torch.Tensor]] = []
        critical_support_hyperedges: list[tuple[list[int], list[int], torch.Tensor]] = []
        critical_hyperedges: list[list[int]] = []
        compute_attribute_hyperedges: list[list[int]] = []
        communication_attribute_hyperedges: list[list[int]] = []
        candidate_scarce_attribute_hyperedges: list[list[int]] = []
        uav_hyperedges: list[list[int]] = []
        edge_feature_map = {
            (task_id, uav_id): edge_pair_features[edge_idx]
            for edge_idx, (task_id, uav_id) in enumerate(snapshot.task_uav_edges)
        }

        service_domain_groups = snapshot.service_domain_hyperedges
        resource_competition_groups = snapshot.resource_competition_hyperedges
        if not service_domain_groups and not resource_competition_groups:
            service_domain_groups = snapshot.collaborative_hyperedges

        for task_ids, uav_ids in service_domain_groups:
            mixed_hyperedge = self._prepare_mixed_hyperedge(
                task_ids,
                uav_ids,
                task_index,
                uav_index,
                edge_feature_map,
            )
            if mixed_hyperedge is not None:
                service_domain_hyperedges.append(mixed_hyperedge)
        for task_ids, uav_ids in resource_competition_groups:
            mixed_hyperedge = self._prepare_mixed_hyperedge(
                task_ids,
                uav_ids,
                task_index,
                uav_index,
                edge_feature_map,
            )
            if mixed_hyperedge is not None:
                resource_competition_hyperedges.append(mixed_hyperedge)
        for task_ids, uav_ids in snapshot.critical_support_hyperedges:
            mixed_hyperedge = self._prepare_mixed_hyperedge(
                task_ids,
                uav_ids,
                task_index,
                uav_index,
                edge_feature_map,
            )
            if mixed_hyperedge is not None:
                critical_support_hyperedges.append(mixed_hyperedge)
        for task_ids in snapshot.critical_hyperedges:
            members = [task_index[task_id] for task_id in task_ids if task_id in task_index]
            if members:
                critical_hyperedges.append(members)
        for task_ids in snapshot.compute_attribute_hyperedges:
            members = [task_index[task_id] for task_id in task_ids if task_id in task_index]
            if members:
                compute_attribute_hyperedges.append(members)
        for task_ids in snapshot.communication_attribute_hyperedges:
            members = [task_index[task_id] for task_id in task_ids if task_id in task_index]
            if members:
                communication_attribute_hyperedges.append(members)
        for task_ids in snapshot.candidate_scarce_attribute_hyperedges:
            members = [task_index[task_id] for task_id in task_ids if task_id in task_index]
            if members:
                candidate_scarce_attribute_hyperedges.append(members)

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
            service_domain_hyperedges,
            resource_competition_hyperedges,
            critical_support_hyperedges,
            critical_hyperedges,
            compute_attribute_hyperedges,
            communication_attribute_hyperedges,
            candidate_scarce_attribute_hyperedges,
            uav_hyperedges,
        )

    def _prepare_mixed_hyperedge(
        self,
        task_ids: list[str],
        uav_ids: list[int],
        task_index: dict[str, int],
        uav_index: dict[int, int],
        edge_feature_map: dict[tuple[str, int], torch.Tensor],
    ) -> tuple[list[int], list[int], torch.Tensor] | None:
        task_members = [task_index[task_id] for task_id in task_ids if task_id in task_index]
        uav_members = [uav_index[uav_id] for uav_id in uav_ids if uav_id in uav_index]
        if not task_members or not uav_members:
            return None

        pair_features = [
            edge_feature_map[(task_id, uav_id)]
            for task_id in task_ids
            for uav_id in uav_ids
            if (task_id, uav_id) in edge_feature_map
        ]
        if pair_features:
            pair_summary = torch.stack(pair_features, dim=0).mean(dim=0)
        else:
            pair_summary = torch.zeros((self.pair_feature_dim,), dtype=torch.float32, device=self.device)
        return task_members, uav_members, pair_summary

    def _build_pair_hyperedge_score_features(
        self,
        snapshot: HeteroGraphSnapshot,
    ) -> dict[tuple[str, int], torch.Tensor]:
        if not config.USE_PAIR_HYPEREDGE_SCORE_FEATURES:
            return {}

        service_pair_counts: dict[tuple[str, int], float] = {}
        resource_pair_counts: dict[tuple[str, int], float] = {}
        service_task_counts: dict[str, float] = {}
        resource_task_counts: dict[str, float] = {}
        service_uav_counts: dict[int, float] = {}
        resource_uav_counts: dict[int, float] = {}
        critical_tasks: set[str] = set()
        critical_pair_counts: dict[tuple[str, int], float] = {}

        for task_ids, uav_ids in snapshot.service_domain_hyperedges:
            for task_id in task_ids:
                service_task_counts[task_id] = service_task_counts.get(task_id, 0.0) + 1.0
            for uav_id in uav_ids:
                service_uav_counts[uav_id] = service_uav_counts.get(uav_id, 0.0) + 1.0
            for task_id in task_ids:
                for uav_id in uav_ids:
                    key = (task_id, uav_id)
                    service_pair_counts[key] = service_pair_counts.get(key, 0.0) + 1.0

        for task_ids, uav_ids in snapshot.resource_competition_hyperedges:
            for task_id in task_ids:
                resource_task_counts[task_id] = resource_task_counts.get(task_id, 0.0) + 1.0
            for uav_id in uav_ids:
                resource_uav_counts[uav_id] = resource_uav_counts.get(uav_id, 0.0) + 1.0
            for task_id in task_ids:
                for uav_id in uav_ids:
                    key = (task_id, uav_id)
                    resource_pair_counts[key] = resource_pair_counts.get(key, 0.0) + 1.0

        for task_ids in snapshot.critical_hyperedges:
            critical_tasks.update(task_ids)
        for task_ids, uav_ids in snapshot.critical_support_hyperedges:
            critical_tasks.update(task_ids)
            for task_id in task_ids:
                for uav_id in uav_ids:
                    key = (task_id, uav_id)
                    critical_pair_counts[key] = critical_pair_counts.get(key, 0.0) + 1.0

        critical_feasible_counts: dict[int, float] = {}
        for task_id, uav_id in snapshot.task_uav_edges:
            if task_id in critical_tasks or (task_id, uav_id) in critical_pair_counts:
                critical_feasible_counts[uav_id] = critical_feasible_counts.get(uav_id, 0.0) + 1.0

        service_den = float(max(len(snapshot.service_domain_hyperedges), 1))
        resource_den = float(max(len(snapshot.resource_competition_hyperedges), 1))
        critical_den = float(max(len(critical_tasks), 1))
        feature_map: dict[tuple[str, int], torch.Tensor] = {}
        for task_id, uav_id in snapshot.task_uav_edges:
            values = [
                min(service_pair_counts.get((task_id, uav_id), 0.0), 1.0),
                min(resource_pair_counts.get((task_id, uav_id), 0.0), 1.0),
                min(critical_pair_counts.get((task_id, uav_id), 0.0), 1.0),
                min(service_uav_counts.get(uav_id, 0.0) / service_den, 1.0),
                min(resource_uav_counts.get(uav_id, 0.0) / resource_den, 1.0),
                min(critical_feasible_counts.get(uav_id, 0.0) / critical_den, 1.0),
                min(service_task_counts.get(task_id, 0.0) / service_den, 1.0),
                min(resource_task_counts.get(task_id, 0.0) / resource_den, 1.0),
            ]
            feature_map[(task_id, uav_id)] = torch.tensor(
                values,
                dtype=torch.float32,
                device=self.device,
            )
        return feature_map

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
            service_domain_hyperedges,
            resource_competition_hyperedges,
            critical_support_hyperedges,
            critical_hyperedges,
            compute_attribute_hyperedges,
            communication_attribute_hyperedges,
            candidate_scarce_attribute_hyperedges,
            uav_hyperedges,
        ) = self._prepare_graph_inputs(snapshot)

        encoded = self.encoder(
            task_features=task_features,
            uav_features=uav_features,
            dependency_edges=dependency_edges,
            task_to_uav_edges=task_to_uav_edges,
            uav_to_task_edges=uav_to_task_edges,
            uav_uav_edges=uav_uav_edges,
            service_domain_hyperedges=service_domain_hyperedges,
            resource_competition_hyperedges=resource_competition_hyperedges,
            critical_support_hyperedges=critical_support_hyperedges,
            critical_hyperedges=critical_hyperedges,
            compute_attribute_hyperedges=compute_attribute_hyperedges,
            communication_attribute_hyperedges=communication_attribute_hyperedges,
            candidate_scarce_attribute_hyperedges=candidate_scarce_attribute_hyperedges,
            uav_hyperedges=uav_hyperedges,
        )

        pair_hyperedge_feature_map = self._build_pair_hyperedge_score_features(snapshot)
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
                pair_feature = edge_pair_features[edge_idx]
                if config.USE_PAIR_HYPEREDGE_SCORE_FEATURES:
                    pair_feature = torch.cat(
                        [
                            pair_feature,
                            pair_hyperedge_feature_map.get(
                                (task_id, uav_id),
                                torch.zeros(
                                    (config.PAIR_HYPEREDGE_SCORE_FEATURE_DIM,),
                                    dtype=torch.float32,
                                    device=self.device,
                                ),
                            ),
                        ],
                        dim=0,
                    )
                pair_emb.append(pair_feature)
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
