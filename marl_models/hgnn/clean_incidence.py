from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


CLEAN_TASK_ENCODER_MLP = "mlp"
CLEAN_TASK_ENCODER_CURRENT_MEAN_HGNN = "current_mean_hgnn"
CLEAN_TASK_ENCODER_STANDARD_WEIGHTED_HGNN = "standard_weighted_hgnn"
CLEAN_TASK_ENCODER_TYPED_GATED_HGNN = "typed_gated_hgnn"
CLEAN_TASK_ENCODER_LEGACY_HGNN = "hgnn"
CLEAN_TASK_ENCODER_TYPES = (
    CLEAN_TASK_ENCODER_MLP,
    CLEAN_TASK_ENCODER_CURRENT_MEAN_HGNN,
    CLEAN_TASK_ENCODER_STANDARD_WEIGHTED_HGNN,
    CLEAN_TASK_ENCODER_TYPED_GATED_HGNN,
)
CLEAN_TASK_ENCODER_CHOICES = (
    CLEAN_TASK_ENCODER_LEGACY_HGNN,
    *CLEAN_TASK_ENCODER_TYPES,
)
HYPEREDGE_TYPE_COUNT = 4
_EPS = 1.0e-12


def normalize_clean_task_encoder_type(encoder_type: str) -> str:
    """Normalize public encoder names while preserving the legacy hgnn alias."""

    value = str(encoder_type)
    if value == CLEAN_TASK_ENCODER_LEGACY_HGNN:
        return CLEAN_TASK_ENCODER_CURRENT_MEAN_HGNN
    if value not in CLEAN_TASK_ENCODER_TYPES:
        raise ValueError(f"unsupported clean task encoder type: {value}")
    return value


def count_trainable_parameters(module: nn.Module) -> int:
    """Return the trainable parameter count for smoke/reporting."""

    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


def build_clean_task_encoder(
    *,
    encoder_type: str,
    task_feature_dim: int,
    hidden_dim: int,
    output_dim: int | None = None,
    num_layers: int = 1,
) -> nn.Module:
    """Build one of the clean task encoder variants.

    The legacy name ``hgnn`` is intentionally accepted and mapped to the
    current mean-incidence HGNN to preserve existing run behavior.
    """

    normalized = normalize_clean_task_encoder_type(encoder_type)
    if normalized == CLEAN_TASK_ENCODER_MLP:
        from marl_models.hgnn.clean_independent_mlp import CleanIndependentTaskMLP

        return CleanIndependentTaskMLP(
            task_feature_dim=task_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
    if normalized == CLEAN_TASK_ENCODER_CURRENT_MEAN_HGNN:
        return CleanIncidenceHGNN(
            task_feature_dim=task_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
        )
    if normalized == CLEAN_TASK_ENCODER_STANDARD_WEIGHTED_HGNN:
        return CleanStandardWeightedIncidenceHGNN(
            task_feature_dim=task_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
        )
    if normalized == CLEAN_TASK_ENCODER_TYPED_GATED_HGNN:
        return CleanTypedGatedIncidenceHGNN(
            task_feature_dim=task_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
        )
    raise AssertionError(f"unhandled clean task encoder type: {normalized}")


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
        hyperedge_type_ids: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        del hyperedge_type_ids
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


class StandardWeightedIncidenceHGNNLayer(nn.Module):
    """Symmetric-normalized incidence propagation layer with W = I."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.hyperedge_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, node_features: torch.Tensor, incidence_matrix: torch.Tensor) -> torch.Tensor:
        if node_features.numel() == 0 or incidence_matrix.numel() == 0:
            return node_features

        node_messages = _standard_weighted_messages(node_features, incidence_matrix)
        updated = self.self_proj(node_features) + self.hyperedge_proj(node_messages)
        return self.activation(self.norm(updated))


class CleanStandardWeightedIncidenceHGNN(nn.Module):
    """Clean task-only HGNN using symmetric-normalized incidence propagation."""

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
        self.layers = nn.ModuleList(
            StandardWeightedIncidenceHGNNLayer(int(hidden_dim)) for _ in range(layer_count)
        )
        self.output_proj = nn.Linear(int(hidden_dim), self.output_dim)
        self.activation = nn.ReLU()

    def forward(
        self,
        task_features: torch.Tensor | np.ndarray,
        incidence_matrix: torch.Tensor | np.ndarray,
        hyperedge_type_ids: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        del hyperedge_type_ids
        task_features_tensor, incidence_tensor = _validated_task_and_incidence(
            task_features,
            incidence_matrix,
            output_dim=self.output_dim,
        )
        if task_features_tensor.shape[0] == 0:
            return task_features_tensor.new_zeros((0, self.output_dim))

        hidden = self.activation(self.input_proj(task_features_tensor))
        if incidence_tensor.shape[1] == 0:
            return self.output_proj(hidden)

        for layer in self.layers:
            hidden = layer(hidden, incidence_tensor)
        return self.output_proj(hidden)


class TypedGatedIncidenceHGNNLayer(nn.Module):
    """Symmetric incidence propagation with learnable hyperedge-type weights and residual gate."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.raw_type_weights = nn.Parameter(torch.zeros(HYPEREDGE_TYPE_COUNT, dtype=torch.float32))
        self.message_projection = nn.Linear(hidden_dim, hidden_dim)
        self.gate_network = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def normalized_type_weights(self) -> torch.Tensor:
        positive = F.softplus(self.raw_type_weights)
        return positive / positive.mean().clamp_min(_EPS)

    def forward(
        self,
        node_features: torch.Tensor,
        incidence_matrix: torch.Tensor,
        hyperedge_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        if node_features.numel() == 0:
            return node_features
        if incidence_matrix.numel() == 0:
            return self.norm(node_features)

        type_ids = _validate_hyperedge_type_ids(
            hyperedge_type_ids,
            expected_edge_count=int(incidence_matrix.shape[1]),
            device=incidence_matrix.device,
        )
        type_weights = self.normalized_type_weights().to(device=incidence_matrix.device, dtype=node_features.dtype)
        edge_weights = type_weights.index_select(0, type_ids)
        node_messages = _typed_weighted_messages(node_features, incidence_matrix, edge_weights)
        message = self.message_projection(node_messages)
        gate = torch.sigmoid(self.gate_network(torch.cat([node_features, message], dim=-1)))
        return self.norm(node_features + gate * message)


class CleanTypedGatedIncidenceHGNN(nn.Module):
    """Clean task-only HGNN with typed hyperedge weights and per-hidden-dim residual gate."""

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
        self.layers = nn.ModuleList(
            TypedGatedIncidenceHGNNLayer(int(hidden_dim)) for _ in range(layer_count)
        )
        self.output_proj = nn.Linear(int(hidden_dim), self.output_dim)
        self.activation = nn.ReLU()

    def forward(
        self,
        task_features: torch.Tensor | np.ndarray,
        incidence_matrix: torch.Tensor | np.ndarray,
        hyperedge_type_ids: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        task_features_tensor, incidence_tensor = _validated_task_and_incidence(
            task_features,
            incidence_matrix,
            output_dim=self.output_dim,
        )
        if task_features_tensor.shape[0] == 0:
            _validate_hyperedge_type_ids(
                hyperedge_type_ids,
                expected_edge_count=int(incidence_tensor.shape[1]),
                device=task_features_tensor.device,
            )
            return task_features_tensor.new_zeros((0, self.output_dim))

        if hyperedge_type_ids is None:
            raise ValueError("hyperedge_type_ids is required for typed_gated_hgnn.")

        type_ids = _validate_hyperedge_type_ids(
            hyperedge_type_ids,
            expected_edge_count=int(incidence_tensor.shape[1]),
            device=task_features_tensor.device,
        )
        hidden = self.activation(self.input_proj(task_features_tensor))
        if incidence_tensor.shape[1] == 0:
            return self.output_proj(hidden)

        for layer in self.layers:
            hidden = layer(hidden, incidence_tensor, type_ids)
        return self.output_proj(hidden)


def _standard_weighted_messages(
    node_features: torch.Tensor,
    incidence_matrix: torch.Tensor,
) -> torch.Tensor:
    edge_degree = incidence_matrix.sum(dim=0)
    node_degree = incidence_matrix.sum(dim=1)
    edge_degree_inv = _safe_reciprocal(edge_degree)
    node_degree_inv_sqrt = _safe_reciprocal_sqrt(node_degree)

    x_norm = node_degree_inv_sqrt.unsqueeze(-1) * node_features
    edge_features = incidence_matrix.transpose(0, 1).matmul(x_norm)
    edge_features = edge_degree_inv.unsqueeze(-1) * edge_features
    node_messages = incidence_matrix.matmul(edge_features)
    return node_degree_inv_sqrt.unsqueeze(-1) * node_messages


def _typed_weighted_messages(
    node_features: torch.Tensor,
    incidence_matrix: torch.Tensor,
    edge_weights: torch.Tensor,
) -> torch.Tensor:
    edge_degree = incidence_matrix.sum(dim=0)
    node_degree = incidence_matrix.matmul(edge_weights.reshape(-1, 1)).reshape(-1)
    edge_degree_inv = _safe_reciprocal(edge_degree)
    node_degree_inv_sqrt = _safe_reciprocal_sqrt(node_degree)

    x_norm = node_degree_inv_sqrt.unsqueeze(-1) * node_features
    edge_features = incidence_matrix.transpose(0, 1).matmul(x_norm)
    edge_features = edge_degree_inv.unsqueeze(-1) * edge_features
    edge_features = edge_weights.unsqueeze(-1) * edge_features
    node_messages = incidence_matrix.matmul(edge_features)
    return node_degree_inv_sqrt.unsqueeze(-1) * node_messages


def _safe_reciprocal(values: torch.Tensor) -> torch.Tensor:
    reciprocal = values.clamp_min(_EPS).reciprocal()
    return reciprocal.masked_fill(values <= 0.0, 0.0)


def _safe_reciprocal_sqrt(values: torch.Tensor) -> torch.Tensor:
    reciprocal_sqrt = values.clamp_min(_EPS).rsqrt()
    return reciprocal_sqrt.masked_fill(values <= 0.0, 0.0)


def _validated_task_and_incidence(
    task_features: torch.Tensor | np.ndarray,
    incidence_matrix: torch.Tensor | np.ndarray,
    *,
    output_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del output_dim
    task_features_tensor = _as_float_tensor(task_features)
    if task_features_tensor.dim() != 2:
        raise ValueError("task_features must be a 2D tensor or array.")
    incidence_tensor = _as_float_tensor(
        incidence_matrix,
        device=task_features_tensor.device,
        dtype=task_features_tensor.dtype,
    )
    if incidence_tensor.dim() != 2:
        raise ValueError("incidence_matrix must be a 2D tensor or array.")
    if incidence_tensor.shape[0] != task_features_tensor.shape[0]:
        raise ValueError("incidence_matrix row count must match task_features row count.")
    return task_features_tensor, incidence_tensor


def _validate_hyperedge_type_ids(
    hyperedge_type_ids: torch.Tensor | np.ndarray | None,
    *,
    expected_edge_count: int,
    device: torch.device,
) -> torch.Tensor:
    if hyperedge_type_ids is None:
        if int(expected_edge_count) == 0:
            return torch.empty((0,), dtype=torch.long, device=device)
        raise ValueError("hyperedge_type_ids is required when incidence_matrix has hyperedges.")
    type_ids = torch.as_tensor(hyperedge_type_ids, device=device)
    if type_ids.dim() != 1:
        raise ValueError("hyperedge_type_ids must be a 1D tensor or array.")
    if int(type_ids.shape[0]) != int(expected_edge_count):
        raise ValueError(
            "hyperedge_type_ids length must equal incidence_matrix column count "
            f"({int(type_ids.shape[0])} != {int(expected_edge_count)})."
        )
    if type_ids.numel() == 0:
        return type_ids.to(dtype=torch.long)
    if torch.is_floating_point(type_ids) and not torch.equal(type_ids, type_ids.round()):
        raise ValueError("hyperedge_type_ids must contain integer type ids.")
    type_ids = type_ids.to(dtype=torch.long)
    if bool(((type_ids < 0) | (type_ids >= HYPEREDGE_TYPE_COUNT)).any().item()):
        raise ValueError(f"hyperedge_type_ids must be in [0, {HYPEREDGE_TYPE_COUNT - 1}].")
    return type_ids


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
