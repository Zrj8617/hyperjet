from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _PoisonArray:
    def __array__(self, dtype=None):  # noqa: ANN001
        raise AssertionError("MLP encoder must not read incidence_matrix or hyperedge_type_ids.")


def main() -> None:
    try:
        import torch
        from marl_models.hgnn import (
            CLEAN_TASK_ENCODER_CHOICES,
            CLEAN_TASK_ENCODER_CURRENT_MEAN_HGNN,
            CLEAN_TASK_ENCODER_MLP,
            CLEAN_TASK_ENCODER_STANDARD_WEIGHTED_HGNN,
            CLEAN_TASK_ENCODER_TYPED_GATED_HGNN,
            CleanIncidenceHGNN,
            CleanTypedGatedIncidenceHGNN,
            build_clean_task_encoder,
            count_trainable_parameters,
            normalize_clean_task_encoder_type,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            print("smoke_clean_hgnn_encoder_variants skipped: torch is not installed in this Python runtime")
            return
        raise

    torch.manual_seed(23)
    task_feature_dim = 5
    hidden_dim = 7
    output_dim = 3
    task_features = torch.randn(4, task_feature_dim)
    incidence = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    hyperedge_type_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    encoder_types = (
        CLEAN_TASK_ENCODER_MLP,
        CLEAN_TASK_ENCODER_CURRENT_MEAN_HGNN,
        CLEAN_TASK_ENCODER_STANDARD_WEIGHTED_HGNN,
        CLEAN_TASK_ENCODER_TYPED_GATED_HGNN,
    )
    param_counts: dict[str, int] = {}

    _assert("hgnn" in CLEAN_TASK_ENCODER_CHOICES, "legacy hgnn alias must remain available.")
    _assert(
        normalize_clean_task_encoder_type("hgnn") == CLEAN_TASK_ENCODER_CURRENT_MEAN_HGNN,
        "legacy hgnn alias must normalize to current_mean_hgnn.",
    )

    for encoder_type in encoder_types:
        model = build_clean_task_encoder(
            encoder_type=encoder_type,
            task_feature_dim=task_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        param_counts[encoder_type] = count_trainable_parameters(model)
        output = model(task_features, incidence, hyperedge_type_ids)
        _assert(output.shape == (4, output_dim), f"{encoder_type} output shape mismatch.")
        _assert(torch.isfinite(output).all().item(), f"{encoder_type} output contains NaN/Inf.")

        empty_features = torch.empty((0, task_feature_dim), dtype=torch.float32)
        empty_incidence = torch.empty((0, 0), dtype=torch.float32)
        empty_types = torch.empty((0,), dtype=torch.long)
        empty_output = model(empty_features, empty_incidence, empty_types)
        _assert(empty_output.shape == (0, output_dim), f"{encoder_type} empty output shape mismatch.")
        _assert(torch.isfinite(empty_output).all().item(), f"{encoder_type} empty output contains NaN/Inf.")

        isolated_features = torch.randn(3, task_feature_dim)
        zero_edge_incidence = torch.empty((3, 0), dtype=torch.float32)
        zero_edge_types = torch.empty((0,), dtype=torch.long)
        isolated_output = model(isolated_features, zero_edge_incidence, zero_edge_types)
        _assert(isolated_output.shape == (3, output_dim), f"{encoder_type} zero-edge output shape mismatch.")
        _assert(torch.isfinite(isolated_output).all().item(), f"{encoder_type} zero-edge output contains NaN/Inf.")

    mean_model = build_clean_task_encoder(
        encoder_type=CLEAN_TASK_ENCODER_CURRENT_MEAN_HGNN,
        task_feature_dim=task_feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )
    _assert(isinstance(mean_model, CleanIncidenceHGNN), "current_mean_hgnn must use the existing CleanIncidenceHGNN class.")

    mlp = build_clean_task_encoder(
        encoder_type=CLEAN_TASK_ENCODER_MLP,
        task_feature_dim=task_feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )
    mlp_output = mlp(task_features, _PoisonArray(), _PoisonArray())
    _assert(mlp_output.shape == (4, output_dim), "MLP output shape mismatch with poison graph inputs.")

    typed = build_clean_task_encoder(
        encoder_type=CLEAN_TASK_ENCODER_TYPED_GATED_HGNN,
        task_feature_dim=task_feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )
    _assert(isinstance(typed, CleanTypedGatedIncidenceHGNN), "typed_gated_hgnn builder returned wrong class.")

    try:
        typed(task_features, incidence, torch.tensor([0, 1, 2], dtype=torch.long))
    except ValueError as exc:
        _assert("length" in str(exc), "length mismatch error should be clear.")
    else:
        raise AssertionError("typed_gated_hgnn must reject type-id length mismatch.")

    try:
        typed(task_features, incidence, torch.tensor([0, 1, 2, 4], dtype=torch.long))
    except ValueError as exc:
        _assert("type_ids" in str(exc) or "hyperedge_type_ids" in str(exc), "invalid type-id error should be clear.")
    else:
        raise AssertionError("typed_gated_hgnn must reject invalid type ids.")

    typed.zero_grad(set_to_none=True)
    typed_output = typed(task_features, incidence, hyperedge_type_ids)
    loss = typed_output.pow(2).mean()
    loss.backward()
    first_layer = typed.layers[0]
    _assert(first_layer.raw_type_weights.grad is not None, "raw_type_weights gradient is missing.")
    _assert(torch.isfinite(first_layer.raw_type_weights.grad).all().item(), "raw_type_weights gradient has NaN/Inf.")
    gate_grads = [
        parameter.grad
        for parameter in first_layer.gate_network.parameters()
        if parameter.requires_grad
    ]
    _assert(gate_grads and all(grad is not None for grad in gate_grads), "residual gate gradients are missing.")
    _assert(all(torch.isfinite(grad).all().item() for grad in gate_grads if grad is not None), "gate gradients have NaN/Inf.")
    gate = torch.sigmoid(
        first_layer.gate_network(
            torch.cat(
                [
                    typed.activation(typed.input_proj(task_features)),
                    torch.zeros((task_features.shape[0], hidden_dim), dtype=torch.float32),
                ],
                dim=-1,
            )
        )
    )
    _assert(gate.shape == (4, hidden_dim), "typed residual gate must be per hidden dimension.")

    print(f"smoke_clean_hgnn_encoder_variants passed; param_counts={param_counts}")


if __name__ == "__main__":
    main()
