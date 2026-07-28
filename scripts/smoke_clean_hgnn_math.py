from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _PoisonTypeIds:
    def __array__(self, dtype=None):  # noqa: ANN001
        raise AssertionError("current_mean_hgnn must not read hyperedge_type_ids.")


def main() -> None:
    try:
        import torch
        from marl_models.hgnn import (
            CLEAN_TASK_ENCODER_CURRENT_MEAN_HGNN,
            CLEAN_TASK_ENCODER_MLP,
            CLEAN_TASK_ENCODER_STANDARD_WEIGHTED_HGNN,
            CLEAN_TASK_ENCODER_TYPED_GATED_HGNN,
            CleanIncidenceHGNN,
            CleanStandardWeightedIncidenceHGNN,
            CleanTypedGatedIncidenceHGNN,
            build_clean_task_encoder,
            normalize_clean_task_encoder_type,
        )
        from marl_models.hgnn.clean_incidence import (
            TypedGatedIncidenceHGNNLayer,
            _standard_weighted_messages,
            _typed_weighted_messages,
            _validate_hyperedge_type_ids,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            print("smoke_clean_hgnn_math skipped: torch is not installed in this Python runtime")
            return
        raise

    torch.manual_seed(31)
    _test_current_mean_alias_and_type_ignore(torch)
    _test_standard_weighted_manual_formula(torch, _standard_weighted_messages)
    _test_typed_weighted_manual_formula(torch, TypedGatedIncidenceHGNNLayer, _typed_weighted_messages)
    _test_type_id_validation(torch, _validate_hyperedge_type_ids)
    _test_empty_and_isolated_regression(torch)
    print("smoke_clean_hgnn_math passed")


def _test_current_mean_alias_and_type_ignore(torch) -> None:  # noqa: ANN001
    from marl_models.hgnn import (
        CLEAN_TASK_ENCODER_CURRENT_MEAN_HGNN,
        CleanIncidenceHGNN,
        build_clean_task_encoder,
        normalize_clean_task_encoder_type,
    )

    _assert(
        normalize_clean_task_encoder_type("hgnn") == CLEAN_TASK_ENCODER_CURRENT_MEAN_HGNN,
        "legacy hgnn alias must normalize to current_mean_hgnn.",
    )
    torch.manual_seed(101)
    legacy = build_clean_task_encoder(
        encoder_type="hgnn",
        task_feature_dim=2,
        hidden_dim=4,
        output_dim=3,
    )
    torch.manual_seed(101)
    current = build_clean_task_encoder(
        encoder_type=CLEAN_TASK_ENCODER_CURRENT_MEAN_HGNN,
        task_feature_dim=2,
        hidden_dim=4,
        output_dim=3,
    )
    _assert(isinstance(legacy, CleanIncidenceHGNN), "legacy hgnn must build CleanIncidenceHGNN.")
    _assert(isinstance(current, CleanIncidenceHGNN), "current_mean_hgnn must build CleanIncidenceHGNN.")

    task_features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    incidence = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
    type_ids = torch.tensor([0, 3], dtype=torch.long)
    legacy_out = legacy(task_features, incidence, type_ids)
    current_out = current(task_features, incidence)
    _assert(torch.allclose(legacy_out, current_out, atol=1.0e-6), "hgnn alias output must match current_mean_hgnn.")

    no_type_out = current(task_features, incidence)
    poisoned_type_out = current(task_features, incidence, _PoisonTypeIds())
    _assert(
        torch.allclose(no_type_out, poisoned_type_out, atol=1.0e-6),
        "current_mean_hgnn output must be independent of hyperedge_type_ids.",
    )


def _test_standard_weighted_manual_formula(torch, standard_kernel) -> None:  # noqa: ANN001
    h = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    x = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=torch.float32,
    )
    edge_degree = h.sum(dim=0)
    node_degree = h.sum(dim=1)
    edge_degree_inv = torch.where(edge_degree > 0.0, 1.0 / edge_degree, torch.zeros_like(edge_degree))
    node_degree_inv_sqrt = torch.where(node_degree > 0.0, 1.0 / torch.sqrt(node_degree), torch.zeros_like(node_degree))
    x_norm = node_degree_inv_sqrt.unsqueeze(-1) * x
    edge_features = h.transpose(0, 1).matmul(x_norm)
    edge_features = edge_degree_inv.unsqueeze(-1) * edge_features
    expected = node_degree_inv_sqrt.unsqueeze(-1) * h.matmul(edge_features)

    actual = standard_kernel(x, h)
    _assert(torch.allclose(actual, expected, atol=1.0e-6), "standard weighted propagation must match manual formula.")
    _assert(torch.isfinite(actual).all().item(), "standard weighted propagation produced NaN/Inf.")


def _test_typed_weighted_manual_formula(torch, layer_cls, typed_kernel) -> None:  # noqa: ANN001
    from marl_models.hgnn import CleanTypedGatedIncidenceHGNN

    h = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    x = torch.tensor(
        [
            [2.0, 1.0],
            [4.0, 3.0],
            [6.0, 5.0],
        ],
        dtype=torch.float32,
    )
    layer = layer_cls(hidden_dim=2)
    with torch.no_grad():
        layer.raw_type_weights.copy_(torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32))
    type_weights = layer.normalized_type_weights()
    _assert(torch.allclose(type_weights.mean(), torch.ones((), dtype=type_weights.dtype), atol=1.0e-6), "type weights mean must be 1.")
    type_ids = torch.tensor([0, 2], dtype=torch.long)
    edge_weights = type_weights.index_select(0, type_ids)

    edge_degree = h.sum(dim=0)
    weighted_node_degree = h.matmul(edge_weights.reshape(-1, 1)).reshape(-1)
    unweighted_node_degree = h.sum(dim=1)
    _assert(
        not torch.allclose(weighted_node_degree, unweighted_node_degree),
        "test setup must distinguish weighted and unweighted node degrees.",
    )
    expected_node_degree = torch.tensor(
        [float(edge_weights[0]), float(edge_weights[0] + edge_weights[1]), float(edge_weights[1])],
        dtype=torch.float32,
    )
    _assert(
        torch.allclose(weighted_node_degree, expected_node_degree, atol=1.0e-6),
        "weighted node degree must equal H @ edge_weights.",
    )

    edge_degree_inv = torch.where(edge_degree > 0.0, 1.0 / edge_degree, torch.zeros_like(edge_degree))
    node_degree_inv_sqrt = torch.where(
        weighted_node_degree > 0.0,
        1.0 / torch.sqrt(weighted_node_degree),
        torch.zeros_like(weighted_node_degree),
    )
    x_norm = node_degree_inv_sqrt.unsqueeze(-1) * x
    edge_features = h.transpose(0, 1).matmul(x_norm)
    edge_features = edge_degree_inv.unsqueeze(-1) * edge_features
    edge_features = edge_weights.unsqueeze(-1) * edge_features
    expected = node_degree_inv_sqrt.unsqueeze(-1) * h.matmul(edge_features)

    actual = typed_kernel(x, h, edge_weights)
    _assert(torch.allclose(actual, expected, atol=1.0e-6), "typed propagation must match weighted manual formula.")
    _assert(torch.isfinite(actual).all().item(), "typed propagation produced NaN/Inf.")

    model = CleanTypedGatedIncidenceHGNN(task_feature_dim=2, hidden_dim=2, output_dim=2)
    output = model(x, h, type_ids)
    output.sum().backward()
    _assert(model.layers[0].raw_type_weights.grad is not None, "raw_type_weights gradient is missing.")
    _assert(torch.isfinite(model.layers[0].raw_type_weights.grad).all().item(), "raw_type_weights gradient has NaN/Inf.")


def _test_type_id_validation(torch, validator) -> None:  # noqa: ANN001
    valid_empty = validator(torch.empty((0,), dtype=torch.long), expected_edge_count=0, device=torch.device("cpu"))
    _assert(valid_empty.shape == (0,), "empty hyperedge_type_ids must be legal for zero hyperedges.")
    valid = validator(torch.tensor([0, 1, 2, 3]), expected_edge_count=4, device=torch.device("cpu"))
    _assert(valid.dtype == torch.long and valid.tolist() == [0, 1, 2, 3], "valid type ids should roundtrip.")

    for bad_types, message in (
        (torch.tensor([0, 1, 2]), "length mismatch"),
        (torch.tensor([-1, 0]), "negative type id"),
        (torch.tensor([0, 4]), "too-large type id"),
    ):
        try:
            validator(bad_types, expected_edge_count=2, device=torch.device("cpu"))
        except ValueError:
            pass
        else:
            raise AssertionError(f"validator must reject {message}.")


def _test_empty_and_isolated_regression(torch) -> None:  # noqa: ANN001
    from marl_models.hgnn import (
        CLEAN_TASK_ENCODER_CURRENT_MEAN_HGNN,
        CLEAN_TASK_ENCODER_MLP,
        CLEAN_TASK_ENCODER_STANDARD_WEIGHTED_HGNN,
        CLEAN_TASK_ENCODER_TYPED_GATED_HGNN,
        build_clean_task_encoder,
    )

    encoder_types = (
        CLEAN_TASK_ENCODER_MLP,
        CLEAN_TASK_ENCODER_CURRENT_MEAN_HGNN,
        CLEAN_TASK_ENCODER_STANDARD_WEIGHTED_HGNN,
        CLEAN_TASK_ENCODER_TYPED_GATED_HGNN,
    )
    for encoder_type in encoder_types:
        model = build_clean_task_encoder(
            encoder_type=encoder_type,
            task_feature_dim=3,
            hidden_dim=5,
            output_dim=4,
        )
        empty_output = model(
            torch.empty((0, 3), dtype=torch.float32),
            torch.empty((0, 0), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
        )
        _assert(empty_output.shape == (0, 4), f"{encoder_type} empty graph shape mismatch.")
        _assert(torch.isfinite(empty_output).all().item(), f"{encoder_type} empty graph produced NaN/Inf.")

        isolated_output = model(
            torch.randn(3, 3),
            torch.empty((3, 0), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
        )
        _assert(isolated_output.shape == (3, 4), f"{encoder_type} zero-hyperedge shape mismatch.")
        _assert(torch.isfinite(isolated_output).all().item(), f"{encoder_type} zero-hyperedge produced NaN/Inf.")

        missing_type_class_output = model(
            torch.randn(3, 3),
            torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32),
            torch.tensor([0, 3], dtype=torch.long),
        )
        _assert(missing_type_class_output.shape == (3, 4), f"{encoder_type} missing-type-class shape mismatch.")
        _assert(torch.isfinite(missing_type_class_output).all().item(), f"{encoder_type} missing type class produced NaN/Inf.")


if __name__ == "__main__":
    main()
