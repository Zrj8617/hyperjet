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


def main() -> int:
    try:
        import torch
    except ModuleNotFoundError:
        print("smoke_clean_task_encoder_ablation skipped: torch is not installed")
        return 0

    from marl_models.hgnn import CleanIndependentTaskMLP

    torch.manual_seed(20260725)
    encoder = CleanIndependentTaskMLP(
        task_feature_dim=12,
        hidden_dim=16,
        output_dim=8,
    )
    features = torch.randn(5, 12)
    incidence_a = torch.zeros(5, 0)
    incidence_b = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    output_a = encoder(features, incidence_a)
    output_b = encoder(features, incidence_b)
    _assert(output_a.shape == (5, 8), "MLP task encoder output shape mismatch")
    _assert(
        torch.equal(output_a, output_b),
        "MLP task encoder must be exactly invariant to incidence values",
    )
    empty = encoder(np.zeros((0, 12), dtype=np.float32), np.zeros((0, 0), dtype=np.float32))
    _assert(tuple(empty.shape) == (0, 8), "MLP task encoder empty shape mismatch")
    output_a.sum().backward()
    _assert(
        all(parameter.grad is not None for parameter in encoder.parameters()),
        "MLP task encoder parameters must receive gradients",
    )
    print("smoke_clean_task_encoder_ablation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
