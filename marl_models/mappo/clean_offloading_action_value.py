from __future__ import annotations

from typing import Any

import torch
from torch import nn


class CleanOffloadingActionValueCritic(nn.Module):
    """Action-conditioned value head used only as an offloading control variate."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError("input_dim must be positive")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        output_layer = self.net[-1]
        assert isinstance(output_layer, nn.Linear)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(self, candidate_context_features: torch.Tensor) -> torch.Tensor:
        if candidate_context_features.dim() != 2:
            raise ValueError("candidate_context_features must be a 2D tensor")
        if int(candidate_context_features.shape[1]) != self.input_dim:
            raise ValueError(
                "candidate_context_features width mismatch: "
                f"expected {self.input_dim}, got {int(candidate_context_features.shape[1])}"
            )
        return self.net(candidate_context_features).squeeze(-1)


def build_rng_neutral_clean_counterfactual_q(
    *, input_dim: int, hidden_dim: int = 128
) -> CleanOffloadingActionValueCritic:
    """Build the auxiliary CPU Q head without advancing the main Torch RNG."""

    rng_state = torch.random.get_rng_state().clone()
    try:
        return CleanOffloadingActionValueCritic(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )
    finally:
        torch.random.set_rng_state(rng_state)


def masked_counterfactual_value(
    *,
    logits: torch.Tensor,
    action_values: torch.Tensor,
    candidate_mask: torch.Tensor,
    selected_action: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached selected-minus-policy-expected Q and legal Q spread."""

    if logits.dim() != 1 or action_values.dim() != 1 or candidate_mask.dim() != 1:
        raise ValueError("logits, action_values, and candidate_mask must be 1D tensors")
    if logits.shape != action_values.shape or logits.shape != candidate_mask.shape:
        raise ValueError("logits, action_values, and candidate_mask must have identical shapes")
    mask = candidate_mask.to(dtype=torch.bool)
    legal_count = int(mask.sum().item())
    if legal_count <= 0:
        raise ValueError("counterfactual value requires at least one legal candidate")
    action_index = int(selected_action)
    if action_index < 0 or action_index >= int(mask.numel()) or not bool(mask[action_index].item()):
        raise ValueError("selected_action must identify a legal candidate")
    if not bool(torch.isfinite(logits[mask]).all().item()):
        raise FloatingPointError("legal offloading logits contain non-finite values")
    if not bool(torch.isfinite(action_values[mask]).all().item()):
        raise FloatingPointError("legal offloading action values contain non-finite values")

    detached_values = action_values.detach()
    legal_values = detached_values[mask]
    spread = legal_values.max() - legal_values.min()
    if legal_count == 1:
        return detached_values[action_index].new_zeros(()), spread

    masked_logits = logits.detach().masked_fill(~mask, torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(masked_logits, dim=0)
    expected = torch.sum(probabilities * detached_values)
    counterfactual = detached_values[action_index] - expected
    return counterfactual.detach(), spread.detach()


def normalize_counterfactual_values(
    values: list[torch.Tensor],
    *,
    epsilon: float = 1e-8,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Population-normalize detached scalar values, with a stable zero fallback."""

    if not values:
        return [], {
            "mean": 0.0,
            "std": 0.0,
            "normalized_std": 0.0,
            "effective_count": 0,
        }
    scalars = [value.detach().reshape(()) for value in values]
    stacked = torch.stack(scalars)
    if not bool(torch.isfinite(stacked).all().item()):
        raise FloatingPointError("counterfactual advantages contain non-finite values")
    mean = stacked.mean()
    std = stacked.std(unbiased=False) if int(stacked.numel()) >= 2 else stacked.new_zeros(())
    if int(stacked.numel()) < 2 or float(std.item()) < float(epsilon):
        normalized = torch.zeros_like(stacked)
    else:
        normalized = (stacked - mean) / std
    normalized_std = (
        normalized.std(unbiased=False) if int(normalized.numel()) >= 2 else normalized.new_zeros(())
    )
    diagnostics = {
        "mean": float(mean.item()),
        "std": float(std.item()),
        "normalized_std": float(normalized_std.item()),
        "effective_count": int(stacked.numel()),
    }
    return [item.detach() for item in normalized.unbind(0)], diagnostics
