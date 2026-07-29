from __future__ import annotations

import math
from typing import Any


def eft_auxiliary_lambda(
    update_index: int,
    *,
    lambda_initial: float,
    constant_through_update: int = 8,
    zero_at_update: int = 20,
) -> float:
    """Approved outer-update schedule for the diagnostic EFT auxiliary loss."""
    update = int(update_index)
    initial = float(lambda_initial)
    constant_end = int(constant_through_update)
    zero_update = int(zero_at_update)
    if update < 0:
        raise ValueError("update_index must be non-negative")
    if not math.isfinite(initial) or initial < 0.0:
        raise ValueError("lambda_initial must be finite and non-negative")
    if constant_end < 0 or zero_update <= constant_end:
        raise ValueError("EFT auxiliary schedule boundaries are invalid")
    if update <= constant_end:
        return initial
    if update >= zero_update:
        return 0.0
    remaining_steps = zero_update - update
    decay_steps = zero_update - constant_end
    return initial * float(remaining_steps) / float(decay_steps)


def compute_eft_auxiliary_objective(
    items: list[dict[str, Any]],
    *,
    regret_scale: float,
    categorical_cls: Any,
    generator: Any | None = None,
    include_debug_vectors: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Current-policy contextual-bandit loss on immutable rollout EFT vectors.

    Each item must contain the current masked logits and the historical mask/EFT
    vector from the same sequential decision. The rollout PPO action is accepted
    only for diagnostics and is never used to select the auxiliary reward.
    """
    if not math.isfinite(float(regret_scale)) or float(regret_scale) <= 0.0:
        raise ValueError("EFT auxiliary regret_scale must be finite and positive")
    if categorical_cls is None:
        raise ModuleNotFoundError("torch Categorical is required for EFT auxiliary loss")

    losses: list[Any] = []
    sampled_raw_regrets: list[Any] = []
    sampled_actions: list[int] = []
    greedy_actions: list[int] = []
    entropies: list[Any] = []
    exact_baselines: list[Any] = []
    advantages: list[Any] = []
    margin5_matches: list[float] = []
    margin20_matches: list[float] = []
    excluded_zero_legal = 0
    excluded_single_legal = 0
    invalid_action_count = 0
    rollout_selected_regrets: list[float] = []

    reference = None
    for item in items:
        logits = item["masked_logits"]
        mask = item["candidate_mask"]
        eft = item["candidate_estimated_finish_times"]
        reference = logits if reference is None else reference
        if logits.dim() != 1 or mask.dim() != 1 or eft.dim() != 1:
            raise ValueError("EFT auxiliary logits, mask, and EFT must be 1D")
        if logits.shape != mask.shape or logits.shape != eft.shape:
            raise ValueError("EFT auxiliary logits/mask/EFT shapes must match")
        legal_count = int(mask.sum().detach().cpu().item())
        if legal_count <= 0:
            excluded_zero_legal += 1
            continue
        legal_eft = eft[mask]
        if not bool(legal_eft.isfinite().all().detach().cpu().item()):
            raise FloatingPointError("EFT auxiliary legal candidate values are non-finite")
        best_eft = legal_eft.min()
        raw_regret = eft.new_zeros(eft.shape)
        raw_regret[mask] = (legal_eft - best_eft).clamp_min(0.0)
        raw_regret = raw_regret.detach()

        rollout_action = item.get("rollout_action")
        if rollout_action is not None:
            rollout_index = int(rollout_action)
            if rollout_index < 0 or rollout_index >= int(mask.numel()) or not bool(
                mask[rollout_index].detach().cpu().item()
            ):
                raise ValueError("rollout action is illegal in historical EFT context")
            rollout_selected_regrets.append(
                float(raw_regret[rollout_index].detach().cpu().item())
            )

        if legal_count == 1:
            excluded_single_legal += 1
            continue

        rewards = (-raw_regret / float(regret_scale)).detach()
        distribution = categorical_cls(logits=logits)
        if generator is None:
            auxiliary_action = distribution.sample()
        else:
            auxiliary_action = distribution.probs.multinomial(
                num_samples=1,
                replacement=True,
                generator=generator,
            ).reshape(())
        action_index = int(auxiliary_action.detach().cpu().item())
        sampled_actions.append(action_index)
        if not bool(mask[action_index].detach().cpu().item()):
            invalid_action_count += 1
            continue

        selected_reward = rewards[action_index]
        exact_baseline = (distribution.probs * rewards).sum()
        advantage = (selected_reward - exact_baseline.detach()).detach()
        loss = -(advantage * distribution.log_prob(auxiliary_action))

        legal_indices = mask.nonzero(as_tuple=False).reshape(-1)
        legal_values = eft[legal_indices]
        order = legal_values.argsort()
        greedy_index = int(legal_indices[order[0]].detach().cpu().item())
        greedy_actions.append(greedy_index)
        margin = float(
            (legal_values[order[1]] - legal_values[order[0]]).detach().cpu().item()
        )
        match = float(action_index == greedy_index)
        if margin >= 5.0:
            margin5_matches.append(match)
        if margin >= 20.0:
            margin20_matches.append(match)

        losses.append(loss)
        sampled_raw_regrets.append(raw_regret[action_index])
        entropies.append(distribution.entropy())
        exact_baselines.append(exact_baseline.detach())
        advantages.append(advantage)

    if invalid_action_count:
        raise AssertionError("masked EFT auxiliary policy sampled an illegal action")
    if losses:
        loss = losses[0].new_tensor(0.0) + losses[0]
        if len(losses) > 1:
            loss = sum(losses[1:], loss)
        loss = loss / float(len(losses))
    elif reference is not None:
        loss = reference.sum() * 0.0
    else:
        raise ValueError("EFT auxiliary objective requires at least one offloading item")

    finite_tensors = [loss, *sampled_raw_regrets, *entropies, *exact_baselines, *advantages]
    if not all(bool(value.isfinite().all().detach().cpu().item()) for value in finite_tensors):
        raise FloatingPointError("EFT auxiliary loss or diagnostics are non-finite")

    effective_count = len(losses)
    sampled_regret_mean = (
        float(sum(value.detach().cpu().item() for value in sampled_raw_regrets) / effective_count)
        if effective_count
        else 0.0
    )
    sampled_regret_values = [
        float(value.detach().cpu().item()) for value in sampled_raw_regrets
    ]
    greedy_agreement = (
        float(sum(int(a == g) for a, g in zip(sampled_actions, greedy_actions)) / effective_count)
        if effective_count
        else 0.0
    )
    diagnostics = {
        "eft_aux_effective_decision_count": int(effective_count),
        "eft_aux_excluded_zero_legal_count": int(excluded_zero_legal),
        "eft_aux_excluded_single_legal_count": int(excluded_single_legal),
        "eft_aux_invalid_action_count": int(invalid_action_count),
        "eft_aux_sampled_raw_regret_mean": float(sampled_regret_mean),
        "eft_aux_sampled_raw_regret_p95": _percentile_nearest(
            sampled_regret_values, 0.95
        ),
        "eft_aux_rollout_chosen_raw_regret_mean": (
            float(sum(rollout_selected_regrets) / len(rollout_selected_regrets))
            if rollout_selected_regrets
            else 0.0
        ),
        "eft_aux_greedy_agreement": float(greedy_agreement),
        "eft_aux_margin_ge_5s_accuracy": (
            float(sum(margin5_matches) / len(margin5_matches)) if margin5_matches else None
        ),
        "eft_aux_margin_ge_20s_accuracy": (
            float(sum(margin20_matches) / len(margin20_matches)) if margin20_matches else None
        ),
        "eft_aux_entropy_mean": (
            float(sum(value.detach().cpu().item() for value in entropies) / effective_count)
            if effective_count
            else 0.0
        ),
    }
    if include_debug_vectors:
        # Smoke-only proof vectors. The formal trainer keeps these disabled so
        # JSONL size scales with updates, not with every sampled decision.
        diagnostics.update(
            {
                "eft_aux_sampled_actions": list(sampled_actions),
                "eft_aux_greedy_actions": list(greedy_actions),
                "eft_aux_exact_baselines": [
                    float(value.detach().cpu().item()) for value in exact_baselines
                ],
                "eft_aux_advantages": [
                    float(value.detach().cpu().item()) for value in advantages
                ],
            }
        )
    return loss, diagnostics


def summarize_historical_eft_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic rollout-action regret metrics, with no policy sampling."""
    selected_regrets: list[float] = []
    selected_matches: list[float] = []
    margin5_matches: list[float] = []
    margin20_matches: list[float] = []
    zero_legal = 0
    single_legal = 0
    for item in items:
        mask = item["candidate_mask"]
        eft = item["candidate_estimated_finish_times"]
        if mask.dim() != 1 or eft.dim() != 1 or mask.shape != eft.shape:
            raise ValueError("historical EFT mask/vector shapes must match")
        legal_indices = mask.nonzero(as_tuple=False).reshape(-1)
        legal_count = int(legal_indices.numel())
        if legal_count <= 0:
            zero_legal += 1
            continue
        if legal_count == 1:
            single_legal += 1
        legal_eft = eft[legal_indices]
        if not bool(legal_eft.isfinite().all().detach().cpu().item()):
            raise FloatingPointError("historical legal candidate EFT values are non-finite")
        action = int(item["rollout_action"])
        if action < 0 or action >= int(mask.numel()) or not bool(
            mask[action].detach().cpu().item()
        ):
            raise ValueError("historical rollout action is illegal")
        best_offset = int(legal_eft.argmin().detach().cpu().item())
        best_action = int(legal_indices[best_offset].detach().cpu().item())
        selected_regrets.append(
            float((eft[action] - legal_eft.min()).clamp_min(0.0).detach().cpu().item())
        )
        match = float(action == best_action)
        selected_matches.append(match)
        if legal_count >= 2:
            sorted_eft = legal_eft.sort().values
            margin = float((sorted_eft[1] - sorted_eft[0]).detach().cpu().item())
            if margin >= 5.0:
                margin5_matches.append(match)
            if margin >= 20.0:
                margin20_matches.append(match)
    return {
        "eft_rollout_decision_count": int(len(selected_regrets)),
        "eft_rollout_zero_legal_count": int(zero_legal),
        "eft_rollout_single_legal_count": int(single_legal),
        "eft_rollout_chosen_raw_regret_mean": (
            float(sum(selected_regrets) / len(selected_regrets)) if selected_regrets else 0.0
        ),
        "eft_rollout_chosen_raw_regret_p95": _percentile_nearest(
            selected_regrets, 0.95
        ),
        "eft_rollout_greedy_agreement": (
            float(sum(selected_matches) / len(selected_matches)) if selected_matches else 0.0
        ),
        "eft_rollout_margin_ge_5s_accuracy": (
            float(sum(margin5_matches) / len(margin5_matches)) if margin5_matches else None
        ),
        "eft_rollout_margin_ge_20s_accuracy": (
            float(sum(margin20_matches) / len(margin20_matches)) if margin20_matches else None
        ),
    }


def _percentile_nearest(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = int(math.ceil(float(quantile) * len(ordered))) - 1
    return float(ordered[max(0, min(index, len(ordered) - 1))])
