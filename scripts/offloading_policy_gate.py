from __future__ import annotations

from collections import Counter
from functools import wraps
import hashlib
import math
from typing import Any

import numpy as np

from environment.assignment import (
    CleanAssignmentBuffer,
    TemporaryReservationState,
    build_offloading_candidate_components,
)


OFFLOADING_POLICIES = (
    "actor_argmax",
    "greedy_eft_teacher",
    "shortest_queue",
    "random_hash",
)
OFFLOADING_GATE_SCHEMA_VERSION = 1
RANDOM_HASH_VERSION = "sha256-v1"


def _torch_no_grad(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        import torch

        with torch.no_grad():
            return function(*args, **kwargs)

    return wrapped


@_torch_no_grad
def select_eval_offloading_actions(
    *,
    policy: str,
    offloading_actor: Any,
    frozen_ready_tasks: list[Any],
    task_embeddings: torch.Tensor | np.ndarray,
    graph_snapshot: Any,
    task_manager: Any,
    uavs: list[Any],
    executor: Any,
    current_time_seconds: float,
    environment_seed: int,
    episode: int,
    slot: int,
    checkpoint_path: str,
    checkpoint_model_seed: int | None,
    uav_service_positions: dict[int, Any] | None = None,
    ue_service_positions: dict[int, Any] | None = None,
    ues: list[Any] | None = None,
) -> tuple[CleanAssignmentBuffer, list[dict[str, Any]]]:
    """Select evaluation assignments without changing the training actor path."""
    import torch

    if policy not in OFFLOADING_POLICIES:
        raise ValueError(f"unsupported offloading policy: {policy}")

    device = next(offloading_actor.parameters()).device
    embedding_tensor = torch.as_tensor(task_embeddings, dtype=torch.float32, device=device)
    if embedding_tensor.dim() != 2 or embedding_tensor.shape[1] != offloading_actor.task_embedding_dim:
        raise ValueError("task_embeddings shape does not match task_embedding_dim")

    reservation = TemporaryReservationState.from_executor(uavs, executor)
    assignments = CleanAssignmentBuffer()
    decisions: list[dict[str, Any]] = []
    for decision_order, task in enumerate(frozen_ready_tasks):
        task_idx = graph_snapshot.task_id_to_idx.get(task.task_id)
        if task_idx is None:
            continue
        dynamic_np, pair_np, mask_np, candidate_uav_ids, estimates = build_offloading_candidate_components(
            task=task,
            uavs=uavs,
            task_manager=task_manager,
            executor=executor,
            state_view=reservation,
            current_time_seconds=float(current_time_seconds),
            uav_service_positions=uav_service_positions,
            ue_service_positions=ue_service_positions,
            ues=ues,
        )
        if dynamic_np.shape[0] == 0 or not bool(mask_np.any()):
            continue

        task_embedding = embedding_tensor[int(task_idx)].detach().cpu().numpy().reshape(1, -1)
        candidate_features_np = np.concatenate(
            [np.repeat(task_embedding, dynamic_np.shape[0], axis=0), dynamic_np, pair_np],
            axis=1,
        ).astype(np.float32)
        candidate_features = torch.as_tensor(candidate_features_np, dtype=torch.float32, device=device)
        mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
        logits = offloading_actor.scorer(candidate_features)
        masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        probabilities = torch.softmax(masked_logits, dim=-1)

        legal_indices = [idx for idx, legal in enumerate(mask_np.tolist()) if bool(legal)]
        actor_idx = int(torch.argmax(masked_logits).item())
        greedy_idx = min(
            legal_indices,
            key=lambda idx: (float(estimates[idx].estimated_finish_time), int(candidate_uav_ids[idx])),
        )
        shortest_idx = min(
            legal_indices,
            key=lambda idx: (
                int(reservation.queue_lengths.get(int(candidate_uav_ids[idx]), 0)),
                float(estimates[idx].estimated_finish_time),
                int(candidate_uav_ids[idx]),
            ),
        )
        if policy == "actor_argmax":
            selected_idx = actor_idx
        elif policy == "greedy_eft_teacher":
            selected_idx = greedy_idx
        elif policy == "shortest_queue":
            selected_idx = shortest_idx
        else:
            selected_idx = legal_indices[
                stable_random_hash_index(
                    environment_seed=environment_seed,
                    episode=episode,
                    slot=slot,
                    task_id=str(task.task_id),
                    legal_uav_ids=[int(candidate_uav_ids[idx]) for idx in legal_indices],
                )
            ]

        selected_uav_id = int(candidate_uav_ids[selected_idx])
        selected_estimate = estimates[selected_idx]
        min_finish = min(float(estimates[idx].estimated_finish_time) for idx in legal_indices)
        queue_lengths = [int(reservation.queue_lengths.get(int(uav_id), 0)) for uav_id in candidate_uav_ids]
        available_times = [float(reservation.available_times.get(int(uav_id), 0.0)) for uav_id in candidate_uav_ids]
        queued_workloads = [float(reservation.queued_workloads.get(int(uav_id), 0.0)) for uav_id in candidate_uav_ids]
        normalized_entropy, margin = _actor_distribution_diagnostics(probabilities, legal_indices)
        ready_time = getattr(task, "ready_time", None)
        decisions.append(
            {
                "schema_version": OFFLOADING_GATE_SCHEMA_VERSION,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_model_seed": checkpoint_model_seed,
                "environment_seed": int(environment_seed),
                "episode": int(episode),
                "slot": int(slot),
                "decision_order": int(decision_order),
                "task_id": str(task.task_id),
                "dag_id": str(task.dag_id),
                "policy": str(policy),
                "candidate_uav_ids": [int(value) for value in candidate_uav_ids],
                "candidate_mask": [bool(value) for value in mask_np.tolist()],
                "valid_candidate_count": int(len(legal_indices)),
                "queue_lengths": queue_lengths,
                "available_times": available_times,
                "queued_workloads": queued_workloads,
                "actor_logits": [float(value) for value in logits.detach().cpu().tolist()],
                "actor_probabilities": [float(value) for value in probabilities.detach().cpu().tolist()],
                "actor_normalized_entropy": normalized_entropy,
                "actor_top1_top2_margin": margin,
                "estimated_finish_times": [
                    float(estimate.estimated_finish_time) if bool(mask_np[idx]) else None
                    for idx, estimate in enumerate(estimates)
                ],
                "estimated_delays": [
                    max(float(estimate.estimated_finish_time) - float(current_time_seconds), 0.0)
                    if bool(mask_np[idx])
                    else None
                    for idx, estimate in enumerate(estimates)
                ],
                "actor_selected_uav_id": int(candidate_uav_ids[actor_idx]),
                "greedy_eft_selected_uav_id": int(candidate_uav_ids[greedy_idx]),
                "shortest_queue_selected_uav_id": int(candidate_uav_ids[shortest_idx]),
                "selected_uav_id": selected_uav_id,
                "selected_estimated_regret": max(
                    float(selected_estimate.estimated_finish_time) - float(min_finish), 0.0
                ),
                "selected_estimated_finish": float(selected_estimate.estimated_finish_time),
                "task_ready_time": None if ready_time is None else float(ready_time),
                "enqueue_time": float(current_time_seconds),
                "realized_assignment_time": None,
                "realized_start_time": None,
                "realized_compute_finish_time": None,
                "realized_final_finish_time": None,
                "realized_upload_time": None,
                "realized_inter_transfer_time": None,
                "realized_queue_resource_wait": None,
                "realized_compute_time": None,
                "realized_return_time": None,
                "selected_finish_error": None,
                "selected_completed": False,
                "selected_realization_status": "pending",
            }
        )
        assignments.append(str(task.task_id), selected_uav_id, int(decision_order))
        reservation.reserve(
            str(task.task_id),
            selected_uav_id,
            estimated_available_time=float(selected_estimate.estimated_finish_time),
            estimated_queued_workload=float(selected_estimate.estimated_queued_workload),
        )

    return assignments, decisions


def stable_random_hash_index(
    *,
    environment_seed: int,
    episode: int,
    slot: int,
    task_id: str,
    legal_uav_ids: list[int],
) -> int:
    if not legal_uav_ids:
        raise ValueError("legal_uav_ids must not be empty")
    canonical = "|".join(
        [
            RANDOM_HASH_VERSION,
            str(int(environment_seed)),
            str(int(episode)),
            str(int(slot)),
            str(task_id),
            ",".join(str(int(uav_id)) for uav_id in legal_uav_ids),
        ]
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % len(legal_uav_ids)


def finalize_decision_realizations(decisions: list[dict[str, Any]], *, env: Any) -> None:
    """Attach outcomes for selected tasks only; unselected candidates stay counterfactual-free."""
    records = getattr(env.executor, "task_records", {})
    for decision in decisions:
        record = records.get(str(decision["task_id"]))
        if record is None:
            decision["selected_realization_status"] = "not_committed"
            continue
        assignment_time = float(record.assignment_time)
        start_time = float(record.start_time)
        upload_time = float(record.upload_time)
        inter_transfer_time = float(record.inter_transfer_time)
        completed = bool(record.completed)
        decision.update(
            {
                "realized_assignment_time": assignment_time,
                "realized_start_time": start_time,
                "realized_compute_finish_time": float(record.compute_finish_time),
                "realized_final_finish_time": float(record.finish_time) if completed else None,
                "realized_upload_time": upload_time,
                "realized_inter_transfer_time": inter_transfer_time,
                "realized_queue_resource_wait": max(
                    start_time - assignment_time - upload_time - inter_transfer_time,
                    0.0,
                ),
                "realized_compute_time": float(record.compute_time),
                "realized_return_time": float(record.return_time),
                "selected_finish_error": (
                    float(record.finish_time) - float(decision["selected_estimated_finish"])
                    if completed
                    else None
                ),
                "selected_completed": completed,
                "selected_realization_status": "completed" if completed else "unfinished_at_eval_end",
            }
        )


def summarize_offloading_decisions(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_counts = Counter(str(int(row["valid_candidate_count"])) for row in decisions)
    errors = [float(row["selected_finish_error"]) for row in decisions if row.get("selected_finish_error") is not None]
    abs_errors = [abs(value) for value in errors]
    agreement_count = sum(
        row.get("actor_selected_uav_id") == row.get("greedy_eft_selected_uav_id") for row in decisions
    )
    return {
        "valid_candidate_count_distribution": dict(sorted(candidate_counts.items(), key=lambda item: int(item[0]))),
        "actor_normalized_entropy_mean": _mean(row.get("actor_normalized_entropy") for row in decisions),
        "actor_entropy_sample_count": int(
            sum(row.get("actor_normalized_entropy") is not None for row in decisions)
        ),
        "actor_top1_top2_margin_mean": _mean(row.get("actor_top1_top2_margin") for row in decisions),
        "actor_margin_sample_count": int(
            sum(row.get("actor_top1_top2_margin") is not None for row in decisions)
        ),
        "actor_greedy_agreement_rate": _fraction(
            row.get("actor_selected_uav_id") == row.get("greedy_eft_selected_uav_id") for row in decisions
        ),
        "actor_greedy_agreement_count": int(agreement_count),
        "actor_greedy_comparison_count": int(len(decisions)),
        "selected_estimated_regret_mean": _mean(row.get("selected_estimated_regret") for row in decisions),
        "selected_estimated_regret_p90": _percentile(
            [float(row["selected_estimated_regret"]) for row in decisions], 90.0
        ),
        "estimator_calibration_count": int(len(errors)),
        "estimator_calibration_mae": _mean(abs_errors),
        "estimator_calibration_bias": _mean(errors),
        "estimator_calibration_p90_abs_error": _percentile(abs_errors, 90.0),
        "realized_cross_uav_transfer_time": float(
            sum(float(row.get("realized_inter_transfer_time") or 0.0) for row in decisions)
        ),
        "realized_queue_resource_wait": float(
            sum(float(row.get("realized_queue_resource_wait") or 0.0) for row in decisions)
        ),
        "_estimator_error_samples": errors,
        "_selected_estimated_regret_samples": [
            float(row["selected_estimated_regret"]) for row in decisions
        ],
        "_actor_entropy_samples": [
            float(row["actor_normalized_entropy"])
            for row in decisions
            if row.get("actor_normalized_entropy") is not None
        ],
        "_actor_margin_samples": [
            float(row["actor_top1_top2_margin"])
            for row in decisions
            if row.get("actor_top1_top2_margin") is not None
        ],
    }


def _actor_distribution_diagnostics(probabilities: torch.Tensor, legal_indices: list[int]) -> tuple[float | None, float | None]:
    import torch

    if not legal_indices:
        return None, None
    legal_probs = probabilities[torch.as_tensor(legal_indices, dtype=torch.long, device=probabilities.device)]
    if len(legal_indices) < 2:
        return None, None
    entropy = -torch.sum(legal_probs * torch.log(torch.clamp(legal_probs, min=1e-12)))
    normalized_entropy = float((entropy / math.log(len(legal_indices))).item())
    top_two = torch.topk(legal_probs, k=2).values
    return normalized_entropy, float((top_two[0] - top_two[1]).item())


def _mean(values: Any) -> float | None:
    resolved = [float(value) for value in values if value is not None]
    return float(np.mean(resolved)) if resolved else None


def _fraction(values: Any) -> float | None:
    resolved = [bool(value) for value in values]
    return float(sum(resolved) / len(resolved)) if resolved else None


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if values else None
