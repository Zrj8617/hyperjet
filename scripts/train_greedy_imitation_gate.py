from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.assignment import (
    CleanAssignmentBuffer,
    TemporaryReservationState,
    build_offloading_candidate_components,
)
from environment.env import Env
from environment.graph_builder import CleanGraphBuilder, CleanGraphSnapshot
from marl_models.mappo.clean_slot_orchestrator import prepare_slot_state
from scripts.offloading_policy_gate import RANDOM_HASH_VERSION, stable_random_hash_index


TRAJECTORY_POLICIES = ("greedy_eft", "random_hash")
TASK_ENCODERS = (
    "hgnn",
    "mlp",
    "current_mean_hgnn",
    "standard_weighted_hgnn",
    "typed_gated_hgnn",
)
MARGIN_GROUPS = (
    "trivial",
    "nearly_tied_margin_lt_1s",
    "weak_margin_1_5s",
    "clear_margin_5_20s",
    "strong_margin_ge_20s",
)


@dataclass(slots=True)
class GateModules:
    task_encoder: Any
    offloading_actor: Any
    optimizer: Any
    torch: Any
    functional: Any
    device: Any
    gradient_batch_decisions: int
    max_grad_norm: float | None


@dataclass(slots=True)
class MetricAccumulator:
    decision_count: int = 0
    valid_slot_ids: set[tuple[str, int, int]] = field(default_factory=set)
    skipped_no_candidate: int = 0
    top1_count: int = 0
    top2_count: int = 0
    masked_random_accuracy_sum: float = 0.0
    cross_entropy_sum: float = 0.0
    regret_values: list[float] = field(default_factory=list)
    rank_corr_values: list[float] = field(default_factory=list)
    finite_loss: bool = True
    finite_logits: bool = True

    def update(self, result: dict[str, Any]) -> None:
        self.decision_count += 1
        self.valid_slot_ids.add(
            (
                str(result["trajectory_policy"]),
                int(result["episode"]),
                int(result["slot"]),
            )
        )
        valid_count = int(result["valid_candidate_count"])
        self.top1_count += int(bool(result["top1_correct"]))
        self.top2_count += int(bool(result["top2_correct"]))
        self.masked_random_accuracy_sum += 1.0 / float(max(valid_count, 1))
        self.cross_entropy_sum += float(result["cross_entropy"])
        self.regret_values.append(float(result["eft_regret"]))
        self.finite_loss = self.finite_loss and bool(result["finite_loss"])
        self.finite_logits = self.finite_logits and bool(result["finite_logits"])
        rank_corr = result.get("logit_neg_eft_rank_correlation")
        if rank_corr is not None and math.isfinite(float(rank_corr)):
            self.rank_corr_values.append(float(rank_corr))

    def merge_skip(self, skipped: int) -> None:
        self.skipped_no_candidate += int(skipped)

    def as_dict(self) -> dict[str, Any]:
        regrets = np.asarray(self.regret_values, dtype=np.float64)
        rank_corrs = np.asarray(self.rank_corr_values, dtype=np.float64)
        random_acc = self.masked_random_accuracy_sum / float(max(self.decision_count, 1))
        top1 = self.top1_count / float(max(self.decision_count, 1))
        top2 = self.top2_count / float(max(self.decision_count, 1))
        return {
            "decision_count": int(self.decision_count),
            "valid_slot_count": int(len(self.valid_slot_ids)),
            "skipped_no_candidate": int(self.skipped_no_candidate),
            "cross_entropy": (
                float(self.cross_entropy_sum / float(self.decision_count))
                if self.decision_count
                else None
            ),
            "nll": (
                float(self.cross_entropy_sum / float(self.decision_count))
                if self.decision_count
                else None
            ),
            "top1_accuracy": float(top1),
            "top2_accuracy": float(top2),
            "masked_random_accuracy": float(random_acc),
            "accuracy_uplift_over_masked_random": float(top1 - random_acc),
            "mean_eft_regret": float(regrets.mean()) if regrets.size else None,
            "median_eft_regret": float(np.median(regrets)) if regrets.size else None,
            "p95_eft_regret": float(np.percentile(regrets, 95)) if regrets.size else None,
            "logit_neg_eft_rank_correlation": (
                float(rank_corrs.mean()) if rank_corrs.size else None
            ),
            "finite_loss": bool(self.finite_loss),
            "finite_logits": bool(self.finite_logits),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a supervised greedy-EFT imitation gate for the current clean "
            "offloading actor input path. This does not use PPO, reward, critic, "
            "GAE, or movement actor training."
        )
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps-per-episode", type=int, default=int(config.EPISODE_LENGTH))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trajectory-policies", nargs="+", choices=TRAJECTORY_POLICIES, default=list(TRAJECTORY_POLICIES))
    parser.add_argument("--task-encoder", choices=TASK_ENCODERS, default="hgnn")
    parser.add_argument("--task-embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--supervised-epochs", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--gradient-batch-decisions", type=int, default=64)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--completed-dag-weight", type=float, default=16.0)
    parser.add_argument("--dag-base-arrival-prob", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("logs") / "greedy_imitation_gate")
    parser.add_argument("--run-name", type=str, default="greedy_imitation_gate")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-samples", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--sample-save-limit",
        type=int,
        default=-1,
        help="Maximum decision samples to write; negative means all samples.",
    )
    parser.add_argument("--closed-loop-eval-episodes", type=int, default=100)
    parser.add_argument("--closed-loop-eval-seed", type=int, default=None)
    parser.add_argument("--skip-closed-loop-eval", action="store_true", default=False)
    parser.add_argument("--smoke", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args = _apply_smoke_overrides(args)
    _validate_args(args)
    if args.dag_base_arrival_prob is not None:
        config.DAG_BASE_ARRIVAL_PROB = float(args.dag_base_arrival_prob)
    run_dir = _create_run_dir(args)
    _write_json(run_dir / "config.json", _build_config(args, run_dir))
    _write_json(run_dir / "run_summary.json", {"status": "initialized", "run_dir": str(run_dir)})

    try:
        modules = _build_modules(args)
    except ModuleNotFoundError as exc:
        _write_json(
            run_dir / "run_summary.json",
            {
                "status": "torch_unavailable",
                "run_dir": str(run_dir),
                "error": str(exc),
                "torch_required": True,
            },
        )
        print("greedy imitation gate skipped: torch is not installed")
        return 2

    sample_writer = _SampleWriter(
        run_dir / "decision_samples.jsonl",
        enabled=bool(args.save_samples),
        limit=int(args.sample_save_limit),
    )
    train_logger = _JsonlWriter(run_dir / "train_metrics.jsonl")
    try:
        summary = _run_gate(args=args, run_dir=run_dir, modules=modules, sample_writer=sample_writer, train_logger=train_logger)
        _save_checkpoint(run_dir / "checkpoints" / "imitation_model.pt", args=args, modules=modules)
        if not bool(args.skip_closed_loop_eval):
            summary["closed_loop_eval"] = _run_closed_loop_eval(args=args, run_dir=run_dir, modules=modules)
        summary["status"] = "completed"
        _write_json(run_dir / "run_summary.json", summary)
        return 0
    except Exception as exc:
        _write_json(
            run_dir / "run_summary.json",
            {
                "status": "failed",
                "run_dir": str(run_dir),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finally:
        sample_writer.close()
        train_logger.close()


def _run_gate(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    modules: GateModules,
    sample_writer: "_SampleWriter",
    train_logger: "_JsonlWriter",
) -> dict[str, Any]:
    split_bounds = _split_bounds(int(args.episodes), float(args.train_fraction), float(args.val_fraction))
    train_summary: dict[str, Any] = {}
    for epoch in range(int(args.supervised_epochs)):
        split_metrics = _make_metric_tree()
        train_loss = _LossAccumulator()
        _generate_decisions(
            args=args,
            modules=modules,
            split_name="train",
            episode_range=range(split_bounds["train"][0], split_bounds["train"][1]),
            train_enabled=True,
            epoch_index=epoch,
            split_metrics=split_metrics,
            sample_writer=sample_writer if epoch == 0 else None,
            loss_accumulator=train_loss,
        )
        train_loss.flush(modules)
        epoch_summary = _summarize_metric_tree(split_metrics)
        epoch_summary["epoch"] = int(epoch)
        epoch_summary["loss_mean"] = train_loss.loss_mean
        epoch_summary["optimizer_steps"] = int(train_loss.optimizer_steps)
        train_logger.write(epoch_summary)
        train_summary[f"epoch_{epoch}"] = epoch_summary

    eval_metrics = _make_metric_tree()
    for split_name in ("val", "test"):
        start, end = split_bounds[split_name]
        if start >= end:
            continue
        _generate_decisions(
            args=args,
            modules=modules,
            split_name=split_name,
            episode_range=range(start, end),
            train_enabled=False,
            epoch_index=int(args.supervised_epochs),
            split_metrics=eval_metrics,
            sample_writer=sample_writer,
            loss_accumulator=None,
        )

    eval_summary = _summarize_metric_tree(eval_metrics)
    split_summary = _build_split_summary(
        split_bounds=split_bounds,
        train_summary=train_summary,
        eval_summary=eval_summary,
        supervised_epochs=int(args.supervised_epochs),
    )
    _write_json(run_dir / "imitation_split_summary.json", split_summary)
    return {
        "schema": "greedy_imitation_gate_v1",
        "run_dir": str(run_dir),
        "git_commit": _git_commit(),
        "task_encoder": str(args.task_encoder),
        "trajectory_policies": list(args.trajectory_policies),
        "split_bounds": split_bounds,
        "train_summary": train_summary,
        "eval_summary": eval_summary,
        "split_summary": split_summary,
    }


def _generate_decisions(
    *,
    args: argparse.Namespace,
    modules: GateModules,
    split_name: str,
    episode_range: range,
    train_enabled: bool,
    epoch_index: int,
    split_metrics: dict[str, MetricAccumulator],
    sample_writer: "_SampleWriter | None",
    loss_accumulator: "_LossAccumulator | None",
) -> None:
    active_episodes = set(int(item) for item in episode_range)
    if not active_episodes:
        return
    stop_episode = int(episode_range.stop)
    for policy in args.trajectory_policies:
        env, graph_builder = _new_seeded_env(args)
        try:
            for episode in range(stop_episode):
                record_episode = int(episode) in active_episodes
                env.reset()
                graph_builder.reset()
                for slot in range(int(args.max_steps_per_episode)):
                    prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
                    env.apply_movement({})
                    ready_tasks = [
                        env.task_manager.get_task(task_id)
                        for task_id in prepared.frozen_ready_task_ids
                    ]
                    ready_tasks = [task for task in ready_tasks if task is not None and task.is_ready]
                    assignment_buffer = CleanAssignmentBuffer()
                    reservation = TemporaryReservationState.from_executor(env.uavs, env.executor)
                    skipped_no_candidate = 0
                    for decision_order, task in enumerate(ready_tasks):
                        sample = _build_decision_sample(
                            trajectory_policy=str(policy),
                            prepared=prepared,
                            task=task,
                            decision_order=int(decision_order),
                            reservation=reservation,
                            env=env,
                            environment_seed=int(args.seed),
                            episode=int(episode),
                            slot=int(slot),
                        )
                        if sample is None:
                            skipped_no_candidate += 1
                            continue
                        if record_episode:
                            result = _score_sample(modules, sample, train_enabled=train_enabled)
                            _update_metric_tree(split_metrics, split_name, sample, result)
                            if sample_writer is not None:
                                sample_writer.write(
                                    {
                                        **sample,
                                        "split": split_name,
                                        "epoch": int(epoch_index),
                                        "model_prediction_idx": int(result["predicted_idx"]),
                                        "model_eft_regret": float(result["eft_regret"]),
                                    }
                                )
                            if train_enabled and loss_accumulator is not None:
                                loss_accumulator.add(result["loss"], modules)

                        behavior_idx = int(sample["behavior_idx"])
                        selected_uav_id = int(sample["candidate_uav_ids"][behavior_idx])
                        selected_finish = float(sample["estimated_finish_times"][behavior_idx])
                        selected_workload = float(sample["estimated_queued_workloads"][behavior_idx])
                        assignment_buffer.append(str(task.task_id), selected_uav_id, int(decision_order))
                        reservation.reserve(
                            str(task.task_id),
                            selected_uav_id,
                            estimated_available_time=selected_finish,
                            estimated_queued_workload=selected_workload,
                        )
                    if record_episode:
                        _merge_skip(split_metrics, split_name, str(policy), skipped_no_candidate)
                    _, _, done, _ = env.commit_and_advance(
                        assignment_buffer=assignment_buffer,
                        offloading_skip_count=skipped_no_candidate,
                    )
                    if done:
                        break
        finally:
            graph_builder.close()


def _build_decision_sample(
    *,
    trajectory_policy: str,
    prepared: Any,
    task: Any,
    decision_order: int,
    reservation: TemporaryReservationState,
    env: Env,
    environment_seed: int,
    episode: int,
    slot: int,
) -> dict[str, Any] | None:
    dynamic, pair, mask, candidate_uav_ids, estimates = build_offloading_candidate_components(
        task=task,
        uavs=env.uavs,
        task_manager=env.task_manager,
        executor=env.executor,
        state_view=reservation,
        current_time_seconds=float(env.current_time_seconds),
        uav_service_positions=env.uav_service_positions,
        ue_service_positions=env.ue_service_positions,
        ues=env.ues,
    )
    legal_indices = [idx for idx, legal in enumerate(mask.tolist()) if bool(legal)]
    if not legal_indices:
        return None
    greedy_idx = min(
        legal_indices,
        key=lambda idx: (
            float(estimates[idx].estimated_finish_time),
            int(candidate_uav_ids[idx]),
        ),
    )
    if trajectory_policy == "greedy_eft":
        behavior_idx = greedy_idx
    elif trajectory_policy == "random_hash":
        random_offset = stable_random_hash_index(
            environment_seed=int(environment_seed),
            episode=int(episode),
            slot=int(slot),
            task_id=str(task.task_id),
            legal_uav_ids=[int(candidate_uav_ids[idx]) for idx in legal_indices],
        )
        behavior_idx = legal_indices[random_offset]
    else:
        raise ValueError(f"unsupported trajectory policy: {trajectory_policy}")

    snapshot: CleanGraphSnapshot = prepared.graph_snapshot
    task_idx = snapshot.task_id_to_idx.get(str(task.task_id))
    if task_idx is None:
        return None
    eft_values = [float(item.estimated_finish_time) for item in estimates]
    workload_values = [float(item.estimated_queued_workload) for item in estimates]
    trajectory_id = _trajectory_id(
        trajectory_policy=trajectory_policy,
        environment_seed=int(environment_seed),
        episode=int(episode),
    )
    sample_id = _sample_id(
        trajectory_id=trajectory_id,
        slot=int(slot),
        decision_order=int(decision_order),
        task_id=str(task.task_id),
    )
    return {
        "schema": "greedy_imitation_decision_sample_v1",
        "sample_id": sample_id,
        "trajectory_policy": str(trajectory_policy),
        "label_mode": "behavior_conditioned_greedy_eft",
        "seed": int(environment_seed),
        "environment_seed": int(environment_seed),
        "episode": int(episode),
        "episode_id": int(episode),
        "trajectory_id": trajectory_id,
        "slot": int(slot),
        "decision_order": int(decision_order),
        "task_id": str(task.task_id),
        "dag_id": str(task.dag_id),
        "task_idx": int(task_idx),
        "task_local_index": int(task_idx),
        "active_task_ids": list(snapshot.active_task_ids),
        "ready_task_ids": list(snapshot.ready_task_ids),
        "pending_task_ids": list(snapshot.pending_task_ids),
        "idx_to_task_id": dict(snapshot.idx_to_task_id),
        "task_features": np.asarray(snapshot.task_features, dtype=np.float32).copy(),
        "incidence_matrix": np.asarray(snapshot.incidence_matrix, dtype=np.float32).copy(),
        "hyperedge_type_ids": np.asarray(snapshot.hyperedge_type_ids, dtype=np.int64).copy(),
        "hyperedges": [list(edge) for edge in snapshot.hyperedges],
        "dag_hyperedges": [list(edge) for edge in snapshot.dag_hyperedges],
        "khop_hyperedges": [list(edge) for edge in snapshot.khop_hyperedges],
        "attribute_hyperedges": [list(edge) for edge in snapshot.attribute_hyperedges],
        "partition_hyperedges": [list(edge) for edge in snapshot.partition_hyperedges],
        "task_id_to_idx": dict(snapshot.task_id_to_idx),
        "frozen_ready_task_ids": list(prepared.frozen_ready_task_ids),
        "dynamic_uav_features": np.asarray(dynamic, dtype=np.float32).copy(),
        "pair_features": np.asarray(pair, dtype=np.float32).copy(),
        "candidate_mask": np.asarray(mask, dtype=bool).copy(),
        "candidate_uav_ids": [int(item) for item in candidate_uav_ids],
        "candidate_uav_id_mapping": [int(item) for item in candidate_uav_ids],
        "greedy_label_idx": int(greedy_idx),
        "behavior_idx": int(behavior_idx),
        "estimated_finish_times": eft_values,
        "estimated_queued_workloads": workload_values,
        "valid_candidate_count": int(len(legal_indices)),
        "greedy_margin": _greedy_margin(eft_values, legal_indices),
    }


def _score_sample(modules: GateModules, sample: dict[str, Any], *, train_enabled: bool) -> dict[str, Any]:
    torch = modules.torch
    with torch.set_grad_enabled(bool(train_enabled)):
        logits = _sample_logits(modules, sample)
        mask = torch.as_tensor(sample["candidate_mask"], dtype=torch.bool, device=modules.device)
        label = torch.as_tensor([int(sample["greedy_label_idx"])], dtype=torch.long, device=modules.device)
        masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        loss = modules.functional.cross_entropy(masked_logits.unsqueeze(0), label)
    finite_logits = bool(torch.isfinite(logits).all().item())
    finite_loss = bool(torch.isfinite(loss).all().item())
    if not finite_logits or not finite_loss:
        raise FloatingPointError("non-finite imitation logits or loss")

    valid_indices = [idx for idx, legal in enumerate(sample["candidate_mask"]) if bool(legal)]
    valid_logits = masked_logits.detach()[valid_indices]
    order = torch.argsort(valid_logits, descending=True).detach().cpu().numpy().tolist()
    ranked_indices = [valid_indices[int(idx)] for idx in order]
    prediction = int(ranked_indices[0])
    label_idx = int(sample["greedy_label_idx"])
    top2 = set(ranked_indices[: min(2, len(ranked_indices))])
    eft = [float(value) for value in sample["estimated_finish_times"]]
    rank_corr = _rank_correlation(
        valid_logits.detach().cpu().numpy().astype(float).tolist(),
        [-eft[idx] for idx in valid_indices],
    )
    return {
        "loss": loss,
        "cross_entropy": float(loss.detach().cpu().item()),
        "finite_loss": finite_loss,
        "finite_logits": finite_logits,
        "predicted_idx": prediction,
        "top1_correct": prediction == label_idx,
        "top2_correct": label_idx in top2,
        "eft_regret": float(eft[prediction] - eft[label_idx]),
        "valid_candidate_count": int(sample["valid_candidate_count"]),
        "trajectory_policy": str(sample["trajectory_policy"]),
        "episode": int(sample["episode"]),
        "slot": int(sample["slot"]),
        "logit_neg_eft_rank_correlation": rank_corr,
    }


def _sample_logits(modules: GateModules, sample: dict[str, Any]) -> Any:
    torch = modules.torch
    task_features = torch.as_tensor(sample["task_features"], dtype=torch.float32, device=modules.device)
    incidence = torch.as_tensor(sample["incidence_matrix"], dtype=torch.float32, device=modules.device)
    hyperedge_type_ids = torch.as_tensor(sample["hyperedge_type_ids"], dtype=torch.long, device=modules.device)
    embeddings = modules.task_encoder(task_features, incidence, hyperedge_type_ids)
    task_embedding = embeddings[int(sample["task_local_index"])].reshape(1, -1)
    dynamic = torch.as_tensor(sample["dynamic_uav_features"], dtype=torch.float32, device=modules.device)
    pair = torch.as_tensor(sample["pair_features"], dtype=torch.float32, device=modules.device)
    repeated = task_embedding.repeat(dynamic.shape[0], 1)
    candidate_features = torch.cat([repeated, dynamic, pair], dim=1)
    return modules.offloading_actor.scorer(candidate_features)


class _LossAccumulator:
    def __init__(self) -> None:
        self.loss: Any | None = None
        self.count = 0
        self.total_loss = 0.0
        self.loss_samples = 0
        self.optimizer_steps = 0
        self.gradient_norm_values: list[float] = []
        self.finite_gradient_norm = True

    def add(self, loss: Any, modules: GateModules) -> None:
        self.loss = loss if self.loss is None else self.loss + loss
        self.count += 1
        self.total_loss += float(loss.detach().cpu().item())
        self.loss_samples += 1
        if self.count >= int(getattr(modules, "gradient_batch_decisions", 64)):
            self.flush(modules)

    def flush(self, modules: GateModules) -> None:
        if self.loss is None or self.count <= 0:
            return
        modules.optimizer.zero_grad(set_to_none=True)
        (self.loss / float(self.count)).backward()
        parameters = list(modules.task_encoder.parameters()) + list(modules.offloading_actor.scorer.parameters())
        max_norm = float(modules.max_grad_norm) if modules.max_grad_norm is not None else float("inf")
        gradient_norm = modules.torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        gradient_norm_value = float(gradient_norm.detach().cpu().item())
        self.gradient_norm_values.append(gradient_norm_value)
        self.finite_gradient_norm = self.finite_gradient_norm and math.isfinite(gradient_norm_value)
        if not math.isfinite(gradient_norm_value):
            raise FloatingPointError("non-finite imitation gradient norm")
        modules.optimizer.step()
        self.optimizer_steps += 1
        self.loss = None
        self.count = 0

    @property
    def loss_mean(self) -> float | None:
        if self.loss_samples <= 0:
            return None
        return float(self.total_loss / float(self.loss_samples))


def _run_closed_loop_eval(*, args: argparse.Namespace, run_dir: Path, modules: GateModules) -> dict[str, Any]:
    eval_seed = int(args.closed_loop_eval_seed if args.closed_loop_eval_seed is not None else args.seed)
    policies = ("random_hash", "greedy_eft", f"imitation_{args.task_encoder}")
    writer = _JsonlWriter(run_dir / "closed_loop_eval_metrics.jsonl")
    aggregate: dict[str, list[dict[str, Any]]] = {policy: [] for policy in policies}
    try:
        for policy in policies:
            env, graph_builder = _new_seeded_env(args, seed=eval_seed)
            try:
                for episode in range(int(args.closed_loop_eval_episodes)):
                    env.reset()
                    graph_builder.reset()
                    episode_reward = 0.0
                    accepted = 0
                    skipped = 0
                    latest_info: dict[str, Any] = {}
                    for slot in range(int(args.max_steps_per_episode)):
                        prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
                        env.apply_movement({})
                        ready_tasks = [
                            env.task_manager.get_task(task_id)
                            for task_id in prepared.frozen_ready_task_ids
                        ]
                        ready_tasks = [task for task in ready_tasks if task is not None and task.is_ready]
                        if policy == "greedy_eft" or policy == "random_hash":
                            assignment_buffer, skipped_slot = _select_static_assignments(
                                policy=policy,
                                frozen_ready_tasks=ready_tasks,
                                prepared=prepared,
                                env=env,
                                environment_seed=eval_seed,
                                episode=episode,
                                slot=slot,
                            )
                        else:
                            assignment_buffer, skipped_slot = _select_imitation_assignments(
                                modules=modules,
                                frozen_ready_tasks=ready_tasks,
                                prepared=prepared,
                                env=env,
                            )
                        _, _, done, latest_info = env.commit_and_advance(
                            assignment_buffer=assignment_buffer,
                            offloading_skip_count=skipped_slot,
                        )
                        episode_reward += float(latest_info["step_reward"])
                        accepted += int(latest_info["newly_assigned_tasks"])
                        skipped += int(skipped_slot)
                        if done:
                            break
                    row = {
                        "policy": str(policy),
                        "seed": int(eval_seed),
                        "episode": int(episode),
                        "episode_reward_total": float(episode_reward),
                        "accepted_assignments": int(accepted),
                        "offloading_skipped_no_candidate": int(skipped),
                        **_episode_metric_subset(latest_info),
                    }
                    writer.write(row)
                    aggregate[policy].append(row)
            finally:
                graph_builder.close()
    finally:
        writer.close()
    summary = {policy: _summarize_episode_rows(rows) for policy, rows in aggregate.items()}
    _write_json(run_dir / "closed_loop_eval_summary.json", summary)
    return summary


def _select_static_assignments(
    *,
    policy: str,
    frozen_ready_tasks: list[Any],
    prepared: Any,
    env: Env,
    environment_seed: int,
    episode: int,
    slot: int,
) -> tuple[CleanAssignmentBuffer, int]:
    reservation = TemporaryReservationState.from_executor(env.uavs, env.executor)
    assignments = CleanAssignmentBuffer()
    skipped = 0
    for decision_order, task in enumerate(frozen_ready_tasks):
        sample = _build_decision_sample(
            trajectory_policy=policy,
            prepared=prepared,
            task=task,
            decision_order=int(decision_order),
            reservation=reservation,
            env=env,
            environment_seed=int(environment_seed),
            episode=int(episode),
            slot=int(slot),
        )
        if sample is None:
            skipped += 1
            continue
        selected = int(sample["greedy_label_idx"] if policy == "greedy_eft" else sample["behavior_idx"])
        selected_uav_id = int(sample["candidate_uav_ids"][selected])
        assignments.append(str(task.task_id), selected_uav_id, int(decision_order))
        reservation.reserve(
            str(task.task_id),
            selected_uav_id,
            estimated_available_time=float(sample["estimated_finish_times"][selected]),
            estimated_queued_workload=float(sample["estimated_queued_workloads"][selected]),
        )
    return assignments, skipped


def _select_imitation_assignments(
    *,
    modules: GateModules,
    frozen_ready_tasks: list[Any],
    prepared: Any,
    env: Env,
) -> tuple[CleanAssignmentBuffer, int]:
    torch = modules.torch
    reservation = TemporaryReservationState.from_executor(env.uavs, env.executor)
    assignments = CleanAssignmentBuffer()
    skipped = 0
    for decision_order, task in enumerate(frozen_ready_tasks):
        sample = _build_decision_sample(
            trajectory_policy="greedy_eft",
            prepared=prepared,
            task=task,
            decision_order=int(decision_order),
            reservation=reservation,
            env=env,
            environment_seed=0,
            episode=0,
            slot=0,
        )
        if sample is None:
            skipped += 1
            continue
        with torch.no_grad():
            logits = _sample_logits(modules, sample)
            mask = torch.as_tensor(sample["candidate_mask"], dtype=torch.bool, device=modules.device)
            masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            selected = int(torch.argmax(masked_logits).detach().cpu().item())
        selected_uav_id = int(sample["candidate_uav_ids"][selected])
        assignments.append(str(task.task_id), selected_uav_id, int(decision_order))
        reservation.reserve(
            str(task.task_id),
            selected_uav_id,
            estimated_available_time=float(sample["estimated_finish_times"][selected]),
            estimated_queued_workload=float(sample["estimated_queued_workloads"][selected]),
        )
    return assignments, skipped


def _make_metric_tree() -> dict[str, MetricAccumulator]:
    return {}


def _update_metric_tree(tree: dict[str, MetricAccumulator], split: str, sample: dict[str, Any], result: dict[str, Any]) -> None:
    keys = [
        f"{split}/overall",
        f"{split}/trajectory_policy/{sample['trajectory_policy']}",
        f"{split}/valid_candidate_count/k{int(sample['valid_candidate_count'])}",
        f"{split}/margin/{_margin_group(sample)}",
    ]
    if int(sample["valid_candidate_count"]) >= 2:
        keys.append(f"{split}/nontrivial_k_ge_2")
    margin = sample.get("greedy_margin")
    if margin is not None and float(margin) >= 5.0:
        keys.append(f"{split}/margin_ge_5s")
    if margin is not None and float(margin) >= 20.0:
        keys.append(f"{split}/margin_ge_20s")
    for key in keys:
        tree.setdefault(key, MetricAccumulator()).update(result)


def _merge_skip(tree: dict[str, MetricAccumulator], split: str, policy: str, skipped: int) -> None:
    keys = [f"{split}/overall", f"{split}/trajectory_policy/{policy}"]
    for key in keys:
        tree.setdefault(key, MetricAccumulator()).merge_skip(int(skipped))


def _summarize_metric_tree(tree: dict[str, MetricAccumulator]) -> dict[str, Any]:
    return {key: value.as_dict() for key, value in sorted(tree.items())}


def _margin_group(sample: dict[str, Any]) -> str:
    if int(sample["valid_candidate_count"]) <= 1:
        return "trivial"
    margin = float(sample["greedy_margin"])
    if margin < 1.0:
        return "nearly_tied_margin_lt_1s"
    if margin < 5.0:
        return "weak_margin_1_5s"
    if margin < 20.0:
        return "clear_margin_5_20s"
    return "strong_margin_ge_20s"


def _greedy_margin(eft_values: list[float], legal_indices: list[int]) -> float | None:
    if len(legal_indices) <= 1:
        return None
    ordered = sorted(float(eft_values[idx]) for idx in legal_indices)
    return float(ordered[1] - ordered[0])


def _rank_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    rx = _rank_values(xs)
    ry = _rank_values(ys)
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _rank_values(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr)
    ranks = np.empty_like(arr)
    start = 0
    while start < len(arr):
        end = start + 1
        while end < len(arr) and arr[order[end]] == arr[order[start]]:
            end += 1
        rank = 0.5 * float(start + end - 1)
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _build_modules(
    args: argparse.Namespace,
    *,
    encoder_seed: int | None = None,
    scorer_seed: int | None = None,
) -> GateModules:
    try:
        import torch
        from torch.nn import functional
        from marl_models.hgnn import build_clean_task_encoder
        from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is required for greedy imitation training") from exc

    _set_seed(int(args.seed), torch=torch)
    configured_task_feature_dim = getattr(args, "task_feature_dim", None)
    if configured_task_feature_dim is not None:
        task_feature_dim = int(configured_task_feature_dim)
    else:
        init_env = Env(completed_dag_weight=float(args.completed_dag_weight), freeze_ue_mobility=True)
        graph_builder = CleanGraphBuilder()
        try:
            _set_seed(int(args.seed), torch=torch)
            init_env.reset()
            graph_builder.reset()
            prepared = prepare_slot_state(env=init_env, graph_builder=graph_builder)
            task_feature_dim = int(prepared.graph_snapshot.task_features.shape[1])
        finally:
            graph_builder.close()
    device = torch.device(str(args.device) if str(args.device) != "cuda" or torch.cuda.is_available() else "cpu")
    _set_seed(int(args.seed if encoder_seed is None else encoder_seed), torch=torch)
    task_encoder = build_clean_task_encoder(
        encoder_type=str(args.task_encoder),
        task_feature_dim=task_feature_dim,
        hidden_dim=int(args.hidden_dim),
        output_dim=int(args.task_embedding_dim),
    )
    if scorer_seed is not None:
        _set_seed(int(scorer_seed), torch=torch)
    offloading_actor = CleanOffloadingActor(
        task_embedding_dim=int(args.task_embedding_dim),
        hidden_dim=int(args.hidden_dim),
    )
    task_encoder.to(device)
    offloading_actor.to(device)
    optimizer = torch.optim.Adam(
        list(task_encoder.parameters()) + list(offloading_actor.scorer.parameters()),
        lr=float(args.lr),
    )
    modules = GateModules(
        task_encoder=task_encoder,
        offloading_actor=offloading_actor,
        optimizer=optimizer,
        torch=torch,
        functional=functional,
        device=device,
        gradient_batch_decisions=int(args.gradient_batch_decisions),
        max_grad_norm=float(args.max_grad_norm),
    )
    return modules


def _new_seeded_env(args: argparse.Namespace, seed: int | None = None) -> tuple[Env, CleanGraphBuilder]:
    _set_seed(int(args.seed if seed is None else seed))
    env = Env(completed_dag_weight=float(args.completed_dag_weight), freeze_ue_mobility=True)
    graph_builder = CleanGraphBuilder()
    env.reset()
    graph_builder.reset()
    prepare_slot_state(env=env, graph_builder=graph_builder)
    return env, graph_builder


def _save_checkpoint(path: Path, *, args: argparse.Namespace, modules: GateModules) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    modules.torch.save(
        {
            "schema": "greedy_imitation_gate_checkpoint_v1",
            "task_encoder": str(args.task_encoder),
            "task_embedding_dim": int(args.task_embedding_dim),
            "hidden_dim": int(args.hidden_dim),
            "task_encoder_state_dict": modules.task_encoder.state_dict(),
            "offloading_actor_state_dict": modules.offloading_actor.state_dict(),
            "config": _namespace_to_dict(args),
        },
        path,
    )


def _episode_metric_subset(info: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "generated_dag_count",
        "completed_dag_count",
        "dag_completion_rate",
        "average_dag_flowtime",
        "avg_uav_queue_length",
        "energy_per_completed_dag",
        "total_task_energy",
        "uav_movement_energy_total",
        "active_dags",
        "frozen_ready_task_count",
        "service_waiting_ues",
        "invalid_assignment_count",
    )
    return {key: info.get(key) for key in keys}


def _summarize_episode_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "episode_reward_total",
        "completed_dag_count",
        "generated_dag_count",
        "dag_completion_rate",
        "average_dag_flowtime",
        "avg_uav_queue_length",
    )
    summary: dict[str, Any] = {"episodes": int(len(rows))}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary[f"{key}_mean"] = float(arr.mean())
        summary[f"{key}_std"] = float(arr.std(ddof=0))
    return summary


def _split_bounds(episodes: int, train_fraction: float, val_fraction: float) -> dict[str, tuple[int, int]]:
    train_end = int(math.floor(float(episodes) * float(train_fraction)))
    val_end = train_end + int(math.floor(float(episodes) * float(val_fraction)))
    train_end = min(max(train_end, 0), int(episodes))
    val_end = min(max(val_end, train_end), int(episodes))
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, int(episodes)),
    }


def _trajectory_id(*, trajectory_policy: str, environment_seed: int, episode: int) -> str:
    return f"{trajectory_policy}:seed{int(environment_seed)}:episode{int(episode)}"


def _sample_id(*, trajectory_id: str, slot: int, decision_order: int, task_id: str) -> str:
    identity = {
        "trajectory_id": str(trajectory_id),
        "slot": int(slot),
        "decision_order": int(decision_order),
        "task_id": str(task_id),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _split_episode_sets(split_bounds: dict[str, tuple[int, int]]) -> dict[str, set[int]]:
    return {
        split: set(range(int(bounds[0]), int(bounds[1])))
        for split, bounds in split_bounds.items()
    }


def _split_leakage_count(split_bounds: dict[str, tuple[int, int]]) -> int:
    episode_sets = _split_episode_sets(split_bounds)
    leakage = 0
    split_names = sorted(episode_sets)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            leakage += len(episode_sets[left].intersection(episode_sets[right]))
    return int(leakage)


def _build_split_summary(
    *,
    split_bounds: dict[str, tuple[int, int]],
    train_summary: dict[str, Any],
    eval_summary: dict[str, Any],
    supervised_epochs: int,
) -> dict[str, Any]:
    split_sets = _split_episode_sets(split_bounds)
    latest_train = train_summary.get(f"epoch_{max(int(supervised_epochs) - 1, 0)}", {})
    summary: dict[str, Any] = {
        "schema": "greedy_imitation_split_summary_v1",
        "split_mode": "episode_level",
        "split_bounds": {key: [int(value[0]), int(value[1])] for key, value in split_bounds.items()},
        "leakage_count": _split_leakage_count(split_bounds),
    }
    for split in ("train", "val", "test"):
        metric_source = latest_train if split == "train" else eval_summary
        overall = metric_source.get(f"{split}/overall", {})
        episodes = sorted(int(item) for item in split_sets[split])
        summary[split] = {
            "episodes": episodes,
            "episode_count": int(len(episodes)),
            "sample_count": int(overall.get("decision_count", 0)) if isinstance(overall, dict) else 0,
        }
    return summary


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.episodes) <= 0:
        raise ValueError("episodes must be positive")
    if int(args.max_steps_per_episode) <= 0:
        raise ValueError("max-steps-per-episode must be positive")
    if int(args.supervised_epochs) <= 0:
        raise ValueError("supervised-epochs must be positive")
    if int(args.gradient_batch_decisions) <= 0:
        raise ValueError("gradient-batch-decisions must be positive")
    if not 0.0 <= float(args.train_fraction) <= 1.0:
        raise ValueError("train-fraction must be in [0, 1]")
    if not 0.0 <= float(args.val_fraction) <= 1.0:
        raise ValueError("val-fraction must be in [0, 1]")
    if float(args.train_fraction) + float(args.val_fraction) > 1.0:
        raise ValueError("train-fraction + val-fraction must be <= 1")
    if not np.isfinite(float(args.completed_dag_weight)) or float(args.completed_dag_weight) < 0.0:
        raise ValueError("completed-dag-weight must be finite and non-negative")
    if args.dag_base_arrival_prob is not None and not 0.0 <= float(args.dag_base_arrival_prob) <= 1.0:
        raise ValueError("dag-base-arrival-prob must be in [0, 1]")


def _apply_smoke_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if not bool(args.smoke):
        return args
    args.episodes = min(int(args.episodes), 4)
    args.max_steps_per_episode = min(int(args.max_steps_per_episode), 12)
    args.supervised_epochs = min(int(args.supervised_epochs), 1)
    args.closed_loop_eval_episodes = min(int(args.closed_loop_eval_episodes), 2)
    args.gradient_batch_decisions = min(int(args.gradient_batch_decisions), 4)
    if args.dag_base_arrival_prob is None:
        args.dag_base_arrival_prob = 1.0
    args.sample_save_limit = 20 if int(args.sample_save_limit) < 0 else int(args.sample_save_limit)
    return args


def _create_run_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(args.run_name)).strip("_")
    run_dir = Path(args.output_dir) / f"{timestamp}_{safe_name or 'greedy_imitation'}_{args.task_encoder}_seed{int(args.seed)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_config(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    return {
        "schema": "greedy_imitation_gate_v1",
        "run_dir": str(run_dir),
        "git_commit": _git_commit(),
        "random_hash_version": RANDOM_HASH_VERSION,
        "fixed_hover": True,
        "ppo_used": False,
        "reward_used_for_training": False,
        "critic_trained": False,
        "movement_actor_trained": False,
        "cli": _namespace_to_dict(args),
    }


def _namespace_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in vars(args).items():
        output[key] = str(value) if isinstance(value, Path) else value
    return output


def _set_seed(seed: int, torch: Any | None = None) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    if torch is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class _JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")

    def write(self, payload: dict[str, Any]) -> None:
        self.handle.write(json.dumps(_jsonable(payload), ensure_ascii=True, sort_keys=True) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class _SampleWriter:
    def __init__(self, path: Path, *, enabled: bool, limit: int) -> None:
        self.enabled = bool(enabled)
        self.limit = int(limit)
        self.count = 0
        self.writer = _JsonlWriter(path) if self.enabled else None

    def write(self, payload: dict[str, Any]) -> None:
        if self.writer is None:
            return
        if self.limit >= 0 and self.count >= self.limit:
            return
        self.writer.write(payload)
        self.count += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
