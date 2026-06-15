from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import dataclasses
import json
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

import config
from environment.env import Env
from marl_models.mappo.assignment_mappo import AssignmentMAPPO
from scripts.static_scheduler_compare import _apply_ablation_config, _override_num_uavs, _override_num_ues
from utils.progress import TerminalProgress


def _zero_actions() -> np.ndarray:
    return np.zeros((config.NUM_UAVS, config.ACTION_DIM), dtype=np.float32)


def _candidate_count_bucket(candidate_count: int) -> str:
    if candidate_count <= 0:
        return "count_0"
    if candidate_count == 1:
        return "count_1"
    if candidate_count == 2:
        return "count_2"
    return "count_3plus"


def _new_execution_bucket() -> dict[str, int]:
    return {
        "total": 0,
        "actor_called": 0,
        "executed": 0,
        "non_executed": 0,
        "no_feasible_candidate_count": 0,
        "executor_override_count": 0,
        "fallback_after_actor_count": 0,
    }


def _finalize_execution_buckets(buckets: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int | None]]:
    finalized: dict[str, dict[str, float | int | None]] = {}
    for key, bucket in sorted(buckets.items()):
        actor_called = int(bucket["actor_called"])
        finalized[str(key)] = {
            **bucket,
            "action_executed_rate": (
                float(bucket["executed"]) / float(actor_called)
                if actor_called > 0
                else None
            ),
        }
    return finalized


def summarize_assignment_execution(records: list[dict]) -> dict:
    non_executed_reason_counts: Counter[str] = Counter()
    non_executed_failure_reason_counts: Counter[str] = Counter()
    by_task_type: dict[str, dict[str, int]] = defaultdict(_new_execution_bucket)
    by_candidate_count: dict[str, dict[str, int]] = defaultdict(_new_execution_bucket)
    by_step: dict[str, dict[str, int]] = defaultdict(_new_execution_bucket)
    executor_override_count = 0
    fallback_after_actor_count = 0
    invalid_actor_action_count = 0
    no_feasible_candidate_count = 0
    for record in records:
        actor_called = bool(record.get("actor_called", True))
        action_executed = bool(record.get("action_executed", False))
        executor_selected_uav = record.get("executor_selected_uav")
        actor_selected_uav = record.get("actor_selected_uav")
        fallback_used = bool(record.get("fallback_used", False))
        overridden = (
            actor_called
            and not action_executed
            and executor_selected_uav is not None
            and executor_selected_uav != actor_selected_uav
        )
        if actor_called and not action_executed:
            reason = str(record.get("non_executed_reason") or "unknown_non_executed")
            non_executed_reason_counts[reason] += 1
            failure_reason = str(record.get("failure_reason") or "unknown_failure_reason")
            non_executed_failure_reason_counts[failure_reason] += 1
        if overridden:
            executor_override_count += 1
        fallback_after_actor = actor_called and not action_executed and fallback_used
        if fallback_after_actor:
            fallback_after_actor_count += 1
        if actor_called and not action_executed and record.get("non_executed_reason") == "invalid_actor_action":
            invalid_actor_action_count += 1
        if not actor_called and record.get("non_executed_reason") == "no_feasible_candidate":
            no_feasible_candidate_count += 1

        task_type = str(record.get("task_type") if record.get("task_type") is not None else "unknown")
        candidate_count = int(record.get("candidate_count", len(record.get("candidate_uav_ids", []))))
        candidate_bucket = _candidate_count_bucket(candidate_count)
        env_step_id = str(record.get("env_step_id", "unknown"))
        for bucket in (by_task_type[task_type], by_candidate_count[candidate_bucket], by_step[env_step_id]):
            bucket["total"] += 1
            bucket["actor_called"] += int(actor_called)
            bucket["executed"] += int(action_executed)
            bucket["non_executed"] += int(actor_called and not action_executed)
            bucket["no_feasible_candidate_count"] += int(not actor_called)
            bucket["executor_override_count"] += int(overridden)
            bucket["fallback_after_actor_count"] += int(fallback_after_actor)

    finalized_by_step = _finalize_execution_buckets(by_step)
    worst_steps = sorted(
        (
            {
                "env_step_id": int(step_id) if step_id.isdigit() else step_id,
                "executed": int(bucket["executed"]),
                "total": int(bucket["actor_called"]),
                "rate": float(bucket["action_executed_rate"]),
            }
            for step_id, bucket in finalized_by_step.items()
            if int(bucket["actor_called"]) > 0
        ),
        key=lambda item: (item["rate"], str(item["env_step_id"])),
    )[:10]
    actor_called_decisions = sum(1 for record in records if record.get("actor_called", True))
    executed_actor_actions = sum(1 for record in records if record.get("action_executed", False))
    return {
        "action_executed_rate": executed_actor_actions / float(max(actor_called_decisions, 1)),
        "actor_called_decisions": actor_called_decisions,
        "executed_actor_actions": executed_actor_actions,
        "invalid_actor_action_count": invalid_actor_action_count,
        "no_feasible_candidate_count": no_feasible_candidate_count,
        "num_assignment_decisions": actor_called_decisions,
        "num_executed_decisions": executed_actor_actions,
        "num_non_executed_decisions": actor_called_decisions - executed_actor_actions,
        "non_executed_reason_counts": dict(sorted(non_executed_reason_counts.items())),
        "non_executed_failure_reason_counts": dict(sorted(non_executed_failure_reason_counts.items())),
        "action_executed_rate_by_task_type": _finalize_execution_buckets(by_task_type),
        "action_executed_rate_by_candidate_count": _finalize_execution_buckets(by_candidate_count),
        "min_step_action_executed_rate": None if not worst_steps else worst_steps[0]["rate"],
        "worst_action_executed_steps": worst_steps,
        "executor_override_count": executor_override_count,
        "fallback_after_actor_count": fallback_after_actor_count,
    }


def _configure(args: argparse.Namespace) -> None:
    if args.num_ues is not None:
        _override_num_ues(args.num_ues)
    if args.num_uavs is not None:
        _override_num_uavs(args.num_uavs, args.seed)
    if args.dag_arrival_prob is not None:
        if not 0.0 <= args.dag_arrival_prob <= 1.0:
            raise ValueError("--dag_arrival_prob must be in [0, 1].")
        config.DAG_ARRIVAL_PROB = float(args.dag_arrival_prob)

    config.STEPS_PER_EPISODE = int(args.steps)
    config.CANDIDATE_POLICY_MODE = args.candidate_policy_mode
    config.EXPANDED_CANDIDATE_MAX_DISTANCE = float(args.expanded_candidate_max_distance)
    config.EXPANDED_CANDIDATE_MAX_QUEUE = int(args.expanded_candidate_max_queue)
    config.EXPANDED_CANDIDATE_DEADLINE_TOLERANCE = float(args.expanded_candidate_deadline_tolerance)
    config.USE_RL_ASSIGNMENT = True
    config.RL_ASSIGNMENT_LOAD_ENCODER_CHECKPOINT = bool(args.encoder_checkpoint)
    config.RL_ASSIGNMENT_TRAIN_ENCODER = False
    config.USE_HGNN_SCORE_ASSIGNMENT = False
    config.USE_SELECTIVE_HGNN_SCORING = False
    config.USE_HGNN_PER_TASK_RESCORING = False
    config.USE_SCORE_RUNTIME_BOUNDED_GUARD = False
    config.USE_SCORE_AGREEMENT_ONLY = False
    config.SCORE_FALLBACK_TO_HEURISTIC = False
    config.USE_PHASE_ONE_DAG_REWARD_SHAPING = False
    config.USE_STAGE_B_MOVEMENT_REWARD = False
    _apply_ablation_config(args.ablation)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train minimal assignment-only PPO with zero UAV movement.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--encoder_checkpoint", default="")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--num_ues", type=int, default=None)
    parser.add_argument("--num_uavs", type=int, default=None)
    parser.add_argument("--dag_arrival_prob", type=float, default=None)
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--candidate_policy_mode", choices=["strict", "expanded"], default=config.CANDIDATE_POLICY_MODE)
    parser.add_argument("--expanded_candidate_max_distance", type=float, default=config.EXPANDED_CANDIDATE_MAX_DISTANCE)
    parser.add_argument("--expanded_candidate_max_queue", type=int, default=config.EXPANDED_CANDIDATE_MAX_QUEUE)
    parser.add_argument(
        "--expanded_candidate_deadline_tolerance",
        type=float,
        default=config.EXPANDED_CANDIDATE_DEADLINE_TOLERANCE,
    )
    parser.add_argument(
        "--ablation",
        choices=["attribute_blind", "critical_only", "attribute_only", "critical_plus_attribute"],
        default="attribute_blind",
    )
    parser.add_argument("--write_alignment_debug", action="store_true")
    parser.add_argument("--non_executed_sample_limit", type=int, default=0)
    args = parser.parse_args()
    _configure(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "assignment_mappo_train_log.jsonl"
    alignment_debug_path = output_dir / "assignment_alignment_debug.jsonl"
    non_executed_samples_path = output_dir / "assignment_non_executed_samples.jsonl"
    checkpoint_path = output_dir / "assignment_mappo_latest.pt"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy = AssignmentMAPPO(device=args.device, encoder_checkpoint=args.encoder_checkpoint)
    progress = TerminalProgress(args.episodes * args.steps, "assignment-ppo")

    alignment_debug_handle = alignment_debug_path.open("w", encoding="utf-8") if args.write_alignment_debug else None
    samples_handle = (
        non_executed_samples_path.open("w", encoding="utf-8")
        if args.non_executed_sample_limit > 0
        else None
    )
    log_handle = None
    try:
        log_handle = log_path.open("w", encoding="utf-8")
        for episode in range(1, args.episodes + 1):
            episode_seed = args.seed + episode - 1
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            env = Env()
            env.set_assignment_policy(policy)
            env.reset()

            episode_reward = 0.0
            latency_total = 0.0
            energy_total = 0.0
            assignment_decisions = 0
            executed_decisions = 0
            fallback_selected_count = 0
            invalid_assignment_count = 0
            episode_rl_records: list[dict] = []

            for step in range(1, args.steps + 1):
                _, rewards, metrics = env.step(_zero_actions())
                global_reward = float(np.mean(rewards))
                done = step == args.steps
                rl_records = env.latest_assignment_rl_records
                for record in rl_records:
                    record["episode"] = int(episode)
                    record["env_step_id"] = int(step)
                policy.store_step(
                    env_step_id=step,
                    rl_records=rl_records,
                    shared_reward=global_reward,
                    done=done,
                )
                assignment_decisions += sum(1 for record in rl_records if record.get("actor_called", True))
                executed_decisions += sum(1 for record in rl_records if record.get("action_executed", False))
                fallback_selected_count += sum(1 for record in rl_records if record.get("fallback_used", False))
                episode_rl_records.extend(rl_records)
                invalid_assignment_count += int(env.task_executor.latest_stats.invalid_actions)
                episode_reward += global_reward
                latency_total += float(metrics[0])
                energy_total += float(metrics[1])
                progress.update(
                    postfix=(
                        f"episode {episode}/{args.episodes} step {step}/{args.steps} "
                        f"reward {episode_reward:.3f} executed {executed_decisions}/{assignment_decisions}"
                    )
                )

            update_metrics = policy.update()
            job_summary = env.task_manager.get_job_summary()
            execution_summary = summarize_assignment_execution(episode_rl_records)
            row = {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "episode": episode,
                "seed": episode_seed,
                "episode_reward": episode_reward,
                "mean_step_reward": episode_reward / max(float(args.steps), 1.0),
                "episode_latency": latency_total,
                "episode_energy": energy_total,
                "fallback_selected_count": fallback_selected_count,
                "invalid_assignment_count": invalid_assignment_count,
                **execution_summary,
                **dataclasses.asdict(update_metrics),
                **job_summary,
            }
            log_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            log_handle.flush()
            if alignment_debug_handle is not None:
                alignment_debug_row = {
                    "episode": episode,
                    "total_decisions": execution_summary["num_assignment_decisions"],
                    "executed_decisions": execution_summary["num_executed_decisions"],
                    "non_executed_decisions": execution_summary["num_non_executed_decisions"],
                    "actor_called_decisions": execution_summary["actor_called_decisions"],
                    "executed_actor_actions": execution_summary["executed_actor_actions"],
                    "invalid_actor_action_count": execution_summary["invalid_actor_action_count"],
                    "no_feasible_candidate_count": execution_summary["no_feasible_candidate_count"],
                    "action_executed_rate": execution_summary["action_executed_rate"],
                    "non_executed_reason_counts": execution_summary["non_executed_reason_counts"],
                    "non_executed_failure_reason_counts": execution_summary["non_executed_failure_reason_counts"],
                    "executor_override_count": execution_summary["executor_override_count"],
                    "fallback_after_actor_count": execution_summary["fallback_after_actor_count"],
                    "worst_action_executed_steps": execution_summary["worst_action_executed_steps"],
                }
                alignment_debug_handle.write(json.dumps(alignment_debug_row, ensure_ascii=False) + "\n")
                alignment_debug_handle.flush()
            if samples_handle is not None:
                sample_fields = (
                    "episode",
                    "env_step_id",
                    "task_id",
                    "task_type",
                    "candidate_count",
                    "candidate_uav_ids",
                    "actor_selected_uav",
                    "executor_selected_uav",
                    "fallback_used",
                    "non_executed_reason",
                    "failure_reason",
                    "selected_uav_failure_reason",
                )
                non_executed_records = [
                    record for record in episode_rl_records if not record.get("action_executed", False)
                ][: args.non_executed_sample_limit]
                for record in non_executed_records:
                    samples_handle.write(
                        json.dumps({field: record.get(field) for field in sample_fields}, ensure_ascii=False) + "\n"
                    )
                samples_handle.flush()
            policy.save(str(checkpoint_path))
    finally:
        if log_handle is not None:
            log_handle.close()
        if alignment_debug_handle is not None:
            alignment_debug_handle.close()
        if samples_handle is not None:
            samples_handle.close()

    progress.finish(postfix=f"saved {checkpoint_path}")
    print(f"log={log_path}")
    print(f"checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
