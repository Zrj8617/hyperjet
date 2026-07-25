from __future__ import annotations

import argparse
from datetime import datetime
import json
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
from environment.graph_builder import CleanGraphBuilder
from marl_models.mappo.clean_slot_orchestrator import prepare_slot_state
from scripts.offloading_policy_gate import RANDOM_HASH_VERSION, stable_random_hash_index


BASELINE_POLICIES = ("random_hash", "greedy_eft")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed-hover no-learning clean assignment baselines."
    )
    parser.add_argument("--policy", choices=BASELINE_POLICIES, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=int(config.EPISODE_LENGTH),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--completed-dag-weight", type=float, default=16.0)
    parser.add_argument("--output-dir", type=Path, default=Path("logs") / "clean_baselines")
    parser.add_argument("--run-name", type=str, default="baseline")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _validate_args(args)
    run_dir = _create_run_dir(args)
    config_payload = _build_config(args, run_dir)
    _write_json(run_dir / "config.json", config_payload)
    _write_json(run_dir / "run_summary.json", {"status": "initialized", **config_payload})

    writer = _ScalarEventWriter(run_dir)
    graph_builder = CleanGraphBuilder()
    try:
        _set_seed(int(args.seed))
        env = Env(completed_dag_weight=float(args.completed_dag_weight))

        # Match the clean training entrypoint's feature-dimension prelude so the
        # first real episode starts from the same NumPy RNG position.
        env.reset()
        graph_builder.reset()
        prepare_slot_state(env=env, graph_builder=graph_builder)

        aggregate: dict[str, float] = {}
        for episode in range(int(args.episodes)):
            env.reset()
            graph_builder.reset()
            episode_reward = 0.0
            accepted_assignments = 0
            skipped_no_candidate = 0
            latest_info: dict[str, Any] = {}
            for episode_step in range(int(args.max_steps_per_episode)):
                prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
                env.apply_movement({})
                ready_tasks = [
                    env.task_manager.get_task(task_id)
                    for task_id in prepared.frozen_ready_task_ids
                ]
                ready_tasks = [
                    task for task in ready_tasks if task is not None and task.is_ready
                ]
                assignments, skipped = select_baseline_assignments(
                    policy=str(args.policy),
                    frozen_ready_tasks=ready_tasks,
                    task_manager=env.task_manager,
                    uavs=env.uavs,
                    executor=env.executor,
                    current_time_seconds=env.current_time_seconds,
                    environment_seed=int(args.seed),
                    episode=int(episode),
                    slot=int(episode_step),
                    uav_service_positions=env.uav_service_positions,
                    ue_service_positions=env.ue_service_positions,
                    ues=env.ues,
                )
                _, _, done, latest_info = env.commit_and_advance(
                    assignment_buffer=assignments,
                    offloading_skip_count=skipped,
                )
                episode_reward += float(latest_info["step_reward"])
                accepted_assignments += int(latest_info["newly_assigned_tasks"])
                skipped_no_candidate += int(latest_info["offloading_skipped_no_candidate"])
                if done:
                    break

            row = {
                "episode": int(episode),
                "policy": str(args.policy),
                "seed": int(args.seed),
                "episode_reward_total": float(episode_reward),
                "accepted_assignments": int(accepted_assignments),
                "offloading_skipped_no_candidate": int(skipped_no_candidate),
                **_episode_metric_subset(latest_info),
            }
            _append_jsonl(run_dir / "episode_metrics.jsonl", row)
            _accumulate(aggregate, row)
            _write_scalars(writer, row, episode)
            _write_json(
                run_dir / "run_summary.json",
                {
                    "status": "running" if episode + 1 < int(args.episodes) else "completed",
                    **config_payload,
                    "last_episode": int(episode),
                    "aggregate_means": _aggregate_means(aggregate, episode + 1),
                    "latest_episode": row,
                },
            )
        return 0
    except Exception as exc:
        _write_json(
            run_dir / "run_summary.json",
            {
                "status": "failed",
                **config_payload,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finally:
        writer.close()
        graph_builder.close()


def select_baseline_assignments(
    *,
    policy: str,
    frozen_ready_tasks: list[Any],
    task_manager: Any,
    uavs: list[Any],
    executor: Any,
    current_time_seconds: float,
    environment_seed: int,
    episode: int,
    slot: int,
    uav_service_positions: dict[int, Any] | None = None,
    ue_service_positions: dict[int, Any] | None = None,
    ues: list[Any] | None = None,
) -> tuple[CleanAssignmentBuffer, int]:
    if policy not in BASELINE_POLICIES:
        raise ValueError(f"unsupported baseline policy: {policy}")
    reservation = TemporaryReservationState.from_executor(uavs, executor)
    assignments = CleanAssignmentBuffer()
    skipped = 0
    for decision_order, task in enumerate(frozen_ready_tasks):
        _, _, mask, candidate_uav_ids, estimates = build_offloading_candidate_components(
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
        legal_indices = [idx for idx, legal in enumerate(mask.tolist()) if bool(legal)]
        if not legal_indices:
            skipped += 1
            continue
        if policy == "greedy_eft":
            selected_idx = min(
                legal_indices,
                key=lambda idx: (
                    float(estimates[idx].estimated_finish_time),
                    int(candidate_uav_ids[idx]),
                ),
            )
        else:
            selected_idx = legal_indices[
                stable_random_hash_index(
                    environment_seed=int(environment_seed),
                    episode=int(episode),
                    slot=int(slot),
                    task_id=str(task.task_id),
                    legal_uav_ids=[
                        int(candidate_uav_ids[idx]) for idx in legal_indices
                    ],
                )
            ]
        selected_uav_id = int(candidate_uav_ids[selected_idx])
        selected_estimate = estimates[selected_idx]
        assignments.append(str(task.task_id), selected_uav_id, int(decision_order))
        reservation.reserve(
            str(task.task_id),
            selected_uav_id,
            estimated_available_time=float(selected_estimate.estimated_finish_time),
            estimated_queued_workload=float(
                selected_estimate.estimated_queued_workload
            ),
        )
    return assignments, skipped


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.episodes) <= 0:
        raise ValueError("episodes must be positive")
    if int(args.max_steps_per_episode) <= 0:
        raise ValueError("max-steps-per-episode must be positive")
    if not np.isfinite(float(args.completed_dag_weight)) or float(
        args.completed_dag_weight
    ) < 0.0:
        raise ValueError("completed-dag-weight must be finite and non-negative")


def _create_run_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(args.run_name)
    ).strip("_")
    run_dir = Path(args.output_dir) / (
        f"{timestamp}_{safe_name or args.policy}_seed{int(args.seed)}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _build_config(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    return {
        "schema": "clean_policy_baseline_v1",
        "run_dir": str(run_dir),
        "policy": str(args.policy),
        "seed": int(args.seed),
        "episodes": int(args.episodes),
        "max_steps_per_episode": int(args.max_steps_per_episode),
        "movement_frozen": True,
        "completed_dag_weight": float(args.completed_dag_weight),
        "max_active_dags_per_ue": 1,
        "dag_base_arrival_probability": float(config.DAG_BASE_ARRIVAL_PROB),
        "random_hash_version": (
            RANDOM_HASH_VERSION if str(args.policy) == "random_hash" else None
        ),
        "git_commit": _git_commit(),
    }


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


def _accumulate(aggregate: dict[str, float], row: dict[str, Any]) -> None:
    for key, value in row.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            aggregate[key] = aggregate.get(key, 0.0) + float(value)


def _aggregate_means(aggregate: dict[str, float], count: int) -> dict[str, float]:
    return {
        key: float(value) / float(max(int(count), 1))
        for key, value in aggregate.items()
        if key != "episode"
    }


def _write_scalars(writer: "_ScalarEventWriter", row: dict[str, Any], step: int) -> None:
    mapping = {
        "episode/reward_total": "episode_reward_total",
        "episode/generated_dag_count": "generated_dag_count",
        "episode/completed_dag_count": "completed_dag_count",
        "episode/dag_completion_rate": "dag_completion_rate",
        "episode/average_dag_flowtime": "average_dag_flowtime",
        "episode/avg_uav_queue_length": "avg_uav_queue_length",
        "episode/frozen_ready_task_count": "frozen_ready_task_count",
        "episode/accepted_assignments": "accepted_assignments",
        "episode/offloading_skipped_no_candidate": "offloading_skipped_no_candidate",
    }
    for tag, key in mapping.items():
        value = row.get(key)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            writer.add_scalar(tag, float(value), int(step))


class _ScalarEventWriter:
    def __init__(self, run_dir: Path) -> None:
        self._writer: Any | None = None
        self._event_cls: Any | None = None
        self._summary_cls: Any | None = None
        try:
            from tensorboard.compat.proto.event_pb2 import Event
            from tensorboard.compat.proto.summary_pb2 import Summary
            from tensorboard.summary.writer.event_file_writer import EventFileWriter

            self._event_cls = Event
            self._summary_cls = Summary
            self._writer = EventFileWriter(str(run_dir))
        except ModuleNotFoundError:
            self._writer = None

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is None:
            return
        summary = self._summary_cls(
            value=[self._summary_cls.Value(tag=str(tag), simple_value=float(value))]
        )
        self._writer.add_event(
            self._event_cls(wall_time=datetime.now().timestamp(), step=int(step), summary=summary)
        )

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
