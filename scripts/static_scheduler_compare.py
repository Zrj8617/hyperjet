from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

import config
from environment.env import Env
from environment.dag_tasks import TASK_STATE_DROPPED, TASK_STATE_FINISHED
from environment.task_execution import AssignmentCandidateRecord, AssignmentDecisionRecord, ScheduledTask
from utils.progress import TerminalProgress


def _refresh_dimension_config() -> None:
    phase_one_obs = (
        config.ENABLE_DYNAMIC_DAG
        and config.ENABLE_PHASE_ONE_EXECUTION
        and not config.ENABLE_LEGACY_REQUEST_PIPELINE
        and config.USE_PHASE_ONE_DEDICATED_OBS
    )
    compact_obs = phase_one_obs and config.USE_MAPPO_COMPACT_OBS
    config.MAX_UAV_NEIGHBORS = max(config.NUM_UAVS - 1, 1)
    config.MAX_ASSOCIATED_UES = min(30, config.NUM_UES // max(config.NUM_UAVS, 1) + 10)
    config.SELF_OBS_DIM = config.PHASE_ONE_SELF_OBS_DIM if phase_one_obs else config.LEGACY_SELF_OBS_DIM
    config.UE_OBS_DIM = (
        config.MAPPO_COMPACT_LOCAL_OBS_DIM
        if compact_obs
        else config.PHASE_ONE_TASK_OBS_DIM
        if phase_one_obs
        else config.LEGACY_UE_OBS_DIM
    )
    config.NEIGHBOR_OBS_DIM = config.PHASE_ONE_NEIGHBOR_OBS_DIM if phase_one_obs else config.LEGACY_NEIGHBOR_OBS_DIM
    config.OBS_DIM_SINGLE = (
        config.SELF_OBS_DIM + (config.MAX_UAV_NEIGHBORS * config.NEIGHBOR_OBS_DIM) + config.UE_OBS_DIM
        if compact_obs
        else config.SELF_OBS_DIM
        + (config.MAX_UAV_NEIGHBORS * config.NEIGHBOR_OBS_DIM)
        + (config.MAX_ASSOCIATED_UES * config.UE_OBS_DIM)
    )


def _override_num_uavs(num_uavs: int, seed: int) -> None:
    if num_uavs <= 1:
        raise ValueError("--num_uavs must be greater than 1.")
    config.NUM_UAVS = num_uavs
    rng = np.random.default_rng(seed)
    config.UAV_STORAGE_CAPACITY = rng.choice(
        np.arange(40 * 10**6, 80 * 10**6, 10**6),
        size=config.NUM_UAVS,
    ).astype(np.int64)
    config.UAV_COMPUTING_CAPACITY = rng.choice(
        np.arange(5 * 10**9, 20 * 10**9, 10**9),
        size=config.NUM_UAVS,
    ).astype(np.int64)
    _refresh_dimension_config()


def _override_num_ues(num_ues: int) -> None:
    if num_ues <= 0:
        raise ValueError("--num_ues must be greater than 0.")
    config.NUM_UES = num_ues
    _refresh_dimension_config()


def _zero_actions() -> np.ndarray:
    return np.zeros((config.NUM_UAVS, config.ACTION_DIM), dtype=np.float32)


def _update_diagnostics(accumulator: dict[str, float], diagnostics: dict[str, float]) -> None:
    for key, value in diagnostics.items():
        accumulator[key] = accumulator.get(key, 0.0) + float(value)


def _finalize_diagnostics(accumulator: dict[str, float], steps: int) -> dict[str, float]:
    averaged_keys = {
        "ready_tasks",
        "active_tasks",
        "feasible_edges",
        "score_edge_count",
        "selective_scoring_enabled",
        "selective_ready_tasks",
        "selective_high_risk_tasks",
        "selective_normal_tasks",
        "selective_high_risk_ratio",
        "selective_score_edges",
    }
    result: dict[str, float] = {}
    for key, value in accumulator.items():
        if key in averaged_keys:
            result[f"avg_{key}"] = value / max(float(steps), 1.0)
        else:
            result[key] = value
    return result


def _mean_metrics(episodes: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for episode in episodes for key in episode})
    return {
        key: float(np.mean([episode.get(key, 0.0) for episode in episodes]))
        for key in keys
        if key not in {"episode", "seed"}
    }


class _AttributionWriter:
    def __init__(self, path: Path, metadata: dict[str, Any]) -> None:
        self._file: TextIO = path.open("w", encoding="utf-8")
        self._first_assignment = True
        self._closed = False
        self._file.write("{\n")
        for key, value in metadata.items():
            self._file.write(f'  "{key}": ')
            json.dump(value, self._file, ensure_ascii=False)
            self._file.write(",\n")
        self._file.write('  "assignments": [')

    def write_assignment(self, assignment: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to a closed attribution writer.")
        if self._first_assignment:
            self._file.write("\n")
            self._first_assignment = False
        else:
            self._file.write(",\n")
        self._file.write("    ")
        json.dump(assignment, self._file, ensure_ascii=False)

    def close(self, task_outcomes: list[dict[str, Any]], dag_outcomes: list[dict[str, Any]]) -> None:
        if self._closed:
            return
        if not self._first_assignment:
            self._file.write("\n")
        self._file.write("  ],\n")
        self._file.write('  "task_outcomes": ')
        json.dump(task_outcomes, self._file, ensure_ascii=False)
        self._file.write(",\n")
        self._file.write('  "dag_outcomes": ')
        json.dump(dag_outcomes, self._file, ensure_ascii=False)
        self._file.write("\n}\n")
        self._file.close()
        self._closed = True


def _candidate_to_dict(candidate: AssignmentCandidateRecord) -> dict[str, float | int | None]:
    return {
        "uav_id": int(candidate.uav_id),
        "planned_start": float(candidate.planned_start),
        "planned_finish": float(candidate.planned_finish),
        "transmission_time": float(candidate.transmission_time),
        "execution_time": float(candidate.execution_time),
        "total_energy": float(candidate.total_energy),
        "score": None if candidate.score is None else float(candidate.score),
        "queue_length": int(candidate.queue_length),
        "available_time": float(candidate.available_time),
    }


def _rank_by_value(candidates: list[AssignmentCandidateRecord], uav_id: int | None, attr: str, reverse: bool = False) -> int | None:
    if uav_id is None:
        return None
    filtered = [candidate for candidate in candidates if getattr(candidate, attr) is not None]
    if not filtered:
        return None
    ordered = sorted(filtered, key=lambda candidate: getattr(candidate, attr), reverse=reverse)
    for rank, candidate in enumerate(ordered, 1):
        if candidate.uav_id == uav_id:
            return rank
    return None


def _get_candidate(candidates: list[AssignmentCandidateRecord], uav_id: int | None) -> AssignmentCandidateRecord | None:
    if uav_id is None:
        return None
    return next((candidate for candidate in candidates if candidate.uav_id == uav_id), None)


def _candidate_field(candidate: AssignmentCandidateRecord | None, field_name: str) -> float | int | None:
    if candidate is None:
        return None
    value = getattr(candidate, field_name)
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return float(value)


def _delta(selected: AssignmentCandidateRecord | None, heuristic: AssignmentCandidateRecord | None, field_name: str) -> float | None:
    selected_value = _candidate_field(selected, field_name)
    heuristic_value = _candidate_field(heuristic, field_name)
    if selected_value is None or heuristic_value is None:
        return None
    return float(selected_value) - float(heuristic_value)


def _is_selective_high_risk_task(env: Env, record: AssignmentDecisionRecord, current_step: int) -> bool:
    task = env.task_manager.tasks.get(record.task_id)
    if task is None:
        return False
    candidate_count = len(record.candidates)
    task_slack = task.remaining_slack(current_step)
    dag_slack = env.task_manager.get_dag_remaining_slack(task.dag_id, current_step)
    dag_completion = env.task_manager.get_dag_completion_ratio(task.dag_id)
    context_slack_threshold = config.SELECTIVE_HGNN_SLACK_THRESHOLD * config.SELECTIVE_HGNN_CONTEXT_SLACK_MULTIPLIER
    critical_path_task = config.SELECTIVE_HGNN_USE_CRITICAL_PATH and env.task_manager.is_critical_path_task(task.task_id)
    successor_unlock_task = config.SELECTIVE_HGNN_USE_SUCCESSOR_UNLOCK and len(task.successors) > 0
    is_high_risk = task_slack <= config.SELECTIVE_HGNN_SLACK_THRESHOLD or dag_slack <= config.SELECTIVE_HGNN_SLACK_THRESHOLD
    if config.USE_STRICT_SELECTIVE_HGNN_SCORING:
        return bool(is_high_risk)
    if config.SELECTIVE_HGNN_USE_CANDIDATE_SCARCITY:
        is_high_risk = is_high_risk or candidate_count <= config.SELECTIVE_HGNN_CANDIDATE_THRESHOLD
    if critical_path_task:
        is_high_risk = is_high_risk or dag_slack <= context_slack_threshold or dag_completion >= config.SELECTIVE_HGNN_COMPLETION_THRESHOLD
    if successor_unlock_task:
        is_high_risk = is_high_risk or dag_slack <= context_slack_threshold
    if config.SELECTIVE_HGNN_USE_DAG_COMPLETION:
        is_high_risk = is_high_risk or (
            dag_completion >= config.SELECTIVE_HGNN_COMPLETION_THRESHOLD
            and (critical_path_task or successor_unlock_task)
        )
    return bool(is_high_risk)


def _serialize_assignment_record(
    env: Env,
    record: AssignmentDecisionRecord,
    episode: int,
    step: int,
) -> dict[str, Any]:
    task = env.task_manager.tasks.get(record.task_id)
    selected_candidate = _get_candidate(record.candidates, record.selected_uav)
    heuristic_candidate = _get_candidate(record.candidates, record.heuristic_uav)
    score_candidate = _get_candidate(record.candidates, record.score_uav)
    teacher_scores: dict[int, float] = {}
    teacher_uav: int | None = None
    if task is not None and record.candidates:
        for candidate in record.candidates:
            scheduled = ScheduledTask(
                task_id=record.task_id,
                uav_id=candidate.uav_id,
                planned_start=float(candidate.planned_start),
                planned_finish=float(candidate.planned_finish),
                transmission_time=float(candidate.transmission_time),
                execution_time=float(candidate.execution_time),
                total_energy=float(candidate.total_energy),
            )
            teacher_scores[candidate.uav_id] = float(
                env.task_executor._compute_teacher_score(  # noqa: SLF001 - diagnostic-only parity with Stage A target.
                    scheduled,
                    task,
                    env.task_manager,
                    env.uavs[candidate.uav_id],
                    step,
                )
            )
        teacher_uav = min(teacher_scores, key=teacher_scores.get)
    teacher_candidate = _get_candidate(record.candidates, teacher_uav)
    planned_finishes = sorted(float(candidate.planned_finish) for candidate in record.candidates)
    best_finish = planned_finishes[0] if planned_finishes else None
    second_best_finish = planned_finishes[1] if len(planned_finishes) > 1 else None
    selected_margin = None
    heuristic_margin = None
    if task is not None:
        selected_finish = _candidate_field(selected_candidate, "planned_finish")
        heuristic_finish = _candidate_field(heuristic_candidate, "planned_finish")
        if selected_finish is not None:
            selected_margin = float(task.deadline) - float(selected_finish)
        if heuristic_finish is not None:
            heuristic_margin = float(task.deadline) - float(heuristic_finish)

    payload: dict[str, Any] = {
        "episode": int(episode),
        "step": int(step),
        "task_id": record.task_id,
        "dag_id": None if task is None else task.dag_id,
        "ue_id": None if task is None else int(task.ue_id),
        "task_type": int(record.task_type),
        "task_level": int(record.task_level),
        "task_state_at_assignment": record.task_state,
        "task_arrival_time": float(record.task_arrival_time),
        "task_deadline": float(record.task_deadline),
        "task_slack": float(record.task_slack),
        "num_predecessors": int(record.num_predecessors),
        "num_successors": int(record.num_successors),
        "is_high_risk_task": bool(_is_selective_high_risk_task(env, record, step)),
        "is_critical_path_task": False if task is None else bool(env.task_manager.is_critical_path_task(task.task_id)),
        "dag_is_high_risk": False if task is None else bool(env.task_manager.is_high_risk_job(task.dag_id)),
        "dag_completion_ratio": 0.0 if task is None else float(env.task_manager.get_dag_completion_ratio(task.dag_id)),
        "dag_remaining_slack": 0.0 if task is None else float(env.task_manager.get_dag_remaining_slack(task.dag_id, step)),
        "candidate_count": int(len(record.candidates)),
        "selection_mode": record.selection_mode or "none",
        "selected_uav": record.selected_uav,
        "heuristic_uav": record.heuristic_uav,
        "score_uav": record.score_uav,
        "raw_score_uav": record.raw_score_uav,
        "raw_disagrees_with_heuristic": bool(record.raw_disagrees_with_heuristic),
        "guard_reason": record.guard_reason,
        "teacher_uav": teacher_uav,
        "disagrees_with_heuristic": bool(record.disagrees_with_heuristic),
        "teacher_disagrees_with_heuristic": bool(teacher_uav is not None and record.heuristic_uav is not None and teacher_uav != record.heuristic_uav),
        "student_disagrees_with_teacher": bool(teacher_uav is not None and record.score_uav is not None and record.score_uav != teacher_uav),
        "num_candidates": int(len(record.candidates)),
        "best_candidate_planned_finish": best_finish,
        "second_best_candidate_planned_finish": second_best_finish,
        "selected_rank_by_planned_finish": _rank_by_value(record.candidates, record.selected_uav, "planned_finish", reverse=False),
        "selected_rank_by_score": _rank_by_value(record.candidates, record.selected_uav, "score", reverse=True),
        "heuristic_rank_by_score": _rank_by_value(record.candidates, record.heuristic_uav, "score", reverse=True),
        "score_rank_by_planned_finish": _rank_by_value(record.candidates, record.score_uav, "planned_finish", reverse=False),
        "teacher_rank_by_planned_finish": _rank_by_value(record.candidates, teacher_uav, "planned_finish", reverse=False),
        "candidates": [_candidate_to_dict(candidate) for candidate in record.candidates],
    }
    for candidate_payload in payload["candidates"]:
        candidate_payload["teacher_score"] = teacher_scores.get(candidate_payload["uav_id"])

    for prefix, candidate in {
        "selected": selected_candidate,
        "heuristic": heuristic_candidate,
        "score": score_candidate,
        "teacher": teacher_candidate,
    }.items():
        payload[f"{prefix}_planned_start"] = _candidate_field(candidate, "planned_start")
        payload[f"{prefix}_planned_finish"] = _candidate_field(candidate, "planned_finish")
        payload[f"{prefix}_transmission_time"] = _candidate_field(candidate, "transmission_time")
        payload[f"{prefix}_execution_time"] = _candidate_field(candidate, "execution_time")
        payload[f"{prefix}_total_energy"] = _candidate_field(candidate, "total_energy")
        payload[f"{prefix}_queue_length"] = _candidate_field(candidate, "queue_length")
        payload[f"{prefix}_available_time"] = _candidate_field(candidate, "available_time")
        payload[f"{prefix}_score"] = _candidate_field(candidate, "score")
        if prefix == "teacher":
            payload[f"{prefix}_teacher_score"] = None if teacher_uav is None else teacher_scores.get(teacher_uav)

    payload["selected_deadline_margin"] = selected_margin
    payload["heuristic_deadline_margin"] = heuristic_margin
    payload["delta_planned_finish"] = _delta(selected_candidate, heuristic_candidate, "planned_finish")
    payload["delta_deadline_margin"] = (
        None if selected_margin is None or heuristic_margin is None else float(selected_margin - heuristic_margin)
    )
    payload["delta_transmission_time"] = _delta(selected_candidate, heuristic_candidate, "transmission_time")
    payload["delta_execution_time"] = _delta(selected_candidate, heuristic_candidate, "execution_time")
    payload["delta_total_energy"] = _delta(selected_candidate, heuristic_candidate, "total_energy")
    payload["delta_queue_length"] = _delta(selected_candidate, heuristic_candidate, "queue_length")
    payload["delta_available_time"] = _delta(selected_candidate, heuristic_candidate, "available_time")
    payload["teacher_delta_planned_finish"] = _delta(teacher_candidate, heuristic_candidate, "planned_finish")
    payload["teacher_delta_deadline_margin"] = (
        None
        if task is None or teacher_candidate is None or heuristic_candidate is None
        else float((task.deadline - teacher_candidate.planned_finish) - (task.deadline - heuristic_candidate.planned_finish))
    )
    payload["student_delta_planned_finish_vs_teacher"] = _delta(score_candidate, teacher_candidate, "planned_finish")
    return payload


def _serialize_task_outcomes(env: Env, episode: int) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for task in env.task_manager.tasks.values():
        outcomes.append(
            {
                "episode": int(episode),
                "task_id": task.task_id,
                "dag_id": task.dag_id,
                "final_state": task.state,
                "assigned_uav": task.assigned_uav,
                "enqueue_time": task.enqueue_time,
                "start_time": task.start_time,
                "finish_time": task.finish_time,
                "finished": task.state == TASK_STATE_FINISHED,
                "finished_on_time": bool(task.finish_time is not None and task.finish_time <= task.deadline),
                "dropped": task.state == TASK_STATE_DROPPED,
                "completion_time": None if task.finish_time is None else float(task.finish_time - task.arrival_time),
            }
        )
    return outcomes


def _serialize_dag_outcomes(env: Env, episode: int) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for job in env.task_manager.jobs.values():
        tasks = env.task_manager.get_job_tasks(job.dag_id)
        finished = all(task.state == TASK_STATE_FINISHED for task in tasks)
        failed = any(task.state == TASK_STATE_DROPPED for task in tasks)
        incomplete = not finished and not failed
        finish_times = [float(task.finish_time) for task in tasks if task.finish_time is not None]
        completion_time = max(finish_times) - float(job.arrival_time) if finished and finish_times else None
        tardiness = (
            max(max(float(task.finish_time) - float(task.deadline), 0.0) for task in tasks if task.finish_time is not None)
            if finished and finish_times
            else None
        )
        on_time = bool(finished and all(task.finish_time is not None and task.finish_time <= task.deadline for task in tasks))
        outcomes.append(
            {
                "episode": int(episode),
                "dag_id": job.dag_id,
                "arrival_time": float(job.arrival_time),
                "total_tasks": int(len(tasks)),
                "successful": bool(finished),
                "failed": bool(failed),
                "incomplete": bool(incomplete),
                "on_time_successful": on_time,
                "completion_time": completion_time,
                "tardiness": tardiness,
                "is_high_risk_job": bool(env.task_manager.is_high_risk_job(job.dag_id)),
            }
        )
    return outcomes


def _configure_run(args: argparse.Namespace) -> None:
    if args.num_ues is not None:
        _override_num_ues(args.num_ues)
    if args.num_uavs is not None:
        _override_num_uavs(args.num_uavs, args.seed)
    if args.dag_arrival_prob is not None:
        if not 0.0 <= args.dag_arrival_prob <= 1.0:
            raise ValueError("--dag_arrival_prob must be in [0, 1].")
        config.DAG_ARRIVAL_PROB = args.dag_arrival_prob

    config.STEPS_PER_EPISODE = args.steps
    config.USE_MAPPO_COMPACT_OBS = True
    config.USE_PHASE_ONE_DEDICATED_OBS = True
    config.USE_STAGE_B_MOVEMENT_REWARD = False
    config.USE_PHASE_ONE_DAG_REWARD_SHAPING = False
    config.USE_HGNN_SCORE_ASSIGNMENT = args.mode == "full"
    config.USE_SELECTIVE_HGNN_SCORING = args.mode == "full" and args.selective_hgnn_score
    config.USE_HGNN_PER_TASK_RESCORING = args.mode == "full" and args.rescore_each_assignment
    config.USE_SCORE_RUNTIME_BOUNDED_GUARD = args.mode == "full" and args.runtime_bounded_guard
    config.SCORE_RUNTIME_FINISH_TOLERANCE = float(args.runtime_finish_tolerance)
    config.USE_SCORE_AGREEMENT_ONLY = args.mode == "full" and args.agreement_only
    config.USE_STRICT_SELECTIVE_HGNN_SCORING = args.mode == "full" and args.strict_selective_hgnn_score
    config.HGNN_SCORE_CHECKPOINT = args.checkpoint if args.mode == "full" else ""
    config.USE_PHASE_ONE_HYPEREDGES = args.ablation != "no_hyperedge"
    config.USE_COLLABORATIVE_HYPEREDGES = args.ablation != "no_hyperedge"
    config.USE_SERVICE_DOMAIN_HYPEREDGES = args.ablation not in {"no_hyperedge", "no_service_domain"}
    config.USE_RESOURCE_COMPETITION_HYPEREDGES = args.ablation not in {
        "no_hyperedge",
        "no_resource_competition",
        "safe_hyperedge_only",
    }
    config.USE_CRITICAL_HYPEREDGES = args.ablation not in {"no_hyperedge", "no_critical"}
    config.USE_CRITICAL_SUPPORT_HYPEREDGES = args.ablation not in {
        "no_hyperedge",
        "no_critical",
        "safe_hyperedge_only",
    }
    config.USE_ATTRIBUTE_HYPEREDGES = False
    config.USE_COMPUTE_ATTRIBUTE_HYPEREDGES = False
    config.USE_COMMUNICATION_ATTRIBUTE_HYPEREDGES = False
    config.USE_CANDIDATE_SCARCE_ATTRIBUTE_HYPEREDGES = False
    config.USE_PAIR_HYPEREDGE_SCORE_FEATURES = args.ablation not in {
        "no_pair_hyperedge_score_feature",
        "safe_hyperedge_only",
    }

    if config.USE_HGNN_SCORE_ASSIGNMENT and not config.HGNN_SCORE_CHECKPOINT:
        raise ValueError("--checkpoint is required for --mode full.")
    if config.HGNN_SCORE_CHECKPOINT and not os.path.exists(config.HGNN_SCORE_CHECKPOINT):
        raise FileNotFoundError(f"HGNN checkpoint not found: {config.HGNN_SCORE_CHECKPOINT}")


def run_static_eval(args: argparse.Namespace) -> dict[str, object]:
    _configure_run(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    episode_rows: list[dict[str, float]] = []
    task_outcomes: list[dict[str, Any]] = []
    dag_outcomes: list[dict[str, Any]] = []
    attribution_writer: _AttributionWriter | None = None
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    progress = None
    if not args.no_progress:
        progress = TerminalProgress(
            args.episodes * args.steps,
            f"static:{args.mode}:seed{args.seed}",
        )
    if args.write_attribution:
        attribution_writer = _AttributionWriter(
            args.attribution_path,
            {
                "created_at": created_at,
                "mode": args.mode,
                "seed": args.seed,
                "episodes": args.episodes,
                "steps": args.steps,
                "num_ues": config.NUM_UES,
                "num_uavs": config.NUM_UAVS,
                "dag_arrival_prob": config.DAG_ARRIVAL_PROB,
                "rescore_each_assignment": config.USE_HGNN_PER_TASK_RESCORING,
                "runtime_bounded_guard": config.USE_SCORE_RUNTIME_BOUNDED_GUARD,
                "runtime_finish_tolerance": config.SCORE_RUNTIME_FINISH_TOLERANCE,
                "agreement_only": config.USE_SCORE_AGREEMENT_ONLY,
                "strict_selective_hgnn_score": config.USE_STRICT_SELECTIVE_HGNN_SCORING,
            },
        )
    for episode in range(1, args.episodes + 1):
        episode_seed = args.seed + episode - 1
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)

        env = Env()
        env.reset()
        reward_total = 0.0
        latency_total = 0.0
        energy_total = 0.0
        fairness_last = 0.0
        offline_total = 0.0
        diagnostics: dict[str, float] = {}

        for step in range(1, args.steps + 1):
            _, rewards, (total_latency, total_energy, fairness, offline_rate) = env.step(_zero_actions())
            reward_total += float(np.sum(rewards))
            latency_total += float(total_latency)
            energy_total += float(total_energy)
            fairness_last = float(fairness)
            offline_total += float(offline_rate)
            _update_diagnostics(diagnostics, env.latest_phase_one_diagnostics)
            if progress is not None:
                progress.update(
                    postfix=(
                        f"episode {episode}/{args.episodes} "
                        f"step {step}/{args.steps} "
                        f"reward {reward_total:.1f}"
                    )
                )
            if args.write_attribution:
                if attribution_writer is None:
                    raise RuntimeError("Attribution writer was not initialized.")
                for record in env.task_executor.latest_assignment_records:
                    attribution_writer.write_assignment(_serialize_assignment_record(env, record, episode, step))

        row: dict[str, float] = {
            "episode": float(episode),
            "seed": float(args.seed),
            "episode_reward": reward_total,
            "episode_latency": latency_total,
            "episode_energy": energy_total,
            "fairness": fairness_last,
            "offline_rate_mean": offline_total / max(float(args.steps), 1.0),
        }
        row.update(_finalize_diagnostics(diagnostics, args.steps))
        row.update(env.task_manager.get_job_summary())
        episode_rows.append(row)
        if args.write_attribution:
            task_outcomes.extend(_serialize_task_outcomes(env, episode))
            dag_outcomes.extend(_serialize_dag_outcomes(env, episode))
    if progress is not None:
        progress.finish(postfix=f"episodes {args.episodes} complete")

    payload: dict[str, Any] = {
        "created_at": created_at,
        "mode": args.mode,
        "seed": args.seed,
        "episodes": args.episodes,
        "steps": args.steps,
        "num_ues": config.NUM_UES,
        "num_uavs": config.NUM_UAVS,
        "dag_arrival_prob": config.DAG_ARRIVAL_PROB,
        "use_hgnn_score": config.USE_HGNN_SCORE_ASSIGNMENT,
        "use_selective_hgnn_score": config.USE_SELECTIVE_HGNN_SCORING,
        "rescore_each_assignment": config.USE_HGNN_PER_TASK_RESCORING,
        "runtime_bounded_guard": config.USE_SCORE_RUNTIME_BOUNDED_GUARD,
        "runtime_finish_tolerance": config.SCORE_RUNTIME_FINISH_TOLERANCE,
        "agreement_only": config.USE_SCORE_AGREEMENT_ONLY,
        "strict_selective_hgnn_score": config.USE_STRICT_SELECTIVE_HGNN_SCORING,
        "ablation": args.ablation,
        "checkpoint": config.HGNN_SCORE_CHECKPOINT,
        "episodes_data": episode_rows,
        "summary": _mean_metrics(episode_rows),
    }
    if attribution_writer is not None:
        attribution_writer.close(task_outcomes, dag_outcomes)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Static zero-movement scheduler comparison.")
    parser.add_argument("--mode", choices=["full", "fallback"], required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--num_ues", type=int, default=None)
    parser.add_argument("--num_uavs", type=int, default=None)
    parser.add_argument("--dag_arrival_prob", type=float, default=None)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--selective_hgnn_score", action="store_true")
    parser.add_argument("--rescore_each_assignment", action="store_true")
    parser.add_argument("--runtime_bounded_guard", action="store_true")
    parser.add_argument("--runtime_finish_tolerance", type=float, default=0.1)
    parser.add_argument("--agreement_only", action="store_true")
    parser.add_argument("--strict_selective_hgnn_score", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=[
            "full",
            "no_hyperedge",
            "no_service_domain",
            "no_resource_competition",
            "no_critical",
            "no_pair_hyperedge_score_feature",
            "safe_hyperedge_only",
        ],
    )
    parser.add_argument("--output_dir", type=str, default="analysis_outputs/static_scheduler_compare")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_attribution", action="store_true")
    args = parser.parse_args()
    args.write_attribution = not args.no_attribution

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"static_{args.mode}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = output_dir / f"{tag}.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite or change --tag.")
    if args.write_attribution:
        args.attribution_path = output_dir / f"{tag}_attribution.json"
        if args.attribution_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Attribution output already exists: {args.attribution_path}. Use --overwrite or change --tag."
            )

    payload = run_static_eval(args)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    summary = payload["summary"]
    print(f"saved={output_path}")
    if args.write_attribution:
        print(f"attribution={args.attribution_path}")
    for key in [
        "dag_on_time_success_rate",
        "dag_high_risk_on_time_success_rate",
        "dag_critical_path_finish_rate",
        "dag_critical_path_on_time_rate",
        "dag_success_rate",
        "dag_failure_rate",
        "avg_score_edge_count",
        "avg_selective_high_risk_ratio",
    ]:
        if key in summary:
            print(f"{key}={summary[key]:.6f}")


if __name__ == "__main__":
    main()
