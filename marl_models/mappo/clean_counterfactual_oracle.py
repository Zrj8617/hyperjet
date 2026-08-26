from __future__ import annotations

import copy
from dataclasses import dataclass
import multiprocessing
from typing import Any, Iterable

import numpy as np

from environment.assignment import (
    CleanAssignmentBuffer,
    TemporaryReservationState,
    build_offloading_candidate_components,
)


@dataclass(frozen=True, slots=True)
class CleanCounterfactualDecisionTrace:
    """Minimal immutable baseline decision trace used by Scheme-B0 branches."""

    task_id: str
    decision_order: int
    selected_uav_id: int
    candidate_uav_ids: tuple[int, ...]
    candidate_mask: tuple[bool, ...]

    @property
    def legal_uav_ids(self) -> tuple[int, ...]:
        return tuple(
            uav_id
            for uav_id, legal in zip(self.candidate_uav_ids, self.candidate_mask)
            if legal
        )


@dataclass(frozen=True, slots=True)
class CleanCounterfactualBranchResult:
    """One current-slot forced-UAV branch result and compact state proof."""

    target_task_id: str
    decision_order: int
    forced_uav_id: int
    baseline_uav_id: int
    candidate_uav_ids: tuple[int, ...]
    suffix_replay_feasible: bool
    current_slot_reward: float | None
    reward_components: dict[str, float]
    completed_task_ids: tuple[str, ...]
    completed_task_count: int
    completed_dag_ids: tuple[str, ...]
    completed_dag_count: int
    executor_queue_lengths_after_commit: dict[int, int]
    uav_available_time_after_commit: dict[int, float]
    replayed_assignments: tuple[tuple[int, str, int], ...]
    task_state_signature: tuple[tuple[Any, ...], ...]
    dag_state_signature: tuple[tuple[Any, ...], ...]

    def minimal_output(self) -> dict[str, Any]:
        """Return only the user-facing B0 correctness fields."""

        return {
            "target_task_id": self.target_task_id,
            "decision_order": self.decision_order,
            "forced_uav_id": self.forced_uav_id,
            "baseline_uav_id": self.baseline_uav_id,
            "candidate_uav_ids": list(self.candidate_uav_ids),
            "suffix_replay_feasible": self.suffix_replay_feasible,
            "current_slot_reward": self.current_slot_reward,
            "reward_components": dict(self.reward_components),
            "completed_task_ids": list(self.completed_task_ids),
            "completed_task_count": self.completed_task_count,
            "completed_dag_ids": list(self.completed_dag_ids),
            "completed_dag_count": self.completed_dag_count,
            "executor_queue_lengths_after_commit": dict(
                self.executor_queue_lengths_after_commit
            ),
        }


def clone_post_movement_pre_offloading_env(env: Any) -> Any:
    """Deep-copy an open post-movement/pre-offloading clean Env snapshot."""

    if not bool(getattr(env, "_prepared_slot_open", False)):
        raise ValueError(
            "Scheme-B0 snapshot requires an open prepared slot before offloading commit"
        )
    if int(getattr(env, "_last_movement_action_count", 0)) != len(env.uavs):
        raise ValueError("Scheme-B0 snapshot requires movement to be applied first")
    if not getattr(env, "uav_service_positions", {}):
        raise ValueError("Scheme-B0 snapshot requires movement to be applied first")
    return copy.deepcopy(env)


def capture_clean_counterfactual_baseline_trace(
    *,
    decision_records: Iterable[Any],
    assignment_buffer: CleanAssignmentBuffer,
) -> tuple[CleanCounterfactualDecisionTrace, ...]:
    """Freeze the actor's real sequential decisions into a process-safe trace."""

    traces: list[CleanCounterfactualDecisionTrace] = []
    for record in sorted(decision_records, key=lambda item: int(item.decision_order)):
        candidate_uav_ids = tuple(int(value) for value in record.candidate_uav_ids)
        candidate_mask = tuple(
            bool(value)
            for value in np.asarray(_to_numpy(record.candidate_mask), dtype=bool).reshape(-1)
        )
        if len(candidate_uav_ids) != len(candidate_mask):
            raise ValueError("baseline candidate IDs and mask have inconsistent lengths")
        selected_uav_id = int(record.selected_uav_id)
        if selected_uav_id not in candidate_uav_ids:
            raise ValueError("baseline selected UAV is absent from candidate ordering")
        selected_index = candidate_uav_ids.index(selected_uav_id)
        if not candidate_mask[selected_index]:
            raise ValueError("baseline selected UAV is not legal")
        traces.append(
            CleanCounterfactualDecisionTrace(
                task_id=str(record.task_id),
                decision_order=int(record.decision_order),
                selected_uav_id=selected_uav_id,
                candidate_uav_ids=candidate_uav_ids,
                candidate_mask=candidate_mask,
            )
        )
    expected_assignments = tuple(
        (int(entry.decision_order), str(entry.task_id), int(entry.uav_id))
        for entry in assignment_buffer.entries
    )
    trace_assignments = tuple(
        (trace.decision_order, trace.task_id, trace.selected_uav_id)
        for trace in traces
    )
    if expected_assignments != trace_assignments:
        raise ValueError("baseline decision trace does not match final assignment buffer")
    return tuple(traces)


def select_first_multicandidate_decision(
    baseline_trace: Iterable[CleanCounterfactualDecisionTrace],
) -> CleanCounterfactualDecisionTrace:
    for trace in baseline_trace:
        if len(trace.legal_uav_ids) >= 2:
            return trace
    raise ValueError("baseline trace has no decision with at least two legal UAVs")


def run_clean_counterfactual_branch(
    *,
    env_snapshot: Any,
    baseline_trace: Iterable[CleanCounterfactualDecisionTrace],
    target_decision_order: int,
    forced_uav_id: int,
) -> CleanCounterfactualBranchResult:
    """Replay one strict single-decision intervention and commit one real slot."""

    result, _ = materialize_clean_counterfactual_branch(
        env_snapshot=env_snapshot,
        baseline_trace=baseline_trace,
        target_decision_order=target_decision_order,
        forced_uav_id=forced_uav_id,
    )
    return result


def materialize_clean_counterfactual_branch(
    *,
    env_snapshot: Any,
    baseline_trace: Iterable[CleanCounterfactualDecisionTrace],
    target_decision_order: int,
    forced_uav_id: int,
) -> tuple[CleanCounterfactualBranchResult, Any]:
    """Replay and return both the B0 result and its independent post-commit Env."""

    env = clone_post_movement_pre_offloading_env(env_snapshot)
    traces = tuple(sorted(baseline_trace, key=lambda item: item.decision_order))
    target = _target_trace(traces, target_decision_order)
    forced_uav = int(forced_uav_id)
    if forced_uav not in target.legal_uav_ids:
        raise ValueError("forced UAV must be legal in the baseline target decision")

    reservation = TemporaryReservationState.from_executor(env.uavs, env.executor)
    assignments = CleanAssignmentBuffer()
    replayed: list[tuple[int, str, int]] = []
    target_seen = False
    for trace in traces:
        task = env.task_manager.get_task(trace.task_id)
        if task is None:
            raise ValueError(f"baseline task is missing from Env snapshot: {trace.task_id}")
        _, _, mask, candidate_uav_ids, estimates = build_offloading_candidate_components(
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
        current_ids = tuple(int(value) for value in candidate_uav_ids)
        current_mask = tuple(bool(value) for value in np.asarray(mask, dtype=bool))
        if trace.decision_order <= target.decision_order:
            if current_ids != trace.candidate_uav_ids or current_mask != trace.candidate_mask:
                raise AssertionError(
                    "baseline prefix/target candidate ordering or legality changed"
                )

        selected_uav = (
            forced_uav
            if trace.decision_order == target.decision_order
            else int(trace.selected_uav_id)
        )
        if trace.decision_order == target.decision_order:
            target_seen = True
        if selected_uav not in current_ids:
            if trace.decision_order > target.decision_order:
                return _censored_result(target, forced_uav, env, replayed), env
            raise AssertionError("prefix/target selected UAV disappeared from candidates")
        selected_index = current_ids.index(selected_uav)
        if not current_mask[selected_index]:
            if trace.decision_order > target.decision_order:
                return _censored_result(target, forced_uav, env, replayed), env
            raise AssertionError("prefix/target selected UAV became illegal")
        estimate = estimates[selected_index]
        assignments.append(trace.task_id, selected_uav, trace.decision_order)
        reservation.reserve(
            trace.task_id,
            selected_uav,
            estimated_available_time=float(estimate.estimated_finish_time),
            estimated_queued_workload=float(estimate.estimated_queued_workload),
        )
        replayed.append((trace.decision_order, trace.task_id, selected_uav))

    if not target_seen:
        raise AssertionError("target decision was not replayed")
    _, _, _, info = env.commit_and_advance(assignment_buffer=assignments)
    latest_stats = env.executor.latest_stats
    result = CleanCounterfactualBranchResult(
        target_task_id=target.task_id,
        decision_order=target.decision_order,
        forced_uav_id=forced_uav,
        baseline_uav_id=target.selected_uav_id,
        candidate_uav_ids=target.candidate_uav_ids,
        suffix_replay_feasible=True,
        current_slot_reward=float(info["step_reward"]),
        reward_components=_reward_components(info),
        completed_task_ids=tuple(str(value) for value in latest_stats.completed_task_ids),
        completed_task_count=int(latest_stats.completed_tasks),
        completed_dag_ids=tuple(str(value) for value in latest_stats.completed_dag_ids),
        completed_dag_count=int(latest_stats.completed_dags),
        executor_queue_lengths_after_commit=_queue_lengths(env),
        uav_available_time_after_commit=_available_times(env),
        replayed_assignments=tuple(replayed),
        task_state_signature=_task_state_signature(env),
        dag_state_signature=_dag_state_signature(env),
    )
    return result, env


def run_clean_counterfactual_branches_serial(
    *,
    env_snapshot: Any,
    baseline_trace: Iterable[CleanCounterfactualDecisionTrace],
    target_decision_order: int,
) -> tuple[CleanCounterfactualBranchResult, ...]:
    traces = tuple(baseline_trace)
    target = _target_trace(traces, target_decision_order)
    return tuple(
        run_clean_counterfactual_branch(
            env_snapshot=env_snapshot,
            baseline_trace=traces,
            target_decision_order=target.decision_order,
            forced_uav_id=uav_id,
        )
        for uav_id in target.legal_uav_ids
    )


def run_clean_counterfactual_branches_process(
    *,
    env_snapshot: Any,
    baseline_trace: Iterable[CleanCounterfactualDecisionTrace],
    target_decision_order: int,
    timeout_seconds: float = 30.0,
) -> tuple[CleanCounterfactualBranchResult, ...]:
    """Run the same serialized Env snapshot under spawn, one forced UAV per worker."""

    traces = tuple(baseline_trace)
    target = _target_trace(traces, target_decision_order)
    context = multiprocessing.get_context("spawn")
    workers: list[tuple[int, Any, Any]] = []
    try:
        for uav_id in target.legal_uav_ids:
            parent, child = context.Pipe(duplex=False)
            process = context.Process(
                target=_branch_process_worker,
                args=(child, env_snapshot, traces, target.decision_order, uav_id),
                name=f"phase4-b0-uav-{uav_id}",
            )
            process.start()
            child.close()
            workers.append((uav_id, parent, process))
        results: dict[int, CleanCounterfactualBranchResult] = {}
        for uav_id, connection, process in workers:
            if not connection.poll(float(timeout_seconds)):
                raise TimeoutError(f"Phase4-B0 branch UAV {uav_id} timed out")
            payload = connection.recv()
            process.join(timeout=float(timeout_seconds))
            if process.is_alive():
                raise TimeoutError(f"Phase4-B0 branch UAV {uav_id} did not exit")
            if int(process.exitcode or 0) != 0 or not payload.get("ok", False):
                raise RuntimeError(
                    f"Phase4-B0 branch UAV {uav_id} failed: {payload.get('error')}"
                )
            results[uav_id] = payload["result"]
        return tuple(results[uav_id] for uav_id in target.legal_uav_ids)
    finally:
        for _, connection, process in workers:
            connection.close()
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)


def _branch_process_worker(
    connection: Any,
    env_snapshot: Any,
    baseline_trace: tuple[CleanCounterfactualDecisionTrace, ...],
    target_decision_order: int,
    forced_uav_id: int,
) -> None:
    try:
        result = run_clean_counterfactual_branch(
            env_snapshot=env_snapshot,
            baseline_trace=baseline_trace,
            target_decision_order=target_decision_order,
            forced_uav_id=forced_uav_id,
        )
        connection.send({"ok": True, "result": result})
    except BaseException as exc:
        connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def _target_trace(
    traces: Iterable[CleanCounterfactualDecisionTrace], target_decision_order: int
) -> CleanCounterfactualDecisionTrace:
    matches = [
        trace for trace in traces if trace.decision_order == int(target_decision_order)
    ]
    if len(matches) != 1:
        raise ValueError("target decision_order must identify exactly one baseline decision")
    return matches[0]


def _censored_result(
    target: CleanCounterfactualDecisionTrace,
    forced_uav_id: int,
    env: Any,
    replayed: list[tuple[int, str, int]],
) -> CleanCounterfactualBranchResult:
    return CleanCounterfactualBranchResult(
        target_task_id=target.task_id,
        decision_order=target.decision_order,
        forced_uav_id=int(forced_uav_id),
        baseline_uav_id=target.selected_uav_id,
        candidate_uav_ids=target.candidate_uav_ids,
        suffix_replay_feasible=False,
        current_slot_reward=None,
        reward_components={},
        completed_task_ids=(),
        completed_task_count=0,
        completed_dag_ids=(),
        completed_dag_count=0,
        executor_queue_lengths_after_commit=_queue_lengths(env),
        uav_available_time_after_commit=_available_times(env),
        replayed_assignments=tuple(replayed),
        task_state_signature=_task_state_signature(env),
        dag_state_signature=_dag_state_signature(env),
    )


def _reward_components(info: dict[str, Any]) -> dict[str, float]:
    names = (
        "step_time_penalty",
        "step_energy_penalty",
        "step_task_energy_penalty",
        "step_movement_energy_penalty",
        "step_completed_dag_bonus",
        "step_movement_position_bonus",
    )
    return {name: float(info[name]) for name in names}


def _queue_lengths(env: Any) -> dict[int, int]:
    return {
        int(uav.id): len(env.executor.uav_queues.get(int(uav.id), []))
        for uav in env.uavs
    }


def _available_times(env: Any) -> dict[int, float]:
    return {
        int(uav.id): float(env.executor.uav_available_time.get(int(uav.id), 0.0))
        for uav in env.uavs
    }


def _task_state_signature(env: Any) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            str(task_id),
            str(task.state),
            None if task.assigned_uav is None else int(task.assigned_uav),
            _optional_float(task.start_time),
            _optional_float(task.finish_time),
            _optional_float(task.compute_finish_time),
            _optional_float(task.reward_completion_time),
            bool(task.reward_settled),
        )
        for task_id, task in sorted(env.task_manager.tasks.items())
    )


def _dag_state_signature(env: Any) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            str(dag_id),
            bool(job.completed),
            _optional_float(job.return_complete_time),
            bool(job.completion_reward_settled),
        )
        for dag_id, job in sorted(env.task_manager.jobs.items())
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)
