from __future__ import annotations

import copy
from dataclasses import dataclass
import multiprocessing
import random
from typing import Any, Iterable

import numpy as np

from environment.graph_builder import CleanGraphBuilder
from marl_models.mappo.clean_counterfactual_oracle import (
    CleanCounterfactualBranchResult,
    CleanCounterfactualDecisionTrace,
    materialize_clean_counterfactual_branch,
)
from marl_models.mappo.clean_slot_orchestrator import (
    encode_prepared_slot,
    prepare_slot_state,
)


@dataclass(slots=True)
class CleanHostRngState:
    python_state: object
    numpy_state: tuple[Any, ...]


@dataclass(slots=True)
class CleanFutureSlotRngTrace:
    future_slot: int
    start_state: CleanHostRngState
    end_state: CleanHostRngState


@dataclass(slots=True)
class CleanCounterfactualContinuationResult:
    branch: CleanCounterfactualBranchResult
    horizon_slots: int
    executed_future_slots: int
    reward_sequence: tuple[float, ...]
    discounted_return: float | None
    target_dag_id: str
    target_dag_completed: bool
    target_dag_completion_time: float | None
    completed_dag_count: int
    rng_trace: tuple[CleanFutureSlotRngTrace, ...]


def capture_clean_host_rng_state() -> CleanHostRngState:
    return CleanHostRngState(
        python_state=copy.deepcopy(random.getstate()),
        numpy_state=_copy_numpy_rng_state(np.random.get_state()),
    )


def restore_clean_host_rng_state(state: CleanHostRngState) -> None:
    random.setstate(copy.deepcopy(state.python_state))
    np.random.set_state(_copy_numpy_rng_state(state.numpy_state))


def clean_host_rng_states_equal(
    left: CleanHostRngState,
    right: CleanHostRngState,
) -> bool:
    return bool(
        left.python_state == right.python_state
        and _numpy_rng_states_equal(left.numpy_state, right.numpy_state)
    )


def run_clean_counterfactual_continuation_branch(
    *,
    env_snapshot: Any,
    baseline_trace: Iterable[CleanCounterfactualDecisionTrace],
    target_decision_order: int,
    forced_uav_id: int,
    initial_rng_state: CleanHostRngState,
    horizon_slots: int,
    gamma: float,
    task_encoder: Any,
    movement_actor: Any,
    offloading_actor: Any,
    device: str | Any = "cpu",
) -> CleanCounterfactualContinuationResult:
    """Commit one B0 intervention, then follow the frozen deterministic policy."""

    import torch

    horizon = int(horizon_slots)
    if horizon not in {1, 5, 10, 20}:
        raise ValueError("Scheme-B1 feasibility supports only horizon 1, 5, 10, or 20")
    discount = float(gamma)
    if not 0.0 <= discount <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    target_trace = _target_trace(baseline_trace, target_decision_order)
    target_task = env_snapshot.task_manager.get_task(target_trace.task_id)
    if target_task is None:
        raise ValueError(f"target task is missing: {target_trace.task_id}")
    target_dag_id = str(target_task.dag_id)

    restore_clean_host_rng_state(initial_rng_state)
    branch_result, env = materialize_clean_counterfactual_branch(
        env_snapshot=env_snapshot,
        baseline_trace=baseline_trace,
        target_decision_order=target_decision_order,
        forced_uav_id=forced_uav_id,
    )
    if not branch_result.suffix_replay_feasible:
        return CleanCounterfactualContinuationResult(
            branch=branch_result,
            horizon_slots=horizon,
            executed_future_slots=0,
            reward_sequence=(),
            discounted_return=None,
            target_dag_id=target_dag_id,
            target_dag_completed=False,
            target_dag_completion_time=None,
            completed_dag_count=_completed_dag_count(env),
            rng_trace=(),
        )

    graph_builder = CleanGraphBuilder()
    graph_builder.reset()
    rewards: list[float] = []
    rng_trace: list[CleanFutureSlotRngTrace] = []
    done = False
    try:
        with torch.no_grad():
            for future_slot in range(1, horizon + 1):
                if done:
                    break
                start_state = capture_clean_host_rng_state()
                prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
                encoded = encode_prepared_slot(
                    prepared_state=prepared,
                    env=env,
                    hgnn=task_encoder,
                    movement_actor=movement_actor,
                    device=device,
                )
                selected_movement = torch.argmax(encoded.movement_logits, dim=-1)
                movement_actions = {
                    int(uav_id): int(selected_movement[index].detach().cpu().item())
                    for index, uav_id in enumerate(encoded.movement_observation.uav_ids)
                }
                env.apply_movement(movement_actions)
                ready_tasks = [
                    env.task_manager.get_task(task_id)
                    for task_id in prepared.frozen_ready_task_ids
                ]
                ready_tasks = [
                    task for task in ready_tasks if task is not None and task.is_ready
                ]
                assignments = offloading_actor.act(
                    frozen_ready_tasks=ready_tasks,
                    task_embeddings=encoded.task_embeddings.detach(),
                    graph_snapshot=prepared.graph_snapshot,
                    task_manager=env.task_manager,
                    uavs=env.uavs,
                    executor=env.executor,
                    current_time_seconds=env.current_time_seconds,
                    uav_service_positions=env.uav_service_positions,
                    ue_service_positions=env.ue_service_positions,
                    ues=env.ues,
                    deterministic=True,
                )
                _, _, done, info = env.commit_and_advance(
                    assignment_buffer=assignments
                )
                rewards.append(float(info["step_reward"]))
                rng_trace.append(
                    CleanFutureSlotRngTrace(
                        future_slot=future_slot,
                        start_state=start_state,
                        end_state=capture_clean_host_rng_state(),
                    )
                )
    finally:
        graph_builder.close()

    job = env.task_manager.jobs.get(target_dag_id)
    completed = bool(job is not None and job.completed)
    completion_time = (
        float(job.return_complete_time)
        if completed and job.return_complete_time is not None
        else None
    )
    discounted_return = sum(
        (discount ** index) * reward for index, reward in enumerate(rewards)
    )
    return CleanCounterfactualContinuationResult(
        branch=branch_result,
        horizon_slots=horizon,
        executed_future_slots=len(rewards),
        reward_sequence=tuple(rewards),
        discounted_return=float(discounted_return),
        target_dag_id=target_dag_id,
        target_dag_completed=completed,
        target_dag_completion_time=completion_time,
        completed_dag_count=_completed_dag_count(env),
        rng_trace=tuple(rng_trace),
    )


def run_clean_counterfactual_continuations_serial(
    *,
    env_snapshot: Any,
    baseline_trace: Iterable[CleanCounterfactualDecisionTrace],
    target_decision_order: int,
    initial_rng_state: CleanHostRngState,
    horizon_slots: int,
    gamma: float,
    task_encoder: Any,
    movement_actor: Any,
    offloading_actor: Any,
    device: str | Any = "cpu",
) -> tuple[CleanCounterfactualContinuationResult, ...]:
    traces = tuple(baseline_trace)
    target = _target_trace(traces, target_decision_order)
    return tuple(
        run_clean_counterfactual_continuation_branch(
            env_snapshot=env_snapshot,
            baseline_trace=traces,
            target_decision_order=target.decision_order,
            forced_uav_id=uav_id,
            initial_rng_state=initial_rng_state,
            horizon_slots=horizon_slots,
            gamma=gamma,
            task_encoder=task_encoder,
            movement_actor=movement_actor,
            offloading_actor=offloading_actor,
            device=device,
        )
        for uav_id in target.legal_uav_ids
    )


def run_clean_counterfactual_continuations_process(
    *,
    env_snapshot: Any,
    baseline_trace: Iterable[CleanCounterfactualDecisionTrace],
    target_decision_order: int,
    initial_rng_state: CleanHostRngState,
    horizon_slots: int,
    gamma: float,
    task_encoder: Any,
    movement_actor: Any,
    offloading_actor: Any,
    device: str | Any = "cpu",
    timeout_seconds: float = 60.0,
) -> tuple[CleanCounterfactualContinuationResult, ...]:
    traces = tuple(baseline_trace)
    target = _target_trace(traces, target_decision_order)
    context = multiprocessing.get_context("spawn")
    workers: list[tuple[int, Any, Any]] = []
    try:
        for uav_id in target.legal_uav_ids:
            parent, child = context.Pipe(duplex=False)
            process = context.Process(
                target=_continuation_process_worker,
                args=(
                    child,
                    env_snapshot,
                    traces,
                    target.decision_order,
                    uav_id,
                    initial_rng_state,
                    horizon_slots,
                    gamma,
                    task_encoder,
                    movement_actor,
                    offloading_actor,
                    device,
                ),
                name=f"phase4-b1-uav-{uav_id}",
            )
            process.start()
            child.close()
            workers.append((uav_id, parent, process))
        results: dict[int, CleanCounterfactualContinuationResult] = {}
        for uav_id, connection, process in workers:
            if not connection.poll(float(timeout_seconds)):
                raise TimeoutError(f"Phase4-B1 branch UAV {uav_id} timed out")
            payload = connection.recv()
            process.join(timeout=float(timeout_seconds))
            if process.is_alive():
                raise TimeoutError(f"Phase4-B1 branch UAV {uav_id} did not exit")
            if int(process.exitcode or 0) != 0 or not payload.get("ok", False):
                raise RuntimeError(
                    f"Phase4-B1 branch UAV {uav_id} failed: {payload.get('error')}"
                )
            results[uav_id] = payload["result"]
        return tuple(results[uav_id] for uav_id in target.legal_uav_ids)
    finally:
        for _, connection, process in workers:
            connection.close()
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)


def first_clean_rng_divergence_slot(
    left: CleanCounterfactualContinuationResult,
    right: CleanCounterfactualContinuationResult,
) -> int | None:
    limit = max(len(left.rng_trace), len(right.rng_trace))
    for index in range(limit):
        if index >= len(left.rng_trace) or index >= len(right.rng_trace):
            return index + 1
        left_slot = left.rng_trace[index]
        right_slot = right.rng_trace[index]
        if not clean_host_rng_states_equal(left_slot.start_state, right_slot.start_state):
            return index + 1
        if not clean_host_rng_states_equal(left_slot.end_state, right_slot.end_state):
            return index + 1
    return None


def clean_continuation_results_equal(
    left: CleanCounterfactualContinuationResult,
    right: CleanCounterfactualContinuationResult,
) -> bool:
    if (
        left.branch != right.branch
        or left.horizon_slots != right.horizon_slots
        or left.executed_future_slots != right.executed_future_slots
        or left.reward_sequence != right.reward_sequence
        or left.discounted_return != right.discounted_return
        or left.target_dag_id != right.target_dag_id
        or left.target_dag_completed != right.target_dag_completed
        or left.target_dag_completion_time != right.target_dag_completion_time
        or left.completed_dag_count != right.completed_dag_count
        or len(left.rng_trace) != len(right.rng_trace)
    ):
        return False
    return all(
        left_slot.future_slot == right_slot.future_slot
        and clean_host_rng_states_equal(left_slot.start_state, right_slot.start_state)
        and clean_host_rng_states_equal(left_slot.end_state, right_slot.end_state)
        for left_slot, right_slot in zip(left.rng_trace, right.rng_trace)
    )


def _continuation_process_worker(
    connection: Any,
    env_snapshot: Any,
    baseline_trace: tuple[CleanCounterfactualDecisionTrace, ...],
    target_decision_order: int,
    forced_uav_id: int,
    initial_rng_state: CleanHostRngState,
    horizon_slots: int,
    gamma: float,
    task_encoder: Any,
    movement_actor: Any,
    offloading_actor: Any,
    device: str | Any,
) -> None:
    try:
        result = run_clean_counterfactual_continuation_branch(
            env_snapshot=env_snapshot,
            baseline_trace=baseline_trace,
            target_decision_order=target_decision_order,
            forced_uav_id=forced_uav_id,
            initial_rng_state=initial_rng_state,
            horizon_slots=horizon_slots,
            gamma=gamma,
            task_encoder=task_encoder,
            movement_actor=movement_actor,
            offloading_actor=offloading_actor,
            device=device,
        )
        connection.send({"ok": True, "result": result})
    except BaseException as exc:
        connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def _target_trace(
    traces: Iterable[CleanCounterfactualDecisionTrace],
    target_decision_order: int,
) -> CleanCounterfactualDecisionTrace:
    matches = [
        trace for trace in traces if trace.decision_order == int(target_decision_order)
    ]
    if len(matches) != 1:
        raise ValueError("target decision_order must identify exactly one decision")
    return matches[0]


def _completed_dag_count(env: Any) -> int:
    return sum(1 for job in env.task_manager.jobs.values() if bool(job.completed))


def _copy_numpy_rng_state(state: tuple[Any, ...]) -> tuple[Any, ...]:
    return (
        str(state[0]),
        np.asarray(state[1], dtype=np.uint32).copy(),
        int(state[2]),
        int(state[3]),
        float(state[4]),
    )


def _numpy_rng_states_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return bool(
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )
