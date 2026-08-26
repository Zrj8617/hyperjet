from __future__ import annotations

from dataclasses import dataclass
import multiprocessing
from typing import Any, Iterable

from environment.graph_builder import CleanGraphBuilder
from marl_models.mappo.clean_counterfactual_oracle import (
    CleanCounterfactualDecisionTrace,
    materialize_clean_counterfactual_branch,
)
from marl_models.mappo.clean_counterfactual_oracle_rng import (
    CleanHostRngState,
    capture_clean_host_rng_state,
    clean_host_rng_states_equal,
    restore_clean_host_rng_state,
)
from marl_models.mappo.clean_slot_orchestrator import (
    encode_prepared_slot,
    prepare_slot_state,
)


@dataclass(frozen=True, slots=True)
class CleanCompletionScaleBranchResult:
    forced_uav_id: int
    target_dag_completion_horizon: int | None
    target_dag_completion_time: float | None
    reward_sequence: tuple[float, ...]
    common_discounted_return: float | None


@dataclass(frozen=True, slots=True)
class CleanCompletionScaleDecisionResult:
    target_task_id: str
    decision_order: int
    baseline_uav_id: int
    candidate_uav_ids: tuple[int, ...]
    branches: tuple[CleanCompletionScaleBranchResult, ...]
    common_completion_horizon: int | None
    first_rng_divergence_slot: int | None
    stop_reason: str


def run_clean_completion_scale_serial(
    *,
    env_snapshot: Any,
    baseline_trace: Iterable[CleanCounterfactualDecisionTrace],
    target_decision_order: int,
    initial_rng_state: CleanHostRngState,
    gamma: float,
    task_encoder: Any,
    movement_actor: Any,
    offloading_actor: Any,
    max_future_slots: int = 100,
    device: str | Any = "cpu",
) -> CleanCompletionScaleDecisionResult:
    traces = tuple(baseline_trace)
    target = _target_trace(traces, target_decision_order)
    cap = _validate_cap(max_future_slots)
    branches: list[dict[str, Any]] = []
    builders: list[CleanGraphBuilder] = []
    try:
        for uav_id in target.legal_uav_ids:
            builder = CleanGraphBuilder()
            builder.reset()
            builders.append(builder)
            restore_clean_host_rng_state(initial_rng_state)
            branch_result, env = materialize_clean_counterfactual_branch(
                env_snapshot=env_snapshot,
                baseline_trace=traces,
                target_decision_order=target.decision_order,
                forced_uav_id=uav_id,
            )
            if not branch_result.suffix_replay_feasible:
                raise ValueError("completion-scale target must have feasible strict suffix replay")
            target_dag_id = str(env.task_manager.get_task(target.task_id).dag_id)
            completion_time = _target_completion_time(env, target_dag_id)
            branches.append(
                {
                    "uav_id": int(uav_id),
                    "env": env,
                    "builder": builder,
                    "rng": capture_clean_host_rng_state(),
                    "rewards": [],
                    "completion_horizon": 0 if completion_time is not None else None,
                    "completion_time": completion_time,
                    "target_dag_id": target_dag_id,
                }
            )

        stop_reason = "future_slot_cap"
        divergence_slot: int | None = None
        if _all_completed(branches):
            stop_reason = "all_target_dags_completed"
        else:
            for future_slot in range(1, cap + 1):
                if not _rng_states_aligned([branch["rng"] for branch in branches]):
                    divergence_slot = future_slot
                    stop_reason = "rng_divergence"
                    break
                for branch in branches:
                    restore_clean_host_rng_state(branch["rng"])
                    reward, done = _advance_one_future_slot(
                        env=branch["env"],
                        graph_builder=branch["builder"],
                        task_encoder=task_encoder,
                        movement_actor=movement_actor,
                        offloading_actor=offloading_actor,
                        device=device,
                    )
                    branch["rewards"].append(reward)
                    branch["rng"] = capture_clean_host_rng_state()
                    if branch["completion_horizon"] is None:
                        completion_time = _target_completion_time(
                            branch["env"], branch["target_dag_id"]
                        )
                        if completion_time is not None:
                            branch["completion_horizon"] = future_slot
                            branch["completion_time"] = completion_time
                    if done and branch["completion_horizon"] is None:
                        raise RuntimeError("environment ended before its target DAG completed")
                if not _rng_states_aligned([branch["rng"] for branch in branches]):
                    divergence_slot = future_slot
                    stop_reason = "rng_divergence"
                    break
                if _all_completed(branches):
                    stop_reason = "all_target_dags_completed"
                    break
        return _finalize_result(
            target=target,
            branches=branches,
            gamma=gamma,
            divergence_slot=divergence_slot,
            stop_reason=stop_reason,
        )
    finally:
        for builder in builders:
            builder.close()


def run_clean_completion_scale_process(
    *,
    env_snapshot: Any,
    baseline_trace: Iterable[CleanCounterfactualDecisionTrace],
    target_decision_order: int,
    initial_rng_state: CleanHostRngState,
    gamma: float,
    task_encoder: Any,
    movement_actor: Any,
    offloading_actor: Any,
    max_future_slots: int = 100,
    device: str | Any = "cpu",
    timeout_seconds: float = 60.0,
) -> CleanCompletionScaleDecisionResult:
    traces = tuple(baseline_trace)
    target = _target_trace(traces, target_decision_order)
    cap = _validate_cap(max_future_slots)
    context = multiprocessing.get_context("spawn")
    workers: list[tuple[int, Any, Any]] = []
    branch_rows: dict[int, dict[str, Any]] = {}
    try:
        for uav_id in target.legal_uav_ids:
            parent, child = context.Pipe(duplex=True)
            process = context.Process(
                target=_completion_worker,
                args=(
                    child,
                    env_snapshot,
                    traces,
                    target.decision_order,
                    uav_id,
                    initial_rng_state,
                    task_encoder,
                    movement_actor,
                    offloading_actor,
                    device,
                ),
                name=f"phase4-completion-uav-{uav_id}",
            )
            process.start()
            child.close()
            workers.append((uav_id, parent, process))
        for uav_id, connection, _ in workers:
            payload = _receive(connection, timeout_seconds, uav_id)
            if not payload.get("ok", False):
                raise RuntimeError(f"completion branch UAV {uav_id}: {payload.get('error')}")
            branch_rows[uav_id] = {
                "uav_id": uav_id,
                "rng": payload["rng"],
                "rewards": [],
                "completion_horizon": payload["completion_horizon"],
                "completion_time": payload["completion_time"],
            }

        stop_reason = "future_slot_cap"
        divergence_slot: int | None = None
        ordered = [branch_rows[uav_id] for uav_id in target.legal_uav_ids]
        if _all_completed(ordered):
            stop_reason = "all_target_dags_completed"
        else:
            for future_slot in range(1, cap + 1):
                if not _rng_states_aligned([branch["rng"] for branch in ordered]):
                    divergence_slot = future_slot
                    stop_reason = "rng_divergence"
                    break
                for _, connection, _ in workers:
                    connection.send(("step", future_slot))
                for uav_id, connection, _ in workers:
                    payload = _receive(connection, timeout_seconds, uav_id)
                    if not payload.get("ok", False):
                        raise RuntimeError(
                            f"completion branch UAV {uav_id}: {payload.get('error')}"
                        )
                    row = branch_rows[uav_id]
                    row["rng"] = payload["rng"]
                    row["rewards"].append(float(payload["reward"]))
                    row["completion_horizon"] = payload["completion_horizon"]
                    row["completion_time"] = payload["completion_time"]
                if not _rng_states_aligned([branch["rng"] for branch in ordered]):
                    divergence_slot = future_slot
                    stop_reason = "rng_divergence"
                    break
                if _all_completed(ordered):
                    stop_reason = "all_target_dags_completed"
                    break
        return _finalize_result(
            target=target,
            branches=ordered,
            gamma=gamma,
            divergence_slot=divergence_slot,
            stop_reason=stop_reason,
        )
    finally:
        for _, connection, process in workers:
            try:
                if process.is_alive():
                    connection.send(("stop", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
            connection.close()
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)


def clean_completion_scale_results_equal(
    left: CleanCompletionScaleDecisionResult,
    right: CleanCompletionScaleDecisionResult,
) -> bool:
    return left == right


def _completion_worker(
    connection: Any,
    env_snapshot: Any,
    baseline_trace: tuple[CleanCounterfactualDecisionTrace, ...],
    target_decision_order: int,
    forced_uav_id: int,
    initial_rng_state: CleanHostRngState,
    task_encoder: Any,
    movement_actor: Any,
    offloading_actor: Any,
    device: str | Any,
) -> None:
    graph_builder = CleanGraphBuilder()
    graph_builder.reset()
    try:
        target = _target_trace(baseline_trace, target_decision_order)
        restore_clean_host_rng_state(initial_rng_state)
        branch_result, env = materialize_clean_counterfactual_branch(
            env_snapshot=env_snapshot,
            baseline_trace=baseline_trace,
            target_decision_order=target_decision_order,
            forced_uav_id=forced_uav_id,
        )
        if not branch_result.suffix_replay_feasible:
            raise ValueError("completion-scale target must have feasible strict suffix replay")
        target_dag_id = str(env.task_manager.get_task(target.task_id).dag_id)
        completion_time = _target_completion_time(env, target_dag_id)
        completion_horizon = 0 if completion_time is not None else None
        connection.send(
            {
                "ok": True,
                "rng": capture_clean_host_rng_state(),
                "completion_horizon": completion_horizon,
                "completion_time": completion_time,
            }
        )
        while True:
            command, future_slot = connection.recv()
            if command == "stop":
                break
            reward, done = _advance_one_future_slot(
                env=env,
                graph_builder=graph_builder,
                task_encoder=task_encoder,
                movement_actor=movement_actor,
                offloading_actor=offloading_actor,
                device=device,
            )
            if completion_horizon is None:
                completion_time = _target_completion_time(env, target_dag_id)
                if completion_time is not None:
                    completion_horizon = int(future_slot)
            if done and completion_horizon is None:
                raise RuntimeError("environment ended before its target DAG completed")
            connection.send(
                {
                    "ok": True,
                    "reward": reward,
                    "rng": capture_clean_host_rng_state(),
                    "completion_horizon": completion_horizon,
                    "completion_time": completion_time,
                }
            )
    except BaseException as exc:
        connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        graph_builder.close()
        connection.close()


def _advance_one_future_slot(
    *,
    env: Any,
    graph_builder: CleanGraphBuilder,
    task_encoder: Any,
    movement_actor: Any,
    offloading_actor: Any,
    device: str | Any,
) -> tuple[float, bool]:
    import torch

    with torch.no_grad():
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
        ready_tasks = [task for task in ready_tasks if task is not None and task.is_ready]
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
        _, _, done, info = env.commit_and_advance(assignment_buffer=assignments)
    return float(info["step_reward"]), bool(done)


def _finalize_result(
    *,
    target: CleanCounterfactualDecisionTrace,
    branches: list[dict[str, Any]],
    gamma: float,
    divergence_slot: int | None,
    stop_reason: str,
) -> CleanCompletionScaleDecisionResult:
    horizons = [branch["completion_horizon"] for branch in branches]
    common_horizon = max(horizons) if all(value is not None for value in horizons) else None
    results = []
    for branch in branches:
        common_return = None
        if common_horizon is not None:
            common_return = float(
                sum(
                    (float(gamma) ** index) * reward
                    for index, reward in enumerate(branch["rewards"][:common_horizon])
                )
            )
        results.append(
            CleanCompletionScaleBranchResult(
                forced_uav_id=int(branch["uav_id"]),
                target_dag_completion_horizon=branch["completion_horizon"],
                target_dag_completion_time=branch["completion_time"],
                reward_sequence=tuple(float(value) for value in branch["rewards"]),
                common_discounted_return=common_return,
            )
        )
    return CleanCompletionScaleDecisionResult(
        target_task_id=target.task_id,
        decision_order=target.decision_order,
        baseline_uav_id=target.selected_uav_id,
        candidate_uav_ids=target.legal_uav_ids,
        branches=tuple(results),
        common_completion_horizon=common_horizon,
        first_rng_divergence_slot=divergence_slot,
        stop_reason=stop_reason,
    )


def _target_trace(
    traces: Iterable[CleanCounterfactualDecisionTrace],
    decision_order: int,
) -> CleanCounterfactualDecisionTrace:
    matches = [trace for trace in traces if trace.decision_order == int(decision_order)]
    if len(matches) != 1:
        raise ValueError("decision_order must identify exactly one target")
    return matches[0]


def _target_completion_time(env: Any, dag_id: str) -> float | None:
    job = env.task_manager.jobs.get(str(dag_id))
    if job is None or not bool(job.completed) or job.return_complete_time is None:
        return None
    return float(job.return_complete_time)


def _all_completed(branches: list[dict[str, Any]]) -> bool:
    return all(branch["completion_horizon"] is not None for branch in branches)


def _rng_states_aligned(states: list[CleanHostRngState]) -> bool:
    return all(clean_host_rng_states_equal(states[0], state) for state in states[1:])


def _validate_cap(value: int) -> int:
    cap = int(value)
    if not 1 <= cap <= 100:
        raise ValueError("completion-scale future-slot cap must be between 1 and 100")
    return cap


def _receive(connection: Any, timeout_seconds: float, uav_id: int) -> dict[str, Any]:
    if not connection.poll(float(timeout_seconds)):
        raise TimeoutError(f"completion branch UAV {uav_id} timed out")
    return connection.recv()
