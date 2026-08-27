from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import multiprocessing
from typing import Any, Iterable

import numpy as np

from environment.graph_builder import CleanGraphBuilder
from marl_models.mappo.clean_counterfactual_oracle import (
    CleanCounterfactualDecisionTrace,
    materialize_clean_counterfactual_branch,
)
from marl_models.mappo.clean_counterfactual_oracle_completion import (
    CleanCompletionScaleBranchResult,
    CleanCompletionScaleDecisionResult,
    _advance_one_future_slot,
    _target_completion_time,
)
from marl_models.mappo.clean_counterfactual_oracle_rng import CleanHostRngState


_SUBSYSTEM_CODES = {"mobility": 1, "arrival": 2, "dag_generation": 3}
_NUMPY_DRAW_NAMES = ("random", "normal", "uniform", "randint", "choice", "permutation")


@dataclass(slots=True)
class CleanSemanticAudit:
    shared_semantic_keys_checked: int
    semantic_key_mismatches: list[dict[str, Any]]
    unrecognized_environment_calls: int


@dataclass(slots=True)
class CleanCommonRandomCompletionResult:
    decision: CleanCompletionScaleDecisionResult
    audit: CleanSemanticAudit


class CleanSemanticCommonRandom:
    """Scheme-B-only semantic NumPy streams derived without global RNG coupling."""

    def __init__(self, initial_rng_state: CleanHostRngState) -> None:
        self._root_entropy = _root_entropy_words(initial_rng_state)
        self._streams: dict[tuple[int, int, str], np.random.RandomState] = {}
        self._audit: dict[tuple[int, int, str], dict[str, Any]] = {}
        self._unrecognized_environment_calls = 0

    def draw(
        self,
        *,
        future_slot: int,
        ue_id: int,
        subsystem: str,
        function: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        key = (int(future_slot), int(ue_id), str(subsystem))
        stream = self._stream(key)
        method_name = "random_sample" if function == "random" else str(function)
        state_before = _random_state_marker(stream)
        value = getattr(stream, method_name)(*args, **(kwargs or {}))
        state_after = _random_state_marker(stream)
        self._audit[key]["draws"].append(
            {
                "function": str(function),
                "state_before": state_before,
                "state_after": state_after,
            }
        )
        return value

    @contextmanager
    def scoped_environment_calls(self, future_slot: int):
        originals: list[tuple[str, Any]] = []
        for name in _NUMPY_DRAW_NAMES:
            original = getattr(np.random, name)

            def wrapped(
                *args: Any,
                __name: str = name,
                __original: Any = original,
                **kwargs: Any,
            ) -> Any:
                semantic = _semantic_call_context()
                if semantic is None:
                    if _called_from_environment():
                        self._unrecognized_environment_calls += 1
                    return __original(*args, **kwargs)
                subsystem, ue_id = semantic
                return self.draw(
                    future_slot=int(future_slot),
                    ue_id=ue_id,
                    subsystem=subsystem,
                    function=__name,
                    args=args,
                    kwargs=kwargs,
                )

            originals.append((name, original))
            setattr(np.random, name, wrapped)
        try:
            yield
        finally:
            for name, original in reversed(originals):
                setattr(np.random, name, original)

    def audit_snapshot(self) -> dict[str, Any]:
        return {
            "keys": {
                _key_text(key): {
                    "seed": int(value["seed"]),
                    "draws": list(value["draws"]),
                }
                for key, value in sorted(self._audit.items())
            },
            "unrecognized_environment_calls": int(self._unrecognized_environment_calls),
        }

    def _stream(self, key: tuple[int, int, str]) -> np.random.RandomState:
        if key not in self._streams:
            future_slot, ue_id, subsystem = key
            code = _SUBSYSTEM_CODES.get(subsystem)
            if code is None:
                raise ValueError(f"unsupported semantic subsystem: {subsystem}")
            sequence = np.random.SeedSequence(
                entropy=self._root_entropy,
                spawn_key=(int(future_slot), int(ue_id), int(code)),
            )
            seed = int(sequence.generate_state(1, dtype=np.uint32)[0])
            self._streams[key] = np.random.RandomState(seed)
            self._audit[key] = {"seed": seed, "draws": []}
        return self._streams[key]


def run_clean_common_random_completion_serial(
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
) -> CleanCommonRandomCompletionResult:
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
            branch_result, env = materialize_clean_counterfactual_branch(
                env_snapshot=env_snapshot,
                baseline_trace=traces,
                target_decision_order=target.decision_order,
                forced_uav_id=uav_id,
            )
            if not branch_result.suffix_replay_feasible:
                raise ValueError("common-random target must be strict-suffix feasible")
            target_dag_id = str(env.task_manager.get_task(target.task_id).dag_id)
            completion_time = _target_completion_time(env, target_dag_id)
            branches.append(
                {
                    "uav_id": int(uav_id),
                    "env": env,
                    "builder": builder,
                    "common_random": CleanSemanticCommonRandom(initial_rng_state),
                    "rewards": [],
                    "completion_horizon": 0 if completion_time is not None else None,
                    "completion_time": completion_time,
                    "target_dag_id": target_dag_id,
                }
            )

        stop_reason = "future_slot_cap"
        if _all_completed(branches):
            stop_reason = "all_target_dags_completed"
        else:
            for future_slot in range(1, cap + 1):
                for branch in branches:
                    with branch["common_random"].scoped_environment_calls(future_slot):
                        reward, done = _advance_one_future_slot(
                            env=branch["env"],
                            graph_builder=branch["builder"],
                            task_encoder=task_encoder,
                            movement_actor=movement_actor,
                            offloading_actor=offloading_actor,
                            device=device,
                        )
                    branch["rewards"].append(reward)
                    if branch["completion_horizon"] is None:
                        completion_time = _target_completion_time(
                            branch["env"], branch["target_dag_id"]
                        )
                        if completion_time is not None:
                            branch["completion_horizon"] = future_slot
                            branch["completion_time"] = completion_time
                    if done and branch["completion_horizon"] is None:
                        raise RuntimeError("environment ended before target DAG completion")
                if _all_completed(branches):
                    stop_reason = "all_target_dags_completed"
                    break
        decision = _finalize_decision(target, branches, gamma, stop_reason)
        audit = audit_clean_semantic_common_random(
            [branch["common_random"].audit_snapshot() for branch in branches]
        )
        return CleanCommonRandomCompletionResult(decision=decision, audit=audit)
    finally:
        for builder in builders:
            builder.close()


def run_clean_common_random_completion_process(
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
) -> CleanCommonRandomCompletionResult:
    traces = tuple(baseline_trace)
    target = _target_trace(traces, target_decision_order)
    cap = _validate_cap(max_future_slots)
    context = multiprocessing.get_context("spawn")
    workers: list[tuple[int, Any, Any]] = []
    rows: dict[int, dict[str, Any]] = {}
    try:
        for uav_id in target.legal_uav_ids:
            parent, child = context.Pipe(duplex=True)
            process = context.Process(
                target=_common_random_worker,
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
                name=f"phase4-crn-uav-{uav_id}",
            )
            process.start()
            child.close()
            workers.append((uav_id, parent, process))
        for uav_id, connection, _ in workers:
            payload = _receive(connection, timeout_seconds, uav_id)
            if not payload.get("ok", False):
                raise RuntimeError(f"CRN branch UAV {uav_id}: {payload.get('error')}")
            rows[uav_id] = {
                "uav_id": uav_id,
                "rewards": [],
                "completion_horizon": payload["completion_horizon"],
                "completion_time": payload["completion_time"],
            }
        ordered = [rows[uav_id] for uav_id in target.legal_uav_ids]
        stop_reason = "future_slot_cap"
        if _all_completed(ordered):
            stop_reason = "all_target_dags_completed"
        else:
            for future_slot in range(1, cap + 1):
                for _, connection, _ in workers:
                    connection.send(("step", future_slot))
                for uav_id, connection, _ in workers:
                    payload = _receive(connection, timeout_seconds, uav_id)
                    if not payload.get("ok", False):
                        raise RuntimeError(f"CRN branch UAV {uav_id}: {payload.get('error')}")
                    row = rows[uav_id]
                    row["rewards"].append(float(payload["reward"]))
                    row["completion_horizon"] = payload["completion_horizon"]
                    row["completion_time"] = payload["completion_time"]
                if _all_completed(ordered):
                    stop_reason = "all_target_dags_completed"
                    break
        audits = []
        for _, connection, _ in workers:
            connection.send(("finish", None))
        for uav_id, connection, _ in workers:
            payload = _receive(connection, timeout_seconds, uav_id)
            if not payload.get("ok", False):
                raise RuntimeError(f"CRN audit UAV {uav_id}: {payload.get('error')}")
            audits.append(payload["audit"])
        return CleanCommonRandomCompletionResult(
            decision=_finalize_decision(target, ordered, gamma, stop_reason),
            audit=audit_clean_semantic_common_random(audits),
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


def audit_clean_semantic_common_random(
    branch_audits: list[dict[str, Any]],
) -> CleanSemanticAudit:
    all_keys = sorted(
        set().union(*(set(audit["keys"]) for audit in branch_audits))
    )
    checked = 0
    mismatches: list[dict[str, Any]] = []
    for key in all_keys:
        values = [audit["keys"][key] for audit in branch_audits if key in audit["keys"]]
        if len(values) < 2:
            continue
        checked += 1
        if any(value != values[0] for value in values[1:]):
            mismatches.append({"semantic_key": key, "branch_values": values})
    return CleanSemanticAudit(
        shared_semantic_keys_checked=checked,
        semantic_key_mismatches=mismatches,
        unrecognized_environment_calls=sum(
            int(audit["unrecognized_environment_calls"]) for audit in branch_audits
        ),
    )


def _common_random_worker(
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
    builder = CleanGraphBuilder()
    builder.reset()
    common_random = CleanSemanticCommonRandom(initial_rng_state)
    try:
        target = _target_trace(baseline_trace, target_decision_order)
        branch_result, env = materialize_clean_counterfactual_branch(
            env_snapshot=env_snapshot,
            baseline_trace=baseline_trace,
            target_decision_order=target_decision_order,
            forced_uav_id=forced_uav_id,
        )
        if not branch_result.suffix_replay_feasible:
            raise ValueError("common-random target must be strict-suffix feasible")
        target_dag_id = str(env.task_manager.get_task(target.task_id).dag_id)
        completion_time = _target_completion_time(env, target_dag_id)
        completion_horizon = 0 if completion_time is not None else None
        connection.send(
            {
                "ok": True,
                "completion_horizon": completion_horizon,
                "completion_time": completion_time,
            }
        )
        while True:
            command, future_slot = connection.recv()
            if command == "stop":
                break
            if command == "finish":
                connection.send({"ok": True, "audit": common_random.audit_snapshot()})
                break
            with common_random.scoped_environment_calls(int(future_slot)):
                reward, done = _advance_one_future_slot(
                    env=env,
                    graph_builder=builder,
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
                raise RuntimeError("environment ended before target DAG completion")
            connection.send(
                {
                    "ok": True,
                    "reward": reward,
                    "completion_horizon": completion_horizon,
                    "completion_time": completion_time,
                }
            )
    except BaseException as exc:
        connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        builder.close()
        connection.close()


def _finalize_decision(
    target: CleanCounterfactualDecisionTrace,
    branches: list[dict[str, Any]],
    gamma: float,
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
        first_rng_divergence_slot=None,
        stop_reason=stop_reason,
    )


def _semantic_call_context() -> tuple[str, int] | None:
    frame = inspect.currentframe()
    frame = None if frame is None else frame.f_back
    while frame is not None:
        filename = frame.f_code.co_filename.replace("\\", "/")
        function = frame.f_code.co_name
        if filename.endswith("/environment/dag_tasks.py") and function == "create_dag_for_ue":
            return "dag_generation", int(frame.f_locals["ue_id"])
        if filename.endswith("/environment/user_equipments.py") and function == "update_position":
            return "mobility", int(frame.f_locals["self"].id)
        if filename.endswith("/environment/env.py") and function == "_process_clean_dag_arrivals":
            return "arrival", int(frame.f_locals["ue"].id)
        frame = frame.f_back
    return None


def _called_from_environment() -> bool:
    frame = inspect.currentframe()
    frame = None if frame is None else frame.f_back
    while frame is not None:
        if "/environment/" in frame.f_code.co_filename.replace("\\", "/"):
            return True
        frame = frame.f_back
    return False


def _root_entropy_words(state: CleanHostRngState) -> tuple[int, ...]:
    numpy_state = state.numpy_state
    words = [int(value) for value in np.asarray(numpy_state[1], dtype=np.uint32)]
    cached_words = np.asarray([float(numpy_state[4])], dtype=np.float64).view(np.uint32)
    words.extend(
        [int(numpy_state[2]), int(numpy_state[3]), *(int(value) for value in cached_words)]
    )
    return tuple(words)


def _random_state_marker(stream: np.random.RandomState) -> dict[str, Any]:
    state = stream.get_state()
    return {
        "generator": str(state[0]),
        "position": int(state[2]),
        "has_cached_gaussian": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _key_text(key: tuple[int, int, str]) -> str:
    return f"slot={key[0]}|ue={key[1]}|subsystem={key[2]}"


def _target_trace(
    traces: Iterable[CleanCounterfactualDecisionTrace], decision_order: int
) -> CleanCounterfactualDecisionTrace:
    matches = [trace for trace in traces if trace.decision_order == int(decision_order)]
    if len(matches) != 1:
        raise ValueError("decision_order must identify exactly one target")
    return matches[0]


def _all_completed(branches: list[dict[str, Any]]) -> bool:
    return all(branch["completion_horizon"] is not None for branch in branches)


def _validate_cap(value: int) -> int:
    cap = int(value)
    if not 1 <= cap <= 100:
        raise ValueError("common-random completion cap must be between 1 and 100")
    return cap


def _receive(connection: Any, timeout_seconds: float, uav_id: int) -> dict[str, Any]:
    if not connection.poll(float(timeout_seconds)):
        raise TimeoutError(f"CRN branch UAV {uav_id} timed out")
    return connection.recv()
