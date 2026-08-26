from __future__ import annotations

from contextlib import contextmanager
import inspect
import random
from typing import Any, Iterable

import numpy as np

from environment.graph_builder import CleanGraphBuilder
from marl_models.mappo.clean_counterfactual_oracle import (
    CleanCounterfactualDecisionTrace,
    materialize_clean_counterfactual_branch,
)
from marl_models.mappo.clean_counterfactual_oracle_completion import (
    _advance_one_future_slot,
)
from marl_models.mappo.clean_counterfactual_oracle_rng import (
    CleanHostRngState,
    capture_clean_host_rng_state,
    clean_host_rng_states_equal,
    restore_clean_host_rng_state,
)


_NUMPY_DRAW_NAMES = ("random", "normal", "uniform", "randint", "choice", "permutation")
_PYTHON_DRAW_NAMES = ("random", "uniform", "randint", "randrange", "choice", "choices", "sample", "gauss")


def localize_clean_rng_call_divergence(
    *,
    env_snapshot: Any,
    baseline_trace: Iterable[CleanCounterfactualDecisionTrace],
    target_decision_order: int,
    initial_rng_state: CleanHostRngState,
    task_encoder: Any,
    movement_actor: Any,
    offloading_actor: Any,
    max_future_slots: int = 100,
    device: str | Any = "cpu",
) -> dict[str, Any]:
    """Stop at the first branch-level Python/NumPy RNG call-sequence difference."""

    traces = tuple(baseline_trace)
    target = _target_trace(traces, target_decision_order)
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
                raise ValueError("RNG localization target must be strict-suffix feasible")
            branches.append(
                {
                    "uav_id": int(uav_id),
                    "env": env,
                    "builder": builder,
                    "rng": capture_clean_host_rng_state(),
                }
            )

        for future_slot in range(1, min(int(max_future_slots), 100) + 1):
            if not _states_aligned(branches):
                return {
                    "task": target.task_id,
                    "decision_order": target.decision_order,
                    "first_divergence_slot": future_slot,
                    "divergence_phase": "slot_start",
                    "calls": {},
                    "branch_context": {
                        str(branch["uav_id"]): _branch_context(branch["env"])
                        for branch in branches
                    },
                }
            slot_traces: dict[int, list[dict[str, Any]]] = {}
            contexts = {
                int(branch["uav_id"]): _branch_context(branch["env"])
                for branch in branches
            }
            for branch in branches:
                restore_clean_host_rng_state(branch["rng"])
                calls: list[dict[str, Any]] = []
                with _trace_environment_rng_calls(calls):
                    _advance_one_future_slot(
                        env=branch["env"],
                        graph_builder=branch["builder"],
                        task_encoder=task_encoder,
                        movement_actor=movement_actor,
                        offloading_actor=offloading_actor,
                        device=device,
                    )
                branch["rng"] = capture_clean_host_rng_state()
                slot_traces[int(branch["uav_id"])] = calls
            mismatch = _first_call_mismatch(slot_traces)
            if mismatch is not None or not _states_aligned(branches):
                return {
                    "task": target.task_id,
                    "decision_order": target.decision_order,
                    "first_divergence_slot": future_slot,
                    "divergence_phase": "rng_call_sequence",
                    "first_call_index": mismatch,
                    "calls": {
                        str(uav_id): (
                            calls[mismatch]
                            if mismatch is not None and mismatch < len(calls)
                            else None
                        )
                        for uav_id, calls in slot_traces.items()
                    },
                    "slot_call_counts": {
                        str(uav_id): len(calls) for uav_id, calls in slot_traces.items()
                    },
                    "branch_context": {
                        str(uav_id): context for uav_id, context in contexts.items()
                    },
                    "direct_cause": _infer_direct_cause(
                        mismatch=mismatch,
                        traces=slot_traces,
                        contexts=contexts,
                        max_active_dags=int(env_snapshot.max_active_dags_per_ue),
                    ),
                }
        return {
            "task": target.task_id,
            "decision_order": target.decision_order,
            "first_divergence_slot": None,
            "divergence_phase": None,
            "calls": {},
            "direct_cause": "no divergence through localization cap",
        }
    finally:
        for builder in builders:
            builder.close()


@contextmanager
def _trace_environment_rng_calls(output: list[dict[str, Any]]):
    originals: list[tuple[Any, str, Any]] = []

    def install(owner: Any, library: str, name: str) -> None:
        original = getattr(owner, name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            location = _environment_call_location()
            if location is None:
                return original(*args, **kwargs)
            before = _state_marker(library)
            value = original(*args, **kwargs)
            after = _state_marker(library)
            output.append(
                {
                    "library": library,
                    "draw_function": name,
                    "draw_occurred": True,
                    **location,
                    "rng_state_before": before,
                    "rng_state_after": after,
                }
            )
            return value

        originals.append((owner, name, original))
        setattr(owner, name, wrapped)

    for name in _NUMPY_DRAW_NAMES:
        install(np.random, "numpy", name)
    for name in _PYTHON_DRAW_NAMES:
        install(random, "python", name)
    try:
        yield
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)


def _environment_call_location() -> dict[str, Any] | None:
    frame = inspect.currentframe()
    if frame is None:
        return None
    frame = frame.f_back
    while frame is not None:
        normalized = frame.f_code.co_filename.replace("\\", "/")
        if "/environment/" in normalized:
            entity_id = None
            local_self = frame.f_locals.get("self")
            if local_self is not None and hasattr(local_self, "id"):
                entity_id = int(local_self.id)
            elif "ue" in frame.f_locals and hasattr(frame.f_locals["ue"], "id"):
                entity_id = int(frame.f_locals["ue"].id)
            elif "ue_id" in frame.f_locals:
                entity_id = int(frame.f_locals["ue_id"])
            return {
                "subsystem": _subsystem(normalized, frame.f_code.co_name),
                "call_site": f"{normalized.rsplit('/', 2)[-2]}/{normalized.rsplit('/', 1)[-1]}:{frame.f_lineno}",
                "function": frame.f_code.co_name,
                "entity_id": entity_id,
            }
        frame = frame.f_back
    return None


def _subsystem(filename: str, function: str) -> str:
    if filename.endswith("user_equipments.py") and function == "update_position":
        return "ue_mobility"
    if filename.endswith("env.py") and function == "_process_clean_dag_arrivals":
        return "dag_arrival"
    if filename.endswith("dag_tasks.py"):
        return "dag_generation"
    if filename.endswith("uavs.py"):
        return "uav_state"
    return "environment_other"


def _state_marker(library: str) -> dict[str, Any]:
    if library == "numpy":
        state = np.random.get_state()
        return {
            "generator": str(state[0]),
            "position": int(state[2]),
            "has_cached_gaussian": int(state[3]),
        }
    state = random.getstate()
    return {"version": int(state[0]), "position": int(state[1][-1])}


def _first_call_mismatch(traces: dict[int, list[dict[str, Any]]]) -> int | None:
    rows = list(traces.values())
    width = max(len(row) for row in rows)
    for index in range(width):
        identities = [
            _call_identity(row[index]) if index < len(row) else None for row in rows
        ]
        if any(identity != identities[0] for identity in identities[1:]):
            return index
    return None


def _call_identity(call: dict[str, Any]) -> tuple[Any, ...]:
    return (
        call["library"],
        call["draw_function"],
        call["subsystem"],
        call["function"],
        call["entity_id"],
    )


def _branch_context(env: Any) -> dict[str, Any]:
    active_jobs: dict[str, list[str]] = {}
    completed_jobs: dict[str, list[str]] = {}
    for ue in env.ues:
        ue_id = int(ue.id)
        active_jobs[str(ue_id)] = [
            str(job.dag_id)
            for job in env.task_manager.jobs.values()
            if int(job.ue_id) == ue_id and not bool(job.completed)
        ]
        completed_jobs[str(ue_id)] = [
            str(job.dag_id)
            for job in env.task_manager.jobs.values()
            if int(job.ue_id) == ue_id and bool(job.completed)
        ]
    return {"active_jobs_by_ue": active_jobs, "completed_jobs_by_ue": completed_jobs}


def _infer_direct_cause(
    *,
    mismatch: int | None,
    traces: dict[int, list[dict[str, Any]]],
    contexts: dict[int, dict[str, Any]],
    max_active_dags: int,
) -> str:
    if mismatch is None:
        return "RNG state changed despite identical traced environment call sequence"
    calls = {
        uav_id: rows[mismatch] if mismatch < len(rows) else None
        for uav_id, rows in traces.items()
    }
    arrival_calls = [
        call for call in calls.values() if call is not None and call["subsystem"] == "dag_arrival"
    ]
    if arrival_calls:
        ue_id = arrival_calls[0].get("entity_id")
        if ue_id is not None:
            counts = {
                uav_id: len(context["active_jobs_by_ue"].get(str(ue_id), []))
                for uav_id, context in contexts.items()
            }
            if min(counts.values()) < max_active_dags <= max(counts.values()):
                return (
                    f"UE {ue_id} active-DAG eligibility differs across branches: "
                    f"active counts {counts}, cap={max_active_dags}; eligible branches draw "
                    "dag_arrival while capped branches skip it"
                )
    return f"first traced call identity differs across branches: {calls}"


def _states_aligned(branches: list[dict[str, Any]]) -> bool:
    first = branches[0]["rng"]
    return all(clean_host_rng_states_equal(first, branch["rng"]) for branch in branches[1:])


def _target_trace(
    traces: Iterable[CleanCounterfactualDecisionTrace],
    decision_order: int,
) -> CleanCounterfactualDecisionTrace:
    matches = [trace for trace in traces if trace.decision_order == int(decision_order)]
    if len(matches) != 1:
        raise ValueError("decision_order must identify exactly one target")
    return matches[0]
