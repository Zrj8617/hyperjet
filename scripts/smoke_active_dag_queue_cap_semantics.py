from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import numpy as np

from environment.assignment import TemporaryReservationState, _dynamic_uav_features
from environment.capacity_factorial_diagnostic import (
    FactorialEpisodeTracker,
    candidate_legality_reasons,
    generate_scenario_tape,
    run_factorial_episode,
)
from environment.dag_tasks import TASK_STATE_READY, TaskNode
from environment.diagnostic_capacity import (
    DIAGNOSTIC_QUEUE_WORKLOAD_NORM_REF,
    DiagnosticCapacityContext,
)


class _Executor:
    def __init__(self) -> None:
        self.uav_queues = {index: [] for index in range(5)}
        self.uav_available_time = {index: 0.0 for index in range(5)}
        self.task_records = {}

    def is_task_scheduled(self, task_id: str) -> bool:
        return str(task_id) in self.task_records


class _Uav:
    def __init__(self, uav_id: int) -> None:
        self.id = uav_id
        self.pos = np.asarray([100.0 + uav_id, 200.0 + uav_id, config.UAV_ALTITUDE], dtype=np.float32)


def _task() -> TaskNode:
    task = TaskNode(
        task_id="diag_task",
        dag_id="diag_dag",
        ue_id=0,
        input_data_size_mb=1.0,
        output_data_size_mb=0.5,
        task_complexity="n",
        task_constant=1,
        num_operation=1_000_000.0,
        level=0,
        source_pos=np.asarray([0.0, 0.0], dtype=np.float32),
    )
    task.state = TASK_STATE_READY
    return task


def main() -> int:
    assert DIAGNOSTIC_QUEUE_WORKLOAD_NORM_REF == 80_000_000.0
    context16 = DiagnosticCapacityContext(hard_queue_cap=16)
    context999 = DiagnosticCapacityContext(hard_queue_cap=999)
    state = TemporaryReservationState(
        queue_lengths={0: 20},
        available_times={0: 0.0},
        queued_workloads={0: 90_000_000.0},
        slot_assigned_counts={0: 20},
    )
    uav = _Uav(0)
    features16 = _dynamic_uav_features(
        uav=uav,
        uav_id=0,
        state_view=state,
        current_time_seconds=0.0,
        uav_service_positions={0: uav.pos[:2]},
        capacity_context=context16,
    )
    features999 = _dynamic_uav_features(
        uav=uav,
        uav_id=0,
        state_view=state,
        current_time_seconds=0.0,
        uav_service_positions={0: uav.pos[:2]},
        capacity_context=context999,
    )
    assert features16.shape == (7,) and np.array_equal(features16, features999)
    assert features16[2] == 1.0 and features16[3] == 0.0 and features16[5] == 1.0 and features16[6] == 1.0
    assert state.remaining_slots(0, context16) == 0
    assert state.remaining_slots(0, context999) == 979

    executor = _Executor()
    task = _task()
    full_state = TemporaryReservationState(
        queue_lengths={index: 16 for index in range(5)},
        slot_assigned_counts={index: 0 for index in range(5)},
    )
    reasons = {
        index: candidate_legality_reasons(
            task=task,
            uav_id=index,
            reservation=full_state,
            valid_uav_ids=set(range(5)),
            executor=executor,
            capacity_context=context16,
        )
        for index in range(5)
    }
    assert all(value == frozenset({"queue_full"}) for value in reasons.values())
    tracker = FactorialEpisodeTracker(tuple(range(5)))
    tracker.observe_executor_queues(executor)
    tracker.observe_decision(task_id=task.task_id, reservation=full_state, reasons_by_uav=reasons, legal_uav_ids=[])
    assert tracker.all_uavs_full_decision_count == 1
    task.state = "WAITING_DEPENDENCY"
    mixed = {
        index: candidate_legality_reasons(
            task=task,
            uav_id=index,
            reservation=full_state,
            valid_uav_ids=set(range(5)),
            executor=executor,
            capacity_context=context16,
        )
        for index in range(5)
    }
    tracker.observe_decision(task_id=task.task_id, reservation=full_state, reasons_by_uav=mixed, legal_uav_ids=[])
    assert tracker.all_uavs_full_decision_count == 1
    assert tracker.max_executor_queue_length_by_uav != tracker.max_temporary_queue_length_by_uav

    consumers = (
        ROOT / "environment" / "env.py",
        ROOT / "environment" / "dag_tasks.py",
        ROOT / "environment" / "assignment.py",
        ROOT / "environment" / "task_execution.py",
        ROOT / "environment" / "metrics.py",
    )
    assert all("active_dag_id" not in path.read_text(encoding="utf-8") for path in consumers)

    original_probability = config.DAG_BASE_ARRIVAL_PROB
    original_multiplier = config.DAG_HOTSPOT_ARRIVAL_MULTIPLIER
    try:
        config.DAG_BASE_ARRIVAL_PROB = 0.0
        config.DAG_HOTSPOT_ARRIVAL_MULTIPLIER = 1.0
        tape = generate_scenario_tape(scenario_seeds=(13,), episodes=1, load_slots=2, episode_slots=200)
    finally:
        config.DAG_BASE_ARRIVAL_PROB = original_probability
        config.DAG_HOTSPOT_ARRIVAL_MULTIPLIER = original_multiplier
    row = run_factorial_episode(
        episode_payload=tape["episodes"][0],
        cell="D",
        policy="random_hash",
        full_tape_checksum=tape["full_tape_checksum"],
        pilot_prefix_checksum=tape["pilot_prefix_checksum"],
    )
    assert row["arrival_opportunity_count"] == 2 * config.NUM_UES
    assert row["no_arrival_event_count"] == 2 * config.NUM_UES
    assert row["offered_dag_count"] == 0
    assert row["average_dag_flowtime"] is None
    assert row["completed_dag_flowtime_sum"] == 0.0
    assert row["completed_dag_flowtime_count"] == 0
    assert row["dag_completion_rate_admitted"] is None
    assert row["dag_completion_rate_offered"] is None
    assert row["technical_pass"] is True
    assert all(
        record["physical_slot"] == record["slot_index"] + 1
        for record in row["assignment_legality_records"]
    )
    print("SMOKE_ACTIVE_DAG_QUEUE_CAP_SEMANTICS_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
