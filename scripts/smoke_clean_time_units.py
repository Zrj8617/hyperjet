"""Time-unit consistency smoke for the clean mainline (Phase 0.2).

This smoke encodes the CORRECT clean time semantics from the spec:
one slot advances simulation time by TIME_SLOT_DURATION seconds, and all
executor timestamps (assignment/start/finish) are seconds on that clock.

Minimal reproduction: a single deterministic entry task whose upload takes
~12.0 s (75 MB at 50 Mbps, UE placed next to the UAV). Under correct
semantics, when the task is assigned in slot 1:

  (1) executor record.assignment_time == 1 * TIME_SLOT_DURATION = 5.0 seconds
  (2) compute_finish ~= 5.0 + 12.0 -> the task completes in slot 3
      (slot k covers sim time up to k * TIME_SLOT_DURATION + TIME_SLOT_DURATION)

EXPECTED TO FAIL (red) before Phase 1: the current env passes the raw slot
index into the executor while task durations are seconds, so each slot only
advances the executor clock by 1 "second" and the task completes around
slot 9 instead of slot 3.

Rule: this smoke must NOT be edited to turn green. It turns green only via
the Phase 1 time-semantics unification in env/executor/estimator.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

# Pin a deterministic minimal scene BEFORE constructing the Env. Process-local
# config mutation only; this smoke runs standalone.
config.DAG_BASE_ARRIVAL_PROB = 0.0          # no random DAG arrivals
config.DAG_MIN_TASKS = 2                    # smallest supported DAG: entry + sink
config.DAG_MAX_TASKS = 2
config.DAG_MAX_LEVELS = 2
config.DAG_MAX_PARENTS = 1
config.INPUT_DATA_SIZE_MB_RANGE = (75.0, 75.0)   # 75 MB * 8 / 50 Mbps = 12.0 s upload
config.OUTPUT_DATA_SIZE_MB_RANGE = (0.5, 0.5)
config.TASK_CONSTANT_RANGE = (1, 1)
config.TASK_COMPLEXITY_PROBS = {"n": 1.0}        # compute time ~0.008 s (negligible)
config.BASE_UPLOAD_BANDWIDTH_MBPS = [50.0]
config.BASE_DOWNLOAD_BANDWIDTH_MBPS = [50.0]
config.BANDWIDTH_LEVEL_PROBS = [1.0]

from environment.dag_tasks import TASK_STATE_COMPLETED  # noqa: E402
from environment.env import Env  # noqa: E402

EXPECTED_ASSIGNMENT_SLOT = 1
EXPECTED_COMPLETION_SLOT = 3
MAX_OBSERVED_SLOTS = 15


def main() -> int:
    np.random.seed(0)
    env = Env()
    env.reset()

    uav = env.uavs[0]
    ue = env.ues[0]
    # Put the UE next to UAV 0 so the upload distance factor is ~1.0. The UE
    # walks <= 3 m in the first prepare step, keeping upload within 12.02 s.
    ue.pos[:2] = uav.pos[:2].copy()

    job = env.task_manager.create_dag_for_ue(
        ue_id=int(ue.id),
        source_pos=ue.pos[:2].copy(),
        current_time_step=0,
    )
    entry_tasks = [
        env.task_manager.tasks[task_id]
        for task_id in job.task_ids
        if not env.task_manager.tasks[task_id].predecessors
    ]
    assert len(entry_tasks) == 1, "pinned scene must produce exactly one entry task"
    entry = entry_tasks[0]

    # Slot 1: freeze ready set, hover, assign the entry task to UAV 0.
    env.prepare_slot_state()
    env.apply_movement({})
    env.commit_and_advance(assignments={entry.task_id: int(uav.id)})
    assert env.last_assignment_buffer.entry_count == 1, "entry task was not committed; smoke is inconclusive"

    record = env.executor.task_records[entry.task_id]
    assert 11.9 <= float(record.upload_time) <= 12.1, (
        f"scenario pinning failed: upload_time={record.upload_time:.3f}s, expected ~12.0s"
    )

    failures: list[str] = []

    expected_assignment_seconds = float(EXPECTED_ASSIGNMENT_SLOT) * float(config.TIME_SLOT_DURATION)
    if abs(float(record.assignment_time) - expected_assignment_seconds) > 1e-6:
        failures.append(
            "assignment_time is not in seconds: "
            f"got {float(record.assignment_time):.3f}, expected {expected_assignment_seconds:.3f} "
            f"(slot {EXPECTED_ASSIGNMENT_SLOT} * TIME_SLOT_DURATION); the executor clock is "
            "being fed the raw slot index"
        )

    completion_slot = None
    if entry.state == TASK_STATE_COMPLETED:
        completion_slot = env.time_step
    else:
        for _ in range(MAX_OBSERVED_SLOTS):
            env.prepare_slot_state()
            env.apply_movement({})
            env.commit_and_advance(assignments={})
            if entry.state == TASK_STATE_COMPLETED:
                completion_slot = env.time_step
                break

    if completion_slot is None:
        failures.append(
            f"entry task did not complete within {MAX_OBSERVED_SLOTS} observed slots "
            f"(a ~12s upload must complete by slot {EXPECTED_COMPLETION_SLOT} under 5s slots)"
        )
    elif int(completion_slot) != EXPECTED_COMPLETION_SLOT:
        failures.append(
            f"completion slot mismatch: got slot {completion_slot}, expected slot "
            f"{EXPECTED_COMPLETION_SLOT}; each slot advances the executor clock by 1 unit "
            "instead of TIME_SLOT_DURATION seconds"
        )

    if failures:
        print("smoke_clean_time_units FAILED (expected red before the Phase 1 time-unit fix):")
        for line in failures:
            print(f"  - {line}")
        return 1

    print("smoke_clean_time_units passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
