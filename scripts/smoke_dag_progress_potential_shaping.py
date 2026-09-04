"""Focused correctness smoke for the approved DAG-progress shaping probe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.dag_tasks import TASK_STATE_COMPLETED  # noqa: E402
from marl_models.mappo.clean_dag_progress_shaping import (  # noqa: E402
    cumulative_dag_progress_potential,
    shape_transition_reward,
)
from marl_models.mappo.clean_slot_orchestrator import (  # noqa: E402
    CleanSlotRolloutBuffer,
    CleanSlotRolloutRecord,
)
from marl_models.mappo.clean_trainer import (  # noqa: E402
    compute_de_shaped_multi_trajectory_gae,
    compute_multi_trajectory_gae,
)


@dataclass
class _Task:
    state: str


@dataclass
class _Job:
    task_ids: list[str]
    completed: bool


@dataclass
class _Manager:
    jobs: dict[str, _Job]
    tasks: dict[str, _Task]


def _record(*, value: float, reward: float, terminated: bool = False) -> CleanSlotRolloutRecord:
    return CleanSlotRolloutRecord(
        slot_index=0,
        graph_snapshot=None,  # type: ignore[arg-type]
        critic_non_graph_input=np.zeros((1,), dtype=np.float32),
        value=float(value),
        reward=float(reward),
        original_reward=float(reward),
        terminated=bool(terminated),
    )


def main() -> int:
    manager = _Manager(
        jobs={"dag_0": _Job(task_ids=["a", "b"], completed=False)},
        tasks={"a": _Task(TASK_STATE_COMPLETED), "b": _Task("IN_SERVICE")},
    )
    before = cumulative_dag_progress_potential(manager, completed_dag_weight=8.0)
    assert before == 4.0
    manager.tasks["b"].state = TASK_STATE_COMPLETED
    manager.jobs["dag_0"].completed = True
    after = cumulative_dag_progress_potential(manager, completed_dag_weight=8.0)
    assert after == 8.0, "completed DAG contribution must not drop"

    shaped, increment = shape_transition_reward(
        2.0,
        potential_before=before,
        potential_after=after,
        gamma=0.9,
        done=False,
    )
    assert shaped == 5.2 and increment == 3.2
    terminal_reward, terminal_increment = shape_transition_reward(
        2.0,
        potential_before=before,
        potential_after=after,
        gamma=0.9,
        done=True,
    )
    assert terminal_reward == 2.0 and terminal_increment == 0.0

    random.seed(17)
    np.random.seed(17)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    _ = cumulative_dag_progress_potential(manager, completed_dag_weight=8.0)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]

    buffer = CleanSlotRolloutBuffer()
    buffer.append(_record(value=0.25, reward=1.0))
    buffer.append(_record(value=-0.5, reward=-0.25, terminated=True))
    buffer.close(bootstrap_value=0.0)
    regular = compute_multi_trajectory_gae([buffer], gamma=0.91, gae_lambda=0.87)
    de_shaped = compute_de_shaped_multi_trajectory_gae(
        [buffer], gamma=0.91, gae_lambda=0.87
    )
    assert np.array_equal(regular[0], de_shaped[0])
    assert np.array_equal(regular[1], de_shaped[1])
    print("dag-progress shaping smoke PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
