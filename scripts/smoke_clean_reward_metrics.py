from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.env import Env


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    np.random.seed(13)
    original_arrival_prob = config.DAG_BASE_ARRIVAL_PROB
    config.DAG_BASE_ARRIVAL_PROB = 0.0
    try:
        env = Env()
        env.reset()
        center = np.array([350.0, 350.0], dtype=np.float32)
        for idx, uav in enumerate(env.uavs):
            uav.pos[:2] = center + np.array([float(idx), 0.0], dtype=np.float32)
        ue = env.ues[0]
        ue.pos[:2] = center.copy()
        job = env.task_manager.create_dag_for_ue(
            ue_id=ue.id,
            source_pos=ue.pos[:2].copy(),
            current_time_step=env.time_step,
        )
        ue.enter_service_waiting(job.dag_id)
        env.task_manager.refresh_ready_states()

        idle_reward_steps = 0
        nonzero_reward_steps = 0
        final_info = {}
        for _ in range(1000):
            ready_tasks = env.task_manager.get_ready_tasks()
            assignments = {task.task_id: int(task.level % config.NUM_UAVS) for task in ready_tasks}
            _, rewards, _, info = env.step(assignments)
            _assert("step_reward" in info, "info should expose step_reward.")
            if info["completed_tasks"] == 0 and info["completed_dags"] == 0:
                _assert(info["step_reward"] == 0.0, "idle unfinished steps should not repeat delay penalties.")
                idle_reward_steps += 1
            if info["step_reward"] != 0.0:
                nonzero_reward_steps += 1
            _assert(len(rewards) == config.NUM_UAVS, "reward list should match NUM_UAVS.")
            _assert(np.isclose(sum(rewards), info["step_reward"]), "global reward should be averaged across UAVs.")
            final_info = info
            if job.completed:
                break

        _assert(job.completed, "DAG should complete during reward smoke.")
        _assert(idle_reward_steps > 0, "smoke should include unfinished idle steps.")
        _assert(nonzero_reward_steps > 0, "reward should change when tasks or DAG complete.")
        _assert(final_info["episode_reward"] != 0.0, "episode_reward should be non-zero.")
        _assert(final_info["completed_dag_count"] >= 1, "completed_dag_count should be at least one.")
        _assert(final_info["generated_dag_count"] >= 1, "generated_DAG_count should be at least one.")
        _assert(final_info["avg_dag_flowtime"] > 0.0, "avg_dag_flowtime should be positive.")
        _assert(final_info["average_dag_flowtime"] == final_info["avg_dag_flowtime"], "average DAG flowtime alias mismatch.")
        _assert(0.0 <= final_info["dag_completion_rate"] <= 1.0, "DAG completion rate should be in [0, 1].")
        _assert(final_info["dag_throughput"] > 0.0, "dag_throughput should be positive.")
        _assert(
            final_info["dag_throughput"] <= final_info["completed_dag_count"] / float(config.TIME_SLOT_DURATION),
            "throughput should be normalized by executed time, not raw slot count.",
        )
        _assert(
            final_info["average_critical_path_task_completion_delay"] >= 0.0,
            "critical path task completion delay should be non-negative.",
        )
        _assert(final_info["energy_per_completed_dag"] > 0.0, "energy per completed DAG should be positive.")
        _assert(final_info["avg_task_execution_delay"] >= 0.0, "avg_task_execution_delay should be non-negative.")
        _assert(final_info["total_task_energy"] > 0.0, "total_task_energy should be positive.")
        _assert(final_info["uav_computation_utilization"] >= 0.0, "utilization should be non-negative.")
        _assert(final_info["avg_uav_queue_length"] >= 0.0, "avg queue length should be non-negative.")
        _assert(final_info["load_balance"] >= 0.0, "load balance should be non-negative.")
        _assert(0.0 <= final_info["action_executed_rate"] <= 1.0, "action executed rate should be in [0, 1].")
        _assert(0.0 <= final_info["invalid_assignment_rate"] <= 1.0, "invalid assignment rate should be in [0, 1].")
        _assert("uav_movement_energy_total" in final_info, "movement energy metric should be present.")
        _assert(job.completion_reward_settled, "DAG completion bonus should be settled exactly once.")
        for task in env.task_manager.get_job_tasks(job.dag_id):
            _assert(task.reward_settled, f"Task {task.task_id} reward should be settled exactly once.")

        move_env = Env()
        move_env.reset()
        for uav in move_env.uavs:
            uav.pos[:2] = np.array([250.0, 250.0], dtype=np.float32)
        _, _, _, move_info = move_env.step({"movement_actions": {0: "+x"}})
        _assert(move_info["step_movement_energy"] > 0.0, "movement energy should be accounted for moving UAV.")
        _assert(move_info["step_movement_energy_penalty"] < 0.0, "movement energy penalty should affect reward.")
        _assert(move_info["step_task_energy_penalty"] == 0.0, "movement-only step should not create task energy penalty.")
    finally:
        config.DAG_BASE_ARRIVAL_PROB = original_arrival_prob

    print("smoke_clean_reward_metrics passed")


if __name__ == "__main__":
    main()
