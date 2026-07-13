from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.env import Env
from environment.metrics import CleanMetricsTracker
from scripts.eval_clean_mainline import build_arg_parser as build_eval_arg_parser, build_eval_config
from scripts.train_clean_mainline import (
    build_arg_parser as build_train_arg_parser,
    build_config_snapshot,
    checkpoint_experiment_controls,
    validate_resume_experiment_controls,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    np.random.seed(13)
    _check_completed_dag_weight_injection()
    _check_reward_control_provenance()
    original_arrival_prob = config.DAG_BASE_ARRIVAL_PROB
    original_input_range = config.INPUT_DATA_SIZE_MB_RANGE
    original_output_range = config.OUTPUT_DATA_SIZE_MB_RANGE
    original_task_constant_range = config.TASK_CONSTANT_RANGE
    config.DAG_BASE_ARRIVAL_PROB = 0.0
    # Keep this smoke independent from Phase 3 load calibration. The large input
    # guarantees at least one unfinished idle slot so the "no repeated delay
    # penalty before completion" assertion remains meaningful.
    config.INPUT_DATA_SIZE_MB_RANGE = (75.0, 75.0)
    config.OUTPUT_DATA_SIZE_MB_RANGE = (1.0, 1.0)
    config.TASK_CONSTANT_RANGE = (6, 6)
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
            current_time_step=env.current_time_seconds,
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
        config.INPUT_DATA_SIZE_MB_RANGE = original_input_range
        config.OUTPUT_DATA_SIZE_MB_RANGE = original_output_range
        config.TASK_CONSTANT_RANGE = original_task_constant_range

    print("smoke_clean_reward_metrics passed")


def _check_completed_dag_weight_injection() -> None:
    baseline_config_weight = float(config.REWARD_COMPLETED_DAG_WEIGHT)

    def _completed_dag_reward(weight: float):
        job = SimpleNamespace(
            completed=True,
            return_complete_time=1.0,
            completion_reward_settled=False,
        )
        task_manager = SimpleNamespace(
            get_job=lambda dag_id: job if dag_id == "dag" else None,
        )
        execution_stats = SimpleNamespace(
            reward_completed_task_ids=[],
            reward_completed_dag_ids=["dag"],
        )
        tracker = CleanMetricsTracker(completed_dag_weight=weight)
        tracker.reset([0])
        reward = tracker.calculate_step_reward(task_manager, execution_stats)
        tracker.reset([0])
        _assert(
            tracker.completed_dag_weight == weight,
            "reset must preserve the environment-level completed-DAG weight.",
        )
        return reward

    reward_w2 = _completed_dag_reward(2.0)
    reward_w16 = _completed_dag_reward(16.0)
    _assert(
        reward_w16.completed_dag_bonus - reward_w2.completed_dag_bonus
        == 14.0 * reward_w2.completed_dags,
        "w16-w2 bonus difference should be 14 times the completed DAG count.",
    )
    for field in (
        "time_penalty",
        "energy_penalty",
        "task_energy_penalty",
        "movement_energy_penalty",
        "movement_position_bonus",
        "completed_tasks",
        "completed_dags",
    ):
        _assert(
            getattr(reward_w2, field) == getattr(reward_w16, field),
            f"reward component {field} must not change with completed-DAG weight.",
        )
    _assert(
        Env().metrics.completed_dag_weight == baseline_config_weight,
        "default Env should retain the configured w_c baseline.",
    )
    _assert(
        float(config.REWARD_COMPLETED_DAG_WEIGHT) == baseline_config_weight,
        "run-level reward injection must not mutate the global config.",
    )
    for invalid in (-1.0, float("inf"), float("nan")):
        try:
            Env(completed_dag_weight=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid completed-DAG weight should fail: {invalid!r}")


def _check_reward_control_provenance() -> None:
    train_args = build_train_arg_parser().parse_args(["--completed-dag-weight", "16"])
    snapshot = build_config_snapshot(train_args)
    _assert(
        snapshot["cli"]["completed_dag_weight"] == 16.0,
        "training CLI snapshot should record w_c=16.",
    )
    _assert(
        snapshot["experiment_controls"]["completed_dag_weight"] == 16.0,
        "training experiment controls should record w_c=16.",
    )
    _assert(
        snapshot["experiment_controls"]["detach_critic_hgnn"] is False,
        "training experiment controls should default to shared critic-HGNN mode.",
    )
    payload = {"config": snapshot}
    _assert(
        checkpoint_experiment_controls(payload)["completed_dag_weight"] == 16.0,
        "checkpoint control resolver should recover w_c=16.",
    )
    validate_resume_experiment_controls(train_args, payload)
    mismatch_args = build_train_arg_parser().parse_args(["--completed-dag-weight", "2"])
    try:
        validate_resume_experiment_controls(mismatch_args, payload)
    except ValueError:
        pass
    else:
        raise AssertionError("resume should reject a completed-DAG weight mismatch.")
    _assert(
        checkpoint_experiment_controls({})["completed_dag_weight"]
        == float(config.REWARD_COMPLETED_DAG_WEIGHT),
        "legacy checkpoints should resolve to the configured w_c=2 baseline.",
    )
    _assert(
        checkpoint_experiment_controls({})["detach_critic_hgnn"] is False,
        "legacy checkpoints should resolve to shared critic-HGNN mode.",
    )
    detach_args = build_train_arg_parser().parse_args(
        ["--completed-dag-weight", "16", "--detach-critic-hgnn"]
    )
    try:
        validate_resume_experiment_controls(detach_args, payload)
    except ValueError:
        pass
    else:
        raise AssertionError("resume should reject a critic-HGNN detach mismatch.")
    eval_config = build_eval_config(
        build_eval_arg_parser().parse_args([]),
        {"completed_dag_weight": 16.0, "detach_critic_hgnn": True},
    )
    _assert(
        eval_config["checkpoint_experiment_controls"]["completed_dag_weight"] == 16.0,
        "evaluation provenance should report the checkpoint-derived reward weight.",
    )
    _assert(
        eval_config["checkpoint_experiment_controls"]["detach_critic_hgnn"] is True,
        "evaluation provenance should report the checkpoint-derived detach boundary.",
    )


if __name__ == "__main__":
    main()
