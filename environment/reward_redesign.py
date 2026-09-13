from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


REWARD_REDENOMINATION_ARMS = (
    "N0",
    "A1",
    "A2",
    "B1",
    "B2",
    "B2D",
    "C1",
    "C2",
    "D1",
    "D2",
    "N0-LOCAL",
)


@dataclass(slots=True)
class RewardRedesignLedger:
    """Episode-local timing ledger for the frozen reward-redesign experiments."""

    arm: str
    flowtime_ref_seconds: float = 500.0
    lambda_task_seconds_per_joule: float = 1.0
    lambda_move_seconds_per_joule: float = 0.10
    last_forecast: dict[str, float] = field(default_factory=dict)
    admitted: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.arm = str(self.arm).upper()
        if self.arm not in REWARD_REDENOMINATION_ARMS:
            raise ValueError(f"unknown reward-redesign arm: {self.arm}")
        if self.arm == "D2":
            self.lambda_move_seconds_per_joule = 0.04

    @property
    def forecast_enabled(self) -> bool:
        return self.arm in {"N0", "N0-LOCAL", "A2", "B2", "B2D", "C2", "D1"}

    @property
    def analytic_forecast_advantage(self) -> bool:
        return self.arm in {"N0", "N0-LOCAL", "B2D"}

    def apply(
        self,
        *,
        env: Any,
        info: dict[str, Any],
        offloading_records: list[Any],
        initial_forecast: dict[str, float],
        final_forecast: dict[str, float],
        done: bool,
    ) -> float:
        original = float(info["step_reward"])
        if self.arm in {"N0", "N0-LOCAL", "A1", "C1"}:
            info.update(self._diagnostics(0.0, 0.0, 0.0, 0.0, 0.0, original))
            return original

        completed_ids = [str(value) for value in info.get("completed_dag_ids", [])]
        completed_flowtime = 0.0
        for dag_id in completed_ids:
            job = env.task_manager.get_job(dag_id)
            if job is not None and job.return_complete_time is not None:
                completed_flowtime += max(
                    float(job.return_complete_time) - float(job.arrival_time), 0.0
                )

        initial_cost = 0.0
        assignment_cost = 0.0
        correction_cost = 0.0
        if self.arm in {"B1", "D2"}:
            correction_cost = completed_flowtime
            if bool(done):
                correction_cost += sum(
                    max(float(env.current_time_seconds) - float(job.arrival_time), 0.0)
                    for job in env.task_manager.jobs.values()
                    if not bool(job.completed)
                )
        else:
            for dag_id, forecast_time in initial_forecast.items():
                if dag_id in self.admitted:
                    continue
                job = env.task_manager.get_job(dag_id)
                if job is None:
                    continue
                initial_cost += float(forecast_time) - float(job.arrival_time)
                self.admitted.add(dag_id)

            assignment_cost = sum(
                float(getattr(record, "forecast_delta_phi", 0.0))
                for record in offloading_records
            )

            for dag_id in completed_ids:
                job = env.task_manager.get_job(dag_id)
                if job is None or job.return_complete_time is None:
                    continue
                anchor = final_forecast.get(dag_id, self.last_forecast.get(dag_id))
                if anchor is not None:
                    correction_cost += float(job.return_complete_time) - float(anchor)
                self.last_forecast.pop(dag_id, None)
            self.last_forecast.update(
                {
                    str(dag_id): float(value)
                    for dag_id, value in final_forecast.items()
                    if dag_id not in completed_ids
                }
            )
            if bool(done):
                for dag_id, anchor in list(self.last_forecast.items()):
                    job = env.task_manager.get_job(dag_id)
                    if job is not None and not bool(job.completed):
                        correction_cost += float(env.current_time_seconds) - float(anchor)

        timing_cost = float(initial_cost + assignment_cost + correction_cost)
        time_reward = -float(timing_cost) / float(self.flowtime_ref_seconds)
        if self.arm == "A2":
            reward = original - float(info.get("step_time_penalty", 0.0)) + time_reward
        else:
            task_cost = (
                float(self.lambda_task_seconds_per_joule)
                * float(info.get("step_task_energy", 0.0))
                / float(self.flowtime_ref_seconds)
            )
            move_cost = (
                float(self.lambda_move_seconds_per_joule)
                * float(info.get("step_movement_energy", 0.0))
                / float(self.flowtime_ref_seconds)
            )
            reward = time_reward - task_cost - move_cost
        info.update(
            self._diagnostics(
                timing_cost,
                initial_cost,
                assignment_cost,
                correction_cost,
                time_reward,
                reward,
            )
        )
        return float(reward)

    def _diagnostics(
        self,
        timing_cost: float,
        initial_cost: float,
        assignment_cost: float,
        correction_cost: float,
        time_reward: float,
        reward: float,
    ) -> dict[str, Any]:
        return {
            "reward_redesign_arm": self.arm,
            "reward_redesign_timing_cost_seconds": float(timing_cost),
            "reward_redesign_initial_cost_seconds": float(initial_cost),
            "reward_redesign_assignment_cost_seconds": float(assignment_cost),
            "reward_redesign_correction_cost_seconds": float(correction_cost),
            "reward_redesign_time_reward": float(time_reward),
            "reward_redesign_total": float(reward),
            "reward_redesign_flowtime_ref_seconds": float(self.flowtime_ref_seconds),
            "reward_redesign_lambda_task": float(self.lambda_task_seconds_per_joule),
            "reward_redesign_lambda_move": float(self.lambda_move_seconds_per_joule),
        }
