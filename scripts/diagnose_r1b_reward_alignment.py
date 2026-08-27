"""Stage R1-B reward component, alignment, and timing audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.env import Env  # noqa: E402
from marl_models.mappo.clean_counterfactual_oracle_common_random import (  # noqa: E402
    CleanSemanticCommonRandom,
    audit_clean_semantic_common_random,
)
from marl_models.mappo.clean_counterfactual_oracle_rng import (  # noqa: E402
    capture_clean_host_rng_state,
    clean_host_rng_states_equal,
)
from scripts.diagnose_r1a_environment_load_feedback import (  # noqa: E402
    POLICIES,
    SEEDS,
    _git_head,
    _pending_task_count,
    _queue_lengths,
    _select_assignments,
    _set_seed,
)


FORMAL_SLOTS = 500
BENCHMARK_SLOTS = 100
CHECKPOINT_SLOTS = (50, 100, 250, 500)
OUTPUT_PATH = ROOT / "logs" / "r1b_reward_alignment.json"
COMPONENT_INFO_KEYS = {
    "time_penalty": "step_time_penalty",
    "task_energy_penalty": "step_task_energy_penalty",
    "movement_energy_penalty": "step_movement_energy_penalty",
    "completed_dag_bonus": "step_completed_dag_bonus",
    "movement_position_bonus": "step_movement_position_bonus",
}
ALIGNMENT_METRICS = {
    "completed_dags": ("higher", "completed_dag_count"),
    "completion_rate": ("higher", "completion_rate"),
    "avg_flowtime": ("lower", "avg_dag_flowtime"),
    "critical_path_delay": ("lower", "avg_critical_path_delay"),
    "task_energy_per_completed_dag": ("lower", "task_energy_per_completed_dag"),
    "avg_queue": ("lower", "avg_uav_queue"),
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixed Stage R1-B reward audit.")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser


def _longest_zero_gap(values: list[float]) -> int:
    longest = 0
    current = 0
    for value in values:
        if float(value) == 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _component_summary(
    component_values: dict[str, list[float]], slots: int
) -> tuple[dict[str, dict[str, Any]], float, dict[str, dict[str, float]]]:
    absolute_total = sum(
        float(np.sum(np.abs(np.asarray(values, dtype=np.float64))))
        for values in component_values.values()
    )
    summaries: dict[str, dict[str, Any]] = {}
    timing: dict[str, dict[str, float]] = {}
    cumulative_components: dict[str, float] = {}
    for component, values in component_values.items():
        array = np.asarray(values, dtype=np.float64)
        cumulative = np.cumsum(array)
        nonzero = np.flatnonzero(array != 0.0)
        absolute_sum = float(np.sum(np.abs(array)))
        cumulative_sum = float(np.sum(array))
        cumulative_components[component] = cumulative_sum
        summaries[component] = {
            "cumulative_sum": cumulative_sum,
            "mean": float(np.mean(array)) if array.size else 0.0,
            "std": float(np.std(array)) if array.size else 0.0,
            "nonzero_slot_count": int(nonzero.size),
            "nonzero_fraction": float(nonzero.size / max(int(slots), 1)),
            "absolute_contribution_sum": absolute_sum,
            "absolute_contribution_share": float(
                absolute_sum / absolute_total if absolute_total > 0.0 else 0.0
            ),
            "first_nonzero_slot": int(nonzero[0] + 1) if nonzero.size else None,
            "longest_zero_gap": _longest_zero_gap(values),
        }
        timing[component] = {
            str(checkpoint): float(cumulative[checkpoint - 1])
            for checkpoint in CHECKPOINT_SLOTS
            if checkpoint <= int(slots)
        }

    denominator = sum(abs(value) for value in cumulative_components.values())
    cancellation_ratio = float(
        1.0 - abs(sum(cumulative_components.values())) / denominator
        if denominator > 0.0
        else 0.0
    )
    return summaries, cancellation_ratio, timing


def run_one(
    *, seed: int, policy: str, slots: int
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    _set_seed(seed)
    policy_rng = np.random.default_rng(int(seed) + 1_000_003)
    env = Env()
    env.reset()
    root_rng_state = capture_clean_host_rng_state()
    common_random = CleanSemanticCommonRandom(root_rng_state)

    reward_total_values: list[float] = []
    component_values = {name: [] for name in COMPONENT_INFO_KEYS}
    completed_task_values: list[int] = []
    completed_dag_values: list[int] = []
    active_dag_values: list[int] = []
    ready_task_values: list[int] = []
    pending_task_values: list[int] = []
    task_energy_values: list[float] = []
    queue_values: list[float] = []
    slot_records: list[dict[str, Any]] = []
    latest_info: dict[str, Any] = {}
    reconstruction_max_error = 0.0

    for slot_index in range(int(slots)):
        with common_random.scoped_environment_calls(slot_index + 1):
            context = env.prepare_slot_state()
            frozen_ready_task_ids = [
                str(value) for value in context["frozen_ready_task_ids"]
            ]
            ready_task_values.append(len(frozen_ready_task_ids))
            pending_task_values.append(_pending_task_count(env))
            env.apply_movement({})
            assignments, skipped = _select_assignments(
                env=env,
                frozen_ready_task_ids=frozen_ready_task_ids,
                policy=policy,
                policy_rng=policy_rng,
            )
            _, _, _, latest_info = env.commit_and_advance(
                assignment_buffer=assignments,
                offloading_skip_count=skipped,
            )

        reward_total = float(latest_info["step_reward"])
        reward_total_values.append(reward_total)
        slot_component_sum = 0.0
        for component, info_key in COMPONENT_INFO_KEYS.items():
            value = float(latest_info[info_key])
            component_values[component].append(value)
            slot_component_sum += value
        reconstruction_max_error = max(
            reconstruction_max_error, abs(reward_total - slot_component_sum)
        )
        completed_task_values.append(int(latest_info["completed_tasks"]))
        completed_dag_values.append(int(latest_info["completed_dags"]))
        active_dag_values.append(int(latest_info["active_dags"]))
        task_energy_values.append(float(latest_info["step_task_energy"]))
        queue_values.extend(float(value) for value in _queue_lengths(env))
        slot_records.append(
            {
                "slot": int(slot_index + 1),
                "reward_total": reward_total,
                **{
                    component: float(component_values[component][-1])
                    for component in COMPONENT_INFO_KEYS
                },
                "completed_tasks": int(latest_info["completed_tasks"]),
                "completed_dags": int(latest_info["completed_dags"]),
                "active_dags": int(latest_info["active_dags"]),
                "ready_tasks": int(ready_task_values[-1]),
                "pending_tasks": int(pending_task_values[-1]),
                "task_energy": float(latest_info["step_task_energy"]),
            }
        )

    summaries, cancellation_ratio, component_timing = _component_summary(
        component_values, int(slots)
    )
    completed = int(round(float(latest_info["completed_dag_count"])))
    admitted = int(latest_info["arrival_admitted_count"])
    generated = int(round(float(latest_info["generated_dag_count"])))
    episode_reward = float(latest_info["episode_reward"])
    total_task_energy = float(latest_info["total_task_energy"])
    row = {
        "seed": int(seed),
        "policy": str(policy),
        "slots": int(slots),
        "slot_records": slot_records,
        "components": summaries,
        "component_cumulative_at_slots": component_timing,
        "reward_total_cumulative_at_slots": {
            str(checkpoint): float(
                np.sum(np.asarray(reward_total_values[:checkpoint], dtype=np.float64))
            )
            for checkpoint in CHECKPOINT_SLOTS
            if checkpoint <= int(slots)
        },
        "cancellation_ratio": cancellation_ratio,
        "component_reconstruction_max_abs_error": reconstruction_max_error,
        "completed_dag_count": completed,
        "arrival_admitted_count": admitted,
        "generated_dag_count": generated,
        "completion_rate": float(latest_info["dag_completion_rate"]),
        "avg_dag_flowtime": float(latest_info["average_dag_flowtime"]),
        "avg_critical_path_delay": float(
            latest_info["average_critical_path_task_completion_delay"]
        ),
        "avg_uav_queue": float(np.mean(queue_values)) if queue_values else 0.0,
        "total_task_energy": total_task_energy,
        "task_energy_per_completed_dag": float(total_task_energy / max(completed, 1)),
        "episode_reward": episode_reward,
        "reward_per_completed_dag": float(episode_reward / max(completed, 1)),
        "reward_per_admitted_dag": float(episode_reward / max(admitted, 1)),
        "episode_end_active_dag_count": int(latest_info["active_dags"]),
        "slot_series_summary": {
            "completed_tasks_total": int(sum(completed_task_values)),
            "completed_dags_total": int(sum(completed_dag_values)),
            "active_dags_mean": float(np.mean(active_dag_values)),
            "ready_tasks_mean": float(np.mean(ready_task_values)),
            "pending_tasks_mean": float(np.mean(pending_task_values)),
            "task_energy_sum": float(np.sum(task_energy_values)),
        },
    }
    return row, common_random.audit_snapshot(), root_rng_state


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    left_rank = _average_ranks(np.asarray(left, dtype=np.float64))
    right_rank = _average_ranks(np.asarray(right, dtype=np.float64))
    if float(np.std(left_rank)) == 0.0 or float(np.std(right_rank)) == 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _alignment_by_seed(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for seed in SEEDS:
        rows = [row for row in runs if int(row["seed"]) == int(seed)]
        reward_values = [float(row["episode_reward"]) for row in rows]
        reward_ranking = [
            str(row["policy"])
            for row in sorted(rows, key=lambda item: (-float(item["episode_reward"]), str(item["policy"])))
        ]
        metrics: dict[str, Any] = {}
        for metric, (direction, key) in ALIGNMENT_METRICS.items():
            raw_values = [float(row[key]) for row in rows]
            better_values = raw_values if direction == "higher" else [-value for value in raw_values]
            metrics[metric] = {
                "direction": direction,
                "ranking_best_to_worst": [
                    str(row["policy"])
                    for row in sorted(
                        rows,
                        key=lambda item: (
                            -float(item[key]) if direction == "higher" else float(item[key]),
                            str(item["policy"]),
                        ),
                    )
                ],
                "spearman_reward_vs_better_metric": _spearman(
                    reward_values, better_values
                ),
            }
        output.append(
            {
                "seed": int(seed),
                "reward_ranking_best_to_worst": reward_ranking,
                "metrics": metrics,
            }
        )
    return output


def _pareto_contradictions(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contradictions: list[dict[str, Any]] = []
    for seed in SEEDS:
        rows = [row for row in runs if int(row["seed"]) == int(seed)]
        for candidate in rows:
            for baseline in rows:
                if candidate is baseline:
                    continue
                no_worse = (
                    int(candidate["completed_dag_count"])
                    >= int(baseline["completed_dag_count"])
                    and float(candidate["avg_dag_flowtime"])
                    <= float(baseline["avg_dag_flowtime"])
                    and float(candidate["task_energy_per_completed_dag"])
                    <= float(baseline["task_energy_per_completed_dag"])
                )
                strictly_better = (
                    int(candidate["completed_dag_count"])
                    > int(baseline["completed_dag_count"])
                    or float(candidate["avg_dag_flowtime"])
                    < float(baseline["avg_dag_flowtime"])
                    or float(candidate["task_energy_per_completed_dag"])
                    < float(baseline["task_energy_per_completed_dag"])
                )
                reward_lower = float(candidate["episode_reward"]) < float(
                    baseline["episode_reward"]
                )
                if no_worse and strictly_better and reward_lower:
                    contradictions.append(
                        {
                            "seed": int(seed),
                            "system_better_policy": str(candidate["policy"]),
                            "reward_higher_policy": str(baseline["policy"]),
                            "system_better_values": {
                                "completed_dags": candidate["completed_dag_count"],
                                "avg_flowtime": candidate["avg_dag_flowtime"],
                                "task_energy_per_completed_dag": candidate[
                                    "task_energy_per_completed_dag"
                                ],
                                "episode_reward": candidate["episode_reward"],
                            },
                            "baseline_values": {
                                "completed_dags": baseline["completed_dag_count"],
                                "avg_flowtime": baseline["avg_dag_flowtime"],
                                "task_energy_per_completed_dag": baseline[
                                    "task_energy_per_completed_dag"
                                ],
                                "episode_reward": baseline["episode_reward"],
                            },
                        }
                    )
    return contradictions


def run_formal(output_path: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    seed_audits: list[dict[str, Any]] = []
    for seed in SEEDS:
        policy_audits: list[dict[str, Any]] = []
        root_states: list[Any] = []
        for policy in POLICIES:
            print(f"Running R1-B seed={seed} policy={policy} slots={FORMAL_SLOTS}", flush=True)
            row, audit, root_state = run_one(
                seed=seed, policy=policy, slots=FORMAL_SLOTS
            )
            runs.append(row)
            policy_audits.append(audit)
            root_states.append(root_state)
        if not all(
            clean_host_rng_states_equal(root_states[0], state)
            for state in root_states[1:]
        ):
            raise AssertionError(f"seed={seed}: CRN root states differ")
        audit = audit_clean_semantic_common_random(policy_audits)
        audit_row = {
            "seed": int(seed),
            "shared_semantic_keys_checked": int(audit.shared_semantic_keys_checked),
            "semantic_key_mismatches": len(audit.semantic_key_mismatches),
            "unrecognized_environment_calls": int(audit.unrecognized_environment_calls),
        }
        seed_audits.append(audit_row)
        if audit_row["semantic_key_mismatches"] or audit_row["unrecognized_environment_calls"]:
            raise AssertionError(f"seed={seed}: strict CRN audit failed")

    payload = {
        "schema": "r1b_reward_alignment_v1",
        "server_code_commit": _git_head(),
        "protocol": {
            "seeds": list(SEEDS),
            "policies": list(POLICIES),
            "slots_per_run": FORMAL_SLOTS,
            "movement": "forced_hover",
            "common_random": "Scheme-B2 semantic CRN",
            "optimizer_updates": 0,
            "cancellation_ratio": "1 - abs(sum(component cumulative sums)) / sum(abs(component cumulative sum))",
            "pareto_criteria": "no worse in completed DAGs, avg flowtime, and task energy per completed DAG; strict in at least one",
        },
        "crn_audit": {
            "by_seed": seed_audits,
            "totals": {
                "shared_semantic_keys_checked": sum(
                    row["shared_semantic_keys_checked"] for row in seed_audits
                ),
                "semantic_key_mismatches": sum(
                    row["semantic_key_mismatches"] for row in seed_audits
                ),
                "unrecognized_environment_calls": sum(
                    row["unrecognized_environment_calls"] for row in seed_audits
                ),
            },
        },
        "runs": runs,
        "alignment_by_seed": _alignment_by_seed(runs),
        "pareto_contradictions": _pareto_contradictions(runs),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {output_path}", flush=True)
    return payload


def run_benchmark() -> dict[str, Any]:
    started = time.perf_counter()
    row, audit, _ = run_one(
        seed=42, policy="nearest_legal", slots=BENCHMARK_SLOTS
    )
    elapsed = float(time.perf_counter() - started)
    unrecognized = int(audit["unrecognized_environment_calls"])
    if unrecognized != 0:
        raise AssertionError("benchmark observed unrecognized environment RNG call")
    return {
        "seed": 42,
        "policy": "nearest_legal",
        "slots": BENCHMARK_SLOTS,
        "wall_clock_seconds": elapsed,
        "unrecognized_environment_calls": unrecognized,
        "component_reconstruction_max_abs_error": row[
            "component_reconstruction_max_abs_error"
        ],
        "completed_dag_count": row["completed_dag_count"],
    }


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.benchmark:
        print("BENCHMARK_JSON " + json.dumps(run_benchmark(), sort_keys=True), flush=True)
        return 0
    run_formal(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
