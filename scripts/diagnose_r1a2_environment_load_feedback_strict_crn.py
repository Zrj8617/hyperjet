"""Stage R1-A2 strict semantic common-random load-feedback control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
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
    SLOTS,
    _git_head,
    _main_row,
    _mean,
    _paired_deltas,
    _pending_task_count,
    _queue_lengths,
    _select_assignments,
    _set_seed,
)


OUTPUT_PATH = ROOT / "logs" / "r1a2_environment_load_feedback_strict_crn.json"
NATIVE_R1A_PATH = ROOT / "logs" / "r1a_environment_load_feedback.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage R1-A2 strict semantic CRN control.")
    parser.add_argument("--native-r1a", type=Path, default=NATIVE_R1A_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser


def run_one_strict_crn(
    *, seed: int, policy: str
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    _set_seed(seed)
    policy_rng = np.random.default_rng(int(seed) + 1_000_003)
    env = Env()
    env.reset()
    root_rng_state = capture_clean_host_rng_state()
    common_random = CleanSemanticCommonRandom(root_rng_state)

    active_dag_samples: list[float] = []
    ready_task_samples: list[float] = []
    pending_task_samples: list[float] = []
    queue_samples: list[float] = []
    executed_assignments = 0
    invalid_assignments = 0
    latest_info: dict[str, Any] = {}

    for slot_index in range(SLOTS):
        with common_random.scoped_environment_calls(slot_index + 1):
            context = env.prepare_slot_state()
            frozen_ready_task_ids = [
                str(value) for value in context["frozen_ready_task_ids"]
            ]
            ready_task_samples.append(float(len(frozen_ready_task_ids)))
            pending_task_samples.append(float(_pending_task_count(env)))

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
            executed_assignments += int(latest_info["newly_assigned_tasks"])
            invalid_assignments += int(latest_info["invalid_assignments"])
            active_dag_samples.append(float(latest_info["active_dags"]))
            queue_samples.extend(float(value) for value in _queue_lengths(env))

    arrival_attempt_count = int(latest_info["arrival_attempt_count"])
    arrival_draw_count = int(latest_info["arrival_draw_count"])
    arrival_admitted_count = int(latest_info["arrival_admitted_count"])
    arrival_blocked_count = int(latest_info["arrival_blocked_count"])
    generated_dag_count = int(round(float(latest_info["generated_dag_count"])))
    completed_dag_count = int(round(float(latest_info["completed_dag_count"])))
    blocked_fraction = float(arrival_blocked_count / max(arrival_attempt_count, 1))

    row = {
        "seed": int(seed),
        "policy": str(policy),
        "slots": SLOTS,
        "arrival_attempt_count": arrival_attempt_count,
        "arrival_draw_count": arrival_draw_count,
        "arrival_admitted_count": arrival_admitted_count,
        "arrival_blocked_count": arrival_blocked_count,
        "active_dag_cap_blocked_count": int(
            latest_info.get("arrival_blocked_reasons", {}).get("active_dag_cap", 0)
        ),
        "generated_dag_count": generated_dag_count,
        "completed_dag_count": completed_dag_count,
        "completion_rate": float(latest_info["dag_completion_rate"]),
        "throughput": float(latest_info["dag_throughput"]),
        "avg_dag_flowtime": float(latest_info["average_dag_flowtime"]),
        "avg_critical_path_delay": float(
            latest_info["average_critical_path_task_completion_delay"]
        ),
        "mean_active_dag": _mean(active_dag_samples),
        "max_active_dag": max(active_dag_samples, default=0.0),
        "mean_ready_tasks": _mean(ready_task_samples),
        "mean_pending_tasks": _mean(pending_task_samples),
        "avg_uav_queue": _mean(queue_samples),
        "max_uav_queue": max(queue_samples, default=0.0),
        "executed_assignment_count": int(executed_assignments),
        "invalid_assignment_count": int(invalid_assignments),
        "task_energy": float(latest_info["total_task_energy"]),
        "movement_energy": float(latest_info["uav_movement_energy_total"]),
        "episode_reward": float(latest_info["episode_reward"]),
        "blocked_fraction": blocked_fraction,
        "net_active_growth": int(generated_dag_count - completed_dag_count),
        "admitted_per_completed": float(
            arrival_admitted_count / max(completed_dag_count, 1)
        ),
    }
    return row, common_random.audit_snapshot(), root_rng_state


def _native_vs_crn(
    native_paired: list[dict[str, Any]], crn_paired: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    native = {
        (int(row["seed"]), str(row["comparison"])): row for row in native_paired
    }
    rows: list[dict[str, Any]] = []
    for crn in crn_paired:
        key = (int(crn["seed"]), str(crn["comparison"]))
        baseline = native[key]
        rows.append(
            {
                "seed": key[0],
                "comparison": key[1],
                "native_R1A_delta_admitted": int(baseline["admitted_delta"]),
                "CRN_R1A2_delta_admitted": int(crn["admitted_delta"]),
                "native_delta_completed": int(baseline["completed_delta"]),
                "CRN_delta_completed": int(crn["completed_delta"]),
            }
        )
    return rows


def _print_table(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    print(" | ".join(columns))
    for row in rows:
        print(" | ".join(str(row[column]) for column in columns))


def main() -> int:
    args = build_arg_parser().parse_args()
    native = json.loads(args.native_r1a.resolve().read_text(encoding="utf-8"))
    if native.get("schema") != "r1a_environment_load_feedback_v1":
        raise ValueError("native R1-A JSON schema mismatch")

    runs: list[dict[str, Any]] = []
    seed_audits: list[dict[str, Any]] = []
    for seed in SEEDS:
        policy_audits: list[dict[str, Any]] = []
        root_states: list[Any] = []
        for policy in POLICIES:
            print(f"Running strict CRN seed={seed} policy={policy} slots={SLOTS}", flush=True)
            row, audit, root_state = run_one_strict_crn(seed=seed, policy=policy)
            runs.append(row)
            policy_audits.append(audit)
            root_states.append(root_state)

        if not all(
            clean_host_rng_states_equal(root_states[0], state)
            for state in root_states[1:]
        ):
            raise AssertionError(f"seed={seed}: policy CRN root states differ")
        audit = audit_clean_semantic_common_random(policy_audits)
        seed_row = {
            "seed": int(seed),
            "shared_semantic_keys_checked": int(audit.shared_semantic_keys_checked),
            "semantic_key_mismatches": len(audit.semantic_key_mismatches),
            "unrecognized_environment_calls": int(audit.unrecognized_environment_calls),
        }
        seed_audits.append(seed_row)
        if seed_row["semantic_key_mismatches"] != 0:
            raise AssertionError(f"seed={seed}: semantic CRN mismatch")
        if seed_row["unrecognized_environment_calls"] != 0:
            raise AssertionError(f"seed={seed}: unrecognized environment RNG call")
        print(f"CRN audit seed={seed}: {seed_row}", flush=True)

    main_table = [_main_row(run) for run in runs]
    paired = _paired_deltas(runs)
    comparison = _native_vs_crn(native["paired_deltas"], paired)
    totals = {
        "shared_semantic_keys_checked": sum(
            int(row["shared_semantic_keys_checked"]) for row in seed_audits
        ),
        "semantic_key_mismatches": sum(
            int(row["semantic_key_mismatches"]) for row in seed_audits
        ),
        "unrecognized_environment_calls": sum(
            int(row["unrecognized_environment_calls"]) for row in seed_audits
        ),
    }
    payload = {
        "schema": "r1a2_environment_load_feedback_strict_crn_v1",
        "server_code_commit": _git_head(),
        "protocol": {
            "seeds": list(SEEDS),
            "policies": list(POLICIES),
            "episodes_per_cell": 1,
            "slots_per_episode": SLOTS,
            "movement": "forced_hover",
            "ready_task_order": "environment_frozen_order",
            "reservation": "sequential_immediate",
            "semantic_key": "(slot, ue_id, subsystem)",
            "semantic_subsystems": ["mobility", "arrival", "dag_generation"],
            "common_random_implementation": "Scheme-B2 CleanSemanticCommonRandom",
            "nearest_legal": "min euclidean distance(task.source_pos, candidate_uav_service_position), tie min uav_id",
            "blocked_fraction": "arrival_blocked_count / max(arrival_attempt_count, 1)",
        },
        "crn_audit": {"by_seed": seed_audits, "totals": totals},
        "runs": runs,
        "main_table": main_table,
        "paired_deltas": paired,
        "native_r1a_vs_crn_r1a2": comparison,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    _print_table(
        main_table,
        (
            "seed",
            "policy",
            "completed",
            "admitted",
            "generated",
            "blocked_fraction",
            "completion_rate",
            "avg_flowtime",
            "avg_queue",
            "episode_reward",
        ),
    )
    print(json.dumps(paired, indent=2), flush=True)
    print(json.dumps(comparison, indent=2), flush=True)
    print(json.dumps(totals, sort_keys=True), flush=True)
    print(f"Wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
