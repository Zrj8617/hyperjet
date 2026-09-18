from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.env import Env
from environment.exogenous_tape import ExogenousTapeReplay, read_tape
from environment.graph_builder import CleanGraphBuilder
from scripts.eval_clean_mainline import (
    _build_modules,
    _checkpoint_model_seed,
    _load_module_state,
    _load_trusted_checkpoint,
    _module_dims_from_checkpoint,
    _require_torch,
    _run_eval_episode,
    _set_eval_mode,
    _set_seed,
)
from scripts.train_clean_mainline import checkpoint_experiment_controls


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one frozen checkpoint on fixed exogenous tapes.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tape-dir", type=Path, required=True)
    parser.add_argument("--tape-ids", type=str, required=True)
    parser.add_argument("--protocol", choices=("forced_hover", "joint"), required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", type=str, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--checkpoint-label", type=str, required=True)
    parser.add_argument("--energy-lambda", type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    tape_ids = _parse_ids(args.tape_ids)
    torch = _require_torch()
    device = torch.device(str(args.device))
    checkpoint_payload = _load_trusted_checkpoint(torch, args.checkpoint)
    controls = checkpoint_experiment_controls(checkpoint_payload)
    enable_kahypar = bool(controls["enable_kahypar"])
    config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES = enable_kahypar
    dims_args = argparse.Namespace(task_embedding_dim=None, hidden_dim=None)
    modules = _build_modules(
        dims=_module_dims_from_checkpoint(checkpoint_payload, dims_args),
        experiment_controls=controls,
        device=device,
    )
    _load_module_state(modules, checkpoint_payload)
    _set_eval_mode(modules)
    checkpoint_seed = _checkpoint_model_seed(checkpoint_payload)
    if checkpoint_seed is not None and int(checkpoint_seed) != int(args.model_seed):
        raise ValueError("checkpoint model seed does not match requested model seed")
    graph_builder = CleanGraphBuilder()
    rows: list[dict[str, Any]] = []
    try:
        for episode, tape_id in enumerate(tape_ids):
            tape_path = args.tape_dir / f"tape_{tape_id:04d}.json.gz"
            tape = ExogenousTapeReplay(read_tape(tape_path))
            _set_seed(int(tape_id), torch=torch)
            env = Env(
                completed_dag_weight=float(controls["completed_dag_weight"]),
                freeze_ue_mobility=False,
                exogenous_tape=tape,
            )
            env.reset()
            graph_builder.reset()
            policy_row = _run_eval_episode(
                env=env,
                graph_builder=graph_builder,
                modules=modules,
                device=device,
                arrival_steps=int(tape.payload["slots"]),
                max_drain_steps=0,
                episode=episode,
                environment_seed=int(tape_id),
                offloading_policy="actor_argmax",
                checkpoint_path=str(args.checkpoint),
                checkpoint_model_seed=checkpoint_seed,
                completed_dag_weight=float(controls["completed_dag_weight"]),
                freeze_movement=args.protocol == "forced_hover",
                detach_critic_hgnn=bool(controls["detach_critic_hgnn"]),
                offloading_counterfactual_coef=float(controls["offloading_counterfactual_coef"]),
                offloading_action_value_loss_coef=float(controls["offloading_action_value_loss_coef"]),
                offloading_lagged_q_coef=float(controls["offloading_lagged_q_coef"]),
                offloading_lagged_q_loss_coef=float(controls["offloading_lagged_q_loss_coef"]),
            )
            policy_row.pop("last_info", None)
            policy_row.pop("_dag_flowtime_samples", None)
            policy_row.pop("_offloading_decisions", None)
            hover_action_ratio = policy_row.get("hover_action_ratio")
            if hover_action_ratio is None:
                raise ValueError("evaluation did not report hover_action_ratio")
            if args.protocol == "forced_hover" and float(hover_action_ratio) != 1.0:
                raise AssertionError("forced_hover protocol must have hover_action_ratio=1")
            rows.append(
                {
                    "arm": str(args.arm),
                    "model_seed": int(args.model_seed),
                    "checkpoint_label": str(args.checkpoint_label),
                    "protocol": str(args.protocol),
                    "tape_id": int(tape_id),
                    "tape_path": str(tape_path),
                    **_unified_cost(
                        env=env,
                        tape=tape,
                        energy_lambda=float(args.energy_lambda),
                    ),
                    "hover_action_ratio": float(hover_action_ratio),
                    "policy_metrics": policy_row,
                }
            )
    finally:
        graph_builder.close()
    status_counts: dict[str, int] = {}
    degraded_count = 0
    type3_slot_count = 0
    invalid_disabled_count = 0
    for row in rows:
        metrics = dict(row.get("policy_metrics", {}))
        for status, count in dict(metrics.get("kahypar_partition_status_counts", {})).items():
            status_counts[str(status)] = status_counts.get(str(status), 0) + int(count)
        degraded_count += int(metrics.get("kahypar_degraded_slot_count", 0))
        type3_slot_count += int(metrics.get("kahypar_partition_nonzero_slot_count", 0))
        invalid_disabled_count += int(metrics.get("kahypar_invalid_disabled_count", 0))
    kahypar_health = {
        "enabled_from_checkpoint": enable_kahypar,
        "partition_status_counts": dict(sorted(status_counts.items())),
        "degraded_count": int(degraded_count),
        "type3_slot_count": int(type3_slot_count),
        "invalid_disabled_count": int(invalid_disabled_count),
        "circuit_open": bool(graph_builder.kahypar_circuit_open),
    }
    kahypar_health["pass"] = bool(
        not enable_kahypar
        or (
            int(degraded_count) == 0
            and int(type3_slot_count) > 0
            and int(invalid_disabled_count) == 0
            and not bool(kahypar_health["circuit_open"])
        )
    )
    result = {
        "schema": "six_arm_fixed_tape_evaluation_v2",
        "status": "completed" if bool(kahypar_health["pass"]) else "failed_kahypar",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": str(args.arm),
        "model_seed": int(args.model_seed),
        "checkpoint": str(args.checkpoint),
        "checkpoint_label": str(args.checkpoint_label),
        "protocol": str(args.protocol),
        "task_encoder": str(controls["task_encoder"]),
        "enable_kahypar": enable_kahypar,
        "kahypar_health": kahypar_health,
        "tape_ids": tape_ids,
        "rows": rows,
        "actual_parameters": controls,
        "evaluation_energy_lambda_task": float(args.energy_lambda),
        "evaluation_energy_lambda_move": float(args.energy_lambda),
        "version": _version_record(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if not bool(kahypar_health["pass"]):
        raise RuntimeError(f"KaHyPar evaluation health gate failed: {kahypar_health}")
    print(json.dumps({"status": "completed", "output": str(args.output), "row_count": len(rows)}, sort_keys=True))


def _unified_cost(
    *, env: Env, tape: ExogenousTapeReplay, energy_lambda: float = 1.0
) -> dict[str, Any]:
    end_seconds = float(tape.payload["slots"]) * float(tape.payload["time_slot_duration"])
    delay_total = 0.0
    completed_ids: list[str] = []
    for offer in tape.offers:
        offer_id = str(offer["offer_id"])
        arrival = float(offer["job"]["arrival_time"])
        job = env.task_manager.get_job(offer_id)
        if job is not None and bool(job.completed) and job.return_complete_time is not None:
            completion = min(float(job.return_complete_time), end_seconds)
            delay_total += max(completion - arrival, 0.0)
            completed_ids.append(offer_id)
        else:
            delay_total += max(end_seconds - arrival, 0.0)
    offer_count = int(len(tape.offers))
    admitted_count = int(len(tape.admitted_offer_ids))
    completed_count = int(len(completed_ids))
    task_energy = float(env.metrics.metrics.total_task_energy)
    move_energy = float(env.metrics.metrics.uav_movement_energy_total)
    delay_component = delay_total / 500.0
    task_energy_lambda = float(energy_lambda)
    move_energy_lambda = float(energy_lambda)
    task_component = task_energy_lambda * task_energy / 500.0
    move_component = move_energy_lambda * move_energy / 500.0
    j_episode = delay_component + task_component + move_component
    denominator = float(max(offer_count, 1))
    completed_flowtimes = [
        float(env.task_manager.get_job(offer_id).return_complete_time - env.task_manager.get_job(offer_id).arrival_time)
        for offer_id in completed_ids
    ]
    values = {
        "N_offer": offer_count,
        "N_admitted": admitted_count,
        "N_rejected": int(offer_count - admitted_count),
        "N_completed": completed_count,
        "admission_rate": admitted_count / denominator,
        "conditional_completion_rate": completed_count / float(max(admitted_count, 1)),
        "end_to_end_completion_rate": completed_count / denominator,
        "delay_seconds_total": delay_total,
        "task_energy_joules_total": task_energy,
        "move_energy_joules_total": move_energy,
        "lambda_task_seconds_per_joule": task_energy_lambda,
        "lambda_move_seconds_per_joule": move_energy_lambda,
        "J_episode": j_episode,
        "J_per_offer": j_episode / denominator,
        "J_delay_component": delay_component,
        "J_task_energy_component": task_component,
        "J_move_energy_component": move_component,
        "J_per_offer_delay_component": delay_component / denominator,
        "J_per_offer_task_energy_component": task_component / denominator,
        "J_per_offer_move_energy_component": move_component / denominator,
        "throughput": completed_count / max(end_seconds, float(config.TIME_SLOT_DURATION)),
        "energy_per_completed_DAG": (task_energy + move_energy) / float(max(completed_count, 1)),
        "completed_DAG_flowtime_mean": None if not completed_flowtimes else float(np.mean(completed_flowtimes)),
        "completed_DAG_flowtime_median": None if not completed_flowtimes else float(np.median(completed_flowtimes)),
        "offer_ids": [str(offer["offer_id"]) for offer in tape.offers],
        "admitted_offer_ids": sorted(tape.admitted_offer_ids),
        "completed_offer_ids": sorted(completed_ids),
    }
    for key, value in values.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite unified metric: {key}")
    return values


def _parse_ids(spec: str) -> list[int]:
    values: list[int] = []
    for part in str(spec).split(","):
        token = part.strip()
        if "-" in token:
            start, end = (int(value) for value in token.split("-", 1))
            values.extend(range(start, end + 1))
        elif token:
            values.append(int(token))
    if not values or len(values) != len(set(values)):
        raise ValueError("tape IDs must be non-empty and unique")
    return values


def _version_record() -> dict[str, Any]:
    def run(*command: str) -> str:
        return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    return {
        "head": run("git", "rev-parse", "HEAD"),
        "dirty": run("git", "status", "--porcelain").splitlines(),
        "script_git_object": run("git", "hash-object", str(Path(__file__).resolve())),
        "tape_module_git_object": run("git", "hash-object", str(ROOT / "environment" / "exogenous_tape.py")),
        "env_git_object": run("git", "hash-object", str(ROOT / "environment" / "env.py")),
    }


if __name__ == "__main__":
    main()
