from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.env import Env
from environment.graph_builder import CleanGraphBuilder
from marl_models.mappo.clean_counterfactual_oracle import (
    capture_clean_counterfactual_baseline_trace,
    clone_post_movement_pre_offloading_env,
    run_clean_counterfactual_branches_serial,
)
from marl_models.mappo.clean_counterfactual_oracle_rng import (
    capture_clean_host_rng_state,
    clean_continuation_results_equal,
    first_clean_rng_divergence_slot,
    restore_clean_host_rng_state,
    run_clean_counterfactual_continuations_process,
    run_clean_counterfactual_continuations_serial,
)
from marl_models.mappo.clean_slot_orchestrator import (
    encode_prepared_slot,
    prepare_slot_state,
)
from scripts.eval_clean_mainline import (
    _build_modules,
    _load_module_state,
    _load_trusted_checkpoint,
    _module_dims_from_checkpoint,
    _set_eval_mode,
)
from scripts.train_clean_mainline import checkpoint_experiment_controls


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase4 Scheme-B1 same-start RNG-divergence feasibility probe."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--oracle-horizon-slots",
        type=int,
        nargs="+",
        choices=(1, 5, 10, 20),
        default=(1, 5, 10),
    )
    parser.add_argument("--max-target-decisions", type=int, default=5)
    parser.add_argument("--max-search-slots", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs") / "phase4_scheme_b1_rng_feasibility.json",
    )
    return parser


def run_feasibility(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not 1 <= int(args.max_target_decisions) <= 5:
        raise ValueError("--max-target-decisions must be between 1 and 5")
    if bool(config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES):
        raise ValueError("Scheme-B1 requires KaHyPar OFF")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    payload = _load_trusted_checkpoint(torch, args.checkpoint)
    controls = checkpoint_experiment_controls(payload)
    if str(controls.get("task_encoder", "hgnn")) != "mlp":
        raise ValueError("Scheme-B1 feasibility requires an MLP checkpoint")
    cli = payload.get("config", {}).get("cli", {})
    gamma = float(cli.get("gamma", 0.99))
    dims = _module_dims_from_checkpoint(
        payload,
        argparse.Namespace(task_embedding_dim=None, hidden_dim=None),
    )
    modules = _build_modules(dims=dims, experiment_controls=controls, device="cpu")
    _load_module_state(modules, payload)
    _set_eval_mode(modules)

    env = Env(
        completed_dag_weight=float(controls["completed_dag_weight"]),
        freeze_ue_mobility=bool(controls.get("freeze_ue_mobility", False)),
    )
    env.reset()
    graph_builder = CleanGraphBuilder()
    graph_builder.reset()
    decisions: list[dict[str, Any]] = []
    try:
        for _ in range(int(args.max_search_slots)):
            if len(decisions) >= int(args.max_target_decisions):
                break
            with torch.no_grad():
                prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
                encoded = encode_prepared_slot(
                    prepared_state=prepared,
                    env=env,
                    hgnn=modules.hgnn,
                    movement_actor=modules.movement_actor,
                    device="cpu",
                )
                selected_movement = torch.argmax(encoded.movement_logits, dim=-1)
                movement_actions = {
                    int(uav_id): int(selected_movement[index].detach().cpu().item())
                    for index, uav_id in enumerate(encoded.movement_observation.uav_ids)
                }
                env.apply_movement(movement_actions)
                env_snapshot = clone_post_movement_pre_offloading_env(env)
                snapshot_rng = capture_clean_host_rng_state()
                ready_tasks = [
                    env.task_manager.get_task(task_id)
                    for task_id in prepared.frozen_ready_task_ids
                ]
                ready_tasks = [
                    task for task in ready_tasks if task is not None and task.is_ready
                ]
                assignment_buffer = modules.offloading_actor.act(
                    frozen_ready_tasks=ready_tasks,
                    task_embeddings=encoded.task_embeddings.detach(),
                    graph_snapshot=prepared.graph_snapshot,
                    task_manager=env.task_manager,
                    uavs=env.uavs,
                    executor=env.executor,
                    current_time_seconds=env.current_time_seconds,
                    uav_service_positions=env.uav_service_positions,
                    ue_service_positions=env.ue_service_positions,
                    ues=env.ues,
                    deterministic=True,
                )
            baseline_trace = capture_clean_counterfactual_baseline_trace(
                decision_records=modules.offloading_actor.latest_records,
                assignment_buffer=assignment_buffer,
            )
            for target in baseline_trace:
                if len(decisions) >= int(args.max_target_decisions):
                    break
                if len(target.legal_uav_ids) < 2:
                    continue
                restore_clean_host_rng_state(snapshot_rng)
                feasibility = run_clean_counterfactual_branches_serial(
                    env_snapshot=env_snapshot,
                    baseline_trace=baseline_trace,
                    target_decision_order=target.decision_order,
                )
                restore_clean_host_rng_state(snapshot_rng)
                if not all(result.suffix_replay_feasible for result in feasibility):
                    continue
                decisions.append(
                    _run_target_decision(
                        env_snapshot=env_snapshot,
                        baseline_trace=baseline_trace,
                        target=target,
                        snapshot_rng=snapshot_rng,
                        horizons=tuple(dict.fromkeys(int(v) for v in args.oracle_horizon_slots)),
                        gamma=gamma,
                        modules=modules,
                    )
                )
                restore_clean_host_rng_state(snapshot_rng)

            restore_clean_host_rng_state(snapshot_rng)
            _, _, done, _ = env.commit_and_advance(
                assignment_buffer=assignment_buffer
            )
            if done:
                break
    finally:
        graph_builder.close()

    if not decisions:
        raise RuntimeError("no eligible multi-candidate decision was found")
    report = {
        "protocol": {
            "task_encoder": "mlp",
            "kahypar_enabled": False,
            "checkpoint": str(args.checkpoint),
            "seed": int(args.seed),
            "gamma": gamma,
            "policy": "frozen checkpoint; movement masked argmax; offloading deterministic=True",
            "rng_start": "identical full random.getstate()/numpy.random.get_state()",
            "rng_comparison": "exact full-state equality; no random tape",
            "optimizer_updates": 0,
        },
        "horizons": {
            str(horizon): _summarize_horizon(decisions, horizon)
            for horizon in tuple(dict.fromkeys(int(v) for v in args.oracle_horizon_slots))
        },
        "decisions": decisions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _run_target_decision(
    *,
    env_snapshot: Any,
    baseline_trace: tuple[Any, ...],
    target: Any,
    snapshot_rng: Any,
    horizons: tuple[int, ...],
    gamma: float,
    modules: Any,
) -> dict[str, Any]:
    task = env_snapshot.task_manager.get_task(target.task_id)
    horizon_rows: dict[str, Any] = {}
    for horizon in horizons:
        serial = run_clean_counterfactual_continuations_serial(
            env_snapshot=env_snapshot,
            baseline_trace=baseline_trace,
            target_decision_order=target.decision_order,
            initial_rng_state=snapshot_rng,
            horizon_slots=horizon,
            gamma=gamma,
            task_encoder=modules.hgnn,
            movement_actor=modules.movement_actor,
            offloading_actor=modules.offloading_actor,
            device="cpu",
        )
        parallel = run_clean_counterfactual_continuations_process(
            env_snapshot=env_snapshot,
            baseline_trace=baseline_trace,
            target_decision_order=target.decision_order,
            initial_rng_state=snapshot_rng,
            horizon_slots=horizon,
            gamma=gamma,
            task_encoder=modules.hgnn,
            movement_actor=modules.movement_actor,
            offloading_actor=modules.offloading_actor,
            device="cpu",
        )
        consistent = len(serial) == len(parallel) and all(
            clean_continuation_results_equal(left, right)
            for left, right in zip(serial, parallel)
        )
        if not consistent:
            raise AssertionError("Scheme-B1 serial/parallel results differ")
        feasible = [row for row in serial if row.branch.suffix_replay_feasible]
        baseline = next(
            row for row in feasible if row.branch.forced_uav_id == target.selected_uav_id
        )
        branch_rows = []
        for row in feasible:
            branch_rows.append(
                {
                    "uav_id": row.branch.forced_uav_id,
                    "reward_sequence": list(row.reward_sequence),
                    "discounted_return": row.discounted_return,
                    "target_dag_completed": row.target_dag_completed,
                    "target_dag_completion_time": row.target_dag_completion_time,
                    "completed_dag_count": row.completed_dag_count,
                    "rng_divergence_slot_vs_baseline": first_clean_rng_divergence_slot(
                        baseline, row
                    ),
                }
            )
        returns = [float(row.discounted_return) for row in feasible]
        best_index = int(np.argmax(np.asarray(returns, dtype=np.float64)))
        horizon_rows[str(horizon)] = {
            "branches": branch_rows,
            "best_uav": feasible[best_index].branch.forced_uav_id,
            "return_spread": max(returns) - min(returns),
            "action_ranking_present": len(set(returns)) > 1,
            "serial_parallel_consistent": True,
            "pair_divergence_slots": [
                first_clean_rng_divergence_slot(left, right)
                for left, right in combinations(feasible, 2)
            ],
        }
    return {
        "task": target.task_id,
        "target_dag_id": None if task is None else str(task.dag_id),
        "decision_order": target.decision_order,
        "candidate_uavs": list(target.legal_uav_ids),
        "baseline_uav": target.selected_uav_id,
        "horizons": horizon_rows,
    }


def _summarize_horizon(decisions: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    rows = [decision["horizons"][str(horizon)] for decision in decisions]
    spreads = np.asarray([row["return_spread"] for row in rows], dtype=np.float64)
    divergences = [value for row in rows for value in row["pair_divergence_slots"]]
    distribution = Counter("aligned" if value is None else str(value) for value in divergences)
    branch_count = sum(len(row["branches"]) for row in rows)
    aligned = sum(value is None for value in divergences)
    return {
        "target_decision_count": len(rows),
        "branch_count": branch_count,
        "rng_aligned_pair_fraction": (
            float(aligned / len(divergences)) if divergences else 1.0
        ),
        "first_rng_divergence_slot_distribution": dict(sorted(distribution.items())),
        "return_spread_mean": float(spreads.mean()),
        "return_spread_median": float(np.median(spreads)),
        "return_spread_max": float(spreads.max()),
        "action_ranking_present": any(row["action_ranking_present"] for row in rows),
        "serial_parallel_consistent": all(
            row["serial_parallel_consistent"] for row in rows
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run_feasibility(args)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
