from __future__ import annotations

import argparse
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
from marl_models.mappo.clean_counterfactual_oracle_completion import (
    clean_completion_scale_results_equal,
    run_clean_completion_scale_process,
    run_clean_completion_scale_serial,
)
from marl_models.mappo.clean_counterfactual_oracle_rng import (
    capture_clean_host_rng_state,
    restore_clean_host_rng_state,
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
        description="Phase4 Scheme-B completion-scale paired feasibility probe."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--h20-reference", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-future-slots", type=int, default=100)
    parser.add_argument("--max-search-slots", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs") / "phase4_scheme_b_completion_scale.json",
    )
    return parser


def run_completion_scale(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if int(args.max_future_slots) != 100:
        raise ValueError("this feasibility protocol fixes the safety cap at 100 slots")
    if bool(config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES):
        raise ValueError("completion-scale feasibility requires KaHyPar OFF")
    if not args.checkpoint.is_file() or not args.h20_reference.is_file():
        raise FileNotFoundError("checkpoint and H20 reference must both exist")
    h20_report = json.loads(args.h20_reference.read_text(encoding="utf-8"))
    expected = [
        (
            str(row["task"]),
            int(row["decision_order"]),
            tuple(int(value) for value in row["candidate_uavs"]),
            int(row["baseline_uav"]),
        )
        for row in h20_report["decisions"]
    ]
    if len(expected) != 5:
        raise ValueError("H20 reference must contain exactly five target decisions")

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    payload = _load_trusted_checkpoint(torch, args.checkpoint)
    controls = checkpoint_experiment_controls(payload)
    if str(controls.get("task_encoder", "hgnn")) != "mlp":
        raise ValueError("completion-scale feasibility requires an MLP checkpoint")
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
    collected: list[dict[str, Any]] = []
    try:
        for _ in range(int(args.max_search_slots)):
            if len(collected) == len(expected):
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
            trace = capture_clean_counterfactual_baseline_trace(
                decision_records=modules.offloading_actor.latest_records,
                assignment_buffer=assignment_buffer,
            )
            for target in trace:
                if len(collected) == len(expected):
                    break
                identity = (
                    target.task_id,
                    target.decision_order,
                    target.legal_uav_ids,
                    target.selected_uav_id,
                )
                if identity != expected[len(collected)]:
                    continue
                restore_clean_host_rng_state(snapshot_rng)
                feasibility = run_clean_counterfactual_branches_serial(
                    env_snapshot=env_snapshot,
                    baseline_trace=trace,
                    target_decision_order=target.decision_order,
                )
                restore_clean_host_rng_state(snapshot_rng)
                if not all(row.suffix_replay_feasible for row in feasibility):
                    raise AssertionError("referenced target is no longer suffix-feasible")
                serial = run_clean_completion_scale_serial(
                    env_snapshot=env_snapshot,
                    baseline_trace=trace,
                    target_decision_order=target.decision_order,
                    initial_rng_state=snapshot_rng,
                    gamma=gamma,
                    task_encoder=modules.hgnn,
                    movement_actor=modules.movement_actor,
                    offloading_actor=modules.offloading_actor,
                    max_future_slots=100,
                    device="cpu",
                )
                parallel = run_clean_completion_scale_process(
                    env_snapshot=env_snapshot,
                    baseline_trace=trace,
                    target_decision_order=target.decision_order,
                    initial_rng_state=snapshot_rng,
                    gamma=gamma,
                    task_encoder=modules.hgnn,
                    movement_actor=modules.movement_actor,
                    offloading_actor=modules.offloading_actor,
                    max_future_slots=100,
                    device="cpu",
                )
                if not clean_completion_scale_results_equal(serial, parallel):
                    raise AssertionError("completion-scale serial/parallel results differ")
                collected.append(_result_row(serial, h20_report["decisions"][len(collected)]))
                restore_clean_host_rng_state(snapshot_rng)
            restore_clean_host_rng_state(snapshot_rng)
            _, _, done, _ = env.commit_and_advance(assignment_buffer=assignment_buffer)
            if done:
                break
    finally:
        graph_builder.close()

    if len(collected) != len(expected):
        raise RuntimeError("failed to recover the exact five H20 target decisions")
    report = {
        "protocol": {
            "checkpoint": str(args.checkpoint),
            "h20_reference": str(args.h20_reference),
            "task_encoder": "mlp",
            "seed": int(args.seed),
            "gamma": gamma,
            "kahypar_enabled": False,
            "max_future_slots_safety_cap": 100,
            "optimizer_updates": 0,
            "policy": "frozen deterministic movement/offloading",
            "rng_comparison": "exact full Python/NumPy state equality",
        },
        "decisions": collected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _result_row(result: Any, h20_reference: dict[str, Any]) -> dict[str, Any]:
    branches = []
    for branch in result.branches:
        branches.append(
            {
                "uav_id": branch.forced_uav_id,
                "target_dag_completion_horizon": branch.target_dag_completion_horizon,
                "target_dag_completion_time": branch.target_dag_completion_time,
                "reward_sequence": list(branch.reward_sequence),
                "g_common": branch.common_discounted_return,
            }
        )
    completed_horizons = [
        branch.target_dag_completion_horizon
        for branch in result.branches
        if branch.target_dag_completion_horizon is not None
    ]
    earliest = min(completed_horizons) if completed_horizons else None
    latest = max(completed_horizons) if len(completed_horizons) == len(result.branches) else None
    completion_spread = latest - earliest if latest is not None and earliest is not None else None
    best_completion = (
        [
            branch.forced_uav_id
            for branch in result.branches
            if branch.target_dag_completion_horizon == earliest
        ]
        if earliest is not None
        else []
    )
    common_returns = {
        branch.forced_uav_id: branch.common_discounted_return
        for branch in result.branches
        if branch.common_discounted_return is not None
    }
    best_return = max(common_returns.values()) if common_returns else None
    best_g_common = (
        [uav_id for uav_id, value in common_returns.items() if value == best_return]
        if best_return is not None
        else []
    )
    h20_returns = {
        int(branch["uav_id"]): float(branch["discounted_return"])
        for branch in h20_reference["horizons"]["20"]["branches"]
    }
    h20_max = max(h20_returns.values())
    h20_best = [uav_id for uav_id, value in h20_returns.items() if value == h20_max]
    return {
        "task": result.target_task_id,
        "decision_order": result.decision_order,
        "baseline_uav": result.baseline_uav_id,
        "candidate_uavs": list(result.candidate_uav_ids),
        "branches": branches,
        "earliest_completion": earliest,
        "latest_completion": latest,
        "completion_spread": completion_spread,
        "common_completion_horizon": result.common_completion_horizon,
        "best_uav_by_completion_time": best_completion,
        "best_uav_by_g_common": best_g_common,
        "h20_best_uav": h20_best,
        "h20_best_matches_completion_best": bool(set(h20_best) & set(best_completion)),
        "first_rng_divergence_slot": result.first_rng_divergence_slot,
        "stop_reason": result.stop_reason,
        "serial_parallel_consistent": True,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run_completion_scale(args)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
