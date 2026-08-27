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
from marl_models.mappo.clean_counterfactual_oracle_common_random import (
    run_clean_common_random_completion_process,
    run_clean_common_random_completion_serial,
)
from marl_models.mappo.clean_counterfactual_oracle_rng import (
    capture_clean_host_rng_state,
    restore_clean_host_rng_state,
)
from marl_models.mappo.clean_slot_orchestrator import encode_prepared_slot, prepare_slot_state
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
        description="Phase4 Scheme-B2 strict semantic common-random completion probe."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--completion-reference", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-future-slots", type=int, default=100)
    parser.add_argument("--max-search-slots", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs") / "phase4_scheme_b2_common_random_completion.json",
    )
    return parser


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if int(args.max_future_slots) != 100:
        raise ValueError("Scheme-B2 fixes the safety cap at 100 future slots")
    if bool(config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES):
        raise ValueError("Scheme-B2 requires KaHyPar OFF")
    reference = json.loads(args.completion_reference.read_text(encoding="utf-8"))
    expected = [
        (
            str(row["task"]),
            int(row["decision_order"]),
            tuple(int(value) for value in row["candidate_uavs"]),
            int(row["baseline_uav"]),
        )
        for row in reference["decisions"]
    ]
    if len(expected) != 5:
        raise ValueError("completion reference must contain exactly five target decisions")

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    payload = _load_trusted_checkpoint(torch, args.checkpoint)
    controls = checkpoint_experiment_controls(payload)
    if str(controls.get("task_encoder", "hgnn")) != "mlp":
        raise ValueError("Scheme-B2 requires an MLP checkpoint")
    gamma = float(payload.get("config", {}).get("cli", {}).get("gamma", 0.99))
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
    builder = CleanGraphBuilder()
    builder.reset()
    collected: list[dict[str, Any]] = []
    try:
        for _ in range(int(args.max_search_slots)):
            if len(collected) == len(expected):
                break
            with torch.no_grad():
                prepared = prepare_slot_state(env=env, graph_builder=builder)
                encoded = encode_prepared_slot(
                    prepared_state=prepared,
                    env=env,
                    hgnn=modules.hgnn,
                    movement_actor=modules.movement_actor,
                    device="cpu",
                )
                movement = torch.argmax(encoded.movement_logits, dim=-1)
                env.apply_movement(
                    {
                        int(uav_id): int(movement[index].detach().cpu().item())
                        for index, uav_id in enumerate(encoded.movement_observation.uav_ids)
                    }
                )
                snapshot = clone_post_movement_pre_offloading_env(env)
                root_rng = capture_clean_host_rng_state()
                ready = [env.task_manager.get_task(task_id) for task_id in prepared.frozen_ready_task_ids]
                ready = [task for task in ready if task is not None and task.is_ready]
                assignments = modules.offloading_actor.act(
                    frozen_ready_tasks=ready,
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
                assignment_buffer=assignments,
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
                restore_clean_host_rng_state(root_rng)
                feasibility = run_clean_counterfactual_branches_serial(
                    env_snapshot=snapshot,
                    baseline_trace=trace,
                    target_decision_order=target.decision_order,
                )
                restore_clean_host_rng_state(root_rng)
                if not all(row.suffix_replay_feasible for row in feasibility):
                    raise AssertionError("referenced target is no longer strict-suffix feasible")
                kwargs = {
                    "env_snapshot": snapshot,
                    "baseline_trace": trace,
                    "target_decision_order": target.decision_order,
                    "initial_rng_state": root_rng,
                    "gamma": gamma,
                    "task_encoder": modules.hgnn,
                    "movement_actor": modules.movement_actor,
                    "offloading_actor": modules.offloading_actor,
                    "max_future_slots": 100,
                    "device": "cpu",
                }
                serial = run_clean_common_random_completion_serial(**kwargs)
                parallel = run_clean_common_random_completion_process(**kwargs)
                if serial.decision != parallel.decision or serial.audit != parallel.audit:
                    raise AssertionError("Scheme-B2 serial/spawn result or audit differs")
                if serial.audit.semantic_key_mismatches:
                    raise AssertionError("Scheme-B2 semantic shared-key mismatch")
                if serial.audit.unrecognized_environment_calls:
                    raise AssertionError("Scheme-B2 saw unrecognized environment RNG calls")
                collected.append(
                    _result_row(
                        serial,
                        gamma=gamma,
                        historical=reference["decisions"][len(collected)],
                    )
                )
                restore_clean_host_rng_state(root_rng)
            restore_clean_host_rng_state(root_rng)
            _, _, done, _ = env.commit_and_advance(assignment_buffer=assignments)
            if done:
                break
    finally:
        builder.close()

    if len(collected) != 5:
        raise RuntimeError("failed to recover the exact five reference decisions")
    report = {
        "protocol": {
            "checkpoint": str(args.checkpoint),
            "completion_reference": str(args.completion_reference),
            "task_encoder": "mlp",
            "seed": int(args.seed),
            "gamma": gamma,
            "kahypar_enabled": False,
            "optimizer_updates": 0,
            "max_future_slots_safety_cap": 100,
            "root_derivation": "NumPy SeedSequence over full MT19937 state words and semantic key",
        },
        "decisions": collected,
        "shared_semantic_keys_checked": sum(
            int(row["shared_semantic_keys_checked"]) for row in collected
        ),
        "semantic_key_mismatches": sum(
            int(row["semantic_key_mismatches"]) for row in collected
        ),
        "serial_parallel_consistent_count": sum(
            int(row["serial_parallel_consistent"]) for row in collected
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _result_row(result: Any, *, gamma: float, historical: dict[str, Any]) -> dict[str, Any]:
    decision = result.decision
    branches = []
    for branch in decision.branches:
        g20 = (
            float(sum((gamma ** index) * value for index, value in enumerate(branch.reward_sequence[:20])))
            if len(branch.reward_sequence) >= 20
            else None
        )
        branches.append(
            {
                "uav_id": branch.forced_uav_id,
                "target_dag_completion_horizon": branch.target_dag_completion_horizon,
                "target_dag_completion_time": branch.target_dag_completion_time,
                "reward_sequence": list(branch.reward_sequence),
                "g20_crn": g20,
                "g_common": branch.common_discounted_return,
            }
        )
    completion_values = {
        row["uav_id"]: row["target_dag_completion_horizon"]
        for row in branches
        if row["target_dag_completion_horizon"] is not None
    }
    g20_values = {row["uav_id"]: row["g20_crn"] for row in branches if row["g20_crn"] is not None}
    common_values = {
        row["uav_id"]: row["g_common"] for row in branches if row["g_common"] is not None
    }
    best_completion = _best_ids(completion_values, maximize=False)
    best_g20 = _best_ids(g20_values, maximize=True)
    best_common = _best_ids(common_values, maximize=True)
    ids = [row["uav_id"] for row in branches]
    return {
        "task": decision.target_task_id,
        "decision_order": decision.decision_order,
        "baseline_uav": decision.baseline_uav_id,
        "candidate_uavs": list(decision.candidate_uav_ids),
        "branches": branches,
        "common_completion_horizon": decision.common_completion_horizon,
        "stop_reason": decision.stop_reason,
        "best_uav_by_completion_time": best_completion,
        "best_uav_by_g20_crn": best_g20,
        "best_uav_by_g_common": best_common,
        "g20_vs_completion_spearman": _spearman(
            [g20_values.get(uav_id) for uav_id in ids],
            [
                -float(completion_values[uav_id]) if uav_id in completion_values else None
                for uav_id in ids
            ],
        ),
        "g20_vs_g_common_spearman": _spearman(
            [g20_values.get(uav_id) for uav_id in ids],
            [common_values.get(uav_id) for uav_id in ids],
        ),
        "historical_h20_best_uav": list(historical["h20_best_uav"]),
        "shared_semantic_keys_checked": result.audit.shared_semantic_keys_checked,
        "semantic_key_mismatches": len(result.audit.semantic_key_mismatches),
        "unrecognized_environment_calls": result.audit.unrecognized_environment_calls,
        "serial_parallel_consistent": True,
    }


def _best_ids(values: dict[int, float | int], *, maximize: bool) -> list[int]:
    if not values:
        return []
    best = (max if maximize else min)(values.values())
    return [key for key, value in values.items() if value == best]


def _spearman(left: list[float | None], right: list[float | None]) -> float | None:
    pairs = [(x, y) for x, y in zip(left, right) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    x_rank = _average_ranks(x)
    y_rank = _average_ranks(y)
    if float(x_rank.std()) == 0.0 or float(y_rank.std()) == 0.0:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


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


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    print(json.dumps(run_probe(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
