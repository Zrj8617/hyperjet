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
from marl_models.mappo.clean_counterfactual_oracle_rng import (
    capture_clean_host_rng_state,
    restore_clean_host_rng_state,
)
from marl_models.mappo.clean_counterfactual_oracle_rng_trace import (
    localize_clean_rng_call_divergence,
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
    parser = argparse.ArgumentParser(description="Phase4 Scheme-B2 RNG divergence localization.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--completion-reference", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-search-slots", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs") / "phase4_scheme_b2_rng_divergence_localization.json",
    )
    return parser


def run_localization(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if bool(config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES):
        raise ValueError("Scheme-B2 requires KaHyPar OFF")
    reference = json.loads(args.completion_reference.read_text(encoding="utf-8"))
    expected = [
        (
            str(row["task"]),
            int(row["decision_order"]),
            tuple(int(value) for value in row["candidate_uavs"]),
            int(row["baseline_uav"]),
            int(row["first_rng_divergence_slot"]),
        )
        for row in reference["decisions"]
    ]
    if len(expected) != 5:
        raise ValueError("completion reference must contain exactly five decisions")

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    payload = _load_trusted_checkpoint(torch, args.checkpoint)
    controls = checkpoint_experiment_controls(payload)
    if str(controls.get("task_encoder", "hgnn")) != "mlp":
        raise ValueError("Scheme-B2 requires an MLP checkpoint")
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
                ready_tasks = [task for task in ready_tasks if task is not None and task.is_ready]
                assignments = modules.offloading_actor.act(
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
                if identity != expected[len(collected)][:4]:
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
                localized = localize_clean_rng_call_divergence(
                    env_snapshot=env_snapshot,
                    baseline_trace=trace,
                    target_decision_order=target.decision_order,
                    initial_rng_state=snapshot_rng,
                    task_encoder=modules.hgnn,
                    movement_actor=modules.movement_actor,
                    offloading_actor=modules.offloading_actor,
                    max_future_slots=expected[len(collected)][4],
                    device="cpu",
                )
                if localized["first_divergence_slot"] != expected[len(collected)][4]:
                    raise AssertionError("localized slot does not match completion-scale result")
                collected.append(localized)
                restore_clean_host_rng_state(snapshot_rng)
            restore_clean_host_rng_state(snapshot_rng)
            _, _, done, _ = env.commit_and_advance(assignment_buffer=assignments)
            if done:
                break
    finally:
        graph_builder.close()

    if len(collected) != 5:
        raise RuntimeError("failed to localize all five referenced decisions")
    report = {
        "protocol": {
            "checkpoint": str(args.checkpoint),
            "completion_reference": str(args.completion_reference),
            "task_encoder": "mlp",
            "seed": int(args.seed),
            "kahypar_enabled": False,
            "optimizer_updates": 0,
            "instrumentation": "temporary wrappers around environment Python/NumPy RNG draws",
        },
        "decisions": collected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    print(json.dumps(run_localization(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
