"""Deterministic fixed-hover R2 checkpoint evaluation under Scheme-B2 CRN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from environment.env import Env  # noqa: E402
from environment.graph_builder import CleanGraphBuilder  # noqa: E402
from marl_models.mappo.clean_counterfactual_oracle_common_random import (  # noqa: E402
    CleanSemanticCommonRandom,
)
from marl_models.mappo.clean_counterfactual_oracle_rng import (  # noqa: E402
    capture_clean_host_rng_state,
)
from marl_models.mappo.clean_slot_orchestrator import (  # noqa: E402
    encode_prepared_slot,
    prepare_slot_state,
)
from scripts.diagnose_r1a_environment_load_feedback import (  # noqa: E402
    _pending_task_count,
    _queue_lengths,
)
from scripts.eval_clean_mainline import (  # noqa: E402
    _build_modules,
    _load_module_state,
    _load_trusted_checkpoint,
    _module_dims_from_checkpoint,
    _set_eval_mode,
)
from scripts.train_clean_mainline import checkpoint_experiment_controls  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--eval-seed", type=int, required=True)
    parser.add_argument("--slots", type=int, default=500)
    parser.add_argument("--completed-dag-weight", type=float, required=True)
    parser.add_argument("--energy-weight", type=float, required=True)
    parser.add_argument(
        "--position-shaping",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _set_seed(seed: int, torch: Any) -> None:
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    config.REWARD_ENERGY_WEIGHT = float(args.energy_weight)
    config.ENABLE_MOVEMENT_POSITION_SHAPING = bool(args.position_shaping)
    _set_seed(int(args.eval_seed), torch)
    device = torch.device(str(args.device))
    checkpoint = _load_trusted_checkpoint(torch, args.checkpoint.resolve())
    controls = checkpoint_experiment_controls(checkpoint)
    if str(controls["task_encoder"]) != "mlp" or not bool(controls["freeze_movement"]):
        raise ValueError("R2 evaluation requires an MLP fixed-movement checkpoint")
    if abs(float(controls["completed_dag_weight"]) - float(args.completed_dag_weight)) > 1e-12:
        raise ValueError("checkpoint completed-DAG weight does not match the R2 arm")
    dims = _module_dims_from_checkpoint(
        checkpoint,
        SimpleNamespace(task_embedding_dim=None, hidden_dim=None),
    )
    modules = _build_modules(dims=dims, experiment_controls=controls, device=device)
    _load_module_state(modules, checkpoint)
    _set_eval_mode(modules)

    env = Env(completed_dag_weight=float(args.completed_dag_weight))
    graph_builder = CleanGraphBuilder()
    env.reset()
    root_rng_state = capture_clean_host_rng_state()
    common_random = CleanSemanticCommonRandom(root_rng_state)
    ready_samples: list[float] = []
    pending_samples: list[float] = []
    queue_samples: list[float] = []
    latest_info: dict[str, Any] = {}
    try:
        for slot in range(1, int(args.slots) + 1):
            with common_random.scoped_environment_calls(slot):
                prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
                with torch.no_grad():
                    encoded = encode_prepared_slot(
                        prepared_state=prepared,
                        env=env,
                        hgnn=modules.hgnn,
                        critic=modules.critic,
                        movement_actor=modules.movement_actor,
                        device=device,
                        detach_critic_hgnn=False,
                    )
                ready_tasks = [
                    env.task_manager.get_task(task_id)
                    for task_id in prepared.frozen_ready_task_ids
                ]
                ready_tasks = [task for task in ready_tasks if task is not None and task.is_ready]
                ready_samples.append(float(len(ready_tasks)))
                pending_samples.append(float(_pending_task_count(env)))
                env.apply_movement({})
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
                _, _, _, latest_info = env.commit_and_advance(
                    assignment_buffer=assignments,
                    offloading_skip_count=len(modules.offloading_actor.latest_skip_events),
                )
                queue_samples.extend(float(value) for value in _queue_lengths(env))
    finally:
        graph_builder.close()

    completed = int(round(float(latest_info["completed_dag_count"])))
    generated = int(round(float(latest_info["generated_dag_count"])))
    admitted = int(latest_info["arrival_admitted_count"])
    result = {
        "schema": "r2_reward_checkpoint_strict_crn_eval_v1",
        "arm": str(args.arm),
        "training_seed": int(args.training_seed),
        "eval_seed": int(args.eval_seed),
        "slots": int(args.slots),
        "checkpoint": str(args.checkpoint.resolve()),
        "reward_override": {
            "completed_dag_weight": float(args.completed_dag_weight),
            "energy_weight": float(args.energy_weight),
            "movement_position_shaping": bool(args.position_shaping),
        },
        "metrics": {
            "completed_dag_count": completed,
            "arrival_admitted_count": admitted,
            "generated_dag_count": generated,
            "completion_rate": float(completed / max(generated, 1)),
            "avg_dag_flowtime": float(latest_info["average_dag_flowtime"]),
            "avg_critical_path_delay": float(
                latest_info["average_critical_path_task_completion_delay"]
            ),
            "avg_uav_queue": float(np.mean(queue_samples)) if queue_samples else 0.0,
            "task_energy": float(latest_info["total_task_energy"]),
            "task_energy_per_completed_dag": float(
                float(latest_info["total_task_energy"]) / max(completed, 1)
            ),
            "episode_reward": float(latest_info["episode_reward"]),
            "mean_ready_tasks": float(np.mean(ready_samples)) if ready_samples else 0.0,
            "mean_pending_tasks": float(np.mean(pending_samples)) if pending_samples else 0.0,
        },
        "semantic_audit_snapshot": common_random.audit_snapshot(),
    }
    return result


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    compact = dict(result)
    compact.pop("semantic_audit_snapshot", None)
    print(json.dumps(compact, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
