from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.env import Env
from environment.graph_builder import CleanGraphBuilder
from marl_models.hgnn import build_clean_task_encoder
from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
from marl_models.mappo.clean_slot_orchestrator import prepare_slot_state
from scripts.train_decision_ppo_bandit_gate import CHECKPOINT_SCHEMA


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Checkpoint-only Stage 1 closed-loop eval.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps-per-episode", type=int, default=200)
    parser.add_argument("--eval-seed", type=int, default=424242)
    parser.add_argument("--modes", nargs="+", choices=["stochastic", "deterministic"], default=["stochastic", "deterministic"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows, metadata = evaluate_checkpoint(args)
    summary = {
        "schema": "decision_ppo_bandit_stage1_closed_loop_v1",
        "technical_pass": True,
        "metadata": metadata,
        "rows": rows,
        "summary": _summarize(rows),
        "pairing_limitation": (
            "same eval seeds are not strict counterfactual pairs because active_dag_cap "
            "makes eligibility and RNG consumption policy-dependent"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary["summary"], ensure_ascii=False, sort_keys=True))
    return 0


def evaluate_checkpoint(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    try:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(args.checkpoint, map_location="cpu")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported checkpoint schema: {payload.get('schema')!r}")
    config = payload.get("config")
    if not isinstance(config, dict) or str(config.get("encoder")) != "mlp":
        raise ValueError("checkpoint lacks a valid Stage 1 MLP config")
    device = torch.device(str(args.device))
    _set_seed(int(args.eval_seed), torch)
    probe_env = Env(
        completed_dag_weight=16.0,
        freeze_ue_mobility=False,
    )
    probe_builder = CleanGraphBuilder()
    probe_env.reset()
    probe_builder.reset()
    probe = prepare_slot_state(env=probe_env, graph_builder=probe_builder)
    task_feature_dim = int(probe.graph_snapshot.task_features.shape[1])
    probe_builder.close()
    encoder = build_clean_task_encoder(
        encoder_type="mlp",
        task_feature_dim=task_feature_dim,
        hidden_dim=int(config["hidden_dim"]),
        output_dim=int(config["task_embedding_dim"]),
    ).to(device)
    actor = CleanOffloadingActor(
        task_embedding_dim=int(config["task_embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    actor.scorer.load_state_dict(payload["scorer_state_dict"], strict=True)
    encoder.eval()
    actor.eval()

    rows: list[dict[str, Any]] = []
    for mode in args.modes:
        deterministic = str(mode) == "deterministic"
        for episode in range(int(args.episodes)):
            scenario_seed = int(args.eval_seed) + int(episode)
            _set_seed(scenario_seed, torch)
            env = Env(completed_dag_weight=16.0, freeze_ue_mobility=False)
            builder = CleanGraphBuilder()
            env.reset()
            builder.reset()
            reward_total = 0.0
            latest_info: dict[str, Any] = {}
            try:
                for _slot in range(int(args.max_steps_per_episode)):
                    prepared = prepare_slot_state(env=env, graph_builder=builder)
                    env.apply_movement({})
                    task_features = torch.as_tensor(
                        np.array(
                            prepared.graph_snapshot.task_features,
                            dtype=np.float32,
                            copy=True,
                        ),
                        dtype=torch.float32,
                        device=device,
                    )
                    with torch.no_grad():
                        embeddings = encoder(task_features)
                    ready_tasks = [
                        env.task_manager.get_task(task_id)
                        for task_id in prepared.frozen_ready_task_ids
                    ]
                    ready_tasks = [task for task in ready_tasks if task is not None]
                    assignments = actor.act(
                        frozen_ready_tasks=ready_tasks,
                        task_embeddings=embeddings,
                        graph_snapshot=prepared.graph_snapshot,
                        task_manager=env.task_manager,
                        uavs=env.uavs,
                        executor=env.executor,
                        current_time_seconds=env.current_time_seconds,
                        uav_service_positions=env.uav_service_positions,
                        ue_service_positions=env.ue_service_positions,
                        ues=env.ues,
                        deterministic=deterministic,
                    )
                    _, _, done, latest_info = env.commit_and_advance(
                        assignment_buffer=assignments,
                        offloading_skip_count=len(actor.latest_skip_events),
                    )
                    reward_total += float(latest_info["step_reward"])
                    if done:
                        break
            finally:
                builder.close()
            rows.append(
                {
                    "mode": str(mode),
                    "episode": int(episode),
                    "scenario_seed": scenario_seed,
                    "episode_reward_total": float(reward_total),
                    **_metric_subset(latest_info),
                }
            )
    return rows, {
        "checkpoint": str(args.checkpoint),
        "training_group": str(payload["group"]),
        "training_seed": int(payload["seed"]),
        "completed_update": int(payload["completed_update"]),
        "eval_seed": int(args.eval_seed),
        "episodes": int(args.episodes),
    }


def _metric_subset(info: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "completed_dag_count",
        "generated_dag_count",
        "dag_completion_rate",
        "average_dag_flowtime",
        "avg_uav_queue_length",
        "arrival_attempt_count",
        "arrival_admitted_count",
        "arrival_blocked_count",
        "arrival_blocked_reasons",
    )
    return {key: info.get(key) for key in keys}


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    metrics = (
        "episode_reward_total",
        "completed_dag_count",
        "generated_dag_count",
        "dag_completion_rate",
        "average_dag_flowtime",
        "avg_uav_queue_length",
    )
    for mode in sorted({str(row["mode"]) for row in rows}):
        selected = [row for row in rows if str(row["mode"]) == mode]
        summary[mode] = {}
        for metric in metrics:
            values = [
                float(row[metric])
                for row in selected
                if row.get(metric) is not None
            ]
            summary[mode][metric] = {
                "mean": None if not values else float(np.mean(values)),
                "std": None if not values else float(np.std(values)),
                "count": len(values),
            }
    return summary


def _set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
