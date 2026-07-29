from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_contextual_bandit_gate as bandit
from scripts import train_greedy_imitation_gate as gate


SCHEMA = "contextual_bandit_sequential_closed_loop_v1"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Checkpoint-only paired closed-loop evaluation for the contextual-bandit "
            "credit diagnostic. It never trains or replays frozen decisions."
        )
    )
    parser.add_argument("--bandit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[42, 86, 1042])
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--eval-seed", type=int, default=2_000_042)
    parser.add_argument("--max-steps-per-episode", type=int, default=200)
    parser.add_argument("--completed-dag-weight", type=float, default=16.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--smoke", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.smoke:
        args.episodes = 1
        args.max_steps_per_episode = 1
        args.device = "cpu"
    _validate_args(args)
    bandit_dir = Path(args.bandit_dir).resolve()
    summary = _load_bandit_summary(bandit_dir, args.training_seeds)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else bandit_dir / "sequential_closed_loop"
    )
    config = _eval_config(args, bandit_dir, summary)
    _initialize_output(output_dir, config)

    rows: list[dict[str, Any]] = []
    for policy in ("masked_random", "greedy_eft"):
        rows.extend(
            _evaluate_policy(
                args=args,
                policy=policy,
                modules=None,
                training_seed=None,
            )
        )
        _write_outputs(output_dir, config, rows)
    for training_seed in args.training_seeds:
        for stage in ("initial", "trained"):
            checkpoint_path = (
                bandit_dir / f"seed_{int(training_seed)}" / f"{stage}_checkpoint.pt"
            )
            modules, identity = load_bandit_checkpoint(
                checkpoint_path,
                expected_seed=int(training_seed),
                expected_stage=stage,
                expected_dataset_checksum=str(summary["dataset_checksum"]),
                device=str(args.device),
            )
            with modules.torch.no_grad():
                policy_rows = _evaluate_policy(
                    args=args,
                    policy=stage,
                    modules=modules,
                    training_seed=int(training_seed),
                )
            for row in policy_rows:
                row["checkpoint_sha256"] = identity["checkpoint_sha256"]
            rows.extend(policy_rows)
            _write_outputs(output_dir, config, rows)
    _write_outputs(output_dir, config, rows, completed=True)
    return 0


def load_bandit_checkpoint(
    checkpoint_path: Path,
    *,
    expected_seed: int,
    expected_stage: str,
    expected_dataset_checksum: str,
    device: str,
) -> tuple[gate.GateModules, dict[str, Any]]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is required for closed-loop evaluation") from exc
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"bandit checkpoint is missing: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    expected = {
        "schema": bandit.CHECKPOINT_SCHEMA,
        "stage": str(expected_stage),
        "encoder": "mlp",
        "training_seed": int(expected_seed),
        "dataset_checksum": str(expected_dataset_checksum),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"checkpoint {key} mismatch: expected {value!r}, got {payload.get(key)!r}"
            )
    module_args = argparse.Namespace(
        seed=int(expected_seed),
        task_encoder="mlp",
        task_feature_dim=int(payload["task_feature_dim"]),
        task_embedding_dim=int(payload["task_embedding_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        lr=3e-4,
        gradient_batch_decisions=64,
        max_grad_norm=0.5,
        completed_dag_weight=16.0,
        device=str(device),
    )
    modules = gate._build_modules(
        module_args,
        encoder_seed=int(expected_seed),
        scorer_seed=0,
    )
    modules.task_encoder.load_state_dict(payload["task_encoder_state_dict"], strict=True)
    modules.offloading_actor.scorer.load_state_dict(
        payload["candidate_scorer_state_dict"], strict=True
    )
    modules.task_encoder.eval()
    modules.offloading_actor.scorer.eval()
    return modules, {
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "stage": str(expected_stage),
        "training_seed": int(expected_seed),
    }


def _evaluate_policy(
    *,
    args: argparse.Namespace,
    policy: str,
    modules: gate.GateModules | None,
    training_seed: int | None,
) -> list[dict[str, Any]]:
    env, graph_builder = gate._new_seeded_env(args, seed=int(args.eval_seed))
    rows: list[dict[str, Any]] = []
    try:
        for episode in range(int(args.episodes)):
            scenario_seed = int(args.eval_seed) + int(episode)
            gate._set_seed(scenario_seed)
            env.reset()
            graph_builder.reset()
            episode_reward = 0.0
            accepted = 0
            skipped = 0
            latest_info: dict[str, Any] = {}
            for slot in range(int(args.max_steps_per_episode)):
                prepared = gate.prepare_slot_state(env=env, graph_builder=graph_builder)
                env.apply_movement({})
                ready_tasks = [
                    env.task_manager.get_task(task_id)
                    for task_id in prepared.frozen_ready_task_ids
                ]
                ready_tasks = [
                    task for task in ready_tasks if task is not None and task.is_ready
                ]
                if policy in {"masked_random", "greedy_eft"}:
                    assignments, skipped_slot = gate._select_static_assignments(
                        policy=("random_hash" if policy == "masked_random" else "greedy_eft"),
                        frozen_ready_tasks=ready_tasks,
                        prepared=prepared,
                        env=env,
                        environment_seed=int(args.eval_seed),
                        episode=int(episode),
                        slot=int(slot),
                    )
                else:
                    if modules is None:
                        raise ValueError(f"{policy} policy requires loaded modules")
                    assignments, skipped_slot = gate._select_imitation_assignments(
                        modules=modules,
                        frozen_ready_tasks=ready_tasks,
                        prepared=prepared,
                        env=env,
                    )
                _, _, done, latest_info = env.commit_and_advance(
                    assignment_buffer=assignments,
                    offloading_skip_count=skipped_slot,
                )
                episode_reward += float(latest_info["step_reward"])
                accepted += int(latest_info["newly_assigned_tasks"])
                skipped += int(skipped_slot)
                if done:
                    break
            rows.append(
                {
                    "policy": str(policy),
                    "training_seed": training_seed,
                    "eval_seed": int(args.eval_seed),
                    "scenario_seed": int(scenario_seed),
                    "episode": int(episode),
                    "episode_reward_total": float(episode_reward),
                    "accepted_assignments": int(accepted),
                    "offloading_skipped_no_candidate": int(skipped),
                    **gate._episode_metric_subset(latest_info),
                    **_arrival_metric_subset(latest_info),
                }
            )
    finally:
        graph_builder.close()
    return rows


def _load_bandit_summary(bandit_dir: Path, seeds: list[int]) -> dict[str, Any]:
    path = bandit_dir / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"bandit summary is missing: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("schema") != bandit.SCHEMA or summary.get("status") != "completed":
        raise ValueError("bandit summary is not a completed formal result")
    if summary.get("technical_pass") is not True:
        raise ValueError("bandit formal gate did not technically pass")
    if int(summary.get("epochs", -1)) != 5:
        raise ValueError("closed-loop requires the fixed five-epoch formal gate")
    if [int(value) for value in summary.get("training_seeds", [])] != [
        int(value) for value in seeds
    ]:
        raise ValueError("requested training seeds differ from the formal summary")
    return summary


def _eval_config(
    args: argparse.Namespace,
    bandit_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": SCHEMA,
        "bandit_dir": str(bandit_dir),
        "dataset_checksum": str(summary["dataset_checksum"]),
        "training_seeds": [int(value) for value in args.training_seeds],
        "episodes": int(args.episodes),
        "eval_seed": int(args.eval_seed),
        "max_steps_per_episode": int(args.max_steps_per_episode),
        "movement": "forced_fixed_empty_movement_action",
        "policies": ["masked_random", "greedy_eft", "initial", "trained"],
        "real_online_sequential_reservation": True,
        "uses_frozen_decision_transitions": False,
    }
    payload["config_hash"] = _json_hash(payload)
    return payload


def _initialize_output(output_dir: Path, config: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("config_hash") != config["config_hash"]:
            raise ValueError("closed-loop output directory has a different config")
    else:
        _write_json(path, config)


def _write_outputs(
    output_dir: Path,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    completed: bool = False,
) -> None:
    lines = "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows
    )
    _write_text(output_dir / "episode_rows.jsonl", lines)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row["policy"])
        if row["training_seed"] is not None:
            key += f":seed_{int(row['training_seed'])}"
        grouped.setdefault(key, []).append(row)
    aggregate = {key: _summarize(value) for key, value in sorted(grouped.items())}
    expected_groups = 2 + 2 * len(config["training_seeds"])
    _write_json(
        output_dir / "summary.json",
        {
            "schema": SCHEMA,
            "status": (
                "completed"
                if completed and len(aggregate) == expected_groups
                else "in_progress"
            ),
            "config_hash": config["config_hash"],
            "group_count": len(aggregate),
            "expected_group_count": expected_groups,
            "aggregate": aggregate,
            "finite": _all_finite(rows),
            "completed_at_utc": (
                datetime.now(timezone.utc).isoformat() if completed else None
            ),
        },
    )


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"episodes": len(rows)}
    for key in (
        "episode_reward_total",
        "completed_dag_count",
        "generated_dag_count",
        "dag_completion_rate",
        "average_dag_flowtime",
        "avg_uav_queue_length",
        "arrival_attempt_count",
        "arrival_draw_count",
        "arrival_sampled_event_count",
        "arrival_admitted_count",
        "arrival_blocked_count",
        "arrival_no_event_count",
    ):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        result[key] = {
            "mean": statistics.fmean(values) if values else None,
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None,
        }
    return result


def _all_finite(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                return False
    return True


def _arrival_metric_subset(info: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "arrival_attempt_count",
        "arrival_draw_count",
        "arrival_sampled_event_count",
        "arrival_admitted_count",
        "arrival_blocked_count",
        "arrival_no_event_count",
        "arrival_blocked_reasons",
    )
    return {key: info.get(key) for key in keys}


def _validate_args(args: argparse.Namespace) -> None:
    if not bool(args.smoke) and not 20 <= int(args.episodes) <= 30:
        raise ValueError("formal paired closed-loop requires 20 to 30 episodes")
    if int(args.max_steps_per_episode) <= 0:
        raise ValueError("max steps per episode must be positive")
    if len(set(args.training_seeds)) != len(args.training_seeds):
        raise ValueError("training seeds must be unique")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(
        path,
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
