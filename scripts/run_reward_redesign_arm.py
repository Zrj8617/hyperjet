from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.reward_redesign import REWARD_REDENOMINATION_ARMS
from scripts.eval_clean_mainline import (
    _load_trusted_checkpoint,
    _require_torch,
    build_arg_parser as build_eval_parser,
    run_evaluation,
)
from scripts.train_clean_mainline import (
    TASK_ENCODER_CHOICES,
    build_arg_parser as build_train_parser,
    resolved_reward_redesign_flags,
    run_training,
)


TB_TAGS = (
    "offload/entropy",
    "eval/completion_rate",
    "eval/avg_dag_flowtime",
    "eval/throughput",
    "eval/agreement_eft_greedy",
    "train/critic_ev",
    "train/episode_reward",
    "train/actor_loss",
    "train/critic_loss",
    "train/approx_kl",
    "move/move_rate_m",
    "forecast/wall_time_frac",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one frozen reward-redesign arm.")
    parser.add_argument("--arm", required=True, choices=REWARD_REDENOMINATION_ARMS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--rollout-horizon", type=int, default=125)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--task-encoder", choices=TASK_ENCODER_CHOICES, default="mlp")
    parser.add_argument("--enable-kahypar", action="store_true", default=False)
    parser.add_argument(
        "--rng-neutral-task-encoder-comparison",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--rng-neutral-reference-encoder-hidden-dim", type=int, default=None
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--task-encoder-hidden-dim", type=int, default=None)
    parser.add_argument("--task-embedding-dim", type=int, default=64)
    parser.add_argument("--reward-energy-lambda", type=float, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--teacher-anneal-total-updates", type=int, default=None)
    parser.add_argument("--rollout-cost-diagnostics", action="store_true", default=False)
    parser.add_argument("--skip-eval", action="store_true", default=False)
    return parser


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.rstrip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _writer(path: Path) -> Any:
    try:
        from tensorboardX import SummaryWriter
    except ModuleNotFoundError:
        from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(str(path))


def _parameter_counts(checkpoint: Path) -> dict[str, int]:
    torch = _require_torch()
    payload = _load_trusted_checkpoint(torch, checkpoint)
    names = ("hgnn", "movement_actor", "offloading_actor", "critic")
    counts = {
        name: int(sum(int(tensor.numel()) for tensor in payload[name].values()))
        for name in names
    }
    counts["total_policy_value_modules"] = int(sum(counts.values()))
    return counts


def _write_tensorboard(
    *,
    run_root: Path,
    train_rows: list[dict[str, Any]],
    evaluation: dict[str, Any],
    training_wall_seconds: float,
) -> None:
    writer = _writer(run_root / "tensorboard")
    try:
        for row in train_rows:
            step = int(row.get("global_slot", 0))
            writer.add_scalar("offload/entropy", float(row.get("ppo_offloading_entropy", 0.0)), step)
            writer.add_scalar("train/critic_ev", float(row.get("ppo_explained_variance", 0.0)), step)
            writer.add_scalar(
                "train/actor_loss",
                float(row.get("ppo_movement_loss", 0.0)) + float(row.get("ppo_offloading_loss", 0.0)),
                step,
            )
            writer.add_scalar("train/critic_loss", float(row.get("ppo_value_loss", 0.0)), step)
            writer.add_scalar("train/approx_kl", float(row.get("ppo_approx_kl", 0.0)), step)
            if bool(row.get("episode_terminal_record", False)):
                writer.add_scalar(
                    "train/episode_reward",
                    float(row.get("episode_reward_total", 0.0)),
                    step,
                )
                hover = row.get("hover_action_ratio")
                if hover is not None:
                    writer.add_scalar("move/move_rate_m", 1.0 - float(hover), step)
        final_step = int(train_rows[-1].get("global_slot", 0)) if train_rows else 0
        forecast_wall = sum(
            float(row.get("ppo_forecast_wall_seconds", 0.0)) for row in train_rows
        )
        writer.add_scalar(
            "forecast/wall_time_frac",
            forecast_wall / max(float(training_wall_seconds), 1e-12),
            final_step,
        )
        for tag, key in (
            ("eval/completion_rate", "DAG_completion_rate"),
            ("eval/avg_dag_flowtime", "Average_DAG_flowtime"),
            ("eval/throughput", "DAG_throughput"),
            ("eval/agreement_eft_greedy", "actor_greedy_agreement_rate"),
        ):
            value = evaluation.get(key)
            if value is not None:
                writer.add_scalar(tag, float(value), final_step)
    finally:
        writer.close()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_root = Path(args.output_root)
    run_root.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    started_wall = perf_counter()

    train_argv = [
            "--episodes", str(int(args.episodes)),
            "--max-steps-per-episode", str(int(args.steps)),
            "--rollout-horizon", str(int(args.rollout_horizon)),
            "--seed", str(int(args.seed)),
            "--device", str(args.device),
            "--task-encoder", str(args.task_encoder),
            "--hidden-dim", str(int(args.hidden_dim)),
            "--task-embedding-dim", str(int(args.task_embedding_dim)),
            "--num-envs", "1",
            "--sampler-backend", "synchronous",
            "--lr", "0.0003",
            "--gamma", "0.99",
            "--gae-lambda", "0.95",
            "--clip-ratio", "0.2",
            "--entropy-coef", "0.01",
            "--value-coef", "0.5",
            "--ppo-epochs", "1",
            "--normalize-value-targets",
            "--checkpoint-interval", "10",
            "--output-dir", str(run_root / "train"),
            "--run-name", str(args.run_name),
            "--reward-redesign-arm", str(args.arm),
            "--no-dag-progress-potential-shaping",
        ]
    if bool(args.enable_kahypar):
        train_argv.append("--enable-kahypar")
    if bool(args.rng_neutral_task_encoder_comparison):
        train_argv.append("--rng-neutral-task-encoder-comparison")
    if args.rng_neutral_reference_encoder_hidden_dim is not None:
        train_argv.extend(
            [
                "--rng-neutral-reference-encoder-hidden-dim",
                str(int(args.rng_neutral_reference_encoder_hidden_dim)),
            ]
        )
    if args.task_encoder_hidden_dim is not None:
        train_argv.extend(
            ["--task-encoder-hidden-dim", str(int(args.task_encoder_hidden_dim))]
        )
    if args.reward_energy_lambda is not None:
        train_argv.extend(
            ["--reward-energy-lambda", str(float(args.reward_energy_lambda))]
        )
    if args.max_updates is not None:
        train_argv.extend(["--max-updates", str(int(args.max_updates))])
    if args.teacher_anneal_total_updates is not None:
        train_argv.extend(
            ["--teacher-anneal-total-updates", str(int(args.teacher_anneal_total_updates))]
        )
    if bool(args.rollout_cost_diagnostics):
        train_argv.append("--rollout-cost-diagnostics")
    train_args = build_train_parser().parse_args(train_argv)
    training_started_wall = perf_counter()
    train_result = run_training(train_args)
    training_wall_seconds = float(perf_counter() - training_started_wall)
    train_dir = Path(train_result["run_dir"])
    checkpoint = train_dir / "checkpoints" / "latest.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    evaluation_wall_seconds = 0.0
    if bool(args.skip_eval):
        evaluation = {"status": "skipped_diagnostic_replay"}
    else:
        eval_argv = [
            "--checkpoint", str(checkpoint),
            "--episodes", "20",
            "--arrival-steps", "500",
            "--max-drain-steps", "0",
            "--seed", "0",
            "--device", str(args.device),
            "--output-dir", str(run_root / "eval"),
            "--run-name", f"{args.run_name}_eval20",
            "--offloading-policy", "actor_argmax",
            "--freeze-movement",
            "--clean-training-rng-prelude",
            "--no-render",
        ]
        if bool(args.enable_kahypar):
            eval_argv.append("--enable-kahypar")
        eval_args = build_eval_parser().parse_args(eval_argv)
        evaluation_started_wall = perf_counter()
        evaluation = run_evaluation(eval_args)
        evaluation_wall_seconds = float(perf_counter() - evaluation_started_wall)
    train_rows = _read_jsonl(train_dir / "train_metrics.jsonl")
    _write_tensorboard(
        run_root=run_root,
        train_rows=train_rows,
        evaluation=evaluation,
        training_wall_seconds=training_wall_seconds,
    )
    forecast_wall_seconds = sum(
        float(row.get("ppo_forecast_wall_seconds", 0.0)) for row in train_rows
    )
    kahypar_status_counts = dict(train_result.get("kahypar_partition_status_counts", {}))
    degraded_count = sum(
        int(count)
        for status, count in kahypar_status_counts.items()
        if str(status).startswith("degraded")
    )
    kahypar_health = {
        "required": bool(args.enable_kahypar),
        "partition_status_counts": kahypar_status_counts,
        "type3_slot_count": int(train_result.get("kahypar_type3_slot_count", 0)),
        "invalid_disabled_count": int(
            train_result.get("kahypar_invalid_disabled_count", 0)
        ),
        "degraded_count": int(degraded_count),
        "circuit_open": bool(train_result.get("kahypar_circuit_open", False)),
    }
    kahypar_health["pass"] = bool(
        not kahypar_health["required"]
        or (
            int(kahypar_health["degraded_count"]) == 0
            and not bool(kahypar_health["circuit_open"])
            and int(kahypar_health["type3_slot_count"]) > 0
            and int(kahypar_health["invalid_disabled_count"]) == 0
        )
    )
    result = {
        "schema": "reward_redesign_arm_v2",
        "status": "completed" if bool(kahypar_health["pass"]) else "failed_kahypar",
        "run_name": str(args.run_name),
        "arm": str(args.arm),
        "seed": int(args.seed),
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": float(perf_counter() - started_wall),
        "training_wall_seconds": training_wall_seconds,
        "evaluation_wall_seconds": evaluation_wall_seconds,
        "forecast_wall_seconds": forecast_wall_seconds,
        "forecast_total_training_wall_frac": (
            forecast_wall_seconds / max(training_wall_seconds, 1e-12)
        ),
        "train": train_result,
        "train_dir": str(train_dir),
        "checkpoint": str(checkpoint),
        "evaluation": evaluation,
        "evaluation_protocol": {
            "environment_seeds": list(range(20)),
            "movement": "forced_hover",
            "offloading": "deterministic_masked_argmax",
            "arrival_steps": 500,
            "max_drain_steps": 0,
            "clean_training_rng_prelude": True,
        },
        "resolved_flags": resolved_reward_redesign_flags(train_args),
        "parameter_counts": train_result.get("parameter_counts"),
        "tensorboard": {"directory": str(run_root / "tensorboard"), "tags": list(TB_TAGS)},
        "actual_parameters": vars(train_args),
        "version": {
            "head": _git("rev-parse", "HEAD"),
            "dirty": _git("status", "--porcelain").splitlines(),
            "runner_git_object": _git("hash-object", str(Path(__file__).resolve())),
            "trainer_git_object": _git("hash-object", str(ROOT / "scripts" / "train_clean_mainline.py")),
        },
    }
    if bool(args.enable_kahypar):
        result["parameter_counts"] = _parameter_counts(checkpoint)
        result["kahypar_health"] = kahypar_health
        result["version"]["hgnn_git_object"] = _git(
            "hash-object", str(ROOT / "marl_models" / "hgnn" / "clean_incidence.py")
        )
    (run_root / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(run_root / "result.json")}))
    if not bool(kahypar_health["pass"]):
        raise RuntimeError(f"KaHyPar health gate failed: {kahypar_health}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
