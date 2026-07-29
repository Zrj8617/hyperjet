from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA = "fixed_movement_mappo_eft_aux_gate_v1"
BANDIT_SUMMARY_SCHEMA = "greedy_eft_contextual_bandit_gate_v1"
GROUPS = ("A", "B", "C", "D")
BANDIT_GROUPS = {"B", "C"}
AUX_GROUPS = {"C", "D"}
DEFAULT_CHECKPOINT_COUNTS = (1, 5, 10, 21, 30)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed-movement MAPPO per-decision EFT auxiliary diagnostic."
    )
    parser.add_argument("--bandit-dir", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 86, 1042])
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--lambda-initial", type=float, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps-per-episode", type=int, default=200)
    parser.add_argument("--rollout-horizon", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--task-embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--completed-dag-weight", type=float, default=16.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=424242)
    parser.add_argument("--eval-arrival-steps", type=int, default=200)
    parser.add_argument("--eval-max-drain-steps", type=int, default=300)
    parser.add_argument(
        "--eval-checkpoint-counts",
        type=str,
        default="0,30",
        help="Completed-update checkpoints evaluated in paired closed loop.",
    )
    parser.add_argument(
        "--checkpoint-update-counts",
        type=str,
        default="1,5,10,21,30",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs") / "fixed_movement_mappo_eft_aux_gate",
    )
    parser.add_argument("--run-name", type=str, default="eft_aux_formal")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--skip-closed-loop-eval", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _validate_args(args)
    if bool(args.pilot):
        args.groups = ["C"]
        args.seeds = [42]
        args.updates = 2
        args.eval_episodes = min(int(args.eval_episodes), 1)
        args.checkpoint_update_counts = "1,2"
        args.eval_checkpoint_counts = "0,2"

    bandit_summary = _read_json(Path(args.bandit_dir) / "summary.json")
    _validate_bandit_summary(bandit_summary, args)
    dataset_checksum = str(bandit_summary["dataset_checksum"])
    regret_scale = float(bandit_summary["regret_scale"]["value"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_dir) / f"{timestamp}_{args.run_name}"
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": SCHEMA,
        "status": "running",
        "pilot": bool(args.pilot),
        "root": str(root),
        "bandit_dir": str(Path(args.bandit_dir)),
        "dataset_checksum": dataset_checksum,
        "regret_scale": {
            "method": bandit_summary["regret_scale"]["method"],
            "value": regret_scale,
            "source": str(Path(args.bandit_dir) / "summary.json"),
        },
        "lambda_initial": float(args.lambda_initial),
        "lambda_schedule": {
            "constant_update_indices": [0, 8],
            "linear_decay_update_indices": [9, 19],
            "first_zero_update_index": 20,
            "zero_update_indices": [20, 29],
        },
        "groups": list(args.groups),
        "seeds": [int(seed) for seed in args.seeds],
        "updates": int(args.updates),
        "common_training_controls": _common_controls(args),
        "variants": [],
    }
    _write_json(root / "manifest.json", manifest)

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for group in args.groups:
            row = _run_variant(
                args=args,
                root=root,
                group=str(group),
                seed=int(seed),
                dataset_checksum=dataset_checksum,
                regret_scale=regret_scale,
            )
            rows.append(row)
            manifest["variants"] = rows
            _write_json(root / "manifest.json", manifest)
            _write_jsonl(root / "variant_rows.jsonl", rows)

    _assert_pairwise_initialization(rows)
    summary = _summarize(
        args=args,
        manifest=manifest,
        rows=rows,
        elapsed_seconds=time.perf_counter() - started,
    )
    manifest["status"] = "completed"
    manifest["elapsed_seconds"] = summary["elapsed_seconds"]
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


def _run_variant(
    *,
    args: argparse.Namespace,
    root: Path,
    group: str,
    seed: int,
    dataset_checksum: str,
    regret_scale: float,
) -> dict[str, Any]:
    cell = f"group_{group}_seed_{seed}"
    cell_root = root / cell
    cell_root.mkdir(parents=True, exist_ok=False)
    bandit_checkpoint = (
        Path(args.bandit_dir) / f"seed_{seed}" / "trained_checkpoint.pt"
        if group in BANDIT_GROUPS
        else None
    )
    if bandit_checkpoint is not None and not bandit_checkpoint.is_file():
        raise FileNotFoundError(
            f"matching-seed bandit checkpoint is missing: {bandit_checkpoint}"
        )
    auxiliary_lambda = float(args.lambda_initial) if group in AUX_GROUPS else 0.0
    base = _training_command_base(
        args=args,
        seed=seed,
        auxiliary_lambda=auxiliary_lambda,
        regret_scale=regret_scale,
        bandit_checkpoint=bandit_checkpoint,
        dataset_checksum=dataset_checksum,
    )

    print(f"[{cell}] initialize update-0 checkpoint", flush=True)
    init_result = _run_json_command(
        [
            *base,
            "--max-updates",
            "0",
            "--output-dir",
            str(cell_root / "initialization"),
            "--run-name",
            f"{cell}_initial",
        ]
    )
    init_dir = Path(init_result["run_dir"])
    initial_checkpoint = init_dir / "checkpoints" / "checkpoint_update_0000.pt"
    if not initial_checkpoint.is_file():
        raise FileNotFoundError(f"initial checkpoint was not created: {initial_checkpoint}")
    init_identity = dict(init_result["offloading_initialization"])

    eval_rows: list[dict[str, Any]] = []
    eval_counts = _parse_counts(str(args.eval_checkpoint_counts))
    if not bool(args.skip_closed_loop_eval) and 0 in eval_counts:
        # This runs before any MAPPO update, including the required B/C check.
        eval_rows.append(
            _run_evaluation(
                args=args,
                cell_root=cell_root,
                group=group,
                seed=seed,
                completed_update_count=0,
                checkpoint=initial_checkpoint,
            )
        )

    print(f"[{cell}] train {int(args.updates)} outer updates", flush=True)
    training_result = _run_json_command(
        [
            *base,
            "--max-updates",
            str(int(args.updates)),
            "--checkpoint-update-counts",
            str(args.checkpoint_update_counts),
            "--resume-checkpoint",
            str(initial_checkpoint),
            "--output-dir",
            str(cell_root / "training"),
            "--run-name",
            f"{cell}_train",
        ]
    )
    training_dir = Path(training_result["run_dir"])
    training_rows = _read_jsonl(training_dir / "train_metrics.jsonl")
    technical = _validate_training_rows(
        rows=training_rows,
        expected_updates=int(args.updates),
        auxiliary_enabled=group in AUX_GROUPS,
    )

    if not bool(args.skip_closed_loop_eval):
        for count in sorted(eval_counts - {0}):
            checkpoint = (
                training_dir
                / "checkpoints"
                / f"checkpoint_update_{int(count):04d}.pt"
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"requested evaluation checkpoint is missing: {checkpoint}"
                )
            eval_rows.append(
                _run_evaluation(
                    args=args,
                    cell_root=cell_root,
                    group=group,
                    seed=seed,
                    completed_update_count=count,
                    checkpoint=checkpoint,
                )
            )

    training_identity = dict(training_result["offloading_initialization"])
    if _identity_key(init_identity) != _identity_key(training_identity):
        raise AssertionError(f"{cell} initialization identity changed across resume")
    return {
        "group": group,
        "seed": int(seed),
        "auxiliary_enabled": bool(group in AUX_GROUPS),
        "initialization_mode": (
            "bandit_checkpoint" if group in BANDIT_GROUPS else "random"
        ),
        "initialization_identity": init_identity,
        "initial_checkpoint": str(initial_checkpoint),
        "training_dir": str(training_dir),
        "completed_update_count": int(training_result["completed_update_count"]),
        "technical_pass": bool(technical["technical_pass"]),
        "technical_diagnostics": technical,
        "evaluation_rows": eval_rows,
    }


def _training_command_base(
    *,
    args: argparse.Namespace,
    seed: int,
    auxiliary_lambda: float,
    regret_scale: float,
    bandit_checkpoint: Path | None,
    dataset_checksum: str,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_clean_mainline.py"),
        "--episodes",
        str(int(args.episodes)),
        "--max-steps-per-episode",
        str(int(args.max_steps_per_episode)),
        "--rollout-horizon",
        str(int(args.rollout_horizon)),
        "--num-envs",
        "1",
        "--sampler-backend",
        "synchronous",
        "--seed",
        str(int(seed)),
        "--lr",
        str(float(args.lr)),
        "--gamma",
        "0.99",
        "--gae-lambda",
        "0.95",
        "--clip-ratio",
        "0.2",
        "--entropy-coef",
        "0.01",
        "--value-coef",
        "0.5",
        "--no-normalize-value-targets",
        "--value-clip-epsilon",
        "0",
        "--completed-dag-weight",
        str(float(args.completed_dag_weight)),
        "--freeze-movement",
        "--task-encoder",
        "mlp",
        "--task-embedding-dim",
        str(int(args.task_embedding_dim)),
        "--hidden-dim",
        str(int(args.hidden_dim)),
        "--ppo-epochs",
        str(int(args.ppo_epochs)),
        "--max-grad-norm",
        "0.5",
        "--eft-auxiliary-lambda-initial",
        str(float(auxiliary_lambda)),
        "--eft-auxiliary-regret-scale",
        str(float(regret_scale)),
        "--eft-auxiliary-sampling-seed",
        str(int(seed) + 7_000_000),
    ]
    if bandit_checkpoint is not None:
        command.extend(
            [
                "--offloading-init-bandit-checkpoint",
                str(bandit_checkpoint),
                "--offloading-init-bandit-dataset-checksum",
                str(dataset_checksum),
            ]
        )
    return command


def _run_evaluation(
    *,
    args: argparse.Namespace,
    cell_root: Path,
    group: str,
    seed: int,
    completed_update_count: int,
    checkpoint: Path,
) -> dict[str, Any]:
    tag = f"group_{group}_seed_{seed}_update_{completed_update_count:04d}"
    print(f"[{tag}] paired fixed-movement closed-loop evaluation", flush=True)
    result = _run_json_command(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_clean_mainline.py"),
            "--checkpoint",
            str(checkpoint),
            "--episodes",
            str(int(args.eval_episodes)),
            "--arrival-steps",
            str(int(args.eval_arrival_steps)),
            "--max-drain-steps",
            str(int(args.eval_max_drain_steps)),
            "--seed",
            str(int(args.eval_seed)),
            "--device",
            str(args.device),
            "--output-dir",
            str(cell_root / "evaluations" / f"update_{completed_update_count:04d}"),
            "--run-name",
            tag,
            "--offloading-policy",
            "actor_argmax",
            "--freeze-movement",
        ]
    )
    return {
        "completed_update_count": int(completed_update_count),
        "checkpoint": str(checkpoint),
        "eval_dir": str(result["run_dir"]),
        "aggregate": result.get("aggregate", result),
    }


def _validate_training_rows(
    *,
    rows: list[dict[str, Any]],
    expected_updates: int,
    auxiliary_enabled: bool,
) -> dict[str, Any]:
    update_rows = [row for row in rows if row.get("ppo_update_step") is not None]
    if len(update_rows) != int(expected_updates):
        raise AssertionError(
            f"expected {expected_updates} PPO update rows, got {len(update_rows)}"
        )
    update_steps = [int(row["ppo_update_step"]) for row in update_rows]
    if update_steps != list(range(1, int(expected_updates) + 1)):
        raise AssertionError(f"unexpected PPO update sequence: {update_steps}")
    invalid_aux = 0
    movement_actions = 0
    nonfinite_paths: list[str] = []
    last_diagnostics: dict[str, Any] = {}
    for row_index, row in enumerate(update_rows):
        diagnostics = dict(row.get("ppo_diagnostics", {}))
        last_diagnostics = diagnostics
        invalid_aux += int(diagnostics.get("eft_aux_invalid_action_count", 0))
        movement_actions += int(row.get("ppo_movement_action_count", 0))
        _collect_nonfinite(row, f"row[{row_index}]", nonfinite_paths)
    if invalid_aux != 0:
        raise AssertionError(f"auxiliary illegal action count is {invalid_aux}")
    if movement_actions != 0:
        raise AssertionError(
            f"forced-hover gate recorded {movement_actions} movement PPO actions"
        )
    if nonfinite_paths:
        raise FloatingPointError(
            "non-finite training diagnostics: " + ", ".join(nonfinite_paths[:10])
        )
    if auxiliary_enabled and not any(
        int(dict(row.get("ppo_diagnostics", {})).get("eft_aux_effective_decision_count", 0))
        > 0
        for row in update_rows
    ):
        raise AssertionError("enabled EFT auxiliary gate had no effective decisions")
    return {
        "technical_pass": True,
        "update_row_count": len(update_rows),
        "invalid_auxiliary_action_count": int(invalid_aux),
        "movement_action_count": int(movement_actions),
        "nonfinite_paths": nonfinite_paths,
        "final_update": update_rows[-1],
        "last_diagnostics": last_diagnostics,
    }


def _assert_pairwise_initialization(rows: list[dict[str, Any]]) -> None:
    by_key = {(row["group"], int(row["seed"])): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})
    for seed in seeds:
        if ("A", seed) in by_key and ("D", seed) in by_key:
            if _identity_key(by_key[("A", seed)]["initialization_identity"]) != _identity_key(
                by_key[("D", seed)]["initialization_identity"]
            ):
                raise AssertionError(f"A/D random initialization mismatch for seed {seed}")
        if ("B", seed) in by_key and ("C", seed) in by_key:
            if _identity_key(by_key[("B", seed)]["initialization_identity"]) != _identity_key(
                by_key[("C", seed)]["initialization_identity"]
            ):
                raise AssertionError(f"B/C bandit initialization mismatch for seed {seed}")


def _identity_key(identity: dict[str, Any]) -> tuple[Any, ...]:
    return (
        identity.get("mode"),
        identity.get("training_seed"),
        identity.get("task_encoder_state_sha256"),
        identity.get("candidate_scorer_state_sha256"),
        identity.get("checkpoint_sha256"),
        identity.get("dataset_checksum"),
    )


def _summarize(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    group_aggregates: dict[str, Any] = {}
    for group in args.groups:
        group_rows = [row for row in rows if row["group"] == group]
        final_regrets = [
            float(row["technical_diagnostics"]["last_diagnostics"].get(
                "eft_rollout_chosen_raw_regret_mean", 0.0
            ))
            for row in group_rows
        ]
        group_aggregates[group] = {
            "seed_count": len(group_rows),
            "final_rollout_chosen_eft_regret_mean": (
                statistics.fmean(final_regrets) if final_regrets else None
            ),
            "final_rollout_chosen_eft_regret_std": (
                statistics.pstdev(final_regrets) if final_regrets else None
            ),
        }
    return {
        "schema": SCHEMA,
        "status": "completed",
        "pilot": bool(args.pilot),
        "elapsed_seconds": float(elapsed_seconds),
        "dataset_checksum": manifest["dataset_checksum"],
        "regret_scale": manifest["regret_scale"],
        "lambda_initial": float(args.lambda_initial),
        "rows": rows,
        "group_aggregates": group_aggregates,
        "technical_pass": all(bool(row["technical_pass"]) for row in rows),
    }


def _common_controls(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "encoder": "mlp",
        "movement": "forced_hover",
        "movement_actor_trainable": False,
        "num_envs": 1,
        "sampler_backend": "synchronous",
        "episodes_upper_bound": int(args.episodes),
        "max_steps_per_episode": int(args.max_steps_per_episode),
        "rollout_horizon": int(args.rollout_horizon),
        "ppo_epochs": int(args.ppo_epochs),
        "lr": float(args.lr),
        "completed_dag_weight": float(args.completed_dag_weight),
        "normalize_value_targets": False,
        "value_clip_epsilon": 0.0,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.updates) <= 0:
        raise ValueError("updates must be positive")
    if not math.isfinite(float(args.lambda_initial)) or float(args.lambda_initial) <= 0.0:
        raise ValueError("lambda-initial must be finite and positive")
    if not args.groups or not args.seeds:
        raise ValueError("at least one group and seed are required")
    if len(set(args.groups)) != len(args.groups) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("groups and seeds must not contain duplicates")
    if int(args.eval_episodes) <= 0 and not bool(args.skip_closed_loop_eval):
        raise ValueError("eval episodes must be positive")
    _parse_counts(str(args.checkpoint_update_counts))
    _parse_counts(str(args.eval_checkpoint_counts))


def _validate_bandit_summary(
    summary: dict[str, Any], args: argparse.Namespace
) -> None:
    if (
        summary.get("schema") != BANDIT_SUMMARY_SCHEMA
        or summary.get("status") != "completed"
    ):
        raise ValueError("bandit summary schema/status is not a completed formal gate")
    if not bool(summary.get("technical_pass", False)):
        raise ValueError("bandit summary technical_pass is false")
    available_seeds = {int(seed) for seed in summary.get("training_seeds", [])}
    required_bandit_seeds = {
        int(seed)
        for seed in args.seeds
        if any(group in BANDIT_GROUPS for group in args.groups)
    }
    missing = required_bandit_seeds - available_seeds
    if missing:
        raise ValueError(f"bandit summary lacks required seeds: {sorted(missing)}")
    scale = summary.get("regret_scale", {})
    if scale.get("method") != "train_legal_candidate_raw_eft_regret_rms":
        raise ValueError("bandit regret scale method does not match the validated gate")
    if not math.isfinite(float(scale.get("value", 0.0))) or float(scale["value"]) <= 0.0:
        raise ValueError("bandit regret scale is invalid")


def _run_json_command(command: list[str]) -> dict[str, Any]:
    print("$ " + " ".join(shlex.quote(part) for part in command), flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
    if completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {command}"
        )
    for line in reversed(completed.stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"command produced no final JSON object: {command}")


def _collect_nonfinite(value: Any, path: str, output: list[str]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            output.append(path)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _collect_nonfinite(item, f"{path}.{key}", output)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _collect_nonfinite(item, f"{path}[{index}]", output)


def _parse_counts(value: str) -> set[int]:
    counts = {int(token.strip()) for token in str(value).split(",") if token.strip()}
    if any(count < 0 for count in counts):
        raise ValueError("checkpoint counts must be non-negative")
    return counts


def _read_json(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects in {path}")
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
