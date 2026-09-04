"""Run the approved paired Stage-1 EFT-anchored offloading-credit probe."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("off", "on")
DEFAULT_TRAIN_SEEDS = (0, 1, 2)
DEFAULT_EVAL_SEEDS = tuple(range(20))
SYSTEM_METRICS = {
    "dag_completion_rate": "DAG_completion_rate",
    "average_dag_flowtime": "Average_DAG_flowtime",
    "dag_throughput": "DAG_throughput",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--leverage-result", type=Path, required=True)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=list(DEFAULT_TRAIN_SEEDS))
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=list(DEFAULT_EVAL_SEEDS))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--rollout-horizon", type=int, default=128)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _validate_args(args)
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "runs", "eval", "curves"):
        (root / name).mkdir()

    leverage = _read_json(Path(args.leverage_result))
    _validate_leverage_reference(leverage, args)
    version = _version_record()
    manifest: dict[str, Any] = {
        "schema": "stage1_eft_anchored_credit_probe_v1",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "mechanism": {
            "name": "historical_eft_regret_advantage",
            "gate": "--offloading-eft-advantage",
            "replaces_return_based_advantage": True,
            "insertion": "offloading head PPO action loss only",
        },
        "design": {
            "arms": list(ARMS),
            "train_seeds": [int(seed) for seed in args.train_seeds],
            "eval_seeds": [int(seed) for seed in args.eval_seeds],
            "episodes": int(args.episodes),
            "max_steps_per_episode": int(args.max_steps_per_episode),
            "rollout_horizon": int(args.rollout_horizon),
            "task_encoder": "mlp",
            "training_movement": "unchanged learned movement",
            "evaluation_movement": "forced_hover",
            "evaluation_arrival_steps": 500,
            "evaluation_max_drain_steps": 0,
            "evaluation_rng_prelude": "run_clean_policy_baseline compatible",
            "gpus": [int(gpu) for gpu in args.gpus],
        },
        "leverage_result": str(Path(args.leverage_result).resolve()),
        "version": version,
        "commands": {"training": [], "evaluation": []},
    }
    _write_json(root / "manifest.json", manifest)
    _write_version_text(root / "run_version.txt", version)

    training_jobs = [
        (arm, int(seed)) for seed in args.train_seeds for arm in ARMS
    ]
    training_rows = _run_gpu_queues(
        training_jobs,
        args.gpus,
        lambda job, gpu: _run_training_cell(args, root, job[0], job[1], gpu),
    )
    for row in training_rows:
        manifest["commands"]["training"].append(row["command"])
    _write_json(root / "manifest.json", manifest)

    initialization_checks = _check_paired_initialization(training_rows, args.train_seeds)
    if not all(
        row["all_tensors_equal"]
        and row["optimizer_state_equal"]
        and row["rng_state_equal"]
        for row in initialization_checks
    ):
        raise AssertionError("paired OFF/ON initialization model/optimizer/RNG states differ")
    config_check = _check_single_variable(training_rows, args.train_seeds)
    if not config_check["pass"]:
        raise AssertionError(f"training arms differ beyond EFT gate: {config_check}")

    evaluation_jobs = [
        (row["arm"], int(row["seed"]), int(eval_seed), row["checkpoint"])
        for row in training_rows
        for eval_seed in args.eval_seeds
    ]
    evaluation_rows = _run_gpu_queues(
        evaluation_jobs,
        args.gpus,
        lambda job, gpu: _run_eval_cell(
            args, root, job[0], job[1], job[2], Path(job[3]), gpu,
            policy="actor_argmax",
        ),
    )
    for row in evaluation_rows:
        manifest["commands"]["evaluation"].append(row["command"])

    # One checkpoint is enough to prove that the evaluator's environment/RNG
    # protocol reproduces the frozen EFT-greedy leverage reference per seed.
    calibration_checkpoint = Path(training_rows[0]["checkpoint"])
    calibration_jobs = [
        (int(eval_seed), calibration_checkpoint) for eval_seed in args.eval_seeds
    ]
    calibration_rows = _run_gpu_queues(
        calibration_jobs,
        args.gpus,
        lambda job, gpu: _run_eval_cell(
            args, root, "calibration", -1, job[0], job[1], gpu,
            policy="greedy_eft_teacher",
        ),
    )
    for row in calibration_rows:
        manifest["commands"]["evaluation"].append(row["command"])

    result = _summarize(
        args=args,
        root=root,
        leverage=leverage,
        version=version,
        training_rows=training_rows,
        evaluation_rows=evaluation_rows,
        calibration_rows=calibration_rows,
        initialization_checks=initialization_checks,
        config_check=config_check,
        commands=manifest["commands"],
    )
    _write_json(root / "result.json", result)
    _plot(result, root / "curves")
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = result["completed_at_utc"]
    manifest["result"] = str(root / "result.json")
    manifest["curves"] = result["curve_paths"]
    _write_json(root / "manifest.json", manifest)
    print(json.dumps({"status": "completed", "result": str(root / "result.json")}, sort_keys=True))
    return 0


def _training_command(args: argparse.Namespace, parent: Path, arm: str, seed: int) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_clean_mainline.py"),
        "--episodes", str(int(args.episodes)),
        "--max-steps-per-episode", str(int(args.max_steps_per_episode)),
        "--rollout-horizon", str(int(args.rollout_horizon)),
        "--num-envs", "1",
        "--sampler-backend", "synchronous",
        "--ppo-epochs", "1",
        "--gamma", "0.99",
        "--gae-lambda", "0.95",
        "--clip-ratio", "0.2",
        "--lr", "0.0003",
        "--entropy-coef", "0.01",
        "--value-coef", "0.5",
        "--normalize-value-targets",
        "--value-clip-epsilon", "0.2",
        "--max-grad-norm", "0.5",
        "--hidden-dim", "128",
        "--task-embedding-dim", "64",
        "--task-encoder", "mlp",
        "--critic-task-pooling", "mean",
        "--completed-dag-weight", "8",
        "--offloading-lr-scale", "1",
        "--movement-lr-scale", "1",
        "--eft-auxiliary-lambda-initial", "0",
        "--eft-auxiliary-regret-scale", "1",
        "--eft-auxiliary-sampling-seed", "0",
        "--checkpoint-interval", str(int(args.episodes)),
        "--seed", str(int(seed)),
        "--device", str(args.device),
        "--run-name", f"stage1_eft_anchor_{arm}_seed{seed}",
        "--output-dir", str(parent),
    ]
    if arm == "on":
        command.append("--offloading-eft-advantage")
    return command


def _run_training_cell(
    args: argparse.Namespace, root: Path, arm: str, seed: int, gpu: int
) -> dict[str, Any]:
    parent = root / "runs" / arm / f"seed{seed}"
    parent.mkdir(parents=True)
    command = _training_command(args, parent, arm, seed)
    log = root / "logs" / f"train_{arm}_seed{seed}.log"
    _run_logged(command, log, gpu)
    run_dir = _only_child_directory(parent)
    summary = _read_json(run_dir / "run_summary.json")
    if summary.get("status") != "completed":
        raise RuntimeError(f"training did not complete: {run_dir}")
    checkpoint = run_dir / "checkpoints" / "latest.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return {
        "arm": arm,
        "seed": int(seed),
        "gpu": int(gpu),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "command": [f"CUDA_VISIBLE_DEVICES={gpu}", *command],
        "completed_update_count": int(summary["completed_update_count"]),
    }


def _run_eval_cell(
    args: argparse.Namespace,
    root: Path,
    arm: str,
    model_seed: int,
    eval_seed: int,
    checkpoint: Path,
    gpu: int,
    *,
    policy: str,
) -> dict[str, Any]:
    parent = root / "eval" / arm / f"model_seed{model_seed}" / f"env_seed{eval_seed}"
    parent.mkdir(parents=True)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "eval_clean_mainline.py"),
        "--checkpoint", str(checkpoint),
        "--episodes", "1",
        "--arrival-steps", "500",
        "--max-drain-steps", "0",
        "--seed", str(int(eval_seed)),
        "--device", str(args.device),
        "--output-dir", str(parent),
        "--run-name", f"stage1_{arm}_model{model_seed}_{policy}",
        "--offloading-policy", policy,
        "--freeze-movement",
        "--clean-training-rng-prelude",
        "--no-render",
    ]
    log = root / "logs" / f"eval_{arm}_model{model_seed}_env{eval_seed}_{policy}.log"
    _run_logged(command, log, gpu)
    run_dir = _only_child_directory(parent)
    summary = _read_json(run_dir / "eval_summary.json")
    if summary.get("status") != "completed":
        raise RuntimeError(f"evaluation did not complete: {run_dir}")
    return {
        "arm": arm,
        "model_seed": int(model_seed),
        "environment_seed": int(eval_seed),
        "policy": policy,
        "gpu": int(gpu),
        "run_dir": str(run_dir),
        "command": [f"CUDA_VISIBLE_DEVICES={gpu}", *command],
        "summary": summary,
    }


def _run_gpu_queues(
    jobs: list[Any],
    gpus: list[int],
    worker: Callable[[Any, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    queues = [[] for _ in gpus]
    for index, job in enumerate(jobs):
        queues[index % len(gpus)].append(job)

    def run_queue(gpu: int, queue: list[Any]) -> list[dict[str, Any]]:
        return [worker(job, int(gpu)) for job in queue]

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(run_queue, int(gpu), queue) for gpu, queue in zip(gpus, queues)]
        for future in as_completed(futures):
            rows.extend(future.result())
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("arm", "")),
            int(row.get("seed", row.get("model_seed", -1))),
            int(row.get("environment_seed", -1)),
        ),
    )


def _run_logged(command: list[str], log: Path, gpu: int) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _check_paired_initialization(rows: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    import torch

    checks = []
    by_key = {(row["arm"], int(row["seed"])): row for row in rows}
    for seed in seeds:
        payloads = {}
        for arm in ARMS:
            run_dir = Path(by_key[(arm, int(seed))]["run_dir"])
            payloads[arm] = torch.load(
                run_dir / "checkpoints" / "checkpoint_update_0000.pt",
                map_location="cpu",
                weights_only=False,
            )
        left = payloads["off"]
        right = payloads["on"]
        tensor_count = 0
        all_equal = True
        for module_name in ("hgnn", "movement_actor", "offloading_actor", "critic"):
            left_state = left[module_name]
            right_state = right[module_name]
            if left_state.keys() != right_state.keys():
                all_equal = False
                continue
            for name in left_state:
                tensor_count += 1
                all_equal = all_equal and bool(torch.equal(left_state[name], right_state[name]))
        checks.append({
            "seed": int(seed),
            "all_tensors_equal": bool(all_equal),
            "compared_tensor_count": int(tensor_count),
            "optimizer_state_equal": _nested_equal(left["optimizer"], right["optimizer"], torch),
            "rng_state_equal": _nested_equal(left.get("rng_state"), right.get("rng_state"), torch),
        })
    return checks


def _nested_equal(left: Any, right: Any, torch: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return bool(torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right))
    if isinstance(left, dict) or isinstance(right, dict):
        return bool(
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_nested_equal(left[key], right[key], torch) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return bool(
            type(left) is type(right)
            and len(left) == len(right)
            and all(_nested_equal(a, b, torch) for a, b in zip(left, right))
        )
    try:
        import numpy as np
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            return bool(isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right))
    except ModuleNotFoundError:
        pass
    return bool(left == right)


def _check_single_variable(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    by_key = {(row["arm"], int(row["seed"])): row for row in rows}
    mismatches: dict[str, list[str]] = {}
    for seed in seeds:
        configs = {}
        for arm in ARMS:
            config = _read_json(Path(by_key[(arm, int(seed))]["run_dir"]) / "config.json")
            cli = dict(config["cli"])
            for key in ("offloading_eft_advantage", "output_dir", "run_name"):
                cli.pop(key, None)
            controls = dict(config["experiment_controls"])
            controls.pop("offloading_eft_advantage", None)
            configs[arm] = {"cli": cli, "experiment_controls": controls, "clean_scene": config["clean_scene"]}
        if configs["off"] != configs["on"]:
            mismatches[str(seed)] = _mapping_diff(configs["off"], configs["on"])
    return {
        "pass": not mismatches,
        "allowed_differences": ["offloading_eft_advantage", "output_dir", "run_name"],
        "mismatches": mismatches,
    }


def _mapping_diff(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        rows = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                rows.append(path)
            else:
                rows.extend(_mapping_diff(left[key], right[key], path))
        return rows
    return [] if left == right else [prefix]


def _summarize(
    *,
    args: argparse.Namespace,
    root: Path,
    leverage: dict[str, Any],
    version: dict[str, Any],
    training_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    initialization_checks: list[dict[str, Any]],
    config_check: dict[str, Any],
    commands: dict[str, Any],
) -> dict[str, Any]:
    training: dict[str, Any] = {arm: {} for arm in ARMS}
    for row in training_rows:
        metrics = _read_jsonl(Path(row["run_dir"]) / "train_metrics.jsonl")
        curve = []
        seen_updates = set()
        for metric_row in metrics:
            update = metric_row.get("ppo_update_step")
            entropy = dict(metric_row.get("ppo_diagnostics", {})).get(
                "rollout_offloading_entropy_normalized_mean"
            )
            if update is None or entropy is None or int(update) in seen_updates:
                continue
            seen_updates.add(int(update))
            curve.append({"update": int(update), "normalized_entropy": float(entropy)})
        training[row["arm"]][str(row["seed"])] = {
            **{key: value for key, value in row.items() if key != "command"},
            "actual_command": row["command"],
            "entropy_curve": curve,
            "initial_entropy_mean": _head_mean(curve),
            "final_entropy_mean": _tail_mean(curve),
        }

    eval_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for row in evaluation_rows:
        eval_by_arm[row["arm"]].append(row)
    evaluation: dict[str, Any] = {}
    for arm, rows in eval_by_arm.items():
        per_model: dict[str, Any] = {}
        for model_seed in args.train_seeds:
            model_rows = [row for row in rows if int(row["model_seed"]) == int(model_seed)]
            per_model[str(model_seed)] = _eval_aggregate(model_rows)
        evaluation[arm] = {
            "per_model_seed": per_model,
            "pooled": _eval_aggregate(rows),
        }

    reference = {
        policy: {
            metric: float(leverage["aggregates"][policy][metric]["mean"])
            for metric in SYSTEM_METRICS
        }
        for policy in ("random", "eft_greedy")
    }
    progress = {}
    exceeds = {}
    for arm in ARMS:
        pooled = evaluation[arm]["pooled"]
        progress[arm] = {}
        exceeds[arm] = {}
        for metric in SYSTEM_METRICS:
            learned = float(pooled[metric]["mean"])
            random_value = reference["random"][metric]
            greedy_value = reference["eft_greedy"][metric]
            if metric == "average_dag_flowtime":
                fraction = (random_value - learned) / (random_value - greedy_value)
                beats = learned < greedy_value
            else:
                fraction = (learned - random_value) / (greedy_value - random_value)
                beats = learned > greedy_value
            progress[arm][metric] = float(fraction)
            exceeds[arm][metric] = bool(beats)

    calibration_differences = []
    for row in calibration_rows:
        seed = int(row["environment_seed"])
        ref = leverage["cells"]["eft_greedy"][str(seed)]["metrics"]
        summary = row["summary"]
        differences = {
            metric: float(summary[eval_name]) - float(ref[metric])
            for metric, eval_name in SYSTEM_METRICS.items()
        }
        calibration_differences.append({"seed": seed, "differences": differences})
    max_calibration_error = max(
        abs(value)
        for row in calibration_differences
        for value in row["differences"].values()
    )
    if max_calibration_error > 1e-10:
        raise AssertionError(
            f"leverage-compatible evaluator did not reproduce EFT greedy: {max_calibration_error}"
        )

    result = {
        "schema": "stage1_eft_anchored_credit_probe_v1",
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": version,
        "design": {
            "train_seeds": [int(seed) for seed in args.train_seeds],
            "eval_seeds": [int(seed) for seed in args.eval_seeds],
            "mechanism": "historical EFT-regret advantage",
            "gate": "--offloading-eft-advantage",
            "replaces_offloading_return_based_advantage": True,
            "training_movement": "unchanged learned movement",
            "evaluation_movement": "forced_hover",
            "evaluation_protocol": "1 episode x 500 arrival slots x 0 drain, per environment seed",
            "single_variable": "offloading_eft_advantage false/true",
        },
        "commands": commands,
        "engineering_checks": {
            "paired_initialization": initialization_checks,
            "single_variable": config_check,
            "gate_off_is_default": True,
            "evaluation_deterministic": True,
            "leverage_protocol_calibration": {
                "pass": True,
                "max_absolute_metric_error": float(max_calibration_error),
                "per_seed": calibration_differences,
            },
        },
        "leverage_reference": {
            "path": str(Path(args.leverage_result).resolve()),
            "random": reference["random"],
            "eft_greedy": reference["eft_greedy"],
        },
        "training": training,
        "evaluation": evaluation,
        "fraction_of_random_to_greedy_gap": progress,
        "strictly_exceeds_eft_greedy": exceeds,
        "curve_paths": {
            "entropy": str(root / "curves" / "entropy_vs_update.png"),
            "system": str(root / "curves" / "system_metrics_reference_band.png"),
            "agreement": str(root / "curves" / "agreement_rate.png"),
        },
    }
    return result


def _eval_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = {}
    for metric, eval_name in SYSTEM_METRICS.items():
        values = [float(row["summary"][eval_name]) for row in rows]
        aggregate[metric] = _mean_std(values)
    agreement_count = sum(int(row["summary"]["actor_greedy_agreement_count"]) for row in rows)
    comparison_count = sum(int(row["summary"]["actor_greedy_comparison_count"]) for row in rows)
    entropy_samples = []
    for row in rows:
        value = row["summary"].get("actor_normalized_entropy_mean")
        if value is not None:
            entropy_samples.append(float(value))
    aggregate["actor_greedy_agreement_rate"] = (
        float(agreement_count / comparison_count) if comparison_count else None
    )
    aggregate["actor_greedy_agreement_count"] = int(agreement_count)
    aggregate["actor_greedy_comparison_count"] = int(comparison_count)
    aggregate["actor_normalized_entropy_mean_across_cells"] = (
        statistics.fmean(entropy_samples) if entropy_samples else None
    )
    aggregate["cell_count"] = len(rows)
    return aggregate


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "sample_std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def _head_mean(curve: list[dict[str, Any]]) -> float | None:
    if not curve:
        return None
    count = max(1, math.ceil(len(curve) * 0.2))
    return float(statistics.fmean(row["normalized_entropy"] for row in curve[:count]))


def _tail_mean(curve: list[dict[str, Any]]) -> float | None:
    if not curve:
        return None
    count = max(1, math.ceil(len(curve) * 0.2))
    return float(statistics.fmean(row["normalized_entropy"] for row in curve[-count:]))


def _plot(result: dict[str, Any], curves: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"off": "#4C78A8", "on": "#E45756"}
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for arm in ARMS:
        for seed, row in result["training"][arm].items():
            curve = row["entropy_curve"]
            axis.plot(
                [point["update"] for point in curve],
                [point["normalized_entropy"] for point in curve],
                color=colors[arm], alpha=0.35, label=f"{arm} seed {seed}",
            )
    axis.set(xlabel="PPO update", ylabel="Normalized offloading entropy", title="EFT-anchored credit probe")
    axis.legend(ncol=2, fontsize=8)
    figure.savefig(curves / "entropy_vs_update.png", dpi=170)
    plt.close(figure)

    metrics = list(SYSTEM_METRICS)
    titles = ["DAG completion rate", "Average DAG flowtime", "DAG throughput"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, metric, title in zip(axes, metrics, titles):
        labels = ["random", "OFF", "ON", "EFT greedy"]
        values = [
            result["leverage_reference"]["random"][metric],
            result["evaluation"]["off"]["pooled"][metric]["mean"],
            result["evaluation"]["on"]["pooled"][metric]["mean"],
            result["leverage_reference"]["eft_greedy"][metric],
        ]
        axis.bar(labels, values, color=["#BAB0AC", colors["off"], colors["on"], "#59A14F"])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
    figure.savefig(curves / "system_metrics_reference_band.png", dpi=170)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    values = [result["evaluation"][arm]["pooled"]["actor_greedy_agreement_rate"] for arm in ARMS]
    axis.bar(list(ARMS), values, color=[colors["off"], colors["on"]])
    axis.set(ylim=(0.0, 1.0), ylabel="Agreement rate", title="Actor argmax vs EFT greedy")
    figure.savefig(curves / "agreement_rate.png", dpi=170)
    plt.close(figure)


def _version_record() -> dict[str, Any]:
    scripts = [
        "scripts/run_stage1_eft_anchored_credit_probe.py",
        "scripts/train_clean_mainline.py",
        "scripts/eval_clean_mainline.py",
        "marl_models/mappo/clean_trainer.py",
        "scripts/offloading_policy_gate.py",
    ]
    return {
        "head": _git("rev-parse", "HEAD"),
        "dirty": _git("status", "--porcelain").splitlines(),
        "git_hash_object": {path: _git("hash-object", path) for path in scripts},
    }


def _write_version_text(path: Path, version: dict[str, Any]) -> None:
    lines = [f"HEAD {version['head']}", f"DIRTY_COUNT {len(version['dirty'])}"]
    lines.extend(version["dirty"])
    lines.append("GIT_HASH_OBJECT")
    lines.extend(f"{value}  {key}" for key, value in version["git_hash_object"].items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _validate_args(args: argparse.Namespace) -> None:
    if list(args.train_seeds) != list(DEFAULT_TRAIN_SEEDS):
        raise ValueError("formal probe requires training seeds 0 1 2")
    if list(args.eval_seeds) != list(DEFAULT_EVAL_SEEDS):
        raise ValueError("formal probe requires leverage evaluation seeds 0..19")
    if int(args.episodes) != 100 or int(args.max_steps_per_episode) != 500:
        raise ValueError("formal short gate is frozen at 100 episodes x 500 slots")
    if int(args.rollout_horizon) != 128:
        raise ValueError("formal rollout horizon is frozen at 128")
    if len(set(args.gpus)) != len(args.gpus) or not args.gpus:
        raise ValueError("--gpus must contain distinct GPU indices")
    if Path(args.output_root).exists():
        raise FileExistsError(args.output_root)


def _validate_leverage_reference(leverage: dict[str, Any], args: argparse.Namespace) -> None:
    design = leverage.get("design", {})
    if leverage.get("schema") != "offloading_leverage_check_v1":
        raise ValueError("unexpected leverage result schema")
    if design.get("movement_policy") != "forced_hover":
        raise ValueError("leverage reference is not forced-hover")
    if list(design.get("seeds", [])) != list(args.eval_seeds):
        raise ValueError("evaluation seeds do not match leverage reference")
    if int(design.get("episodes_per_seed", 0)) != 1 or int(design.get("max_steps_per_episode", 0)) != 500:
        raise ValueError("leverage horizon does not match 1 x 500 protocol")


def _only_child_directory(parent: Path) -> Path:
    children = [path for path in parent.iterdir() if path.is_dir()]
    if len(children) != 1:
        raise RuntimeError(f"expected one run directory under {parent}, found {len(children)}")
    return children[0]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
