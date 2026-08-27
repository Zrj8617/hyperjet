"""Prepare and launch the GPU-queued R2 reward-ablation learnability bundle."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 86, 1042)
EVAL_SEEDS = (42, 86, 1042)
ARMS = ("control", "no_position", "low_cancel", "energy_balanced")
MODULE_KEYS = ("hgnn", "movement_actor", "offloading_actor", "critic")


@dataclass
class ActiveProcess:
    gpu: str
    job: dict[str, Any]
    stage: str
    process: subprocess.Popen[Any]
    log_handle: Any
    log_path: Path
    eval_index: int = 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiments" / "manifests" / "r2_reward_ablation_mlp.json",
    )
    parser.add_argument(
        "--offline-sweep",
        type=Path,
        default=ROOT / "logs" / "r2_reward_offline_sweep.json",
    )
    parser.add_argument(
        "--r1a2-reference",
        type=Path,
        default=ROOT / "logs" / "r1a2_environment_load_feedback_strict_crn.json",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "logs" / "r2_reward_bundle")
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "logs" / "r2_reward_bundle_summary.json",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _arm_controls(sweep: dict[str, Any]) -> dict[str, dict[str, Any]]:
    low_cancel = float(sweep["selected_low_cancel_weight"])
    energy_balanced = float(sweep["selected_energy_weight"])
    return {
        "control": {
            "completed_dag_weight": 8.0,
            "energy_weight": 0.05,
            "position_shaping": True,
        },
        "no_position": {
            "completed_dag_weight": 8.0,
            "energy_weight": 0.05,
            "position_shaping": False,
        },
        "low_cancel": {
            "completed_dag_weight": low_cancel,
            "energy_weight": 0.05,
            "position_shaping": True,
        },
        "energy_balanced": {
            "completed_dag_weight": 8.0,
            "energy_weight": energy_balanced,
            "position_shaping": True,
        },
    }


def _training_command(
    *,
    args: argparse.Namespace,
    arm: str,
    seed: int,
    controls: dict[str, Any],
    output_dir: Path,
    run_name: str,
    max_updates: int | None,
    resume_checkpoint: Path | None = None,
) -> list[str]:
    command = [
        str(args.python),
        str(ROOT / "scripts" / "train_r2_reward_arm.py"),
        "--r2-arm",
        arm,
        "--r2-energy-weight",
        str(float(controls["energy_weight"])),
        "--r2-position-shaping" if bool(controls["position_shaping"]) else "--no-r2-position-shaping",
        "--episodes",
        "1000",
        "--max-steps-per-episode",
        "500",
        "--rollout-horizon",
        "128",
        "--num-envs",
        "1",
        "--sampler-backend",
        "synchronous",
        "--ppo-epochs",
        "1",
        "--gamma",
        "0.99",
        "--gae-lambda",
        "0.95",
        "--clip-ratio",
        "0.2",
        "--lr",
        "0.0003",
        "--entropy-coef",
        "0.01",
        "--value-coef",
        "0.5",
        "--normalize-value-targets",
        "--value-clip-epsilon",
        "0.2",
        "--max-grad-norm",
        "0.5",
        "--hidden-dim",
        "128",
        "--task-embedding-dim",
        "64",
        "--task-encoder",
        "mlp",
        "--completed-dag-weight",
        str(float(controls["completed_dag_weight"])),
        "--freeze-movement",
        "--offloading-lr-scale",
        "1",
        "--movement-lr-scale",
        "1",
        "--offloading-counterfactual-coef",
        "0",
        "--offloading-action-value-loss-coef",
        "0",
        "--offloading-lagged-q-coef",
        "0",
        "--offloading-lagged-q-loss-coef",
        "0",
        "--eft-auxiliary-lambda-initial",
        "0",
        "--checkpoint-interval",
        "50",
        "--seed",
        str(int(seed)),
        "--device",
        "cuda",
        "--run-name",
        run_name,
        "--output-dir",
        str(output_dir),
    ]
    if max_updates is not None:
        command.extend(["--max-updates", str(int(max_updates))])
        if int(max_updates) == 0:
            command.extend(["--checkpoint-update-counts", "0"])
    if resume_checkpoint is not None:
        command.extend(["--resume-checkpoint", str(resume_checkpoint)])
    return command


def _last_json_line(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("process output contains no JSON object")


def _run_capture(command: list[str], *, gpu: str, log_path: Path) -> dict[str, Any]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "$ " + " ".join(shlex.quote(part) for part in command) + "\n"
        + completed.stdout
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log_path}")
    return _last_json_line(completed.stdout)


def prepare_initializations(
    args: argparse.Namespace,
    controls_by_arm: dict[str, dict[str, Any]],
    gpu_ids: list[str],
) -> dict[str, Any]:
    import torch

    root = args.output_root.resolve() / "initializations"
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ARMS:
            cell = root / f"{arm}_seed{seed}"
            result = _run_capture(
                _training_command(
                    args=args,
                    arm=arm,
                    seed=seed,
                    controls=controls_by_arm[arm],
                    output_dir=cell,
                    run_name=f"r2_{arm}_seed{seed}_initial",
                    max_updates=0,
                ),
                gpu=gpu_ids[0],
                log_path=cell / "initialization.out",
            )
            checkpoint = Path(result["run_dir"]) / "checkpoints" / "checkpoint_update_0000.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            rows.append({"arm": arm, "seed": seed, "checkpoint": str(checkpoint.resolve())})

    equality_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        reference = _load_checkpoint(torch, Path(seed_rows[0]["checkpoint"]))
        compared_tensors = 0
        for row in seed_rows[1:]:
            candidate = _load_checkpoint(torch, Path(row["checkpoint"]))
            for module_key in MODULE_KEYS:
                left = reference[module_key]
                right = candidate[module_key]
                if set(left) != set(right):
                    raise AssertionError(f"seed={seed} {module_key} state keys differ")
                for key in left:
                    if not torch.equal(left[key], right[key]):
                        raise AssertionError(
                            f"seed={seed} initial tensor mismatch: {row['arm']} {module_key}.{key}"
                        )
                    compared_tensors += 1
        equality_rows.append(
            {
                "seed": seed,
                "arms": list(ARMS),
                "direct_tensor_equality": True,
                "pairwise_reference_comparisons": len(seed_rows) - 1,
                "compared_tensors": compared_tensors,
            }
        )
    payload = {
        "schema": "r2_reward_initialization_equality_v1",
        "rows": rows,
        "equality_by_seed": equality_rows,
    }
    _write_json(args.output_root.resolve() / "initialization_equality.json", payload)
    return payload


def _load_checkpoint(torch: Any, path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location="cpu")


def _spawn(
    command: list[str], *, gpu: str, job: dict[str, Any], stage: str, log_path: Path
) -> ActiveProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    handle.write("$ " + " ".join(shlex.quote(part) for part in command) + "\n")
    handle.flush()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return ActiveProcess(
        gpu=str(gpu),
        job=job,
        stage=stage,
        process=process,
        log_handle=handle,
        log_path=log_path,
    )


def _eval_command(
    args: argparse.Namespace,
    active: ActiveProcess,
    eval_seed: int,
) -> tuple[list[str], Path, Path]:
    job = active.job
    eval_dir = Path(job["job_root"]) / "evaluations"
    output = eval_dir / f"eval_seed{eval_seed}.json"
    log_path = eval_dir / f"eval_seed{eval_seed}.out"
    controls = job["controls"]
    command = [
        str(args.python),
        str(ROOT / "scripts" / "evaluate_r2_reward_checkpoint_strict_crn.py"),
        "--checkpoint",
        str(job["final_checkpoint"]),
        "--arm",
        str(job["arm"]),
        "--training-seed",
        str(job["seed"]),
        "--eval-seed",
        str(eval_seed),
        "--slots",
        "500",
        "--completed-dag-weight",
        str(controls["completed_dag_weight"]),
        "--energy-weight",
        str(controls["energy_weight"]),
        "--position-shaping" if controls["position_shaping"] else "--no-position-shaping",
        "--device",
        "cuda",
        "--output",
        str(output),
    ]
    return command, log_path, output


def _training_diagnostic_paths(run_dir: Path) -> dict[str, Any]:
    return {
        "train_metrics_jsonl": str((run_dir / "train_metrics.jsonl").resolve()),
        "run_summary_json": str((run_dir / "run_summary.json").resolve()),
        "checkpoint": str((run_dir / "checkpoints" / "latest.pt").resolve()),
    }


def _finalize_summary(
    args: argparse.Namespace,
    jobs: list[dict[str, Any]],
    initialization: dict[str, Any],
    sweep: dict[str, Any],
) -> dict[str, Any]:
    from marl_models.mappo.clean_counterfactual_oracle_common_random import (
        audit_clean_semantic_common_random,
    )

    eval_rows: list[dict[str, Any]] = []
    audit_by_seed: list[dict[str, Any]] = []
    for job in jobs:
        for eval_path in job.get("evaluation_outputs", []):
            payload = _load_json(Path(eval_path))
            eval_rows.append(
                {key: value for key, value in payload.items() if key != "semantic_audit_snapshot"}
            )
    for seed in EVAL_SEEDS:
        snapshots = []
        for job in jobs:
            path = Path(job["job_root"]) / "evaluations" / f"eval_seed{seed}.json"
            if path.is_file():
                snapshots.append(_load_json(path)["semantic_audit_snapshot"])
        if len(snapshots) != len(jobs):
            audit_by_seed.append(
                {
                    "seed": seed,
                    "checkpoint_count": len(snapshots),
                    "shared_semantic_keys_checked": 0,
                    "semantic_key_mismatches": None,
                    "unrecognized_environment_calls": None,
                }
            )
            continue
        audit = audit_clean_semantic_common_random(snapshots)
        audit_by_seed.append(
            {
                "seed": seed,
                "checkpoint_count": len(snapshots),
                "shared_semantic_keys_checked": int(audit.shared_semantic_keys_checked),
                "semantic_key_mismatches": len(audit.semantic_key_mismatches),
                "unrecognized_environment_calls": int(audit.unrecognized_environment_calls),
            }
        )
    crn_pass = all(
        row["semantic_key_mismatches"] == 0
        and row["unrecognized_environment_calls"] == 0
        for row in audit_by_seed
    )
    reference = _load_json(args.r1a2_reference) if args.r1a2_reference.is_file() else None
    all_jobs_pass = all(job.get("status") == "completed" for job in jobs)
    return {
        "schema": "r2_reward_ablation_mlp_bundle_summary_v1",
        "status": "completed" if all_jobs_pass and crn_pass else "failed",
        "offline_sweep": sweep,
        "initialization_equality": initialization["equality_by_seed"],
        "jobs": jobs,
        "evaluation_runs": eval_rows,
        "crn_audit": {
            "by_eval_seed": audit_by_seed,
            "pass": crn_pass,
            "semantic_key": "(slot, ue_id, subsystem)",
        },
        "r1a2_reference_lines": None if reference is None else reference.get("main_table", reference.get("runs")),
    }


def launch(args: argparse.Namespace, controls_by_arm: dict[str, dict[str, Any]], gpu_ids: list[str]) -> None:
    sweep = _load_json(args.offline_sweep)
    initialization = _load_json(args.output_root.resolve() / "initialization_equality.json")
    init_by_key = {
        (row["arm"], int(row["seed"])): Path(row["checkpoint"])
        for row in initialization["rows"]
    }
    jobs: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            job_root = args.output_root.resolve() / "jobs" / f"{arm}_seed{seed}"
            jobs.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "controls": controls_by_arm[arm],
                    "initial_checkpoint": str(init_by_key[(arm, seed)].resolve()),
                    "job_root": str(job_root),
                    "status": "queued",
                    "evaluation_outputs": [],
                }
            )
    pending = list(jobs)
    active: dict[str, ActiveProcess] = {}
    status_path = args.output_root.resolve() / "launcher_status.json"
    while pending or active:
        for gpu in gpu_ids:
            if gpu in active or not pending:
                continue
            job = pending.pop(0)
            job["status"] = "training"
            job["gpu"] = gpu
            training_root = Path(job["job_root"]) / "training"
            command = _training_command(
                args=args,
                arm=job["arm"],
                seed=int(job["seed"]),
                controls=job["controls"],
                output_dir=training_root,
                run_name=f"r2_{job['arm']}_seed{job['seed']}_formal",
                max_updates=None,
                resume_checkpoint=Path(job["initial_checkpoint"]),
            )
            active[gpu] = _spawn(
                command,
                gpu=gpu,
                job=job,
                stage="training",
                log_path=Path(job["job_root"]) / "training.out",
            )
            job["worker_pid"] = int(active[gpu].process.pid)
            job["started_at"] = datetime.now(timezone.utc).isoformat()

        _write_json(
            status_path,
            {
                "status": "running",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "pending_jobs": len(pending),
                "active": [
                    {
                        "gpu": item.gpu,
                        "arm": item.job["arm"],
                        "seed": item.job["seed"],
                        "stage": item.stage,
                        "pid": item.process.pid,
                    }
                    for item in active.values()
                ],
                "jobs": jobs,
            },
        )
        for gpu, item in list(active.items()):
            return_code = item.process.poll()
            if return_code is None:
                continue
            item.log_handle.close()
            if return_code != 0:
                item.job["status"] = "failed"
                item.job["failed_stage"] = item.stage
                item.job["return_code"] = int(return_code)
                item.job["finished_at"] = datetime.now(timezone.utc).isoformat()
                del active[gpu]
                continue
            if item.stage == "training":
                result = _last_json_line(item.log_path.read_text(encoding="utf-8"))
                run_dir = Path(result["run_dir"])
                item.job["training_run_dir"] = str(run_dir.resolve())
                item.job["completed_update_count"] = int(result["completed_update_count"])
                item.job["training_diagnostics"] = _training_diagnostic_paths(run_dir)
                item.job["final_checkpoint"] = str(
                    (run_dir / "checkpoints" / "latest.pt").resolve()
                )
                command, log_path, output = _eval_command(args, item, EVAL_SEEDS[0])
                next_item = _spawn(
                    command,
                    gpu=gpu,
                    job=item.job,
                    stage="evaluation",
                    log_path=log_path,
                )
                next_item.eval_index = 0
                next_item.job["evaluation_outputs"].append(str(output.resolve()))
                active[gpu] = next_item
                continue
            next_index = int(item.eval_index) + 1
            if next_index < len(EVAL_SEEDS):
                command, log_path, output = _eval_command(args, item, EVAL_SEEDS[next_index])
                next_item = _spawn(
                    command,
                    gpu=gpu,
                    job=item.job,
                    stage="evaluation",
                    log_path=log_path,
                )
                next_item.eval_index = next_index
                next_item.job["evaluation_outputs"].append(str(output.resolve()))
                active[gpu] = next_item
            else:
                item.job["status"] = "completed"
                item.job["finished_at"] = datetime.now(timezone.utc).isoformat()
                del active[gpu]
        if pending or active:
            time.sleep(max(float(args.poll_seconds), 0.2))

    summary = _finalize_summary(args, jobs, initialization, sweep)
    _write_json(args.summary.resolve(), summary)
    _write_json(
        status_path,
        {
            "status": summary["status"],
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "jobs": jobs,
            "summary": str(args.summary.resolve()),
        },
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest = _load_json(args.manifest)
    if manifest.get("schema") != "r2_reward_ablation_mlp_manifest_v1":
        raise ValueError("R2 manifest schema mismatch")
    sweep = _load_json(args.offline_sweep)
    if sweep.get("schema") != "r2_reward_offline_sweep_v1":
        raise ValueError("R2 offline sweep schema mismatch")
    gpu_ids = [value.strip() for value in str(args.gpu_ids).split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("at least one GPU id is required")
    controls = _arm_controls(sweep)
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    if bool(args.prepare_only):
        result = prepare_initializations(args, controls, gpu_ids)
        print(json.dumps(result, sort_keys=True))
        return 0
    launch(args, controls, gpu_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
