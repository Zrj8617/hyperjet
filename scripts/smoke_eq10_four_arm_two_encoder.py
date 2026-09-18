from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("B2", "C2A", "C2B", "C2")
ENCODERS = ("mlp", "typed_gated_hgnn")
ENCODER_HIDDEN_DIMS = {"mlp": 773, "typed_gated_hgnn": 128}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the eight EQ10 five-episode smoke cells.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6")
    parser.add_argument(
        "--reuse-completed",
        action="store_true",
        help="Validate already-completed smoke cells without launching them again.",
    )
    return parser


def _command(*, arm: str, encoder: str, output_root: Path) -> tuple[list[str], Path, Path]:
    name = f"20260918_smoke_{arm}_{encoder.upper()}_EQ10_seed5"
    run_root = output_root / name
    log_path = output_root / f"{name}.log"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_reward_redesign_arm.py"),
        "--arm",
        arm,
        "--seed",
        "5",
        "--episodes",
        "5",
        "--steps",
        "500",
        "--rollout-horizon",
        "125",
        "--device",
        "cuda:0",
        "--task-encoder",
        encoder,
        "--hidden-dim",
        "128",
        "--task-encoder-hidden-dim",
        str(ENCODER_HIDDEN_DIMS[encoder]),
        "--task-embedding-dim",
        "64",
        "--reward-energy-lambda",
        "1.0",
        "--output-root",
        str(run_root),
        "--run-name",
        name,
        "--skip-eval",
    ]
    if encoder == "typed_gated_hgnn":
        command.extend(
            [
                "--rng-neutral-task-encoder-comparison",
                "--rng-neutral-reference-encoder-hidden-dim",
                str(ENCODER_HIDDEN_DIMS["mlp"]),
            ]
        )
    if arm in {"C2A", "C2B", "C2"}:
        command.extend(["--teacher-anneal-total-updates", "2000"])
    return command, run_root, log_path


def _run_cell(command: list[str], log_path: Path, gpu: int) -> None:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _expected_flags(arm: str, encoder: str) -> dict[str, Any]:
    return {
        "offloading_eft_advantage": arm in {"C2A", "C2"},
        "movement_position_advantage": arm in {"C2B", "C2"},
        "forecast_enabled": True,
        "offloading_forecast_advantage": False,
        "teacher_anneal_total_updates": 0 if arm == "B2" else 2000,
        "reward_redesign_lambda_task": 1.0,
        "reward_redesign_lambda_move": 1.0,
        "task_encoder": encoder,
    }


def _validate_result(payload: dict[str, Any], *, arm: str, encoder: str) -> None:
    if payload.get("status") != "completed":
        raise ValueError(f"{arm}/{encoder}: smoke did not complete")
    if payload.get("resolved_flags") != _expected_flags(arm, encoder):
        raise ValueError(f"{arm}/{encoder}: resolved flags differ from frozen matrix")
    actual = payload["actual_parameters"]
    expected = {
        "episodes": 5,
        "max_steps_per_episode": 500,
        "rollout_horizon": 125,
        "num_envs": 1,
        "sampler_backend": "synchronous",
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "ppo_epochs": 1,
        "normalize_value_targets": True,
        "checkpoint_interval": 10,
        "dag_progress_potential_shaping": False,
        "hidden_dim": 128,
        "task_embedding_dim": 64,
        "task_encoder_hidden_dim": ENCODER_HIDDEN_DIMS[encoder],
        "task_encoder": encoder,
        "rng_neutral_task_encoder_comparison": encoder == "typed_gated_hgnn",
        "rng_neutral_reference_encoder_hidden_dim": (
            ENCODER_HIDDEN_DIMS["mlp"] if encoder == "typed_gated_hgnn" else None
        ),
        "reward_energy_lambda": 1.0,
        "reward_redesign_arm": arm,
        "seed": 5,
    }
    mismatches = {
        key: {"actual": actual.get(key), "expected": value}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{arm}/{encoder}: actual parameter mismatch: {mismatches}")
    if int(payload["train"]["completed_update_count"]) != 20:
        raise ValueError(f"{arm}/{encoder}: expected exactly 20 smoke updates")


def _config_diff(payloads: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    ignored = {
        "reward_redesign_arm",
        "seed",
        "task_encoder",
        "task_encoder_hidden_dim",
        "output_dir",
        "run_name",
        "teacher_anneal_total_updates",
        "offloading_eft_advantage",
        "movement_position_advantage",
        "rng_neutral_task_encoder_comparison",
        "rng_neutral_reference_encoder_hidden_dim",
        "_offloading_initialization_identity",
    }
    baseline = payloads[("B2", "mlp")]["actual_parameters"]
    differences: dict[str, Any] = {}
    for (arm, encoder), payload in payloads.items():
        actual = payload["actual_parameters"]
        diff = {
            key: {"baseline": baseline.get(key), "candidate": actual.get(key)}
            for key in sorted(set(baseline) | set(actual))
            if key not in ignored and baseline.get(key) != actual.get(key)
        }
        if diff:
            differences[f"{arm}/{encoder}"] = diff
    return differences


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = args.output_root.resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f"existing output root required: {output_root}")
    if args.result.exists():
        raise FileExistsError(f"refusing to overwrite result: {args.result}")
    gpus = tuple(int(value) for value in str(args.gpus).split(","))

    jobs: list[tuple[str, str, list[str], Path, Path, int]] = []
    index = 0
    for arm in ARMS:
        for encoder in ENCODERS:
            command, run_root, log_path = _command(
                arm=arm, encoder=encoder, output_root=output_root
            )
            if args.reuse_completed:
                if not (run_root / "result.json").is_file() or not log_path.is_file():
                    raise FileNotFoundError(
                        f"completed smoke result and log required for reuse: {run_root}"
                    )
            elif run_root.exists() or log_path.exists():
                raise FileExistsError(f"refusing to overwrite smoke target: {run_root}")
            jobs.append((arm, encoder, command, run_root, log_path, gpus[index % len(gpus)]))
            index += 1

    if not args.reuse_completed:
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = {
                pool.submit(_run_cell, command, log_path, gpu): (arm, encoder)
                for arm, encoder, command, _, log_path, gpu in jobs
            }
            for future in as_completed(futures):
                future.result()

    payloads: dict[tuple[str, str], dict[str, Any]] = {}
    cells: list[dict[str, Any]] = []
    for arm, encoder, _, run_root, log_path, gpu in jobs:
        result_path = run_root / "result.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        _validate_result(payload, arm=arm, encoder=encoder)
        payloads[(arm, encoder)] = payload
        cells.append(
            {
                "arm": arm,
                "encoder": encoder,
                "gpu": gpu,
                "result_path": str(result_path),
                "log_path": str(log_path),
                "resolved_flags": payload["resolved_flags"],
                "parameter_counts": payload["parameter_counts"],
                "wall_seconds": payload["wall_seconds"],
            }
        )

    config_differences = _config_diff(payloads)
    if config_differences:
        raise ValueError(f"non-treatment configuration differences found: {config_differences}")
    counts = {
        encoder: payloads[("B2", encoder)]["parameter_counts"]
        for encoder in ENCODERS
    }
    for key in ("movement_actor", "offloading_actor", "actor_total", "critic"):
        if counts["mlp"][key] != counts["typed_gated_hgnn"][key]:
            raise ValueError(f"non-encoder parameter count differs for {key}")
    total_mlp = int(counts["mlp"]["total"])
    total_hgnn = int(counts["typed_gated_hgnn"]["total"])
    relative_difference = abs(total_hgnn - total_mlp) / max(total_hgnn, total_mlp)
    if relative_difference > 0.10:
        raise ValueError("MLP/HGNN total parameter difference exceeds 10%")

    result = {
        "schema": "eq10_four_arm_two_encoder_smoke_v1",
        "status": "pass",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "cells": cells,
        "resolved_flag_matrix": {
            f"{arm}/{encoder}": payloads[(arm, encoder)]["resolved_flags"]
            for arm in ARMS
            for encoder in ENCODERS
        },
        "parameter_counts": counts,
        "total_parameter_relative_difference": relative_difference,
        "non_treatment_config_diff": config_differences,
    }
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
