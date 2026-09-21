from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("B2", "C2A", "C2B", "C2")
SEED = 5
ENCODER = "typed_gated_hgnn"
ENCODER_HIDDEN_DIM = 128
REFERENCE_MLP_HIDDEN_DIM = 773
BASELINE_PREFIX = "20260918_smoke"
RUN_PREFIX = "20260921_smoke"


def _csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(token.strip()) for token in value.split(",") if token.strip())
    if len(values) != len(ARMS):
        raise argparse.ArgumentTypeError(
            f"exactly {len(ARMS)} GPU assignments are required"
        )
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the four EQ10 typed-gated HGNN KaHyPar-on smoke cells."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-audit-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--gpu-assignment", type=_csv_ints, required=True)
    parser.add_argument(
        "--run-prefix",
        default=RUN_PREFIX,
        help="Run-name prefix; use a new value for every non-overwriting retry.",
    )
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Validate already completed non-overwritten smoke cells without rerunning them.",
    )
    return parser


def _name(arm: str, run_prefix: str) -> str:
    return f"{run_prefix}_{arm}_TYPED_GATED_HGNN_KAHYPAR_EQ10_seed{SEED}"


def _baseline_path(root: Path, arm: str) -> Path:
    name = f"{BASELINE_PREFIX}_{arm}_TYPED_GATED_HGNN_EQ10_seed{SEED}"
    return root / name / "result.json"


def _command(
    *, arm: str, output_root: Path, run_prefix: str
) -> tuple[list[str], Path, Path]:
    name = _name(arm, run_prefix)
    run_root = output_root / name
    log_path = output_root / f"{name}.log"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_reward_redesign_arm.py"),
        "--arm",
        arm,
        "--seed",
        str(SEED),
        "--episodes",
        "5",
        "--steps",
        "500",
        "--rollout-horizon",
        "125",
        "--device",
        "cuda:0",
        "--task-encoder",
        ENCODER,
        "--hidden-dim",
        "128",
        "--task-encoder-hidden-dim",
        str(ENCODER_HIDDEN_DIM),
        "--task-embedding-dim",
        "64",
        "--reward-energy-lambda",
        "1.0",
        "--enable-kahypar",
        "--rng-neutral-task-encoder-comparison",
        "--rng-neutral-reference-encoder-hidden-dim",
        str(REFERENCE_MLP_HIDDEN_DIM),
        "--output-root",
        str(run_root),
        "--run-name",
        name,
        "--skip-eval",
    ]
    if arm in {"C2A", "C2B", "C2"}:
        command.extend(["--teacher-anneal-total-updates", "2000"])
    return command, run_root, log_path


def _run(command: list[str], log_path: Path, gpu: int) -> None:
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
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _expected_flags(arm: str) -> dict[str, Any]:
    return {
        "offloading_eft_advantage": arm in {"C2A", "C2"},
        "movement_position_advantage": arm in {"C2B", "C2"},
        "forecast_enabled": True,
        "offloading_forecast_advantage": False,
        "teacher_anneal_total_updates": 0 if arm == "B2" else 2000,
        "reward_redesign_lambda_task": 1.0,
        "reward_redesign_lambda_move": 1.0,
        "task_encoder": ENCODER,
    }


def _non_treatment_diff(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    ignored = {
        "enable_kahypar",
        "output_dir",
        "run_name",
    }
    baseline_parameters = dict(baseline["actual_parameters"])
    candidate_parameters = dict(candidate["actual_parameters"])
    return {
        key: {
            "baseline": baseline_parameters.get(key),
            "candidate": candidate_parameters.get(key),
        }
        for key in sorted(set(baseline_parameters) | set(candidate_parameters))
        if key not in ignored
        and baseline_parameters.get(key) != candidate_parameters.get(key)
    }


def _canonical_parameter_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts = dict(payload.get("parameter_counts") or {})
    task_encoder = counts.get("task_encoder", counts.get("hgnn"))
    total = counts.get("total", counts.get("total_policy_value_modules"))
    required = {
        "task_encoder": task_encoder,
        "movement_actor": counts.get("movement_actor"),
        "offloading_actor": counts.get("offloading_actor"),
        "critic": counts.get("critic"),
        "total": total,
    }
    if any(value is None for value in required.values()):
        raise ValueError(f"incomplete parameter counts: {counts}")
    return {key: int(value) for key, value in required.items()}


def _validate(
    *, arm: str, baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    if candidate.get("status") != "completed":
        raise ValueError(f"{arm}: smoke did not complete")
    if candidate.get("resolved_flags") != _expected_flags(arm):
        raise ValueError(f"{arm}: resolved flags differ from the frozen matrix")
    actual = candidate["actual_parameters"]
    if not bool(actual.get("enable_kahypar")):
        raise ValueError(f"{arm}: KaHyPar is not enabled")
    if str(actual.get("task_encoder")) != ENCODER:
        raise ValueError(f"{arm}: task encoder is not {ENCODER}")
    if int(candidate["train"]["completed_update_count"]) != 20:
        raise ValueError(f"{arm}: expected exactly 20 smoke updates")
    candidate_counts = _canonical_parameter_counts(candidate)
    baseline_counts = _canonical_parameter_counts(baseline)
    if candidate_counts != baseline_counts:
        raise ValueError(f"{arm}: parameter counts differ from KaHyPar-off baseline")
    differences = _non_treatment_diff(baseline, candidate)
    if differences:
        raise ValueError(f"{arm}: non-treatment config differences: {differences}")
    health = candidate.get("kahypar_health")
    if not isinstance(health, dict) or not bool(health.get("pass")):
        raise ValueError(f"{arm}: KaHyPar health gate failed: {health}")
    if not bool(health.get("required")):
        raise ValueError(f"{arm}: KaHyPar health was not required")
    if int(health.get("degraded_count", -1)) != 0:
        raise ValueError(f"{arm}: degraded KaHyPar slots were observed")
    if bool(health.get("circuit_open")):
        raise ValueError(f"{arm}: KaHyPar circuit opened")
    if int(health.get("type3_slot_count", 0)) <= 0:
        raise ValueError(f"{arm}: no partition/type-3 hyperedges were observed")
    return {
        "arm": arm,
        "resolved_flags": candidate["resolved_flags"],
        "parameter_counts": candidate_counts,
        "kahypar_health": health,
        "non_treatment_config_diff": differences,
        "baseline_result": baseline,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = args.output_root.resolve()
    baseline_root = args.baseline_audit_root.resolve()
    if not output_root.is_dir() or not baseline_root.is_dir():
        raise FileNotFoundError("existing output and baseline audit roots are required")
    if args.result.exists():
        raise FileExistsError(f"refusing to overwrite smoke result: {args.result}")

    jobs: list[tuple[str, list[str], Path, Path, int]] = []
    baselines: dict[str, dict[str, Any]] = {}
    for arm, gpu in zip(ARMS, args.gpu_assignment, strict=True):
        baseline_path = _baseline_path(baseline_root, arm)
        if not baseline_path.is_file():
            raise FileNotFoundError(baseline_path)
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline.get("status") != "completed":
            raise ValueError(f"incomplete baseline: {baseline_path}")
        baselines[arm] = baseline
        command, run_root, log_path = _command(
            arm=arm,
            output_root=output_root,
            run_prefix=str(args.run_prefix),
        )
        if args.validate_existing:
            if not (run_root / "result.json").is_file() or not log_path.is_file():
                raise FileNotFoundError(f"incomplete existing smoke target: {run_root}")
        elif run_root.exists() or log_path.exists():
            raise FileExistsError(f"refusing to overwrite smoke target: {run_root}")
        jobs.append((arm, command, run_root, log_path, gpu))

    # KaHyPar uses a spawned persistent worker.  On this server, simultaneous
    # cold starts can contend on storage long enough for all workers to exceed
    # their response deadline and open the circuit.  Smoke cells are therefore
    # deliberately serialized; production runs are staggered, then overlap.
    if not args.validate_existing:
        for _, command, _, log_path, gpu in jobs:
            _run(command, log_path, gpu)

    cells = []
    for arm, _, run_root, log_path, gpu in jobs:
        result_path = run_root / "result.json"
        candidate = json.loads(result_path.read_text(encoding="utf-8"))
        validation = _validate(
            arm=arm,
            baseline=baselines[arm],
            candidate=candidate,
        )
        validation.pop("baseline_result")
        cells.append(
            {
                **validation,
                "gpu": gpu,
                "result_path": str(result_path),
                "log_path": str(log_path),
                "baseline_result_path": str(_baseline_path(baseline_root, arm)),
                "wall_seconds": candidate.get("wall_seconds"),
            }
        )

    result = {
        "schema": "eq10_kahypar_four_arm_smoke_v1",
        "status": "pass",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "arms": list(ARMS),
        "seed": SEED,
        "task_encoder": ENCODER,
        "run_prefix": str(args.run_prefix),
        "enable_kahypar": True,
        "cells": cells,
    }
    args.result.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
