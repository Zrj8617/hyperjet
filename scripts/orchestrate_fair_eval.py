from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.exogenous_tape import read_tape
from scripts.generate_fair_eval_tapes import scene_parameters
APPROVED_ARMS = ("B1", "B2", "C2A", "C2B", "C2", "C1")
EQ10_TREATMENTS = (
    "B2_MLP_EQ10",
    "B2_TYPED_GATED_HGNN_EQ10",
    "C2A_MLP_EQ10",
    "C2A_TYPED_GATED_HGNN_EQ10",
    "C2B_MLP_EQ10",
    "C2B_TYPED_GATED_HGNN_EQ10",
    "C2_MLP_EQ10",
    "C2_TYPED_GATED_HGNN_EQ10",
)
APPROVED_SEEDS = (5, 86, 617)
TAPE_GENERATION_HEAD = "2450a40f4b877c6155d284a3f5d3c5d0cca61130"
CANDIDATE_EPISODES = (320, 360, 400, 450, 500)
TEST_PROTOCOLS = ("joint", "forced_hover")
VALIDATION_PROTOCOL = "joint"


def _csv_strings(value: str) -> tuple[str, ...]:
    values = tuple(token.strip().upper() for token in value.split(",") if token.strip())
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("values must be non-empty and unique")
    return values


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(token.strip()) for token in value.split(",") if token.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("values must be comma-separated integers") from exc
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("values must be non-empty and unique")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run six-arm fixed-tape joint validation and locked-checkpoint tests."
    )
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--tape-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=28)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arms", type=_csv_strings, required=True)
    parser.add_argument("--seeds", type=_csv_ints, required=True)
    parser.add_argument("--run-prefix", type=str, required=True)
    parser.add_argument("--gpus", type=_csv_ints, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    arms = tuple(args.arms)
    seeds = tuple(args.seeds)
    expected_arms = EQ10_TREATMENTS if str(args.run_prefix) == "20260918" else APPROVED_ARMS
    if arms != expected_arms:
        raise ValueError(f"--arms must be {','.join(expected_arms)} in this order")
    if seeds != APPROVED_SEEDS:
        raise ValueError(f"--seeds must be {','.join(map(str, APPROVED_SEEDS))}")
    if str(args.run_prefix) not in {"20260915", "20260918"}:
        raise ValueError("--run-prefix must be 20260915 or 20260918")
    if args.output_root.exists() or args.manifest.exists():
        raise FileExistsError("fair-evaluation output or manifest already exists")
    validation_tape_audit = _validate_tapes(
        tape_dir=args.tape_dir,
        tape_ids=tuple(range(100, 120)),
        required_split="validation",
    )
    args.output_root.mkdir(parents=True)
    state: dict[str, Any] = {
        "schema": "six_arm_fair_eval_orchestrator_v2",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_root": str(args.audit_root.resolve()),
        "tape_dir": str(args.tape_dir.resolve()),
        "output_root": str(args.output_root.resolve()),
        "run_prefix": str(args.run_prefix),
        "arms_order": list(arms),
        "seeds": list(seeds),
        "validation_tape_ids": list(range(100, 120)),
        "test_tape_ids": list(range(200, 250)),
        "candidate_episodes": list(CANDIDATE_EPISODES),
        "selection_protocol": VALIDATION_PROTOCOL,
        "test_protocols": list(TEST_PROTOCOLS),
        "validation_tape_audit": validation_tape_audit,
        "test_tape_audit": None,
        "test_tapes_read_after_selection_locked": None,
        "max_workers": int(args.max_workers),
        "gpus": list(args.gpus),
        "version": _version_record(),
        "arms": _audit_runs(
            args.audit_root,
            arms,
            seeds,
            run_prefix=str(args.run_prefix),
        ),
    }
    _write(args.manifest, state)
    _run_all(args=args, state=state, arms=arms, seeds=seeds)
    state["status"] = "completed"
    state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write(args.manifest, state)
    return 0


def _run_all(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    arms: tuple[str, ...],
    seeds: tuple[int, ...],
) -> None:
    validation_jobs: list[dict[str, Any]] = []
    for arm in arms:
        for seed in seeds:
            run = state["arms"][arm][str(seed)]
            for episode in CANDIDATE_EPISODES:
                checkpoint = (
                    Path(run["train_dir"])
                    / "checkpoints"
                    / f"checkpoint_ep_{episode:04d}.pt"
                )
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                validation_jobs.append(
                    _job(
                        args=args,
                        arm=arm,
                        seed=seed,
                        checkpoint=checkpoint,
                        checkpoint_label=f"ep{episode:04d}",
                        split="validation",
                        tape_ids="100-119",
                        protocol=VALIDATION_PROTOCOL,
                    )
                )
    _run_jobs(
        validation_jobs,
        max_workers=int(args.max_workers),
        gpus=tuple(args.gpus),
    )

    selections: dict[str, Any] = {}
    for arm in arms:
        selections[arm] = {}
        for seed in seeds:
            candidates = []
            for episode in CANDIDATE_EPISODES:
                path = _output_path(
                    args,
                    arm,
                    seed,
                    f"ep{episode:04d}",
                    "validation",
                    VALIDATION_PROTOCOL,
                )
                payload = json.loads(path.read_text(encoding="utf-8"))
                score = statistics.fmean(
                    float(row["J_per_offer"]) for row in payload["rows"]
                )
                candidates.append((score, episode, str(payload["checkpoint"]), str(path)))
            score, episode, checkpoint, source = min(
                candidates, key=lambda item: (item[0], item[1])
            )
            selections[arm][str(seed)] = {
                "selected_episode": int(episode),
                "selected_checkpoint": checkpoint,
                "joint_validation_J_per_offer_mean": float(score),
                "selection_source": source,
                "selection_protocol": VALIDATION_PROTOCOL,
                "tie_break": "earlier_checkpoint",
                "candidate_sources": [item[3] for item in candidates],
            }
    state["checkpoint_selections"] = selections
    state["status"] = "selection_locked"
    _write(args.manifest, state)
    _write(args.output_root / "selection_six_arm.json", selections)

    state["test_tape_audit"] = _validate_tapes(
        tape_dir=args.tape_dir,
        tape_ids=tuple(range(200, 250)),
        required_split="test",
    )
    state["test_tapes_read_after_selection_locked"] = True
    _write(args.manifest, state)

    test_jobs: list[dict[str, Any]] = []
    for arm in arms:
        for seed in seeds:
            selected = selections[arm][str(seed)]
            final_checkpoint = Path(state["arms"][arm][str(seed)]["checkpoint"])
            for label, checkpoint in (
                ("budget_selected", Path(selected["selected_checkpoint"])),
                ("final_ep0500", final_checkpoint),
            ):
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                for protocol in TEST_PROTOCOLS:
                    test_jobs.append(
                        _job(
                            args=args,
                            arm=arm,
                            seed=seed,
                            checkpoint=checkpoint,
                            checkpoint_label=label,
                            split="test",
                            tape_ids="200-249",
                            protocol=protocol,
                        )
                    )
    _run_jobs(
        test_jobs,
        max_workers=int(args.max_workers),
        gpus=tuple(args.gpus),
    )


def _audit_runs(
    audit_root: Path,
    arms: tuple[str, ...],
    seeds: tuple[int, ...],
    *,
    run_prefix: str,
) -> dict[str, Any]:
    audited: dict[str, Any] = {}
    for arm in arms:
        audited[arm] = {}
        for seed in seeds:
            result_path = audit_root / f"{run_prefix}_{arm}_seed{seed}" / "result.json"
            if not result_path.is_file():
                raise FileNotFoundError(result_path)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") != "completed":
                raise ValueError(f"incomplete source run: {result_path}")
            actual = dict(result["actual_parameters"])
            is_eq10 = arm.endswith("_EQ10")
            expected_arm, expected_encoder = (
                _treatment_identity(arm)
                if is_eq10
                else (arm, str(actual.get("task_encoder")))
            )
            if (
                str(actual["reward_redesign_arm"]) != expected_arm
                or str(actual["task_encoder"]) != expected_encoder
                or int(actual["seed"]) != seed
            ):
                raise ValueError(f"source run identity mismatch: {result_path}")
            flags = result.get("resolved_flags", {})
            if is_eq10 and (
                float(flags.get("reward_redesign_lambda_task", -1.0)) != 1.0
                or float(flags.get("reward_redesign_lambda_move", -1.0)) != 1.0
            ):
                raise ValueError(f"source run energy coefficients are not EQ10: {result_path}")
            train_dir = Path(result["train_dir"])
            rows = [
                json.loads(line)
                for line in (train_dir / "train_metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            teacher_weights: dict[str, float] = {}
            checkpoints: dict[str, str] = {}
            for episode in CANDIDATE_EPISODES:
                eligible = [row for row in rows if int(row["episode"]) == episode - 1]
                if not eligible:
                    raise ValueError(
                        f"missing episode boundary for {arm} seed {seed} episode {episode}"
                    )
                teacher_weights[str(episode)] = float(
                    eligible[-1].get("ppo_diagnostics", {}).get("teacher_weight", 0.0)
                )
                checkpoint = train_dir / "checkpoints" / f"checkpoint_ep_{episode:04d}.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                checkpoints[str(episode)] = str(checkpoint)
            audited[arm][str(seed)] = {
                "result_path": str(result_path),
                "train_dir": str(train_dir),
                "checkpoint": str(result["checkpoint"]),
                "actual_parameters": actual,
                "resolved_flags": result.get("resolved_flags"),
                "candidate_teacher_weights": teacher_weights,
                "candidate_checkpoints_read": checkpoints,
                "source_version": result.get("version"),
            }
    return audited


def _job(
    *,
    args: argparse.Namespace,
    arm: str,
    seed: int,
    checkpoint: Path,
    checkpoint_label: str,
    split: str,
    tape_ids: str,
    protocol: str,
) -> dict[str, Any]:
    output = _output_path(args, arm, seed, checkpoint_label, split, protocol)
    return {
        "arm": arm,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_label": checkpoint_label,
        "split": split,
        "tape_ids": tape_ids,
        "protocol": protocol,
        "tape_dir": str(args.tape_dir),
        "output": str(output),
        "log": str(output.with_suffix(".log")),
    }


def _output_path(
    args: argparse.Namespace,
    arm: str,
    seed: int,
    label: str,
    split: str,
    protocol: str,
) -> Path:
    return args.output_root / split / f"{arm}_seed{seed}_{label}_{protocol}.json"


def _run_jobs(
    jobs: list[dict[str, Any]], *, max_workers: int, gpus: tuple[int, ...]
) -> None:
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_one, job, index, gpus): job
            for index, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            future.result()


def _run_one(job: dict[str, Any], index: int, gpus: tuple[int, ...]) -> None:
    output = Path(job["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        raise FileExistsError(f"refusing to reuse evaluation output: {output}")
    gpu = gpus[index % len(gpus)]
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_fair_eval_batch.py"),
        "--checkpoint",
        job["checkpoint"],
        "--tape-dir",
        job["tape_dir"],
        "--tape-ids",
        job["tape_ids"],
        "--protocol",
        job["protocol"],
        "--device",
        "cuda:0",
        "--output",
        job["output"],
        "--arm",
        job["arm"],
        "--model-seed",
        str(job["seed"]),
        "--checkpoint-label",
        job["checkpoint_label"],
        "--energy-lambda",
        "1.0",
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    with Path(job["log"]).open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _version_record() -> dict[str, Any]:
    return {
        "head": _git("rev-parse", "HEAD"),
        "dirty": _git("status", "--porcelain").splitlines(),
        "orchestrator_git_object": _git("hash-object", str(Path(__file__).resolve())),
        "evaluator_git_object": _git(
            "hash-object", str(ROOT / "scripts" / "run_fair_eval_batch.py")
        ),
        "tape_module_git_object": _git(
            "hash-object", str(ROOT / "environment" / "exogenous_tape.py")
        ),
        "env_git_object": _git("hash-object", str(ROOT / "environment" / "env.py")),
    }


def _validate_tapes(
    *, tape_dir: Path, tape_ids: tuple[int, ...], required_split: str
) -> dict[str, Any]:
    manifest_path = tape_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_scene = scene_parameters()
    if manifest.get("schema") != "hyperuav_exogenous_tape_manifest_v2":
        raise ValueError("fixed-tape manifest schema is not v2")
    if manifest.get("scene_parameters") != expected_scene:
        raise ValueError("fixed-tape manifest scene parameters do not match runtime")
    if manifest.get("version", {}).get("head") != TAPE_GENERATION_HEAD:
        raise ValueError("fixed tapes do not come from the frozen 2450a40 tape commit")
    if manifest.get("version", {}).get("dirty"):
        raise ValueError("fixed tapes were generated from a dirty worktree")
    if manifest.get("splits", {}).get(required_split) != list(tape_ids):
        raise ValueError(f"fixed-tape {required_split} split does not match approved IDs")
    rows = {int(row["tape_id"]): row for row in manifest.get("tapes", [])}
    offer_count = 0
    task_count = 0
    for tape_id in tape_ids:
        path = (tape_dir / f"tape_{tape_id:04d}.json.gz").resolve()
        row = rows.get(tape_id)
        if row is None or Path(row["path"]).resolve() != path:
            raise ValueError(f"fixed-tape manifest path mismatch for tape {tape_id}")
        payload = read_tape(path)
        _validate_tape_payload(payload=payload, tape_id=tape_id, expected_scene=expected_scene)
        offer_count += int(payload["offer_count"])
        task_count += sum(len(offer["tasks"]) for slot in payload["slot_rows"] for offer in slot["offers"])
    return {
        "status": "pass",
        "split": required_split,
        "tape_ids": list(tape_ids),
        "tape_count": len(tape_ids),
        "offer_count": offer_count,
        "task_count": task_count,
        "scene_parameters": expected_scene,
        "manifest_path": str(manifest_path.resolve()),
    }


def _validate_tape_payload(
    *, payload: dict[str, Any], tape_id: int, expected_scene: dict[str, object]
) -> None:
    if int(payload["tape_id"]) != tape_id or int(payload["slots"]) != 500:
        raise ValueError(f"fixed-tape identity or length mismatch for tape {tape_id}")
    if payload.get("generation_context", {}).get("scene_parameters") != expected_scene:
        raise ValueError(f"fixed-tape generation context mismatch for tape {tape_id}")
    scene = payload["scene"]
    observed_scene = {
        "area_width": int(scene["area_width"]),
        "area_height": int(scene["area_height"]),
        "num_uavs": int(scene["num_uavs"]),
        "num_ues": int(scene["num_ues"]),
        "time_slot_duration": float(payload["time_slot_duration"]),
        "hotspot_radius": float(scene["hotspot_radius"]),
    }
    for key, value in observed_scene.items():
        if value != expected_scene[key]:
            raise ValueError(f"fixed-tape scene mismatch for {key} in tape {tape_id}")
    upload_levels = set(expected_scene["upload_bandwidth_mbps"])
    download_levels = set(expected_scene["download_bandwidth_mbps"])
    input_low, input_high = expected_scene["input_data_size_mb_range"]
    output_low, output_high = expected_scene["output_data_size_mb_range"]
    constant_low, constant_high = expected_scene["task_constant_range"]
    task_low, task_high = expected_scene["dag_task_count_range"]
    complexity_names = set(expected_scene["task_complexity_probabilities"])
    for slot in payload["slot_rows"]:
        for offer in slot["offers"]:
            job = offer["job"]
            tasks = offer["tasks"]
            if not task_low <= len(tasks) <= task_high:
                raise ValueError(f"DAG task count mismatch in tape {tape_id}")
            if float(job["base_upload_bandwidth_mbps"]) not in upload_levels:
                raise ValueError(f"upload bandwidth mismatch in tape {tape_id}")
            if float(job["base_download_bandwidth_mbps"]) not in download_levels:
                raise ValueError(f"download bandwidth mismatch in tape {tape_id}")
            for task in tasks:
                if not input_low <= float(task["input_data_size_mb"]) <= input_high:
                    raise ValueError(f"task input range mismatch in tape {tape_id}")
                if not output_low <= float(task["output_data_size_mb"]) <= output_high:
                    raise ValueError(f"task output range mismatch in tape {tape_id}")
                if not constant_low <= int(task["task_constant"]) <= constant_high:
                    raise ValueError(f"task constant range mismatch in tape {tape_id}")
                if str(task["task_complexity"]) not in complexity_names:
                    raise ValueError(f"task complexity mismatch in tape {tape_id}")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.rstrip()


def _treatment_identity(treatment: str) -> tuple[str, str]:
    suffixes = {
        "_MLP_EQ10": "mlp",
        "_TYPED_GATED_HGNN_EQ10": "typed_gated_hgnn",
    }
    for suffix, encoder in suffixes.items():
        if treatment.endswith(suffix):
            return treatment[: -len(suffix)], encoder
    raise ValueError(f"invalid EQ10 treatment name: {treatment}")


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
