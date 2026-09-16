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
from scripts.orchestrate_fair_eval import _validate_tape_payload


ARMS = ("B2", "C1", "C2")
SEEDS = (5, 86, 617)
CANDIDATES = (320, 360, 400, 450, 500)
RUN_PREFIX = "20260915_TYPED_GATED_HGNN"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the frozen typed-gated HGNN screen.")
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--tape-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--gpus", type=str, required=True)
    parser.add_argument("--max-workers", type=int, default=28)
    return parser


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.rstrip()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _version() -> dict[str, Any]:
    files = {
        "orchestrator": Path(__file__).resolve(),
        "evaluator": ROOT / "scripts" / "run_fair_eval_batch.py",
        "eval_core": ROOT / "scripts" / "eval_clean_mainline.py",
        "summarizer": ROOT / "scripts" / "summarize_20260915_typed_gated_hgnn_three_arm_screen.py",
    }
    return {
        "head": _git("rev-parse", "HEAD"),
        "dirty": _git("status", "--porcelain").splitlines(),
        "git_objects": {name: _git("hash-object", str(path)) for name, path in files.items()},
    }


def _load_shared_manifest(tape_dir: Path) -> dict[str, Any]:
    manifest = json.loads((tape_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "hyperuav_exogenous_tape_manifest_v2":
        raise ValueError("fixed-tape manifest schema mismatch")
    if manifest.get("scene_parameters") != scene_parameters():
        raise ValueError("fixed-tape scene mismatch")
    version = dict(manifest.get("version", {}))
    if version.get("dirty"):
        raise ValueError("tapes were generated from a dirty worktree")
    expected_objects = {
        "config_git_object": _git("hash-object", str(ROOT / "config.py")),
        "generator_git_object": _git(
            "hash-object", str(ROOT / "scripts" / "generate_fair_eval_tapes.py")
        ),
        "tape_module_git_object": _git(
            "hash-object", str(ROOT / "environment" / "exogenous_tape.py")
        ),
    }
    for key, expected in expected_objects.items():
        if version.get(key) != expected:
            raise ValueError(f"fixed-tape object compatibility failed: {key}")
    return manifest


def _validate_split(
    *, manifest: dict[str, Any], tape_dir: Path, split: str, tape_ids: tuple[int, ...]
) -> dict[str, Any]:
    if manifest.get("splits", {}).get(split) != list(tape_ids):
        raise ValueError(f"fixed-tape {split} split mismatch")
    rows = {int(row["tape_id"]): dict(row) for row in manifest.get("tapes", [])}
    for tape_id in tape_ids:
        path = (tape_dir / f"tape_{tape_id:04d}.json.gz").resolve()
        row = rows.get(tape_id)
        if row is None or Path(row["path"]).resolve() != path:
            raise ValueError(f"fixed-tape path mismatch for {tape_id}")
        payload = read_tape(path)
        _validate_tape_payload(payload=payload, tape_id=tape_id, expected_scene=scene_parameters())
        if payload.get("generation_context", {}).get("version") != manifest.get("version"):
            raise ValueError(f"fixed-tape generation context mismatch for {tape_id}")
    return {"status": "pass", "split": split, "tape_ids": list(tape_ids)}


def _audit_runs(audit_root: Path) -> dict[str, Any]:
    audited: dict[str, Any] = {}
    for arm in ARMS:
        audited[arm] = {}
        for seed in SEEDS:
            result_path = audit_root / f"{RUN_PREFIX}_{arm}_seed{seed}" / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") != "completed" or not result.get("kahypar_health", {}).get("pass"):
                raise ValueError(f"incomplete or unhealthy source run: {result_path}")
            actual = dict(result["actual_parameters"])
            if (
                str(actual.get("task_encoder")) != "typed_gated_hgnn"
                or not bool(actual.get("enable_kahypar"))
                or str(actual.get("reward_redesign_arm")) != arm
                or int(actual.get("seed")) != seed
            ):
                raise ValueError(f"source identity mismatch: {result_path}")
            train_dir = Path(result["train_dir"])
            checkpoints = {}
            for episode in CANDIDATES:
                checkpoint = train_dir / "checkpoints" / f"checkpoint_ep_{episode:04d}.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                checkpoints[str(episode)] = str(checkpoint)
            audited[arm][str(seed)] = {
                "result_path": str(result_path),
                "train_dir": str(train_dir),
                "checkpoints": checkpoints,
            }
    return audited


def _job_output(root: Path, arm: str, seed: int, split: str, label: str, protocol: str) -> Path:
    base = root / arm / f"seed{seed}"
    if split == "validation":
        return base / "validation" / f"checkpoint_{label}.json"
    return base / "test" / f"{label}_{protocol}.json"


def _run_job(job: dict[str, Any], gpu: int) -> None:
    output = Path(job["output"])
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_fair_eval_batch.py"),
        "--checkpoint", job["checkpoint"],
        "--tape-dir", job["tape_dir"],
        "--tape-ids", job["tape_ids"],
        "--protocol", job["protocol"],
        "--device", "cuda:0",
        "--output", str(output),
        "--arm", job["arm"],
        "--model-seed", str(job["seed"]),
        "--checkpoint-label", job["checkpoint_label"],
    ]
    environment = dict(os.environ)
    environment.update(
        {"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    )
    with output.with_suffix(".log").open("w", encoding="utf-8") as handle:
        subprocess.run(
            command, cwd=ROOT, env=environment, stdout=handle,
            stderr=subprocess.STDOUT, check=True,
        )
    payload = json.loads(output.read_text(encoding="utf-8"))
    if payload.get("status") != "completed" or not payload.get("kahypar_health", {}).get("pass"):
        raise RuntimeError(f"evaluation batch failed health gate: {output}")


def _run_jobs(jobs: list[dict[str, Any]], gpus: tuple[int, ...], max_workers: int) -> None:
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_job, job, gpus[index % len(gpus)]): job
            for index, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            future.result()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output_root.exists() or args.manifest.exists():
        raise FileExistsError("fair-eval output already exists")
    launch = json.loads(args.launch_manifest.read_text(encoding="utf-8"))
    if launch.get("status") != "launched_unmonitored":
        raise ValueError("unexpected HGNN launch manifest")
    gpus = tuple(int(token.strip()) for token in args.gpus.split(",") if token.strip())
    shared_manifest = _load_shared_manifest(args.tape_dir)
    validation_audit = _validate_split(
        manifest=shared_manifest,
        tape_dir=args.tape_dir,
        split="validation",
        tape_ids=tuple(range(100, 120)),
    )
    args.output_root.mkdir(parents=True)
    state = {
        "schema": "typed_gated_hgnn_three_arm_fair_eval_v1",
        "status": "validation",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "shared_tape_manifest_read": True,
        "test_tape_payload_read_count": 0,
        "validation_tape_audit": validation_audit,
        "test_tape_audit": None,
        "runs": _audit_runs(args.audit_root),
        "version": _version(),
    }
    _write(args.manifest, state)

    validation_jobs = []
    for arm in ARMS:
        for seed in SEEDS:
            for episode in CANDIDATES:
                label = f"ep{episode:04d}"
                validation_jobs.append(
                    {
                        "arm": arm, "seed": seed,
                        "checkpoint": state["runs"][arm][str(seed)]["checkpoints"][str(episode)],
                        "checkpoint_label": label,
                        "tape_dir": str(args.tape_dir), "tape_ids": "100-119",
                        "protocol": "joint",
                        "output": str(_job_output(args.output_root, arm, seed, "validation", label, "joint")),
                    }
                )
    _run_jobs(validation_jobs, gpus, int(args.max_workers))

    selections: dict[str, Any] = {}
    for arm in ARMS:
        selections[arm] = {}
        for seed in SEEDS:
            candidates = []
            for episode in CANDIDATES:
                label = f"ep{episode:04d}"
                path = _job_output(args.output_root, arm, seed, "validation", label, "joint")
                payload = json.loads(path.read_text(encoding="utf-8"))
                score = statistics.fmean(float(row["J_per_offer"]) for row in payload["rows"])
                candidates.append((score, episode, str(payload["checkpoint"]), str(path)))
            score, episode, checkpoint, source = min(candidates, key=lambda item: (item[0], item[1]))
            lock = {
                "status": "locked", "arm": arm, "seed": seed,
                "selected_episode": episode, "selected_checkpoint": checkpoint,
                "joint_validation_J_per_offer_mean": score, "selection_source": source,
                "candidate_sources": [item[3] for item in candidates],
            }
            _write(args.output_root / arm / f"seed{seed}" / "selection_lock.json", lock)
            selections[arm][str(seed)] = lock
    _write(args.output_root / "selection_manifest.json", selections)
    state["status"] = "selection_locked"
    state["checkpoint_selections"] = selections
    _write(args.manifest, state)

    state["test_tape_audit"] = _validate_split(
        manifest=shared_manifest,
        tape_dir=args.tape_dir,
        split="test",
        tape_ids=tuple(range(200, 250)),
    )
    _write(args.manifest, state)
    test_jobs = []
    for arm in ARMS:
        for seed in SEEDS:
            selected = selections[arm][str(seed)]["selected_checkpoint"]
            final = state["runs"][arm][str(seed)]["checkpoints"]["500"]
            for label, checkpoint in (("selected", selected), ("final_ep0500", final)):
                for protocol in ("joint", "forced_hover"):
                    test_jobs.append(
                        {
                            "arm": arm, "seed": seed, "checkpoint": checkpoint,
                            "checkpoint_label": label, "tape_dir": str(args.tape_dir),
                            "tape_ids": "200-249", "protocol": protocol,
                            "output": str(_job_output(args.output_root, arm, seed, "test", label, protocol)),
                        }
                    )
    _run_jobs(test_jobs, gpus, int(args.max_workers))
    state["test_tape_payload_read_count"] = len(test_jobs) * 50
    state["status"] = "completed"
    state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write(args.manifest, state)
    print(json.dumps({"status": "completed", "manifest": str(args.manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
