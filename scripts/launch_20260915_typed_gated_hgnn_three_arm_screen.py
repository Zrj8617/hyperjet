from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl_models.hgnn import build_clean_task_encoder
from marl_models.mappo.clean_movement_actor import CleanMovementActor
from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
from marl_models.mappo.clean_ppo import CleanCentralizedCritic, clean_critic_input_dim
import config
from scripts.run_reward_redesign_arm import _parser as build_arm_runner_parser
from scripts.train_clean_mainline import (
    _build_rng_neutral_comparison_task_encoder,
    build_arg_parser as build_train_parser,
    resolved_reward_redesign_flags,
)


ARMS = ("B2", "C1", "C2")
SEEDS = (5, 86, 617)
RUN_PREFIX = "20260915_TYPED_GATED_HGNN"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Queue and launch the frozen typed-gated HGNN screen.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--queue-status", type=Path, required=True)
    parser.add_argument("--mlp-manifest", type=Path, required=True)
    parser.add_argument("--gpus", type=str, required=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    return parser


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.rstrip()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _read_mlp_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "six_arm_realign_production_launch_v1":
        raise ValueError("unexpected MLP launch manifest schema")
    return payload


def _mlp_control_records(manifest: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for row in manifest.get("runs", []):
        arm = str(row["arm"])
        seed = int(row["seed"])
        if arm not in ARMS or seed not in SEEDS:
            continue
        result_path = Path(row["result_path"])
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") != "completed":
                raise RuntimeError(f"MLP control failed: {result_path}")
            result["control_record_source"] = str(result_path)
            records[(arm, seed)] = result
            continue
        config_paths = sorted((result_path.parent / "train").glob("*/config.json"))
        if len(config_paths) != 1:
            raise RuntimeError(f"expected one actual MLP config record for {arm}/{seed}")
        config_payload = json.loads(config_paths[0].read_text(encoding="utf-8"))
        cli = dict(config_payload["cli"])
        records[(arm, seed)] = {
            "run_name": cli["run_name"],
            "actual_parameters": cli,
            "resolved_flags": resolved_reward_redesign_flags(argparse.Namespace(**cli)),
            "control_record_source": str(config_paths[0]),
        }
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if set(records) != expected:
        raise ValueError("MLP manifest does not contain the required B2/C1/C2 controls")
    return records


def _training_args(*, arm: str, seed: int) -> argparse.Namespace:
    args = build_train_parser().parse_args(
        [
            "--episodes", "500",
            "--max-steps-per-episode", "500",
            "--rollout-horizon", "125",
            "--seed", str(seed),
            "--device", "cuda:0",
            "--task-encoder", "typed_gated_hgnn",
            "--rng-neutral-task-encoder-comparison",
            "--enable-kahypar",
            "--reward-redesign-arm", arm,
            "--no-dag-progress-potential-shaping",
        ]
    )
    if arm in {"C1", "C2", "C2A", "C2B"}:
        args.offloading_eft_advantage = arm != "C2B"
        args.movement_position_advantage = arm != "C2A"
    return args


def _runner_default_preflight() -> dict[str, Any]:
    args = build_arm_runner_parser().parse_args(
        [
            "--arm", "B2",
            "--seed", "5",
            "--output-root", "unused",
            "--run-name", "unused",
        ]
    )
    observed = {
        "task_encoder": args.task_encoder,
        "enable_kahypar": args.enable_kahypar,
        "rng_neutral_task_encoder_comparison": args.rng_neutral_task_encoder_comparison,
        "episodes": args.episodes,
        "steps": args.steps,
        "rollout_horizon": args.rollout_horizon,
        "teacher_anneal_total_updates": args.teacher_anneal_total_updates,
    }
    expected = {
        "task_encoder": "mlp",
        "enable_kahypar": False,
        "rng_neutral_task_encoder_comparison": False,
        "episodes": 500,
        "steps": 500,
        "rollout_horizon": 125,
        "teacher_anneal_total_updates": None,
    }
    if observed != expected:
        raise AssertionError(f"MLP runner defaults changed: {observed}")
    return {"status": "pass", "resolved_defaults": observed}


def _validate_controls(
    mlp_results: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    audited: dict[str, dict[str, Any]] = {}
    frozen_fields = {
        "episodes": 500,
        "max_steps_per_episode": 500,
        "rollout_horizon": 125,
        "num_envs": 1,
        "sampler_backend": "synchronous",
        "lr": 3.0e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "ppo_epochs": 1,
        "hidden_dim": 128,
        "task_embedding_dim": 64,
        "normalize_value_targets": True,
        "max_grad_norm": 0.5,
        "checkpoint_interval": 10,
        "dag_progress_potential_shaping": False,
        "completed_dag_weight": 8.0,
        "critic_task_pooling": "mean",
        "teacher_anneal_total_updates": None,
    }
    for arm in ARMS:
        for seed in SEEDS:
            result = mlp_results[(arm, seed)]
            actual = dict(result["actual_parameters"])
            for key, expected in frozen_fields.items():
                if actual.get(key) != expected:
                    raise ValueError(f"MLP control {arm}/{seed} {key} mismatch")
            if str(actual.get("task_encoder")) != "mlp" or bool(actual.get("enable_kahypar", False)):
                raise ValueError(f"MLP control {arm}/{seed} representation mismatch")
            hgnn_args = _training_args(arm=arm, seed=seed)
            hgnn_flags = resolved_reward_redesign_flags(hgnn_args)
            if hgnn_flags != result.get("resolved_flags"):
                raise ValueError(f"resolved reward flags differ for {arm}/{seed}")
            resolved_updates = int(hgnn_args.episodes) * (
                (int(hgnn_args.max_steps_per_episode) + int(hgnn_args.rollout_horizon) - 1)
                // int(hgnn_args.rollout_horizon)
            )
            if arm in {"C1", "C2"} and resolved_updates != 2000:
                raise ValueError(f"resolved teacher clock mismatch for {arm}/{seed}")
            audited[f"{arm}/seed{seed}"] = {
                "mlp_result": str(result.get("run_name")),
                "control_record_source": str(result["control_record_source"]),
                "raw_teacher_anneal_total_updates": actual.get("teacher_anneal_total_updates"),
                "resolved_teacher_updates": resolved_updates if arm in {"C1", "C2"} else None,
                "resolved_flags": hgnn_flags,
            }
    return audited


def _rng_preflight() -> dict[str, Any]:
    import torch

    task_feature_dim = 16
    for seed in SEEDS:
        torch.manual_seed(seed)
        mlp_offloading = CleanOffloadingActor(task_embedding_dim=64, hidden_dim=128)
        build_clean_task_encoder(
            encoder_type="mlp",
            task_feature_dim=task_feature_dim,
            hidden_dim=128,
            output_dim=64,
        )
        mlp_movement = CleanMovementActor(task_embedding_dim=64, hidden_dim=128)
        mlp_critic = CleanCentralizedCritic(
            input_dim=clean_critic_input_dim(64, config.NUM_UAVS, task_pooling="mean"),
            hidden_dim=128,
            task_pooling="mean",
        )
        expected_state = torch.get_rng_state().clone()

        torch.manual_seed(seed)
        typed_offloading = CleanOffloadingActor(task_embedding_dim=64, hidden_dim=128)
        typed = _build_rng_neutral_comparison_task_encoder(
            builder=build_clean_task_encoder,
            torch=torch,
            encoder_type="typed_gated_hgnn",
            task_feature_dim=task_feature_dim,
            hidden_dim=128,
            output_dim=64,
            training_seed=seed,
            enabled=True,
        )
        typed_movement = CleanMovementActor(task_embedding_dim=64, hidden_dim=128)
        typed_critic = CleanCentralizedCritic(
            input_dim=clean_critic_input_dim(64, config.NUM_UAVS, task_pooling="mean"),
            hidden_dim=128,
            task_pooling="mean",
        )
        if not torch.equal(torch.get_rng_state(), expected_state):
            raise AssertionError(f"global Torch RNG mismatch for seed {seed}")
        for label, mlp_module, typed_module in (
            ("offloading_actor", mlp_offloading, typed_offloading),
            ("movement_actor", mlp_movement, typed_movement),
            ("critic", mlp_critic, typed_critic),
        ):
            mlp_state = mlp_module.state_dict()
            typed_state = typed_module.state_dict()
            if mlp_state.keys() != typed_state.keys() or any(
                not torch.equal(mlp_state[key], typed_state[key]) for key in mlp_state
            ):
                raise AssertionError(f"shared-module initialization mismatch: {label}/seed{seed}")
        for parameter in typed.parameters():
            if not bool(torch.isfinite(parameter).all().item()):
                raise AssertionError(f"non-finite typed encoder initialization for seed {seed}")
    return {
        "status": "pass",
        "seeds": list(SEEDS),
        "typed_encoder_seed": "training_seed",
        "shared_modules_equal": ["offloading_actor", "movement_actor", "critic"],
        "global_torch_rng_equal": True,
    }


def _gpu_memory_snapshot(gpus: tuple[int, ...]) -> list[dict[str, int]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    requested = set(gpus)
    rows = []
    for line in completed.stdout.splitlines():
        index, free_mib, total_mib = (int(token.strip()) for token in line.split(","))
        if index in requested:
            rows.append({"gpu": index, "free_mib": free_mib, "total_mib": total_mib})
    if {row["gpu"] for row in rows} != requested:
        raise RuntimeError("nvidia-smi did not report every requested GPU")
    return sorted(rows, key=lambda row: row["gpu"])


def _kahypar_preflight() -> dict[str, str]:
    import kahypar

    version = str(getattr(kahypar, "__version__", "unknown"))
    if version != "1.3.7":
        raise RuntimeError(f"KaHyPar 1.3.7 required, found {version}")
    return {"status": "pass", "version": version, "module": str(kahypar.__file__)}


def _version() -> dict[str, Any]:
    files = {
        "launcher": Path(__file__).resolve(),
        "runner": ROOT / "scripts" / "run_reward_redesign_arm.py",
        "trainer": ROOT / "scripts" / "train_clean_mainline.py",
        "hgnn": ROOT / "marl_models" / "hgnn" / "clean_incidence.py",
        "clean_trainer": ROOT / "marl_models" / "mappo" / "clean_trainer.py",
        "reward_ledger": ROOT / "environment" / "reward_redesign.py",
        "evaluator": ROOT / "scripts" / "run_fair_eval_batch.py",
        "eval_core": ROOT / "scripts" / "eval_clean_mainline.py",
        "orchestrator": ROOT / "scripts" / "orchestrate_20260915_typed_gated_hgnn_three_arm_screen.py",
        "summarizer": ROOT / "scripts" / "summarize_20260915_typed_gated_hgnn_three_arm_screen.py",
    }
    return {
        "head": _git("rev-parse", "HEAD"),
        "dirty": _git("status", "--porcelain").splitlines(),
        "git_objects": {name: _git("hash-object", str(path)) for name, path in files.items()},
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = args.output_root.resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f"existing output root required: {output_root}")
    for path in (args.manifest,):
        if path.exists():
            raise FileExistsError(path)
    gpus = tuple(int(token.strip()) for token in str(args.gpus).split(",") if token.strip())
    if len(gpus) != 7 or len(set(gpus)) != 7:
        raise ValueError("exactly seven unique GPU ids are required")

    mlp_results = _mlp_control_records(_read_mlp_manifest(args.mlp_manifest))
    version = _version()
    if version["dirty"]:
        raise RuntimeError("formal launch requires a clean server worktree")
    controls_audit = _validate_controls(mlp_results)
    runner_default_audit = _runner_default_preflight()
    rng_audit = _rng_preflight()
    kahypar_audit = _kahypar_preflight()

    targets: list[dict[str, Any]] = []
    index = 0
    for arm in ARMS:
        for seed in SEEDS:
            run_name = f"{RUN_PREFIX}_{arm}_seed{seed}"
            run_root = output_root / run_name
            log_path = output_root / f"{run_name}.log"
            if run_root.exists() or log_path.exists():
                raise FileExistsError(f"refusing to overwrite {run_name}")
            targets.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "run_name": run_name,
                    "run_root": run_root,
                    "log_path": log_path,
                }
            )
            index += 1

    runs: list[dict[str, Any]] = []
    processes = []
    pre_launch_gpu_memory = _gpu_memory_snapshot(gpus)
    extra_gpu_memory = None
    extra_gpus: tuple[int, ...] = ()
    launched_at = datetime.now(timezone.utc)
    for index, target in enumerate(targets):
        if index < len(gpus):
            gpu = gpus[index]
        else:
            if extra_gpu_memory is None:
                time.sleep(10)
                extra_gpu_memory = _gpu_memory_snapshot(gpus)
                extra_gpus = tuple(
                    row["gpu"]
                    for row in sorted(
                        extra_gpu_memory,
                        key=lambda row: (-row["free_mib"], row["gpu"]),
                    )[:2]
                )
            gpu = extra_gpus[index - len(gpus)]
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_reward_redesign_arm.py"),
            "--arm", str(target["arm"]),
            "--seed", str(target["seed"]),
            "--episodes", "500",
            "--steps", "500",
            "--rollout-horizon", "125",
            "--device", "cuda:0",
            "--task-encoder", "typed_gated_hgnn",
            "--enable-kahypar",
            "--rng-neutral-task-encoder-comparison",
            "--skip-eval",
            "--output-root", str(target["run_root"]),
            "--run-name", str(target["run_name"]),
        ]
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with target["log_path"].open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        processes.append(process)
        runs.append(
            {
                "arm": target["arm"],
                "seed": target["seed"],
                "gpu": gpu,
                "run_name": target["run_name"],
                "run_root": str(target["run_root"]),
                "log_path": str(target["log_path"]),
                "result_path": str(target["run_root"] / "result.json"),
                "pid": int(process.pid),
                "command": command,
            }
        )

    time.sleep(3)
    exited = [int(process.pid) for process in processes if process.poll() is not None]
    if exited:
        raise RuntimeError(f"formal run exited during one-time launch check: {exited}")

    manifest = {
        "schema": "typed_gated_hgnn_three_arm_launch_v1",
        "status": "launched_unmonitored",
        "launched_at_utc": launched_at.isoformat(),
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "task_encoder": "typed_gated_hgnn",
        "enable_kahypar": True,
        "episodes": 500,
        "steps": 500,
        "rollout_horizon": 125,
        "mlp_controls_audit": controls_audit,
        "runner_default_preflight": runner_default_audit,
        "rng_preflight": rng_audit,
        "kahypar_preflight": kahypar_audit,
        "gpu_memory_before_launch": pre_launch_gpu_memory,
        "gpu_memory_before_extra_runs": extra_gpu_memory,
        "version": version,
        "runs": runs,
    }
    _write(args.manifest, manifest)
    _write(
        args.queue_status,
        {
            "schema": "typed_gated_hgnn_queue_v1",
            "status": "launched_unmonitored",
            "launched_at_utc": launched_at.isoformat(),
            "manifest": str(args.manifest),
            "pids": [int(row["pid"]) for row in runs],
        },
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
