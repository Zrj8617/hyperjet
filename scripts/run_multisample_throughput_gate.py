from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_COUNTS = (1, 2, 4, 8)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sequential num_envs equivalence/throughput gates."
    )
    parser.add_argument(
        "--num-envs",
        nargs="+",
        type=int,
        default=list(DEFAULT_ENV_COUNTS),
    )
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-steps-per-episode", type=int, default=64)
    parser.add_argument("--rollout-horizon", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "multisample_throughput_gate",
    )
    parser.add_argument("--execute", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _validate_args(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_dir)
    if not root.is_absolute():
        root = ROOT / root
    gate_root = root / f"{timestamp}_multisample_process_background_load"
    gate_root.mkdir(parents=True, exist_ok=False)
    cases = [_build_case(args, gate_root, value) for value in args.num_envs]
    manifest = {
        "schema": "multisample_throughput_gate_v1",
        "label": "multisample_process_background_load",
        "git_commit": _git_commit(),
        "gate_root": str(gate_root),
        "execute": bool(args.execute),
        "num_envs": [int(value) for value in args.num_envs],
        "episodes_total": int(args.episodes),
        "max_steps_per_episode": int(args.max_steps_per_episode),
        "total_environment_slots": int(args.episodes)
        * int(args.max_steps_per_episode),
        "rollout_horizon_per_environment": int(args.rollout_horizon),
        "cases": cases,
    }
    _write_json(gate_root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=True, indent=2))
    if not args.execute:
        return 0

    results: list[dict[str, Any]] = []
    for case in cases:
        result = _run_case(case, gate_root)
        results.append(result)
        _write_json(
            gate_root / "gate_status.json",
            {
                "status": "failed" if result["return_code"] != 0 else "running",
                "completed": results,
            },
        )
        if result["return_code"] != 0:
            return 1
    _write_json(
        gate_root / "gate_status.json",
        {"status": "completed", "completed": results},
    )
    _write_json(gate_root / "throughput_summary.json", _summarize(results))
    return 0


def _build_case(
    args: argparse.Namespace,
    gate_root: Path,
    num_envs: int,
) -> dict[str, Any]:
    cell_id = f"multisample_num_envs_{int(num_envs)}"
    command = [
        sys.executable,
        "scripts/train_clean_mainline.py",
        "--episodes",
        str(int(args.episodes)),
        "--max-steps-per-episode",
        str(int(args.max_steps_per_episode)),
        "--rollout-horizon",
        str(int(args.rollout_horizon)),
        "--num-envs",
        str(int(num_envs)),
        "--sampler-backend",
        "process",
        "--seed",
        str(int(args.seed)),
        "--device",
        str(args.device),
        "--task-encoder",
        "hgnn",
        "--freeze-movement",
        "--completed-dag-weight",
        "16",
        "--ppo-epochs",
        "1",
        "--no-normalize-value-targets",
        "--value-clip-epsilon",
        "0",
        "--offloading-counterfactual-coef",
        "0",
        "--offloading-action-value-loss-coef",
        "0",
        "--offloading-lagged-q-coef",
        "0",
        "--offloading-lagged-q-loss-coef",
        "0",
        "--checkpoint-interval",
        "0",
        "--output-dir",
        str(gate_root / "runs"),
        "--run-name",
        cell_id,
    ]
    return {
        "cell_id": cell_id,
        "label": "multisample_process_background_load",
        "num_envs": int(num_envs),
        "total_environment_slots": int(args.episodes)
        * int(args.max_steps_per_episode),
        "command": command,
        "stdout_log": str(gate_root / "logs" / f"{cell_id}.log"),
    }


def _run_case(case: dict[str, Any], gate_root: Path) -> dict[str, Any]:
    log_path = Path(case["stdout_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    peak_gpu_memory_mib = 0.0
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            case["command"],
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            peak_gpu_memory_mib = max(
                peak_gpu_memory_mib,
                _process_gpu_memory_mib(int(process.pid)),
            )
            time.sleep(0.5)
        return_code = int(process.returncode)
    elapsed = max(time.perf_counter() - started, 1e-9)
    training_result = _read_last_json_object(log_path)
    total_slots = int(
        training_result.get("environment_slots_this_run", training_result.get("global_slot", 0))
        if training_result is not None
        else 0
    )
    return {
        **case,
        "return_code": return_code,
        "elapsed_seconds": float(elapsed),
        "observed_environment_slots": int(total_slots),
        "environment_slots_per_second": float(total_slots / elapsed),
        "peak_gpu_memory_mib": float(peak_gpu_memory_mib),
        "training_result": training_result,
    }


def _process_gpu_memory_mib(pid: int) -> float:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return 0.0
    total = 0.0
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(int(pid)):
            try:
                total += float(fields[1])
            except ValueError:
                pass
    return total


def _read_last_json_object(path: Path) -> dict[str, Any] | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next(
        (
            float(row["environment_slots_per_second"])
            for row in results
            if int(row["num_envs"]) == 1
        ),
        None,
    )
    rows = []
    for result in results:
        throughput = float(result["environment_slots_per_second"])
        rows.append(
            {
                "num_envs": int(result["num_envs"]),
                "elapsed_seconds": float(result["elapsed_seconds"]),
                "environment_slots_per_second": throughput,
                "observed_environment_slots": int(
                    result["observed_environment_slots"]
                ),
                "speedup_vs_num_envs_1": (
                    None
                    if baseline is None or baseline <= 0.0
                    else float(throughput / baseline)
                ),
                "peak_gpu_memory_mib": float(result["peak_gpu_memory_mib"]),
                "return_code": int(result["return_code"]),
            }
        )
    return {
        "label": "multisample_process_background_load",
        "status": (
            "completed"
            if all(int(row["return_code"]) == 0 for row in results)
            else "failed"
        ),
        "results": rows,
    }


def _validate_args(args: argparse.Namespace) -> None:
    values = [int(value) for value in args.num_envs]
    if values != list(DEFAULT_ENV_COUNTS):
        raise ValueError("throughput gate requires --num-envs 1 2 4 8")
    if int(args.episodes) < max(values):
        raise ValueError("episodes must be at least the largest num_envs value")
    if int(args.episodes) <= 0 or int(args.max_steps_per_episode) <= 0:
        raise ValueError("episode and step counts must be positive")
    if int(args.rollout_horizon) <= 0:
        raise ValueError("rollout horizon must be positive")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
