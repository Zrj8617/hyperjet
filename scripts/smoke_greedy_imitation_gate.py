from __future__ import annotations

from pathlib import Path
import json
import os
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts import train_greedy_imitation_gate
except ModuleNotFoundError as exc:
    if exc.name in {"numpy", "torch"}:
        print(f"smoke_greedy_imitation_gate skipped: {exc.name} is not installed")
        raise SystemExit(0) from exc
    raise


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    parser = train_greedy_imitation_gate.build_arg_parser()
    args = parser.parse_args(
        [
            "--smoke",
            "--episodes",
            "4",
            "--max-steps-per-episode",
            "12",
            "--supervised-epochs",
            "1",
            "--closed-loop-eval-episodes",
            "2",
            "--task-encoder",
            "mlp",
            "--trajectory-policies",
            "greedy_eft",
            "random_hash",
            "--output-dir",
            str(ROOT / ".codex_tmp_greedy_imitation_gate"),
            "--run-name",
            "smoke",
        ]
    )
    args = train_greedy_imitation_gate._apply_smoke_overrides(args)
    _assert(args.episodes <= 4, "smoke should cap episodes")
    _assert(args.max_steps_per_episode <= 12, "smoke should cap episode steps")
    _assert(args.dag_base_arrival_prob == 1.0, "smoke should force task arrivals")

    with _workspace_temp_dir("run") as tmp_dir:
        exit_code = train_greedy_imitation_gate.main(
            [
                "--smoke",
                "--episodes",
                "4",
                "--max-steps-per-episode",
                "12",
                "--supervised-epochs",
                "1",
                "--closed-loop-eval-episodes",
                "2",
                "--task-encoder",
                "mlp",
                "--trajectory-policies",
                "greedy_eft",
                "random_hash",
                "--output-dir",
                str(tmp_dir),
                "--run-name",
                "smoke_main",
            ]
        )
        if not _torch_available():
            _assert(exit_code == 2, "non-torch environment should return unavailable code")
            print("smoke_greedy_imitation_gate skipped: torch is not installed")
            return 0
        _assert(exit_code == 0, "torch environment should complete smoke gate")
        run_dirs = [path for path in Path(tmp_dir).iterdir() if path.is_dir()]
        _assert(run_dirs, "smoke should create a run directory")
        latest = sorted(run_dirs)[-1]
        required = [
            latest / "config.json",
            latest / "run_summary.json",
            latest / "train_metrics.jsonl",
            latest / "decision_samples.jsonl",
            latest / "imitation_split_summary.json",
            latest / "closed_loop_eval_metrics.jsonl",
            latest / "closed_loop_eval_summary.json",
            latest / "checkpoints" / "imitation_model.pt",
        ]
        for path in required:
            _assert(path.is_file(), f"missing smoke artifact: {path.name}")
        summary = json.loads((latest / "run_summary.json").read_text(encoding="utf-8"))
        _assert(summary["status"] == "completed", "summary should mark completed")
        sample = json.loads((latest / "decision_samples.jsonl").read_text(encoding="utf-8").splitlines()[0])
        for key in (
            "task_features",
            "incidence_matrix",
            "hyperedge_type_ids",
            "task_id_to_idx",
            "idx_to_task_id",
            "active_task_ids",
            "ready_task_ids",
            "pending_task_ids",
            "task_local_index",
            "candidate_mask",
            "candidate_uav_ids",
            "greedy_label_idx",
            "estimated_finish_times",
            "valid_candidate_count",
            "trajectory_policy",
            "label_mode",
        ):
            _assert(key in sample, f"decision sample missing {key}")
        _assert(
            len(sample["hyperedge_type_ids"]) == len(sample["incidence_matrix"][0]) if sample["incidence_matrix"] else len(sample["hyperedge_type_ids"]) == 0,
            "hyperedge_type_ids must align with incidence columns",
        )
        split_summary = json.loads((latest / "imitation_split_summary.json").read_text(encoding="utf-8"))
        _assert(split_summary["split_mode"] == "episode_level", "split summary should use episode-level split")
        _assert(split_summary["leakage_count"] == 0, "split summary should report zero leakage")
    print("smoke_greedy_imitation_gate passed")
    return 0


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


class _workspace_temp_dir:
    def __init__(self, name: str) -> None:
        self.path = ROOT / ".codex_tmp_greedy_imitation_gate" / f"{name}_{os.getpid()}_{random.randint(0, 1_000_000)}"

    def __enter__(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
        parent = ROOT / ".codex_tmp_greedy_imitation_gate"
        if parent.exists():
            try:
                if not any(parent.iterdir()):
                    parent.rmdir()
            except PermissionError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
