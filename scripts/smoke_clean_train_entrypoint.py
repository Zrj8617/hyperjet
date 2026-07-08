from __future__ import annotations

from pathlib import Path
import json
import os
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl_models.mappo.clean_trainer import CleanCheckpointManager
from scripts import train_clean_mainline


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = train_clean_mainline.build_arg_parser()
    args = parser.parse_args(
        [
            "--smoke",
            "--episodes",
            "99",
            "--max-steps-per-episode",
            "99",
            "--rollout-horizon",
            "99",
            "--output-dir",
            str(ROOT / ".codex_tmp_train_entrypoint"),
            "--run-name",
            "entrypoint_smoke",
        ]
    )
    args = train_clean_mainline.apply_smoke_overrides(args)
    _assert(args.episodes <= 2, "smoke mode should cap episodes.")
    _assert(args.max_steps_per_episode <= 30, "smoke mode should cap episode steps.")
    _assert(args.rollout_horizon <= 8, "smoke mode should cap rollout horizon.")

    forbidden = ["clean_" + "mappo", "clean_assignment_" + "policy", "train_clean_assignment_" + "mappo"]
    source = (ROOT / "scripts" / "train_clean_mainline.py").read_text(encoding="utf-8")
    for token in forbidden:
        _assert(token not in source, f"train entrypoint should not reference legacy clean entrypoint token: {token}")

    with _workspace_temp_dir("setup") as tmp_dir:
        args.output_dir = Path(tmp_dir)
        run_dir = train_clean_mainline.create_run_directory(args)
        train_clean_mainline.initialize_run_files(run_dir, args)
        _assert((run_dir / "checkpoints").is_dir(), "run directory should include checkpoints/.")
        _assert((run_dir / "plots").is_dir(), "run directory should include plots/.")
        _assert((run_dir / "config.json").is_file(), "run directory should include config.json.")
        _assert((run_dir / "run_summary.json").is_file(), "run directory should include run_summary.json.")
        config_payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        _assert(config_payload["resume_semantics"].startswith("checkpoint restore"), "config should describe resume semantics.")
        summary_payload = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        _assert(summary_payload["torch_required_for_training"] is True, "summary should say torch is required.")

        manager = CleanCheckpointManager(run_dir / "checkpoints")
        try:
            manager.save(
                modules=None,
                optimizer=None,
                episode=0,
                global_slot=0,
                update_step=0,
                config_snapshot={},
                safe_boundary=False,
            )
        except RuntimeError:
            pass
        except ModuleNotFoundError:
            raise AssertionError("unsafe checkpoint should fail before torch availability is checked.")
        else:
            raise AssertionError("checkpoint manager must reject unsafe boundaries.")

    with _workspace_temp_dir("main") as tmp_dir:
        exit_code = train_clean_mainline.main(
            [
                "--smoke",
                "--episodes",
                "1",
                "--max-steps-per-episode",
                "2",
                "--rollout-horizon",
                "1",
                "--output-dir",
                str(tmp_dir),
                "--run-name",
                "entrypoint_main",
            ]
        )
        if _torch_available():
            _assert(exit_code == 0, "torch environment should run smoke training entrypoint.")
            run_dirs = [path for path in Path(tmp_dir).iterdir() if path.is_dir()]
            _assert(run_dirs, "torch smoke should create a run directory.")
            latest = sorted(run_dirs)[-1]
            _assert((latest / "train_metrics.jsonl").is_file(), "torch smoke should write train_metrics.jsonl.")
            _assert((latest / "checkpoints" / "latest.pt").is_file(), "torch smoke should save latest checkpoint.")
        else:
            _assert(exit_code == 2, "non-torch environment should return clear training-unavailable code.")
            run_dirs = [path for path in Path(tmp_dir).iterdir() if path.is_dir()]
            _assert(run_dirs, "non-torch entrypoint should still create run setup files before failing.")
            latest = sorted(run_dirs)[-1]
            _assert((latest / "config.json").is_file(), "non-torch setup should still write config.json.")
            _assert((latest / "run_summary.json").is_file(), "non-torch setup should still write run_summary.json.")
            print("smoke_clean_train_entrypoint passed; torch training branch skipped")
            return

    print("smoke_clean_train_entrypoint passed")


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


class _workspace_temp_dir:
    def __init__(self, name: str) -> None:
        self.path = ROOT / ".codex_tmp_train_entrypoint" / f"{name}_{os.getpid()}_{np.random.randint(0, 1_000_000)}"

    def __enter__(self) -> str:
        self.path.mkdir(parents=True, exist_ok=True)
        return str(self.path)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
        parent = ROOT / ".codex_tmp_train_entrypoint"
        if parent.exists():
            try:
                if not any(parent.iterdir()):
                    parent.rmdir()
            except PermissionError:
                pass


if __name__ == "__main__":
    main()
