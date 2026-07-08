"""Smoke checks for the HyperUAV clean full-experiment launch plan."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import launch_clean_experiments


OLD_ENTRY_TOKENS = (
    "clean_mappo.py",
    "clean_assignment_policy.py",
    "train_clean_assignment_mappo.py",
    "main.py",
    "train.py",
    "tune",
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_args(output_dir: Path) -> argparse.Namespace:
    parser = launch_clean_experiments.build_arg_parser()
    return parser.parse_args(
        [
            "--seeds",
            "0",
            "1",
            "--episodes",
            "500",
            "--max-steps-per-episode",
            "500",
            "--rollout-horizon",
            "128",
            "--eval-episodes",
            "20",
            "--arrival-steps",
            "500",
            "--max-drain-steps",
            "500",
            "--output-dir",
            str(output_dir),
            "--run-prefix",
            "smoke_hypergraph",
        ]
    )


def _check_document() -> None:
    doc_path = ROOT / "docs" / "hyperuav_clean_experiment_launch.md"
    _assert(doc_path.exists(), "launch checklist document should exist.")
    text = doc_path.read_text(encoding="utf-8")
    for needle in (
        "Gate 1",
        "Gate 2",
        "Gate 3",
        "sanity_report.json",
        "overall_pass",
        "train_clean_mainline.py",
        "eval_clean_mainline.py",
        "plot_clean_metrics.py",
        "baseline/ablation",
        "not implemented",
    ):
        _assert(needle in text, f"launch document missing {needle!r}.")


def _check_launcher_source() -> None:
    source = (ROOT / "scripts" / "launch_clean_experiments.py").read_text(encoding="utf-8")
    for token in OLD_ENTRY_TOKENS:
        _assert(token not in source, f"launcher should not reference old entry token {token!r}.")


def _check_plan_schema(plan: dict) -> None:
    _assert(plan["schema"] == "hyperuav_clean_experiment_plan_v1", "unexpected plan schema.")
    _assert(plan["dry_run"] is True, "launcher should default to dry-run.")
    _assert(plan["execute_requested"] is False, "dry-run plan should not request execution.")
    _assert(plan["execution_supported"] is False, "T18 should not support actual execution.")
    _assert(plan["implemented_method"] == "hypergraph_mainline_only", "only hypergraph mainline should be planned.")
    _assert(plan["baseline_ablation_status"] == "not_implemented", "baseline/ablation should be marked future work.")
    _assert(len(plan["gates"]) == 3, "plan should contain three launch gates.")
    _assert(any("Gate 1" in gate["name"] for gate in plan["gates"]), "Gate 1 should be present.")
    _assert(any("Gate 2" in gate["name"] for gate in plan["gates"]), "Gate 2 should be present.")
    _assert(any("Gate 3" in gate["name"] for gate in plan["gates"]), "Gate 3 should be present.")
    _assert(len(plan["experiments"]) == 2, "two seeds should produce two experiment plans.")
    _assert(plan["commands"], "plan should include clean commands.")

    joined_commands = "\n".join(plan["commands"])
    for token in ("train_clean_mainline.py", "eval_clean_mainline.py", "plot_clean_metrics.py"):
        _assert(token in joined_commands, f"plan commands missing {token}.")
    for token in OLD_ENTRY_TOKENS:
        _assert(token not in joined_commands, f"plan commands should not reference {token!r}.")

    required_failure_items = {
        "command",
        "stdout/stderr",
        "traceback",
        "git log -2 --oneline",
        "torch version/cuda availability",
        "config.json",
        "run_summary.json",
        "last 20 lines of train_metrics.jsonl",
        "eval_summary.json",
        "sanity_report.json",
        "tensor shape/device/mask dtype diagnostics",
    }
    _assert(
        required_failure_items.issubset(set(plan["failure_return_package"])),
        "failure return package should include required diagnostics.",
    )


def main() -> None:
    tmp_root = ROOT / ".codex_tmp_clean_experiment_launch"
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    try:
        _check_document()
        _check_launcher_source()

        args = _make_args(tmp_root)
        _assert(args.dry_run is True, "arg parser should default --dry-run to true.")
        _assert(args.execute is False, "arg parser should default --execute to false.")

        run_dir = launch_clean_experiments.create_launch_run_dir(args.output_dir, args.run_prefix)
        plan = launch_clean_experiments.build_experiment_plan(args, run_dir)
        plan_path = run_dir / "experiment_plan.json"
        launch_clean_experiments.write_experiment_plan(plan, plan_path)
        _assert(plan_path.exists(), "experiment_plan.json should be written.")
        _check_plan_schema(_read_json(plan_path))

        dry_run_code = launch_clean_experiments.main(
            [
                "--seeds",
                "0",
                "--output-dir",
                str(tmp_root / "main_call"),
                "--run-prefix",
                "main_call",
            ]
        )
        _assert(dry_run_code == 0, "dry-run launcher main should return 0.")

        execute_code = launch_clean_experiments.main(
            [
                "--seeds",
                "0",
                "--output-dir",
                str(tmp_root / "execute_call"),
                "--run-prefix",
                "execute_call",
                "--execute",
            ]
        )
        _assert(execute_code == 2, "T18 --execute should be refused without starting training.")
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)

    print("smoke_clean_experiment_launch passed")


if __name__ == "__main__":
    main()
