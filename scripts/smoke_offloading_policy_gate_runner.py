from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.offloading_policy_gate import OFFLOADING_POLICIES
from scripts.run_offloading_policy_gate import aggregate_gate_root, build_gate_cells


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    output_root = ROOT / ".codex_tmp_offloading_gate_runner" / f"smoke_{os.getpid()}"
    checkpoints = {42: Path("seed42.pt"), 86: Path("seed86.pt"), 1042: Path("seed1042.pt")}
    cells = build_gate_cells(
        checkpoints=checkpoints,
        environment_seeds=[4242, 4243, 4244, 4245, 4246],
        policies=list(OFFLOADING_POLICIES),
        python=Path(sys.executable),
        output_root=output_root,
        log_root=output_root.parent / "logs",
        arrival_steps=200,
        max_drain_steps=500,
        device="cuda",
    )
    _assert(len(cells) == 60, "gate plan must contain exactly 60 cells")
    _assert(len({cell["cell_id"] for cell in cells}) == 60, "gate cell ids must be unique")
    for cell in cells:
        command = cell["command"]
        _assert(command[command.index("--episodes") + 1] == "1", "each gate cell must run one episode")
        _assert(command[command.index("--arrival-steps") + 1] == "200", "arrival protocol mismatch")
        _assert("--offloading-policy" in command, "gate command must select a policy")
        _assert("--freeze-ue-mobility" not in command, "UE mobility must be inherited from checkpoint")

    try:
        output_root.mkdir(parents=True)
        (output_root / "gate_manifest.json").write_text(
            json.dumps(
                {
                    "cell_count": 4,
                    "cells": [
                        {"cell_id": f"cell_{policy}", "model_seed": 42}
                        for policy in OFFLOADING_POLICIES
                    ],
                }
            ),
            encoding="utf-8",
        )
        for index, policy in enumerate(OFFLOADING_POLICIES):
            run_dir = output_root / "cells" / f"cell_{policy}" / "run"
            run_dir.mkdir(parents=True)
            row = {
                "checkpoint_model_seed": 42,
                "environment_seed": 4242,
                "offloading_policy": policy,
                "arrival_generated_DAG_count": 10.0,
                "arrival_completed_DAG_count": float(index + 1),
                "completed_DAG_count": float(index + 2),
                "DAG_throughput": 0.1 * (index + 1),
                "_dag_flowtime_samples": [10.0 + index, 20.0 + index],
                "arrival_backlog_DAG_count": 3,
                "final_backlog_DAG_count": 1,
                "offloading_action_count": 2,
                "actor_normalized_entropy_mean": 0.9,
                "actor_top1_top2_margin_mean": 0.05,
                "actor_greedy_agreement_count": 1,
                "actor_greedy_comparison_count": 2,
                "_selected_estimated_regret_samples": [0.0, 1.0],
                "_estimator_error_samples": [-1.0, 2.0],
                "realized_cross_uav_transfer_time": 4.0,
                "realized_queue_resource_wait": 5.0,
                "hover_action_ratio": 0.2,
                "mean_uav_displacement_per_slot": 3.0,
                "kahypar_degraded_slot_count": 0,
            }
            (run_dir / "eval_metrics.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (run_dir / "offloading_decisions.jsonl").write_text("{}\n{}\n", encoding="utf-8")
            (run_dir / "eval_summary.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        summary = aggregate_gate_root(output_root)
        _assert(summary["episode_count"] == 4, "aggregate episode count mismatch")
        _assert(summary["status"] == "completed", "all four policies should complete the synthetic aggregate")
        greedy = summary["by_policy"]["greedy_eft_teacher"]
        _assert(greedy["estimator_calibration_count"] == 2, "aggregate calibration count mismatch")
        _assert(greedy["actor_greedy_agreement_rate"] == 0.5, "aggregate agreement mismatch")
        _assert(summary["paired_vs_actor"]["greedy_eft_teacher"]["paired_cell_count"] == 1, "paired comparison count mismatch")
    finally:
        if output_root.exists():
            shutil.rmtree(output_root)
        parent = output_root.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()

    print("smoke_offloading_policy_gate_runner passed")


if __name__ == "__main__":
    main()
