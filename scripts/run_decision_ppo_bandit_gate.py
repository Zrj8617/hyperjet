from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train_decision_ppo_bandit_gate.py"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch matched Stage 1 A/B cells.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 86, 1042])
    parser.add_argument("--groups", nargs="+", choices=["S1-A", "S1-B"], default=["S1-A", "S1-B"])
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--slots-per-update", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("logs") / "decision_ppo_bandit")
    parser.add_argument("--run-name", type=str, default="stage1_formal")
    parser.add_argument("--pilot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows: list[dict[str, Any]] = []
    identities: dict[int, dict[str, Any]] = {}
    for seed in args.seeds:
        for group in args.groups:
            command = [
                sys.executable,
                str(TRAIN),
                "--group",
                str(group),
                "--seed",
                str(int(seed)),
                "--updates",
                str(int(args.updates)),
                "--slots-per-update",
                str(int(args.slots_per_update)),
                "--ppo-epochs",
                "3",
                "--device",
                str(args.device),
                "--output-dir",
                str(args.output_dir),
                "--run-name",
                str(args.run_name),
            ]
            if bool(args.pilot):
                command.append("--pilot")
            completed = subprocess.run(
                command,
                cwd=str(ROOT),
                check=True,
                text=True,
                capture_output=True,
            )
            output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
            if not output_lines:
                raise RuntimeError(f"{group} seed {seed} produced no summary")
            row = json.loads(output_lines[-1])
            identity = row["initialization_identity"]
            previous = identities.get(int(seed))
            if previous is None:
                identities[int(seed)] = identity
            elif previous != identity:
                raise AssertionError(f"S1-A/S1-B initialization mismatch for seed {seed}")
            rows.append(row)
            print(output_lines[-1], flush=True)
    summary = {
        "schema": "decision_ppo_bandit_stage1_comparison_v1",
        "technical_pass": all(bool(row["technical_pass"]) for row in rows),
        "cell_count": len(rows),
        "rows": rows,
        "pairing_limitation": (
            "fixed evaluator seeds are not strict counterfactual pairs because "
            "active_dag_cap creates policy-dependent eligibility and RNG consumption"
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
