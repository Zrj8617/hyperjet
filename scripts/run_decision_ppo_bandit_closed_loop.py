from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts" / "eval_decision_ppo_bandit_closed_loop.py"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate all Stage 1 gate checkpoints.")
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--groups", nargs="+", choices=["S1-A", "S1-B"], default=["S1-A", "S1-B"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 86, 1042])
    parser.add_argument("--checkpoint-updates", type=str, default="0,1,5,10,20,30")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=424242)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    updates = [int(value.strip()) for value in args.checkpoint_updates.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for group in args.groups:
            run_dir = _find_unique_run(
                args.training_root,
                run_name=str(args.run_name),
                group=str(group),
                seed=int(seed),
            )
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            if not bool(summary.get("technical_pass")):
                raise RuntimeError(f"training cell did not technically pass: {run_dir}")
            for update in updates:
                checkpoint = run_dir / "checkpoints" / f"checkpoint_update_{int(update):04d}.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
                output = (
                    args.output_dir
                    / f"{group}_seed{int(seed)}_update{int(update):04d}.json"
                )
                command = [
                    sys.executable,
                    str(EVALUATOR),
                    "--checkpoint",
                    str(checkpoint),
                    "--episodes",
                    str(int(args.episodes)),
                    "--eval-seed",
                    str(int(args.eval_seed)),
                    "--modes",
                    "stochastic",
                    "deterministic",
                    "--device",
                    str(args.device),
                    "--output",
                    str(output),
                ]
                subprocess.run(command, cwd=str(ROOT), check=True)
                payload = json.loads(output.read_text(encoding="utf-8"))
                if not bool(payload.get("technical_pass")):
                    raise RuntimeError(f"closed-loop cell failed: {output}")
                rows.append(
                    {
                        "group": str(group),
                        "seed": int(seed),
                        "completed_update": int(update),
                        "training_run_dir": str(run_dir),
                        "checkpoint": str(checkpoint),
                        "result_file": str(output),
                        "summary": payload["summary"],
                    }
                )
                print(
                    f"completed {group} seed={seed} update={update}",
                    flush=True,
                )
    aggregate = {
        "schema": "decision_ppo_bandit_stage1_closed_loop_comparison_v1",
        "technical_pass": True,
        "cell_count": len(rows),
        "episodes_per_mode": int(args.episodes),
        "eval_seed": int(args.eval_seed),
        "rows": rows,
        "pairing_limitation": (
            "same eval seeds are not strict counterfactual pairs because active_dag_cap "
            "makes eligibility and RNG consumption policy-dependent"
        ),
    }
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"technical_pass": True, "cell_count": len(rows)}, sort_keys=True))
    return 0


def _find_unique_run(root: Path, *, run_name: str, group: str, seed: int) -> Path:
    matches = sorted(root.glob(f"*_{run_name}_{group}_seed{int(seed)}"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one run for {group} seed {seed}, found {len(matches)}"
        )
    return matches[0]


if __name__ == "__main__":
    raise SystemExit(main())
