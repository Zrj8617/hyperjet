from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.capacity_factorial_diagnostic import (
    FORMAL_SCENARIO_SEEDS,
    PILOT_EPISODES,
    analyze_factorial_rows,
    load_scenario_tape,
)
from scripts.run_active_dag_queue_cap_factorial import POLICIES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze strict paired 2x2 capacity-factorial rows.")
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--tape", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=PILOT_EPISODES)
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line.strip()]
    tape = load_scenario_tape(args.tape)
    result = analyze_factorial_rows(
        rows,
        policies=POLICIES,
        scenario_seeds=FORMAL_SCENARIO_SEEDS,
        episode_count=int(args.episodes),
        expected_full_tape_checksum=tape["full_tape_checksum"],
        expected_prefix_checksum=tape["pilot_prefix_checksum"],
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"technical_pass": result["technical_pass"], "gate_errors": result["gate_errors"]}, sort_keys=True))
    return 0 if result["technical_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
