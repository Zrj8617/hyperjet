"""Combine completed same-slot bootstrap calibration shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_decision_q_same_slot_bootstrap import _summary


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pids", type=int, nargs=3, required=True)
    parser.add_argument("--shards", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status = args.output / "finalizer_status.json"
    while any(_alive(pid) for pid in args.pids):
        status.write_text(
            json.dumps({"status": "waiting", "pids": args.pids}, indent=2),
            encoding="utf-8",
        )
        time.sleep(30)
    missing = [str(path) for path in args.shards if not path.is_file()]
    if missing:
        status.write_text(
            json.dumps({"status": "failed", "missing": missing}, indent=2),
            encoding="utf-8",
        )
        return 1
    rows = [
        json.loads(line)
        for path in args.shards
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    decisions = args.output / "same_slot_bootstrap_decisions.jsonl"
    summary = args.output / "same_slot_bootstrap_summary.json"
    decisions.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary.write_text(json.dumps(_summary(rows), indent=2, sort_keys=True), encoding="utf-8")
    status.write_text(
        json.dumps(
            {
                "status": "complete",
                "decision_count": len(rows),
                "decisions": str(decisions),
                "summary": str(summary),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
