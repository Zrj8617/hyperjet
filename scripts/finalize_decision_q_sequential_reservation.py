"""Wait for the three diagnostic shards, then combine and summarize them."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


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
    parser.add_argument("--order0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "finalizer_status.json"
    while any(_alive(pid) for pid in args.pids):
        status_path.write_text(
            json.dumps({"status": "waiting", "pids": args.pids}, indent=2),
            encoding="utf-8",
        )
        time.sleep(30)
    missing = [str(path) for path in args.shards if not path.is_file()]
    if missing:
        status_path.write_text(
            json.dumps({"status": "failed", "missing": missing}, indent=2),
            encoding="utf-8",
        )
        return 1
    combined = args.output / "sequential_multi_root_decisions.jsonl"
    combined.write_text(
        "".join(path.read_text(encoding="utf-8") for path in args.shards),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/diagnose_decision_q_v2_ranking_crn.py",
            "--mode",
            "multi-root-summary",
            "--source-decisions",
            str(combined),
            "--output",
            str(args.output),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_decision_q_sequential_reservation.py",
            "--order0",
            str(args.order0),
            "--order-positive",
            str(combined),
            "--output",
            str(args.output / "order0_vs_order_positive.json"),
        ],
        check=True,
    )
    status_path.write_text(
        json.dumps({"status": "complete", "combined": str(combined)}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
