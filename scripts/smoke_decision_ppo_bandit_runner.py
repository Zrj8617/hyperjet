from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        import torch
    except ModuleNotFoundError:
        print("SKIP smoke_decision_ppo_bandit_runner: torch unavailable")
        return 0
    with tempfile.TemporaryDirectory(prefix="decision_bandit_smoke_") as temp:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train_decision_ppo_bandit_gate.py"),
            "--group",
            "S1-B",
            "--seed",
            "42",
            "--updates",
            "1",
            "--slots-per-update",
            "2",
            "--ppo-epochs",
            "3",
            "--device",
            "cpu",
            "--output-dir",
            temp,
            "--run-name",
            "smoke",
        ]
        subprocess.run(command, cwd=str(ROOT), check=True, timeout=180)
        summaries = list(Path(temp).glob("*/summary.json"))
        assert len(summaries) == 1
        text = summaries[0].read_text(encoding="utf-8")
        assert '"technical_pass": true' in text
        assert '"active_dag_cap_pairing_limitation"' in text
    print("PASS smoke_decision_ppo_bandit_runner")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
