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
        print("SKIP smoke_decision_ppo_bandit_closed_loop: torch unavailable")
        return 0
    with tempfile.TemporaryDirectory(prefix="decision_bandit_eval_smoke_") as temp:
        temp_path = Path(temp)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "train_decision_ppo_bandit_gate.py"),
                "--group",
                "S1-A",
                "--seed",
                "42",
                "--updates",
                "1",
                "--slots-per-update",
                "1",
                "--ppo-epochs",
                "3",
                "--device",
                "cpu",
                "--output-dir",
                str(temp_path),
                "--run-name",
                "eval_smoke",
            ],
            cwd=str(ROOT),
            check=True,
            timeout=180,
        )
        checkpoints = list(temp_path.glob("*/checkpoints/checkpoint_update_0000.pt"))
        assert len(checkpoints) == 1
        output = temp_path / "eval.json"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "eval_decision_ppo_bandit_closed_loop.py"),
                "--checkpoint",
                str(checkpoints[0]),
                "--episodes",
                "1",
                "--max-steps-per-episode",
                "2",
                "--modes",
                "stochastic",
                "deterministic",
                "--device",
                "cpu",
                "--output",
                str(output),
            ],
            cwd=str(ROOT),
            check=True,
            timeout=180,
        )
        text = output.read_text(encoding="utf-8")
        assert '"technical_pass": true' in text
        assert '"pairing_limitation"' in text
    print("PASS smoke_decision_ppo_bandit_closed_loop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
