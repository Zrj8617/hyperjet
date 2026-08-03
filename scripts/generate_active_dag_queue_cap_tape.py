from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.capacity_factorial_diagnostic import (
    EPISODE_SLOTS,
    FORMAL_EPISODES,
    FORMAL_SCENARIO_SEEDS,
    LOAD_SLOTS,
    generate_scenario_tape,
    save_scenario_tape_create_only,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the frozen formal capacity-factorial tape.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    tape = generate_scenario_tape(
        scenario_seeds=FORMAL_SCENARIO_SEEDS,
        episodes=FORMAL_EPISODES,
        load_slots=LOAD_SLOTS,
        episode_slots=EPISODE_SLOTS,
    )
    save_scenario_tape_create_only(tape, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "episode_rows": len(tape["episodes"]),
                "full_tape_checksum": tape["full_tape_checksum"],
                "pilot_prefix_checksum": tape["pilot_prefix_checksum"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
