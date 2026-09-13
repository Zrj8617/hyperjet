from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.exogenous_tape import generate_tape, write_tape


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate policy-independent HyperUAV evaluation tapes.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tape-ids", type=str, required=True, help="Comma-separated IDs and inclusive ranges, e.g. 100-119,200-249")
    parser.add_argument("--slots", type=int, default=500)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for tape_id in _parse_ids(args.tape_ids):
        path = args.output_dir / f"tape_{tape_id:04d}.json.gz"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite tape: {path}")
        payload = generate_tape(tape_id=tape_id, slots=int(args.slots))
        write_tape(path, payload)
        rows.append({"tape_id": tape_id, "path": str(path), "slots": int(args.slots), "offer_count": int(payload["offer_count"])})
    manifest = {"schema": "hyperuav_exogenous_tape_manifest_v1", "tapes": rows}
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


def _parse_ids(spec: str) -> list[int]:
    values: list[int] = []
    for part in str(spec).split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start, end = (int(value) for value in token.split("-", 1))
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    if not values or len(values) != len(set(values)):
        raise ValueError("tape IDs must be non-empty and unique")
    return values


if __name__ == "__main__":
    main()
