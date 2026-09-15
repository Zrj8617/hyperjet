from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.exogenous_tape import generate_tape, write_tape


def scene_parameters() -> dict[str, object]:
    return {
        "area_width": int(config.AREA_WIDTH),
        "area_height": int(config.AREA_HEIGHT),
        "num_uavs": int(config.NUM_UAVS),
        "num_ues": int(config.NUM_UES),
        "time_slot_duration": float(config.TIME_SLOT_DURATION),
        "hotspot_radius": float(config.HOTSPOT_RADIUS),
        "dag_base_arrival_probability": float(config.DAG_BASE_ARRIVAL_PROB),
        "dag_hotspot_arrival_multiplier": float(config.DAG_HOTSPOT_ARRIVAL_MULTIPLIER),
        "dag_task_count_range": [int(config.DAG_MIN_TASKS), int(config.DAG_MAX_TASKS)],
        "dag_max_levels": int(config.DAG_MAX_LEVELS),
        "dag_max_parents": int(config.DAG_MAX_PARENTS),
        "input_data_size_mb_range": [float(value) for value in config.INPUT_DATA_SIZE_MB_RANGE],
        "output_data_size_mb_range": [float(value) for value in config.OUTPUT_DATA_SIZE_MB_RANGE],
        "task_constant_range": [int(value) for value in config.TASK_CONSTANT_RANGE],
        "task_complexity_probabilities": dict(config.TASK_COMPLEXITY_PROBS),
        "upload_bandwidth_mbps": [float(value) for value in config.BASE_UPLOAD_BANDWIDTH_MBPS],
        "download_bandwidth_mbps": [float(value) for value in config.BASE_DOWNLOAD_BANDWIDTH_MBPS],
        "bandwidth_level_probabilities": [float(value) for value in config.BANDWIDTH_LEVEL_PROBS],
        "uav_compute_rate_cycles_per_second": float(config.UAV_COMPUTE_RATE_OPS_PER_SEC),
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.rstrip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate policy-independent HyperUAV evaluation tapes.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tape-ids", type=str, required=True, help="Comma-separated IDs and inclusive ranges, e.g. 100-119,200-249")
    parser.add_argument("--slots", type=int, default=500)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene = scene_parameters()
    version = {
        "head": _git("rev-parse", "HEAD"),
        "dirty": _git("status", "--porcelain").splitlines(),
        "generator_git_object": _git("hash-object", str(Path(__file__).resolve())),
        "tape_module_git_object": _git(
            "hash-object", str(ROOT / "environment" / "exogenous_tape.py")
        ),
        "config_git_object": _git("hash-object", str(ROOT / "config.py")),
    }
    rows = []
    for tape_id in _parse_ids(args.tape_ids):
        path = args.output_dir / f"tape_{tape_id:04d}.json.gz"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite tape: {path}")
        payload = generate_tape(tape_id=tape_id, slots=int(args.slots))
        payload["generation_context"] = {
            "scene_parameters": scene,
            "version": version,
        }
        write_tape(path, payload)
        rows.append(
            {
                "tape_id": tape_id,
                "path": str(path.resolve()),
                "slots": int(args.slots),
            }
        )
    manifest = {
        "schema": "hyperuav_exogenous_tape_manifest_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scene_parameters": scene,
        "version": version,
        "splits": {"validation": list(range(100, 120)), "test": list(range(200, 250))},
        "tapes": rows,
    }
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
