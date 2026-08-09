from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.stage1_temperature_analysis import replay_static_record
from environment.stage1_temperature_sampling import FROZEN_TEMPERATURES, canonical_sha256, file_sha256


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _closed_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["training_seed"]),
        int(row["episode_index"]),
        int(row["sampling_replicate"]),
        float(row["temperature"]),
    )


def _static_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["checkpoint_sha256"]),
        int(row["evaluation_scenario_seed"]),
        int(row["sampling_replicate"]),
        int(row["slot_index"]),
        str(row["stable_task_id"]),
        int(row["decision_order"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge create-only Stage 1 temperature shard outputs.")
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("output directory is create-only")
    output.mkdir(parents=True)
    for child in ("static_corpus", "static_replay", "closed_loop", "analysis"):
        (output / child).mkdir()

    shard_dirs = sorted(path for path in args.shard_root.resolve().iterdir() if (path / "run_manifest.json").is_file())
    if not shard_dirs:
        raise FileNotFoundError("no shard manifests found")

    closed_rows: list[dict[str, Any]] = []
    static_rows: list[dict[str, Any]] = []
    checkpoint_metadata: dict[str, Any] = {}
    logical_tape_sha256 = None
    checkpoint_set = None
    for shard in shard_dirs:
        manifest = json.loads((shard / "run_manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("technical_pass") or not manifest.get("partial_shard"):
            raise ValueError(f"invalid shard manifest: {shard}")
        if logical_tape_sha256 is None:
            logical_tape_sha256 = manifest["logical_tape_sha256"]
            checkpoint_set = manifest["checkpoint_set"]
        if manifest["logical_tape_sha256"] != logical_tape_sha256 or manifest["checkpoint_set"] != checkpoint_set:
            raise ValueError("shard identity mismatch")
        if file_sha256(shard / "closed_loop" / "episodes.jsonl") != manifest["closed_loop_sha256"]:
            raise ValueError(f"closed-loop checksum mismatch: {shard}")
        if file_sha256(shard / "static_corpus" / "records.jsonl") != manifest["static_corpus_sha256"]:
            raise ValueError(f"static corpus checksum mismatch: {shard}")
        for seed, meta in manifest["checkpoint_metadata"].items():
            if seed in checkpoint_metadata and checkpoint_metadata[seed] != meta:
                raise ValueError(f"checkpoint metadata mismatch for seed {seed}")
            checkpoint_metadata[seed] = meta
        closed_rows.extend(_jsonl(shard / "closed_loop" / "episodes.jsonl"))
        static_rows.extend(_jsonl(shard / "static_corpus" / "records.jsonl"))

    closed_by_key = {_closed_key(row): row for row in closed_rows}
    static_by_key = {_static_key(row): row for row in static_rows}
    if len(closed_by_key) != len(closed_rows):
        raise ValueError("duplicate closed-loop rows")
    if len(static_by_key) != len(static_rows):
        raise ValueError("duplicate static corpus rows")
    expected_closed = 3 * 20 * 5 * len(FROZEN_TEMPERATURES)
    if len(closed_rows) != expected_closed:
        raise AssertionError(f"closed-loop row count {len(closed_rows)} != {expected_closed}")
    expected_closed_keys = {
        (seed, scenario, replicate, temperature)
        for seed in (42, 86, 1042)
        for scenario in range(20)
        for replicate in range(5)
        for temperature in FROZEN_TEMPERATURES
    }
    if set(closed_by_key) != expected_closed_keys:
        raise ValueError("closed-loop key coverage mismatch")
    for row in static_rows:
        if canonical_sha256({key: value for key, value in row.items() if key != "record_sha256"}) != row["record_sha256"]:
            raise ValueError("static record checksum mismatch")

    closed_rows = [closed_by_key[key] for key in sorted(closed_by_key)]
    static_rows = [static_by_key[key] for key in sorted(static_by_key)]
    closed_path = output / "closed_loop" / "episodes.jsonl"
    corpus_path = output / "static_corpus" / "records.jsonl"
    replay_path = output / "static_replay" / "records.jsonl"
    _write_jsonl(closed_path, closed_rows)
    _write_jsonl(corpus_path, static_rows)
    corpus_sha256 = file_sha256(corpus_path)
    with replay_path.open("x", encoding="utf-8", newline="\n") as handle:
        for static in static_rows:
            for replay_temperature in FROZEN_TEMPERATURES:
                replay = replay_static_record(static, replay_temperature)
                replay["source_corpus_sha256"] = corpus_sha256
                handle.write(json.dumps(replay, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")

    summary = {
        "schema": "stage1_temperature_run_manifest_v1",
        "phase": "formal",
        "checkpoint_set": checkpoint_set,
        "technical_pass": True,
        "partial_shard": False,
        "merged_from_shards": [path.name for path in shard_dirs],
        "closed_loop_rows": len(closed_rows),
        "static_corpus_records": len(static_rows),
        "static_replay_records": len(static_rows) * len(FROZEN_TEMPERATURES),
        "static_corpus_sha256": corpus_sha256,
        "static_replay_sha256": file_sha256(replay_path),
        "closed_loop_sha256": file_sha256(closed_path),
        "checkpoint_metadata": checkpoint_metadata,
        "logical_tape_sha256": logical_tape_sha256,
        "pairing_limitation": "matched starts and keyed noise are not strict counterfactual pairs after trajectories diverge",
        "deployment_temperature_selected": False,
    }
    (output / "run_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"closed_loop_rows": len(closed_rows), "static_corpus_records": len(static_rows), "shards": len(shard_dirs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
