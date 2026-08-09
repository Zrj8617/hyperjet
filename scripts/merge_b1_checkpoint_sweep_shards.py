from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.stage1_temperature_analysis import group_static_rows
from environment.stage1_temperature_sampling import file_sha256


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [row[field] for row in rows if row.get(field) is not None]
    return float(statistics.fmean(values)) if values else None


def _difficulty_tiers() -> dict[str, set[int]]:
    return {
        "congested": set(range(0, 7)),
        "middle": set(range(7, 13)),
        "loose": set(range(13, 20)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge B1 checkpoint sweep shards and summarize each update point.")
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    points: list[dict[str, Any]] = []
    for update in (0, 30, 100, 200):
        shard_root = args.shard_root / f"u{update}"
        output = args.output_root / f"u{update}"
        if output.exists():
            raise FileExistsError(f"output exists: {output}")
        output.mkdir(parents=True)
        for child in ("closed_loop", "static_corpus", "static_replay", "analysis"):
            (output / child).mkdir()

        shard_dirs = sorted(path for path in shard_root.iterdir() if (path / "run_manifest.json").is_file())
        if len(shard_dirs) != 60:
            raise AssertionError(f"u{update} shard count {len(shard_dirs)} != 60")
        closed_rows: list[dict[str, Any]] = []
        static_rows: list[dict[str, Any]] = []
        logical_tape_sha256 = None
        checkpoint_metadata: dict[str, Any] = {}
        for shard in shard_dirs:
            manifest = json.loads((shard / "run_manifest.json").read_text(encoding="utf-8"))
            if manifest["phase"] != "sweep" or int(manifest["checkpoint_update"]) != update:
                raise ValueError(f"wrong shard update: {shard}")
            if logical_tape_sha256 is None:
                logical_tape_sha256 = manifest["logical_tape_sha256"]
            elif manifest["logical_tape_sha256"] != logical_tape_sha256:
                raise ValueError("tape mismatch")
            closed_rows.extend(_jsonl(shard / "closed_loop" / "episodes.jsonl"))
            static_rows.extend(_jsonl(shard / "static_corpus" / "records.jsonl"))
            checkpoint_metadata.update(manifest["checkpoint_metadata"])

        closed_rows.sort(key=lambda row: (int(row["training_seed"]), int(row["episode_index"]), int(row["sampling_replicate"])))
        if len(closed_rows) != 180:
            raise AssertionError(f"u{update} closed rows {len(closed_rows)} != 180")
        keys = {(int(row["training_seed"]), int(row["episode_index"]), int(row["sampling_replicate"])) for row in closed_rows}
        if len(keys) != 180:
            raise ValueError(f"u{update} duplicate closed rows")
        static_rows.sort(key=lambda row: (str(row["checkpoint_sha256"]), int(row["evaluation_scenario_seed"]), int(row["sampling_replicate"]), int(row["slot_index"]), str(row["stable_task_id"]), int(row["decision_order"])))

        closed_path = output / "closed_loop" / "episodes.jsonl"
        corpus_path = output / "static_corpus" / "records.jsonl"
        replay_path = output / "static_replay" / "records.jsonl"
        _write_jsonl(closed_path, closed_rows)
        _write_jsonl(corpus_path, static_rows)
        _write_jsonl(replay_path, static_rows)

        static_grouped = group_static_rows(static_rows)
        static_values = list(static_grouped.values())
        point = {
            "completed_update": update,
            "logical_tape_sha256": logical_tape_sha256,
            "closed_loop_rows": len(closed_rows),
            "static_corpus_records": len(static_rows),
            "completed_dag_count": _mean(closed_rows, "completed_dag_count"),
            "avg_uav_queue_length": _mean(closed_rows, "avg_uav_queue_length"),
            "average_dag_flowtime": _mean(closed_rows, "average_dag_flowtime"),
            "deterministic_margin20_accuracy": _mean(static_values, "deterministic_margin20_accuracy"),
            "deterministic_greedy_agreement": _mean(static_values, "deterministic_greedy_agreement"),
            "normalized_entropy": _mean(static_values, "normalized_entropy"),
            "max_action_probability": _mean(static_values, "max_action_probability"),
            "by_difficulty_tier": {},
            "run_dir": str(output),
        }
        for name, tier in _difficulty_tiers().items():
            point["by_difficulty_tier"][name] = _mean([row for row in closed_rows if int(row["episode_index"]) in tier], "completed_dag_count")
        manifest = {
            "schema": "stage1_temperature_run_manifest_v1",
            "phase": "sweep",
            "checkpoint_set": "b1_sweep",
            "checkpoint_update": update,
            "technical_pass": True,
            "closed_loop_rows": len(closed_rows),
            "static_corpus_records": len(static_rows),
            "static_replay_records": len(static_rows),
            "closed_loop_sha256": file_sha256(closed_path),
            "static_corpus_sha256": file_sha256(corpus_path),
            "static_replay_sha256": file_sha256(replay_path),
            "checkpoint_metadata": checkpoint_metadata,
            "logical_tape_sha256": logical_tape_sha256,
            "deployment_temperature_selected": False,
        }
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        points.append(point)

    by_update = {point["completed_update"]: point["completed_dag_count"] for point in points}
    if by_update[30] <= 97 and abs(by_update[30] - by_update[200]) <= 5:
        verdict = "feature_change"
    elif by_update[0] >= 100 and by_update[30] > by_update[100] > by_update[200]:
        verdict = "objective_harmful"
    else:
        verdict = "non_monotonic_needs_diff"
    summary = {
        "schema": "b1_checkpoint_sweep_v1",
        "logical_tape_sha256": points[0].get("logical_tape_sha256"),
        "points": points,
        "verdict": verdict,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps({"points": len(points), "verdict": verdict}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
