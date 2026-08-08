"""Export clean mainline train_metrics.jsonl to TensorBoard event files.

Usage:
    python scripts/export_clean_metrics_tensorboard.py --run-dir logs/clean_mainline/<run_dir>
    python scripts/export_clean_metrics_tensorboard.py --run-dir logs/clean_mainline/<run_dir> --output-dir logs/tensorboard

Requires either `tensorboardX` or `tensorboard` to be installed (both write
TF event files that TensorBoard can read). The script picks tensorboardX
first, then falls back to tensorboard's EventFileWriter.

Viewing:
    tensorboard --logdir logs/tensorboard --port 6006
    # then open http://localhost:6006 in a browser.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any


TOP_LEVEL_SCALARS = [
    "reward",
    "DAG_completion_rate",
    "average_dag_flowtime",
    "avg_uav_queue_length",
    "hover_action_ratio",
    "mean_uav_displacement_per_slot",
    "ppo_explained_variance",
    "ppo_value_loss",
    "ppo_movement_entropy",
    "ppo_offloading_entropy",
    "ppo_offloading_loss",
    "ppo_movement_loss",
]

# Keys nested under ppo_diagnostics.
NESTED_SCALARS = [
    "rollout_offloading_entropy_normalized_mean",
    "rollout_movement_entropy_normalized_mean",
    "rollout_offloading_valid_candidates_mean",
    "eft_rollout_greedy_agreement",
    "eft_rollout_chosen_raw_regret_mean",
]


def _resolve_value(row: dict[str, Any], key: str) -> float | None:
    if key in NESTED_SCALARS:
        value = row.get("ppo_diagnostics", {}).get(key)
    else:
        value = row.get(key)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _build_writer(output_dir: Path) -> Any:
    try:
        from tensorboardX import SummaryWriter

        return SummaryWriter(str(output_dir))
    except ModuleNotFoundError:
        pass
    try:
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        class _TbWriter:
            def __init__(self, run_dir: Path) -> None:
                self._writer = EventFileWriter(str(run_dir))
                self._event_cls = Event
                self._summary_cls = Summary

            def add_scalar(self, tag: str, value: float, step: int) -> None:
                summary = self._summary_cls(
                    value=[
                        self._summary_cls.Value(tag=str(tag), simple_value=float(value))
                    ]
                )
                self._writer.add_event(
                    self._event_cls(
                        wall_time=datetime.now().timestamp(),
                        step=int(step),
                        summary=summary,
                    )
                )

            def close(self) -> None:
                self._writer.flush()
                self._writer.close()

        return _TbWriter(output_dir)
    except ModuleNotFoundError:
        pass
    raise RuntimeError(
        "Neither tensorboardX nor tensorboard is installed. Install one of them, e.g.:\n"
        "    python -m pip install tensorboardX\n"
        "or\n"
        "    python -m pip install tensorboard"
    )


def _load_rows(metrics_jsonl: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(metrics_jsonl, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export clean mainline train_metrics.jsonl to TensorBoard events."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run directory containing train_metrics.jsonl.")
    parser.add_argument("--output-dir", type=Path, default=Path("logs") / "tensorboard")
    parser.add_argument("--step-key", type=str, default="ppo_update_step", help="JSONL key used as the TensorBoard step (default ppo_update_step).")
    args = parser.parse_args(argv)

    metrics_path = Path(args.run_dir) / "train_metrics.jsonl"
    if not metrics_path.is_file():
        print(f"train_metrics.jsonl not found under {args.run_dir}", file=sys.stderr)
        return 2
    rows = _load_rows(metrics_path)
    if not rows:
        print(f"no rows in {metrics_path}", file=sys.stderr)
        return 2

    run_name = Path(args.run_dir).name
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = _build_writer(output_dir)
    written = 0
    for row_index, row in enumerate(rows):
        step_value = row.get(args.step_key)
        if step_value is None:
            step_value = row_index
        step = int(float(step_value))
        for key in TOP_LEVEL_SCALARS + NESTED_SCALARS:
            value = _resolve_value(row, key)
            if value is None:
                continue
            writer.add_scalar(key, value, step)
            written += 1
    writer.close()

    print(
        f"exported {len(rows)} steps, {written} scalars -> {output_dir} "
        f"(view with: tensorboard --logdir {Path(args.output_dir)} --port 6006)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
