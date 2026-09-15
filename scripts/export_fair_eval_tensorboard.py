"""Export parameterized fixed-tape evaluation rows as TensorBoard scalars."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


PROTOCOLS = ("forced_hover", "joint")
CHECKPOINT_LABELS = ("budget_selected", "final_ep0500")
METRICS = (
    "J_episode",
    "J_per_offer",
    "J_delay_component",
    "J_task_energy_component",
    "J_move_energy_component",
    "J_per_offer_delay_component",
    "J_per_offer_task_energy_component",
    "J_per_offer_move_energy_component",
    "N_offer",
    "N_admitted",
    "N_completed",
    "admission_rate",
    "conditional_completion_rate",
    "end_to_end_completion_rate",
    "completed_DAG_flowtime_mean",
    "completed_DAG_flowtime_median",
    "throughput",
    "task_energy_joules_total",
    "move_energy_joules_total",
    "energy_per_completed_DAG",
    "hover_action_ratio",
)


def _csv_strings(value: str) -> tuple[str, ...]:
    values = tuple(token.strip().upper() for token in value.split(",") if token.strip())
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("values must be non-empty and unique")
    return values


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(token.strip()) for token in value.split(",") if token.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("values must be comma-separated integers") from exc
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("values must be non-empty and unique")
    return values


def _summary_writer(log_dir: Path):
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(log_dir=str(log_dir))


def _load_rows(result_path: Path, *, expected_count: int) -> list[dict]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    if data.get("schema") != "six_arm_fair_evaluation_result_v1":
        raise ValueError(f"unexpected result schema: {data.get('schema')!r}")
    if data.get("status") != "pass":
        raise ValueError("only a PASS six-arm result may be exported")
    rows = data["test_rows"]
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} test rows, got {len(rows)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arms", type=_csv_strings, required=True)
    parser.add_argument("--seeds", type=_csv_ints, required=True)
    args = parser.parse_args()
    arms = tuple(args.arms)
    seeds = tuple(args.seeds)

    if args.output_root.exists():
        raise FileExistsError(f"output root already exists: {args.output_root}")

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    expected_row_count = (
        len(arms) * len(seeds) * len(PROTOCOLS) * len(CHECKPOINT_LABELS) * 50
    )
    for row in _load_rows(args.result, expected_count=expected_row_count):
        grouped[(row["arm"], int(row["model_seed"]))].append(row)

    expected_groups = {(arm, seed) for arm in arms for seed in seeds}
    if set(grouped) != expected_groups:
        raise ValueError(f"unexpected arm/seed groups: {sorted(grouped)}")

    manifest_runs = []
    for arm in arms:
        for seed in seeds:
            rows = sorted(
                grouped[(arm, seed)],
                key=lambda row: (
                    row["protocol"],
                    row["checkpoint_label"],
                    int(row["tape_id"]),
                ),
            )
            if len(rows) != 200:
                raise ValueError(f"{arm}/seed{seed}: expected 200 rows, got {len(rows)}")

            combinations: dict[tuple[str, str], set[int]] = defaultdict(set)
            for row in rows:
                combinations[(row["protocol"], row["checkpoint_label"])].add(
                    int(row["tape_id"])
                )
            expected_combinations = {
                (protocol, checkpoint)
                for protocol in PROTOCOLS
                for checkpoint in CHECKPOINT_LABELS
            }
            if set(combinations) != expected_combinations:
                raise ValueError(f"{arm}/seed{seed}: unexpected protocol/checkpoint combinations")
            for combination, tape_ids in combinations.items():
                if tape_ids != set(range(200, 250)):
                    raise ValueError(
                        f"{arm}/seed{seed}/{combination}: expected test tapes 200-249"
                    )

            run_dir = args.output_root / arm / f"seed{seed}"
            writer = _summary_writer(run_dir)
            for row in rows:
                prefix = f"{row['protocol']}/{row['checkpoint_label']}"
                tape_id = int(row["tape_id"])
                for metric in METRICS:
                    writer.add_scalar(f"{prefix}/{metric}", float(row[metric]), tape_id)
            writer.close()
            manifest_runs.append(
                {
                    "run": f"FormalEval/{arm}/seed{seed}",
                    "directory": str(run_dir),
                    "row_count": len(rows),
                    "scalar_tag_count": len(PROTOCOLS) * len(CHECKPOINT_LABELS) * len(METRICS),
                    "points_per_tag": 50,
                }
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "fair_eval_tensorboard_export_v1",
        "source_result": str(args.result),
        "step_semantics": "independent test tape id (200-249)",
        "primary_prefix": "joint/budget_selected",
        "secondary_prefix": "forced_hover",
        "arms_order": list(arms),
        "seeds": list(seeds),
        "metrics": list(METRICS),
        "runs": manifest_runs,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
