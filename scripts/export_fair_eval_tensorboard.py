"""Export fixed-tape C1/B2/C2 evaluation rows as TensorBoard scalars."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


ARMS = ("C1", "B2", "C2")
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
)


def _summary_writer(log_dir: Path):
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(log_dir=str(log_dir))


def _load_rows(result_path: Path) -> list[dict]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    if data.get("schema") != "c1_b2_c2_fair_reevaluation_result_v1":
        raise ValueError(f"unexpected result schema: {data.get('schema')!r}")
    if data.get("status") != "pass" or data.get("track_a", {}).get("status") != "pass":
        raise ValueError("only a PASS Track A result may be exported")
    rows = data["track_a"]["test_rows"]
    if len(rows) != 1800:
        raise ValueError(f"expected 1800 test rows, got {len(rows)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"output root already exists: {args.output_root}")

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in _load_rows(args.result):
        grouped[(row["arm"], int(row["model_seed"]))].append(row)

    expected_groups = {(arm, seed) for arm in ARMS for seed in range(3)}
    if set(grouped) != expected_groups:
        raise ValueError(f"unexpected arm/seed groups: {sorted(grouped)}")

    manifest_runs = []
    for arm in ARMS:
        for seed in range(3):
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
        "primary_prefix": "forced_hover/budget_selected",
        "secondary_prefix": "joint",
        "excluded": "Track B peak-epoch diagnostics",
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
