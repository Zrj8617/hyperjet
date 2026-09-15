"""Build a two-tag, uniform-semantics TensorBoard view for formal training runs."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


TAGS = ("train/episode_reward", "train/critic_loss")


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


def _read_run(inputs: tuple[str, Path]) -> dict[str, Any]:
    run_name, audit_root = inputs
    result_path = audit_root / run_name / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    train_dir = Path(result["train_dir"])
    metrics_path = train_dir / "train_metrics.jsonl"

    episode_points = []
    critic_points = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            step = int(row["global_slot"])
            critic_points.append((step, float(row["ppo_value_loss"])))
            if bool(row.get("episode_terminal_record", False)):
                episode_points.append((step, float(row["episode_reward_total"])))

    if len(episode_points) != 500:
        raise ValueError(f"{run_name}: expected 500 complete episodes, got {len(episode_points)}")
    if len(critic_points) != 2000:
        raise ValueError(f"{run_name}: expected 2000 PPO updates, got {len(critic_points)}")
    if episode_points[0][0] != 500 or episode_points[-1][0] != 250000:
        raise ValueError(f"{run_name}: unexpected episode step range")
    if critic_points[-1][0] != 250000:
        raise ValueError(f"{run_name}: unexpected critic final step")
    return {
        "run": run_name,
        "result_path": str(result_path),
        "train_dir": str(train_dir),
        "metrics_path": str(metrics_path),
        "episode_points": episode_points,
        "critic_points": critic_points,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--arms", type=_csv_strings, required=True)
    parser.add_argument("--seeds", type=_csv_ints, required=True)
    parser.add_argument("--run-prefix", type=str, required=True)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"output root already exists: {args.output_root}")

    run_names = tuple(
        f"{args.run_prefix}_{arm}_seed{seed}"
        for arm in args.arms
        for seed in args.seeds
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        records = list(
            executor.map(_read_run, ((run_name, args.audit_root) for run_name in run_names))
        )

    manifest_runs = []
    for record in records:
        run_dir = args.output_root / record["run"]
        writer = _summary_writer(run_dir)
        for step, value in record["episode_points"]:
            writer.add_scalar("train/episode_reward", value, step)
        for step, value in record["critic_points"]:
            writer.add_scalar("train/critic_loss", value, step)
        writer.close()
        manifest_runs.append(
            {
                "run": f"UnifiedTrain/{record['run']}",
                "result_path": record["result_path"],
                "train_dir": record["train_dir"],
                "metrics_path": record["metrics_path"],
                "tags": {
                    "train/episode_reward": {
                        "source": "episode-terminal episode_reward_total",
                        "point_count": 500,
                        "first_step": 500,
                        "last_step": 250000,
                    },
                    "train/critic_loss": {
                        "source": "per-update ppo_value_loss",
                        "point_count": 2000,
                        "last_step": 250000,
                    },
                },
            }
        )

    manifest = {
        "schema": "unified_train_tensorboard_export_v1",
        "run_count": len(manifest_runs),
        "reward_comparability": (
            "aggregation window is uniform; reward formulas still differ across arms"
        ),
        "arms_order": list(args.arms),
        "seeds": list(args.seeds),
        "run_prefix": str(args.run_prefix),
        "runs": manifest_runs,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
