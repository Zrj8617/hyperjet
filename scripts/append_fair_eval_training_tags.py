"""Append two original training scalars to the nine FormalEval runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ARMS = ("C1", "B2", "C2")
TRAIN_TAGS = ("train/episode_reward", "train/critic_loss")
EXPECTED_COUNTS = {
    "C1": {"train/episode_reward": 500, "train/critic_loss": 2000},
    "B2": {"train/episode_reward": 2000, "train/critic_loss": 2000},
    "C2": {"train/episode_reward": 500, "train/critic_loss": 2000},
}


def _summary_writer(log_dir: Path):
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(log_dir=str(log_dir), filename_suffix=".training-context")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--formal-eval-root", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    if result.get("schema") != "c1_b2_c2_fair_reevaluation_result_v1":
        raise ValueError(f"unexpected result schema: {result.get('schema')!r}")
    sources = result["track_a"]["run_sources"]

    manifest = {"schema": "fair_eval_training_context_v1", "runs": []}
    for arm in ARMS:
        for seed in range(3):
            target_dir = args.formal_eval_root / arm / f"seed{seed}"
            marker = target_dir / "training_tags_manifest.json"
            if marker.exists():
                manifest["runs"].append(
                    json.loads(marker.read_text(encoding="utf-8"))
                )
                continue

            source_result_path = Path(sources[arm][str(seed)]["result_path"])
            source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
            source_event_dir = Path(source_result["tensorboard"]["directory"])
            accumulator = EventAccumulator(
                str(source_event_dir), size_guidance={"scalars": 0}
            )
            accumulator.Reload()

            writer = _summary_writer(target_dir)
            tag_records = {}
            for tag in TRAIN_TAGS:
                events = accumulator.Scalars(tag)
                expected_count = EXPECTED_COUNTS[arm][tag]
                if len(events) != expected_count:
                    raise ValueError(
                        f"{arm}/seed{seed}/{tag}: expected {expected_count} points, "
                        f"got {len(events)}"
                    )
                for event in events:
                    writer.add_scalar(tag, event.value, event.step, walltime=event.wall_time)
                tag_records[tag] = {
                    "point_count": len(events),
                    "first_step": events[0].step,
                    "last_step": events[-1].step,
                }
            writer.close()

            run_record = {
                "run": f"FormalEval/{arm}/seed{seed}",
                "source_result": str(source_result_path),
                "source_event_dir": str(source_event_dir),
                "tags": tag_records,
            }
            marker.write_text(
                json.dumps(run_record, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            manifest["runs"].append(run_record)

    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
