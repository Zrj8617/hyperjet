"""Build a card-grouped TensorBoard view for C1/B2/C2 comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ARMS = ("C1", "B2", "C2")
TRAIN_RUN_PREFIX = {
    "C1": "20260907_C1",
    "B2": "20260906_B2",
    "C2": "20260908_C2",
}
FORMAL_PREFIX = "01_FormalFixedTape"
TRAIN_PREFIX = "02_TrainDiagnostic"
TRAIN_TAGS = ("train/episode_reward", "train/critic_loss")


def _summary_writer(log_dir: Path):
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(log_dir=str(log_dir))


def _load_scalars(event_dir: Path) -> dict[str, list]:
    accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    return {
        tag: accumulator.Scalars(tag)
        for tag in accumulator.Tags().get("scalars", [])
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"output root already exists: {args.output_root}")

    manifest_runs = []
    for arm in ARMS:
        for seed in range(3):
            formal_dir = args.formal_root / arm / f"seed{seed}"
            train_dir = args.train_root / f"{TRAIN_RUN_PREFIX[arm]}_seed{seed}"
            formal_scalars = {
                tag: events
                for tag, events in _load_scalars(formal_dir).items()
                if tag.startswith(("forced_hover/", "joint/"))
            }
            train_scalars = _load_scalars(train_dir)

            if len(formal_scalars) != 80:
                raise ValueError(
                    f"{arm}/seed{seed}: expected 80 formal tags, "
                    f"got {len(formal_scalars)}"
                )
            missing_train = set(TRAIN_TAGS) - set(train_scalars)
            if missing_train:
                raise ValueError(
                    f"{arm}/seed{seed}: missing training tags {sorted(missing_train)}"
                )

            output_dir = args.output_root / arm / f"seed{seed}"
            writer = _summary_writer(output_dir)
            for tag, events in sorted(formal_scalars.items()):
                if len(events) != 50:
                    raise ValueError(
                        f"{arm}/seed{seed}/{tag}: expected 50 tape points, "
                        f"got {len(events)}"
                    )
                for event in events:
                    writer.add_scalar(
                        f"{FORMAL_PREFIX}/{tag}",
                        event.value,
                        event.step,
                        walltime=event.wall_time,
                    )

            train_counts = {}
            for tag in TRAIN_TAGS:
                events = train_scalars[tag]
                expected = 500 if tag == "train/episode_reward" else 2000
                if len(events) != expected:
                    raise ValueError(
                        f"{arm}/seed{seed}/{tag}: expected {expected} points, "
                        f"got {len(events)}"
                    )
                for event in events:
                    writer.add_scalar(
                        f"{TRAIN_PREFIX}/{tag}",
                        event.value,
                        event.step,
                        walltime=event.wall_time,
                    )
                train_counts[tag] = len(events)
            writer.close()

            manifest_runs.append(
                {
                    "run": f"MethodCompareCards/{arm}/seed{seed}",
                    "formal_tag_count": len(formal_scalars),
                    "formal_points_per_tag": 50,
                    "training_points": train_counts,
                }
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "method_compare_tensorboard_export_v1",
        "card_groups": {
            FORMAL_PREFIX: "fixed-tape metrics used for method ranking",
            TRAIN_PREFIX: "training diagnostics; not a cross-reward ranking metric",
        },
        "runs": manifest_runs,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
