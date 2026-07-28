from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts import train_greedy_imitation_gate as gate
except ModuleNotFoundError as exc:
    if exc.name == "numpy":
        print("smoke_greedy_imitation_split skipped: numpy is not installed")
        raise SystemExit(0) from exc
    raise


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    split_bounds = gate._split_bounds(episodes=10, train_fraction=0.6, val_fraction=0.2)
    _assert(split_bounds == {"train": (0, 6), "val": (6, 8), "test": (8, 10)}, "unexpected split bounds.")
    episode_sets = gate._split_episode_sets(split_bounds)
    _assert(not episode_sets["train"].intersection(episode_sets["val"]), "train/val episodes leaked.")
    _assert(not episode_sets["train"].intersection(episode_sets["test"]), "train/test episodes leaked.")
    _assert(not episode_sets["val"].intersection(episode_sets["test"]), "val/test episodes leaked.")
    _assert(gate._split_leakage_count(split_bounds) == 0, "split leakage count should be zero.")

    train_summary = {
        "epoch_0": {
            "train/overall": {
                "decision_count": 123,
            }
        }
    }
    eval_summary = {
        "val/overall": {
            "decision_count": 45,
        },
        "test/overall": {
            "decision_count": 67,
        },
    }
    summary = gate._build_split_summary(
        split_bounds=split_bounds,
        train_summary=train_summary,
        eval_summary=eval_summary,
        supervised_epochs=1,
    )
    _assert(summary["split_mode"] == "episode_level", "split mode must be episode_level.")
    _assert(summary["leakage_count"] == 0, "summary leakage_count must be zero.")
    _assert(summary["train"]["episodes"] == [0, 1, 2, 3, 4, 5], "train episodes mismatch.")
    _assert(summary["val"]["episodes"] == [6, 7], "val episodes mismatch.")
    _assert(summary["test"]["episodes"] == [8, 9], "test episodes mismatch.")
    _assert(summary["train"]["sample_count"] == 123, "train sample count mismatch.")
    _assert(summary["val"]["sample_count"] == 45, "val sample count mismatch.")
    _assert(summary["test"]["sample_count"] == 67, "test sample count mismatch.")

    trajectory_ids_by_split: dict[str, set[str]] = {}
    for split, episodes in episode_sets.items():
        trajectory_ids_by_split[split] = {
            gate._trajectory_id(
                trajectory_policy=policy,
                environment_seed=42,
                episode=episode,
            )
            for policy in gate.TRAJECTORY_POLICIES
            for episode in episodes
        }
    _assert(
        not trajectory_ids_by_split["train"].intersection(trajectory_ids_by_split["val"]),
        "train/val trajectory ids leaked.",
    )
    _assert(
        not trajectory_ids_by_split["train"].intersection(trajectory_ids_by_split["test"]),
        "train/test trajectory ids leaked.",
    )
    _assert(
        not trajectory_ids_by_split["val"].intersection(trajectory_ids_by_split["test"]),
        "val/test trajectory ids leaked.",
    )

    for forbidden in (
        "clean_mappo",
        "clean_assignment_policy",
        "train_clean_assignment_mappo",
    ):
        _assert(forbidden not in sys.modules, f"smoke must not import legacy module {forbidden}.")

    print("smoke_greedy_imitation_split passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
