from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import greedy_imitation_dataset as dataset
import config


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    with _workspace_temp_dir("determinism") as temp_dir:
        first_dir = temp_dir / "first"
        second_dir = temp_dir / "second"
        with _without_partition():
            first_samples, first_manifest = _generate(first_dir)
            second_samples, second_manifest = _generate(second_dir)
        _assert(
            first_manifest["dataset_checksum"] == second_manifest["dataset_checksum"],
            "same seed/config must produce the same dataset checksum",
        )
        _assert(
            dataset.sample_ids_by_split(first_samples) == dataset.sample_ids_by_split(second_samples),
            "same seed/config must produce the same sample IDs and split",
        )
        _assert(first_manifest["leakage_count"] == 0, "first dataset must have zero leakage")
        _assert(second_manifest["leakage_count"] == 0, "second dataset must have zero leakage")

        loaded_samples, loaded_manifest = dataset.load_frozen_dataset(first_dir)
        _assert(
            dataset.canonical_json(loaded_samples) == dataset.canonical_json(first_samples),
            "dataset load/save round trip changed samples",
        )
        _assert(
            loaded_manifest["dataset_checksum"] == first_manifest["dataset_checksum"],
            "round-trip manifest checksum mismatch",
        )
        if loaded_samples:
            original_value = loaded_samples[0]["task_features"][0][0]
            loaded_samples[0]["task_features"][0][0] = float(original_value) + 123.0
            reloaded_samples, _ = dataset.load_frozen_dataset(first_dir)
            _assert(
                reloaded_samples[0]["task_features"][0][0] == original_value,
                "mutating a loaded historical array must not mutate the frozen dataset on disk",
            )

        manifest_disk = json.loads(
            (first_dir / dataset.MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        for key in (
            "schema",
            "git_commit",
            "creation_time_utc",
            "dataset_seed",
            "episodes",
            "max_steps_per_episode",
            "trajectory_policies",
            "scenario_config_snapshot",
            "task_feature_dimension",
            "pair_feature_dimension",
            "dynamic_uav_feature_dimension",
            "sample_count",
            "episode_count",
            "trajectory_count",
            "train_episode_ids",
            "val_episode_ids",
            "test_episode_ids",
            "train_trajectory_ids",
            "val_trajectory_ids",
            "test_trajectory_ids",
            "train_sample_count",
            "val_sample_count",
            "test_sample_count",
            "leakage_count",
            "dataset_file_path",
            "dataset_checksum",
        ):
            _assert(key in manifest_disk, f"dataset manifest missing {key}")

    with _workspace_temp_dir("zero_samples") as temp_dir:
        with _without_partition():
            zero_samples, zero_manifest = dataset.generate_frozen_dataset(
                dataset_dir=temp_dir / "dataset",
                dataset_seed=7,
                episodes=2,
                max_steps_per_episode=2,
                trajectory_policies=["greedy_eft"],
                train_fraction=0.5,
                val_fraction=0.0,
                dag_base_arrival_prob=0.0,
            )
        _assert(zero_samples == [], "zero-arrival dataset should contain zero decisions")
        _assert(zero_manifest["sample_count"] == 0, "zero-sample manifest count mismatch")
        reloaded_zero, _ = dataset.load_frozen_dataset(temp_dir / "dataset")
        _assert(reloaded_zero == [], "zero-sample dataset round trip failed")

    print("smoke_greedy_imitation_dataset passed")
    return 0


def _generate(path: Path):
    return dataset.generate_frozen_dataset(
        dataset_dir=path,
        dataset_seed=42,
        episodes=2,
        max_steps_per_episode=1,
        trajectory_policies=["greedy_eft", "random_hash"],
        train_fraction=0.5,
        val_fraction=0.0,
        dag_base_arrival_prob=0.1,
    )


@contextmanager
def _without_partition():
    original = config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES
    config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES = False
    try:
        yield
    finally:
        config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES = original


class _workspace_temp_dir:
    def __init__(self, name: str) -> None:
        self.path = ROOT / ".codex_tmp_greedy_imitation_dataset" / (
            f"{name}_{os.getpid()}_{random.randint(0, 1_000_000)}"
        )

    def __enter__(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
        parent = ROOT / ".codex_tmp_greedy_imitation_dataset"
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
