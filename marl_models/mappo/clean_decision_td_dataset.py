from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PHASE4_RAW_DATASET_SCHEMA = "hyperuav_phase4_decision_td_raw_v1"
PHASE4_RHO_ZERO_TOLERANCE = 1e-12


class CleanDecisionTDRawCapture:
    """Write one compact, immutable raw diagnostic shard per shadow-Q update."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        selected_q_input_dim: int,
        source_checkpoint: str | None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.selected_q_input_dim = int(selected_q_input_dim)
        if self.selected_q_input_dim <= 0:
            raise ValueError("selected Q input dimension must be positive")
        self.sample_count = 0
        self.shard_count = 0
        metadata = {
            "schema": PHASE4_RAW_DATASET_SCHEMA,
            "storage": "per_update_npz_shards",
            "rho_zero_absolute_tolerance": PHASE4_RHO_ZERO_TOLERANCE,
            "selected_q_input_dim": self.selected_q_input_dim,
            "source_checkpoint": source_checkpoint,
            "resume_semantics": "restart_from_new_episode_only",
            "contains_graph_snapshot": False,
            "target_frozen_at_capture": True,
        }
        (self.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )

    def write_batch(
        self,
        *,
        update_step: int,
        transitions: Iterable[Any],
        selected_q_inputs: np.ndarray,
        bootstrap_values: np.ndarray,
        td_targets: np.ndarray,
        phase3b_targets: np.ndarray,
        online_selected_q_predictions: np.ndarray,
    ) -> Path:
        rows = list(transitions)
        count = len(rows)
        inputs = np.asarray(selected_q_inputs, dtype=np.float32)
        bootstrap = np.asarray(bootstrap_values, dtype=np.float32).reshape(-1)
        targets = np.asarray(td_targets, dtype=np.float32).reshape(-1)
        phase3b = np.asarray(phase3b_targets, dtype=np.float32).reshape(-1)
        predictions = np.asarray(
            online_selected_q_predictions, dtype=np.float32
        ).reshape(-1)
        if inputs.shape != (count, self.selected_q_input_dim):
            raise ValueError("raw selected Q inputs have an unexpected shape")
        for values in (bootstrap, targets, phase3b, predictions):
            if values.size != count:
                raise ValueError("raw diagnostic columns have inconsistent lengths")
        if not all(
            bool(np.isfinite(values).all())
            for values in (inputs, bootstrap, targets, phase3b, predictions)
        ):
            raise FloatingPointError("raw diagnostic dataset contains non-finite values")

        states = [row.state for row in rows]
        shard_path = self.output_dir / f"update_{int(update_step):04d}.npz"
        if shard_path.exists():
            raise FileExistsError(f"raw diagnostic shard already exists: {shard_path}")
        np.savez_compressed(
            shard_path,
            update_step=np.full(count, int(update_step), dtype=np.int64),
            episode_index=np.asarray([state.episode_index for state in states], dtype=np.int64),
            lane_index=np.asarray([state.lane_index for state in states], dtype=np.int64),
            slot_index=np.asarray([state.slot_index for state in states], dtype=np.int64),
            decision_order=np.asarray([state.decision_order for state in states], dtype=np.int64),
            task_id=np.asarray([state.task_id for state in states]),
            dag_id=np.asarray([state.dag_id for state in states]),
            selected_action=np.asarray([state.selected_action for state in states], dtype=np.int64),
            selected_uav_id=np.asarray([state.selected_uav_id for state in states], dtype=np.int64),
            delta=np.asarray([row.delta for row in rows], dtype=np.int64),
            rho=np.asarray([row.rho for row in rows], dtype=np.float64),
            bootstrap_value=bootstrap,
            td_target=targets,
            phase3b_slot_advantage_target=phase3b,
            online_selected_q_prediction=predictions,
            terminated=np.asarray([row.terminated for row in rows], dtype=bool),
            truncated=np.asarray([row.truncated for row in rows], dtype=bool),
            unresolved=np.asarray([row.unresolved for row in rows], dtype=bool),
            selected_q_input=inputs,
        )
        self.sample_count += count
        self.shard_count += 1
        return shard_path

    def summary(self) -> dict[str, Any]:
        return {
            "phase4_raw_capture_enabled": True,
            "phase4_raw_capture_sample_count": int(self.sample_count),
            "phase4_raw_capture_shard_count": int(self.shard_count),
            "phase4_raw_capture_directory": str(self.output_dir),
            "phase4_rho_zero_tolerance": PHASE4_RHO_ZERO_TOLERANCE,
        }


def load_clean_decision_td_raw_dataset(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = Path(path)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != PHASE4_RAW_DATASET_SCHEMA:
        raise ValueError("unsupported Phase4 raw diagnostic dataset schema")
    shards = sorted(root.glob("update_*.npz"))
    if not shards:
        raise ValueError("Phase4 raw diagnostic dataset has no shards")
    columns: dict[str, list[np.ndarray]] = {}
    for shard in shards:
        with np.load(shard, allow_pickle=False) as payload:
            for key in payload.files:
                columns.setdefault(key, []).append(np.asarray(payload[key]))
    combined = {key: np.concatenate(values, axis=0) for key, values in columns.items()}
    count = int(combined["td_target"].shape[0])
    if int(combined["selected_q_input"].shape[0]) != count:
        raise ValueError("raw dataset input/target row counts differ")
    return combined, metadata


def clean_decision_slot_position_flags(dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    count = int(dataset["td_target"].shape[0])
    keys = list(
        zip(
            dataset["episode_index"].tolist(),
            dataset["lane_index"].tolist(),
            dataset["slot_index"].tolist(),
        )
    )
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(tuple(int(value) for value in key), []).append(index)
    decision_count = np.zeros(count, dtype=np.int64)
    is_first = np.zeros(count, dtype=bool)
    is_last = np.zeros(count, dtype=bool)
    is_singleton = np.zeros(count, dtype=bool)
    for indices in groups.values():
        orders = np.asarray(dataset["decision_order"])[indices]
        if int(np.unique(orders).size) != len(indices):
            raise ValueError("decision_order must be unique within a slot")
        first_index = indices[int(np.argmin(orders))]
        last_index = indices[int(np.argmax(orders))]
        decision_count[indices] = len(indices)
        is_first[first_index] = True
        is_last[last_index] = True
        if len(indices) == 1:
            is_singleton[first_index] = True
    return {
        "decision_count_in_slot": decision_count,
        "is_first_decision": is_first,
        "is_last_decision": is_last,
        "is_middle_decision": ~(is_first | is_last),
        "is_singleton": is_singleton,
        "is_multi_decision_slot": decision_count > 1,
    }
