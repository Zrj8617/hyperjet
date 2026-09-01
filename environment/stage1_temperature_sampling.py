from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


GUMBEL_SCHEMA = "stage1_temperature_gumbel_v1"
FROZEN_TEMPERATURES = (1.0, 0.75, 0.5, 0.25)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Any) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_temperature(temperature: float) -> float:
    value = float(temperature)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("temperature must be finite and positive")
    return value


def sampling_key(
    *,
    checkpoint_sha256: str,
    evaluation_scenario_seed: int,
    slot_index: int,
    stable_task_id: str,
    decision_order: int,
    sampling_replicate: int,
    candidate_uav_id: int,
) -> list[Any]:
    checkpoint_hash = str(checkpoint_sha256).lower()
    if len(checkpoint_hash) != 64 or any(character not in "0123456789abcdef" for character in checkpoint_hash):
        raise ValueError("checkpoint_sha256 must be a 64-character lowercase hexadecimal digest")
    if int(slot_index) < 0 or int(decision_order) < 0 or int(sampling_replicate) < 0:
        raise ValueError("slot, decision order, and sampling replicate must be nonnegative")
    if not str(stable_task_id):
        raise ValueError("stable_task_id must be nonempty")
    return [
        GUMBEL_SCHEMA,
        checkpoint_hash,
        int(evaluation_scenario_seed),
        int(slot_index),
        str(stable_task_id),
        int(decision_order),
        int(sampling_replicate),
        int(candidate_uav_id),
    ]


def keyed_uniform_from_key(key: Sequence[Any]) -> float:
    digest = hashlib.sha256(canonical_json_bytes(list(key))).digest()
    prefix = int.from_bytes(digest[:8], byteorder="big", signed=False)
    mantissa = prefix >> 11
    value = (float(mantissa) + 0.5) / float(1 << 53)
    if not 0.0 < value < 1.0:
        raise AssertionError("keyed uniform must be in the open unit interval")
    return value


def keyed_gumbel_from_key(key: Sequence[Any]) -> float:
    uniform = keyed_uniform_from_key(key)
    return -math.log(-math.log(uniform))


def candidate_gumbels(
    *,
    checkpoint_sha256: str,
    evaluation_scenario_seed: int,
    slot_index: int,
    stable_task_id: str,
    decision_order: int,
    sampling_replicate: int,
    candidate_uav_ids: Iterable[int],
) -> np.ndarray:
    return np.asarray(
        [
            keyed_gumbel_from_key(
                sampling_key(
                    checkpoint_sha256=checkpoint_sha256,
                    evaluation_scenario_seed=evaluation_scenario_seed,
                    slot_index=slot_index,
                    stable_task_id=stable_task_id,
                    decision_order=decision_order,
                    sampling_replicate=sampling_replicate,
                    candidate_uav_id=int(uav_id),
                )
            )
            for uav_id in candidate_uav_ids
        ],
        dtype=np.float64,
    )


def legal_temperature_probabilities(
    logits: Sequence[float] | np.ndarray,
    candidate_mask: Sequence[bool] | np.ndarray,
    temperature: float,
) -> np.ndarray:
    resolved_temperature = validate_temperature(temperature)
    raw_logits, mask = _validated_logits_mask(logits, candidate_mask)
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        raise ValueError("at least one legal candidate is required")
    scaled = raw_logits[legal] / resolved_temperature
    shifted = scaled - float(np.max(scaled))
    weights = np.exp(shifted)
    denominator = float(np.sum(weights))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise FloatingPointError("temperature softmax denominator must be finite and positive")
    probabilities = np.zeros_like(raw_logits, dtype=np.float64)
    probabilities[legal] = weights / denominator
    return probabilities


def deterministic_masked_argmax(
    logits: Sequence[float] | np.ndarray,
    candidate_mask: Sequence[bool] | np.ndarray,
    candidate_uav_ids: Sequence[int],
) -> int:
    raw_logits, mask = _validated_logits_mask(logits, candidate_mask)
    uav_ids = _validated_uav_ids(candidate_uav_ids, raw_logits.shape[0])
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        raise ValueError("at least one legal candidate is required")
    best_value = float(np.max(raw_logits[legal]))
    tied = [int(index) for index in legal if float(raw_logits[index]) == best_value]
    return min(tied, key=lambda index: int(uav_ids[index]))


@dataclass(frozen=True, slots=True)
class TemperatureSample:
    selected_index: int
    selected_uav_id: int
    probabilities: np.ndarray
    gumbels: np.ndarray


def keyed_temperature_sample(
    *,
    logits: Sequence[float] | np.ndarray,
    candidate_mask: Sequence[bool] | np.ndarray,
    candidate_uav_ids: Sequence[int],
    temperature: float,
    checkpoint_sha256: str,
    evaluation_scenario_seed: int,
    slot_index: int,
    stable_task_id: str,
    decision_order: int,
    sampling_replicate: int,
) -> TemperatureSample:
    resolved_temperature = validate_temperature(temperature)
    raw_logits, mask = _validated_logits_mask(logits, candidate_mask)
    uav_ids = _validated_uav_ids(candidate_uav_ids, raw_logits.shape[0])
    probabilities = legal_temperature_probabilities(raw_logits, mask, resolved_temperature)
    legal = np.flatnonzero(mask)
    legal_gumbels = candidate_gumbels(
        checkpoint_sha256=checkpoint_sha256,
        evaluation_scenario_seed=evaluation_scenario_seed,
        slot_index=slot_index,
        stable_task_id=stable_task_id,
        decision_order=decision_order,
        sampling_replicate=sampling_replicate,
        candidate_uav_ids=uav_ids[legal],
    )
    gumbels = np.zeros_like(raw_logits, dtype=np.float64)
    gumbels[legal] = legal_gumbels
    scores = raw_logits[legal] / resolved_temperature + gumbels[legal]
    best_score = float(np.max(scores))
    tied_positions = [position for position, score in enumerate(scores) if float(score) == best_score]
    selected_position = min(tied_positions, key=lambda position: int(uav_ids[int(legal[position])]))
    selected_index = int(legal[selected_position])
    probabilities.setflags(write=False)
    gumbels.setflags(write=False)
    return TemperatureSample(
        selected_index=selected_index,
        selected_uav_id=int(uav_ids[selected_index]),
        probabilities=probabilities,
        gumbels=gumbels,
    )


def distribution_diagnostics(
    probabilities: Sequence[float] | np.ndarray,
    candidate_mask: Sequence[bool] | np.ndarray,
) -> dict[str, float]:
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    mask = np.asarray(candidate_mask, dtype=bool).reshape(-1)
    if values.shape != mask.shape:
        raise ValueError("probabilities and candidate_mask must have the same shape")
    legal = np.flatnonzero(mask)
    if legal.size < 2:
        raise ValueError("distribution diagnostics require at least two legal candidates")
    legal_probabilities = values[legal]
    if not bool(np.isfinite(legal_probabilities).all()) or bool((legal_probabilities < 0.0).any()):
        raise ValueError("legal probabilities must be finite and nonnegative")
    if not math.isclose(float(np.sum(legal_probabilities)), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("legal probabilities must sum to one")
    entropy = -float(np.sum(legal_probabilities * np.log(np.clip(legal_probabilities, 1e-300, 1.0))))
    sorted_probabilities = np.sort(legal_probabilities)
    return {
        "normalized_entropy": entropy / math.log(int(legal.size)),
        "max_action_probability": float(sorted_probabilities[-1]),
        "top1_top2_probability_margin": float(sorted_probabilities[-1] - sorted_probabilities[-2]),
    }


def _validated_logits_mask(
    logits: Sequence[float] | np.ndarray,
    candidate_mask: Sequence[bool] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw_logits = np.asarray(logits, dtype=np.float64).reshape(-1).copy()
    mask = np.asarray(candidate_mask, dtype=bool).reshape(-1).copy()
    if raw_logits.shape != mask.shape:
        raise ValueError("logits and candidate_mask must have the same shape")
    if raw_logits.size == 0:
        raise ValueError("candidate vectors must be nonempty")
    if not bool(np.isfinite(raw_logits[mask]).all()):
        raise ValueError("legal logits must be finite")
    return raw_logits, mask


def _validated_uav_ids(candidate_uav_ids: Sequence[int], count: int) -> np.ndarray:
    uav_ids = np.asarray([int(value) for value in candidate_uav_ids], dtype=np.int64)
    if uav_ids.shape != (int(count),):
        raise ValueError("candidate_uav_ids must match candidate count")
    if len(set(int(value) for value in uav_ids)) != int(count):
        raise ValueError("candidate_uav_ids must be unique")
    return uav_ids
