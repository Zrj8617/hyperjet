from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.stage1_temperature_sampling import (
    GUMBEL_SCHEMA,
    candidate_gumbels,
    canonical_json_bytes,
    deterministic_masked_argmax,
    distribution_diagnostics,
    keyed_gumbel_from_key,
    keyed_temperature_sample,
    keyed_uniform_from_key,
    legal_temperature_probabilities,
    sampling_key,
)


def main() -> int:
    checkpoint_hash = "ab" * 32
    key = sampling_key(
        checkpoint_sha256=checkpoint_hash,
        evaluation_scenario_seed=424242,
        slot_index=7,
        stable_task_id="dag_task_0003",
        decision_order=2,
        sampling_replicate=1,
        candidate_uav_id=4,
    )
    expected_bytes = json.dumps(key, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    assert key[0] == GUMBEL_SCHEMA
    assert canonical_json_bytes(key) == expected_bytes
    digest = hashlib.sha256(expected_bytes).digest()
    mantissa = int.from_bytes(digest[:8], "big") >> 11
    expected_uniform = (mantissa + 0.5) / float(1 << 53)
    assert keyed_uniform_from_key(key) == expected_uniform
    assert keyed_gumbel_from_key(key) == -math.log(-math.log(expected_uniform))

    logits = np.asarray([2.0, 1.8, -4.0, 0.5], dtype=np.float64)
    mask = np.asarray([True, True, False, True], dtype=bool)
    uav_ids = [4, 2, 0, 3]
    probabilities = legal_temperature_probabilities(logits, mask, 1.0)
    reference = np.exp(logits[mask] - np.max(logits[mask]))
    reference /= reference.sum()
    assert np.allclose(probabilities[mask], reference, atol=1e-15, rtol=0.0)
    assert probabilities[2] == 0.0
    assert deterministic_masked_argmax([2.0, 2.0], [True, True], [4, 2]) == 1
    for temperature in (1.0, 0.75, 0.5, 0.25):
        assert deterministic_masked_argmax(logits / temperature, mask, uav_ids) == 0

    samples = [
        keyed_temperature_sample(
            logits=logits,
            candidate_mask=mask,
            candidate_uav_ids=uav_ids,
            temperature=temperature,
            checkpoint_sha256=checkpoint_hash,
            evaluation_scenario_seed=424242,
            slot_index=7,
            stable_task_id="dag_task_0003",
            decision_order=2,
            sampling_replicate=1,
        )
        for temperature in (1.0, 0.75, 0.5, 0.25)
    ]
    for sample in samples[1:]:
        assert np.array_equal(sample.gumbels, samples[0].gumbels)
    different = candidate_gumbels(
        checkpoint_sha256=checkpoint_hash,
        evaluation_scenario_seed=424242,
        slot_index=7,
        stable_task_id="dag_task_0003",
        decision_order=2,
        sampling_replicate=2,
        candidate_uav_ids=uav_ids,
    )
    assert not np.array_equal(different, samples[0].gumbels)
    diagnostics = distribution_diagnostics(samples[0].probabilities, mask)
    assert 0.0 <= diagnostics["normalized_entropy"] <= 1.0
    assert 0.0 <= diagnostics["top1_top2_probability_margin"] <= 1.0
    try:
        samples[0].gumbels[0] = 0.0
    except ValueError:
        pass
    else:
        raise AssertionError("returned Gumbels must be immutable")
    print("PASS smoke_stage1_temperature_sampling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
