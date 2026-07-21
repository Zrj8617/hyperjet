from __future__ import annotations

import copy
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from marl_models.mappo.clean_trainer import (
    CleanPPOUpdateConfig,
    _normalized_clipped_value_loss,
    _validate_value_configuration,
)
from scripts.train_clean_mainline import (
    build_arg_parser,
    build_config_snapshot,
    checkpoint_experiment_controls,
    validate_resume_experiment_controls,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _loss(value: float, old: float, target: float, mean: float, scale: float, clip: float):
    return _normalized_clipped_value_loss(
        value=torch.tensor(value, dtype=torch.float32, requires_grad=True),
        old_value=torch.tensor(old, dtype=torch.float32),
        target=torch.tensor(target, dtype=torch.float32),
        target_mean=torch.tensor(mean, dtype=torch.float32),
        target_scale=torch.tensor(scale, dtype=torch.float32),
        clip_epsilon=clip,
    )


def main() -> None:
    raw_loss, raw_clipped = _loss(3.0, 2.0, 1.0, 0.0, 1.0, 0.0)
    _assert(torch.isclose(raw_loss, torch.tensor(2.0)), "disabled objective must match legacy MSE")
    _assert(raw_clipped.item() == 0.0, "disabled value clip must report zero clip fraction")

    base_loss, _ = _loss(14.0, 12.0, 10.0, 10.0, 2.0, 0.0)
    shifted_loss, _ = _loss(107.0, 106.0, 105.0, 105.0, 1.0, 0.0)
    _assert(torch.isclose(base_loss, shifted_loss), "normalized loss must be affine-scale consistent")

    clipped_loss, was_clipped = _loss(1.0, 0.0, 1.0, 0.0, 1.0, 0.2)
    _assert(torch.isclose(clipped_loss, torch.tensor(0.32)), "value clip must use the pessimistic PPO loss")
    _assert(was_clipped.item() == 1.0, "out-of-range value update must be reported as clipped")

    _validate_value_configuration(
        CleanPPOUpdateConfig(normalize_value_targets=True, value_clip_epsilon=0.2)
    )
    for invalid in (-0.1, float("inf"), float("nan")):
        try:
            _validate_value_configuration(CleanPPOUpdateConfig(value_clip_epsilon=invalid))
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid value clip should fail: {invalid!r}")

    parser = build_arg_parser()
    args = parser.parse_args([])
    _assert(args.normalize_value_targets is True, "mainline must enable target normalization by default")
    _assert(math.isclose(args.value_clip_epsilon, 0.2), "mainline value clip default mismatch")
    snapshot = build_config_snapshot(args)
    controls = checkpoint_experiment_controls({"config": snapshot})
    _assert(controls["normalize_value_targets"] is True, "checkpoint lost normalization control")
    _assert(math.isclose(controls["value_clip_epsilon"], 0.2), "checkpoint lost value clip control")
    validate_resume_experiment_controls(args, {"config": snapshot})

    mismatch = copy.deepcopy(args)
    mismatch.value_clip_epsilon = 0.1
    try:
        validate_resume_experiment_controls(mismatch, {"config": snapshot})
    except ValueError:
        pass
    else:
        raise AssertionError("resume must reject a value clip mismatch")

    legacy = checkpoint_experiment_controls({})
    _assert(legacy["normalize_value_targets"] is False, "legacy normalization must stay disabled")
    _assert(legacy["value_clip_epsilon"] == 0.0, "legacy value clip must stay disabled")
    print("clean value objective smoke: PASS")


if __name__ == "__main__":
    main()
