from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from environment.env import Env
from environment.user_equipments import UE
from scripts.eval_clean_mainline import _resolve_eval_ue_mobility, build_arg_parser as build_eval_parser
from scripts.train_clean_mainline import (
    build_arg_parser as build_train_parser,
    build_config_snapshot,
    checkpoint_experiment_controls,
    validate_resume_experiment_controls,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _rng_states_equal(left: tuple, right: tuple) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _positions(env: Env) -> list[np.ndarray]:
    return [ue.pos.copy() for ue in env.ues]


def _check_default_neutrality_and_hotspot_covariate() -> None:
    np.random.seed(801)
    default_env = Env()
    default_env.reset()
    default_initial = _positions(default_env)
    default_count = default_env.initial_hotspot_ue_count
    default_env.prepare_slot_state()
    default_after = _positions(default_env)
    default_rng = np.random.get_state()

    np.random.seed(801)
    explicit_moving = Env(freeze_ue_mobility=False)
    explicit_moving.reset()
    _assert(explicit_moving.initial_hotspot_ue_count == default_count, "moving hotspot count changed")
    explicit_moving.prepare_slot_state()
    explicit_after = _positions(explicit_moving)
    explicit_rng = np.random.get_state()

    _assert(all(np.array_equal(a, b) for a, b in zip(default_initial, _positions_from_seed(801))), "initial positions changed")
    _assert(all(np.array_equal(a, b) for a, b in zip(default_after, explicit_after)), "default movement changed")
    _assert(_rng_states_equal(default_rng, explicit_rng), "default and explicit moving RNG states differ")
    _assert(any(not np.array_equal(a, b) for a, b in zip(default_initial, default_after)), "moving mode did not move any UE")

    np.random.seed(801)
    fixed_env = Env(freeze_ue_mobility=True)
    fixed_env.reset()
    fixed_initial = _positions(fixed_env)
    _assert(fixed_env.initial_hotspot_ue_count == default_count, "fixed hotspot count is not comparable")
    fixed_env.prepare_slot_state()
    _assert(all(np.array_equal(a, b) for a, b in zip(fixed_initial, _positions(fixed_env))), "fixed UE position changed")


def _positions_from_seed(seed: int) -> list[np.ndarray]:
    np.random.seed(seed)
    env = Env()
    env.reset()
    return _positions(env)


def _check_boundary_rng_alignment() -> None:
    np.random.seed(1902)
    moving = UE(0)
    moving.pos = np.array([float(config.AREA_WIDTH) - 0.01, 10.0, 0.0], dtype=np.float32)
    moving.speed = float(config.UE_GM_MAX_SPEED)
    moving.theta = 0.0
    moving.velocity = moving._velocity_from_polar()
    fixed = deepcopy(moving)
    fixed_initial_position = fixed.pos.copy()

    state = np.random.get_state()
    moving.update_position(commit_position=True)
    moving_rng = np.random.get_state()
    np.random.set_state(state)
    fixed.update_position(commit_position=False)
    fixed_rng = np.random.get_state()

    _assert(_rng_states_equal(moving_rng, fixed_rng), "boundary path consumed different RNG draws")
    _assert(np.array_equal(fixed.pos, fixed_initial_position), "fixed boundary UE moved")
    _assert(not np.array_equal(moving.pos, fixed.pos), "forced boundary case did not distinguish commit behavior")
    _assert(moving.speed == fixed.speed, "mobility speed state diverged")
    _assert(moving.theta == fixed.theta, "mobility direction state diverged")
    _assert(np.array_equal(moving.velocity, fixed.velocity), "mobility velocity state diverged")


def _check_control_provenance() -> None:
    train_parser = build_train_parser()
    fixed_args = train_parser.parse_args(["--completed-dag-weight", "16", "--freeze-ue-mobility"])
    fixed_args._offloading_initialization_identity = {"mode": "random"}
    snapshot = build_config_snapshot(fixed_args)
    _assert(snapshot["cli"]["freeze_ue_mobility"] is True, "training CLI snapshot lost fixed mode")
    _assert(snapshot["experiment_controls"]["freeze_ue_mobility"] is True, "experiment controls lost fixed mode")
    controls = checkpoint_experiment_controls({"config": snapshot})
    _assert(controls["freeze_ue_mobility"] is True, "checkpoint did not recover fixed mode")
    _assert(checkpoint_experiment_controls({})["freeze_ue_mobility"] is False, "legacy checkpoint is not moving mode")
    validate_resume_experiment_controls(fixed_args, {"config": snapshot})

    moving_args = train_parser.parse_args(["--completed-dag-weight", "16"])
    try:
        validate_resume_experiment_controls(moving_args, {"config": snapshot})
    except ValueError:
        pass
    else:
        raise AssertionError("resume accepted a UE mobility mismatch")

    malformed = {"config": {"cli": {"freeze_ue_mobility": "yes"}}}
    try:
        checkpoint_experiment_controls(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("malformed checkpoint UE mobility mode was accepted")

    eval_parser = build_eval_parser()
    inherited_args = eval_parser.parse_args([])
    _assert(_resolve_eval_ue_mobility(inherited_args, controls) is True, "eval did not inherit fixed mode")
    matching_args = eval_parser.parse_args(["--freeze-ue-mobility"])
    _assert(_resolve_eval_ue_mobility(matching_args, controls) is True, "eval rejected matching fixed mode")
    conflict_args = eval_parser.parse_args(["--no-freeze-ue-mobility"])
    try:
        _resolve_eval_ue_mobility(conflict_args, controls)
    except ValueError:
        pass
    else:
        raise AssertionError("eval accepted a conflicting UE mobility override")


def main() -> None:
    _check_default_neutrality_and_hotspot_covariate()
    _check_boundary_rng_alignment()
    _check_control_provenance()
    print("smoke_clean_static_ue passed")


if __name__ == "__main__":
    main()
