from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import random
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl_models.mappo.clean_slot_orchestrator import CleanSlotRolloutBuffer
from marl_models.mappo.clean_trainer import compute_multi_trajectory_gae
from scripts import run_multisample_throughput_gate
from scripts.train_clean_mainline import (
    _EnvironmentRNGState,
    _SamplerLane,
    _activate_lane_rng,
    _derive_environment_seed,
    build_arg_parser,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _record(*, reward: float, value: float, terminated: bool = False):
    return type(
        "Record",
        (),
        {
            "reward": float(reward),
            "value": float(value),
            "terminated": bool(terminated),
        },
    )()


def _closed_buffer(records: list[object], bootstrap: float) -> CleanSlotRolloutBuffer:
    buffer = CleanSlotRolloutBuffer()
    for record in records:
        buffer.append(record)
    buffer.close(bootstrap_value=float(bootstrap))
    return buffer


def _gae_isolation_check() -> None:
    first = _closed_buffer(
        [
            _record(reward=1.0, value=0.0),
            _record(reward=1.0, value=0.0),
        ],
        bootstrap=10.0,
    )
    second = _closed_buffer(
        [_record(reward=100.0, value=0.0, terminated=True)],
        bootstrap=999.0,
    )
    returns, advantages = compute_multi_trajectory_gae(
        [first, second],
        gamma=1.0,
        gae_lambda=1.0,
    )
    _assert(
        np.allclose(returns, np.asarray([12.0, 11.0, 100.0], dtype=np.float32)),
        "GAE must bootstrap each environment independently",
    )
    _assert(
        np.allclose(advantages, returns),
        "zero value predictions should make advantages equal returns",
    )


def _rng_isolation_check() -> None:
    random.seed(7)
    np.random.seed(7)
    outer_python = random.getstate()
    outer_numpy = np.random.get_state()
    lane_a = _SamplerLane(
        lane_index=0,
        environment_seed=11,
        env=None,
        graph_builder=None,
        rng_state=_EnvironmentRNGState(random.Random(11).getstate(), np.random.RandomState(11).get_state()),
    )
    lane_b = _SamplerLane(
        lane_index=1,
        environment_seed=13,
        env=None,
        graph_builder=None,
        rng_state=_EnvironmentRNGState(random.Random(13).getstate(), np.random.RandomState(13).get_state()),
    )
    with _activate_lane_rng(lane_a):
        a_first = (random.random(), float(np.random.random()))
    with _activate_lane_rng(lane_b):
        b_first = (random.random(), float(np.random.random()))
    with _activate_lane_rng(lane_a):
        a_second = (random.random(), float(np.random.random()))
    reference_a = random.Random(11)
    reference_a_np = np.random.RandomState(11)
    _assert(
        a_first
        == (reference_a.random(), float(reference_a_np.random_sample())),
        "lane A first draw must match its isolated seed",
    )
    _assert(
        a_second
        == (reference_a.random(), float(reference_a_np.random_sample())),
        "lane A stream must continue independently after lane B runs",
    )
    _assert(a_first != b_first, "different environment seeds should produce different streams")
    _assert(random.getstate() == outer_python, "lane activation must restore outer Python RNG")
    restored_numpy = np.random.get_state()
    _assert(
        restored_numpy[0] == outer_numpy[0]
        and np.array_equal(restored_numpy[1], outer_numpy[1])
        and restored_numpy[2:] == outer_numpy[2:],
        "lane activation must restore outer NumPy RNG",
    )


def _cli_and_gate_check() -> None:
    parser = build_arg_parser()
    for value in (1, 2, 4, 8):
        parsed = parser.parse_args(["--num-envs", str(value)])
        _assert(parsed.num_envs == value, f"CLI should accept num_envs={value}")
    seeds = [_derive_environment_seed(42, index) for index in range(8)]
    _assert(len(set(seeds)) == 8, "derived lane seeds must be unique")

    args = Namespace(
        num_envs=[1, 2, 4, 8],
        episodes=8,
        max_steps_per_episode=64,
        rollout_horizon=16,
        seed=42,
        device="cuda",
    )
    run_multisample_throughput_gate._validate_args(args)
    case = run_multisample_throughput_gate._build_case(
        args,
        Path("runs") / "multisample_test",
        8,
    )
    command = case["command"]
    _assert("multisample_num_envs_8" in command, "gate run name must carry multisample")
    _assert(command[command.index("--num-envs") + 1] == "8", "gate must forward num_envs")
    _assert(
        command[command.index("--sampler-backend") + 1] == "process",
        "throughput gate must exercise the true process sampler",
    )


def main() -> int:
    _gae_isolation_check()
    _rng_isolation_check()
    _cli_and_gate_check()
    print("smoke_multisample_rollout passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
