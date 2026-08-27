"""Process-local reward override wrapper for one R2 clean MLP training run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402


ARMS = ("control", "no_position", "low_cancel", "energy_balanced")


def build_wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--r2-arm", choices=ARMS, required=True)
    parser.add_argument("--r2-energy-weight", type=float, required=True)
    parser.add_argument(
        "--r2-position-shaping",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    wrapper_args, training_argv = build_wrapper_parser().parse_known_args(argv)
    energy_weight = float(wrapper_args.r2_energy_weight)
    if not math.isfinite(energy_weight) or energy_weight < 0.0:
        raise ValueError("R2 energy weight must be finite and non-negative")

    config.REWARD_ENERGY_WEIGHT = energy_weight
    config.ENABLE_MOVEMENT_POSITION_SHAPING = bool(wrapper_args.r2_position_shaping)

    from scripts import train_clean_mainline

    parser = train_clean_mainline.build_arg_parser()
    args = parser.parse_args(training_argv)
    args.r2_arm = str(wrapper_args.r2_arm)
    args.r2_energy_weight = energy_weight
    args.r2_position_shaping = bool(wrapper_args.r2_position_shaping)
    if not bool(args.freeze_movement):
        raise ValueError("R2 reward bundle requires --freeze-movement")
    if str(args.task_encoder) != "mlp":
        raise ValueError("R2 reward bundle requires --task-encoder mlp")
    if any(
        float(value) != 0.0
        for value in (
            args.offloading_counterfactual_coef,
            args.offloading_action_value_loss_coef,
            args.offloading_lagged_q_coef,
            args.offloading_lagged_q_loss_coef,
            args.eft_auxiliary_lambda_initial,
        )
    ) or bool(args.offloading_eft_advantage):
        raise ValueError("R2 reward bundle requires all guidance/auxiliary/Q paths off")

    result = train_clean_mainline.run_training(args)
    result["r2_reward_override"] = {
        "arm": str(wrapper_args.r2_arm),
        "completed_dag_weight": float(args.completed_dag_weight),
        "energy_weight": energy_weight,
        "movement_position_shaping": bool(wrapper_args.r2_position_shaping),
    }
    run_dir = Path(result["run_dir"])
    (run_dir / "r2_reward_override.json").write_text(
        json.dumps(result["r2_reward_override"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
