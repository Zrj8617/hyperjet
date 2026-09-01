from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_clean_mainline


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _expect_rejected(arguments: list[str]) -> None:
    args = train_clean_mainline.build_arg_parser().parse_args(arguments)
    try:
        train_clean_mainline.validate_clean_counterfactual_credit_controls(args)
    except ValueError:
        return
    raise AssertionError(f"incompatible controls were accepted: {arguments}")


def main() -> int:
    parser = train_clean_mainline.build_arg_parser()
    baseline = parser.parse_args([])
    variant = parser.parse_args(["--clean-counterfactual-credit"])
    train_clean_mainline.validate_clean_counterfactual_credit_controls(baseline)
    train_clean_mainline.validate_clean_counterfactual_credit_controls(variant)

    _assert(baseline.clean_counterfactual_credit is False, "default must be disabled")
    _assert(variant.clean_counterfactual_credit is True, "variant flag was not parsed")
    controls = train_clean_mainline.build_config_snapshot(variant)["experiment_controls"]
    expected = {
        "clean_counterfactual_credit": True,
        "clean_counterfactual_credit_mode": "action_conditioned_v1",
        "counterfactual_beta": 0.25,
        "counterfactual_q_loss_coef": 0.5,
        "gradient_clipping": "base_and_q_separate",
    }
    for key, value in expected.items():
        _assert(controls[key] == value, f"unexpected {key}: {controls[key]!r}")

    _expect_rejected(
        [
            "--clean-counterfactual-credit",
            "--offloading-counterfactual-coef",
            "0.25",
            "--offloading-action-value-loss-coef",
            "0.5",
        ]
    )
    _expect_rejected(
        [
            "--clean-counterfactual-credit",
            "--offloading-lagged-q-coef",
            "0.25",
            "--offloading-lagged-q-loss-coef",
            "0.5",
        ]
    )
    _expect_rejected(["--clean-counterfactual-credit", "--decision-critic"])
    _expect_rejected(["--clean-counterfactual-credit", "--offloading-eft-advantage"])
    _expect_rejected(
        ["--clean-counterfactual-credit", "--eft-auxiliary-lambda-initial", "0.1"]
    )
    _expect_rejected(
        [
            "--clean-counterfactual-credit",
            "--offloading-init-bandit-checkpoint",
            "teacher.pt",
        ]
    )

    legacy_controls = train_clean_mainline.checkpoint_experiment_controls({})
    _assert(
        legacy_controls["clean_counterfactual_credit"] is False,
        "old checkpoints must resolve to disabled",
    )
    variant_payload = {"config": train_clean_mainline.build_config_snapshot(variant)}
    _assert(
        train_clean_mainline.checkpoint_experiment_controls(variant_payload)
        ["clean_counterfactual_credit"]
        is True,
        "variant checkpoint controls were not recovered",
    )
    for requested, payload in ((baseline, variant_payload), (variant, {})):
        try:
            train_clean_mainline.validate_resume_experiment_controls(requested, payload)
        except ValueError as exc:
            _assert(
                "clean counterfactual credit mismatch" in str(exc),
                f"resume failed for the wrong reason: {exc}",
            )
        else:
            raise AssertionError("baseline/variant cross-resume must be rejected")

    print("dry_run_phase3B_clean_counterfactual_credit PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
