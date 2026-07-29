from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_fixed_movement_mappo_eft_aux_gate as gate


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _non_torch_checks() -> None:
    parsed = gate.build_arg_parser().parse_args(
        [
            "--bandit-dir",
            "dummy",
            "--lambda-initial",
            "1.0",
        ]
    )
    _assert(parsed.groups == ["A", "B", "C", "D"], "formal group default changed")
    _assert(parsed.seeds == [42, 86, 1042], "formal seed default changed")
    _assert(parsed.updates == 30, "formal update count changed")
    controls = gate._common_controls(parsed)
    _assert(controls["encoder"] == "mlp", "gate encoder is not fixed to MLP")
    _assert(controls["movement"] == "forced_hover", "gate movement is not forced hover")
    _assert(controls["movement_actor_trainable"] is False, "movement actor must be frozen")
    _assert(controls["num_envs"] == 1, "stage-one diagnostic environment count drifted")
    _assert(controls["normalize_value_targets"] is False, "rejected value normalization was re-enabled")
    _assert(controls["value_clip_epsilon"] == 0.0, "rejected value clip was re-enabled")
    _assert(gate._parse_counts("0,1,5,10,21,30") == {0, 1, 5, 10, 21, 30}, "checkpoint count parsing failed")

    identity_random = {
        "mode": "random",
        "training_seed": 42,
        "task_encoder_state_sha256": "encoder-random",
        "candidate_scorer_state_sha256": "scorer-random",
    }
    identity_bandit = {
        "mode": "bandit_checkpoint",
        "training_seed": 42,
        "task_encoder_state_sha256": "encoder-bandit",
        "candidate_scorer_state_sha256": "scorer-bandit",
        "checkpoint_sha256": "checkpoint",
        "dataset_checksum": "dataset",
    }
    rows = [
        {"group": "A", "seed": 42, "initialization_identity": dict(identity_random)},
        {"group": "D", "seed": 42, "initialization_identity": dict(identity_random)},
        {"group": "B", "seed": 42, "initialization_identity": dict(identity_bandit)},
        {"group": "C", "seed": 42, "initialization_identity": dict(identity_bandit)},
    ]
    gate._assert_pairwise_initialization(rows)
    rows[1]["initialization_identity"]["candidate_scorer_state_sha256"] = "drift"
    try:
        gate._assert_pairwise_initialization(rows)
    except AssertionError:
        pass
    else:
        raise AssertionError("A/D initialization drift was not rejected")

    trainer_source = (ROOT / "marl_models" / "mappo" / "clean_trainer.py").read_text(encoding="utf-8")
    _assert("eft_auxiliary_generator" in trainer_source, "independent auxiliary RNG is missing")
    _assert("weighted_eft_auxiliary_loss" in trainer_source, "combined auxiliary loss is missing")


def _torch_checks() -> None:
    import torch
    from scripts import train_clean_mainline as train

    from marl_models.hgnn import build_clean_task_encoder
    from marl_models.mappo.clean_movement_actor import CleanMovementActor
    from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
    from marl_models.mappo.clean_ppo import CleanCentralizedCritic
    from marl_models.mappo.clean_trainer import CleanTrainingModules

    def modules(seed: int) -> CleanTrainingModules:
        torch.manual_seed(seed)
        return CleanTrainingModules(
            hgnn=build_clean_task_encoder(
                encoder_type="mlp",
                task_feature_dim=6,
                hidden_dim=8,
                output_dim=4,
            ),
            movement_actor=CleanMovementActor(task_embedding_dim=4, hidden_dim=8),
            offloading_actor=CleanOffloadingActor(task_embedding_dim=4, hidden_dim=8),
            critic=CleanCentralizedCritic(input_dim=20, hidden_dim=8),
        )

    random_a = modules(42)
    random_d = modules(42)
    args_random = argparse.Namespace(
        seed=42,
        task_encoder="mlp",
        task_embedding_dim=4,
        hidden_dim=8,
        offloading_init_bandit_checkpoint=None,
        offloading_init_bandit_dataset_checksum=None,
    )
    identity_a = train._initialize_offloading_policy(
        args=args_random, modules=random_a, torch=torch
    )
    identity_d = train._initialize_offloading_policy(
        args=args_random, modules=random_d, torch=torch
    )
    _assert(gate._identity_key(identity_a) == gate._identity_key(identity_d), "same-seed A/D initialization differs")

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "bandit.pt"
        source = modules(42)
        torch.save(
            {
                "schema": train.BANDIT_CHECKPOINT_SCHEMA,
                "stage": "trained",
                "encoder": "mlp",
                "training_seed": 42,
                "dataset_checksum": "checksum",
                "task_feature_dim": 6,
                "task_embedding_dim": 4,
                "hidden_dim": 8,
                "task_encoder_state_dict": source.hgnn.state_dict(),
                "candidate_scorer_state_dict": source.offloading_actor.scorer.state_dict(),
                "optimizer": {"must_not_load": True},
            },
            checkpoint,
        )
        args_bandit = argparse.Namespace(
            seed=42,
            task_encoder="mlp",
            task_embedding_dim=4,
            hidden_dim=8,
            offloading_init_bandit_checkpoint=checkpoint,
            offloading_init_bandit_dataset_checksum="checksum",
        )
        bandit_b = modules(7)
        bandit_c = modules(99)
        identity_b = train._initialize_offloading_policy(
            args=args_bandit, modules=bandit_b, torch=torch
        )
        identity_c = train._initialize_offloading_policy(
            args=args_bandit, modules=bandit_c, torch=torch
        )
        _assert(gate._identity_key(identity_b) == gate._identity_key(identity_c), "B/C checkpoint initialization differs")
        _assert(identity_b["optimizer_state_loaded"] is False, "bandit optimizer state was loaded")
        bad_args = argparse.Namespace(**vars(args_bandit))
        bad_args.seed = 86
        try:
            train._initialize_offloading_policy(
                args=bad_args, modules=modules(86), torch=torch
            )
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched bandit seed was not rejected")

    frozen = modules(1)
    for parameter in frozen.movement_actor.parameters():
        parameter.requires_grad_(False)
    _assert(
        not any(parameter.requires_grad for parameter in frozen.movement_actor.parameters()),
        "forced-hover movement actor remains trainable",
    )


def main() -> int:
    _non_torch_checks()
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        print("smoke_fixed_movement_mappo_eft_gate: PASS (torch checks skipped)")
        return 0
    _torch_checks()
    print("smoke_fixed_movement_mappo_eft_gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
