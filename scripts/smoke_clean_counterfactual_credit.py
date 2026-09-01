from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _same_state(left: object, right: object) -> bool:
    import torch

    left_state = left.state_dict()
    right_state = right.state_dict()
    return left_state.keys() == right_state.keys() and all(
        torch.equal(left_state[key], right_state[key]) for key in left_state
    )


def main() -> int:
    try:
        import torch
    except ModuleNotFoundError:
        print("smoke_clean_counterfactual_credit skipped: torch is not installed")
        return 0

    import config
    from environment.assignment import (
        CLEAN_OFFLOADING_PAIR_FEATURE_DIM,
        CLEAN_OFFLOADING_UAV_FEATURE_DIM,
    )
    from marl_models.hgnn import build_clean_task_encoder
    from marl_models.mappo.clean_movement_actor import CleanMovementActor
    from marl_models.mappo.clean_offloading_action_value import (
        build_rng_neutral_clean_counterfactual_q,
        masked_counterfactual_value,
    )
    from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
    from marl_models.mappo.clean_ppo import (
        CleanCentralizedCritic,
        clean_critic_input_dim,
    )
    from marl_models.mappo.clean_slot_orchestrator import (
        CleanOffloadingRolloutRecord,
        CleanSlotRolloutBuffer,
        CleanSlotRolloutRecord,
    )
    from marl_models.mappo.clean_trainer import (
        CLEAN_COUNTERFACTUAL_BETA,
        CLEAN_COUNTERFACTUAL_Q_LOSS_COEF,
        CleanCheckpointManager,
        CleanPPOUpdateConfig,
        CleanPPOUpdater,
        CleanTrainingModules,
        build_single_optimizer,
    )
    from scripts import eval_clean_mainline, train_clean_mainline

    embedding_dim = 6
    hidden_dim = 10
    task_feature_dim = 4
    critic_input_dim = clean_critic_input_dim(embedding_dim, config.NUM_UAVS)

    def build_modules(*, encoder_type: str, with_q: bool, seed: int):
        torch.manual_seed(seed)
        offloading_actor = CleanOffloadingActor(
            task_embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )
        q_head = (
            build_rng_neutral_clean_counterfactual_q(
                input_dim=offloading_actor.candidate_feature_dim + critic_input_dim,
                hidden_dim=hidden_dim,
            )
            if with_q
            else None
        )
        encoder = build_clean_task_encoder(
            encoder_type=encoder_type,
            task_feature_dim=task_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
        )
        modules = CleanTrainingModules(
            hgnn=encoder,
            movement_actor=CleanMovementActor(
                task_embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
            ),
            offloading_actor=offloading_actor,
            critic=CleanCentralizedCritic(
                input_dim=critic_input_dim,
                hidden_dim=hidden_dim,
            ),
            offloading_action_value_critic=q_head,
        )
        next_random = torch.rand(5)
        return modules, next_random

    baseline_modules, baseline_next = build_modules(
        encoder_type="mlp", with_q=False, seed=42
    )
    variant_modules, variant_next = build_modules(
        encoder_type="mlp", with_q=True, seed=42
    )
    for name in ("hgnn", "offloading_actor", "movement_actor", "critic"):
        _assert(
            _same_state(getattr(baseline_modules, name), getattr(variant_modules, name)),
            f"RNG-neutral construction changed {name}",
        )
    _assert(
        torch.equal(baseline_next, variant_next),
        "Q construction advanced the main CPU Torch RNG",
    )
    q_head = variant_modules.offloading_action_value_critic
    q_probe = torch.randn(3, q_head.input_dim)
    _assert(
        torch.equal(q_head(q_probe), torch.zeros(3)),
        "clean counterfactual Q output must initialize to exact zero",
    )
    zero_correction, _ = masked_counterfactual_value(
        logits=torch.tensor([0.2, -0.1]),
        action_values=q_head(q_probe[:2]),
        candidate_mask=torch.tensor([True, True]),
        selected_action=0,
    )
    _assert(float(zero_correction.item()) == 0.0, "initial correction must be zero")
    _assert(not zero_correction.requires_grad, "counterfactual correction must be detached")

    task_features = np.asarray(
        [[1.0, 0.2, 0.4, 0.8], [0.3, 1.0, 0.7, 0.1]],
        dtype=np.float32,
    )
    snapshot = SimpleNamespace(
        task_features=task_features,
        incidence_matrix=np.eye(2, dtype=np.float32),
        hyperedge_type_ids=np.zeros(2, dtype=np.int64),
    )
    non_graph_dim = critic_input_dim - embedding_dim

    def make_records(modules: CleanTrainingModules) -> list[CleanSlotRolloutRecord]:
        with torch.no_grad():
            embeddings = modules.hgnn(
                torch.as_tensor(task_features),
                torch.eye(2),
                torch.zeros(2, dtype=torch.long),
            )
        rows: list[CleanSlotRolloutRecord] = []
        for slot_index, (reward, selected, terminated) in enumerate(
            ((-1.0, 0, False), (2.0, 1, True))
        ):
            dynamic = np.linspace(
                0.1,
                0.9,
                2 * CLEAN_OFFLOADING_UAV_FEATURE_DIM,
                dtype=np.float32,
            ).reshape(2, CLEAN_OFFLOADING_UAV_FEATURE_DIM)
            pair = np.linspace(
                0.2 + slot_index,
                1.0 + slot_index,
                2 * CLEAN_OFFLOADING_PAIR_FEATURE_DIM,
                dtype=np.float32,
            ).reshape(2, CLEAN_OFFLOADING_PAIR_FEATURE_DIM)
            task_embedding = embeddings[slot_index].reshape(1, -1)
            features = torch.cat(
                [
                    task_embedding.expand(2, -1),
                    torch.as_tensor(dynamic),
                    torch.as_tensor(pair),
                ],
                dim=1,
            )
            with torch.no_grad():
                probabilities = torch.softmax(
                    modules.offloading_actor.scorer(features), dim=0
                )
            offloading = CleanOffloadingRolloutRecord(
                task_id=f"task_{slot_index}",
                task_local_index=slot_index,
                decision_order=0,
                candidate_uav_ids=[0, 1],
                dynamic_uav_features=dynamic,
                pair_features=pair,
                candidate_mask=np.asarray([True, True]),
                selected_action=selected,
                selected_uav_id=selected,
                old_log_probability=float(torch.log(probabilities[selected]).item()),
                entropy=float(
                    -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum().item()
                ),
                old_masked_probabilities=probabilities.numpy().copy(),
            )
            rows.append(
                CleanSlotRolloutRecord(
                    slot_index=slot_index,
                    graph_snapshot=snapshot,
                    critic_non_graph_input=np.linspace(
                        -0.5, 0.5, non_graph_dim, dtype=np.float32
                    ),
                    value=0.0,
                    reward=reward,
                    terminated=terminated,
                    offloading_records=[offloading],
                )
            )
        return rows

    records = make_records(baseline_modules)
    variant_updater = CleanPPOUpdater(
        modules=variant_modules,
        optimizer=build_single_optimizer(variant_modules, lr=1e-3),
        config=CleanPPOUpdateConfig(clean_counterfactual_credit=True),
        device="cpu",
    )
    direct = variant_updater._loss(
        records=records,
        returns=torch.tensor([-1.0, 2.0]),
        old_values=torch.zeros(2),
        value_target_mean=torch.zeros(()),
        value_target_scale=torch.ones(()),
        advantages=torch.tensor([-1.0, 1.0]),
    )
    q_parameters = list(q_head.parameters())
    q_gradients = torch.autograd.grad(
        direct["offloading_action_value_loss"],
        q_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    _assert(
        any(g is not None and float(g.abs().sum()) > 0.0 for g in q_gradients),
        "Q regression must update the Q head",
    )
    for module_name in ("hgnn", "offloading_actor", "movement_actor", "critic"):
        parameters = list(getattr(variant_modules, module_name).parameters())
        gradients = torch.autograd.grad(
            direct["offloading_action_value_loss"],
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        _assert(
            all(g is None or float(g.abs().sum()) == 0.0 for g in gradients),
            f"Q regression leaked gradients into {module_name}",
        )

    baseline_for_update = copy.deepcopy(baseline_modules)
    variant_for_update = copy.deepcopy(variant_modules)
    baseline_updater = CleanPPOUpdater(
        modules=baseline_for_update,
        optimizer=build_single_optimizer(baseline_for_update, lr=1e-3),
        config=CleanPPOUpdateConfig(),
        device="cpu",
    )
    isolated_updater = CleanPPOUpdater(
        modules=variant_for_update,
        optimizer=build_single_optimizer(variant_for_update, lr=1e-3),
        config=CleanPPOUpdateConfig(clean_counterfactual_credit=True),
        device="cpu",
    )

    def buffer_from(records_to_use: list[CleanSlotRolloutRecord]):
        buffer = CleanSlotRolloutBuffer()
        for row in records_to_use:
            buffer.append(row)
        buffer.close(bootstrap_value=0.0)
        return buffer

    baseline_stats = baseline_updater.update(buffer_from(records))
    variant_stats = isolated_updater.update(buffer_from(records))
    _assert(
        math.isclose(
            baseline_stats.diagnostics["grad_pre_clip_global"],
            variant_stats.diagnostics["grad_pre_clip_global"],
            rel_tol=0.0,
            abs_tol=1e-7,
        ),
        "Q gradient changed the base PPO gradient norm",
    )
    _assert(
        variant_stats.diagnostics["grad_pre_clip_clean_counterfactual_q"] > 0.0,
        "separate Q gradient norm was not recorded",
    )
    _assert(
        variant_stats.diagnostics["gradient_clipping"] == "base_and_q_separate",
        "variant did not use split gradient clipping",
    )
    _assert(
        variant_stats.diagnostics["clean_counterfactual_probability_action_count"] == 2,
        "post-update probability diagnostics missed recorded decisions",
    )
    for key in (
        "clean_counterfactual_probability_l1_delta_mean",
        "clean_counterfactual_probability_kl_mean",
        "clean_counterfactual_selected_probability_delta_mean",
    ):
        _assert(
            math.isfinite(float(variant_stats.diagnostics[key])),
            f"non-finite probability diagnostic: {key}",
        )
    for name in ("hgnn", "offloading_actor", "movement_actor", "critic"):
        _assert(
            _same_state(getattr(baseline_for_update, name), getattr(variant_for_update, name)),
            f"initial zero correction changed the base update for {name}",
        )

    changed_q_modules = copy.deepcopy(variant_modules)
    with torch.no_grad():
        changed_q_modules.offloading_actor.scorer.net[-1].weight.add_(0.05)
    changed_updater = CleanPPOUpdater(
        modules=changed_q_modules,
        optimizer=build_single_optimizer(changed_q_modules, lr=1e-3),
        config=CleanPPOUpdateConfig(clean_counterfactual_credit=True),
        device="cpu",
    )
    zero_credit = changed_updater._loss(
        records=records,
        returns=torch.tensor([-1.0, 2.0]),
        old_values=torch.zeros(2),
        value_target_mean=torch.zeros(()),
        value_target_scale=torch.ones(()),
        advantages=torch.tensor([-1.0, 1.0]),
    )
    output_layer = changed_q_modules.offloading_action_value_critic.net[-1]
    with torch.no_grad():
        output_layer.weight.fill_(0.1)
        output_layer.bias.zero_()
    changed = changed_updater._loss(
        records=records,
        returns=torch.tensor([-1.0, 2.0]),
        old_values=torch.zeros(2),
        value_target_mean=torch.zeros(()),
        value_target_scale=torch.ones(()),
        advantages=torch.tensor([-1.0, 1.0]),
    )
    _assert(
        not torch.equal(changed["offloading_loss"], zero_credit["offloading_loss"]),
        "non-flat Q did not change offloading credit",
    )
    _assert(
        torch.equal(changed["movement_loss"], zero_credit["movement_loss"]),
        "counterfactual credit changed movement loss",
    )
    _assert(
        torch.equal(changed["value_loss"], zero_credit["value_loss"]),
        "counterfactual credit changed central critic loss",
    )

    temp_root = ROOT / ".codex_tmp_clean_counterfactual_credit" / f"smoke_{os.getpid()}"
    try:
        temp_root.mkdir(parents=True, exist_ok=False)
        manager = CleanCheckpointManager(temp_root / "checkpoints")
        args = train_clean_mainline.build_arg_parser().parse_args(
            [
                "--clean-counterfactual-credit",
                "--task-encoder",
                "mlp",
                "--task-embedding-dim",
                str(embedding_dim),
                "--hidden-dim",
                str(hidden_dim),
            ]
        )
        checkpoint = manager.save(
            modules=variant_for_update,
            optimizer=isolated_updater.optimizer,
            episode=0,
            global_slot=2,
            update_step=1,
            config_snapshot=train_clean_mainline.build_config_snapshot(args),
            safe_boundary=True,
        )
        payload = manager.read(checkpoint)
        _assert(
            "offloading_action_value_critic" in payload,
            "variant checkpoint is missing Q state",
        )
        controls = train_clean_mainline.checkpoint_experiment_controls(payload)
        _assert(controls["clean_counterfactual_credit"] is True, "checkpoint mode lost")
        eval_modules = eval_clean_mainline._build_modules(
            dims={
                "task_feature_dim": task_feature_dim,
                "task_embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
            },
            experiment_controls=controls,
            device=torch.device("cpu"),
        )
        eval_clean_mainline._load_module_state(eval_modules, payload)
        _assert(
            eval_modules.offloading_action_value_critic is not None,
            "evaluation did not reconstruct the clean counterfactual Q head",
        )
        restored_modules, _ = build_modules(
            encoder_type="mlp", with_q=True, seed=7
        )
        restored_optimizer = build_single_optimizer(restored_modules, lr=1e-3)
        manager.restore(
            modules=restored_modules,
            optimizer=restored_optimizer,
            payload=payload,
        )
        _assert(
            _same_state(variant_for_update.offloading_action_value_critic, restored_modules.offloading_action_value_critic),
            "Q checkpoint round trip failed",
        )
        try:
            manager.restore(
                modules=baseline_modules,
                optimizer=build_single_optimizer(baseline_modules, lr=1e-3),
                payload=payload,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("variant checkpoint resumed into baseline modules")

        for encoder_type in ("mlp", "hgnn"):
            run_args = train_clean_mainline.build_arg_parser().parse_args(
                [
                    "--smoke",
                    "--episodes",
                    "1",
                    "--max-steps-per-episode",
                    "4",
                    "--rollout-horizon",
                    "2",
                    "--max-updates",
                    "1",
                    "--checkpoint-interval",
                    "1",
                    "--device",
                    "cpu",
                    "--task-encoder",
                    encoder_type,
                    "--clean-counterfactual-credit",
                    "--output-dir",
                    str(temp_root / f"training_{encoder_type}"),
                    "--run-name",
                    f"counterfactual_{encoder_type}",
                ]
            )
            result = train_clean_mainline.run_training(run_args)
            _assert(
                int(result["completed_update_count"]) == 1,
                f"{encoder_type} smoke did not complete one PPO update",
            )
            latest = result["latest_update"]
            _assert(latest is not None, f"{encoder_type} smoke produced no update stats")
            _assert(
                math.isfinite(float(latest["total_loss"])),
                f"{encoder_type} smoke produced a non-finite loss",
            )
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root)
        parent = temp_root.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()

    _assert(CLEAN_COUNTERFACTUAL_BETA == 0.25, "counterfactual beta changed")
    _assert(CLEAN_COUNTERFACTUAL_Q_LOSS_COEF == 0.5, "Q loss coefficient changed")
    print("smoke_clean_counterfactual_credit PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
