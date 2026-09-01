from __future__ import annotations

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


def main() -> int:
    try:
        import torch
    except ModuleNotFoundError:
        print("smoke_clean_offloading_counterfactual skipped: torch is not installed")
        return 0

    from marl_models.mappo.clean_offloading_action_value import (
        CleanOffloadingActionValueCritic,
        masked_counterfactual_value,
        normalize_counterfactual_values,
    )
    from environment.assignment import (
        CLEAN_OFFLOADING_PAIR_FEATURE_DIM,
        CLEAN_OFFLOADING_UAV_FEATURE_DIM,
    )
    import config
    from marl_models.hgnn import CleanIncidenceHGNN
    from marl_models.mappo.clean_movement_actor import CleanMovementActor
    from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
    from marl_models.mappo.clean_ppo import CleanCentralizedCritic, clean_critic_input_dim
    from marl_models.mappo.clean_slot_orchestrator import (
        CleanOffloadingRolloutRecord,
        CleanSlotRolloutBuffer,
        CleanSlotRolloutRecord,
    )
    from marl_models.mappo.clean_trainer import (
        CleanPPOUpdateConfig,
        CleanPPOUpdater,
        CleanCheckpointManager,
        CleanTrainingModules,
        build_single_optimizer,
    )
    from scripts import eval_clean_mainline, train_clean_mainline

    critic = CleanOffloadingActionValueCritic(input_dim=7, hidden_dim=11)
    zero_values = critic(torch.randn(4, 7))
    _assert(torch.equal(zero_values, torch.zeros_like(zero_values)), "Q output layer must initialize to exact zero")

    logits = torch.tensor([0.0, math.log(3.0), 100.0], requires_grad=True)
    q_values = torch.tensor([1.0, 5.0, 999.0], requires_grad=True)
    mask = torch.tensor([True, True, False])
    counterfactual, spread = masked_counterfactual_value(
        logits=logits,
        action_values=q_values,
        candidate_mask=mask,
        selected_action=1,
    )
    # Legal probabilities are 1/4 and 3/4, so E[Q] = 4 and Q_selected - E[Q] = 1.
    _assert(torch.allclose(counterfactual, torch.tensor(1.0)), "masked counterfactual value mismatch")
    _assert(torch.allclose(spread, torch.tensor(4.0)), "legal Q spread mismatch")
    _assert(not counterfactual.requires_grad and not spread.requires_grad, "counterfactual outputs must be detached")

    single, single_spread = masked_counterfactual_value(
        logits=torch.tensor([2.0, -3.0]),
        action_values=torch.tensor([7.0, 8.0]),
        candidate_mask=torch.tensor([False, True]),
        selected_action=1,
    )
    _assert(float(single.item()) == 0.0 and float(single_spread.item()) == 0.0, "single legal action must have zero counterfactual value")

    normalized, diagnostics = normalize_counterfactual_values(
        [torch.tensor(-1.0, requires_grad=True), torch.tensor(1.0, requires_grad=True)]
    )
    normalized_tensor = torch.stack(normalized)
    _assert(torch.allclose(normalized_tensor, torch.tensor([-1.0, 1.0])), "population normalization mismatch")
    _assert(not normalized_tensor.requires_grad, "normalized counterfactual values must be detached")
    _assert(diagnostics["effective_count"] == 2, "counterfactual count mismatch")
    _assert(abs(float(diagnostics["normalized_std"]) - 1.0) < 1e-7, "normalized std mismatch")

    zero_normalized, zero_diagnostics = normalize_counterfactual_values([torch.tensor(3.0), torch.tensor(3.0)])
    _assert(torch.equal(torch.stack(zero_normalized), torch.zeros(2)), "flat values must normalize to zero")
    _assert(float(zero_diagnostics["normalized_std"]) == 0.0, "flat normalized std must be zero")
    empty, empty_diagnostics = normalize_counterfactual_values([])
    _assert(empty == [] and empty_diagnostics["effective_count"] == 0, "empty normalization mismatch")

    failure_cases = [
        lambda: masked_counterfactual_value(
            logits=torch.zeros(2),
            action_values=torch.zeros(2),
            candidate_mask=torch.zeros(2, dtype=torch.bool),
            selected_action=0,
        ),
        lambda: masked_counterfactual_value(
            logits=torch.zeros(2),
            action_values=torch.zeros(2),
            candidate_mask=torch.tensor([True, False]),
            selected_action=1,
        ),
        lambda: masked_counterfactual_value(
            logits=torch.zeros(2),
            action_values=torch.tensor([0.0, float("nan")]),
            candidate_mask=torch.ones(2, dtype=torch.bool),
            selected_action=0,
        ),
    ]
    for case in failure_cases:
        try:
            case()
        except (ValueError, FloatingPointError):
            pass
        else:
            raise AssertionError("invalid counterfactual input should fail")

    embedding_dim = 6
    hidden_dim = 10
    critic_input_dim = clean_critic_input_dim(embedding_dim, config.NUM_UAVS)
    offloading_actor = CleanOffloadingActor(
        task_embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    )
    action_value_critic = CleanOffloadingActionValueCritic(
        input_dim=offloading_actor.candidate_feature_dim + critic_input_dim,
        hidden_dim=hidden_dim,
    )
    modules = CleanTrainingModules(
        hgnn=CleanIncidenceHGNN(task_feature_dim=4, hidden_dim=hidden_dim, output_dim=embedding_dim),
        movement_actor=CleanMovementActor(task_embedding_dim=embedding_dim, hidden_dim=hidden_dim),
        offloading_actor=offloading_actor,
        critic=CleanCentralizedCritic(input_dim=critic_input_dim, hidden_dim=hidden_dim),
        offloading_action_value_critic=action_value_critic,
    )
    non_graph_dim = critic_input_dim - embedding_dim
    snapshot = SimpleNamespace(
        task_features=np.asarray([[1.0, 0.2, 0.4, 0.8], [0.3, 1.0, 0.7, 0.1]], dtype=np.float32),
        incidence_matrix=np.eye(2, dtype=np.float32),
        hyperedge_type_ids=np.zeros(2, dtype=np.int64),
    )

    def _slot(slot_index: int, reward: float, selected_action: int, terminated: bool):
        dynamic = np.linspace(
            0.1,
            0.9,
            2 * CLEAN_OFFLOADING_UAV_FEATURE_DIM,
            dtype=np.float32,
        ).reshape(2, CLEAN_OFFLOADING_UAV_FEATURE_DIM)
        pair = np.linspace(
            0.2 + 0.1 * slot_index,
            1.0 + 0.1 * slot_index,
            2 * CLEAN_OFFLOADING_PAIR_FEATURE_DIM,
            dtype=np.float32,
        ).reshape(2, CLEAN_OFFLOADING_PAIR_FEATURE_DIM)
        offloading = CleanOffloadingRolloutRecord(
            task_id=f"task_{slot_index}",
            task_local_index=slot_index,
            decision_order=0,
            candidate_uav_ids=[0, 1],
            dynamic_uav_features=dynamic,
            pair_features=pair,
            candidate_mask=np.asarray([True, True]),
            selected_action=selected_action,
            selected_uav_id=selected_action,
            old_log_probability=-math.log(2.0),
            entropy=math.log(2.0),
        )
        return CleanSlotRolloutRecord(
            slot_index=slot_index,
            graph_snapshot=snapshot,
            critic_non_graph_input=np.linspace(-0.5, 0.5, non_graph_dim, dtype=np.float32),
            value=0.0,
            reward=reward,
            terminated=terminated,
            offloading_records=[offloading],
        )

    records = [_slot(0, -1.0, 0, False), _slot(1, 2.0, 1, True)]
    updater = CleanPPOUpdater(
        modules=modules,
        optimizer=build_single_optimizer(modules, lr=1e-3),
        config=CleanPPOUpdateConfig(
            ppo_epochs=2,
            offloading_counterfactual_coef=0.25,
            offloading_action_value_loss_coef=0.5,
        ),
        device="cpu",
    )
    direct_loss = updater._loss(
        records=records,
        returns=torch.tensor([-1.0, 2.0]),
        old_values=torch.zeros(2),
        value_target_mean=torch.zeros(()),
        value_target_scale=torch.ones(()),
        advantages=torch.tensor([-1.0, 1.0]),
    )
    q_parameters = list(action_value_critic.parameters())
    hgnn_parameters = list(modules.hgnn.parameters())
    q_grads = torch.autograd.grad(
        direct_loss["offloading_action_value_loss"],
        q_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    hgnn_grads = torch.autograd.grad(
        direct_loss["offloading_action_value_loss"],
        hgnn_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    _assert(any(grad is not None and float(grad.abs().sum()) > 0.0 for grad in q_grads), "Q loss must update the Q head")
    _assert(all(grad is None or float(grad.abs().sum()) == 0.0 for grad in hgnn_grads), "Q loss must not update HGNN")

    buffer = CleanSlotRolloutBuffer()
    for record in records:
        buffer.append(record)
    buffer.close(bootstrap_value=0.0)
    stats = updater.update(buffer)
    _assert(math.isfinite(stats.offloading_action_value_loss) and stats.offloading_action_value_loss > 0.0, "enabled Q loss must be finite and positive")
    _assert(stats.diagnostics["grad_pre_clip_offloading_action_value"] > 0.0, "Q gradient diagnostic must be positive")
    _assert(stats.diagnostics["offloading_action_value_hgnn_grad_norm"] == 0.0, "Q-to-HGNN gradient diagnostic must be zero")
    _assert(stats.diagnostics["offloading_counterfactual_effective_action_count"] == 2, "enabled counterfactual action count mismatch")

    temp_dir = ROOT / ".codex_tmp_offloading_counterfactual" / f"smoke_{os.getpid()}"
    try:
        temp_dir.mkdir(parents=True, exist_ok=False)
        manager = CleanCheckpointManager(temp_dir / "checkpoints")
        checkpoint = manager.save(
            modules=modules,
            optimizer=updater.optimizer,
            episode=1,
            global_slot=2,
            update_step=updater.update_step,
            config_snapshot={
                "cli": {
                    "completed_dag_weight": 16.0,
                    "detach_critic_hgnn": False,
                    "freeze_ue_mobility": False,
                    "offloading_counterfactual_coef": 0.25,
                    "offloading_action_value_loss_coef": 0.5,
                }
            },
            safe_boundary=True,
        )
        payload = manager.read(checkpoint)
        _assert("offloading_action_value_critic" in payload, "enabled checkpoint must persist Q state")
        manager.restore(modules=modules, optimizer=updater.optimizer, payload=payload)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        parent = temp_dir.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()

    legacy_modules = CleanTrainingModules(
        hgnn=modules.hgnn,
        movement_actor=modules.movement_actor,
        offloading_actor=modules.offloading_actor,
        critic=modules.critic,
    )
    legacy_updater = CleanPPOUpdater(
        modules=legacy_modules,
        optimizer=build_single_optimizer(legacy_modules, lr=1e-3),
        config=CleanPPOUpdateConfig(),
        device="cpu",
    )
    legacy_advantages = torch.tensor([-1.0, 1.0])
    legacy_loss = legacy_updater._loss(
        records=records,
        returns=torch.tensor([-1.0, 2.0]),
        old_values=torch.zeros(2),
        value_target_mean=torch.zeros(()),
        value_target_scale=torch.ones(()),
        advantages=legacy_advantages,
    )
    reference_terms = []
    for slot_index, record in enumerate(records):
        task_features = torch.as_tensor(record.graph_snapshot.task_features, dtype=torch.float32)
        incidence = torch.as_tensor(record.graph_snapshot.incidence_matrix, dtype=torch.float32)
        embeddings = legacy_modules.hgnn(task_features, incidence)
        offloading = record.offloading_records[0]
        task_embedding = embeddings[offloading.task_local_index].reshape(1, -1)
        dynamic = torch.as_tensor(offloading.dynamic_uav_features, dtype=torch.float32)
        pair = torch.as_tensor(offloading.pair_features, dtype=torch.float32)
        features = torch.cat([task_embedding.expand(dynamic.shape[0], -1), dynamic, pair], dim=1)
        mask = torch.as_tensor(offloading.candidate_mask, dtype=torch.bool)
        logits = legacy_modules.offloading_actor.scorer(features).masked_fill(
            ~mask, torch.finfo(torch.float32).min
        )
        distribution = torch.distributions.Categorical(logits=logits)
        new_log_prob = distribution.log_prob(torch.tensor(offloading.selected_action))
        ratio = torch.exp(new_log_prob - float(offloading.old_log_probability))
        advantage = legacy_advantages[slot_index]
        reference_terms.append(
            -torch.minimum(
                ratio * advantage,
                torch.clamp(ratio, 0.8, 1.2) * advantage,
            )
        )
    _assert(
        torch.equal(legacy_loss["offloading_loss"], torch.stack(reference_terms).mean()),
        "disabled mode must preserve the original slot-level offloading PPO loss",
    )
    try:
        CleanPPOUpdater(
            modules=legacy_modules,
            optimizer=build_single_optimizer(legacy_modules, lr=1e-3),
            config=CleanPPOUpdateConfig(offloading_counterfactual_coef=0.25),
            device="cpu",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("half-enabled counterfactual coefficients must be rejected")

    for beta, eta in ((0.25, 0.0), (0.0, 0.5)):
        try:
            train_clean_mainline._validated_offloading_action_value_controls(beta, eta)
        except ValueError:
            pass
        else:
            raise AssertionError("train CLI controls must reject a half-enabled pair")
    checkpoint_controls = train_clean_mainline.checkpoint_experiment_controls(
        {
            "config": {
                "cli": {
                    "completed_dag_weight": 16.0,
                    "offloading_counterfactual_coef": 0.25,
                    "offloading_action_value_loss_coef": 0.5,
                }
            }
        }
    )
    _assert(checkpoint_controls["offloading_counterfactual_coef"] == 0.25, "checkpoint beta provenance mismatch")
    _assert(checkpoint_controls["offloading_action_value_loss_coef"] == 0.5, "checkpoint eta provenance mismatch")
    resume_args = train_clean_mainline.build_arg_parser().parse_args(
        [
            "--completed-dag-weight",
            "16",
            "--offloading-counterfactual-coef",
            "0.5",
            "--offloading-action-value-loss-coef",
            "0.5",
        ]
    )
    try:
        train_clean_mainline.validate_resume_experiment_controls(
            resume_args,
            {
                "config": {
                    "cli": {
                        "completed_dag_weight": 16.0,
                        "detach_critic_hgnn": False,
                        "freeze_ue_mobility": False,
                        "offloading_counterfactual_coef": 0.25,
                        "offloading_action_value_loss_coef": 0.5,
                    }
                }
            },
        )
    except ValueError:
        pass
    else:
        raise AssertionError("resume must reject a counterfactual coefficient mismatch")

    eval_controls = {
        "offloading_counterfactual_coef": 0.25,
        "offloading_action_value_loss_coef": 0.5,
    }
    eval_modules = eval_clean_mainline._build_modules(
        dims={"task_feature_dim": 4, "task_embedding_dim": embedding_dim, "hidden_dim": hidden_dim},
        experiment_controls=eval_controls,
        device=torch.device("cpu"),
    )
    _assert(eval_modules.offloading_action_value_critic is not None, "eval must instantiate enabled Q head")
    eval_payload = {
        "hgnn": eval_modules.hgnn.state_dict(),
        "movement_actor": eval_modules.movement_actor.state_dict(),
        "offloading_actor": eval_modules.offloading_actor.state_dict(),
        "critic": eval_modules.critic.state_dict(),
        "offloading_action_value_critic": eval_modules.offloading_action_value_critic.state_dict(),
    }
    eval_clean_mainline._load_module_state(eval_modules, eval_payload)
    missing_q_payload = dict(eval_payload)
    missing_q_payload.pop("offloading_action_value_critic")
    try:
        eval_clean_mainline._load_module_state(eval_modules, missing_q_payload)
    except ValueError:
        pass
    else:
        raise AssertionError("eval must reject an enabled checkpoint missing Q state")

    print("smoke_clean_offloading_counterfactual passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
