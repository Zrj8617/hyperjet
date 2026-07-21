from __future__ import annotations

from dataclasses import fields
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


def _expect_failure(callable_, message: str) -> None:
    try:
        callable_()
    except (ValueError, FloatingPointError, RuntimeError):
        return
    raise AssertionError(message)


def main() -> int:
    try:
        import torch
    except ModuleNotFoundError:
        print("smoke_clean_lagged_residual_q skipped: torch is not installed")
        return 0

    import config
    from environment.assignment import (
        CLEAN_OFFLOADING_PAIR_FEATURE_DIM,
        CLEAN_OFFLOADING_UAV_FEATURE_DIM,
    )
    from marl_models.hgnn import CleanIncidenceHGNN
    from marl_models.mappo.clean_lagged_residual_q import (
        CleanLaggedOutcomeTracker,
        CleanLaggedQSample,
        CleanLaggedResidualQCritic,
        build_rng_neutral_lagged_residual_q_critic,
        lagged_residual_target,
    )
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
        _lagged_q_regression_loss,
        build_single_optimizer,
    )
    from scripts import eval_clean_mainline, train_clean_mainline

    completed_target, completed_residual = lagged_residual_target(
        assignment_time=10.0,
        outcome_time=70.0,
        estimated_incremental_delay=20.0,
        scale_seconds=200.0,
        censored=False,
    )
    _assert(math.isclose(completed_residual, 40.0), "completed residual arithmetic mismatch")
    _assert(
        math.isclose(completed_target, -math.tanh(0.2)),
        "completed residual target mismatch",
    )
    censored_target, censored_residual = lagged_residual_target(
        assignment_time=10.0,
        outcome_time=20.0,
        estimated_incremental_delay=50.0,
        scale_seconds=200.0,
        censored=True,
    )
    _assert(censored_residual == 0.0 and censored_target == 0.0, "censor lower bound mismatch")
    _expect_failure(
        lambda: lagged_residual_target(
            assignment_time=2.0,
            outcome_time=1.0,
            estimated_incremental_delay=0.0,
            scale_seconds=200.0,
            censored=False,
        ),
        "outcome before assignment must fail",
    )
    _expect_failure(
        lambda: lagged_residual_target(
            assignment_time=0.0,
            outcome_time=1.0,
            estimated_incremental_delay=0.0,
            scale_seconds=float("nan"),
            censored=False,
        ),
        "non-finite target input must fail",
    )

    tracker = CleanLaggedOutcomeTracker(scale_seconds=200.0, censor_weight=0.25)
    tracker.start_episode(7)
    jobs = {
        "dag_done": SimpleNamespace(completed=False, return_complete_time=None),
        "dag_censored": SimpleNamespace(completed=False, return_complete_time=None),
    }
    tasks = {
        "task_done": SimpleNamespace(dag_id="dag_done"),
        "task_censored": SimpleNamespace(dag_id="dag_censored"),
    }
    manager = SimpleNamespace(
        get_task=lambda task_id: tasks.get(task_id),
        get_job=lambda dag_id: jobs.get(dag_id),
    )
    env = SimpleNamespace(task_manager=manager)
    base_record = dict(
        task_local_index=0,
        decision_order=0,
        candidate_uav_ids=[0, 1],
        dynamic_uav_features=np.zeros((2, 1), dtype=np.float32),
        pair_features=np.zeros((2, 1), dtype=np.float32),
        candidate_mask=np.asarray([True, True]),
        selected_action=1,
        selected_uav_id=1,
        old_log_probability=-math.log(2.0),
        entropy=math.log(2.0),
        candidate_features=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        critic_global_context=np.asarray([5.0, 6.0], dtype=np.float32),
        selected_estimated_finish_time=25.0,
        selected_estimated_incremental_delay=15.0,
    )
    done_record = CleanOffloadingRolloutRecord(
        task_id="task_done",
        dag_id="dag_done",
        assignment_time_seconds=10.0,
        **base_record,
    )
    censored_record = CleanOffloadingRolloutRecord(
        task_id="task_censored",
        dag_id="dag_censored",
        assignment_time_seconds=12.0,
        **base_record,
    )
    tracker.register_rollout_actions(
        slot_record=SimpleNamespace(slot_index=3, offloading_records=[done_record]), env=env
    )
    _assert(tracker.pending_count == 1 and tracker.pop_finalized() == [], "pending action must cross a rollout")
    tracker.register_rollout_actions(
        slot_record=SimpleNamespace(slot_index=4, offloading_records=[censored_record]), env=env
    )
    jobs["dag_done"].completed = True
    jobs["dag_done"].return_complete_time = 55.0
    _assert(tracker.resolve_completed(env=env) == 1, "completed DAG must resolve its action")
    completed_samples = tracker.pop_finalized()
    _assert(len(completed_samples) == 1 and not completed_samples[0].censored, "completed sample mismatch")
    _assert(completed_samples[0].selected_input.tolist() == [3.0, 4.0, 5.0, 6.0], "selected input mismatch")
    tracker.finalize_censored(episode_end_time=80.0)
    censored_samples = tracker.pop_finalized()
    _assert(len(censored_samples) == 1 and censored_samples[0].censored, "censored sample mismatch")
    _assert(censored_samples[0].weight == 0.25, "censor weight mismatch")
    _assert(
        all(name not in {"old_log_prob", "old_log_probability"} for name in (f.name for f in fields(CleanLaggedQSample))),
        "historical Q samples must not carry PPO log probabilities",
    )
    tracker_summary = tracker.finish_episode()
    _assert(
        tracker_summary == {
            "registered": 2,
            "completed": 1,
            "censored": 1,
            "unresolved_before_censoring": 1,
            "pending_after_clear": 0,
        },
        "tracker terminal summary mismatch",
    )
    tracker.start_episode(8)
    tracker.register_rollout_actions(
        slot_record=SimpleNamespace(slot_index=0, offloading_records=[done_record]), env=env
    )
    _assert(tracker.resolve_completed(env=env) == 1, "task IDs may repeat after an episode reset")
    tracker.pop_finalized()
    tracker.finish_episode()

    embedding_dim = 6
    hidden_dim = 10
    task_feature_dim = 4
    critic_input_dim = clean_critic_input_dim(embedding_dim, config.NUM_UAVS)
    actor = CleanOffloadingActor(task_embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    torch.manual_seed(9876)
    rng_before_q_build = torch.random.get_rng_state().clone()
    rng_neutral_q = build_rng_neutral_lagged_residual_q_critic(
        input_dim=actor.candidate_feature_dim + critic_input_dim,
        hidden_dim=hidden_dim,
    )
    _assert(
        torch.equal(torch.random.get_rng_state(), rng_before_q_build),
        "auxiliary Q construction must not advance the actor Torch RNG stream",
    )
    _assert(
        torch.equal(rng_neutral_q(torch.randn(2, rng_neutral_q.input_dim)), torch.zeros(2)),
        "RNG-neutral Q must retain exact zero output initialization",
    )
    lagged_q = CleanLaggedResidualQCritic(
        input_dim=actor.candidate_feature_dim + critic_input_dim,
        hidden_dim=hidden_dim,
    )
    _assert(
        torch.equal(lagged_q(torch.randn(3, lagged_q.input_dim)), torch.zeros(3)),
        "lagged Q output layer must initialize to zero",
    )
    modules = CleanTrainingModules(
        hgnn=CleanIncidenceHGNN(
            task_feature_dim=task_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
        ),
        movement_actor=CleanMovementActor(task_embedding_dim=embedding_dim, hidden_dim=hidden_dim),
        offloading_actor=actor,
        critic=CleanCentralizedCritic(input_dim=critic_input_dim, hidden_dim=hidden_dim),
        offloading_lagged_q_critic=lagged_q,
    )
    updater = CleanPPOUpdater(
        modules=modules,
        optimizer=build_single_optimizer(modules, lr=1e-3),
        config=CleanPPOUpdateConfig(
            ppo_epochs=2,
            offloading_lagged_q_coef=0.25,
            offloading_lagged_q_loss_coef=0.5,
        ),
        device="cpu",
    )
    candidate_count = 2
    dynamic = np.linspace(
        0.1,
        0.9,
        candidate_count * CLEAN_OFFLOADING_UAV_FEATURE_DIM,
        dtype=np.float32,
    ).reshape(candidate_count, CLEAN_OFFLOADING_UAV_FEATURE_DIM)
    pair = np.linspace(
        0.2,
        1.0,
        candidate_count * CLEAN_OFFLOADING_PAIR_FEATURE_DIM,
        dtype=np.float32,
    ).reshape(candidate_count, CLEAN_OFFLOADING_PAIR_FEATURE_DIM)
    snapshot = SimpleNamespace(
        task_features=np.asarray([[1.0, 0.2, 0.4, 0.8]], dtype=np.float32),
        incidence_matrix=np.ones((1, 1), dtype=np.float32),
    )
    with torch.no_grad():
        embedding = modules.hgnn(
            torch.as_tensor(snapshot.task_features), torch.as_tensor(snapshot.incidence_matrix)
        )[0]
        candidate_features = torch.cat(
            [
                embedding.reshape(1, -1).expand(candidate_count, -1),
                torch.as_tensor(dynamic),
                torch.as_tensor(pair),
            ],
            dim=1,
        ).numpy()
    context = np.linspace(-0.5, 0.5, critic_input_dim, dtype=np.float32)
    offloading = CleanOffloadingRolloutRecord(
        task_id="task_train",
        task_local_index=0,
        decision_order=0,
        candidate_uav_ids=[0, 1],
        dynamic_uav_features=dynamic,
        pair_features=pair,
        candidate_mask=np.asarray([True, True]),
        selected_action=1,
        selected_uav_id=1,
        old_log_probability=-math.log(2.0),
        entropy=math.log(2.0),
        dag_id="dag_train",
        assignment_time_seconds=1.0,
        candidate_features=candidate_features.copy(),
        critic_global_context=context.copy(),
        selected_estimated_finish_time=20.0,
        selected_estimated_incremental_delay=19.0,
    )
    slot = CleanSlotRolloutRecord(
        slot_index=0,
        graph_snapshot=snapshot,
        critic_non_graph_input=context[embedding_dim:].copy(),
        value=0.0,
        reward=1.0,
        terminated=True,
        offloading_records=[offloading],
    )
    historical_sample = CleanLaggedQSample(
        episode_index=0,
        slot_index=0,
        task_id="old_task",
        dag_id="old_dag",
        selected_input=np.concatenate([candidate_features[1], context]).astype(np.float32),
        target=-0.5,
        weight=1.0,
        censored=False,
        residual_seconds=100.0,
    )
    q_loss, q_diagnostics = _lagged_q_regression_loss(
        module=lagged_q, samples=[historical_sample], device="cpu"
    )
    _assert(math.isclose(float(q_loss.item()), 0.125), "zero-init smooth-L1 arithmetic mismatch")
    q_grads = torch.autograd.grad(q_loss, list(lagged_q.parameters()), allow_unused=True)
    _assert(any(g is not None and float(g.abs().sum()) > 0.0 for g in q_grads), "Q loss must train Q")
    _assert(q_diagnostics["offloading_lagged_q_training_sample_count"] == 1, "Q sample diagnostic mismatch")
    empty_q_loss, empty_q_diagnostics = _lagged_q_regression_loss(
        module=lagged_q, samples=[], device="cpu"
    )
    _assert(empty_q_loss.item() == 0.0, "empty Q sample batch must have zero loss")
    _assert(
        empty_q_diagnostics["offloading_lagged_q_training_sample_count"] == 0,
        "empty Q sample diagnostic mismatch",
    )

    zero_corrections, _ = updater._precompute_lagged_q_corrections([slot])
    _assert(float(zero_corrections[0].item()) == 0.0, "zero Q must give zero correction")
    offloading.candidate_features = np.zeros_like(candidate_features)
    offloading.candidate_features[0, 0] = 1.0
    offloading.candidate_features[1, 0] = 2.0
    with torch.no_grad():
        for parameter in lagged_q.parameters():
            parameter.zero_()
        lagged_q.net[0].weight[0, 0] = 1.0
        lagged_q.net[2].weight[0, 0] = 1.0
        lagged_q.net[4].weight[0, 0] = 0.1
    frozen_corrections, correction_diags = updater._precompute_lagged_q_corrections([slot])
    _assert(len(frozen_corrections) == 1, "one current action must produce one frozen correction")
    _assert(
        correction_diags["offloading_lagged_q_legal_spread_mean"] > 0.0
        and float(frozen_corrections[0].item()) > 0.0,
        "candidate-dependent Q must give a candidate-dependent correction",
    )
    q_state_before_mutation = {
        key: value.detach().clone() for key, value in lagged_q.state_dict().items()
    }
    first_loss = updater._loss(
        records=[slot],
        returns=torch.tensor([1.0]),
        advantages=torch.tensor([1.0]),
        lagged_q_samples=[historical_sample],
        frozen_lagged_corrections=frozen_corrections,
    )["offloading_loss"].detach().clone()
    with torch.no_grad():
        for parameter in lagged_q.parameters():
            parameter.add_(10.0)
    second_loss = updater._loss(
        records=[slot],
        returns=torch.tensor([1.0]),
        advantages=torch.tensor([1.0]),
        lagged_q_samples=[historical_sample],
        frozen_lagged_corrections=frozen_corrections,
    )["offloading_loss"].detach().clone()
    _assert(torch.equal(first_loss, second_loss), "frozen correction must not change during PPO epochs")
    lagged_q.load_state_dict(q_state_before_mutation)
    _assert(correction_diags["offloading_lagged_q_current_action_count"] == 1, "current action diagnostic mismatch")

    buffer = CleanSlotRolloutBuffer()
    buffer.append(slot)
    buffer.close(bootstrap_value=0.0)
    stats = updater.update(buffer, lagged_q_samples=[historical_sample], lagged_q_pending_count=2)
    _assert(math.isfinite(stats.offloading_lagged_q_loss), "lagged Q loss must be finite")
    _assert(stats.diagnostics["offloading_lagged_q_pending_count"] == 2, "pending diagnostic mismatch")
    _assert(stats.diagnostics["offloading_lagged_q_direct_grad_q"] > 0.0, "Q direct gradient must be positive")
    _assert(stats.diagnostics["offloading_lagged_q_direct_grad_hgnn"] == 0.0, "Q loss must not train HGNN")
    _assert(stats.diagnostics["offloading_lagged_q_direct_grad_actor"] == 0.0, "Q loss must not train actor")
    _assert(stats.diagnostics["offloading_lagged_q_direct_grad_critic"] == 0.0, "Q loss must not train critic")

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
    legacy_loss = legacy_updater._loss(
        records=[slot], returns=torch.tensor([1.0]), advantages=torch.tensor([1.0])
    )
    _assert(legacy_loss["offloading_lagged_q_loss"].item() == 0.0, "default-off Q loss must be zero")
    _assert(legacy_modules.offloading_lagged_q_critic is None, "default mode must not instantiate Q")

    for beta, eta in ((0.25, 0.0), (0.0, 0.5)):
        _expect_failure(
            lambda beta=beta, eta=eta: train_clean_mainline._validated_offloading_lagged_q_controls(
                beta, eta, 200.0, 0.25
            ),
            "half-enabled lagged Q controls must fail",
        )
    _expect_failure(
        lambda: train_clean_mainline._resolved_offloading_lagged_q_controls(
            SimpleNamespace(
                offloading_counterfactual_coef=0.25,
                offloading_action_value_loss_coef=0.5,
                offloading_lagged_q_coef=0.25,
                offloading_lagged_q_loss_coef=0.5,
                offloading_lagged_q_scale_seconds=200.0,
                offloading_lagged_q_censor_weight=0.25,
            )
        ),
        "v1 and v2 must be mutually exclusive",
    )
    controls = train_clean_mainline.checkpoint_experiment_controls({})
    _assert(controls["offloading_lagged_q_coef"] == 0.0, "legacy checkpoint beta default mismatch")
    _assert(controls["offloading_lagged_q_loss_coef"] == 0.0, "legacy checkpoint eta default mismatch")
    _assert(controls["offloading_lagged_q_scale_seconds"] == 200.0, "legacy checkpoint scale mismatch")
    _assert(controls["offloading_lagged_q_censor_weight"] == 0.25, "legacy checkpoint censor mismatch")

    args = train_clean_mainline.build_arg_parser().parse_args(
        [
            "--offloading-lagged-q-coef", "0.25",
            "--offloading-lagged-q-loss-coef", "0.5",
            "--offloading-lagged-q-scale-seconds", "200",
            "--offloading-lagged-q-censor-weight", "0.25",
        ]
    )
    checkpoint_config = {
        "config": {
            "cli": {
                "completed_dag_weight": float(config.REWARD_COMPLETED_DAG_WEIGHT),
                "offloading_lagged_q_coef": 0.25,
                "offloading_lagged_q_loss_coef": 0.5,
                "offloading_lagged_q_scale_seconds": 200.0,
                "offloading_lagged_q_censor_weight": 0.25,
                "normalize_value_targets": True,
                "value_clip_epsilon": 0.2,
            }
        }
    }
    train_clean_mainline.validate_resume_experiment_controls(args, checkpoint_config)
    mismatch = train_clean_mainline.build_arg_parser().parse_args(
        [
            "--offloading-lagged-q-coef", "0.5",
            "--offloading-lagged-q-loss-coef", "0.5",
        ]
    )
    _expect_failure(
        lambda: train_clean_mainline.validate_resume_experiment_controls(mismatch, checkpoint_config),
        "resume must reject lagged Q mismatch",
    )

    temp_dir = ROOT / ".codex_tmp_lagged_residual_q" / f"smoke_{os.getpid()}"
    try:
        temp_dir.mkdir(parents=True, exist_ok=False)
        manager = CleanCheckpointManager(temp_dir / "checkpoints")
        checkpoint = manager.save(
            modules=modules,
            optimizer=updater.optimizer,
            episode=1,
            global_slot=1,
            update_step=updater.update_step,
            config_snapshot=checkpoint_config["config"],
            safe_boundary=True,
        )
        payload = manager.read(checkpoint)
        _assert("offloading_lagged_residual_q" in payload, "checkpoint must persist lagged Q")
        manager.restore(modules=modules, optimizer=updater.optimizer, payload=payload)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        parent = temp_dir.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()

    eval_modules = eval_clean_mainline._build_modules(
        dims={
            "task_feature_dim": task_feature_dim,
            "task_embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
        },
        experiment_controls={
            "offloading_lagged_q_coef": 0.25,
            "offloading_lagged_q_loss_coef": 0.5,
        },
        device=torch.device("cpu"),
    )
    _assert(eval_modules.offloading_lagged_q_critic is not None, "eval must instantiate lagged Q")
    eval_payload = {
        "hgnn": eval_modules.hgnn.state_dict(),
        "movement_actor": eval_modules.movement_actor.state_dict(),
        "offloading_actor": eval_modules.offloading_actor.state_dict(),
        "critic": eval_modules.critic.state_dict(),
        "offloading_lagged_residual_q": eval_modules.offloading_lagged_q_critic.state_dict(),
    }
    eval_clean_mainline._load_module_state(eval_modules, eval_payload)
    missing_q = dict(eval_payload)
    missing_q.pop("offloading_lagged_residual_q")
    _expect_failure(
        lambda: eval_clean_mainline._load_module_state(eval_modules, missing_q),
        "eval must reject missing lagged Q state",
    )

    print("smoke_clean_lagged_residual_q passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
