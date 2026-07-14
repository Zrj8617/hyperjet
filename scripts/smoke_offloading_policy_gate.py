from __future__ import annotations

from pathlib import Path
import json
import os
import random
import shutil
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.assignment import OffloadingCandidateEstimate
from environment.env import Env
from environment.graph_builder import CleanGraphBuilder
from scripts import offloading_policy_gate as gate


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    try:
        import torch
        from torch import nn
        from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            print("smoke_offloading_policy_gate skipped: torch is not installed")
            return
        raise

    _test_legacy_actor_regression(torch, CleanOffloadingActor)
    _test_policy_semantics_and_rng(torch, nn)
    _test_realization_and_summary()
    _test_full_eval_entrypoint(torch)
    print("smoke_offloading_policy_gate passed")


def _test_legacy_actor_regression(torch, actor_cls) -> None:
    import config

    original_arrival_prob = config.DAG_BASE_ARRIVAL_PROB
    config.DAG_BASE_ARRIVAL_PROB = 0.0
    graph_builder = CleanGraphBuilder()
    try:
        env = Env()
        env.reset()
        ue = env.ues[0]
        job = env.task_manager.create_dag_for_ue(
            ue_id=ue.id,
            source_pos=ue.pos[:2].copy(),
            current_time_step=env.current_time_seconds,
        )
        ue.enter_service_waiting(job.dag_id)
        env.task_manager.refresh_ready_states()
        snapshot = graph_builder.build(env.task_manager, env.uavs, env.time_step, executor=env.executor)
        ready_tasks = env.task_manager.get_ready_tasks()
        task_embedding_dim = 8
        embeddings = torch.linspace(
            -1.0,
            1.0,
            steps=max(len(snapshot.task_ids) * task_embedding_dim, 1),
            dtype=torch.float32,
        ).reshape(len(snapshot.task_ids), task_embedding_dim)
        torch.manual_seed(123)
        actor = actor_cls(task_embedding_dim=task_embedding_dim, hidden_dim=16)
        kwargs = {
            "frozen_ready_tasks": ready_tasks,
            "task_embeddings": embeddings,
            "graph_snapshot": snapshot,
            "task_manager": env.task_manager,
            "uavs": env.uavs,
            "executor": env.executor,
            "current_time_seconds": env.current_time_seconds,
            "uav_service_positions": {int(uav.id): uav.pos[:2].copy() for uav in env.uavs},
            "ue_service_positions": {int(item.id): item.pos[:2].copy() for item in env.ues},
            "ues": env.ues,
        }
        legacy = actor.act(**kwargs, deterministic=True).to_assignment_dict()
        selected, decisions = gate.select_eval_offloading_actions(
            policy="actor_argmax",
            offloading_actor=actor,
            environment_seed=4242,
            episode=0,
            slot=0,
            checkpoint_path="synthetic.pt",
            checkpoint_model_seed=42,
            **kwargs,
        )
        _assert(selected.to_assignment_dict() == legacy, "actor_argmax must match the legacy deterministic actor")
        _assert(len(decisions) == len(legacy), "actor regression decision count mismatch")
        _assert(all(row["selected_uav_id"] in row["candidate_uav_ids"] for row in decisions), "selection must map to a candidate")
    finally:
        graph_builder.close()
        config.DAG_BASE_ARRIVAL_PROB = original_arrival_prob


def _test_policy_semantics_and_rng(torch, nn) -> None:
    class _Scorer(nn.Module):
        def forward(self, features):
            return torch.linspace(2.0, 1.0, steps=features.shape[0], device=features.device)

    class _Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.task_embedding_dim = 2
            self.scorer = _Scorer()

    tasks = [
        SimpleNamespace(task_id="task_0", dag_id="dag_0", ready_time=0.0),
        SimpleNamespace(task_id="task_1", dag_id="dag_0", ready_time=0.0),
    ]
    snapshot = SimpleNamespace(task_id_to_idx={"task_0": 0, "task_1": 1})
    uavs = [SimpleNamespace(id=0), SimpleNamespace(id=1)]
    executor = SimpleNamespace(uav_queues={0: [], 1: []}, task_records={}, uav_available_time={0: 0.0, 1: 0.0})
    actor = _Actor()
    original_builder = gate.build_offloading_candidate_components

    def fake_builder(*, task, state_view, **kwargs):
        del kwargs
        q0 = int(state_view.queue_lengths.get(0, 0))
        finishes = [5.0, 5.0] if task.task_id == "task_tie" else [5.0 + 10.0 * q0, 6.0]
        mask = np.asarray([task.task_id != "task_illegal", True], dtype=bool)
        estimates = [
            OffloadingCandidateEstimate(
                task_id=task.task_id,
                uav_id=idx,
                legal=bool(mask[idx]),
                dynamic_uav_features=np.zeros(7, dtype=np.float32),
                pair_features=np.zeros(8, dtype=np.float32),
                estimated_finish_time=finish,
                estimated_queued_workload=1.0,
            )
            for idx, finish in enumerate(finishes)
        ]
        return (
            np.zeros((2, 7), dtype=np.float32),
            np.zeros((2, 8), dtype=np.float32),
            mask,
            [0, 1],
            estimates,
        )

    kwargs = {
        "offloading_actor": actor,
        "frozen_ready_tasks": tasks,
        "task_embeddings": torch.zeros((2, 2)),
        "graph_snapshot": snapshot,
        "task_manager": SimpleNamespace(),
        "uavs": uavs,
        "executor": executor,
        "current_time_seconds": 0.0,
        "environment_seed": 4242,
        "episode": 0,
        "slot": 3,
        "checkpoint_path": "synthetic.pt",
        "checkpoint_model_seed": 42,
    }
    gate.build_offloading_candidate_components = fake_builder
    try:
        greedy, greedy_rows = gate.select_eval_offloading_actions(policy="greedy_eft_teacher", **kwargs)
        _assert([entry.uav_id for entry in greedy.entries] == [0, 1], "greedy must use sequential reservation")
        _assert(all(entry.uav_id in {0, 1} for entry in greedy.entries), "greedy produced an illegal UAV")
        shortest, _ = gate.select_eval_offloading_actions(policy="shortest_queue", **kwargs)
        _assert([entry.uav_id for entry in shortest.entries] == [0, 1], "shortest queue tie-break/reservation mismatch")
        _assert(greedy_rows[0]["selected_estimated_regret"] == 0.0, "greedy regret must be zero")

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        random_one, _ = gate.select_eval_offloading_actions(policy="random_hash", **kwargs)
        random_two, _ = gate.select_eval_offloading_actions(policy="random_hash", **kwargs)
        _assert(random_one.to_assignment_dict() == random_two.to_assignment_dict(), "random_hash must be reproducible")
        _assert(random.getstate() == python_state, "random_hash changed Python RNG")
        after_numpy = np.random.get_state()
        _assert(numpy_state[0] == after_numpy[0] and np.array_equal(numpy_state[1], after_numpy[1]) and numpy_state[2:] == after_numpy[2:], "random_hash changed NumPy RNG")
        _assert(torch.equal(torch_state, torch.random.get_rng_state()), "random_hash changed Torch RNG")
        if torch.cuda.is_available():
            _assert(
                all(torch.equal(before, after) for before, after in zip(cuda_states, torch.cuda.get_rng_state_all())),
                "random_hash changed CUDA RNG",
            )

        illegal_task = SimpleNamespace(task_id="task_illegal", dag_id="dag_0", ready_time=0.0)
        illegal_kwargs = dict(kwargs)
        illegal_kwargs.update(
            frozen_ready_tasks=[illegal_task],
            task_embeddings=torch.zeros((1, 2)),
            graph_snapshot=SimpleNamespace(task_id_to_idx={"task_illegal": 0}),
        )
        for policy in gate.OFFLOADING_POLICIES:
            legal_only, _ = gate.select_eval_offloading_actions(policy=policy, **illegal_kwargs)
            _assert(legal_only.entries[0].uav_id == 1, f"{policy} must ignore an illegal candidate")

        tie_task = SimpleNamespace(task_id="task_tie", dag_id="dag_0", ready_time=0.0)
        tie_kwargs = dict(kwargs)
        tie_kwargs.update(
            frozen_ready_tasks=[tie_task],
            task_embeddings=torch.zeros((1, 2)),
            graph_snapshot=SimpleNamespace(task_id_to_idx={"task_tie": 0}),
        )
        for policy in ("greedy_eft_teacher", "shortest_queue"):
            tied, _ = gate.select_eval_offloading_actions(policy=policy, **tie_kwargs)
            _assert(tied.entries[0].uav_id == 0, f"{policy} must break a full tie by UAV id")
    finally:
        gate.build_offloading_candidate_components = original_builder


def _test_realization_and_summary() -> None:
    decision = {
        "task_id": "task_0",
        "valid_candidate_count": 2,
        "actor_normalized_entropy": 0.9,
        "actor_top1_top2_margin": 0.1,
        "actor_selected_uav_id": 0,
        "greedy_eft_selected_uav_id": 0,
        "selected_estimated_regret": 1.0,
        "selected_estimated_finish": 9.0,
        "selected_finish_error": None,
    }
    record = SimpleNamespace(
        assignment_time=1.0,
        start_time=5.0,
        compute_finish_time=8.0,
        finish_time=10.0,
        upload_time=1.0,
        inter_transfer_time=0.5,
        compute_time=3.0,
        return_time=2.0,
        completed=True,
    )
    env = SimpleNamespace(executor=SimpleNamespace(task_records={"task_0": record}))
    gate.finalize_decision_realizations([decision], env=env)
    _assert(np.isclose(decision["realized_queue_resource_wait"], 2.5), "realized queue wait mismatch")
    _assert(np.isclose(decision["selected_finish_error"], 1.0), "calibration error mismatch")
    summary = gate.summarize_offloading_decisions([decision])
    _assert(summary["estimator_calibration_count"] == 1, "calibration count mismatch")
    _assert(np.isclose(summary["estimator_calibration_mae"], 1.0), "calibration MAE mismatch")


def _test_full_eval_entrypoint(torch) -> None:
    import config
    from scripts import eval_clean_mainline

    temp_root = ROOT / ".codex_tmp_offloading_gate_full_eval" / f"smoke_{os.getpid()}"
    checkpoint = temp_root / "synthetic.pt"
    original_arrival_probability = config.DAG_BASE_ARRIVAL_PROB
    original_kahypar_enabled = config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES
    try:
        temp_root.mkdir(parents=True)
        dims = {
            # GraphBuilder appends the normalized topological index to the 11
            # base DAG task features used by the clean checkpoint.
            "task_feature_dim": int(config.DAG_TASK_FEATURE_DIM) + 1,
            "task_embedding_dim": 8,
            "hidden_dim": 16,
        }
        torch.manual_seed(4242)
        modules = eval_clean_mainline._build_modules(
            dims=dims,
            experiment_controls={
                "offloading_counterfactual_coef": 0.0,
                "offloading_action_value_loss_coef": 0.0,
            },
            device=torch.device("cpu"),
        )
        torch.save(
            {
                "hgnn": modules.hgnn.state_dict(),
                "movement_actor": modules.movement_actor.state_dict(),
                "offloading_actor": modules.offloading_actor.state_dict(),
                "critic": modules.critic.state_dict(),
                "config": {
                    "cli": {
                        "seed": 42,
                        "task_embedding_dim": 8,
                        "hidden_dim": 16,
                        "completed_dag_weight": 16.0,
                        "detach_critic_hgnn": False,
                        "freeze_ue_mobility": False,
                    }
                },
            },
            checkpoint,
        )
        config.DAG_BASE_ARRIVAL_PROB = 1.0
        config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES = False
        movement_distributions = []
        for policy in gate.OFFLOADING_POLICIES:
            args = eval_clean_mainline.build_arg_parser().parse_args(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--episodes",
                    "1",
                    "--arrival-steps",
                    "1",
                    "--max-drain-steps",
                    "0",
                    "--seed",
                    "4242",
                    "--device",
                    "cpu",
                    "--output-dir",
                    str(temp_root / policy),
                    "--run-name",
                    policy,
                    "--offloading-policy",
                    policy,
                ]
            )
            summary = eval_clean_mainline.run_evaluation(args)
            _assert(summary["status"] == "completed", f"full eval did not complete for {policy}")
            _assert(summary["offloading_policy"] == policy, "summary policy provenance mismatch")
            _assert(summary["checkpoint_model_seed"] == 42, "checkpoint seed provenance mismatch")
            _assert(summary["completed_dag_weight"] == 16.0, "checkpoint reward provenance mismatch")
            _assert(summary["detach_critic_hgnn"] is False, "checkpoint detach provenance mismatch")
            _assert(summary["freeze_ue_mobility"] is False, "moving UE provenance mismatch")
            run_dir = Path(summary["run_dir"])
            metric_lines = [line for line in (run_dir / "eval_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
            decision_lines = [line for line in (run_dir / "offloading_decisions.jsonl").read_text(encoding="utf-8").splitlines() if line]
            _assert(len(metric_lines) == 1, "full eval must write one episode row")
            row = json.loads(metric_lines[0])
            _assert(row["offloading_action_count"] > 0, "synthetic full eval should make an offloading decision")
            _assert(len(decision_lines) == row["offloading_action_count"], "decision JSONL count mismatch")
            _assert(row["environment_seed"] == 4242, "environment seed provenance mismatch")
            _assert(row["offloading_policy"] == policy, "episode policy provenance mismatch")
            config_payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            _assert(config_payload["protocol"]["offloading_policy"] == policy, "config policy mismatch")
            _assert(config_payload["protocol"]["ue_mobility_mode"] == "moving", "config UE mobility mismatch")
            movement_distributions.append(row["movement_action_distribution"])
        _assert(
            all(distribution == movement_distributions[0] for distribution in movement_distributions[1:]),
            "single-slot movement must be identical when only offloading policy changes",
        )
    finally:
        config.DAG_BASE_ARRIVAL_PROB = original_arrival_probability
        config.ENABLE_KAHYPAR_PARTITION_HYPEREDGES = original_kahypar_enabled
        if temp_root.exists():
            shutil.rmtree(temp_root)
        parent = temp_root.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()


if __name__ == "__main__":
    main()
