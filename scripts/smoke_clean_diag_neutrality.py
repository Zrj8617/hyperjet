"""Behavior-neutrality smoke for the Phase 4 Commit 1 diagnostics.

Verifies that the pure-diagnostics additions to CleanPPOUpdater do not change
training behavior:

  1. Build ONE rollout buffer from a real short env episode.
  2. Run the PPO update on two identical module copies, once with
     hgnn_grad_decomposition_interval=0 (decomposition off) and once with =1
     (decomposition every update).
  3. Assert bitwise-identical parameters after the update, identical losses,
     and that torch.autograd.grad never polluted .grad.

Requires torch; forces CPU for determinism.
"""

from __future__ import annotations

import copy
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    try:
        import torch
    except ModuleNotFoundError:
        print("smoke_clean_diag_neutrality skipped: torch is not available.")
        return 0

    from environment.env import Env
    from environment.graph_builder import CleanGraphBuilder
    from environment.dag_tasks import TASK_STATE_READY_UNSCHEDULED
    from marl_models.hgnn import CleanIncidenceHGNN
    from marl_models.mappo.clean_movement_actor import CleanMovementActor
    from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
    from marl_models.mappo.clean_ppo import CleanCentralizedCritic, clean_critic_input_dim
    from marl_models.mappo.clean_slot_orchestrator import (
        CleanSlotRolloutBuffer,
        encode_prepared_slot,
        prepare_slot_state,
    )
    from marl_models.mappo.clean_trainer import (
        CleanPPOUpdateConfig,
        CleanPPOUpdater,
        CleanTrainingModules,
        build_single_optimizer,
        close_rollout_with_bootstrap,
    )
    from torch.distributions import Categorical

    sys.path.insert(0, str(ROOT / "scripts"))
    from train_clean_mainline import _collect_clean_slot  # noqa: E402

    device = torch.device("cpu")
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)

    env = Env()
    graph_builder = CleanGraphBuilder()
    env.reset()
    graph_builder.reset()
    prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
    task_feature_dim = int(prepared.graph_snapshot.task_features.shape[1])
    emb_dim = 16
    modules = CleanTrainingModules(
        hgnn=CleanIncidenceHGNN(task_feature_dim=task_feature_dim, hidden_dim=16, output_dim=emb_dim),
        movement_actor=CleanMovementActor(task_embedding_dim=emb_dim, hidden_dim=16),
        offloading_actor=CleanOffloadingActor(task_embedding_dim=emb_dim, hidden_dim=16),
        critic=CleanCentralizedCritic(input_dim=clean_critic_input_dim(emb_dim, config.NUM_UAVS), hidden_dim=16),
    )

    # Collect a short real rollout (6 slots).
    encoded = encode_prepared_slot(
        prepared_state=prepared, env=env, hgnn=modules.hgnn, critic=modules.critic,
        movement_actor=modules.movement_actor, device=device,
    )
    buffer = CleanSlotRolloutBuffer()
    for _ in range(6):
        slot_record, done, _ = _collect_clean_slot(
            env=env, modules=modules, encoded_state=encoded, categorical_cls=Categorical,
            device=device, task_state_ready=TASK_STATE_READY_UNSCHEDULED, freeze_movement=False,
        )
        buffer.append(slot_record)
        if done:
            break
        prepared = prepare_slot_state(env=env, graph_builder=graph_builder)
        encoded = encode_prepared_slot(
            prepared_state=prepared, env=env, hgnn=modules.hgnn, critic=modules.critic,
            movement_actor=modules.movement_actor, device=device,
        )
    close_rollout_with_bootstrap(buffer=buffer, next_encoded_state=encoded, terminated=False)
    _assert(len(buffer) >= 3, "neutrality smoke needs a non-trivial rollout.")

    def run_update(decomposition_interval: int):
        mods = CleanTrainingModules(
            hgnn=copy.deepcopy(modules.hgnn),
            movement_actor=copy.deepcopy(modules.movement_actor),
            offloading_actor=copy.deepcopy(modules.offloading_actor),
            critic=copy.deepcopy(modules.critic),
        )
        optimizer = build_single_optimizer(mods, lr=1e-3)
        updater = CleanPPOUpdater(
            modules=mods, optimizer=optimizer,
            config=CleanPPOUpdateConfig(ppo_epochs=2, hgnn_grad_decomposition_interval=decomposition_interval),
            device=device,
        )
        stats = updater.update(buffer)
        grads = {}
        for name in ("hgnn", "movement_actor", "offloading_actor", "critic"):
            for pname, param in getattr(mods, name).named_parameters():
                grads[f"{name}.{pname}"] = None if param.grad is None else param.grad.detach().clone()
        return mods, stats, grads

    mods_off, stats_off, grads_off = run_update(0)
    mods_on, stats_on, grads_on = run_update(1)

    # 1) Parameters must be bitwise identical.
    for name in ("hgnn", "movement_actor", "offloading_actor", "critic"):
        sd_off = getattr(mods_off, name).state_dict()
        sd_on = getattr(mods_on, name).state_dict()
        _assert(sd_off.keys() == sd_on.keys(), f"{name} state dict keys differ.")
        for key in sd_off:
            _assert(torch.equal(sd_off[key], sd_on[key]), f"{name}.{key} differs with diagnostics on.")

    # 2) Loss scalars must match exactly.
    for field_name in ("movement_loss", "offloading_loss", "value_loss", "total_loss", "grad_norm"):
        _assert(
            getattr(stats_off, field_name) == getattr(stats_on, field_name),
            f"stats.{field_name} differs with diagnostics on.",
        )

    # 3) Decomposition present only when enabled; grad diagnostics always present.
    _assert("hgnn_actor_grad_norm" in stats_on.diagnostics, "decomposition missing when enabled.")
    _assert("hgnn_actor_grad_norm" not in stats_off.diagnostics, "decomposition present when disabled.")
    for key in ("grad_pre_clip_critic", "grad_post_clip_hgnn", "grad_clip_scale",
                "rollout_offloading_entropy_normalized_mean"):
        _assert(key in stats_on.diagnostics and key in stats_off.diagnostics, f"missing diagnostic {key}.")
    cosine = stats_on.diagnostics["hgnn_actor_value_cosine"]
    _assert(cosine is None or -1.0 - 1e-6 <= float(cosine) <= 1.0 + 1e-6, "cosine out of range.")

    # 4) .grad left by the final backward+clip must be bitwise identical, and
    #    torch.autograd.grad must not have polluted it when decomposition ran.
    _assert(grads_off.keys() == grads_on.keys(), ".grad key sets differ.")
    for key in grads_off:
        a, b = grads_off[key], grads_on[key]
        if a is None or b is None:
            _assert(a is None and b is None, f".grad presence differs for {key}.")
        else:
            _assert(torch.equal(a, b), f".grad differs for {key} with diagnostics on.")

    # 5) Regression (frozen rollout without any policy actions): the epoch-0
    #    decomposition must not crash when the actor losses are graph-free
    #    constants; cosine must be reported as None.
    empty_buffer = CleanSlotRolloutBuffer()
    for record in buffer.records:
        stripped = copy.deepcopy(record)
        stripped.movement_records = []
        stripped.offloading_records = []
        empty_buffer.append(stripped)
    empty_buffer.close(bootstrap_value=0.0)
    mods_e = CleanTrainingModules(
        hgnn=copy.deepcopy(modules.hgnn),
        movement_actor=copy.deepcopy(modules.movement_actor),
        offloading_actor=copy.deepcopy(modules.offloading_actor),
        critic=copy.deepcopy(modules.critic),
    )
    updater_e = CleanPPOUpdater(
        modules=mods_e, optimizer=build_single_optimizer(mods_e, lr=1e-3),
        config=CleanPPOUpdateConfig(ppo_epochs=1, hgnn_grad_decomposition_interval=1),
        device=device,
    )
    stats_e = updater_e.update(empty_buffer)
    _assert(stats_e.diagnostics["hgnn_actor_grad_norm"] == 0.0, "graph-free actor term should report norm 0.")
    _assert(stats_e.diagnostics["hgnn_actor_value_cosine"] is None, "graph-free cosine should be None.")

    print("smoke_clean_diag_neutrality passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
