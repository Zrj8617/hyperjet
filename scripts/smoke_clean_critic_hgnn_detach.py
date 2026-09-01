"""Focused gradient-path smoke for the critic-to-HGNN detach experiment."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _grad_norm(loss, parameters, torch) -> float:
    params = [parameter for parameter in parameters if parameter.requires_grad]
    gradients = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    squared = sum(
        float(gradient.detach().pow(2).sum().item())
        for gradient in gradients
        if gradient is not None
    )
    return squared ** 0.5


def main() -> int:
    try:
        import torch
    except ModuleNotFoundError:
        print("smoke_clean_critic_hgnn_detach skipped: torch is not available")
        return 0

    from marl_models.hgnn import CleanIncidenceHGNN
    from marl_models.mappo.clean_movement_actor import CleanMovementActor
    from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
    from marl_models.mappo.clean_ppo import CleanCentralizedCritic, clean_critic_input_dim
    from marl_models.mappo.clean_slot_orchestrator import (
        CleanMovementRolloutRecord,
        CleanSlotRolloutBuffer,
        CleanSlotRolloutRecord,
    )
    from marl_models.mappo.clean_trainer import (
        CleanPPOUpdateConfig,
        CleanPPOUpdater,
        CleanTrainingModules,
        build_single_optimizer,
    )

    torch.manual_seed(29)
    np.random.seed(29)
    task_feature_dim = 4
    embedding_dim = 8
    hidden_dim = 12
    modules = CleanTrainingModules(
        hgnn=CleanIncidenceHGNN(
            task_feature_dim=task_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
        ),
        movement_actor=CleanMovementActor(
            task_embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        ),
        offloading_actor=CleanOffloadingActor(
            task_embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        ),
        critic=CleanCentralizedCritic(
            input_dim=clean_critic_input_dim(embedding_dim, config.NUM_UAVS),
            hidden_dim=hidden_dim,
        ),
    )
    non_graph_dim = clean_critic_input_dim(embedding_dim, config.NUM_UAVS) - embedding_dim
    snapshot = SimpleNamespace(
        task_features=np.asarray(
            [
                [1.0, 0.2, 0.4, 0.8],
                [0.3, 1.0, 0.7, 0.1],
                [0.5, 0.6, 1.0, 0.9],
            ],
            dtype=np.float32,
        ),
        incidence_matrix=np.eye(3, dtype=np.float32),
        hyperedge_type_ids=np.zeros((3,), dtype=np.int64),
    )
    movement_record = CleanMovementRolloutRecord(
        uav_id=0,
        uav_index=0,
        uav_features=np.asarray([0.2, 0.4, 0.1, 0.3, 0.6, 0.8], dtype=np.float32),
        ready_task_indices=[0, 1],
        pending_task_indices=[2],
        ready_count_normalized=2.0 / 3.0,
        pending_count_normalized=1.0 / 3.0,
        movement_mask=np.ones((int(config.CLEAN_MOVEMENT_ACTION_DIM),), dtype=bool),
        selected_action=1,
        old_log_probability=-1.0,
        entropy=1.0,
    )
    record = CleanSlotRolloutRecord(
        slot_index=0,
        graph_snapshot=snapshot,
        critic_non_graph_input=np.linspace(-0.5, 0.5, non_graph_dim, dtype=np.float32),
        value=0.0,
        reward=3.0,
        terminated=True,
        movement_records=[movement_record],
    )
    returns = torch.as_tensor([3.0], dtype=torch.float32)
    advantages = torch.as_tensor([1.0], dtype=torch.float32)
    value_loss_inputs = {
        "old_values": torch.as_tensor([0.0], dtype=torch.float32),
        "value_target_mean": torch.zeros((), dtype=torch.float32),
        "value_target_scale": torch.ones((), dtype=torch.float32),
    }

    def _updater(detach: bool | None):
        kwargs = {} if detach is None else {"detach_critic_hgnn": detach}
        return CleanPPOUpdater(
            modules=modules,
            optimizer=build_single_optimizer(modules, lr=1e-3),
            config=CleanPPOUpdateConfig(
                hgnn_grad_decomposition_interval=1,
                **kwargs,
            ),
            device="cpu",
        )

    default_loss = _updater(None)._loss(
        records=[record], returns=returns, advantages=advantages, **value_loss_inputs
    )
    shared_loss = _updater(False)._loss(
        records=[record], returns=returns, advantages=advantages, **value_loss_inputs
    )
    for name in default_loss:
        if isinstance(default_loss[name], torch.Tensor):
            _assert(
                torch.equal(default_loss[name], shared_loss[name]),
                f"explicit shared mode should match the default for {name}.",
            )
        else:
            _assert(
                default_loss[name] == shared_loss[name],
                f"explicit shared diagnostics should match the default for {name}.",
            )

    shared_value_hgnn = _grad_norm(
        shared_loss["value_loss"], modules.hgnn.parameters(), torch
    )
    shared_value_critic = _grad_norm(
        shared_loss["value_loss"], modules.critic.parameters(), torch
    )
    _assert(shared_value_hgnn > 0.0, "shared value loss should update the HGNN.")
    _assert(shared_value_critic > 0.0, "shared value loss should update the critic.")

    detach_updater = _updater(True)
    detach_loss = detach_updater._loss(
        records=[record], returns=returns, advantages=advantages, **value_loss_inputs
    )
    detach_value_hgnn = _grad_norm(
        detach_loss["value_loss"], modules.hgnn.parameters(), torch
    )
    detach_value_critic = _grad_norm(
        detach_loss["value_loss"], modules.critic.parameters(), torch
    )
    actor_term = (
        detach_loss["movement_loss"]
        + detach_loss["offloading_loss"]
        - detach_updater.config.movement_entropy_coef * detach_loss["movement_entropy"]
        - detach_updater.config.offloading_entropy_coef * detach_loss["offloading_entropy"]
    )
    detach_actor_hgnn = _grad_norm(actor_term, modules.hgnn.parameters(), torch)
    _assert(detach_value_hgnn == 0.0, "detached value loss must not update the HGNN.")
    _assert(detach_value_critic > 0.0, "detached value loss must still update the critic.")
    _assert(detach_actor_hgnn > 0.0, "detach mode must preserve actor-to-HGNN gradients.")

    buffer = CleanSlotRolloutBuffer()
    buffer.append(record)
    buffer.close(bootstrap_value=0.0)
    stats = detach_updater.update(buffer)
    diagnostics = stats.diagnostics
    _assert(diagnostics["critic_hgnn_detached"] is True, "detach diagnostic is missing.")
    _assert(diagnostics["hgnn_value_grad_norm"] == 0.0, "value HGNN norm should be zero.")
    _assert(diagnostics["hgnn_actor_value_cosine"] is None, "zero value gradient has no cosine.")
    _assert(diagnostics["hgnn_actor_grad_norm"] > 0.0, "actor HGNN norm should remain non-zero.")
    _assert(diagnostics["grad_pre_clip_critic"] > 0.0, "critic gradient should remain non-zero.")

    print("smoke_clean_critic_hgnn_detach passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
