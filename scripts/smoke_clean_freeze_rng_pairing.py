"""RNG-alignment smoke for --freeze-movement (Phase 4 Commit 2).

The frozen-movement branch draws and discards the same movement sample the
learned path would consume. This smoke verifies the guarantee it provides:
after one identically-seeded slot, the torch and numpy RNG states are
IDENTICAL between a frozen and a learned collect, so subsequent offloading
sampling consumes an aligned random stream. It aligns sampling-stream
consumption only; trajectories still diverge later through the movement
treatment itself.

Requires torch; forces CPU.
"""

from __future__ import annotations

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
        print("smoke_clean_freeze_rng_pairing skipped: torch is not available.")
        return 0

    from environment.env import Env
    from environment.graph_builder import CleanGraphBuilder
    from environment.dag_tasks import TASK_STATE_READY_UNSCHEDULED
    from marl_models.hgnn import CleanIncidenceHGNN
    from marl_models.mappo.clean_movement_actor import CleanMovementActor
    from marl_models.mappo.clean_offloading_actor import CleanOffloadingActor
    from marl_models.mappo.clean_ppo import CleanCentralizedCritic, clean_critic_input_dim
    from marl_models.mappo.clean_slot_orchestrator import encode_prepared_slot, prepare_slot_state
    from marl_models.mappo.clean_trainer import CleanTrainingModules
    from torch.distributions import Categorical

    sys.path.insert(0, str(ROOT / "scripts"))
    from train_clean_mainline import _collect_clean_slot  # noqa: E402

    device = torch.device("cpu")

    def run_one_slot(freeze_movement: bool):
        random.seed(23)
        np.random.seed(23)
        torch.manual_seed(23)
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
            critic=CleanCentralizedCritic(
                input_dim=clean_critic_input_dim(emb_dim, config.NUM_UAVS), hidden_dim=16
            ),
        )
        encoded = encode_prepared_slot(
            prepared_state=prepared, env=env, hgnn=modules.hgnn, critic=modules.critic,
            movement_actor=modules.movement_actor, device=device,
        )
        _, _, info = _collect_clean_slot(
            env=env, modules=modules, encoded_state=encoded, categorical_cls=Categorical,
            device=device, task_state_ready=TASK_STATE_READY_UNSCHEDULED,
            freeze_movement=freeze_movement,
        )
        return info, torch.get_rng_state().clone(), np.random.get_state()

    info_frozen, torch_state_frozen, np_state_frozen = run_one_slot(freeze_movement=True)
    info_learned, torch_state_learned, np_state_learned = run_one_slot(freeze_movement=False)

    # Precondition for stream comparison: both paths must have taken the same
    # number of offloading samples in this first slot (identical frozen ready
    # set; candidate legality does not depend on the movement outcome).
    _assert(
        int(info_frozen["offloading_action_count"]) == int(info_learned["offloading_action_count"]),
        "offloading action counts differ in slot 1; RNG stream comparison is not applicable.",
    )
    _assert(int(info_frozen["offloading_action_count"]) >= 1, "slot 1 should produce at least one offloading action.")

    _assert(
        torch.equal(torch_state_frozen, torch_state_learned),
        "torch RNG state diverged: frozen path consumed a different number of samples.",
    )
    _assert(np_state_frozen[0] == np_state_learned[0], "numpy RNG algorithm mismatch.")
    _assert(
        np.array_equal(np_state_frozen[1], np_state_learned[1])
        and np_state_frozen[2:] == np_state_learned[2:],
        "numpy RNG state diverged between frozen and learned collects.",
    )
    _assert(bool(info_frozen["movement_frozen"]) is True, "frozen info flag missing.")
    _assert(bool(info_learned["movement_frozen"]) is False, "learned info flag wrong.")

    print("smoke_clean_freeze_rng_pairing passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
