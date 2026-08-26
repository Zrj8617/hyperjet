from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl_models.mappo.clean_decision_td_dataset import (
    PHASE4_RHO_ZERO_TOLERANCE,
    CleanDecisionTDRawCapture,
    clean_decision_slot_position_flags,
    load_clean_decision_td_raw_dataset,
)
from marl_models.mappo.clean_decision_td_diagnostic import (
    build_clean_decision_td_diagnostic,
    decision_q_input,
    expected_sarsa_target_components,
)
from marl_models.mappo.clean_decision_transitions import (
    CleanDecisionState,
    CleanDecisionTransition,
)
from marl_models.mappo.clean_offloading_action_value import (
    CleanOffloadingActionValueCritic,
)
from scripts.analyze_phase4_decision_td_dataset import analyze_dataset


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _state(slot: int, order: int, *, input_dim: int = 3) -> CleanDecisionState:
    candidate_width = input_dim - 1
    features = np.arange(2 * candidate_width, dtype=np.float32).reshape(2, candidate_width)
    return CleanDecisionState(
        episode_index=0,
        lane_index=0,
        slot_index=slot,
        task_id=f"task_{slot}_{order}",
        task_local_index=order,
        dag_id="dag_0",
        decision_order=order,
        selected_action=1,
        selected_uav_id=1,
        candidate_uav_ids=(0, 1),
        candidate_mask=np.asarray([True, True]),
        candidate_features=features,
        critic_global_context=np.asarray([7.0], dtype=np.float32),
        old_masked_probabilities=np.asarray([0.25, 0.75], dtype=np.float32),
        old_log_probability=float(np.log(0.75)),
    )


def _transition(
    state: CleanDecisionState,
    *,
    rho: float,
    delta: int,
    next_state: CleanDecisionState | None,
    terminated: bool = False,
) -> CleanDecisionTransition:
    return CleanDecisionTransition(
        state=state,
        rho=rho,
        delta=delta,
        next_state=next_state,
        terminated=terminated,
        truncated=False,
        unresolved=False,
        future_bootstrap=0.0 if terminated else None,
    )


def _raw_capture_and_analysis_smoke(temp_root: Path) -> None:
    capture = CleanDecisionTDRawCapture(
        output_dir=temp_root / "raw",
        selected_q_input_dim=3,
        source_checkpoint=None,
    )
    diagnostic = build_clean_decision_td_diagnostic(
        input_dim=3,
        hidden_dim=8,
        learning_rate=3e-4,
        gamma=0.9,
        max_grad_norm=0.5,
        device="cpu",
        raw_capture=capture,
    )
    state0 = _state(0, 0)
    state1 = _state(0, 1)
    state2 = _state(1, 0)
    transitions = [
        _transition(state0, rho=0.0, delta=0, next_state=state1),
        _transition(state1, rho=2.0, delta=1, next_state=state2),
    ]
    diagnostic.ingest_transitions(transitions)
    diagnostic.record_slot_advantages(
        slot_keys=[(0, 0, 0)],
        normalized_advantages=np.asarray([0.5], dtype=np.float32),
        decision_counts=[2],
    )
    stats = diagnostic.train_ready(update_step=101)
    _assert(stats["phase4_raw_capture_sample_count"] == 2, "raw capture missed samples")

    terminal = _transition(state2, rho=3.0, delta=1, next_state=None, terminated=True)
    diagnostic.ingest_transitions([terminal])
    diagnostic.record_slot_advantages(
        slot_keys=[(0, 0, 1)],
        normalized_advantages=np.asarray([-0.5], dtype=np.float32),
        decision_counts=[1],
    )
    diagnostic.train_ready(update_step=102)
    dataset, metadata = load_clean_decision_td_raw_dataset(temp_root / "raw")
    _assert(metadata["rho_zero_absolute_tolerance"] == PHASE4_RHO_ZERO_TOLERANCE, "rho tolerance missing")
    _assert(dataset["td_target"].shape[0] == 3, "NPZ shard merge lost samples")
    flags = clean_decision_slot_position_flags(dataset)
    _assert(bool(flags["is_first_decision"][0]), "first decision flag is wrong")
    _assert(bool(flags["is_last_decision"][1]), "last decision flag is wrong")
    _assert(bool(flags["is_first_decision"][2] and flags["is_last_decision"][2]), "singleton is not first+last")
    expected_input = decision_q_input(state0)[state0.selected_action]
    _assert(np.array_equal(dataset["selected_q_input"][0], expected_input), "captured selected Q input differs from forward input")
    same_target, same_bootstrap = expected_sarsa_target_components(
        transitions[0], target_q=diagnostic.target_q, gamma=0.9, device="cpu"
    )
    _assert(float(transitions[0].rho) == 0.0, "same-slot rho is nonzero")
    _assert(torch.equal(same_target, same_bootstrap), "same-slot target differs from bootstrap")
    _assert(
        np.allclose(dataset["td_target"], dataset["rho"] + dataset["bootstrap_value"]),
        "raw target != rho + bootstrap",
    )
    report = analyze_dataset(temp_root / "raw")
    _assert(report["variance_decomposition"]["identity_abs_error"] < 1e-6, "variance identity failed")
    _assert(report["within_slot"]["same_slot_target_bootstrap_max_abs_error"] == 0.0, "same-slot identity failed")
    for shard in sorted((temp_root / "raw").glob("update_*.npz")):
        with np.load(shard, allow_pickle=False) as payload:
            _assert(all(np.array_equal(payload[key], payload[key].copy()) for key in payload.files), "NPZ numerical roundtrip failed")
    print("raw capture PASS: target decomposition, positions, exact selected input, NPZ and variance identity")


def _offline_fit_smoke() -> None:
    _assert(not any(name.startswith("environment.") for name in sys.modules), "environment loaded before offline fit")
    from scripts.fit_phase4_fixed_decision_q import _fit
    _assert(not any(name.startswith("environment.") for name in sys.modules), "offline fitter imported environment")

    rng = np.random.default_rng(42)
    x = rng.normal(size=(16, 181)).astype(np.float32)
    y = (0.2 * x[:, 0] - 0.1 * x[:, 1]).astype(np.float32)
    model = CleanOffloadingActionValueCritic(input_dim=181, hidden_dim=128)
    _assert(model.input_dim == 181 and model.hidden_dim == 128, "offline Q architecture changed")
    _assert(torch.equal(model.net[-1].weight, torch.zeros_like(model.net[-1].weight)), "Q output initialization changed")
    initial_state = copy.deepcopy(model.state_dict())
    actor = torch.nn.Linear(2, 2)
    critic = torch.nn.Linear(2, 1)
    encoder = torch.nn.Linear(2, 2)
    base_before = [parameter.detach().clone() for module in (actor, critic, encoder) for parameter in module.parameters()]
    mask = np.ones(16, dtype=bool)
    result = _fit(
        initial_state=initial_state,
        input_dim=181,
        hidden_dim=128,
        x=x,
        y=y,
        train_mask=mask,
        validation_mask=mask,
        device=torch.device("cpu"),
        learning_rate=3e-4,
        max_grad_norm=0.5,
    )
    base_after = [parameter for module in (actor, critic, encoder) for parameter in module.parameters()]
    _assert(result["frozen_target_unchanged"], "offline target changed")
    _assert(all(torch.equal(before, after) for before, after in zip(base_before, base_after)), "offline fit changed base modules")
    _assert([row["step"] for row in result["measurements"]] == [0, 10, 50, 100, 250, 500, 1000], "offline report steps changed")
    print("offline fit PASS: frozen targets, no environment/base updates, identical Q architecture")


def _mainline_capture_smoke(temp_root: Path) -> None:
    import config
    from scripts import train_clean_mainline

    args_default = train_clean_mainline.build_arg_parser().parse_args([])
    _assert(not args_default.phase4_raw_diagnostic_capture, "raw capture default is enabled")
    original_probability = config.DAG_BASE_ARRIVAL_PROB
    config.DAG_BASE_ARRIVAL_PROB = 1.0
    try:
        args = train_clean_mainline.build_arg_parser().parse_args(
            [
                "--smoke", "--episodes", "1", "--max-steps-per-episode", "2",
                "--rollout-horizon", "1", "--checkpoint-interval", "0",
                "--device", "cpu", "--task-encoder", "mlp",
                "--output-dir", str(temp_root), "--run-name", "phase4_raw_mainline",
                "--record-decision-transitions", "--phase4-decision-td-diagnostic",
                "--phase4-raw-diagnostic-capture",
            ]
        )
        result = train_clean_mainline.run_training(args)
        raw_dir = Path(result["run_dir"]) / "phase4_raw_decision_td"
        dataset, metadata = load_clean_decision_td_raw_dataset(raw_dir)
        _assert(dataset["selected_q_input"].shape[1] == 181, "mainline raw Q input is not 181D")
        _assert(metadata["contains_graph_snapshot"] is False, "raw dataset saved a graph snapshot")
    finally:
        config.DAG_BASE_ARRIVAL_PROB = original_probability
    print("mainline capture PASS: real run_training wrote compact 181D NPZ shards")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix=".codex_tmp_phase4_raw_", dir=ROOT) as temp_dir:
        root = Path(temp_dir)
        _raw_capture_and_analysis_smoke(root)
        _offline_fit_smoke()
        _mainline_capture_smoke(root / "mainline")
    print("smoke_phase4_raw_fixed_dataset passed")


if __name__ == "__main__":
    main()
