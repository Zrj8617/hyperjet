from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl_models.mappo.clean_decision_td_dataset import load_clean_decision_td_raw_dataset
from marl_models.mappo.clean_offloading_action_value import (
    CleanOffloadingActionValueCritic,
    build_rng_neutral_clean_counterfactual_q,
)


REPORT_STEPS = (0, 10, 50, 100, 250, 500, 1000)
SPLIT_SEED = 42
TRAIN_FRACTION = 0.8


def _group_split(dataset: dict[str, np.ndarray], eligible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eligible = np.asarray(eligible, dtype=bool)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index, values in enumerate(
        zip(
            dataset["episode_index"].tolist(),
            dataset["lane_index"].tolist(),
            dataset["slot_index"].tolist(),
        )
    ):
        if eligible[index]:
            groups.setdefault(tuple(int(value) for value in values), []).append(index)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("fixed-dataset split requires at least two eligible slot groups")
    order = np.arange(len(keys))
    np.random.default_rng(SPLIT_SEED).shuffle(order)
    train_group_count = min(max(int(np.floor(TRAIN_FRACTION * len(keys))), 1), len(keys) - 1)
    train_keys = {keys[index] for index in order[:train_group_count]}
    train = np.zeros(eligible.size, dtype=bool)
    validation = np.zeros(eligible.size, dtype=bool)
    for key, indices in groups.items():
        (train if key in train_keys else validation)[indices] = True
    return train, validation


def _loss_ev(model: Any, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float | None]:
    model.eval()
    with torch.no_grad():
        prediction = model(x)
        loss = 0.5 * (prediction - y).pow(2).mean()
        variance = y.var(unbiased=False)
        ev = (
            1.0 - (y - prediction).var(unbiased=False) / variance
            if float(variance.item()) > 1e-12
            else None
        )
    return float(loss.item()), None if ev is None else float(ev.item())


def _fit(
    *,
    initial_state: dict[str, torch.Tensor],
    input_dim: int,
    hidden_dim: int,
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    device: torch.device,
    learning_rate: float,
    max_grad_norm: float,
) -> dict[str, Any]:
    model = CleanOffloadingActionValueCritic(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    frozen_y = torch.as_tensor(y, dtype=torch.float32, device=device).clone().detach()
    frozen_y_before = frozen_y.clone()
    train_index = torch.as_tensor(train_mask, dtype=torch.bool, device=device)
    validation_index = torch.as_tensor(validation_mask, dtype=torch.bool, device=device)
    if not bool(train_index.any()) or not bool(validation_index.any()):
        raise ValueError("fixed-dataset train and validation partitions must be non-empty")
    rows: list[dict[str, Any]] = []
    for step in range(max(REPORT_STEPS) + 1):
        if step in REPORT_STEPS:
            train_loss, train_ev = _loss_ev(
                model, x_tensor[train_index], frozen_y[train_index]
            )
            validation_loss, validation_ev = _loss_ev(
                model, x_tensor[validation_index], frozen_y[validation_index]
            )
            rows.append(
                {
                    "step": step,
                    "train_loss": train_loss,
                    "train_ev": train_ev,
                    "validation_loss": validation_loss,
                    "validation_ev": validation_ev,
                }
            )
        if step == max(REPORT_STEPS):
            break
        model.train()
        prediction = model(x_tensor[train_index])
        loss = 0.5 * (prediction - frozen_y[train_index]).pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()), float(max_grad_norm))
        optimizer.step()
    if not torch.equal(frozen_y, frozen_y_before):
        raise RuntimeError("offline fixed target changed during fitting")
    return {
        "train_sample_count": int(train_index.sum().item()),
        "validation_sample_count": int(validation_index.sum().item()),
        "target_std": float(frozen_y.std(unbiased=False).item()),
        "measurements": rows,
        "frozen_target_unchanged": True,
    }


def _read_checkpoint_online_q(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        payload = torch.load(path, map_location="cpu")
    return payload["extra_state"]["phase4_decision_td_diagnostic"]["online_q"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--checkpoint-online-q", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    args = parser.parse_args()

    dataset, metadata = load_clean_decision_td_raw_dataset(args.dataset)
    x = np.asarray(dataset["selected_q_input"], dtype=np.float32)
    y = np.asarray(dataset["td_target"], dtype=np.float32).reshape(-1)
    input_dim = int(metadata["selected_q_input_dim"])
    if x.ndim != 2 or int(x.shape[1]) != input_dim:
        raise ValueError("fixed-dataset selected Q input dimension mismatch")
    if input_dim != 181 or int(args.hidden_dim) != 128:
        raise ValueError("Phase4 fixed-dataset pilot requires the existing 181x128 Q architecture")

    torch.manual_seed(SPLIT_SEED)
    fresh_model = build_rng_neutral_clean_counterfactual_q(
        input_dim=input_dim, hidden_dim=int(args.hidden_dim)
    )
    fresh_state = copy.deepcopy(fresh_model.state_dict())
    checkpoint_state = _read_checkpoint_online_q(args.checkpoint_online_q)
    architecture_probe = CleanOffloadingActionValueCritic(
        input_dim=input_dim, hidden_dim=int(args.hidden_dim)
    )
    architecture_probe.load_state_dict(copy.deepcopy(checkpoint_state))

    all_rows = np.ones(y.size, dtype=bool)
    same_slot = np.asarray(dataset["delta"], dtype=np.int64) == 0
    overall_train, overall_validation = _group_split(dataset, all_rows)
    same_slot_train, same_slot_validation = _group_split(dataset, same_slot)
    conditions = {
        "overall_train_validation": (overall_train, overall_validation),
        "same_data_memorization": (all_rows, all_rows),
        "same_slot_train_validation": (same_slot_train, same_slot_validation),
    }
    initializations = {
        "fresh_identical_q": fresh_state,
        "checkpoint_online_q_update100": checkpoint_state,
    }
    device = torch.device(args.device)
    results: dict[str, Any] = {}
    for initialization_name, initial_state in initializations.items():
        results[initialization_name] = {}
        for condition_name, (train_mask, validation_mask) in conditions.items():
            results[initialization_name][condition_name] = _fit(
                initial_state=initial_state,
                input_dim=input_dim,
                hidden_dim=int(args.hidden_dim),
                x=x,
                y=y,
                train_mask=train_mask,
                validation_mask=validation_mask,
                device=device,
                learning_rate=float(args.learning_rate),
                max_grad_norm=float(args.max_grad_norm),
            )

    report = {
        "schema": "hyperuav_phase4_fixed_decision_q_fit_v1",
        "dataset": str(args.dataset),
        "checkpoint_online_q": str(args.checkpoint_online_q),
        "environment_accessed": False,
        "actor_or_ppo_accessed": False,
        "split_seed": SPLIT_SEED,
        "split_unit": "episode_lane_slot",
        "train_fraction": TRAIN_FRACTION,
        "report_steps": list(REPORT_STEPS),
        "q_architecture": {
            "class": "CleanOffloadingActionValueCritic",
            "input_dim": input_dim,
            "hidden_dim": int(args.hidden_dim),
            "activation": "ReLU",
            "output_initialization": "zero",
        },
        "optimizer": {
            "type": "Adam",
            "learning_rate": float(args.learning_rate),
            "max_grad_norm": float(args.max_grad_norm),
            "loss": "0.5*MSE",
            "target_normalization": False,
        },
        "results": results,
    }
    output = args.output or (args.dataset / "fixed_dataset_q_fit.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
