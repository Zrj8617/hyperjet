from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from marl_models.mappo.clean_ppo import (
    CleanCentralizedCritic,
    clean_critic_input_dim,
    pool_clean_critic_task_embeddings,
)
from scripts.train_clean_mainline import (
    build_arg_parser,
    build_config_snapshot,
    checkpoint_experiment_controls,
    run_training,
    validate_resume_experiment_controls,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _pooling_checks() -> None:
    embeddings = np.asarray([[1.0, 3.0], [3.0, 7.0]], dtype=np.float32)
    legacy_mean = embeddings.mean(axis=0).astype(np.float32)
    mean_pool = pool_clean_critic_task_embeddings(embeddings, "mean")
    _assert(np.array_equal(mean_pool, legacy_mean), "mean pooling must exactly preserve the Phase2 implementation.")

    expected = np.asarray([2.0, 5.0, 3.0, 7.0, 1.0, 2.0], dtype=np.float32)
    rich_pool = pool_clean_critic_task_embeddings(embeddings, "mean-max-std")
    _assert(np.allclose(rich_pool, expected), "mean-max-std must concatenate [mean, max, population std].")

    _assert(clean_critic_input_dim(64, 5, "mean") == 102, "Phase2 mean critic dimension must remain 102.")
    _assert(
        clean_critic_input_dim(64, 5, "mean-max-std") == 230,
        "Phase3-A mean-max-std critic dimension must be 230.",
    )

    empty = np.zeros((0, 2), dtype=np.float32)
    empty_pool = pool_clean_critic_task_embeddings(empty, "mean-max-std")
    _assert(empty_pool.shape == (6,), "empty rich pool must retain three embedding-width blocks.")
    _assert(np.isfinite(empty_pool).all() and np.count_nonzero(empty_pool) == 0, "empty rich pool must be finite zeros.")

    single = np.asarray([[4.0, -2.0]], dtype=np.float32)
    single_pool = pool_clean_critic_task_embeddings(single, "mean-max-std")
    _assert(np.array_equal(single_pool[:2], single[0]), "single-task mean must equal the task embedding.")
    _assert(np.array_equal(single_pool[2:4], single[0]), "single-task max must equal the task embedding.")
    _assert(np.count_nonzero(single_pool[4:]) == 0, "single-task population std must be zero.")


def _torch_pooling_checks() -> None:
    import torch

    embeddings_np = np.asarray([[1.0, 3.0], [3.0, 7.0]], dtype=np.float32)
    embeddings = torch.as_tensor(embeddings_np)
    for pooling in ("mean", "mean-max-std"):
        numpy_pool = pool_clean_critic_task_embeddings(embeddings_np, pooling)
        torch_pool = pool_clean_critic_task_embeddings(embeddings, pooling).detach().cpu().numpy()
        _assert(np.array_equal(numpy_pool, torch_pool), f"NumPy/Torch pooling mismatch for {pooling}.")

    empty = torch.zeros((0, 2), dtype=torch.float32)
    single = torch.tensor([[4.0, -2.0]], dtype=torch.float32)
    _assert(torch.isfinite(pool_clean_critic_task_embeddings(empty, "mean-max-std")).all().item(), "Torch empty pool must be finite.")
    _assert(
        torch.count_nonzero(pool_clean_critic_task_embeddings(single, "mean-max-std")[-2:]).item() == 0,
        "Torch single-task population std must be zero.",
    )

    critic = CleanCentralizedCritic(input_dim=230, hidden_dim=16, task_pooling="mean-max-std")
    _assert(critic.task_pooling == "mean-max-std", "critic must retain its runtime pooling mode.")
    _assert(all("task_pooling" not in key for key in critic.state_dict()), "task_pooling must not enter state_dict.")


def _checkpoint_control_checks() -> None:
    parser = build_arg_parser()
    args_a = parser.parse_args(["--critic-task-pooling", "mean"])
    args_b = parser.parse_args(["--critic-task-pooling", "mean-max-std"])
    snapshot_a = build_config_snapshot(args_a)
    snapshot_b = build_config_snapshot(args_b)
    _assert(snapshot_a["experiment_controls"]["critic_task_pooling"] == "mean", "A checkpoint config must save mean.")
    _assert(
        snapshot_b["experiment_controls"]["critic_task_pooling"] == "mean-max-std",
        "B checkpoint config must save mean-max-std.",
    )

    payload_a = {"config": snapshot_a}
    _assert(checkpoint_experiment_controls(payload_a)["critic_task_pooling"] == "mean", "A checkpoint must restore mean.")
    legacy_payload = {"config": {"cli": {}, "experiment_controls": {}}}
    _assert(
        checkpoint_experiment_controls(legacy_payload)["critic_task_pooling"] == "mean",
        "legacy checkpoints without the field must resolve to mean.",
    )
    try:
        validate_resume_experiment_controls(args_b, payload_a)
    except ValueError as exc:
        _assert("critic task pooling mismatch" in str(exc), "A-to-B resume must fail for the pooling mismatch.")
    else:
        raise AssertionError("A checkpoint must not resume in B mode.")


def _module_construction_checks() -> None:
    import torch

    from scripts.eval_clean_mainline import _build_modules as build_eval_modules
    from scripts.train_clean_mainline import _build_process_worker_modules

    parser = build_arg_parser()
    args_a = parser.parse_args(["--critic-task-pooling", "mean"])
    args_b = parser.parse_args(["--critic-task-pooling", "mean-max-std"])
    controls_a = checkpoint_experiment_controls({"config": build_config_snapshot(args_a)})
    controls_b = checkpoint_experiment_controls({"config": build_config_snapshot(args_b)})
    dims = {"task_feature_dim": 12, "task_embedding_dim": 64, "hidden_dim": 128}
    eval_a = build_eval_modules(dims=dims, experiment_controls=controls_a, device=torch.device("cpu"))
    eval_b = build_eval_modules(dims=dims, experiment_controls=controls_b, device=torch.device("cpu"))
    _assert(eval_a.critic.net[0].in_features == 102, "evaluation must reconstruct the A critic dimension.")
    _assert(eval_b.critic.net[0].in_features == 230, "evaluation must reconstruct the B critic dimension.")
    _assert(eval_b.critic.task_pooling == "mean-max-std", "evaluation must restore B pooling.")

    torch.manual_seed(71)
    worker_a = _build_process_worker_modules(
        task_feature_dim=12,
        task_embedding_dim=64,
        hidden_dim=16,
        task_encoder="mlp",
        critic_task_pooling="mean",
        device=torch.device("cpu"),
    )
    torch.manual_seed(71)
    worker_b = _build_process_worker_modules(
        task_feature_dim=12,
        task_embedding_dim=64,
        hidden_dim=16,
        task_encoder="mlp",
        critic_task_pooling="mean-max-std",
        device=torch.device("cpu"),
    )
    _assert(worker_b.critic.net[0].in_features == 230, "process worker must construct the B critic dimension.")
    _assert(worker_b.critic.task_pooling == "mean-max-std", "process worker must use B pooling.")
    for module_name in ("hgnn", "movement_actor", "offloading_actor"):
        state_a = getattr(worker_a, module_name).state_dict()
        state_b = getattr(worker_b, module_name).state_dict()
        _assert(
            all(torch.equal(state_a[key], state_b[key]) for key in state_a),
            f"A/B pooling must not change {module_name} initialization for the same seed.",
        )


def _b_end_to_end_smoke() -> None:
    import torch

    temp_root = ROOT / ".codex_tmp_critic_task_pooling" / f"run_{os.getpid()}"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True)
    try:
        args = build_arg_parser().parse_args(
            [
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
                "mlp",
                "--critic-task-pooling",
                "mean-max-std",
                "--eft-auxiliary-lambda-initial",
                "0",
                "--output-dir",
                str(temp_root),
                "--run-name",
                "smoke_critic_task_pooling",
            ]
        )
        result = run_training(args)
        latest_update = result.get("latest_update")
        _assert(latest_update is not None and int(latest_update["update_step"]) == 1, "B smoke must complete one PPO update.")
        _assert(np.isfinite(float(latest_update["total_loss"])), "B smoke PPO loss must be finite.")
        _assert(result["critic_task_pooling"] == "mean-max-std", "B smoke result must report its pooling mode.")

        run_dirs = [path for path in temp_root.iterdir() if path.is_dir()]
        _assert(len(run_dirs) == 1, "B smoke must create exactly one run directory.")
        run_dir = run_dirs[0]
        saved_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        _assert(
            saved_config["experiment_controls"]["critic_task_pooling"] == "mean-max-std",
            "B run config must persist mean-max-std.",
        )
        checkpoint = torch.load(run_dir / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
        controls = checkpoint_experiment_controls(checkpoint)
        _assert(controls["critic_task_pooling"] == "mean-max-std", "B checkpoint must persist mean-max-std.")
        first_weight = checkpoint["critic"]["net.0.weight"]
        _assert(tuple(first_weight.shape) == (128, 230), "B checkpoint critic first layer must be [128, 230].")
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root)
        parent = temp_root.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()


def main() -> None:
    _pooling_checks()
    _checkpoint_control_checks()
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        print("smoke_clean_critic_task_pooling non-torch checks passed; torch checks skipped")
        return
    _torch_pooling_checks()
    _module_construction_checks()
    _b_end_to_end_smoke()
    print("smoke_clean_critic_task_pooling passed")


if __name__ == "__main__":
    main()
