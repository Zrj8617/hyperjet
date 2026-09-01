from __future__ import annotations

import json
from pathlib import Path
import shlex
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from marl_models.mappo.clean_ppo import clean_critic_input_dim
from scripts.train_clean_mainline import build_arg_parser


MANIFEST_PATH = ROOT / "experiments" / "manifests" / "phase3A_critic_task_pooling_ablation_20260825.json"
PHASE2_MANIFEST_PATH = ROOT / "experiments" / "manifests" / "phase2_clean_no_eft_actor_guidance_20260825.json"
OUTPUT_ROOT = "logs/formal_phase3A_critic_task_pooling_ablation_1000ep_3seed_20260825"
SEEDS = (42, 86, 1042)
GPU_MAP = {
    "mlp": {42: 0, 86: 2, 1042: 3},
    "hgnn": {42: 4, 86: 5, 1042: 6},
}
POOLING_SUFFIX = {"mean": "mean", "mean-max-std": "mean_max_std"}
POOLING_DIMENSION = {"mean": 102, "mean-max-std": 230}
FORBIDDEN_FLAGS = (
    "--decision-critic",
    "--offloading-eft-advantage",
    "--offloading-init-bandit-checkpoint",
    "--resume-checkpoint",
)
NON_TRAINING_IDENTITY_FIELDS = {"critic_task_pooling", "run_name", "output_dir"}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_command(command: str) -> tuple[int, Any]:
    tokens = shlex.split(command, posix=True)
    _assert(len(tokens) >= 4, f"incomplete training command: {command}")
    _assert(tokens[0].startswith("CUDA_VISIBLE_DEVICES="), "command must start with CUDA_VISIBLE_DEVICES")
    gpu = int(tokens[0].split("=", 1)[1])
    _assert(tokens[2] == "scripts/train_clean_mainline.py", "command must use scripts/train_clean_mainline.py")
    return gpu, build_arg_parser().parse_args(tokens[3:])


def _training_config(args: Any) -> dict[str, Any]:
    values = vars(args).copy()
    for field in NON_TRAINING_IDENTITY_FIELDS:
        values.pop(field, None)
    return values


def _expected_ids() -> list[str]:
    return [
        f"{encoder}_seed{seed}_{POOLING_SUFFIX[pooling]}"
        for pooling in ("mean", "mean-max-std")
        for encoder in ("mlp", "hgnn")
        for seed in SEEDS
    ]


def main() -> int:
    manifest = _load_json(MANIFEST_PATH)
    phase2 = _load_json(PHASE2_MANIFEST_PATH)
    experiments = manifest.get("experiments", [])
    _assert(len(experiments) == 12, "Phase3-A manifest must contain exactly 12 experiments")
    _assert([item["id"] for item in experiments] == _expected_ids(), "Phase3-A experiment IDs/order mismatch")
    _assert(manifest.get("fresh_run") is True, "manifest fresh_run must be true")
    _assert(manifest.get("checkpoint_source") is None, "manifest checkpoint_source must be null")
    _assert(manifest.get("output_root") == OUTPUT_ROOT, "Phase3-A output root mismatch")

    phase2_by_key = {(item["task_encoder"], int(item["seed"])): item for item in phase2["experiments"]}
    parsed: dict[tuple[str, int, str], Any] = {}
    output_dirs: set[str] = set()
    rows: list[dict[str, Any]] = []

    for item in experiments:
        experiment_id = str(item["id"])
        encoder = str(item["task_encoder"])
        seed = int(item["seed"])
        pooling = str(item["critic_task_pooling"])
        command = str(item["command"])
        gpu, args = _parse_command(command)

        _assert(item.get("fresh_run") is True, f"{experiment_id}: fresh_run must be true")
        _assert(item.get("checkpoint_source") is None, f"{experiment_id}: checkpoint_source must be null")
        for flag in FORBIDDEN_FLAGS:
            _assert(flag not in shlex.split(command, posix=True), f"{experiment_id}: forbidden flag {flag}")
        _assert(args.resume_checkpoint is None, f"{experiment_id}: resume checkpoint must be None")
        _assert(args.offloading_init_bandit_checkpoint is None, f"{experiment_id}: bandit checkpoint must be None")
        _assert(args.decision_critic is False, f"{experiment_id}: decision critic must be disabled")
        _assert(args.offloading_eft_advantage is False, f"{experiment_id}: EFT advantage must be disabled")
        _assert(float(args.eft_auxiliary_lambda_initial) == 0.0, f"{experiment_id}: EFT auxiliary lambda must be zero")

        _assert(args.task_encoder == encoder, f"{experiment_id}: task encoder mismatch")
        _assert(int(args.seed) == seed, f"{experiment_id}: seed mismatch")
        _assert(args.critic_task_pooling == pooling, f"{experiment_id}: critic pooling mismatch")
        _assert(gpu == int(item["gpu"]) == GPU_MAP[encoder][seed], f"{experiment_id}: GPU mapping mismatch")
        expected_model = "MLP" if encoder == "mlp" else "HGNN"
        _assert(item["model"] == expected_model, f"{experiment_id}: model label mismatch")
        _assert(bool(args.enable_kahypar) == (encoder == "hgnn"), f"{experiment_id}: KaHyPar setting mismatch")

        critic_dim = clean_critic_input_dim(
            int(args.task_embedding_dim),
            config.NUM_UAVS,
            task_pooling=args.critic_task_pooling,
        )
        _assert(critic_dim == POOLING_DIMENSION[pooling], f"{experiment_id}: critic input dimension mismatch")

        output_dir = str(args.output_dir).replace("\\", "/")
        _assert(output_dir == str(item["output_dir"]), f"{experiment_id}: command/manifest output mismatch")
        _assert(output_dir.startswith(f"{OUTPUT_ROOT}/"), f"{experiment_id}: output must stay under Phase3-A root")
        _assert("formal_phase2_clean_no_eft_actor_guidance" not in output_dir, f"{experiment_id}: Phase2 output collision")
        _assert(output_dir not in output_dirs, f"{experiment_id}: duplicate output directory")
        output_dirs.add(output_dir)

        phase2_gpu, phase2_args = _parse_command(str(phase2_by_key[(encoder, seed)]["command"]))
        _assert(phase2_gpu == gpu, f"{experiment_id}: GPU changed from Phase2")
        _assert(
            _training_config(args) == _training_config(phase2_args),
            f"{experiment_id}: training configuration differs from Phase2 beyond pooling/output identity",
        )

        parsed[(encoder, seed, pooling)] = args
        rows.append(
            {
                "experiment": experiment_id,
                "encoder": encoder,
                "seed": seed,
                "gpu": gpu,
                "pooling": pooling,
                "critic_input_dim": critic_dim,
                "output": output_dir,
            }
        )

    for encoder in ("mlp", "hgnn"):
        for seed in SEEDS:
            args_a = parsed[(encoder, seed, "mean")]
            args_b = parsed[(encoder, seed, "mean-max-std")]
            _assert(
                _training_config(args_a) == _training_config(args_b),
                f"{encoder} seed{seed}: A/B training controls differ beyond critic pooling/output identity",
            )
            print(
                f"PAIR PASS {encoder}_seed{seed}: actor/encoder/PPO controls identical; "
                "allowed differences=critic_task_pooling, critic_input_dim, run/output identity"
            )

    for row in rows:
        print(
            "RUN PASS "
            f"{row['experiment']} encoder={row['encoder']} seed={row['seed']} gpu={row['gpu']} "
            f"pooling={row['pooling']} critic_dim={row['critic_input_dim']} output={row['output']}"
        )
    print("argparse PASS")
    print("pooling PASS")
    print("dimension PASS")
    print("fresh-run PASS")
    print("EFT disabled PASS")
    print("checkpoint/resume PASS")
    print("Phase2 inheritance and A/B fairness PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
