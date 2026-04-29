from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import datetime
from pathlib import Path

import torch

import config
from marl_models.hgnn.pretrain import save_pretrained_scheduler, train_score_imitation
from marl_models.hgnn.supervision import collect_score_supervision_dataset, save_supervision_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=config.SCORE_PRETRAIN_EPISODES)
    parser.add_argument("--steps_per_episode", type=int, default=config.SCORE_PRETRAIN_STEPS_PER_EPISODE)
    parser.add_argument("--epochs", type=int, default=config.SCORE_PRETRAIN_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.SCORE_PRETRAIN_LR)
    parser.add_argument("--mode", type=str, default=config.SCORE_PRETRAIN_MODE, choices=["top1", "ranking", "soft"])
    parser.add_argument("--action_mode", type=str, default=config.SCORE_PRETRAIN_ACTION_MODE, choices=["zero", "random"])
    parser.add_argument("--output_dir", type=str, default="pretrained_score")
    parser.add_argument("--save_dataset", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[score-pretrain] device={device} mode={args.mode} action_mode={args.action_mode}")
    print("[score-pretrain] Stage 1/2: collect supervision samples")
    samples = collect_score_supervision_dataset(args.episodes, args.steps_per_episode, action_mode=args.action_mode)
    print(f"[score-pretrain] Collected {len(samples)} graph samples.")

    if args.save_dataset:
        dataset_path = f"{args.output_dir}/score_supervision_{timestamp}.json"
        save_supervision_dataset(samples, dataset_path)
        print(f"[score-pretrain] Saved dataset to {dataset_path}")

    print(f"[score-pretrain] Stage 2/2: start {args.mode} imitation training")
    scheduler, metrics = train_score_imitation(samples, args.epochs, args.lr, device, mode=args.mode)
    model_path = save_pretrained_scheduler(scheduler, f"{args.output_dir}/{timestamp}")
    metrics_path = f"{args.output_dir}/{timestamp}/metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump([dataclasses.asdict(metric) for metric in metrics], f, ensure_ascii=False, indent=2)

    print(f"[score-pretrain] Saved scheduler to {model_path}")
    print(f"[score-pretrain] Saved metrics to {metrics_path}")
    if metrics:
        print(
            f"[score-pretrain] Final epoch: loss={metrics[-1].avg_loss:.6f}, top1_acc={metrics[-1].top1_accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
