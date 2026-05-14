from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

import config
from marl_models.hgnn.scheduler import PhaseOneGraphScheduler
from marl_models.hgnn.supervision import GraphSupervisionSample
from utils.progress import TerminalProgress


@dataclass(slots=True)
class ScorePretrainMetrics:
    epoch: int
    avg_loss: float
    top1_accuracy: float


def _build_soft_target(eft_values: list[float], device: torch.device) -> torch.Tensor:
    eft_tensor = torch.tensor(eft_values, dtype=torch.float32, device=device)
    min_eft = torch.min(eft_tensor)
    normalized_gap = (eft_tensor - min_eft) / max(float(torch.max(torch.abs(eft_tensor))), config.EPSILON)
    return torch.softmax(-normalized_gap / max(config.SCORE_SOFT_TARGET_TAU, config.EPSILON), dim=0)


def train_score_imitation(
    samples: list[GraphSupervisionSample],
    epochs: int,
    learning_rate: float,
    device: str,
    mode: str = "top1",
) -> tuple[PhaseOneGraphScheduler, list[ScorePretrainMetrics]]:
    scheduler = PhaseOneGraphScheduler(device=device)
    scheduler.train()
    optimizer = torch.optim.Adam(scheduler.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    ranking_loss_fn = nn.MarginRankingLoss(margin=config.SCORE_RANKING_MARGIN, reduction="none")
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    metrics: list[ScorePretrainMetrics] = []
    epoch_progress = TerminalProgress(epochs, f"train:{mode}")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_tasks = 0
        correct = 0

        for sample_idx in torch.randperm(len(samples)).tolist():
            sample = samples[sample_idx]
            output = scheduler.forward_graph(sample.snapshot)
            edge_score_lookup = {edge_key: output.edge_scores[idx] for idx, edge_key in enumerate(output.edge_keys)}

            sample_loss: torch.Tensor | None = None
            sample_task_count = 0
            for target in sample.targets:
                logits = []
                label_index = None
                valid_uav_ids: list[int] = []
                for idx, uav_id in enumerate(target.feasible_uav_ids):
                    edge_key = (target.task_id, uav_id)
                    if edge_key not in edge_score_lookup:
                        continue
                    logits.append(edge_score_lookup[edge_key])
                    valid_uav_ids.append(uav_id)
                    if uav_id == target.heuristic_best_uav:
                        label_index = len(logits) - 1

                if label_index is None or len(logits) < 2:
                    continue

                logits_tensor = torch.stack(logits, dim=0).unsqueeze(0)
                labels = torch.tensor([label_index], dtype=torch.long, device=logits_tensor.device)

                if mode == "top1":
                    task_loss = loss_fn(logits_tensor, labels)
                elif mode == "ranking":
                    score_lookup = target.heuristic_score_by_uav
                    pair_losses: list[torch.Tensor] = []
                    for i, uav_pos in enumerate(valid_uav_ids):
                        for j, uav_neg in enumerate(valid_uav_ids):
                            if i == j:
                                continue
                            score_pos = score_lookup[uav_pos]
                            score_neg = score_lookup[uav_neg]
                            if score_pos >= score_neg:
                                continue
                            pos_score = edge_score_lookup[(target.task_id, uav_pos)].view(1)
                            neg_score = edge_score_lookup[(target.task_id, uav_neg)].view(1)
                            target_rank = torch.ones(1, device=pos_score.device)
                            raw_pair_loss = ranking_loss_fn(pos_score, neg_score, target_rank).view(())
                            teacher_gap = max(float(score_neg - score_pos), 0.0)
                            gap_weight = 1.0 + teacher_gap / max(abs(float(score_neg)), config.EPSILON)
                            pair_losses.append(raw_pair_loss * gap_weight)
                    if not pair_losses:
                        task_loss = loss_fn(logits_tensor, labels)
                    else:
                        ranking_term = torch.stack(pair_losses).mean()
                        top1_term = loss_fn(logits_tensor, labels)
                        task_loss = ranking_term + config.SCORE_RANKING_TOP1_WEIGHT * top1_term
                elif mode == "soft":
                    score_lookup = target.heuristic_score_by_uav
                    teacher_scores = [float(score_lookup[uav_id]) for uav_id in valid_uav_ids]
                    soft_target = _build_soft_target(teacher_scores, logits_tensor.device).unsqueeze(0)
                    log_probs = F.log_softmax(logits_tensor, dim=1)
                    task_loss = kl_loss_fn(log_probs, soft_target)
                else:
                    raise ValueError(f"Unknown score pretrain mode: {mode}")
                sample_loss = task_loss if sample_loss is None else sample_loss + task_loss
                sample_task_count += 1

                predicted = int(torch.argmax(logits_tensor, dim=1).item())
                if predicted == label_index:
                    correct += 1

            if sample_loss is None or sample_task_count == 0:
                continue

            optimizer.zero_grad()
            sample_loss.backward()
            optimizer.step()

            total_loss += float(sample_loss.item())
            total_tasks += sample_task_count

        avg_loss = total_loss / max(total_tasks, 1)
        accuracy = correct / max(total_tasks, 1)
        metrics.append(ScorePretrainMetrics(epoch=epoch, avg_loss=avg_loss, top1_accuracy=accuracy))
        epoch_progress.update(postfix=f"loss {avg_loss:.4f} top1 {accuracy:.4f}")

    epoch_progress.finish(postfix=f"final loss {metrics[-1].avg_loss:.4f} top1 {metrics[-1].top1_accuracy:.4f}" if metrics else "done")
    scheduler.eval()
    return scheduler, metrics


def save_pretrained_scheduler(scheduler: PhaseOneGraphScheduler, output_dir: str) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "phase_one_graph_scheduler.pt"
    torch.save(scheduler.state_dict(), model_path)
    return str(model_path)
