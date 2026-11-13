"""S1 飞行专用 minibatch；GAE 直接复用通用完整回合缓存。"""

import numpy as np
import torch

from methods_v2.learning.buffer import RolloutBuffer


class FlightRolloutBuffer(RolloutBuffer):
    """只组装飞行动作所需字段，不保存放置特征。"""

    def feed_forward_generator(self, advantages, rng):
        """按时隙打乱，保留五架 UAV 的飞行动作和团队价值。"""
        size = len(self.samples) // self.args.num_mini_batch
        order = rng.permutation(len(self.samples))
        for start in range(0, size * self.args.num_mini_batch, size):
            indices = order[start:start + size]
            samples = [self.samples[index] for index in indices]
            agents = samples[0]["observations"].shape[0]
            batch = {name: torch.from_numpy(np.concatenate([
                sample[name] for sample in samples
            ])) for name in ("observations", "flight", "log_probs")}
            batch["states"] = torch.from_numpy(np.stack([
                sample["state"] for sample in samples
            ]))
            batch["value_preds"] = torch.from_numpy(self.value_preds[indices])
            batch["returns"] = torch.from_numpy(self.returns[indices])
            batch["advantages"] = torch.from_numpy(np.repeat(
                advantages[indices], agents, axis=0,
            ))
            yield batch


class PlacementRolloutBuffer(RolloutBuffer):
    """放置按真实任务数拼接；空任务时隙仍保留 critic 样本。"""

    def feed_forward_generator(self, advantages, rng):
        """先由训练器按时隙归一化优势，再展开给各时隙的子任务。"""
        size = len(self.samples) // self.args.num_mini_batch
        order = rng.permutation(len(self.samples))
        for start in range(0, size * self.args.num_mini_batch, size):
            indices = order[start:start + size]
            samples = [self.samples[index] for index in indices]
            agents = samples[0]["observations"].shape[0]
            task_counts = [len(sample["owners"]) for sample in samples]
            batch = {name: torch.from_numpy(np.concatenate([
                sample[name] for sample in samples
            ])) for name in ("observations", "task_features", "available_actions",
                             "placements", "log_probs")}
            batch["owners"] = torch.from_numpy(np.concatenate([
                sample["owners"] + index * agents for index, sample in enumerate(samples)
            ]))
            batch["states"] = torch.from_numpy(np.stack([sample["state"] for sample in samples]))
            batch["value_preds"] = torch.from_numpy(self.value_preds[indices])
            batch["returns"] = torch.from_numpy(self.returns[indices])
            batch["advantages"] = torch.from_numpy(np.repeat(advantages[indices], task_counts, axis=0))
            yield batch
