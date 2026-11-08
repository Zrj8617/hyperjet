"""完整回合缓存与 GAE；来源 on-policy utils/shared_buffer.py 的 compute_returns/feed_forward_generator，commit de66d7a，MIT 许可见 LICENSE。"""

import numpy as np
import torch


class RolloutBuffer:
    """保存完整回合，用紧凑任务列表支持每步不同的任务数。"""

    def __init__(self, args):
        self.args = args
        self.episodes = []

    def add_episode(self, samples, rewards, values):
        """保存一个已经结束的回合，不跨回合自举。"""
        self.episodes.append((samples, np.asarray(rewards, dtype=np.float32).reshape(-1, 1),
                              np.asarray(values, dtype=np.float32).reshape(-1, 1)))

    def compute_returns(self, value_normalizer):
        """使用官方 GAE + ValueNorm 分支，最后一步 continuation 为零。"""
        self.samples, predictions, returns = [], [], []
        for samples, rewards, values in self.episodes:
            extended = np.concatenate((values, np.zeros((1, 1), dtype=np.float32)))
            denormalized = value_normalizer.denormalize(extended)
            episode_returns = np.zeros_like(rewards)
            gae = 0.0
            for step in reversed(range(len(rewards))):
                continuation = float(step + 1 < len(rewards))
                delta = rewards[step] + self.args.gamma * denormalized[step + 1] * continuation - denormalized[step]
                gae = delta + self.args.gamma * self.args.gae_lambda * continuation * gae
                episode_returns[step] = gae + denormalized[step]
            self.samples.extend(samples)
            predictions.append(values)
            returns.append(episode_returns)
        self.value_preds = np.concatenate(predictions)
        self.returns = np.concatenate(returns)

    def feed_forward_generator(self, advantages, rng):
        """按时隙打乱，每个时隙保留所有智能体，价值只取一次。"""
        size = len(self.samples) // self.args.num_mini_batch
        order = rng.permutation(len(self.samples))
        for start in range(0, size * self.args.num_mini_batch, size):
            indices = order[start:start + size]
            samples = [self.samples[index] for index in indices]
            agents = samples[0]["observations"].shape[0]
            batch = {name: torch.from_numpy(np.concatenate([sample[name] for sample in samples]))
                     for name in ("observations", "task_features", "available_actions", "flight", "placements", "log_probs")}
            batch["states"] = torch.from_numpy(np.stack([sample["state"] for sample in samples]))
            batch["owners"] = torch.from_numpy(np.concatenate([
                sample["owners"] + index * agents for index, sample in enumerate(samples)
            ]))
            batch["value_preds"] = torch.from_numpy(self.value_preds[indices])
            batch["returns"] = torch.from_numpy(self.returns[indices])
            batch["advantages"] = torch.from_numpy(np.repeat(advantages[indices], agents, axis=0))
            yield batch
