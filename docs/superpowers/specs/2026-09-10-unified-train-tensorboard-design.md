# UnifiedTrain TensorBoard 统一训练口径（2026-09-10）

**Status:** APPROVED by user

## 目标

在现有 `http://127.0.0.1:16008/` 中新增独立 `UnifiedTrain` 分组，修正历史 run 中同名 `train/episode_reward` 实际口径不一致的问题。原始实验文件、原 TensorBoard run 和 `FormalEval` 固定-tape结果均保持不变。

## 范围

覆盖当前33个正式训练 run：

- 2026-09-06：N0、A2、B1、B2、B2D、D1、D2，各3 seeds；
- 2026-09-07：A1、C1、N0-local，各3 seeds；
- 2026-09-08：C2，3 seeds。

## 统一定义

- run 名：`UnifiedTrain/<原始 run 名>`。
- `train/episode_reward`：只读取原始 `train_metrics.jsonl` 中 `episode_terminal_record=true` 的行，取 `episode_reward_total`，横轴为该行 `global_slot`。每个 run 必须恰好500点。
- `train/critic_loss`：读取每个 PPO update 行的 `ppo_value_loss`，横轴为 `global_slot`。每个 run 必须恰好2000点。
- 不使用最后一个 slot 近似 episode，不使用 TensorBoard 平滑值，不用 `episode_reward` 做除法，不改变原始数值。

## 边界

统一后的 `train/episode_reward` 只解决“完整 episode 累计窗口”一致性。不同奖励臂使用不同奖励公式，因此该曲线仍不能作为跨奖励臂胜负依据。C1/B2/C2 的正式胜负继续使用 `FormalEval/*/forced_hover/budget_selected/J_per_offer` 固定-tape主协议。

## 验证

- 必须恰好生成33个 `UnifiedTrain` run；
- 每个 run 必须且只能有上述两个 scalar tag；
- 每个 run 的点数必须分别为500和2000；
- `train/episode_reward` 的首末 step 必须为500和250000，`train/critic_loss` 的末 step 必须为250000；
- TensorBoard API 必须发现全部33个 run 和两个 tag。
