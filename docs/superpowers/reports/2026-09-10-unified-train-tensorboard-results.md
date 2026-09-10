# UnifiedTrain TensorBoard 统一训练口径结果（2026-09-10）

**状态：PASS**

当前33个正式训练 run 已全部从原始 `train_metrics.jsonl` 重建为独立 `UnifiedTrain/<原始 run>` 视图。原始 event、原 run 和 `FormalEval` 固定-tape结果均未覆盖。

## 审计结论

- 旧 2026-09-06 event 中的 `train/episode_reward` 实际写入 rollout/update 总奖励，共2000点，不是完整 episode总奖励。
- 所有33个原始 JSONL 都包含500条 `episode_terminal_record=true` 的完整 episode记录以及2000条 PPO update记录。
- 统一视图的 `train/episode_reward` 只取终止记录的 `episode_reward_total`，每个 run 恰好500点，step为500至250000。
- 统一视图的 `train/critic_loss` 取每个 update 的 `ppo_value_loss`，每个 run 恰好2000点，最终 step为250000。
- 未使用最后一个 slot、平滑值、除法近似或原错误 event 的同名数据。

## 覆盖范围

| 批次 | 实验臂 | seeds | run数 |
|---|---|---:|---:|
| 20260906 | N0、A2、B1、B2、B2D、D1、D2 | 3 | 21 |
| 20260907 | A1、C1、N0-local | 3 | 9 |
| 20260908 | C2 | 3 | 3 |
| 合计 | 11个标签臂 | — | 33 |

## 验证

- EventAccumulator：33/33 PASS；每个 run 只有两个指定 tag；500/2000点数全部正确。
- TensorBoard API：发现33个 `UnifiedTrain` run，错误 run 数为0。
- 当前 TensorBoard 共75个 run：33个原训练 run、9个 `FormalEval` run、33个 `UnifiedTrain` run。

## 使用方式

- 地址：`http://127.0.0.1:16008/#timeseries`
- Runs过滤：`UnifiedTrain/`
- Tags过滤：`train/episode_reward|train/critic_loss`
- TensorBoard显示平滑仅影响画面，不影响已写入原始值；严格读数时将 smoothing 调为0。

## 解释边界

统一视图只统一了累计窗口和字段语义。不同奖励臂的环境奖励公式仍不同，因此不能直接用 `train/episode_reward` 的高低判断跨奖励臂胜负。C1/B2/C2 的正式比较继续以 `FormalEval/*/forced_hover/budget_selected/J_per_offer` 为主。

## 路径与版本

- 服务器事件根：`/data2/zrj2025/uav-results/audits/20260910_unified_train_tensorboard`
- TensorBoard view：`/data2/zrj2025/uav-results/tensorboard_views/reward_redesign/UnifiedTrain`
- 导出器：`scripts/export_unified_train_tensorboard.py`
- 校验器：`scripts/check_unified_train_tensorboard.py`
- 设计 commit：`f15b8f97857585a43be037e62d5f96fa992b8d03`
- 执行时工作树为 dirty（76条）；未清理或覆盖既有改动。

