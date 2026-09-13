# mini-spec: 补齐缺失臂 + 20-seed 补评（2026-09-07）

**状态：** APPROVED（用户）　**执行：** Codex　**类型：** 训练 + 评估

## 为什么跑这一批
上一批 21 run 存在两个致命缺口：
1. **A1(最原始奖励)从未在可比条件下运行** —— 之前复用的 Stage-1 OFF 臂是 100ep/20-seed 评估,而 A2 是 500ep/1-seed 评估,**不可比**。⇒ 用户原始需求第 2 点从未被测试。
2. **C1(EFT 老师退火)从未运行** —— 用户原始需求第 4 点空缺。
3. 另需找回被 Claude 指令冲突毁掉的 local-ΔΦ 诊断臂。

## 必须先修的两个缺陷
- **评估协议**：训练 runner 每个 checkpoint 必须评估 **20 个环境 seed**(上一批只评了 1 个)。口径与 leverage check / Stage-1 完全一致：forced-hover、deterministic masked argmax、同一 seed 集。
- **TensorBoard 口径**：`train/episode_reward`(应为整 episode 累计)、`move/move_rate_m`(应为整 episode 移动动作占比)、`forecast/wall_time_frac`(应为 forecast/总训练墙钟)——三者当前写的是局部 rollout/slot 值,与标签语义不符,必须修正。

## 三个新臂（各 3 seeds × 500 episodes）
| 臂 | 定义 |
|---|---|
| **A1** | **最原始奖励**：时延(完成时结算,原公式)+任务能耗+移动能耗+`+8` 完成奖励+覆盖塑形;slot GAE;无老师;所有新 gate 全关。**必须与 A2 完全同条件**(同 500ep、同超参、同评估协议) |
| **C1** | **老师退火**：前 50% updates 开 EFT-anchored offloading advantage + movement centroid advantage;中间 10% 线性退火老师权重到 0;后 40% 纯 **A1 原始奖励** + slot GAE。**完整记录 entropy 在退火前/中/后的轨迹**——这是判据 |
| **N0-local** | 与 N0 完全相同(解析 ΔΦ actor advantage,batch 标准化),**唯一区别 η=0**：ΔΦ 只计当前 DAG,不计跨 DAG 外部性 |

## 20-seed 补评（纯评估,不重训）
对**全部 9 个配置 × 3 seeds = 27 个 checkpoint** 重新评估：
- 上一批 6 个 distinct：`N0 / A2 / B1 / B2(=D1) / B2D / D2`
- 本批 3 个新配置：`A1 / C1 / N0-local`
每 checkpoint × 20 环境 seed,口径同上。产出统一对照表,含 `random / eft_greedy / Stage-1 EFT-anchored` 参考线。

## 判据
- **C1**：退火后 entropy 是否弹回 ~0.999(未学住)或维持在低位(学住了);系统指标是否维持在 greedy 之上。
- **A1 vs A2**：用户第 2 点的真实答案,首次可比。
- **N0-local vs N0**：local 明显更好 ⇒ 跨 DAG 外部性被过度加权,ΔΦ 尚可救;local 也差 ⇒ ΔΦ 路线整体关闭。

## 不做
不重建预注册版 B2D(微状态 critic + variable-discount GAE)——N0 已证明 ΔΦ 信号本身会把策略带偏,换更复杂的信用通路传同一个坏信号大概率白费。
