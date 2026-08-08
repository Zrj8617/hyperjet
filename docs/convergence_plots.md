# HyperUAV 收敛图说明文档

> 日期：2026-08-08
> 用途：配合[收敛方法报告](convergence_method_report.md)和
> [进度报告](progress_report.md)，用图说明"训练是怎么收敛的、怎么看收敛图"。
> 图片来源：`logs/clean_mainline/formal_mlp_centroid_seed*` 与
> `logs/clean_mainline/formal_hgnn_centroid_seed*` 运行目录下的 `plots/`。

## 1. 怎么看一张收敛图（通用判断标准）

判断"收敛"看三个特征：

1. **快速爬升/下降后进入平台**：指标在前 20~30 次更新内快速变化，之后不再
   系统性上升或下降，只在噪声带内波动；
2. **多个随机种子几乎重合**：不同 seed 的曲线最终落在同一个水平（std 很小）；
3. **训练过程稳定**：损失不发散、无 NaN/Inf。

> 注意：单次 rollout（128 个时隙）的指标噪声较大，论文/报告请以
> **滑动平均 + 多集评估**为准。

## 2. 最优配置：MLP + 服务质心信号（3 种子均值 58.2s / 0.942）

以下均为 seed 0，120 次更新。

### 2.1 DAG 完成率与吞吐（最直观的收敛证据）

![MLP 完成率与吞吐](../logs/clean_mainline/20260807_162005_formal_mlp_centroid_seed0_seed0/plots/train_completion_throughput.png)

**怎么看**：完成率从第 1 次更新的 ~0.2 快速爬升，约 30 次更新后进入
**0.90~0.95 的平台期**，之后不再上升——策略"定型"了。吞吐（每秒完成的
DAG 数）同步上升并稳定。

### 2.2 流时间与能耗（性能收敛）

![MLP 流时间与能耗](../logs/clean_mainline/20260807_162005_formal_mlp_centroid_seed0_seed0/plots/train_energy_flowtime.png)

**怎么看**：平均流时间从 ~450s 一路掉到 **~60s** 后稳住；能耗在低位小幅波动。
"快速下降 → 平台"说明无人机学会了飞到用户区、卸载学会了选对 UAV。

### 2.3 奖励曲线（最"抖"的一张，用滑动平均看趋势）

![MLP 奖励曲线](../logs/clean_mainline/20260807_162005_formal_mlp_centroid_seed0_seed0/plots/train_reward.png)

**怎么看**：每个点是单次 rollout 的总奖励，噪声很大（DAG 完成奖励稀疏 +
时间惩罚波动），但整体**趋势向上并围绕平台波动**。论文里对这条曲线做
滑动平均后就是平滑的收敛曲线。

### 2.4 损失曲线（训练稳定性）

![MLP 损失曲线](../logs/clean_mainline/20260807_162005_formal_mlp_centroid_seed0_seed0/plots/train_losses.png)

**怎么看**：策略损失/值损失全程有限、无发散（单次波动属于正常），说明优化
过程稳定，没有出现梯度爆炸或 NaN。

## 3. 对照配置：HGNN + 服务质心信号（3 种子均值 66.4s / 0.927）

seed 0，120 次更新。曲线形状与 MLP 完全一致（快速收敛 → 平台），只是最终
平台略高。

![HGNN 完成率与吞吐](../logs/clean_mainline/20260807_120142_formal_hgnn_centroid_seed0_seed0/plots/train_completion_throughput.png)

![HGNN 流时间与能耗](../logs/clean_mainline/20260807_120142_formal_hgnn_centroid_seed0_seed0/plots/train_energy_flowtime.png)

## 4. 跨种子一致性（3 个种子都收敛到同一水平）

以 MLP 的完成率/流时间为例，seed 1 的曲线与 seed 0 几乎重合：

![MLP seed1 流时间与能耗](../logs/clean_mainline/20260807_172021_formal_mlp_centroid_seed1_seed1/plots/train_energy_flowtime.png)

3 种子正式评估汇总：

| 配置 | 流时间（3 种子均值±std） | 完成率（均值±std） |
|---|---:|---:|
| MLP + 质心 | **58.2 ± 3.1s** | **0.942 ± 0.007** |
| HGNN + 质心 | 66.4 ± 1.6s | 0.927 ± 0.007 |

流时间标准差只有 1~3 秒、完成率标准差 0.007——不同随机种子跑到几乎相同的
结果，这是"真收敛"最硬的证据。

## 5. 收敛特征总结

| 图 | 收敛特征 | 结论 |
|---|---|---|
| 完成率/吞吐 | 快速爬升后 0.9+ 平台 | 卸载+移动策略学会 |
| 流时间 | 450s → ~60s 后平台 | 性能收敛 |
| 能耗 | 低位稳定 | 无乱飞浪费 |
| 奖励 | 上升后围绕平台波动 | 用滑动平均呈现 |
| 损失 | 有限、无发散 | 训练稳定 |
| 多种子 | 曲线/指标几乎重合 | 结果可复现 |

> 如需在 Git 仓库外共享本文档，请连同 `logs/clean_mainline/formal_*` 的
> `plots/` 目录一起拷贝（`logs/` 已在 `.gitignore` 中，不随仓库分发）。
