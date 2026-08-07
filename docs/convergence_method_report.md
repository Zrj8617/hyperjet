# HyperUAV 实验收敛方法报告

> 日期：2026-08-07
> 分支：`zrj_3multisample`
> 目的：向同事说明"用了什么方法让实验收敛、收敛结果是什么"，便于对齐和复现
> 配套文档：[进度报告](progress_report.md)（记录完整实验历史）

## 1. 背景：之前为什么一直不收敛

项目目标：多 UAV 学会"往哪飞 + 任务卸载给谁"，为地面用户提供边缘计算服务，
最小化 DAG 任务处理时间、提高完成率、控制能耗。

V3~V8 全部不收敛的观测证据：

| 观测 | 数值 |
|---|---|
| 训练后 DAG 完成率 | ≈0.50~0.55（与随机基线 0.505 无区别） |
| 卸载策略熵（归一化） | ≈0.9999（几乎全随机） |
| critic explained variance | 全程 ≈0（学不会） |
| 移动策略熵 | ≈1.0（完全没学） |

根因（基于全部日志诊断）：

1. **critic 学不会**：GAE 优势全是噪声，策略无法更新；
2. **移动头结构性零梯度**：5 架 UAV 共享同一 slot advantage，均匀策略下期望
   梯度为 0（`movement_loss` 长期 ≈1e-8）；
3. **卸载头梯度太小**：每决策梯度仅 0.03~0.15，权重几乎不动；
4. **环境过饱和**：多候选卸载决策平均只剩 1.6 个，可学习信号少。

结论：**不修改信用分配结构，只调超参救不回来**。

## 2. 收敛方法：五个关键组件

### 2.1 每决策 EFT 优势（卸载）

- 开关：`--offloading-eft-advantage`
- 做法：每个卸载决策用自己的"决策时刻 EFT 后悔值"（所选 UAV 预计完成时间
  − 最优候选预计完成时间，批量标准化）作为 PPO 优势，替代共享 slot advantage。
- 作用：绕过学不会的 critic，给每个卸载决策一个独立、可信的学习信号。

### 2.2 服务质心移动信号（移动）

- 开关：`--movement-position-advantage`
- 做法：每架 UAV 用"**移动后到服务需求质心的距离**（负向归一化）− 能耗惩罚"
  作为自己的移动优势。服务需求质心 = 正在等待服务的 UE 位置均值
  （`environment/env.py::compute_service_demand_centroid`，无等待 UE 时退回
  热点中心/全部 UE 均值）。
- 作用：无人机学会"飞到用户密集区一次，然后悬停服务"，而不是追着单个任务
  乱飞（追单任务曾导致能耗 3846/DAG）或躺平悬停。
- 这是**收敛提升最大的单一改动**：流时间从 ~150s 降到 ~58~66s。

### 2.3 独立学习率（解决"梯度太小"）

- `--offloading-lr-scale 10`：卸载头单独 10 倍学习率；
- `--movement-lr-scale 5`：移动头单独 5 倍学习率（10 倍会不稳定）。
- 作用：两个头的梯度只有全局梯度的几十分之一，共用学习率时权重几乎不动。

### 2.4 切断 critic 对编码器的污染

- 开关：`--detach-critic-hgnn`
- 做法：critic 的 value loss 梯度不再流入共享任务编码器（HGNN/MLP）。
- 作用：学不会的 critic 不再搅乱编码器特征（HGNN 下效果明显）。

### 2.5 移动能耗惩罚（平衡"移动"与"节能"）

- 参数：`config.CLEAN_MOVEMENT_ENERGY_PENALTY_SIGNAL = 0.0045`
- 历程：0.01 → 无人机躺平悬停；0.003 → 乱飞（能耗 3846/DAG）；
  **0.0045 → 平衡**（先飞到用户区，之后悬停服务）。

## 3. 收敛结果

### 3.1 最终正式结果（3 种子 × 120 次更新 + 正式评估）

正式评估协议：确定性策略 + 200 时隙到达阶段 + 300 时隙排空阶段，10 episodes/seed。

| 配置 | 流时间 | 完成率 | 能耗/DAG | 无效分配 | 移动行为 |
|---|---:|---:|---:|---:|---|
| 随机基线 | 459s | 0.505 | — | — | 悬停 |
| 贪心基线 | 177s | 0.735 | — | — | 悬停 |
| **MLP + 本方法** | **58.2 ± 3.1s** | **0.942 ± 0.007** | 202.1 ± 17.1 | 0 | 飞到用户区后悬停 |
| HGNN + 本方法 | 66.4 ± 1.6s | 0.927 ± 0.007 | **192.7 ± 4.4** | 0 | 飞到用户区后悬停 |

- **MLP + 本方法流时间比贪心快 3 倍，完成率高 20.7 个百分点**；
- 3 种子几乎零波动（流时间 std ≤ 3s，完成率 std ≤ 0.007）——强收敛证据。

### 3.2 收敛过程证据（seed 0，120 次更新）

| 更新数 | 卸载熵 | 移动熵 | 完成率 | 流时间 |
|---:|---:|---:|---:|---:|
| 10 | 0.70~0.81 | 0.99 | 0.545 | ~440s |
| 30 | 0.27~0.33 | 0.64~0.89 | 0.90~0.93 | ~76~183s |
| 60 | 0.17~0.18 | 0.11~0.20 | 0.91~0.98 | ~67~82s |
| 90 | 0.14~0.16 | 0.05~0.16 | 0.93~0.95 | ~38~62s |
| 120 | 0.12~0.18 | 0.15~0.26 | 0.89~0.94 | ~57~124s |

策略熵在 ~30 次更新内从接近 1 降到 0.2 以下（策略定型），完成率稳定在
0.9+，之后只有单次 rollout 的固有噪声（正式评估取 10 集平均）。

## 4. 复现命令

```powershell
python scripts/train_clean_mainline.py --episodes 200 --max-steps-per-episode 200 --rollout-horizon 128 --ppo-epochs 3 --lr 3e-4 --entropy-coef 0.01 --task-encoder mlp --detach-critic-hgnn --offloading-eft-advantage --offloading-lr-scale 10 --movement-position-advantage --movement-lr-scale 5 --completed-dag-weight 8 --max-updates 120 --seed 0 --run-name formal_mlp_centroid_seed0 --output-dir logs\clean_mainline

# 评估
python scripts/eval_clean_mainline.py --checkpoint <run>/checkpoints/latest.pt --episodes 10 --arrival-steps 200 --max-drain-steps 300 --seed 0 --run-name eval --output-dir logs\clean_eval
```

把 `--task-encoder mlp` 换成 `hgnn` 即得 HGNN 版本。依赖当前 config：
`REWARD_COMPLETED_DAG_WEIGHT=8`、`ENABLE_MOVEMENT_POSITION_SHAPING=True`、
`CLEAN_MOVEMENT_ENERGY_PENALTY_SIGNAL=0.0045`、KaHyPar 暂关。

## 5. 对论文的意义与诚实说明

1. **收敛提升主要来自"服务质心移动信号"**（与编码器无关），加上每决策
   EFT 优势解决了 MAPPO 联合策略不收敛问题；
2. 当前 **MLP 优于 HGNN**（流时间快 12%、完成率高 1.5pp），HGNN 仅能耗略低；
3. 建议论文定位：核心贡献 = "**每决策信用分配 + 服务质心移动**的联合训练
   方法"；HGNN 作为编码器消融（与 MLP 相当）；超图完整形态（KaHyPar
   分区超边）列为未来工作（需 Linux 服务器）；
4. 遗留事项：critic 至今未学会（方法绕过它）；移动行为为"飞到用户区后
   悬停"而非持续跟踪单用户。

## 6. 与同事对齐的要点

- 训练能收敛，是因为**每个决策/每架 UAV 都有自己的学习信号**，不再依赖
  学不会的 critic 共享积分；
- 移动信号必须指向"服务需求质心"而不是"最近任务"（后者导致乱飞/躺平）；
- 卸载头、移动头需要**各自的放大学习率**，否则梯度太小学不动；
- 结果可复现：3 种子几乎零波动，MLP 58.2s / 0.942，HGNN 66.4s / 0.927；
- 详细实验历史见[进度报告](progress_report.md)。
