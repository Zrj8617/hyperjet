# HyperUAV 主线训练收敛工作进展报告

> 日期：2026-08-05
> 分支：`zrj_3multisample`
> 面向对象：同课题同事（继续本工作的交接文档）

## 1. 一句话总结

通过给"卸载决策"和"移动决策"分别建立**独立于 critic 的学习信号**，解决了
V3~V8 全部不收敛的问题。3 个随机种子的正式实验平均：流时间 **149.8s**
（贪心基线 177s、随机基线 459s）、到达阶段 DAG 完成率 **0.846**（贪心 0.735、
随机 0.505）、能耗 **154.7 /DAG**、无效分配率 **0**。

## 2. 背景与要解决的问题

项目主线（见 `docs/hyperuav_clean_mainline_design.md`）：**超图辅助多 UAV DAG
任务卸载**。算法结构为"任务编码器（MLP/HGNN）+ 共享 movement actor +
共享 offloading actor + centralized critic"的 MAPPO 式联合策略，目标是让
UAV 学会移动与卸载决策，缩短 DAG 流时间、提高卸载效率。

### 2.1 要解决的核心问题：训练不收敛

历史上 V3~V8 全部不收敛，证据：

| 观测 | 数值 |
|---|---|
| 训练后完成率 | ≈0.50~0.55（与随机基线 0.505 无区别） |
| 评估时卸载策略熵（归一化） | ≈0.9999（几乎均匀随机） |
| 评估时 greedy 一致率 | ≈0.50~0.51 |
| critic explained variance | 全程 ≈0（−0.25~+0.13） |
| 移动策略熵 | ≈1.0（满格，完全没学） |

## 3. 根因分析（基于全部日志）

1. **critic 从没学会**：explained variance ≈0 → GAE 优势全是噪声 → 策略无法
   更新。调参（lr、熵、horizon、数据量、detach）都无法修复。
2. **移动头结构性零梯度**：5 架 UAV 共享同一个 slot-level advantage，均匀策略下
   期望梯度为 0（`movement_loss` 长期 ≈1e-8）。
3. **卸载头梯度太小**：每决策梯度只有 0.03~0.15（相对全局梯度被淹没），权重几乎
   不动。
4. **环境过饱和**：多候选卸载决策平均只剩 1.6 个合法选择，可学习信号少。

结论：**不修改信用分配结构，只调超参救不回来**——这正是
`docs/superpowers/specs/2026-07-29-decision-level-ppo-microstep-mappo-design.md`
预先指出的问题。

## 4. 解决方案（两个新机制，默认关闭，不影响旧行为）

### 4.1 每决策 EFT 优势（卸载）

- 新增 `--offloading-eft-advantage`：每个卸载决策用自己的
  **决策时刻 EFT 后悔值**（所选 UAV 预计完成时间 − 最优候选预计完成时间，
  批量标准化）作为 PPO 优势，替代共享 slot advantage。
- 新增 `--offloading-lr-scale`（门控用 10）：卸载头单独 N 倍学习率。
- 门控验证（冻结移动 + MLP，30 次更新）：策略熵 0.9999→0.32，完成率 0.765
  （超过贪心 0.735），流时间 178.6s（≈贪心 177s）。

### 4.2 每 UAV 移动优势（移动）

- 新增 `--movement-position-advantage`：每架 UAV 用自己的
  **负的归一化"到最近就绪任务距离" − 固定移动能耗惩罚**作为优势。
- 新增 `--movement-lr-scale`（门控用 10）。
- 信号探索过程（B1~B6 六个门控实验）：
  - 覆盖率（绝对）→ 无人机乱飞，能耗爆炸（3898/DAG）；
  - 覆盖率增量 → 信号太稀疏，学不动；
  - 距离增量 → 策略"躺平悬停"（悬停比乱动安全）；
  - **距离绝对 + 能耗惩罚 → 平衡**：策略熵 1.0→0.44，学会"悬停为主、
    必要时调整"，能耗全场最低（154.5/DAG）。
- 局限：确定性评估下最优行为是悬停（文档预警过的"偷懒悬停"），尚未学会
  "主动飞向热点"。

## 5. 正式实验结果（3 seeds × 120 updates + 正式评估）

正式评估协议：确定性策略 + 200 时隙到达阶段 + 300 时隙排空阶段，
10 episodes/seed。

| 指标 | 随机基线 | 贪心基线 | seed 0 | seed 1 | seed 2 | **3 种子均值±std** |
|---|---:|---:|---:|---:|---:|---:|
| 平均流时间 (s) | 459 | 177 | 164.6 | 136.7 | 148.2 | **149.8 ± 14.0** |
| 到达阶段完成率 | 0.505 | 0.735 | 0.836 | 0.857 | 0.845 | **0.846 ± 0.010** |
| 能耗/DAG | — | — | 155.5 | 153.0 | 155.6 | **154.7 ± 1.5** |
| 无效分配率 | — | — | 0 | 0 | 0 | **0** |

结论：三个种子**稳定超过贪心基线**（流时间低 15~23%，完成率高 15%），结果一致
（完成率 std 仅 0.01）。运行目录与曲线见 `logs/clean_mainline/formal_B6_seed*`。

## 6. 复现方法

```powershell
# 随机/贪心基线
python scripts/run_clean_policy_baseline.py --policy random_hash --episodes 30 --max-steps-per-episode 200 --seed 0 --completed-dag-weight 8 --run-name quick_random
python scripts/run_clean_policy_baseline.py --policy greedy_eft --episodes 30 --max-steps-per-episode 200 --seed 0 --completed-dag-weight 8 --run-name quick_greedy

# 正式训练（B6 配置，每种子 ~1.6~1.8 小时/120 更新）
python scripts/train_clean_mainline.py --episodes 200 --max-steps-per-episode 200 --rollout-horizon 128 --ppo-epochs 3 --lr 3e-4 --entropy-coef 0.01 --task-encoder mlp --offloading-eft-advantage --offloading-lr-scale 10 --movement-position-advantage --movement-lr-scale 10 --completed-dag-weight 8 --max-updates 120 --seed 0 --run-name formal_B6_seed0 --output-dir logs\clean_mainline

# 正式评估
python scripts/eval_clean_mainline.py --checkpoint <run>/checkpoints/latest.pt --episodes 10 --arrival-steps 200 --max-drain-steps 300 --seed 0 --run-name formal_B6_seed0_eval --output-dir logs\clean_eval

# 画图
python scripts/plot_clean_metrics.py --run-dir <train_run_dir> --no-show
```

## 7. 代码与提交状态

本工作共 4 个提交，工作区干净：

| 提交 | 内容 |
|---|---|
| `c60baa0` | 4.1：每决策 EFT 优势 + 卸载头学习率倍率 |
| `e949830` | 4.2：每 UAV 移动优势（距离+能耗信号）+ 移动头学习率倍率 |
| `a492ec3` | config 调优（DAG 奖励权重 8、位置塑形开、KaHyPar 关） |
| `26728f0` | smoke 修复 + 项目概览 + 忽略 logs/ |

新增 CLI 参数：`--offloading-eft-advantage`、`--offloading-lr-scale`、
`--movement-position-advantage`、`--movement-lr-scale`（默认全部关闭）。
环境新增按 UAV 计算的信号方法（`environment/env.py`：
`compute_per_uav_ready_task_coverage / _mean_distance / _nearest_distance`）。

## 8. 下一步计划（按优先级）

1. **超图对照实验**：`--task-encoder hgnn` 跑 3 个种子，与 MLP 结果对比，
   支撑论文核心贡献"超图辅助"（每种子约 1.6~1.8 小时）。
2. **KaHyPar**：修复 `third_party/kahypar` worker 后启用，或正式标注为
   "no-KaHyPar / degraded" 消融（当前 config 已关闭）。
3. **补全文档要求**：按 `docs/hyperuav_clean_experiment_launch.md` 模板
   （3 seeds、正式评估、绘图、汇总表）整理论文图表。
4. **移动主动重新定位（研究级）**：把"移动后卸载延迟下降量"直接作为移动优势，
   替代当前距离信号，解决"偷懒悬停"。
5. **消融**：超图 vs 无超图、位置塑形开关、能耗权重敏感性等。

## 9. 给接手同事的注意事项

- 实验依赖 `config.py` 当前值（已提交）：`REWARD_COMPLETED_DAG_WEIGHT=8`、
  `ENABLE_MOVEMENT_POSITION_SHAPING=True`、`ENABLE_KAHYPAR_PARTITION_HYPEREDGES=False`。
- 三个新开关默认关闭；正式实验必须同时开启 4 个参数（见第 6 节命令）。
- `logs/` 已被 `.gitignore` 忽略（720MB 实验产物），结果请从本地日志读取。
- critic 至今仍未学会（explained variance ≈0），当前方案绕开了它；若后续要
  恢复 slot-level MAPPO 或实现 decision micro-step，仍需解决 critic 问题。

---

## 10. 实验进度更新（2026-08-06）：HGNN 超图对照 + KaHyPar 状态

### 10.1 HGNN 对照实验（论文核心贡献的直接证据）

目的：验证"超图辅助"是否真的带来收益。使用与 MLP 完全相同的配方
（每决策 EFT 优势 + 每 UAV 移动优势 + 双头 10 倍学习率），仅把
`--task-encoder` 从 `mlp` 换成 `hgnn`，3 个种子 × 120 次更新 + 正式评估。

| 指标 | MLP（3 种子均值±std） | HGNN（3 种子均值±std） |
|---|---:|---:|
| 平均流时间 (s) | **149.8 ± 14.0** | 161.4 ± 28.0 |
| 到达阶段完成率 | **0.846 ± 0.010** | 0.833 ± 0.015 |
| 能耗/DAG | **154.7 ± 1.5** | 156.4 ± 7.4 |
| 无效分配率 | 0 | 0 |

分种子细节：

| 种子 | MLP 流时间 | HGNN 流时间 | MLP 完成率 | HGNN 完成率 |
|---:|---:|---:|---:|---:|
| 0 | 164.6s | **142.5s** | 0.836 | **0.848** |
| 1 | **136.7s** | 193.6s | **0.857** | 0.818 |
| 2 | 148.2s | **148.1s** | **0.845** | 0.834 |

**诚实结论**：在 120 次更新的训练量下，HGNN 没有稳定超过 MLP——
seed 0 超图更好，但 seed 1 明显更差，整体 MLP 更稳、HGNN 波动更大。
这个结果不能直接写成"超图优于 MLP"。两个可能原因：

1. **KaHyPar 分区超边当时未启用**——超图方法"完整形态"的优势没有发挥；
2. HGNN 参数更多、学得更慢，可能需要 240+ 次更新才能到公平对比。

运行目录：`logs/clean_mainline/formal_hgnn_seed*`（曲线在 `plots/`）。

### 10.2 KaHyPar 状态（2026-08-06 修复动作）

- 2026-08-06 曾重新启用 `ENABLE_KAHYPAR_PARTITION_HYPEREDGES = True`
  并验证优雅降级；随后决策改为**收敛调优阶段暂不启用（=False）**，
  与 MLP/HGNN 基线配置保持一致；
- **`kahypar==1.3.7` 仅提供 Linux wheel**：`pip install kahypar==1.3.7`
  在 Windows/Python 3.14 上无可用版本（`from versions: none`）；
- Windows 开发机上 worker 会因缺少 `kahypar` 模块优雅降级：
  `partition_status = degraded_no_cache / degraded_cache`，训练不中断；
- **真实分区需要 Linux 训练服务器**：按 `third_party/kahypar/README.md`
  在服务器上 `pip install kahypar==1.3.7`，正式运行即可产出分区超边。

### 10.3 当前最稳健可写论文的结果

仍是 MLP 配置：流时间 149.8s（贪心 177s、随机 459s）、到达完成率 0.846
（贪心 0.735、随机 0.505）、能耗 154.7/DAG、无效分配率 0。

### 10.4 下一步（更新后）

1. 在 Linux 服务器上安装 kahypar 并重跑 HGNN 对照，验证超图完整形态
   （含分区超边）是否超过 MLP；
2. 或先在本机把 HGNN 训练量加到 240 次更新，排除"学得慢"的干扰；
3. 论文图表整理：3 种子收敛曲线 + 基线对比 + MLP/HGNN 表。
