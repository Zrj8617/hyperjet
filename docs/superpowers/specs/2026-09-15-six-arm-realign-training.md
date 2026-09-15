# 六臂场景重标定 —— 第二阶段：18 个正式 run 与固定 tape 评测

日期：2026-09-15
分支：`param-realign-20260914`
前置：`docs/superpowers/specs/2026-09-15-six-arm-realign-gating.md` 已验收通过
（报告 commit `6722311`，服务器 HEAD `70c862a`，G1–G9 全部 PASS）

## 0. 范围

训练 6 个奖励/教师臂 × 3 个 seed = **18 个正式 run**，在新场景下用固定外生 tape 完成
validation 选点与 test 评测，产出可用于论文的 2×2 因子结论。

第一阶段已完成的内容（C2A/C2B 开关、rollout 125、编排去硬编码、joint 选点、
候选 checkpoint 320–500、hover ratio、2×2 汇总、λ_move sweep）**不再重做**，
本 spec 只负责跑与分析。

## 1. 实验矩阵

seed：**5、86、617**（六臂共用同一组）

| 臂 | 奖励 | 卸载老师 | 飞行老师 | 在 2×2 中的位置 |
|---|---|:-:|:-:|---|
| B1 | 实际 flowtime 记账 | ✗ | ✗ | 2×2 之外（记账方式对照） |
| B2 | 增量式三段账 | ✗ | ✗ | 都没有 |
| C2A | 同 B2 | ✓ | ✗ | 只有卸载老师 |
| C2B | 同 B2 | ✗ | ✓ | 只有飞行老师 |
| C2 | 同 B2 | ✓ | ✓ | 都有 |
| C1 | 原始奖励 | ✓ | ✓ | 2×2 之外（奖励定义对照） |

## 2. 训练参数（六臂逐字节相同，只有 `--reward-redesign-arm` 与 `--seed` 变）

| 组 | 参数 | 值 |
|---|---|---|
| 回合 | `--episodes` | 500 |
| | `--max-steps-per-episode` | 500 |
| | `--rollout-horizon` | **125**（4 次更新/episode，全程 2000 次） |
| | `--num-envs` / `--sampler-backend` | 1 / synchronous |
| PPO | `--lr` | 3e-4 |
| | `--gamma` | **0.99** |
| | `--gae-lambda` | 0.95 |
| | `--clip-ratio` | 0.2 |
| | `--entropy-coef` / `--value-coef` | 0.01 / 0.5 |
| | `--ppo-epochs` | 1 |
| | `--normalize-value-targets` | 开 |
| 网络 | `--task-encoder` | **mlp**（全部，HGNN 是下一阶段） |
| | `--hidden-dim` / `--task-embedding-dim` | 128 / 64 |
| 教师 | `teacher_anneal_total_updates` | 2000（仅 C1/C2/C2A/C2B） |
| | `hold` / `end` | 0.50 / 0.60 → ep 250 起退火、ep 300 归零 |
| 其他 | `--no-dag-progress-potential-shaping` | 固定 |
| | `--checkpoint-interval` | 10 |

场景参数取分支当前 `config.py`（commit `0aa78f0` 起）：5 UAV / 60 UE / 每 DAG 5–8 任务 /
`arrival 0.0145` / 输入 0.1–1.0 MB / 输出 0.15–0.75 MB / `TASK_CONSTANT 5e5–1.5e6` /
上行 [1.75,3.5,7.0] 下行 [3.5,7.0,14.0] Mbps / 算力 1e9 cycles·s⁻¹。

**启动前必须断言**：六个 run 的 resolved flag 与门禁 spec §5 表逐格一致；
任一不符立即停止（硬停止 §8.1）。

## 3. 固定外生 tape

- **必须按新场景重新生成**，目录另起，**绝不复用**旧场景的 100–119 / 200–249 文件
- validation：ID 100–119（20 条）；test：ID 200–249（50 条）
- tape 与策略无关：offer 序列、DAG 属性、UE 轨迹预生成；未接纳的 offer 仍记为 offer
- manifest 记录实际实验 commit 与完整场景参数；replay 前逐项校验 UAV/UE 数、时隙长度、
  任务属性范围、带宽档与算力
- **test tape 在选点锁定之前不得被读取**

## 4. 评测协议

1. **validation（只跑 joint）**：每个 run 在候选 checkpoint
   **320 / 360 / 400 / 450 / 500** 上跑 20 条 validation tape，取 `J_per_offer` 均值最低者；
   平局取更早的 checkpoint。
2. **锁定**：把选中的 checkpoint 写进 manifest，冻结。
3. **test（两个协议都跑，同一个 checkpoint）**：
   - `--protocol joint` → **主表**
   - `--protocol forced_hover` → 附表（卸载隔离诊断）
   - 同时对 `final_ep0500` 跑一次（不经选择，单独报告）
4. test 结果**不得反过来换 checkpoint**。

统一代价：

```
J_episode   = [ Σ_G L_G + 1.0·E_task + 0.10·E_move ] / 500
J_per_offer = J_episode / max(N_offer, 1)
L_G = C_G − a_G（完成）；T_end − a_G（未完成或未接纳，截尾计入）
```

**λ_move 敏感性**：用同一批 test 结果离线重算 `λ_move ∈ {0.10, 0.04, 0.0}` 三档的
J/offer 与臂间排名。三档排名一致则标注结论稳健；出现翻转必须在结论中明示
"排名由该人为权重决定"。

## 5. 必报指标

每臂 × 每 checkpoint 标签 × 每协议：

- `J_per_offer` 及三个分量（delay / task energy / move energy）
- admission / conditional / end-to-end 完成率
- `completed_DAG_flowtime` 均值与中位数、throughput
- **`hover_action_ratio`**（joint test；每 seed 均值 + 两层 bootstrap 95% CI）
- 训练期 episode 300–500（教师归零后）的 `hover_action_ratio`
- forced_hover 协议下 hover ratio 恒为 1，仅作协议自检

所有臂间差值报**两层 bootstrap 95% CI**（先重采样 3 个训练 seed，再在 seed 内重采样 tape）
与 **3/3 · 2/3 · 1/3 的 seed 方向**。50 条 test tape 不得当作 50 个独立训练重复。

## 6. 分析口径（措辞纪律）

```
移动老师平均主效应 = 0.5 × [(C2B − B2) + (C2 − C2A)]
卸载老师平均主效应 = 0.5 × [(C2A − B2) + (C2 − C2B)]
交互效应           = C2 − C2A − C2B + B2
```

- 单独的 `C2 − C2A`、`C2 − C2B` 仍报告，但必须标注为**条件效应**
- `C2 − C1` 称为**完整奖励定义的效应**（差异不止记账时序，还含 `+8` 完成奖励、
  覆盖整形 `+0.5·q_t` 和不同参考尺度）
- `B2 − B1` 称为**两个不同目标函数的对比**。**禁止**表述为"同一实际代价的分期记账效应"
  —— 该说法已于 2026-09-15 证伪（G9-3：`b2_total ≈ 0.51 × b1_total`，缺口 100% 由
  未记账的预测漂移解释；`Σ ΔΦ` 只占预测总漂移的约 43%）
- **禁止**跨奖励臂比较训练奖励曲线；训练曲线只看 `UnifiedTrain/` 视图，且只用于
  稳定性判读，不参与排名

**预期与关注点**：`C2A`（有卸载老师、无飞行老师、B2 奖励中无任何正向位置信号）
是 hover collapse 的高危配置。若其 joint hover ratio 显著高于 C2B/C2，即为
"飞行老师是防塌陷必需品"的直接证据，应写入结论而非当作异常。

## 7. 机时与调度

- 训练：18 run × 250k slot，用满 7 张卡自行分配显存（可一卡多进程）
- 评测：
  - joint validation：18 × 5 候选 × 20 tape = **1800** 个评测 episode
  - test：18 × 2 checkpoint × 2 协议 × 50 tape = **3600** 个
  - 合计约 5400 个 500-slot 评测 episode（纯推理）
- 命名含年月日：`YYYYMMDD_<ARM>_seed<SEED>`，接 TensorBoard
- 结果写入已有结果根，不新建目录树，**不得覆盖任何旧 run**
- 挂上服务器后停止监控，返回 PID / log path / result path / 预计完成时间

## 8. 硬停止条件（触发即停并报告，不得自行决策）

1. 任一 run 的 resolved flag 与门禁 spec §5 表不符
2. tape 校验失败（场景参数与 manifest 不一致，或误用旧 tape 目录）
3. test tape 在选点锁定前被读取
4. 任一 run 出现 NaN / 训练崩溃 / checkpoint 缺失
5. 任一 run 的实际参数与本 spec §2 不符
6. 存在覆盖旧结果目录的风险
7. 需要修改本 spec 的任何冻结决定

## 9. 交付物

- `docs/superpowers/reports/2026-09-15-six-arm-realign-training-results.md`
- `docs/superpowers/reports/2026-09-15-six-arm-realign-training-result.json`

必须包含：18 run 的实际参数回填（**以 run 记录为准，不得从 config 推断**）、
resolved flag 实测表、选点清单与 validation 分数、两个协议的 test 表、
三档 λ_move 的排名、2×2 主效应与交互项及其 CI 与 seed 方向、hover ratio 表、
服务器 HEAD / dirty 行数 / 每个脚本的 git object hash、触发的硬停止条件（如有）。

## 10. 给 Codex 的提示词

```
读仓库根 AGENTS.md（特别是 §4 第 11 条），再完整阅读
docs/superpowers/specs/2026-09-15-six-arm-realign-training.md，按该 spec 执行。

前置门禁已通过：docs/superpowers/specs/2026-09-15-six-arm-realign-gating.md，
报告 commit 6722311，G1-G9 全部 PASS。C2A/C2B 开关、rollout 125、joint 选点、
候选 checkpoint 320-500、hover ratio、2x2 汇总、lambda_move sweep 都已实现，不要重做。

本轮任务：训练 6 臂 x 3 seed = 18 个正式 run，然后按固定 tape 完成 validation 选点
与 test 评测，产出 2x2 因子结论。

固定要求：每个实验跑 500 轮次；用满 7 张卡最快跑完，自行检查显存并分配，可一卡多进程；
规范化命名含年月日并接 TensorBoard；结果写入已有目录不新建；挂上服务器后停止监控，
返回 PID / log path / result path 和预计完成时间，据此定定时任务；尽可能节省额度，
不要浪费时间在小地方；记录服务器实际 HEAD、dirty 行数、每个脚本的 git object hash。

三条最容易出错的地方，请特别小心：
1) 固定 tape 必须按新场景重新生成，目录另起，绝对不要复用旧场景的 100-119 / 200-249；
2) 选点只用 joint validation，锁定后同一个 checkpoint 再跑 joint 和 forced_hover 两个
   test；test tape 在锁定前不得读取；
3) 所有参数以 run 的实际记录回填报告，不要从 config.py 推断。

一路自主执行，中途不要问我。触发 spec §8 任一硬停止条件就停下报告，禁止近似降级。
结果写：
docs/superpowers/reports/2026-09-15-six-arm-realign-training-results.md
docs/superpowers/reports/2026-09-15-six-arm-realign-training-result.json
```
