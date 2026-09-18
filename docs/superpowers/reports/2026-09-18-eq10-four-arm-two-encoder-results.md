# 2026-09-18 EQ10 四臂 × 双编码器结果（正式训练已启动）

## 1. 当前状态与结论边界

- 状态：`formal_training_launched`。
- 8 个五 episode smoke 单元已全部通过；24 个正式训练任务已于 2026-09-18 21:40:26（Asia/Shanghai）在 7 张 GPU 上后台启动。
- 按冻结协议，启动后不做 tail/轮询。训练完成前不读取测试 tape、不做统一评估，也不作 MLP 与 typed-gated HGNN 的效果结论。
- 预计正式训练约需 48–72 小时；这是由本次 smoke 的实测吞吐与正式并发数外推，不是完成承诺。

机器可读阶段结果：`docs/superpowers/reports/2026-09-18-eq10-four-arm-two-encoder-result.json`。

## 2. 已核验前提（不是沿用文档声明）

### 2.1 奖励账本与实验矩阵

直接执行奖励配置代码确认：

- 本实验 B2/C2A/C2B/C2 均为 `lambda_task = lambda_move = 1.0`。
- 未传 EQ10 参数时，B2/C2A/C2B/C2 仍保持旧默认 `1.0/0.10`，D2 保持 `1.0/0.04`，所以 gate OFF 兼容旧行为。
- 训练矩阵为 4 arms × 2 encoders × 3 seeds（5/86/617），共 24 个任务；每任务 500 episodes、每 episode 500 steps、rollout horizon 125。
- C2A/C2B/C2 的 teacher anneal 总更新数为 2000；B2 为 0。
- KaHyPar 未启用。

### 2.2 tape 与场景

复用 tape 根目录：`/data2/zrj2025/uav-results/audits/20260915_six_arm_realign_tapes`；其生成版本为 `2450a40`。

只读取 validation IDs 100–119 做了场景审计，共 20 条 tape、11,094 个 offers、71,995 个 tasks。已逐项核对：500×500 区域、5 UAV、60 UE、slot 5 s、arrival 0.0145、每 DAG 5–8 tasks、4 levels、每任务最多 3 parents、input 0.1–1.0、output 0.15–0.75、task constant 0.5M–1.5M、上行带宽集合 1.75/3.5/7 MHz、下行 3.5/7/14 MHz、UAV compute 1e9。

本阶段没有读取测试 tape payload；测试集保持封存，等 validation 选点锁定后才允许统一测试。

### 2.3 容量对齐与 RNG-neutral

实际构图参数量：

| 编码器 | task encoder | actor total | critic | total |
|---|---:|---:|---:|---:|
| MLP（encoder hidden 773） | 59,585 | 90,630 | 29,825 | 180,040 |
| typed-gated HGNN（encoder hidden 128） | 59,588 | 90,630 | 29,825 | 180,043 |

总参数相对差为 `0.001666%`，低于 10% 门槛；非编码器模块参数量完全一致。typed-gated HGNN 使用独立 RNG 上下文构造，公共训练模块的初始化序列与容量对齐 MLP 保持一致。

### 2.4 旧 movement 基线的日志复核

从 2026-09-15 固定 tape 联合预算选点的机器可读结果重新计算 `m_effective = move_energy / (500 J × 5 UAV × 500 slots)`，三 seed 聚合值为：B2 `0.0507%`、C2A `0.2248%`、C2B `0.4893%`、C2 `0.6077%`。

这与任务说明中的“B2 约 0.36%、C2 约 2.3%”不一致。因此后者不作为本实验 gate 或推论前提；这里只保留从旧原始结果重算出的数值作为历史参照。

## 3. Smoke gate

机器可读结果：`/data2/zrj2025/uav-results/audits/20260918_eq10_four_arm_two_encoder_smoke.json`。

- 8/8 单元完成，每单元 5 episodes、20 PPO updates。
- 全部 arm/encoder 的实际开关、EQ10 系数、teacher 配置与冻结矩阵一致。
- 非 treatment 配置差异为空。
- MLP/HGNN 容量门通过。
- typed-gated HGNN 的 DAG、k-hop、attribute、partition 四类归一化超边权重在检查单元的全部 20 条训练诊断中存在。
- 首次汇总曾把编码器专属初始化元数据误列为非 treatment 配置；训练本身均已完成。修正汇总分类后复用原 8 个结果验收，没有重跑或覆盖 smoke 训练。

## 4. 正式训练启动记录

服务器工作树：`/data2/zrj2025/HyperUAV-gating-20260915`。

- 实际 HEAD：`ba6de8b91afba4da330c6c40fbde0ec70f6133ba`。
- 启动时 `git status --porcelain`：空（0 dirty files）。
- 启动清单：`/data2/zrj2025/uav-results/audits/20260918_eq10_four_arm_two_encoder_training_launch.json`。
- 24 个后台 PID：1477200–1477223；每个任务的精确 PID/GPU/log/result 路径见启动清单和机器可读阶段结果。
- GPU 分配数：GPU 0/1/2/3/4/5/6 分别为 4/2/2/5/2/5/4 个任务。启动前 GPU 1、2、4 已有其它负载，因此把更多任务分配给当时较空闲的 0、3、5、6，同时仍使用全部 7 张卡。

版本标识（由 `git hash-object` 记录，不用作内容断言）：

| 文件 | 标识 |
|---|---|
| `scripts/launch_eq10_four_arm_two_encoder_production.py` | `44ffbf5bb8ba81281ebc213dcd5fae44ea04e5c6` |
| `scripts/run_reward_redesign_arm.py` | `ebe18536d1b4f603afb7d8ffc703d28f6060ead2` |
| `scripts/train_clean_mainline.py` | `870bae5f935a4d70a97659fb163160cf09c1a4d2` |
| `environment/reward_redesign.py` | `31041d8cbe8f4cbdb165ffbd8e8d34d561b018e8` |

## 5. 待完成工作

训练完成后才执行冻结的统一评估：先用 validation tapes 选定预算/检查 hard stops，锁定选择后再读取 test tapes；评价目标统一为

`J = (flowtime + 1.0 × task_energy + 1.0 × move_energy) / 500`。

届时把 validation/test 指标、选点依据、原始 JSON/log 路径和最终结论追加到本报告，并更新同名机器可读结果。当前没有触发算法级 hard stop；“训练尚未完成”只是正常运行状态。
