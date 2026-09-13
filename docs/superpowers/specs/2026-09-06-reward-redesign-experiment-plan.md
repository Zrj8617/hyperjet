# 奖励重设计实验方案（2026-09-06 定稿 v1 · 待用户额外要求后冻结）

## 0. 前置工程与两个测量探针（全部先于任何训练臂）

| # | 内容 | 验证/产出 |
|---|---|---|
| P0-a | 拆字段：`estimated_compute_finish_time` / `estimated_return_finish_time` | **纯增量,行为不变**;gate-OFF 逐值复现基线 |
| P0-b | **`Q_forecast` 影子时钟**：仅供预测,模拟 executor 分配阶段(sink 只推进到 compute_finish),槽начале从 executor 真实状态初始化,按接受顺序独立推进 | 不改 actor 的 `Q_policy`(仍含回传)、不改 reservation、不改 RNG、不改环境轨迹 |
| P0-c | forecast DP：**数组化 `O(E·U²)`**,预缓存拓扑/传输矩阵,NumPy 小矩阵,禁止 deepcopy 与重复调用通信模型。**禁用** 5³×5 父节点组合朴素枚举 | 单独 smoke 验证正确性 |
| P0-d | 微决策 reward 通路 + **微状态 critic**(新 schema) | 默认关闭;gate-OFF 复现基线 |
| **M1** | **P_residual 实测探针**：slot 500 停止新到达 → 冻结参考策略继续 drain → 记录每个未完成 DAG 的 `R_g = C_g − T` → 20 seed 求均值 → 取整到 50/100 档 | **500s 只是占位符,必须用实测值替换** |
| **M2** | **ΔΦ 标定探针**（无训练,固定策略,20 seed）：记录 `zero_fraction`(阈值 1e-9)、p25/50/75/90/95/99、mean positive ΔΦ、target-DAG Δ、cross-DAG Δ、affected/actually-changed DAG count、**forecast wall time** | 用于定尺度;并据 `forecast_wall/collection_wall ≤ 10%` 决定用 exact-global 还是降级方案 |

**降级方案(仅当 M2 显示 exact-global 过慢)**：当前 DAG 精确 + top-K 精确 + **敏感度尾部近似**
`ΔΦ ≈ ΔT̂_G^exact + ΔQ_u · Σ_{H≠G} g_{H,u}`，缓存 `queue_headroom[H,u]` 与 `reroute_gap[H,u]`。
**禁止**直接丢弃 top-K 之外的 DAG(会系统性低估公共资源外部性)。

## 1. 冻结的系数与尺度

```
F_ref = 500 s      E_task_ref = 250 J      E_move_ref = 2500 J/slot
λ_task = 1.0 s/J   →  β_task = 0.5
λ_move = 0.10 s/J  →  β_move = 0.5        （按目标运行区间 m_ref = 0.30 定）
B_complete = P / F_ref = 1.0              （新尺度主实验）
```

- **λ_move 训练内固定,禁止按实测 m 动态调整**：`λ_move(m)·E_move(m) ∝ (1/m)·m ≈ const`,动态调整会**摧毁"少移动"的边际激励**;EMA/滑窗只能减缓漂移,不能消除。
- `λ_move = 0.04` 保留为**预注册对照**;敏感性分析可扫 `0.06 / 0.10 / 0.14`,但每次训练内固定。
- **完成奖励与未完成 residual 惩罚二选一**,不可并用(固定到达数下二者只差一个常数)。

## 2. 时延奖励的两种分解（必须账本对齐）

两臂必须归约到**同一个 undiscounted 截尾 flowtime**：
`−(1/F_ref)[ Σ_{completed}(C_G−a_G) + Σ_{unfinished}(C̄_G−a_G) ]`

**Arm H（holding 分解,"不增量"）**
```
R_H = −(1/500) [ Σ_t 5·A_t  +  Σ_{unfinished} R̂_G(H) ]
```

**Arm Δ（forecast-increment 分解,"增量"）— 必须补齐三项,缺一不可**
```
R_Δ = −(1/500) [ Σ_G (T̂_G^0 − a_G)        ← DAG admission 时结算,记在 slot boundary,
                                              不归因给该 DAG 的第一个卸载动作
               + Σ_k ΔΦ_k                  ← 每个微决策结算,Φ(S)=Σ_{active G} T̂_G
               + Σ_G (C̄_G − T̂_G^last) ]    ← 实际完成/终止截尾校正
```

⚠️ **只用 `−ΣΔΦ` 是 assignment-regret 代理,不是 flowtime**。初始项 `T̂_initial` 受此前决策影响,**不是可忽略常数**——省略它,策略可以把拥塞后果推到"下一批 DAG 出生之前"从账本里逃掉。若只用 regret 口径,实验结论必须命名为 **assignment-regret surrogate**,不得称作 flowtime reward。

⚠️ `γ_slot = 0.99` 下两臂的 **discounted** return 不同——这正是要研究的 credit/timing 差异,但**不得宣称两个训练目标在折扣后数学等价**。

## 3. 实验臂

**A 组 — 单变量消融（保留原始奖励结构,含 +8）**

| 臂 | 配置 | 说明 |
|---|---|---|
| A1 | 原始奖励 | ✅ 已有(entropy 0.9998 不学) |
| **A2** | 原始奖励,仅时延项 → **Arm Δ 完整账本**,其余(+8=8.0/能耗/覆盖)不动 | 🆕 主。**结论受限**:只能说"增量时延在 completion-dominated reward 下是否有效" |

**B 组 — 新尺度主实验（重标定,完整账本 A/B）**

| 臂 | 配置 |
|---|---|
| **B-H** | 新尺度 + **Arm H**(holding + residual,无完成奖励) |
| **B-Δ** | 新尺度 + **Arm Δ**(初始项 + ΣΔΦ + 校正,无完成奖励) |
| B-Δ-local | 同 B-Δ 但 η=0(只本 DAG) — 外部性消融,后跑 |
| B-move04 | 同 B-H 但 λ_move=0.04 — 预注册系数对照 |

**C 组 — 老师退火**：等 A/B 结果,退火进入表现最好的奖励。

**可选 D 组（不同问题,别与 B 组混谈）**：两臂都用 holding 基础奖励,B 额外加 `γ_n·Ψ(s')−Ψ(s)`,`Ψ=−Φ/scale`,终止势为 0 → 测"势能塑形能否改善 credit",而非用 surrogate 替换真实 flowtime。

## 4. 微决策 credit 的实现约束

状态链：`D(t,1) → … → D(t,m) → B(t) → D(t+1,1)`（无 ready task 时 `B(t)→B(t+1)`）

| | reward | γ_n | λ_n |
|---|---|---|---|
| 微决策 D | `−ΔΦ_n/S` | **1** | **1** |
| 槽边界 B | DAG 初始项 + 完成校正 + holding + 能耗 + 完成奖励 + 终止截尾 | 0.99 | 0.95 |

variable-discount GAE：`d_n=γ_n(1−terminal_n)`;`δ_n=r_n+d_n V_old(s_{n+1})−V_old(s_n)`;`A_n=δ_n+d_n λ_n A_{n+1}`;`R̂_n=A_n+V_old(s_n)`。
PPO 用冻结、batch 标准化后的 `A_n`;critic 回归**未标准化**的 `R̂_n`;old value/advantage/target 在 PPO epochs 前一次算完并冻结。
**choice decision 进 actor loss;forced/skip/boundary 只进 critic loss,不进 actor loss 与 actor-advantage 标准化。**

**微状态 critic 输入必须包含**(不可复用 slot-level 输入,否则 critic 看不到 reservation 演化)：当前任务与合法候选、每 UAV 的 **`Q_policy` 与 `Q_forecast`**、queue length/workload、本槽已分配数、剩余 frozen-ready 摘要、active DAG 全局摘要、decision order、boundary flag。**两条实验臂用相同 critic schema。**

终止：已发放截尾结算 ⇒ 视为该有限时域目标的真 terminal,`V(next)=0`;**不得同时发截尾结算又从 episode 外 bootstrap**。

## 5. 评估口径与论文表述

- **正式跨策略评估用固定外生 offer tape**(预生成每 UE 每 slot 的潜在到达与 DAG 属性;UE 忙碌时计为 **blocked** 而非不生成)。训练环境可暂保留原生 closed-loop。
- 必须同时报三个率：`r_admission=N_adm/N_off`、`r_conditional=N_comp/N_adm`、**`r_end-to-end=N_comp/N_off`（唯一跨策略可比）**。
- 原 `ρ=0.84` 是 admitted 口径,**不能**直接用于 offered 口径(端到端可能仅约 0.43–0.49),须按 fixed-tape 基准重新标定。
- **论文不得称"最小化物理总能耗"**,应称"归一化的时延—任务服务能耗—移动运营成本复合目标";并同时报告原始任务能耗、移动能耗、总焦耳数。当前 hover=0 / move=500J 的抽象更宜称 **mobility/relocation cost** 而非完整推进能耗。

## 6. 统一测量

每臂 3 seeds、short gate、与 `random / eft_greedy` 参考带同口径。
必测：entropy 轨迹、critic EV、completion(三个率) / flowtime / throughput、agreement rate、**实测移动率 m**、forecast wall-time 占比。
