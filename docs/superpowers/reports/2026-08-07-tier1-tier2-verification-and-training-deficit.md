# Tier 1–2 核查结果：工作点结论、以及一个被漏掉的主因

日期：2026-08-07
数据：`analysis_inbox/`（Tier 1 原件 + Tier 2 汇总，源 SHA-256 与 manifest 一致）
状态：**所有待验证项已闭环。发现第二个独立主因，且它比 clip 更便宜。**

---

## 1. 你的判读全部正确，我逐条复核过

| 你的结论 | 复核 |
| --- | --- |
| p50 从首档 17–18 s 升至 85–103 s 后走平 | ✅ 三个 checkpoint 一致（16.99/17.92/18.45 → 85.1/91.8/92.1），第 4 档起进入平台 |
| 属 active-DAG cap 稳住的高压稳态，非无界增长 | ✅ 见 §2 |
| 首档 p50 < 40 s，不支持"单次服务本身就慢" | ✅ 但首档 p90 已达 52–56 s，饱和率 17–21%，所以"慢"的尾巴一开始就有 |
| 2–3 个合法候选约 39%，非多数 | ✅ 精确 38.5%/39.1%/38.8% |
| 后半程平均合法候选降至 3.0–3.2 | ✅ 从首档 5.00 单调降至 3.0–3.2 后走平 |
| 平均 UAV 队列 ≈ 12.80/16 | ✅ 12.48–12.96（随温度） |
| `dag_completion_rate` 标准差 ≈ 0.093 | ✅ 各温度 0.088–0.093 |
| p99 = 271–281 s，触及 Tier 3 条件 | ✅ 但结论是**不要做**，见 §5 |

---

## 2. 工作点结论：高压但稳定，归一化参考值偏小约 4–5 倍

### 2.1 稳定性

`min_legal_incremental_delay` 的 p50 轨迹（seed 均值）：

```
slot 0-19   →  17.8 s     饱和率 20%     平均合法候选 5.00
slot 20-39  →  51.1 s     饱和率 63%     平均合法候选 4.62
slot 40-59  →  84.9 s     饱和率 88%     平均合法候选 3.74
slot 60-79  →  91.8 s     饱和率 91%     平均合法候选 3.34
slot 80-199 →  85-103 s   饱和率 85-93%  平均合法候选 3.0-3.2   ← 平台
```

**前 60 个时隙是瞬态爬升，之后进入平台。不是无界增长。**

但要清楚**它为什么稳**：`active_dag_cap=1` 起的是**准入控制**的作用。
每个 UE 在其 DAG 完成前不再产生新 DAG，所以到达率被完成率反向锁住。
这是闭环流控掩盖过载的经典形态——**系统稳定，但稳定在一个很拥挤的点上**。

### 2.2 拥挤程度（来自 `closed_loop/episodes.jsonl`，1200 行）

| 指标 | T=1.0 | T=0.25 |
| --- | --- | --- |
| `completed_dag_count` | 50.41 ± 13.21 | 60.43 ± 15.36 |
| `generated_dag_count` | 91.49 ± 11.16 | 99.32 ± 12.15 |
| `dag_completion_rate` | 0.542 ± 0.088 | 0.600 ± 0.090 |
| `average_dag_flowtime` | 431.9 ± 78.1 s | 367.7 ± 73.7 s |
| `avg_uav_queue_length` | 12.96 / 16 | 12.48 / 16 |
| `admitted_incomplete_backlog` | 41.1 | 38.9 |

- **约 45% 的 DAG 在 200 个时隙内没完成**（backlog 39–41 / generated 91–99）
- 队列持续占用 **78–81%**，5 架 UAV 里常态有约 2 架满
- 平均 DAG 流时间 368–432 s，而整个 episode 只有 1000 s

### 2.3 关于"98.64% 节流代理"——这个归一化会误导

`arrival_blocked_count / (blocked + generated) = 0.9864` 的分母不对。
`arrival_attempt_count` 每时隙对每个 UE 都 +1，所以每 episode 固定是 `60 × 200 = 12000`。

正确的读法：

```
blocked / 12000     = 0.569    ← UE 平均有 in-flight DAG 的时间占比
generated / 12000   = 0.00785  ← 实际每 UE-slot 到达率
（未受阻抽样：94 / 5173 = 0.0182，与 base 0.0145 + hotspot ×2 相符，到达过程本身正常）
```

**56.9% 的 UE-时隙被自身在途 DAG 阻塞**，这才是节流强度。
98.64% 那个数只是在说"阻塞次数远多于成功到达"，而由于一个 DAG 会连续阻塞几十个时隙，
这个不等式是恒成立的，不携带负载信息。

### 2.4 对 clip 的直接结论

稳态 `min_legal_incremental_delay` 分布（三 checkpoint 一致）：

```
p10 ≈ 13 s    p25 ≈ 35 s    p50 ≈ 70 s    p75 ≈ 111 s    p90 ≈ 159 s    p99 ≈ 275 s    max ≈ 500 s
```

而 `CLEAN_NORM_AVAIL_TIME_REF = 40 s` **落在 p25–p30 之间**。

> **参考值应该在 160–200 s 量级（约 4–5×），或者干脆换成无界保序变换。**

而且既然是稳态而非发散，**修归一化是有意义的，不需要先改负载**。
我上一轮提的"分支 A / 分支 B"到此关闭：**答案是 A**，且不必动到达率或容量。

---

## 3. ⚠️ 被漏掉的第二个主因：训练严重不足（而且这个更便宜）

这是我看训练日志才发现的，它和 clip 是**两个独立问题**。

### 3.1 证据

来自 `analysis/formal_analysis.json`（评估侧）：

```
normalized_entropy      = 0.9852 – 0.9928     （均匀分布 = 1.0）
max_action_probability  = 0.3385 – 0.3560     （平均 3.9 个合法候选，均匀 ≈ 0.26）
sampled_greedy_agreement = 0.3307 – 0.3413    （随机 ≈ 0.26）
deterministic_greedy_agreement = 0.6214 – 0.6381
```

**采样策略几乎就是均匀随机。** 而 argmax 却有 62–64% 的正确率。
所以 logits 的**方向**是对的，**幅度**远远不够，softmax 压根没被拉开。

来自 `tier1/training/seed*/updates.jsonl`（训练侧），**这才是决定性的**：

| 指标 | 全部 30 个 update、全部 3 个 seed |
| --- | --- |
| `approx_kl` | **≈ 0.0000**（量级 1e-4，且时常为负 = 估计噪声） |
| `clip_fraction` | **恒为 0.0000**（PPO 裁剪从未触发过一次） |
| `ratio_std` | 0.002 – 0.016（新旧策略比值 ≈ 1.000） |
| `normalized_entropy` | update 1 = **1.0000** → update 30 = 0.990–0.997 |
| `top1_top2_probability_margin` | 0.0019 → 0.039–0.053 |
| `raw_eft_regret_mean` | 30 个 update **无明显下降趋势**（seed86: 52.5→62.8） |
| 训练总决策数 | **5,331 / 6,331 / 5,796** |

模型参数量约 **36,800**（encoder 9,920 + scorer 26,881）。

> **用约 6,000 个决策训练 36,800 个参数，30 个 update 里熵只降了 0.3%–1.0%，
> PPO 裁剪一次都没触发。策略基本没离开初始化。**

### 3.2 为什么方向学到了、幅度没学到

`incremental_delay` 是 EFT 排序的**单特征精确充分统计量**（未饱和时）。
只要在这一维上带一点点权重，argmax 就能对——所以方向收敛得极快。
但要把 softmax 拉开，需要 logits 的**尺度**增长若干倍，那是慢得多的过程。

这解释了一个此前看似矛盾的现象：
未饱和区 argmax 准确率 96–98%（学得很好），而熵却是 0.99（几乎没学）。
**两者说的是同一个策略的不同侧面，不冲突。**

### 3.3 收益已经被温度扫描量化了

`T=0.25` 等价于把 logits **乘 4**，这正是"幅度不够"的粗暴补偿。效果（配对比较，1200 行）：

```
completed_dag_count    +10.02 / episode   (+19.9%)   配对 SE=0.42   t ≈ 23.6
average_dag_flowtime   -64.20 s           (-14.9%)   配对 SE=2.36   t ≈ -27.2
dag_completion_rate    +0.057                        配对 SE=0.0032 t ≈ 17.7
```

且 `closed_loop_guardrails` 在 T=0.75/0.5/0.25 **全部 pass**，无任何 material degradation。

**仅仅把现有 logits 锐化，闭环吞吐就 +20%。而训练充分本来就应该自己产生这个锐化。**

### 3.4 关键：诊断证明"加大训练"是安全的

`clip_fraction` 恒为 0 → PPO 的信任域从未生效一次
`approx_kl` ≈ 1e-4 → 离不稳定还差 2 个数量级

**所以可以放心大幅提高训练强度**，不存在通常担心的策略崩溃风险。
`sample_count` 每 update 只有 60–360，梯度噪声也很大，batch 应该同时加大。

---

## 4. 修正我上一轮的一个判断

我上一轮说"H2（训练不足）已被实证否定，理由是未饱和区 96–98%"。**这个否定是错的。**

正确的分解是：

| 症状 | 机制 | 证据 | 修法 | 成本 |
| --- | --- | --- | --- | --- |
| **argmax 排序不足**（0.643/0.661/0.672） | clip 天花板 | 未饱和 96–98% vs 饱和 56–58%，margin20 中饱和占 76–81% | 改归一化参考值 | 改 actor 输入 + 重训 + 独立 gate |
| **概率不集中**（熵 0.985–0.993） | 训练严重不足 | KL≈0、clip_frac=0、总决策 6k、熵 30 update 只降 1% | **只加 `--updates`** | **零代码改动** |

**你在第一轮就说"当前问题同时包括概率不集中和 argmax 排序不足"——这个判断完全正确，
现在两半都有了各自独立确认的机制。**

---

## 5. Tier 3：建议不做，并撤回"奖励端饱和"假设

- 奖励 clip 触发阈值 = `CLEAN_REWARD_TIME_REF × CLEAN_REWARD_TIME_CLIP = 60 × 10 = 600 s`
- 实测 `min_legal_incremental_delay` 的 **p99 = 271–281 s，max = 484–514 s**，全部 < 600 s
- 且 `_task_incremental_delay`（`metrics.py:298-312`）算的是**相对最晚父任务完成时刻的单跳时延**，
  比"从 DAG 到达起的累计等待"更小

结论：**`CLEAN_REWARD_TIME_CLIP` 实际上不生效。"观测与奖励同步饱和"这条正式撤回。**
不值得为此花一次 replay。（若将来要留档，一次单 replicate 只读 replay 即可闭环，但没有必要。）

---

## 6. 一个新发现的方法论坑：`dag_completion_rate` 是被混淆的指标

`generated_dag_count` **本身就依赖策略**：
策略越好 → DAG 完成越快 → UE 越早解除 active-DAG cap → 产生的 DAG 越多。

实测：T=0.25 生成 99.32，T=1.0 生成 91.49，**差 8.5% 完全由策略造成**。

于是 `dag_completion_rate = completed / generated` 是**两个策略依赖量之比**，
一个更好的策略可能因为分母变大而让比值看起来提升不明显，极端情况下甚至下降。

> **主指标应该用 `completed_dag_count`（绝对吞吐），
> `dag_completion_rate` 只作为辅助。**

`run_manifest.json` 里的 `active_dag_cap_pairing_limitation` 其实已经点到这件事，
但它的影响范围比"配对不严格"更大——**它使一个常用指标本身失去单调性**。

### 顺带：配对比较的收益已量化

| 指标 | 配对 SE | 非配对 SE | 功效提升 |
| --- | --- | --- | --- |
| `completed_dag_count` | 0.424 | 1.170 | **2.76×** |
| `average_dag_flowtime` | 2.359 | 6.201 | **2.63×** |
| `dag_completion_rate` | 0.0032 | 0.0072 | **2.24×** |

冻结 tape + 配对 bootstrap 应该成为所有策略对比的默认通道。

---

## 7. 原始结论全部核实

| 第一轮的结论 | 状态 |
| --- | --- |
| `technical_pass` | ✅ True |
| 降温改善 sampled policy 与闭环 | ✅ 配对 t≈23.6，guardrails 三档全 pass |
| deterministic margin20 = 64%–67% | ✅ 0.6428 / 0.6607 / 0.6723 |
| 分类 `ranking_limited_margin20` | ✅ `classification="ranking_limited"`，`reasons=["ranking_limited_margin20"]`（措辞区分见前述） |
| 同时存在概率不集中与 argmax 排序不足 | ✅ 且两者机制现已分离确认 |
| 不继续扫更低温度 | ✅ 三档 `common_pass=False`；且 `max_achievable_regret_reduction=0.54–0.61 ≥ 0.50` 本已通过，唯一未过项就是 margin20 |

---

## 8. 修改后的行动顺序

**P0 —— 加大训练量（零代码改动，直接跑）**

```
--updates 300 (10×)   --slots-per-update 256 (2×，降梯度噪声)
可选 --lr 1e-3（clip_fraction 恒为 0，信任域从未生效，提 lr 安全）
```

盯四条曲线：`normalized_entropy` / `max_action_probability` /
`behavior.margin20_accuracy` / `approx_kl`（这些字段训练脚本**已经在记了**）。

**预注册预测（可证伪）：**

- 熵从 0.99 显著下降，`max_action_probability` 明显上升
- 闭环 `completed_dag_count` 在 **T=1.0** 下达到甚至超过当前 T=0.25 的水平（≈60）
- **`deterministic_margin20_accuracy` 仍停在 0.66–0.72 附近**

**若第三条成立 → clip 天花板得到最干净的确认**（同样的模型、同样的输入，
只是训练充分了，排序天花板仍在），此时改归一化的理由无可辩驳。
**若 deterministic margin20 反而升到 0.9 以上 → clip 假设需要重新审视。**

**P1 —— 依 P0 结果决定归一化修法**
参考值 40 s → 160–200 s，或换 `x/(x+ref)` 这类无界保序变换。属 actor 输入变更，需独立 design + gate。

**P2 —— 评估口径切换**
主指标改用 `completed_dag_count`，所有对比走冻结 tape + 配对 bootstrap。

**P3 —— 才轮到超图注入点、UAV 节点、headroom 测试**
（见 `2026-08-07-hypergraph-injection-point-and-novelty.md`）

**不需要：** 改到达率、改容量、改 reward、跑 Tier 3、动 `phase4_p0_baseline_200slot`。
