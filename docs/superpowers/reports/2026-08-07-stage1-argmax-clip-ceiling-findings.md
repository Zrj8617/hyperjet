# Stage 1 argmax 排序失败：D0 结论与下一步

日期：2026-08-07
分支：`zrj_3multisample`　基础提交：`6b6cc75` + 未提交 working tree
数据来源：`/data2/zrj2025/HyperUAV/logs/stage1_temperature_followup/20260806_formal_v1/static_corpus/records.jsonl`（服务器侧 D0 离线分析）
状态：**D0 完成并定论。未修改代码、未进入 D1、未重训。**

---

## 1. 结论：clip 天花板确立

D0 的三项判据全部指向同一结论，且互相独立：

| 判据 | 观测 | 判定 |
| --- | --- | --- |
| margin≥20 上 clip 后 argmin 保留率 | **19%–24%** | 远低于 §决策表设定的 0.90 阈值 |
| 精确最高-logit 平局率 | **0%** | 排除"argmax 退化为最小 UAV ID 破平" |
| 未饱和 vs 饱和 accuracy | **96–98% vs 56–58%** | 排除普遍训练不足与表达能力不足 |

**H1（归一化饱和造成可识别性损失）确立。H2（训练不足）与 H3（scorer 表达能力）被实证否定。**

### 1.1 数字自洽性验证

用 seed42 的分层数据反推整体 margin≥20 accuracy：

```
0.192 × 0.9813  +  0.808 × 0.5844  =  0.188 + 0.472  =  0.660
```

与正式实验报告的 deterministic margin20 accuracy 64%–67% 一致。三个 seed 同样成立。
**分层分解完全解释了原始指标，不存在未归因的残差。** 这是 D0 结论最强的一条支撑。

### 1.2 为什么平局率是 0% 而不是接近 1

`incremental_delay` 饱和不意味着整行特征相同。同一决策内各候选仍然有差异的维度：

- 7 维 dynamic UAV 特征：`pos_x`、`pos_y`、`queue_length/16`、`remaining_slots/16`、`queued_workload`、`slot_assigned`
- pair 中依赖距离的项：`transfer_time`、`return_time` 及对应能耗

（`compute_time` 对所有 UAV 相同——`num_operation / UAV_COMPUTE_RATE_OPS_PER_SEC`，
`UAV_COMPUTE_RATE_OPS_PER_SEC` 是全局常量，不随 UAV 变化，因此它在候选间**不携带任何判别信息**。）

所以 logits 仍然互不相等，argmax 不退化。这解释了饱和子集上 56%–58% 的准确率
——显著高于 5 候选的随机基线（约 20%–25%），也高于 19%–24% 的 clip 天花板：

> **actor 在饱和区退化为"按队列长度/位置的代理策略"。**
> 这个代理与 EFT 相关但不等于 EFT，因此稳定地错约 42%。

### 1.3 可迁移性证据（决定下一步是否值得做的关键）

未饱和区 96%–98% 的准确率证明：**scorer 已经学会了 `incremental_delay → 更低分` 的正确单调映射**。
缺的不是模型能力、不是训练量、不是标签，**只是特征的动态范围**。

因此若能在保序前提下恢复动态范围，margin≥20 accuracy 的上界预期在 **0.96–0.98** 附近，
远高于 `static_temperature_gate` 要求的 0.90。这是一个可量化的收益，值得投入一次重训。

---

## 2. 一个必须先解决的岔路：是特征错了，还是负载错了

D0 报告了一个容易被忽略的数字：**全候选饱和率约 70%（69.47%–72.52%）**。

这意味着在约 70% 的决策中，**连最优 UAV 的 `incremental_delay` 都 ≥ 40 秒**（即 ≥ 8 个时隙）。
系统长期处于深度积压状态。粗算也支持这一点：

```
到达：NUM_UES=60 × 每时隙到达率 ≈ 1 个 DAG/时隙 × DAG_MIN..MAX_TASKS(5..8)
      ≈ 6–10 个任务/时隙
服务：NUM_UAVS=5 × TIME_SLOT_DURATION=5s = 25 UAV·秒/时隙
负载：以 nlogn 为主（概率 0.7）的平均 compute_time 约数秒 ⇒ 每时隙到达工作量与服务能力同量级
```

于是有两种互斥解释，**它们导向不同的修法**：

**分支 A —— 深度积压是预期工作点。**
则 `CLEAN_NORM_AVAIL_TIME_REF = 8 × TIME_SLOT_DURATION = 40 s` 相对真实延迟分布小了一个数量级，
归一化参考值本身选错了。修特征是正解。

**分支 B —— 当前过载是标定问题，预期工作点应是轻载。**
则修特征只是在给一个负载标定问题打补丁；而且在全面积压时，
min-EFT 本身退化为"选最短队列"，EFT-greedy 作为学习目标的信息量也随之下降。

### 解决岔路的零成本测量（D0.5-A）

从**现有 corpus** 直接算，无需任何新 rollout：

```
对全部决策，min_legal_incremental_delay = min_i (eft_i) − 5.0 × (slot_index + 1)
报告：P10 / P25 / P50 / P75 / P90 / P99，以及随 slot_index 的时间演化曲线
```

判读：

- 若 P50 在 **40–200 s** 量级且**随 slot_index 趋于平稳** → 分支 A，稳态积压，直接修特征。
- 若 P50 **随 slot_index 单调发散**（队列一路涨到 cap 16 不回落）→ 分支 B，系统不稳定，
  此时应先复核负载标定，**修特征无法解决根因**。

这一步只需要读一遍 corpus，几分钟 CPU。**建议在写任何设计文档之前先出这条曲线。**

---

## 3. 下一步：D0.5-B 变换选型（仍然零 rollout、零重训）

前提：D0.5-A 判定为分支 A。

修法必然要改 `_normalize_pair_features`（`environment/assignment.py:497-509`），
这**属于 actor 输入变更**，超出上一轮"不改 actor 输入"的约束，需要你显式批准，
并走独立 design + gate + 重训。但**参数选择可以完全离线完成**，不必先动代码。

### 3.1 选型判据：分离度匹配，而非仅仅保序

所有严格单调变换的 argmin 保留率都是 100%（浮点精度内），所以"保序"不构成区分度。
真正的判据是**分离度是否够大到能被 scorer 用上**。用现有数据构造经验参考：

```
对每个 margin≥20 决策、每个候选变换 φ：
    Δφ = φ(次优 incremental_delay) − φ(最优 incremental_delay)

参考基准 Δ* := 当前"未饱和 margin≥20 子集"（actor 已达 96–98%）上 Δ 的 P10

选型条件：在当前"饱和 margin≥20 子集"上，≥90% 的决策满足 Δφ ≥ Δ*
```

这个判据的好处是：基准来自 actor **实际已经做对的**样本，不是拍脑袋定的阈值。

### 3.2 候选变换

| # | 形式 | 优点 | 风险 |
| --- | --- | --- | --- |
| 1 | `clip(x / ref, 0, 1)`，ref 扫 {40, 80, 160, 320, 640} | 改动最小，输入仍在 [0,1] | 仍有 clip；ref 增大会**压缩低延迟区的分离度**，而那正是当前 96–98% 的来源 |
| 2 | `x / (x + ref)` | 恒在 [0,1)，**永不 clip**，严格单调，无新超参形式 | 大 x 处 Δ 衰减快 |
| 3 | `log1p(x / ref) / log1p(x_cap / ref)` | 相对分辨率近似均匀，两端都保留分离度 | 引入 `x_cap`，多一个需要标定的常量 |
| 4 | **保留原维**，另加一维 `(x − min_legal) / (max_legal − min_legal + eps)` | 分离度由构造保证为 O(1)，与负载水平完全解耦 | pair 维度 8→9，actor 输入维变化；特征依赖候选集合，语义从绝对量变为相对量 |

同一次评估应同时覆盖 `queue_waiting_time`（pair idx 2，共用同一个 40 s ref）。
另外需要在 D1 里确认 dynamic 特征 idx 4 `available_delta`（**同样 /40 clip**）是否一并饱和；
若是，则判别信息实际上是被**两处**同时截断的，只修 pair 不够。

### 3.3 我的倾向

**先看 #2 和 #4。**

- **#2** 是最小侵入的"去 clip"改法：保持 [0,1] 输入范围、保持单一常量、永不丢序。
- **#4** 是唯一**与负载水平无关**的方案——无论系统积压多深，相对编码的分离度都是 O(1)。
  代价是多一维输入。既然无论如何都要重训，这个代价是可接受的。

若 D0.5-B 显示 #2 在饱和子集上无法达到 Δ* 判据（大 x 处衰减太快是可预见的风险），
则 #4 是唯一稳健的选择。

---

## 4. 好消息：评估侧一行都不用改

`Stage1TemperatureDiagnosticEnv` + 冻结 tape + `deterministic_masked_argmax`
构成一套现成的、与修法无关的验收闸门：

- `build_manifest`（`stage1_temperature_tape.py:262-297`）记录的 config 字段为
  `num_ues / num_uavs / active_dag_cap / queue_cap / area / uav_altitude / hotspot_radius /
  time_slot_duration / dag 到达参数 / ue 移动参数`——
  **不包含任何 `CLEAN_NORM_*` 参考值**。
- 因此修改归一化常量**不会使已冻结的 tape 失效**，`validate_manifest` 仍然通过，
  `logical_tape_sha256` 不变，新旧 checkpoint 可以在**逐 bit 相同的场景材料**上直接对比。

需要改的只有 `FROZEN_CHECKPOINTS`（`stage1_temperature_diagnostic.py:13-17`）中新增三条
路径 + SHA-256 条目。这是本次修复中最省事的一环。

---

## 5. 验收闸门（重训后）

在同一冻结 tape 上，用同一套 `run_stage1_temperature_followup.py --phase formal` 复跑：

| 指标 | 阈值 | 来源 |
| --- | --- | --- |
| `deterministic_margin20_accuracy` | ≥ **0.90**（三个 seed 全部） | `deterministic_reachability` 已有判据 |
| 全候选饱和率 | ≤ **0.05** | 新增，验证机制确实被消除 |
| margin≥20 clip 后 argmin 保留率 | ≥ **0.98** | 新增 |
| 未饱和子集 accuracy | 不低于当前 0.96 | **防退化**：确认低延迟区能力没有被新归一化损坏 |
| `checkpoint_guardrail` 五项闭环指标 | 无 material degradation | 已有实现（`stage1_temperature_analysis.py:105-113`） |
| `technical_pass` | True | 已有实现 |

**若 deterministic margin20 达到 ≥0.90，则 `deterministic_reachability` 转为 `reachable`，
temperature 通道重新打开，Stage 2 的前置条件才第一次成立。**

重训成本与原 S1-B 相同：3 seeds × 30 updates × 128 slots × 3 epochs，不需要扩预算
（D0 已证明当前预算足以在信息完整时达到 96–98%）。

---

## 6. 门控状态

| 项 | 状态 |
| --- | --- |
| D0（离线 corpus 分析） | ✅ 完成，结论 = clip 天花板 |
| D0.5-A（负载分支判定） | ⬜ 待执行，零成本，**下一步应先做这个** |
| D0.5-B（变换选型） | ⬜ 待执行，零成本，依赖 D0.5-A = 分支 A |
| D1（corpus 富化重跑） | ⬜ 仅在需要确认 `available_delta` 是否同步饱和、或需任务类型/通信-计算分层时才做 |
| 修改 `_normalize_pair_features` | ⛔ **需你显式批准**（属 actor 输入变更），且必须在 D0.5-B 出参数后 |
| 重训 | ⛔ 需独立 design + gate |
| Stage 2 | ⛔ 维持关闭 |
