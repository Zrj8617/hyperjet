# 数据请求：用与 20260806_formal_v1 匹配的只读数据验证负载状态

日期：2026-08-07
目的：把"当前系统是否结构性过载"从**推测**变成**结论**，用与最新正式结果同源的数据。
原则：**全部只读**。不重跑训练、不改任何代码、不改配置。
Tier 1–2 完全不需要新的 rollout。

放置位置：`D:\CodeFile\HyperUAV\analysis_inbox\`（任意新建目录都行，我能读到）。

---

## 背景：为什么需要这一步

上一轮我用 `runs/phase4_p0_baseline_200slot`（2026-07-11，`queue_cap=80`、
无 active-DAG cap、无冻结 commit/manifest/checksum）测出的 `ρ≈1.38`，
外推到了 `20260806_formal_v1`。这是无效外推，已在
`2026-08-07-research-direction-assessment.md` §0.4 逐条作废。

**当前已确立的只有 clip 天花板**（8-6 D0 独立证实）。
"结构性过载"和"奖励端饱和"都还是假设。

其中**奖励端饱和这条，靠读代码就基本可以否掉**：

```python
# environment/metrics.py:321-325
def _norm_time(self, value: float) -> float:
    normalized = max(float(value), 0.0) / max(float(config.CLEAN_REWARD_TIME_REF), 1.0)  # /60.0
    return min(normalized, float(getattr(config, "CLEAN_REWARD_TIME_CLIP", float("inf"))))  # min(.., 10.0)
```

clip 生效阈值是 **delay > 600 秒**（60 × 10），不是我原先以为的接近 60 秒。
而且 `_task_incremental_delay`（`metrics.py:298-312`）算的是
**相对最晚父任务完成时刻（入口任务则相对 DAG 到达时刻）的单跳时延**，
不是从 episode 起点累积。单跳 600 秒是极端长尾。
**所以奖励 clip 大概率不生效**，Tier 3 只是为了确认，优先级最低。

---

## Tier 1：直接拷贝现有文件（无需任何计算）

全部来自 `20260806_formal_v1` 与其对应的训练 run，都是已冻结产物。

| # | 文件 | 大小估计 | 用途 |
| --- | --- | --- | --- |
| 1 | `20260806_formal_v1/run_manifest.json` | KB | 校验口径、checkpoint 元数据、`logical_tape_sha256` |
| 2 | `20260806_formal_v1/analysis/` 下的分析输出 JSON | KB | 落实 `technical_pass`、`deterministic_reachability`、`static_temperature_gates`、`closed_loop_guardrails` |
| 3 | **`20260806_formal_v1/closed_loop/episodes.jsonl`** | 1200 行，约 1 MB | **最关键的一个**，见下 |
| 4 | `logs/decision_ppo_bandit/*_stage1_formal_S1-B_seed{42,86,1042}/config.json` | KB × 3 | 确认训练期配置与评估配置一致 |
| 5 | 同上三个 run 的 `updates.jsonl` | 30 行 × 3 | 训练收敛性（`behavior.raw_eft_regret_*`、`approx_kl`、`clip_fraction`、`ratio_std`） |

### 为什么第 3 项是关键

`run_stage1_temperature_followup.py:72` 写入的每行已经包含：

```
avg_uav_queue_length, admitted_incomplete_backlog, arrival_blocked_count,
generated_dag_count, admitted_dag_count, completed_dag_count,
dag_completion_rate, average_dag_flowtime, episode_reward_total,
invalid_assignment_count
```

这已经足够做**一阶负载判读**，不需要任何新计算：

- `avg_uav_queue_length` 相对 `queue_cap=16` → 队列压力
- `arrival_blocked_count / (arrival_blocked_count + generated_dag_count)` →
  active-DAG cap 的节流比例。**这个比例高 = 系统被拥塞节流 = 高负载**，
  这是当前配置下 ρ 的直接代理
- `admitted_incomplete_backlog = generated − completed` → 积压是否在增长
- `dag_completion_rate` 的**跨 episode 标准差** → 替代旧的 σ=0.099，
  这是"配对比较是否必要"的真实依据
- 四个温度下的上述指标对比 → 顺便把"降温改善闭环"这条从推测变成结论

**如果只能给我一个文件，就给这个。**

---

## Tier 2：从现有 static corpus 离线派生（不重跑，只算，回传汇总 JSON）

`static_corpus/records.jsonl` 有 116,783 条，太大不必拷贝。
请 Codex 在服务器上跑一个只读脚本，回传下面的**汇总 JSON**即可。

时间口径已经确认过（你们上一轮验证的）：

```python
incremental_delay_i = max(eft_i - (slot_index + 1) * 5.0, 0.0)
```

### 派生指标规格

按 `checkpoint_sha256` 分组（3 组），每组输出：

```jsonc
{
  "checkpoint_sha256": "...",
  "decision_count": 0,

  // A. 负载：最优候选的延迟分布 —— 回答"系统有多堵"
  "min_legal_incremental_delay": {
    "p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0
  },

  // B. 负载演化 —— 回答"是稳态积压还是无界增长"（最关键）
  //    slot_index 按 20 一档分成 10 档，每档给 min_legal_incremental_delay 的 p50/p90
  "by_slot_bucket": [
    {"slot_lo": 0, "slot_hi": 19, "p50": 0.0, "p90": 0.0,
     "saturation_rate": 0.0, "mean_legal_candidates": 0.0, "decision_count": 0}
    // ... 共 10 档
  ],

  // C. 决策空间 —— 回答"策略到底有多少可选"
  "legal_candidate_count_hist": {"2": 0, "3": 0, "4": 0, "5": 0},

  // D. 饱和与 margin 的交叉（用于复核 D0，可选）
  "saturation_rate_overall": 0.0,
  "saturation_rate_margin20": 0.0
}
```

**判读规则（我会照这个读）：**

| 观测 | 结论 |
| --- | --- |
| B 中 `p50` 随 slot_bucket **走平** | 稳态积压。ρ ≈ 1 但被 active-DAG cap 稳住，工作点可接受 |
| B 中 `p50` 随 slot_bucket **单调上升且不回落** | 队列无界增长，ρ > 1 确立，**必须先修工作点** |
| B 中 `p50` 从第一档就已 ≥ 40 s | 延迟不是积累出来的，是**单次服务本身就慢**，问题在算力/任务规模标定而非到达率 |
| C 中 `legal_candidate_count` 大量为 2 或 3 | 队列常满导致候选被 mask 掉，**策略的决策空间被压缩**，这本身就是过载的强证据 |

C 这一项特别值得看：如果多数决策只剩 2 个合法候选，那"学一个好策略"的空间本来就很小，
这会直接影响论文的可行性判断。

---

## Tier 3：仅在 Tier 1–2 显示极端延迟时才做（需要一次只读 replay）

目的：确认 `CLEAN_REWARD_TIME_CLIP` 到底有没有生效。

做法：在冻结 tape 上跑一次 **T=1.0、单 replicate** 的只读 replay，
对每个进入奖励结算的任务记录 `_task_incremental_delay(task, task_manager)` 的**原始值**，
输出百分位。

```jsonc
{
  "task_incremental_delay_raw": {
    "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0,
    "fraction_over_600s": 0.0   // ← 唯一真正要看的数
  }
}
```

判读：`fraction_over_600s < 0.01` → 奖励 clip 不生效，**"奖励端饱和"这条正式撤回**。

**除非 Tier 2 的 `min_legal_incremental_delay.p99` 已经逼近几百秒，否则不必做这一步。**

---

## 我会用这些数据回答的三个问题

1. **当前工作点是什么** —— 稳态积压 / 无界增长 / 单次服务慢，三选一
2. **`clip 天花板` 该怎么修** —— 若是无界增长，先修负载；若是稳态，直接改归一化
3. **`dag_completion_rate` 的真实跨 episode 方差** —— 决定后续所有对比是否必须配对

---

## 明确不需要的东西

- 不需要 `static_corpus/records.jsonl` 本体（太大，派生汇总即可）
- 不需要重跑 `20260806_formal_v1`
- 不需要任何训练
- 不需要 `runs/phase4_p0_baseline_200slot`（已作废，**不要再拿它做任何推论**）
