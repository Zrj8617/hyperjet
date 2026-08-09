# 启发式基线线：状态与第 3 轮任务书

日期：2026-08-07
**这条线与 B1（时延软归一化）完全独立，两边不要互相引用。**

---

## 0. 这条线要回答的唯一问题

> **这个环境里，存在 greedy-EFT 拿不到的收益吗？**

如果有 → RL 有可赢空间，超图方向成立。
如果没有 → 环境里没有值得学的结构，必须先改场景，再谈方法。

**这是整个项目最高价值的未决问题，而且不需要训练任何模型。**

---

## 1. 已完成（第 1–2 轮，**不要重做，不要重新讨论**）

已新增 4 个文件（全部 untracked，零修改已有文件）：可插拔评估框架 +
`greedy-EFT` / `random` / `HeftUpwardRankOrderPolicy` 三个策略 + smoke。

**已定死的设计决策**（第 1 轮确认过，不要再提方案）：

| 项 | 决定 |
| --- | --- |
| HEFT 适配方式 | `rank_u` 降序决定**时隙内任务处理顺序**（OrderPolicy），UAV 选择仍用 EFT 最小、tie-break 取最小 `uav_id`。**不改 `_ready_sort_key` 本身** |
| `c(t,s)` | 方案 B：`parent.output_data_size_mb * 8 / job.base_upload_bandwidth_mbps * mean_factor`，`mean_factor` = 10 个无序 UAV 对的 `1+(d/100)²` 均值，每 episode 算一次（UAV 全程悬停） |
| sink 回传 | `--heft-include-return` 开关，**ret0 / ret1 两个都跑** |
| 跨 DAG 比 `rank_u` | 保持原始秒数直接比较，并逐决策记录 `rank_u` 与 `base_upload_bandwidth_mbps` |
| 钩子签名 | keyword-only；`SelectPolicy` 额外收 `DecisionContext(evaluation_scenario_seed, slot_index, stable_task_id, decision_order, policy_replicate)`；keyed 哈希的 key 含 `policy_name` |
| 不适用字段 | 保留字段名置 `null`（列对齐，配对分析直接可用） |
| 输出字段 | `policy_name` / `order_policy` / `select_policy` / `heft_include_return` 四个分开记 |
| tape | 放仓库外，`--tape-dir` 传绝对路径。`logical_tape_sha256` 必须等于 `cf0086e047e33931a867ff94104f4615627164d1afb29bbd6b7c9b133bbfacf4` |
| 超边 | `--partition-hyperedges` 默认 off（启发式无编码器读 incidence，纯提速） |

**已核验的三处偏离**（均正确，无需回退）：
`slot_index` 序列是 `range(0,200)`（`env.py:259` 取自增前的值）；
torch 断言改为 `cuda_initialized is False`；
`--partition-hyperedges` 默认 off。

---

## 2. 两个未闭合项（第 3 轮开始前必须处理）

**① 锚点 1（唯一的外部真值，必须闭合）**
需要 `analysis_inbox/corpus_slot0_anchor.jsonl`。服务器上一次性提取：

```python
import json
seen, out = set(), []
for line in open("logs/stage1_temperature_followup/20260806_formal_v1/static_corpus/records.jsonl"):
    r = json.loads(line)
    if r["slot_index"] == 0 and r["decision_order"] == 0:
        k = r["evaluation_scenario_seed"]
        if k not in seen:
            seen.add(k); out.append(r)
open("analysis_inbox/corpus_slot0_anchor.jsonl","w").write(
    "".join(json.dumps(r, sort_keys=True)+"\n" for r in out))
```

去重后 20 行（该位置状态完全由 tape 决定，与 checkpoint / replicate 无关）。
放好后 `--sections E` 一条命令跑完。

**② D8（E_part 惰性验证）** 需要服务器上的 kahypar。

---

## 3. 第 3 轮任务：跑批 + 分析（**实现已完成，本轮不写新策略**）

### 批次

在**全部 20 个场景**、同一条冻结 tape 上跑：

| 策略 | replicate 数 |
| --- | --- |
| `random` | 5（随机策略需要误差棒） |
| `greedy_eft` | 1（确定性） |
| `heft_ret0 + greedy_eft` | 1 |
| `heft_ret1 + greedy_eft` | 1 |

合计 160 个 episode。无 GPU，预计 30 分钟内。

> ⚠️ **CPU 争抢**：B1 的训练 rollout 是纯 Python、很吃 CPU。
> 本批次要么在 B1 训练开始前跑完，要么接受被拖慢。

### 分析

1. **逐场景配对对照**：每个策略的 `completed_dag_count` vs 下表 RL 基准，按场景配对
2. **按场景难度分层**：用 RL 基准值排序分成难/中/易三档，分别看各策略的相对表现
3. **顺序分歧度**：HEFT 的 `rank_u` 降序与 `_ready_sort_key` 原顺序的差异程度
   （Kendall τ 或逆序对比例）。**这条直接回答「决策顺序到底有没有 leverage」**
4. 辅助指标：`average_dag_flowtime`、`admitted_incomplete_backlog`

---

## 4. 对照基准：RL 在 T=1.0 的逐场景 `completed_dag_count` 均值

来源 `20260807_long_v1_formal`，3 seed × 5 replicate，同一条 tape。
**全局均值 108.48。** 原始逐 episode 值在 `analysis_inbox/round2/e2/closed_loop/episodes.jsonl`。

| 场景 | RL | 场景 | RL |
| --- | --- | --- | --- |
| 0 | 127.3 | 10 | 114.9 |
| 1 | 97.1 | 11 | 114.3 |
| 2 | 80.4 | 12 | 123.9 |
| 3 | 132.2 | 13 | 92.1 |
| 4 | 119.1 | 14 | 103.4 |
| 5 | 109.7 | 15 | 135.3 |
| 6 | 102.6 | 16 | 128.6 |
| 7 | 104.1 | 17 | 106.1 |
| 8 | 99.3 | 18 | 110.1 |
| 9 | 62.8 | 19 | 106.3 |

---

## 5. 已有的早期信号（第 2 轮锚点 6，n=5，**不显著，需在 20 场景上确认**）

greedy 在场景 0–4 上的完成数 `[147, 84, 61, 161, 149]`，对上同场景 RL：

| 场景 | greedy | RL | 差 |
| --- | --- | --- | --- |
| 0 | 147 | 127.3 | +19.7 |
| 1 | 84 | 97.1 | **−13.1** |
| 2 | 61 | 80.4 | **−19.4** |
| 3 | 161 | 132.2 | +28.8 |
| 4 | 149 | 119.1 | +29.9 |

**模式**：greedy 在宽松场景（0/3/4）大幅领先，在拥堵场景（1/2，RL 也只有 80–97）反而输 13–24%。

推测：greedy-EFT 短视，高负载下会往当前最优的 UAV 上堆，加剧队列失衡。

**第 3 轮要重点验证这个模式在 20 个场景上是否成立**——它若成立，就是论文里
「为什么需要学习型策略」最直接的论据；若不成立，则 greedy 是硬天花板，
需要重新考虑环境设计。

---

## 6. 交付

放 `analysis_inbox/heuristic_baseline/`：

- 各策略的逐 episode JSONL（字段与 `closed_loop/episodes.jsonl` 同构 + 策略字段）
- 一份分析汇总 JSON：逐场景对照、难度分层结果、顺序分歧度
- 锚点全量结果（期望 15 passed / 1 skipped，D8 在服务器上应可转为 passed）

---

## 7. 禁止事项

- 不训练任何模型、不加载 checkpoint、不用 GPU
- **不修改任何已有文件**（B1 正在改 `config.py` / `environment/assignment.py` /
  `scripts/train_decision_ppo_bandit_gate.py`，不要碰）
- 不改 `_ready_sort_key`
- 不要回头重新讨论第 1 轮已定死的设计决策
- 不涉及 B1、归一化、Stage 2、超图架构

> **B1 的改动不影响本线结果**：启发式用的是 `estimated_finish_time`，
> 在归一化之前算出，不受影响。基线跑一次永久有效，不需要因 B1 重跑。
