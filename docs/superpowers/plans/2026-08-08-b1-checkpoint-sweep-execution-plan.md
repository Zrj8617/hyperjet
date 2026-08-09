# 执行方案：B1 训练中途快照的闭环扫描

日期：2026-08-08
性质：**只评估，不训练**。使用 B1 已存在的中途 checkpoint，不产生任何新训练。
目的：判定 B1 闭环退化（108.48 → 92.16）的成因属于「特征改动」还是「朝 EFT 目标训练本身有害」。

---

## 1. 背景（已确立的事实，不需要重新验证）

| 策略 | 闭环 `completed_dag_count` | margin20 排序准确率 |
| --- | ---: | ---: |
| greedy-EFT（手写规则） | 108.65 | 1.000（定义） |
| RL `long_v1`（B1 前） | 108.48 | 0.635–0.651 |
| HEFT | 107.45 | — |
| **RL `b1_softnorm_v1`（B1 后）** | **92.16** | **0.998–0.999** |
| shortest_queue | 55.50 | — |
| random | 48.07 | — |

关键观察：**排序准确率 0.64 与 1.00 的闭环结果几乎相同（108.48 vs 108.65），
而 0.999 那个特定策略反常地掉到 92.16。**

三个解释假设已被逐一证伪：
「变得太像 greedy」（greedy 自己是 108.65）、
「继承 greedy 的拥堵弱点」（实测中档跌最狠 −26.61，拥堵档只 −17.89）、
「pre-B1 退化成负载均衡」（shortest_queue 仅 55.50）。

**本方案不再提出新假设，只做一次能二分的测量。**

---

## 2. 要测什么

B1 训练时按 `--checkpoint-updates 0,30,100,200,300` 保存了快照。
对 **update ∈ {0, 30, 100, 200}** 做闭环评估（300 已有结果），
得到一条「训练进度 → 闭环吞吐」曲线，同时得到每个点的排序准确率。

### 预注册判读（跑之前固定）

| 观测 | 结论 |
| --- | --- |
| update 30 就已经 ≈ 92，之后基本平坦 | **特征改动是原因**。朝 EFT 训练不是主因，问题在 B1 改的那三个特征 |
| 从 ≈108 起步、随训练单调下降到 92 | **朝 EFT 目标训练本身有害**。这直接证伪 Stage 1 的目标设定 |
| 非单调（先升后降 / 中间凸起） | 两种机制叠加，需要后续行为 diff，**停下来报告** |
| update 0（随机初始化）≈ 48–56 | 健全性锚点，与 random 基线一致即说明评估链路正常 |

**这三种结果都有价值，不存在「白跑」的分支。**

---

## 3. 代码改动（两处，都是最小改动）

### 3.1 `scripts/run_stage1_temperature_followup.py` —— 新增 `sweep` 相位

现有 `_validate_controls` 硬性要求 formal 相位使用全部四个冻结温度、
20 个场景、5 个 replicate。扫描不需要温度维度。

```python
parser.add_argument("--phase", choices=("pilot", "formal", "sweep"), required=True)
```

`_validate_controls` 中为 `sweep` 增加一组允许值：

```
temperatures        == (1.0,)              # 只跑 T=1.0
scenario_indices    == tuple(range(20))    # 全部 20 个场景，配对单位不变
sampling_replicates == (0, 1, 2)           # 3 个 replicate（formal 是 5）
max_physical_slots  == 200                 # 不变
```

**formal 与 pilot 的校验逻辑一个字符不许改。**

`FROZEN_TEMPERATURES` 常量不许改；sweep 分支单独使用 `(1.0,)`。

### 3.2 `environment/stage1_temperature_diagnostic.py` —— 新增扫描注册表

现有注册表按 seed 索引，扫描需要 (seed, update) 二元索引。新增：

```python
# 由执行者在训练产物上计算 SHA-256 后填入
# key: (training_seed, completed_update)  value: (relative_path, sha256)
B1_SWEEP_CHECKPOINTS: dict[tuple[int, int], tuple[str, str]] = {}
```

在 `CHECKPOINT_SETS` 中注册为 `"b1_sweep"`。
runner 在 `--checkpoint-set b1_sweep` 时，额外接受 `--checkpoint-update <int>`，
按 `(seed, update)` 取条目，并把 `expected_completed_update` 设为该 update 值。

`load_frozen_checkpoint` 的 SHA-256 校验、路径后缀校验、
`encoder == "mlp"`、`graph_dim == 12` 全部保留。

**`FROZEN_CHECKPOINTS`、`FROZEN_CHECKPOINTS_LONG`、`FROZEN_CHECKPOINTS_B1` 三个既有注册表不许改。**

---

## 4. 运行

对 `update ∈ {0, 30, 100, 200}` 各跑一次：

```
--phase sweep
--checkpoint-set b1_sweep
--checkpoint-update <UPDATE>
--tape-dir <与 20260806/20260807/20260808 相同>   # logical_tape_sha256 = cf0086e0…facf4
--temperatures 1.0
--sampling-replicates 0 1 2
--scenario-indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
--max-physical-slots 200
--device cuda
--output-dir logs/stage1_temperature_followup/20260808_b1_sweep_u<UPDATE>
```

规模：每个 update 点 = 20 场景 × 3 replicate × 3 seed = **180 episode**。
四个点合计 **720 episode**，按 formal 的实测速率（4.2 h / 1200 episode）约 **2.5 小时**。

**保留 static corpus 记录**（`record_static` 在 T=1.0 时本就为真），
因为每个 update 点的排序准确率是本次分析的另一半。

`--device` 必须是 `cuda`，与 formal 各轮保持一致，避免引入设备混淆项。

---

## 5. 分析与交付

对每个 update 点计算（可直接复用 `analyze_stage1_temperature_followup.py` 与
`analyze_normalization_transform_selection.py` 的 Part A）：

- `completed_dag_count` 均值（T=1.0，配对单位 `(seed, episode_index, replicate)`）
- `deterministic_margin20_accuracy`
- `deterministic_greedy_agreement`
- `normalized_entropy` / `max_action_probability`
- `avg_uav_queue_length`、`average_dag_flowtime`
- 按 pre-B1 难度分档（拥堵 7 / 中档 6 / 宽松 7）的 `completed_dag_count`

汇总成一张表：

```jsonc
{
  "schema": "b1_checkpoint_sweep_v1",
  "logical_tape_sha256": "cf0086e0...facf4",
  "points": [
    {"completed_update": 0,   "completed_dag_count": 0.0, "deterministic_margin20_accuracy": 0.0,
     "deterministic_greedy_agreement": 0.0, "normalized_entropy": 0.0,
     "avg_uav_queue_length": 0.0, "average_dag_flowtime": 0.0,
     "by_difficulty_tier": {"congested": 0.0, "middle": 0.0, "loose": 0.0}}
    // 30, 100, 200；300 直接引用 20260808_b1_softnorm_v1_formal_parallel 的结果
  ],
  "verdict": "feature_change | objective_harmful | non_monotonic_needs_diff"
}
```

交付到 `analysis_inbox/round5/`：
汇总 JSON + 四个 run 的 `run_manifest.json` + 四个 `closed_loop/episodes.jsonl`。
**static corpus 本体不要拷**（约数百 MB），只回传派生指标。

---

## 6. 禁止改动

- reward、`DECISION_BANDIT_REGRET_SCALE`、`CLEAN_REWARD_*`
- 环境、容量、到达率、`active_dag_cap`、`CLEAN_MAX_QUEUE_PER_UAV`
- **归一化：`CLEAN_NORM_DELAY_SOFT_REF`、`_soft_delay_norm`、`_normalize_pair_features`、
  `_dynamic_uav_features` 一律不许动**（本方案是诊断，不是修复）
- 网络结构、编码器类型、`_ready_sort_key`、PPO/critic/GAE
- `FROZEN_TEMPERATURES` 常量、formal/pilot 的校验逻辑
- 已有 run 目录、`analysis_inbox/` 下既有文件
- **不重新训练任何模型。不进入 Stage 2。**

---

## 7. 结束条件

产出汇总 JSON 与 `verdict` 后**立即停止**。
不要基于结果自行修改归一化、调整 ref、或启动任何重训。
