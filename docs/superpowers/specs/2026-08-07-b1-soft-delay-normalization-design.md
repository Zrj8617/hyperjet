# B1 设计：时延特征软归一化（消除 clip 天花板）

日期：2026-08-07
前置门禁：E1/E2/E3 判定 `clip_ceiling_confirmed`（三 seed margin20 选对率 0.635–0.651 < 0.75）
选型依据：`analysis_inbox/round3/transform_selection{,_normalized}.json`
性质：**actor 输入变更 + 重训**。需在执行前获得明确批准。

---

## 1. 目的与预注册假设

**假设**：`incremental_delay` 是 EFT 排序的精确充分统计量，
当前 `clip(x/40, 0, 1)` 在约 70% 的决策上把它截成常数，构成排序天花板。
换成严格单调、永不截断的变换后，margin≥20 选对率应从 0.64 升到 ≥0.90。

**支撑证据**（已确立，不再重复验证）：

| 事实 | 数值 |
| --- | --- |
| 未饱和子集准确率 | 0.974 – 0.986 |
| 饱和子集准确率 | 0.568 – 0.583 |
| margin20 上 argmin 保持率（当前变换） | **0.1706** |
| 训练强度提高 33× 后选对率 | 无改善（0.659 → 0.643，变化完全由饱和占比 0.791→0.834 解释） |

---

## 2. 变更内容（精确到代码）

### 2.1 `config.py` —— 新增常量，**不改动 `CLEAN_NORM_AVAIL_TIME_REF`**

```python
# 软时延归一化参考值（秒）。用于 x/(x+ref)，严格单调、永不截断。
# 取值依据：实测 min_legal_incremental_delay 的 p99 ≈ 143–149（20260807_long_v1_formal）
CLEAN_NORM_DELAY_SOFT_REF: float = 160.0
```

> ⚠️ **必须新增常量而不是修改旧常量。** `CLEAN_NORM_AVAIL_TIME_REF` 还被
> `marl_models/mappo/clean_movement_actor.py:208` 与 `clean_ppo.py:331,390` 使用（Stage 2 路径）。
> 复用会静默改变 Stage 2 行为。

### 2.2 `environment/assignment.py` —— 新增辅助函数

```python
def _soft_delay_norm(value: float, ref: float) -> float:
    """严格单调、无界不截断的时延归一化，输出落在 [0, 1)。"""
    x = max(float(value), 0.0)
    r = max(float(ref), 1e-9)
    return float(x / (x + r))
```

### 2.3 `_normalize_pair_features`（`assignment.py:497`）

pair 特征第 **2**（`queue_waiting_time`）和第 **5**（`incremental_delay`）改用软归一化，
其余六维（transfer / compute / return 时间与三项能耗）**保持原有 clip 不变**。

```python
raw = np.asarray(values, dtype=np.float32)
out = np.clip(raw / scales, 0.0, 1.0)          # scales 中 index 2/5 的值不再生效
soft_ref = float(config.CLEAN_NORM_DELAY_SOFT_REF)
for idx in (2, 5):
    out[idx] = _soft_delay_norm(float(raw[idx]), soft_ref)
return out.astype(np.float32)
```

### 2.4 `_dynamic_uav_features`（`assignment.py:431`）

dynamic 特征第 **4** 维 `available_delta` 改用同一变换、同一 `ref`：

```python
# 原： np.clip(available_delta / max_available_time, 0.0, 1.0)
_soft_delay_norm(available_delta, float(config.CLEAN_NORM_DELAY_SOFT_REF))
```

**三个特征共用同一 `ref`。** 理由：三者都是以秒计的等待/时延；
`queue_waiting_time` 是 `incremental_delay` 的分量，若用不同 `ref` 会使二者尺度不可比，
反而破坏网络可利用的结构关系；同时避免引入三个独立魔数。

### 2.5 `environment/stage1_temperature_diagnostic.py`

按 `long_v1` 的既有模式新增：

```python
FROZEN_CHECKPOINTS_B1: dict[int, tuple[str, str, int]] = {}   # 训练后回填
CHECKPOINT_SETS = {"formal_v1": ..., "long_v1": ..., "b1_v1": FROZEN_CHECKPOINTS_B1}
```

`--checkpoint-set` 的 `choices` 相应增加 `b1_v1`。

### 2.6 新增冒烟测试 `scripts/smoke_clean_soft_delay_normalization.py`

至少覆盖：

- 严格单调：`x1 < x2 ⟹ φ(x1) < φ(x2)`（含 `x` 跨越旧阈值 40 的用例）
- 值域：`0 ≤ φ(x) < 1`，任何有限输入都不等于 1.0
- **argmin 保持**：合成一组时延，验证 `argmin(φ(x)) == argmin(x)`，包括全部 > 40 的情形
- `ref` 合法性：`CLEAN_NORM_DELAY_SOFT_REF > 0`
- 未改动维度不变：pair 的 0/1/3/4/6/7 维与 dynamic 的 0/1/2/3/5/6 维数值与改动前一致

---

## 3. 附带的免费加速（**仅在 bit 中性验证通过后启用**）

Stage 1 使用 MLP 编码器，而 `CleanIndependentTaskMLP.forward` 第一行即 `del incidence_matrix`
——**完全不读关联矩阵**。但四类超边全程在算，其中 KaHyPar 每 5 时隙 spawn 一次子进程。
按 E1 规模（76,800 slot/seed）计约 **15,360 次子进程往返**，全部作废。

拟在 B1 期间置 `False`：

```
ENABLE_DAG_DEPENDENCY_EDGES
ENABLE_KHOP_DEPENDENCY_HYPEREDGES
ENABLE_ATTRIBUTE_HYPEREDGES
ENABLE_KAHYPAR_PARTITION_HYPEREDGES
```

### 启用前必须通过的中性验证

**评估路径的逐条哈希比对**（比训练侧的检验更严格、更直接）：

1. 取任一已冻结 checkpoint，跑 **1 个 checkpoint × 1 个场景 × 1 个 replicate × T=1.0** 的评估
2. 超边全开 / 全关各一次
3. 比对该 episode 全部 static corpus 记录的 `record_sha256`

**逐条完全一致 → 证明中性，可启用。任何一条不一致 → 放弃该加速，超边保持原状。**

同时做一次计时探针（固定 seed 跑 2 个 update，开关各一次，比墙钟时间），
用于记录实际收益。

---

## 4. 明确不改动的东西

- **不上多环境。** 移植 + 等价性验证约 2–4 天，而 B1 全程约 9 小时，不划算；
  且 B1 是带预注册判据的测量，不应引入未验证变量。**多环境留给 Stage 2 之前。**
- **`--device` 保持 `cuda`。** CPU/CUDA 末位浮点差异会在与 `20260807_long_v1_formal`
  的对照中引入设备混淆项。
- 不改 reward、`DECISION_BANDIT_REGRET_SCALE`、`CLEAN_REWARD_*`
- 不改环境、容量、到达率、`active_dag_cap`、`CLEAN_MAX_QUEUE_PER_UAV`
- 不改网络结构、`hidden_dim`、`task_embedding_dim`、编码器类型（仍为 `mlp`）
- 不改 `_ready_sort_key`、PPO/critic/GAE 任何路径
- 不改 `CLEAN_NORM_PAIR_TIME_REF`（20 s，transfer/compute/return 仍用 clip）
- 不动 `20260806_formal_v1`、`20260807_long_v1_formal`、`runs/`
- **不进入 Stage 2**

### 一个可预见的既有文件冲突

`scripts/smoke_active_dag_queue_cap_semantics.py` 直接导入并断言 `_dynamic_uav_features` 的输出，
第 4 维改动后**很可能失败**。

**处理方式**：改动后跑全部 smoke。任何失败**先报告，不要直接改**——
失败可能是期望值过时（可更新），也可能是语义被破坏（必须停止）。这两者要由人判断。

---

## 5. 训练与评估配置

### 训练（沿用 E1 已验证收敛的配置，少一个变量）

```
--group S1-B --seed {42, 86, 1042}
--updates 300 --ppo-epochs 10 --slots-per-update 256
--lr 3e-4 --clip-ratio 0.2 --max-grad-norm 0.5 --chunk-decisions 64
--task-embedding-dim 64 --hidden-dim 128 --completed-dag-weight 16.0
--checkpoint-updates 0,30,100,200,300
--run-name stage1_b1_softnorm_v1
```

### 评估

回填 `FROZEN_CHECKPOINTS_B1` 后：

```
--phase formal --checkpoint-set b1_v1
--tape-dir <与 20260806/20260807 相同>     # logical_tape_sha256 = cf0086e0…facf4
--temperatures 1.0 0.75 0.5 0.25
--sampling-replicates 0 1 2 3 4
--scenario-indices 0..19
--max-physical-slots 200
```

随后跑 `analyze_stage1_temperature_followup.py`，
以及 `analyze_normalization_transform_selection.py` 的 Part A（分层准确率）。

---

## 6. 预注册闸门

### 主判据

| `deterministic_margin20_accuracy`（三 seed 全部） | 结论 |
| --- | --- |
| **≥ 0.90** | **通过。** clip 天花板消除，Stage 1 收尾 |
| 0.85 – 0.90 | 接近但未达，进入 §7 分支 A |
| < 0.85 | 未达预期，进入 §7 分支 B |

### 分层判据（用**旧口径**分层，保证与既有结果可比）

按「全部合法候选的 `incremental_delay ≥ 40 s`」这一**旧规则**给决策打标签，
仅作为分层标签使用，不参与新特征计算：

| 子集 | 当前 | B1 要求 |
| --- | --- | --- |
| 「曾经饱和」 | 0.568 – 0.583 | **≥ 0.85**（这是修复的直接度量） |
| 「曾经未饱和」 | 0.974 – 0.986 | **≥ 0.96**（防退化，确认新变换没有损害原本正常的区域） |

### 结构判据

- margin20 上 `argmin_preserved == 1.0000`（由构造保证，需实证）
- `technical_pass == True`，`invalid_assignment_count == 0`

### 闭环防退化

T=1.0，与 `20260807_long_v1_formal` 在同一 tape 上按 `(seed, episode, replicate)` 配对：

- `completed_dag_count` 的配对差值 95% bootstrap 置信下界 **> −2%**（当前基准 108.48）
- `average_dag_flowtime` 不显著恶化

### 训练健康度

沿用 E1 口径监控 `normalized_entropy` / `max_action_probability` /
`behavior.margin20_accuracy` / `approx_kl` / `clip_fraction`。
中止条件：连续 5 个 update 出现 `approx_kl > 0.05` 或 `clip_fraction > 0.35`。

---

## 7. 失败分支（预注册，不许事后改）

**分支 A —— 0.85 ≤ acc < 0.90**
`ref` 取值不是主因但仍有余量。跑一轮 `ref ∈ {80, 300}` 的对照（各 1 个 seed 即可），
取最优后再补齐三 seed。**不要改变换族。**

**分支 B —— acc < 0.85**
回到分层数据判断是哪一档没救回来：

- 「曾经饱和」子集仍在 0.6 附近 → 变换族有问题，考虑增加一维**逐决策相对编码**
  （离线分析显示相对编码的分离度上界 0.43，是 `x/(x+160)` 的约 3–5 倍）。
  **这属于 actor 输入维度变更，需要另开设计。**
- 「曾经饱和」升上去但「曾经未饱和」掉下来 → `ref` 过大压缩了低时延分辨率，往小调

**分支 C —— 闭环退化**
立即停止并报告。排序准确率提升却导致吞吐下降，意味着 EFT-greedy 本身不是好目标，
这会直接影响 Stage 1 的整体定位，**不是参数问题**。

---

## 8. 交付物

- 三个 seed 的 `config.json` + `updates.jsonl`
- 评估 run 的 `run_manifest.json` / `analysis/*.json` / `closed_loop/episodes.jsonl`
- 分层准确率 JSON（Part A 口径）
- 超边中性验证结果（哈希比对结论 + 计时探针）
- smoke 全量结果

放 `analysis_inbox/round4/`。**产出后停止，不要进入 Stage 2 或架构改动。**
