# 离线派生规格：归一化变换选型 + 分层准确率验证

日期：2026-08-07
性质：**纯离线只读分析**。不训练、不重跑、不改任何已有文件、不需要 GPU。
输入：`logs/stage1_temperature_followup/20260807_long_v1_formal/static_corpus/records.jsonl`（237,494 条）
输出：一个汇总 JSON（回传本地即可，corpus 本体不用拷）

---

## 目的

1. **补上一个缺口**：验证新 corpus 的分层准确率是否仍为 ~0.97（未饱和）/ ~0.57（饱和）。
   若是，「选对率下降 1.5pp 完全来自饱和子集占比上升」这个解释闭环。
2. **把 B1 的归一化参数从"拍板"变成"算出来"**：
   用分离度匹配判据，在候选变换里选出 `ref`。

---

## 关键口径（照抄，不要自行推断）

```python
t = (slot_index + 1) * 5.0                      # 已验证过的时间换算
delay_i = max(eft_i - t, 0.0)                   # 逐候选
legal = np.flatnonzero(np.asarray(candidate_mask, dtype=bool))
```

> ⚠️ **corpus 里非法候选的 `eft` 是 `0.0` 而不是 `+inf`。
> 所有取 min / 排序的操作必须先按 `candidate_mask` 过滤。** 这是本项目已经踩过的坑。

```python
d = np.sort(delay[legal])                       # 升序
d1, d2 = d[0], d[1]                             # 最优、次优（要求 len(legal) >= 2）
gap = d2 - d1                                   # 秒
margin20 = (gap >= 20.0)
saturated = bool((delay[legal] >= 40.0).all())  # 全部合法候选都被截断
correct = (deterministic_actor_uav_id == greedy_eft_uav_id)
```

`SATURATION_REF = 40.0` 即当前的 `CLEAN_NORM_AVAIL_TIME_REF`。

---

## Part A —— 分层准确率验证

按 `checkpoint_sha256` 分组，在 `margin20 == True` 的子集上输出：

```jsonc
{
  "margin20_count": 0,
  "saturated_fraction": 0.0,
  "accuracy_overall": 0.0,
  "accuracy_saturated": 0.0,        // 预期 ≈ 0.57
  "accuracy_unsaturated": 0.0,      // 预期 ≈ 0.97
  "accuracy_by_legal_k": {"2":0.0,"3":0.0,"4":0.0,"5":0.0}
}
```

**判读**：若 `accuracy_unsaturated ∈ [0.95, 0.99]` 且 `accuracy_saturated ∈ [0.53, 0.61]`，
则解释闭环；若显著偏离，停下来报告，不要自行解释。

---

## Part B —— 候选变换的分离度

### 参考基准

```
Δ* = 在「未饱和 & margin20」子集上，当前变换 clip(x/40, 0, 1) 的
     Δ_current = φ(d2) − φ(d1) 的 P10
```

这个基准的合理性来自 Part A：actor 在这个子集上已经能达到 96–98%，
所以「达到同等分离度」是一个由数据支撑的充分条件，而不是拍脑袋的阈值。

### 待评估变换

| 组 | 形式 | `ref` 取值 |
| --- | --- | --- |
| 当前（基准） | `clip(x/ref, 0, 1)` | 40 |
| 放大 clip（对照组） | `clip(x/ref, 0, 1)` | 80, 160, 200, 320 |
| **无界保序（主选）** | `x / (x + ref)` | 20, 30, 50, 80, 120 |
| 对数 | `log1p(x/ref)` | 5, 10, 20, 50 |
| 相对编码（**上界参考**，非候选） | `(x − d1) / (d_max − d1 + 1e-9)`，逐决策 | — |

「相对编码」不作为实际候选（它把绝对时延信息丢掉了），只用来给出**可达分离度的上界**，
方便判断其他变换离上界有多远。

### 每个变换输出

```jsonc
{
  "transform": "x/(x+ref)",
  "ref": 50,
  "argmin_preserved_margin20": 1.0,      // 严格单调变换应为 1.0；不是 1.0 说明实现或精度有问题
  "delta_percentiles_unsat_m20": {"p10":0.0,"p50":0.0,"p90":0.0},
  "delta_percentiles_sat_m20":   {"p10":0.0,"p50":0.0,"p90":0.0},
  "fraction_sat_m20_meeting_delta_star": 0.0,   // ← 选型主判据
  "delta_percentiles_all_m20":   {"p10":0.0,"p50":0.0,"p90":0.0}
}
```

### 选型判据（预注册）

> 在「饱和 & margin20」子集上，**`fraction_sat_m20_meeting_delta_star ≥ 0.90`**
> 的变换中，取 `ref` 最小的那个（`ref` 越小则低时延区分辨率越高）。

若无任何候选达到 0.90，**停下来报告**，不要自行放宽阈值——
那意味着需要考虑增加一维相对特征，属于架构决策。

---

## Part C —— 延迟分布补充（用于文档与论文）

按 checkpoint 输出：

```jsonc
{
  "delay_rank1_percentiles": {"p10":0,"p25":0,"p50":0,"p75":0,"p90":0,"p99":0,"max":0},
  "delay_rank2_percentiles": {"p10":0,"p50":0,"p90":0},
  "gap_percentiles":         {"p10":0,"p25":0,"p50":0,"p75":0,"p90":0,"p99":0},
  "legal_candidate_count_hist": {"2":0,"3":0,"4":0,"5":0}
}
```

---

## 交付

- 一个脚本（新增文件，命名建议 `scripts/analyze_normalization_transform_selection.py`）
- 一个汇总 JSON，放 `analysis_inbox/round3/transform_selection.json`
- JSON 顶层带 `source_path`、`source_sha256`、`source_record_count`、
  `saturation_ref`、`delta_star`（按 checkpoint 分别记录）

**不要修改任何已有文件。输出路径 create-only。**
分析完成即停，不要开始实现归一化修改。
