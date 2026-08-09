# 补充派生：量程归一化后的变换重选（附修订后的选型规则）

日期：2026-08-07
性质：**纯离线只读**，对 `analyze_normalization_transform_selection.py` 的最小补充。
不训练、不重跑 rollout、不改任何已有文件。

---

## 为什么要重算

原判据比较的是 `Δ = φ(d2) − φ(d1)` 的**绝对值**，但不同变换的输出量程不同：

- `clip(x/40,0,1)` 与 `x/(x+ref)` 输出在 `[0,1]`
- `log1p(x/5)` 在观测范围内输出到约 `4.43`

于是 `log1p` 的 Δ 天然被放大约 4.4 倍，**跨族比较不是等价的**。
原结果选出 `log1p, ref=5`，但那至少部分是量纲 artifact。

对神经网络而言，第一层是 `Σ wᵢxᵢ`，绝对量纲会被权重吸收；
**真正有意义的是分离度相对于该特征自身动态范围的比值**——那才是权重被校准的尺度。
而且其余 14 维特征都在 `[0,1]`，量程一致才可比。

---

## 重算方法（极小改动）

对每个候选变换 φ 定义

```
φ̃(x) = φ(x) / φ(X_hi)
Δ̃    = Δ / φ(X_hi)          # 逐条 Δ 除以同一个常数即可
```

**`delta_star` 不变**：基准是 `clip(x/40,0,1)`，其 `φ(X_hi) = 1`（因为 `X_hi > 40`），
除以 1 不改变任何值。所以只需要重算候选变换的 Δ 分位数与达标率。

### `X_hi` 取两档，检验结论稳健性

| 档 | 取值 | 依据 |
| --- | --- | --- |
| A | **150** | ≈ 三个 checkpoint 的 `delay_rank1` p99（143–149）。少量样本归一化后 >1，无害 |
| B | **400** | ≈ 观测最大值（355–417），全部落入 `[0,1]` |

**两档下的排序若一致 → 结论对 `X_hi` 不敏感，可直接采用。
若不一致 → 停下来报告，不要自行选一档。**

### 扩充 ref 扫描范围

`x/(x+ref)` 当前扫到 120 时归一化分离度仍在上升，需要看是否会转折：
**追加 ref ∈ {160, 200, 300}**。其余变换族保持原有取值。

---

## 修订后的选型规则（本次重算**之前**预注册）

原规则「达标者中取 ref 最小」在跨族比较下语义不明确，现修订为：

1. **硬性门槛**：`argmin_preserved_margin20 == 1.0`
   → 这一条直接排除整个 `clip` 族（实测 0.171 / 0.794 / 0.971 / 0.989 / 1.000 均未达 1.0，
   仅 ref=320 的 0.9996 接近但未达）
2. **达标条件**：归一化后 `fraction_sat_m20_meeting_delta_star ≥ 0.90`，三个 checkpoint 全部满足
3. **主排序**：在达标者中，取归一化后 `delta_percentiles_sat_m20.p10` **最大**的
   （用 p10 而非 p50，因为 `delta_star` 本身就是 P10，失败集中在最差那一档）
4. **破平**（差异 < 5% 视为并列，按此优先级）：
   a. 输出天然落在 `[0,1)`、无需额外缩放常数
   b. `ref` 更小

**规则变更理由**：原规则的动机是「ref 越小低时延分辨率越高」，
那是单族内的合理代理；跨族且量程归一化之后，直接比较分离度本身更贴合原始意图。
本修订在看到重算结果之前写定。

---

## 输出

追加写入 `analysis_inbox/round3/transform_selection_normalized.json`（create-only）：

```jsonc
{
  "schema": "...",
  "x_hi_variants": [150.0, 400.0],
  "delta_star": { "<ckpt>": 0.0 },              // 与原文件一致，用于交叉核对
  "by_x_hi": {
    "150.0": { "<ckpt>": [ { "transform":"", "ref":0,
                             "phi_at_x_hi":0.0,
                             "argmin_preserved_margin20":0.0,
                             "delta_norm_percentiles_sat_m20": {"p10":0,"p50":0,"p90":0},
                             "delta_norm_percentiles_unsat_m20": {"p10":0,"p50":0,"p90":0},
                             "fraction_sat_m20_meeting_delta_star":0.0 } ] },
    "400.0": { }
  },
  "ranking_stable_across_x_hi": true,
  "qualifying_candidates": [],
  "selected_candidate": {},
  "selection_status": "selected | inconclusive | none_qualified"
}
```

若 `ranking_stable_across_x_hi` 为 `false`，或没有候选达标，
**`selection_status` 置为对应值并停止，不要自行放宽任何阈值。**

---

## 一并确定（不需要新数据）

`CLEAN_NORM_AVAIL_TIME_REF = 40 s` 目前被**三个**特征共用：

| 特征 | 位置 |
| --- | --- |
| `incremental_delay` | pair index 5 |
| `queue_waiting_time` | pair index 2 |
| `available_delta` | dynamic index 4 |

corpus 里只有 `eft`，**推不出后两个**，本次分析无法覆盖它们。

**决定：B1 对这三个特征采用同一变换族、同一 `ref`。** 理由：

- 三者都是「以秒计的等待/时延」，同一时间尺度
- `queue_waiting_time` 是 `incremental_delay` 的分量，用不同 `ref` 会让二者尺度不可比，
  反而破坏网络能利用的结构关系
- 避免引入三个各自独立的魔数

将来若要单独标定，需要一次 corpus 富化（记录原始 pair/dynamic 特征），
**但这不阻塞 B1**。

---

## 结束条件

产出 JSON 并给出 `selected_candidate` 后**立即停止**。
不要开始修改 `_normalize_pair_features` 或任何配置常量。
