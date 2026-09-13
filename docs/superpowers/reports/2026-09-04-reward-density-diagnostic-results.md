# REPORT: 2026-09-04 reward-density diagnostic

**对应 spec：** `docs/superpowers/specs/2026-09-04-reward-density-diagnostic.md`
**Codex 执行判定：** **FAIL（未达到 shaped return 多项核心指标显著上升的通过条件；待 Claude 独立复核）**

## 1. 版本现实（AGENTS.md §3）

### 实际服务器执行副本

- 运行目录：`/data2/zrj2025/HyperUAV`
- `git rev-parse HEAD`：`a9501872a450d937786de1ecb88b969dbbf0c1ab`
- `git status --porcelain`：13 行 dirty；3 个 tracked modified、10 个 untracked diagnostic scripts：
  - `M marl_models/mappo/clean_offloading_decision_q_credit.py`
  - `M scripts/smoke_clean_offloading_decision_q_credit.py`
  - `M scripts/train_clean_mainline.py`
  - `?? scripts/diagnose_boundary_anchored_decision_credit_phase1.py`
  - `?? scripts/diagnose_boundary_critic_phase2a.py`
  - `?? scripts/diagnose_decision_q_same_slot_bootstrap.py`
  - `?? scripts/diagnose_decision_q_v2_ranking_crn.py`
  - `?? scripts/diagnose_phase3_counterfactual_horizon.py`
  - `?? scripts/diagnose_phase4a_multi_alternative_credit.py`
  - `?? scripts/finalize_decision_q_same_slot_bootstrap.py`
  - `?? scripts/finalize_decision_q_sequential_reservation.py`
  - `?? scripts/summarize_decision_q_sequential_reservation.py`
  - `?? scripts/summarize_phase2b_local_credit_ranking.py`
- `git hash-object scripts/diagnose_phase3_counterfactual_horizon.py`：`fb8fce229256d6433ee084fbf6ed223984081f27`
- `git hash-object scripts/diagnose_phase4a_multi_alternative_credit.py`：`d35ef413680f193cd65b0e93687c74af99efccdb`
- 启动：2026-09-04 16:04:38 +08:00；完成：约 17:18 +08:00；PID `3446016`

### 本地编辑源（写 report 时）

- `git rev-parse HEAD`：`e9dbe9628ac42c8d9547dc70796b903e089e1d27`
- `git status --porcelain`：写 report 前 19 行 dirty；未清理或覆盖任何既有改动。
- 两个执行脚本 hash 与服务器运行记录一致。

## 2. 实验口径

- 完整复用 Phase 4A 的 27 decisions、8 semantic CRN roots、每个 decision 的全部 5 个 legal actions、exact pre-decision snapshots、冻结 deterministic suffix policy 与 H={40,60,80}。
- 唯一计分变化：逐 transition 记录 active DAG 的 `(dag_id, completed_task_count, total_task_count)`，并计算：
  - `Phi(s) = 8.0 * sum_active_DAG(completed_tasks / total_tasks)`；
  - `r' = r + 0.99 * Phi(s') - Phi(s)`。
- 同一次 branch rollout 同时保存原 reward/control 与 progress-shaped return；没有第二套环境转移。
- 共 27 × 8 × 5 = 1,080 branches、86,400 physical-slot transitions；每条 branch 均有完整 80 rewards 和 81 个 state-boundary progress/Phi records，H=40/60/80 censor 均为 0。

## 3. Pooled all-legal 结果

括号内为 shaped 相对 control 的绝对百分点变化。

| 指标 | H=40 control → shaped | H=60 control → shaped | H=80 control → shaped |
|---|---:|---:|---:|
| Top-1 vs existing truth | 22.2% (6/27) → 33.3% (9/27), **+11.1pp** | 29.6% (8/27) → 22.2% (6/27), **−7.4pp** | 33.3% (9/27) → 40.7% (11/27), **+7.4pp** |
| Pairwise accuracy vs truth | 54.8% (148/270) → 64.4% (174/270), **+9.6pp** | 66.3% (179/270) → 61.9% (167/270), **−4.4pp** | 64.1% (173/270) → 65.6% (177/270), **+1.5pp** |
| Accuracy on 63 truth-CI-resolved pairs | 58.7% (37/63) → 69.8% (44/63), **+11.1pp** | 74.6% (47/63) → 68.3% (43/63), **−6.3pp** | 71.4% (45/63) → 73.0% (46/63), **+1.6pp** |
| Spearman mean vs truth | 0.104 → 0.315 | 0.352 → 0.263 | 0.359 → 0.363 |
| Credit all-pair CI resolved | 11.9% (32/270) → 15.9% (43/270), **+4.1pp** | 12.6% (34/270) → 15.2% (41/270), **+2.6pp** | 11.5% (31/270) → 11.9% (32/270), **+0.4pp** |
| Actual-relative credit CI resolved | 9.3% (10/108) → 12.0% (13/108), **+2.8pp** | 13.0% (14/108) → 17.6% (19/108), **+4.6pp** | 9.3% (10/108) → 12.0% (13/108), **+2.8pp** |
| Root Top-1 stability | 44.0% → 41.2% | 36.6% → 41.7% | 36.6% → 34.3% |
| Root pairwise stability | 65.9% → 64.4% | 64.1% → 65.5% | 64.7% → 64.2% |

### “约 23% pair 可解析率”的口径澄清

Phase 4A 的 `63/270 = 23.3%` 是 **existing multi-root truth oracle 自身**的 CI-resolved pair 比例。truth reference 在本实验中冻结，因此 control/shaped 都仍是 63/270，它不可能因 return 重计分而变化。

本实验真正可被 shaping 改变的，是 8-root credit-return estimator 的 pair CI-resolved 比例：control 最好 12.6%，shaped 最好 15.9%；实际动作相对其它动作的 credit CI-resolved 比例从 control 最好 13.0% 提到 shaped 最好 17.6%。两者均只是小幅提升，仍有超过 82% 的 actual-relative comparisons 无法排除 0。

## 4. 配对不确定性与一致性

对相同 27 decisions 做 fixed-seed、20,000 次 decision-cluster paired bootstrap；区间为 shaped − control 的 95% percentile CI。该分析是结果审计，不参与任何 rollout RNG。

| 指标 | H=40 差值 [95% CI] | H=60 差值 [95% CI] | H=80 差值 [95% CI] |
|---|---:|---:|---:|
| Top-1 | +11.1pp [−3.7, +25.9] | −7.4pp [−22.2, +7.4] | +7.4pp [−11.1, +25.9] |
| Pairwise accuracy | **+9.6pp [+2.2, +17.4]** | −4.4pp [−9.6, +0.7] | +1.5pp [−4.1, +7.0] |
| Truth-resolved pair accuracy | +11.1pp [−8.2, +27.6] | −6.3pp [−16.1, +1.7] | +1.6pp [0.0, +5.5] |
| Credit pair CI resolved | +4.1pp [−1.9, +10.4] | +2.6pp [−3.7, +9.6] | +0.4pp [−4.4, +4.8] |
| Actual-relative CI resolved | +2.8pp [−4.6, +12.0] | +4.6pp [−6.5, +15.7] | +2.8pp [−1.9, +7.4] |

只有 H=40 的全 pair accuracy 改善给出明确正区间；Top-1 和两类 credit-CI-resolved 指标的区间均跨 0。H=60 的 ranking 指标反向。

跨 seed 也不一致：

- H=40 shaped Top-1：seed 0/1/2 = 44.4% / 44.4% / 11.1%；
- H=60 shaped Top-1：33.3% / 22.2% / 11.1%；
- H=80 shaped Top-1：44.4% / 44.4% / 33.3%。

H=80 的 pooled ranking 数字最高（Top-1 40.7%、pairwise 65.6%、Spearman 0.363），但 root Top-1 stability 从 36.6% 降至 34.3%，且其主要 paired 改善区间跨 0。H=40 是唯一有明确 pairwise improvement 的 horizon；H=60 有最高 actual-relative CI-resolved 17.6%，却同时降低 Top-1 和 pairwise accuracy。因此 shaped 结果没有形成一个由所有主指标共同支持的稳定最优 horizon。

## 5. Reward density 与 potential 恒等式审计

- 原 reward 非零 transition：94.15%；shaped：99.80%。原 reward 本身并非“几乎处处为零”。
- 原 reward 正值 transition：40.02%；shaped：87.66%。progress shaping 确实把正向反馈显著前移并加密。
- shaped 与原 reward 不同的 transition：98.79%。
- 对每条 branch、每个 H 验证：
  `G'_H = G_H + gamma^H * Phi(s_H) - Phi(s_0)`；最大绝对误差 `3.41e-13`。

这说明实现严格符合冻结的 potential-based 公式。也说明本 audit 中 truncated-return ranking 的变化完全来自 horizon 边界项；dense step decomposition 本身不会凭空增加完整 discounted return 的信息。

## 6. 工程门禁

- `training_performed=false`；optimizer steps = 0。
- 全部 legal actions branched；27 decisions × 8 roots × 5 actions 完整。
- Phase 3 actual/first-alternative reference branches 全部复现。
- shaping OFF 的 reward sequences、action returns、decision statistics、pooled/by-seed/by-phase metrics **逐值复现 Phase 4A**。
- 参数逐 tensor 不变。
- H=40/60/80 censor = 0。
- semantic shared checks = 1,724,091；semantic mismatch = 0；unrecognized environment RNG = 0。
- smoke serial/process complete result equal = true；formal 按 Phase 4A 原口径不重复 serial 对照。
- `py_compile`、`git diff --check`：PASS。

## 7. 结论边界与判定

### 支持的结论

1. 该 Phi 确实把正向 reward 变得更 dense，公式与 RNG/参数门禁正确。
2. 它没有使 Phase 4A 的完整 all-legal ranking 指标在各 horizon 上一致、显著上升；H=60 还出现退化。
3. credit CI resolvability 的最好值仅从约 13% 提到 16%–18% 区间，绝大多数 action comparisons 仍是低 SNR。
4. 因此按 mini-spec 的预注册门槛，本假设 **FAIL**：不能把“缺少 DAG progress 正向 dense signal”认定为当前 offloading credit 失败的主因，也没有依据直接进入 shaped training。

### 不支持的过度结论

- 不能由本实验断言所有 reward timing/design 问题都已排除；这里只检验了冻结的 active-DAG completion-progress Phi。
- exact discounted shaped return 会 telescoping；本 audit 主要检验有限 horizon 的 terminal-progress boundary 是否改善 counterfactual ranking。它不是 actor-critic/GAE 训练实验，因此不能单独证明 dense TD feedback 对优化完全无帮助。
- 不能把 H=80 的 40.7% Top-1 当成稳定 oracle：n=27、paired CI 跨 0、root stability 仅 34.3%。

### 下一步边界

依照 spec，停止在本任务：**不接入训练、不改 reward/state/PPO、不启动新 target 实验**。后续由用户与 Claude 在 `state sufficiency` 和 `credit estimand definition` 之间裁决；本结果更支持把注意力转回这两个上游问题，而不是直接批准 shaped training。

## 8. 原始产物路径

- 正式 `result.json`：`/data2/zrj2025/uav-results/audits/boundary-anchored-decision-credit/reward_density_diagnostic/formal/result.json`（约 227 MB）
- 正式 log：`/data2/zrj2025/uav-results/audits/boundary-anchored-decision-credit/reward_density_diagnostic/formal/formal.log`
- 版本记录：`/data2/zrj2025/uav-results/audits/boundary-anchored-decision-credit/reward_density_diagnostic/formal/run_version.txt`
- 运行脚本快照：同一 formal 目录下的 `diagnose_phase3_counterfactual_horizon.py` 与 `diagnose_phase4a_multi_alternative_credit.py`
- smoke：`/data2/zrj2025/uav-results/audits/boundary-anchored-decision-credit/reward_density_diagnostic/smoke/result.json`

## 9. Codex 自检

- gate OFF 复现旧行为：是，formal 全量逐值复现 Phase 4A。
- 单变量：是，只改变 return 的离线计分，并同时保留 control。
- RNG-neutral：是，progress/Phi 只读环境状态，不发起 RNG；semantic gates 全过。
- 参数未变：是，checkpoint modules 前后逐 tensor 相等。
- 未训练、未改主线算法、未提交：是。
