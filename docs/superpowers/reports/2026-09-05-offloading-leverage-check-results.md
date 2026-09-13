# REPORT: 2026-09-05 offloading leverage check

**对应 spec：** `docs/superpowers/specs/2026-09-05-offloading-leverage-check.md`
**Codex 执行裁决：** **LARGE LEVERAGE / PASS**（等待 Claude 独立复核与最终裁决）。

## 1. 版本现实（AGENTS.md §3）

- 正式执行目录：`/data2/zrj2025/HyperUAV`
- 服务器实际 `git rev-parse HEAD`：`a9501872a450d937786de1ecb88b969dbbf0c1ab`
- 服务器 `git status --porcelain`：dirty 21 行；完整列表已冻结在正式目录 `run_version.txt`。
- 本次纯评估冻结提交：`f8258e3`（`Add deterministic offloading leverage audit`）。
- 正式执行文件 `git hash-object`：
  - `scripts/run_clean_policy_baseline.py`：`a06bb8b7605425c183c2df75258647744a580629`
  - `scripts/run_offloading_leverage_check.py`：`9f769e66e817c8a8638ec0e9cc7a5f56313c6226`
- 后台 launcher PID：`225933`；正式完成 `60/60` cells。

## 2. 冻结设计与工程门禁

- 复用 harness：`scripts/run_clean_policy_baseline.py`。该 harness 直接复用 `environment.assignment.build_offloading_candidate_components()` 给出的合法 mask、候选 UAV 和 `estimated_finish_time`，没有另造 EFT 估计器或环境。
- 三档 offloading：
  - `random`：独立 policy PRNG 在合法候选索引上均匀抽样；不消耗 environment/global RNG；
  - `eft_greedy`：合法候选中 `estimated_finish_time` 最小，UAV ID 作稳定 tie-break；
  - `eft_worst`：合法候选中 `estimated_finish_time` 最大，UAV ID 作稳定 tie-break。
- 移动策略：**forced hover**。三档每步均提交相同空 movement map；60/60 cells 的 `uav_movement_energy_total=0`。
- 20 个配对环境 seeds：`0..19`；每档每 seed 1 个独立完整 500-slot episode，共 60 cells。
- deterministic smoke：同一 5-seed × 3-arm smoke 重跑后 aggregates/comparisons 逐值一致。
- 正式 protocol check：三档 seed 集相同；冻结配置字段 0 mismatch；全部结果 finite。
- 无训练：harness 不构造训练 model/optimizer，`optimizer_step_count=0`、`model_parameter_tensor_count=0`。因此不存在参数更新；“逐 tensor 不变”在此 no-model harness 中为空集成立。
- CRN 范围：同 seed、同环境初始化与初始 RNG 流；random 使用独立 RNG，不污染环境 RNG。不同 policy 改变完成/准入状态后，closed-loop 条件随机调用仍可能分叉；本次不是 Scheme-B2 式 strict semantic-tape replay。

## 3. 三档原始均值

20 seeds 的均值；括号内为 seed 间 sample std。completion/throughput 越高越好，其余越低越好。

| policy | completion rate | avg DAG flowtime | throughput | energy/completed DAG | avg queue length | load balance CV |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.7702 (0.0802) | 624.21 (206.08) | 0.05858 (0.01968) | 204.33 (23.19) | 13.994 (1.399) | 0.2872 (0.0902) |
| eft_greedy | **0.8413 (0.0962)** | **331.66 (193.87)** | **0.09714 (0.04273)** | **175.75 (20.08)** | **11.387 (3.682)** | **0.1753 (0.0747)** |
| eft_worst | 0.7578 (0.0792) | 603.03 (187.79) | 0.05806 (0.01790) | 200.92 (27.24) | 14.056 (0.964) | 0.3093 (0.0949) |

`load_balance` 是各 UAV completed workload 的变异系数，越低表示越均衡。

## 4. 指定比较一：eft_greedy vs random

“相对改善”统一按 beneficial direction 报告：higher-is-better 指标为 `(greedy-random)/random`；lower-is-better 指标为 `(random-greedy)/random`。

| metric | absolute delta (greedy-random) | relative improvement | favorable seeds | paired raw-delta 95% CI |
|---|---:|---:|---:|---:|
| DAG completion rate | +0.07107 | **+9.23%** | 19/20 | [+0.04361, +0.09852] |
| avg DAG flowtime | -292.55 | **+46.87%** | 20/20 | [-353.05, -232.04] |
| throughput | +0.03856 | **+65.82%** | 20/20 | [+0.02441, +0.05271] |
| energy/completed DAG | -28.59 | +13.99% | 20/20 | [-38.26, -18.91] |
| avg queue length | -2.607 | +18.63% | 18/20 | [-3.994, -1.220] |
| load balance CV | -0.11187 | +38.96% | 18/20 | [-0.16498, -0.05876] |

预注册 large-leverage 门槛为“flowtime 改善 ≥15% **或** completion 相对改善 ≥10%”。本结果 flowtime 改善 `46.87%`，明确越过门槛；completion `9.23%` 虽略低于其单独门槛，但方向在 19/20 seeds 一致；throughput 在 20/20 seeds 改善。

## 5. 指定比较二：best(greedy) vs worst 跨度

| metric | absolute span (greedy-worst) | relative beneficial span | favorable seeds |
|---|---:|---:|---:|
| DAG completion rate | +0.08342 | **11.01%** | 18/20 |
| avg DAG flowtime | -271.36 | **45.00%** | 20/20 |
| throughput | +0.03908 | **67.31%** | 20/20 |
| energy/completed DAG | -25.17 | 12.53% | 20/20 |
| avg queue length | -2.669 | 18.99% | 18/20 |
| load balance CV | -0.13403 | 43.33% | 18/20 |

`eft_worst` 是逐决策局部 EFT 最大规则，不是系统级 adversarial oracle；它不保证在每个汇总指标上都严格差于 random。实际 random flowtime `624.21` 反而略高于 worst 的 `603.03`，同时 random completion/throughput 略好。completed-only flowtime 还受“哪些 DAG 能完成”的选择/censoring 影响，因此 leverage 主判断应联合看 completion 与 throughput，而不是只看 completed-DAG flowtime。

## 6. Workload feedback 补充

策略完成得更快会使 UE 更早解除 active-DAG cap，从而准入更多工作；正式数据延续了 charter 已确认的 policy-dependent workload feedback：

| policy | generated DAG | completed DAG | accepted task assignments | skipped/no-candidate |
|---|---:|---:|---:|---:|
| random | 186.15 | 146.45 | 1118.95 | 4574.20 |
| eft_greedy | **279.25** | **242.85** | **1718.00** | 3994.80 |
| eft_worst | 187.85 | 145.15 | 1134.45 | **3559.60** |

greedy 在承受更高 admitted workload 的同时，仍把 completion rate、绝对 completed count 与 throughput 全部提高；因此 leverage 不是仅由较轻 workload 或单一比率分母造成。

## 7. 裁决与结论边界

**Codex 裁决：LARGE LEVERAGE / PASS。**

本检查支持：

- 在 forced-hover clean 场景中，offloading policy 对系统级结果有很大 leverage；
- EFT greedy 相对近均匀 random 可大幅改善 flowtime、throughput、energy、queue 与 workload balance；
- “单决策 long-horizon credit 低 SNR”不等于“整条 offloading policy 无关紧要”。两组证据并不矛盾；大量局部决策的系统级累积效应很强。

本检查不支持：

- EFT 就是论文最终策略或可直接作为 actor teacher；
- 单个 EFT action 的长期 counterfactual ranking 必然可靠；
- learned movement 下效应量完全相同；本次为隔离 offloading 而固定 hover；
- 进入 HGNN vs MLP 最终结论；
- strict semantic CRN 下的精确 causal effect。本次是 deterministic paired common-seed closed-loop evaluation。

按预注册判读，论文“卸载决策具有实质系统 leverage”的前提成立。结合 Phase 4A 的 per-decision credit 低 SNR，后续若获用户批准，应优先考虑 **DAG-local / 聚合 estimand**，而不是继续构造固定 horizon 全局-return 的 per-decision target。本报告本身不推进阶段、不修改主线。

## 8. 原始产物路径

- 正式根目录：`/data2/zrj2025/uav-results/audits/offloading-leverage-check/formal_20260904_230441`
- `result.json`：`.../result.json`
- launcher log：`.../launcher.log`
- 60 份 cell log：`.../logs/{random,eft_greedy,eft_worst}_seed{0..19}.log`
- 60 份 episode/config/summary：`.../runs/{random,eft_greedy,eft_worst}/seed{0..19}/...`
- 版本记录：`.../run_version.txt`

## 9. Codex 自检

- 现成环境/harness 复用：PASS。
- EFT 使用现成 `estimated_finish_time`：PASS。
- 三档移动完全一致：PASS（forced hover；movement energy 0）。
- ≥5 paired seeds：PASS（20 seeds）。
- deterministic repeat：PASS。
- same-seed CRN + random RNG-neutral：PASS；非 strict semantic replay，已明确边界。
- optimizer step=0 / 无模型参数更新：PASS。
- 版本 HEAD + dirty + script hash-object：PASS。
- 自动进入下一阶段：否。
