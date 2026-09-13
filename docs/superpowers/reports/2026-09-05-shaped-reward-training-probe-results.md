# REPORT: 2026-09-05 shaped-reward training probe

**对应 spec：** `docs/superpowers/specs/2026-09-05-shaped-reward-training-probe.md`（v3）
**Codex 执行裁决：** **FAIL；并带终止边界项的正确性保留意见。** 等待 Claude 独立复核与最终裁决。

## 1. 版本现实（AGENTS.md §3）

- 正式执行目录：`/data2/zrj2025/HyperUAV`
- 服务器实际 `git rev-parse HEAD`：`a9501872a450d937786de1ecb88b969dbbf0c1ab`
- 服务器 `git status --porcelain`：dirty 19 行；完整列表已冻结在正式目录的 `run_version.txt`。
- shaping 实现冻结提交：`3883bdf53c52cd6be835c23613ade1700f43db97`
- 六卡 launcher 冻结提交：`3ea3150a4a4ed963ab6102f32ce9fec634594bc1`
- 正式执行文件的 `git hash-object`：
  - `scripts/train_clean_mainline.py`：`fd59e99982c3754b629b7143dc1f1865f547910a`
  - `scripts/run_shaped_reward_training_probe.py`：`6a28b1b558da24374e5d965243fae90026fec0fa`
  - `marl_models/mappo/clean_dag_progress_shaping.py`：`d1838332437e8bf9c7aee9bc7e06cf3d700d322d`
  - `marl_models/mappo/clean_trainer.py`：`351dccff2eb7257ba061698aec42e1265464086d`
- 正式任务：launcher PID `62156`；2026-09-04 21:46:32 +08:00 启动；6/6 runs completed。
- GPU 映射：seed0 OFF/ON=`0/2`，seed1 OFF/ON=`3/4`，seed2 OFF/ON=`5/6`。
- 先前单卡目录 `formal_20260904_213428` 已停止并标记 `SUPERSEDED`，未混入本结果。

## 2. 实验完整性

- 设计：MLP clean baseline；OFF/ON 各 100 episodes × seeds 0/1/2；每 episode 500 slots；每 run 400 PPO updates。
- 6 个 `run_summary.json` 均为 `status=completed`，所有汇总数值 finite。
- 每个 seed 的 OFF/ON 初始 HGNN、movement actor、offloading actor、critic、optimizer、RNG state 逐值相同。
- 配置的实质差异只有 `dag_progress_potential_shaping=false/true`；run name 与 output path 是产物隔离所需差异。
- 正式 ON 日志逐行满足 `shaped reward = original reward + F`，最大绝对误差 `1.78e-15`。
- 300/300 个 ON episode 的终止记录均满足 `F=0`；未强制清零终止后的 potential。
- 实验前独立 smoke 已确认：gate OFF 的旧日志字段、模型、critic、optimizer 与 RNG 状态逐值复现；truncation 保留正常 shaping；完整 500-step done transition 的 `F=0`。

## 3. Primary metrics

以下均为每个 run **后 20%** 的均值。

| seed | shaped-EV OFF | shaped-EV ON | de-shaped-EV OFF | de-shaped-EV ON | normalized offloading entropy OFF | ON |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00946 | -0.01655 | 0.00946 | 0.43488 | 0.999846 | 0.999980 |
| 1 | 0.01061 | -0.05781 | 0.01061 | 0.42466 | 0.999613 | 0.999920 |
| 2 | 0.00595 | -0.02048 | 0.00595 | 0.44832 | 0.999931 | 0.999954 |
| 3-seed mean | **0.00867** | **-0.03162** | **0.00867** | **0.43595** | **0.999797** | **0.999952** |

### 3.1 Shaped-target critic EV：FAIL

- ON−OFF 配对差为 `[-0.02601, -0.06842, -0.02643]`，3/3 同向变差；均值差 `-0.04029`。
- ON 后半程 EV 为正的 update 比例仅 `[0.490, 0.500, 0.475]`，低于 OFF 的 `[0.645, 0.655, 0.710]`。
- ON 后 20% 的 seed 内标准差为 `[0.243, 0.432, 0.206]`，而 OFF 仅 `[0.031, 0.032, 0.022]`。ON 不但没有“从约 0 持续爬升并保持为正”，反而更负、更不稳定。

### 3.2 De-shaped EV：数值为正，但不能挽救 gate

- ON 后 20% 均值约 `0.436`，3/3 高于 OFF；但它从前 20% 就已约 `[0.531, 0.512, 0.531]`，并非随训练从 0 爬起。
- ON 后 20% 的 seed 内标准差约 `0.81–0.83`，全程最小值约 `-1.47` 至 `-1.25`，波动很大。
- 该诊断由 `V_original=V_shaped+Φ` 与同一 value-bootstrapped GAE target 构造；随机初始化早期即出现约 0.52 EV，说明确定性的 Φ 同时进入 prediction/target 后贡献了大量可解释方差。它不是“critic 已学会原始长期价值”的独立证据。
- spec 要求 shaped-EV 与 de-shaped-EV **同时改善**。shaped-EV 明确失败，因此无论如何不能判 PASS。

### 3.3 Offloading entropy：FAIL

- OFF 与 ON 的 normalized entropy 都长期接近 1；ON 更接近完全均匀，配对差 `[+0.000134, +0.000307, +0.000023]`。
- 这不是 collapse，但也不满足 spec 的“又不长期完全均匀”。100 episodes 内没有形成有辨识度的 offloading policy，不能称为 entropy 稳定性改善。

## 4. Guard system metrics

后 20% episode 均值：

| seed | completion OFF | ON | flowtime OFF | ON | throughput OFF | ON |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7580 | 0.7864 | 619.90 | 547.95 | 0.05430 | 0.06146 |
| 1 | 0.7558 | 0.7668 | 648.00 | 579.41 | 0.05282 | 0.05734 |
| 2 | 0.7674 | 0.7964 | 628.90 | 530.69 | 0.05484 | 0.06328 |
| 3-seed mean | **0.7604** | **0.7832** | **632.27** | **552.68** | **0.05399** | **0.06069** |

配对 ON−OFF 结果 3/3 同向：

- completion：`+0.02280`（约 `+3.0%` relative）；
- average flowtime：`-79.59`（约 `-12.6%`，越低越好）；
- throughput：`+0.006707`（约 `+12.4%` relative）。

这些是有兴趣的方向性信号，但只是 training trajectory、3 seeds、100 episodes，尚无 deterministic evaluation，且受下一节的终止边界项问题污染。因此只能表述为“guard metrics 未恶化且方向改善”，不能作为正式性能结论或覆盖 primary gate 的失败。

## 5. v3 终止约定的数学正确性保留意见

实现严格服从 v3：done transition 令 `F=0`，不清零 Φ，不改基线 bootstrap。这确实避免了约 `-Φ` 的终止负 spike。但 spec 中“整段 discounted return 只差约 `1e-9`”与正式超参数不符：

- `γ=0.99`，`T=500`，所以 `γ^499 = 0.0066368516`，不是接近 `1e-12`；
- 按 300 个 ON episode 终止前的实际 Φ，遗留边界项 `γ^499 Φ(s_499)`：均值 `8.423`，范围 `[6.283, 11.104]`；
- 该量级约等于一次原始 completed-DAG bonus `+8`，且随终止时完成进度变化，是 policy-dependent 的。

所以 v3 实际运行的是“dense PBRS + 一个非忽略的 terminal-progress boundary term”，不再是严格 telescoping 到常数的标准 policy-invariant PBRS。系统指标改善可能部分来自这个有效终端目标变化，不能全部归因于 reward density/critic learnability。此处是对结果解释的限制，不擅自改 terminal/bootstrap，也不补跑新变体。

## 6. 裁决与结论边界

**Codex 裁决：FAIL。不要据此批准正式 1000-episode shaped training。**

理由：

1. Primary gate 1 失败：shaped-EV 3/3 更差且后段为负；
2. Primary gate 2 失败：offloading entropy 仍近乎完全均匀；
3. de-shaped EV 虽高，但早期即高且高度波动，不能独立证明 learned original value；
4. 系统指标虽 3/3 改善，但 short training trajectory 不足以作性能结论，并受约 `+8.4` 的 policy-dependent terminal boundary term 混杂。

本实验支持的最窄结论是：**当前累计 DAG-progress shaping 没有让 critic 对实际 shaped training target 变得可学，也没有让 offloading actor 在 100 episodes 内形成非均匀策略。** 它不支持“reward density 已解决 critic/actor 问题”，也不支持把 observed system improvement 解释为严格 policy-invariant PBRS 的收益。

按 spec stop rule，应停止自动扩展该 shaped-training 配置，交由 Claude 独立复核后决定转向 estimand/state，或先修正并重新冻结 terminal invariance 与 de-shaped-EV 的测量定义。

## 7. 原始产物路径

- 正式根目录：`/data2/zrj2025/uav-results/audits/shaped-reward-training-probe/formal_parallel_20260904_214632`
- `result.json`：`.../result.json`
- 汇总曲线：`.../curves/training_probe_curves.png`
- launcher log：`.../launcher.log`
- 六份训练 log：`.../logs/{off,on}_seed{0,1,2}.log`
- 六组 run config/checkpoint/metrics：`.../runs/{off,on}/seed{0,1,2}/...`
- 版本记录：`.../run_version.txt`

## 8. Codex 自检

- gate OFF 复现旧行为：PASS（pre-change/post-change smoke 旧字段、模型、optimizer、RNG 逐值相同）。
- 单变量与配对初始化：PASS；除 gate 与产物路径/name 外配置一致，三个 seed 初始模型/optimizer/RNG 逐值相同。
- RNG-neutral：PASS；shaping 不抽 RNG，配对初始 RNG state 相同。
- 终止实现符合用户批准的 v3：PASS（300/300 terminal `F=0`，未强制 Φ=0）。
- “discounted boundary 约 1e-9 / 严格目标守恒”：**FAIL**（实际均值 `8.423`）。
- 自动进入下一阶段：否。
