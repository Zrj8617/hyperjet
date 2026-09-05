# REPORT: Stage-1 EFT-anchored offloading credit probe（2026-09-06）

**对应 spec：** `docs/superpowers/specs/2026-09-06-stage1-eft-anchored-credit-probe.md`

**Codex 执行裁决：PASS。** 三个训练 seed 的 offloading entropy 均显著下降；三个最终策略在 forced-hover deterministic evaluation 中均在 completion、flowtime、throughput 三项上超过 `eft_greedy`。等待 Claude 独立复核和用户批准阶段推进。

## 1. 使用的 EFT 机制与接入位置

本实验选择仓库现有的 **historical EFT-regret advantage**，通过 `--offloading-eft-advantage` 开启；没有使用 `clean_eft_auxiliary.py` 的 resampled auxiliary loss。

- 每个实际执行的 offloading action 使用决策时已有的候选 EFT 向量，构造 `-(EFT_chosen - min(EFT_legal)) / scale`，并在 rollout decision batch 内标准化。
- 接入点是 `marl_models/mappo/clean_trainer.py` 的 offloading head PPO action loss。
- **它替换了 offloading head 原来的 slot-level return/GAE advantage。** 它不替换 critic 的环境-return value target，也不替换 movement head 的 return-based advantage。
- reward、state、environment、网络结构、critic、GAE、PPO clipping、entropy coefficient 和 optimizer 均未改动；EFT auxiliary lambda 保持 `0`。
- OFF 组不传该 flag，走现有默认基线；ON 组仅增加该 flag。

## 2. 版本现实（AGENTS.md §3）

- 本次实现 freeze commit：`47b5868`（只包含批准的 spec、runner 和默认关闭的 eval RNG-prelude）。
- 服务器执行目录：`/data2/zrj2025/HyperUAV`
- 服务器实际 `git rev-parse HEAD`：`a9501872a450d937786de1ecb88b969dbbf0c1ab`
- 正式启动时 `git status --porcelain`：dirty **24 行**；完整列表冻结在正式目录 `run_version.txt` 和机器可读结果中。
- 正式执行文件的 `git hash-object`：
  - `scripts/run_stage1_eft_anchored_credit_probe.py`：`0890a252b2b907dc927413b0f9dda224036d0329`
  - `scripts/train_clean_mainline.py`：`fd59e99982c3754b629b7143dc1f1865f547910a`
  - `scripts/eval_clean_mainline.py`：`520672aacf5825f888010ff4e95560e712fba891`
  - `marl_models/mappo/clean_trainer.py`：`351dccff2eb7257ba061698aec42e1265464086d`
  - `scripts/offloading_policy_gate.py`：`ccbca4af49e2f75c662d703f74b9059ef831d116`

## 3. 实际运行参数与完整性

- 训练：OFF/ON × seeds `0,1,2`，每 run `100 episodes × 500 slots`，共 6 runs。
- 每 run 实际完成 `400` 个 PPO updates；`rollout_horizon=128`、`num_envs=1`、`ppo_epochs=1`。
- 实际算法参数：`gamma=0.99`、`gae_lambda=0.95`、`clip_ratio=0.2`、`lr=3e-4`、`entropy_coef=0.01`、`value_coef=0.5`、value-target normalization ON、`value_clip_epsilon=0.2`、`max_grad_norm=0.5`。
- 模型：MLP task encoder，hidden dim `128`，task embedding dim `64`，completed-DAG weight `8`。
- 训练 movement 设置：保持现有 learned movement，不额外冻结；OFF/ON 相同。
- 评估：每个训练 checkpoint 均在环境 seeds `0..19` 上跑一个完整 `500 arrival slots + 0 drain` episode；movement 强制 hover；actor 使用 deterministic masked argmax。共 120 个 learned-policy cells，另加 20 个 EFT-greedy 口径校准 cells。
- 三个 seed 的 OFF/ON 初始 H/GNN encoder、movement actor、offloading actor、critic 共 32 个参数 tensor 逐值相同；optimizer state 与 RNG state 也逐值相同。
- 实际 config 比较除 `offloading_eft_advantage`、run name、output path 外无差异。
- evaluator 在全部 20 个 seed 上逐值复现 leverage check 的 EFT-greedy completion、flowtime、throughput，最大绝对误差为 **0**。

## 4. Offloading entropy 轨迹

以下为三个 seed 在指定 update 的 normalized offloading entropy 均值：

| update | OFF | ON |
|---:|---:|---:|
| 1 | 0.999991 | 0.999991 |
| 20 | 0.999987 | 0.999888 |
| 50 | 0.999974 | 0.999085 |
| 100 | 0.999957 | 0.982726 |
| 200 | 0.999940 | 0.693683 |
| 300 | 0.999895 | 0.598133 |
| 400 | **0.999753** | **0.541189** |

后 20% updates 的 seed 内均值：

| seed | OFF | ON | ON−OFF |
|---:|---:|---:|---:|
| 0 | 0.999846 | 0.600740 | -0.399106 |
| 1 | 0.999613 | 0.544945 | -0.454668 |
| 2 | 0.999931 | 0.590767 | -0.409164 |

结论：entropy 明显离开约 `0.9998` 的均匀策略，且 3/3 seeds 同向；OFF 仍复现近乎均匀的现基线行为。确定性评估中的 ON actor mean normalized entropy 为 `0.545607`，与训练尾段一致。

## 5. Forced-hover deterministic 系统指标

OFF/ON 是 3 个训练 checkpoint × 20 个环境 seed 的 pooled mean；random 和 EFT-greedy 是 2026-09-05 leverage check 的同口径参考值。

| policy | DAG completion | avg DAG flowtime | throughput |
|---|---:|---:|---:|
| random reference | 0.770195 | 624.210 | 0.05858 |
| trained OFF | 0.770934 | 584.451 | 0.06173 |
| EFT-greedy reference | 0.841262 | 331.664 | 0.09714 |
| **trained ON** | **0.891094** | **246.451** | **0.11522** |

### 5.1 ON 相对 OFF

- completion：`+0.120160`，相对 **+15.59%**；60 个配对 cells 中 58 个改善。
- flowtime：`-337.99997`，相对 **-57.83%**；60/60 改善。
- throughput：`+0.053493`，相对 **+86.66%**；60/60 改善。

### 5.2 ON 相对 EFT-greedy

- completion：`+0.049832`，相对 **+5.92%**。
- flowtime：`-85.213`，相对 **-25.69%**。
- throughput：`+0.01808`，相对 **+18.61%**。
- 相对 random→greedy gap 的完成比例分别为：completion `170.1%`、flowtime `129.1%`、throughput `146.9%`。

三个独立训练 checkpoint 的 20-seed 均值全部超过 EFT-greedy：

| model seed | completion | flowtime | throughput | actor↔EFT agreement |
|---:|---:|---:|---:|---:|
| 0 | 0.888037 | 259.126 | 0.11188 | 0.77298 |
| 1 | 0.906442 | 205.238 | 0.12500 | 0.77012 |
| 2 | 0.878803 | 274.989 | 0.10878 | 0.76304 |

按单个 model×environment cell 比较 ON 与相同环境 seed 的 EFT-greedy，completion/throughput 各有 45/60 cells 更好，flowtime 有 42/60 更好。不是每个环境 seed 都支配 greedy，但 3/3 checkpoint 的跨 seed 均值和 pooled mean 均严格超过。

## 6. 与 EFT-greedy 的逐决策一致率

- OFF：`38,845 / 70,623 = 55.00%`
- ON：`91,675 / 119,243 = 76.88%`

ON 明显学到了 EFT 排序方向，但仍有 **23.12%** 的决策与 myopic EFT-greedy 不同。结合系统指标超过 greedy，可以确定最终策略不是逐决策的硬复制。

但这里必须限制解释：本轮使用的是 **MLP**，且唯一直接 action-specific teacher 就是 EFT regret。因此“低于 100% agreement 且系统超过 greedy”尚不能证明模型已经学到 EFT 之外的 DAG/超图结构；它也可能来自函数逼近的平滑、非完全模仿、closed-loop queue/workload feedback 或策略诱导的 workload 差异。HGNN 是否能利用 DAG 结构取得额外增益，必须留给下一阶段的正式 HGNN vs MLP 对照。

## 7. 裁决与阶段边界

**Codex 裁决：PASS。** mini-spec 的两个核心条件均满足：

1. entropy 3/3 seeds 从近均匀显著下降；
2. completion、flowtime、throughput 不仅达到 EFT-greedy，而且三个 checkpoint 的 20-seed 均值都超过 EFT-greedy。

本实验支持以下结论：

- offloading actor、optimizer 和 PPO action-update path 在 action-specific、低噪声的 EFT anchor 下能够稳定学习；
- 当前 Stage-1 的直接瓶颈是 environment-return credit，而不是 actor 网络或 optimizer 完全失效；
- EFT-anchored credit 能取回并超过 leverage check 所量化的 greedy 系统收益。

本实验不支持以下结论：

- EFT anchor 已是论文最终算法；
- learned policy 的每个非 greedy 动作都有可靠的长期因果优势；
- MLP 已学习 DAG/超图结构；
- 自动进入 HGNN vs MLP 最终结论。

按 spec，本结果具备申请结束 Stage 1、进入正式 HGNN vs MLP 对照的数值条件；阶段推进仍需 Claude 独立核对和用户批准。

## 8. 原始与仓库产物

- 服务器正式根目录：`/data2/zrj2025/uav-results/audits/stage1-eft-anchored-credit-probe/formal_20260905_104527`
- 服务器 `result.json`：`.../formal_20260905_104527/result.json`
- launcher log：`.../formal_20260905_104527_launcher.log`
- 完整训练/eval logs：`.../formal_20260905_104527/logs/`
- 版本记录：`.../formal_20260905_104527/run_version.txt`
- 仓库机器可读结果：`docs/superpowers/reports/2026-09-06-stage1-eft-anchored-credit-probe-result.json`
- 曲线：
  - `docs/superpowers/reports/2026-09-06-stage1-eft-anchored-credit-probe-entropy.png`
  - `docs/superpowers/reports/2026-09-06-stage1-eft-anchored-credit-probe-system.png`
  - `docs/superpowers/reports/2026-09-06-stage1-eft-anchored-credit-probe-agreement.png`

## 9. Codex 自检

- 6/6 training runs completed，均为 400 updates：PASS。
- 120/120 learned-policy eval + 20/20 leverage calibration eval completed：PASS。
- gate OFF 为现有默认路径：PASS。
- 单变量 config audit：PASS。
- 初始参数、optimizer、RNG 逐值相同：PASS。
- reward/state/environment/PPO 其余控制不变：PASS。
- forced-hover、seeds `0..19`、500-slot 评估口径与 leverage check 逐值校准：PASS。
- entropy、系统三指标、agreement rate 均已报告：PASS。
- 自动推进阶段：否，等待独立复核与用户批准。
