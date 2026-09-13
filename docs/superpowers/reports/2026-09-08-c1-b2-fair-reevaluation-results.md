# C1 / B2 / C2 公平重评与峰值 epoch 诊断结果（2026-09-08）

**对应 spec：** `docs/superpowers/specs/2026-09-08-c1-b2-fair-reevaluation.md` 及用户批准的 C2 扩展。
**执行状态：** PASS（Track A、Track B 与 C2 扩展均完成；未触发第 10 节硬停止条件）。
**结论口径：** 正式结论只来自固定外生 tape 的 Track A；Track B 是事后按原生 rollout reward 选峰的辅助诊断。

## 1. 结论先行

### 等预算 validation 选点

| 对比（候选−对照） | ΔJ/offer | 两层 95% CI | 有利 seed | Δ端到端完成率 |
|---|---:|---:|---:|---:|
| C1−B2 | -0.4847 | [-0.6758, -0.1639] | 3/3 | 0.2026 |
| C2−B2 | -0.6061 | [-0.7971, -0.3058] | 3/3 | 0.2540 |
| C2−C1 | -0.1214 | [-0.1549, -0.0840] | 3/3 | 0.0514 |

### 最终 episode 500

| 对比（候选−对照） | ΔJ/offer | 两层 95% CI | 有利 seed | Δ端到端完成率 |
|---|---:|---:|---:|---:|
| C1−B2 | -0.5898 | [-0.6245, -0.5562] | 3/3 | 0.2410 |
| C2−B2 | -0.6048 | [-0.6387, -0.5686] | 3/3 | 0.2472 |
| C2−C1 | -0.0150 | [-0.0342, 0.0027] | 3/3 | 0.0062 |

正式主协议的方向由上表给出。区间将训练 seed 与 test tape 两层不确定性都计入；只有 3 个训练 seed，不能把 150 个 seed×tape 单元当成 150 个独立训练重复。C1 与 B2 同时改变奖励和信用通路，因此任何 C1−B2 差异都不是老师的独立因果效应。C2−B2 才隔离同为 B2 环境奖励时的老师预热，C2−C1 则比较同款老师下撤老师后的 B2 增量奖励与原始奖励。

## 2. Track A：固定外生 tape 正式公平评估

### 2.1 Checkpoint 选择

| 臂 | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| C1 | 250 | 50 | 50 |
| B2 | 200 | 50 | 250 |
| C2 | 50 | 50 | 50 |

### 2.2 等预算 validation 选点 · forced-hover 主协议

| 臂 | J/offer | J/episode | delay | task E | move E | admission | conditional | end-to-end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 1.2386 | 693.21 | 590.16 | 103.05 | 0.00 | 0.6263 | 0.9199 | 0.5815 |
| B2 | 1.7234 | 962.89 | 890.19 | 72.71 | 0.00 | 0.4431 | 0.8377 | 0.3788 |
| C2 | 1.1173 | 625.29 | 515.03 | 110.27 | 0.00 | 0.6735 | 0.9320 | 0.6328 |

### 2.2 等预算 validation 选点 · deterministic joint-policy 次协议

| 臂 | J/offer | J/episode | delay | task E | move E | admission | conditional | end-to-end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 1.0075 | 563.45 | 443.22 | 119.28 | 0.95 | 0.7272 | 0.9538 | 0.6956 |
| B2 | 1.7234 | 962.89 | 890.19 | 72.71 | 0.00 | 0.4431 | 0.8377 | 0.3788 |
| C2 | 1.2020 | 671.76 | 449.26 | 119.60 | 102.89 | 0.7251 | 0.9543 | 0.6925 |

### 2.2 最终 episode 500 · forced-hover 主协议

| 臂 | J/offer | J/episode | delay | task E | move E | admission | conditional | end-to-end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 1.2859 | 719.34 | 619.20 | 100.14 | 0.00 | 0.6085 | 0.9140 | 0.5615 |
| B2 | 1.8757 | 1047.60 | 982.44 | 65.16 | 0.00 | 0.3903 | 0.8174 | 0.3205 |
| C2 | 1.2709 | 711.15 | 610.14 | 101.01 | 0.00 | 0.6145 | 0.9148 | 0.5677 |

### 2.2 最终 episode 500 · deterministic joint-policy 次协议

| 臂 | J/offer | J/episode | delay | task E | move E | admission | conditional | end-to-end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 1.5206 | 849.82 | 668.40 | 97.26 | 84.16 | 0.5839 | 0.9010 | 0.5355 |
| B2 | 1.8832 | 1051.91 | 986.05 | 65.38 | 0.48 | 0.3899 | 0.8177 | 0.3200 |
| C2 | 1.1941 | 667.88 | 560.37 | 106.60 | 0.91 | 0.6473 | 0.9296 | 0.6050 |

### 2.3 可复核性

- validation 文件：90；test 文件：36；逐 tape test 行：1800。
- 全部策略共享同一套 50 条 test tape，offer ID 逐值一致；每个 tape ID 只解析到一个预生成 tape 路径。
- `J_episode` 分量恒等式与 `J_per_offer` 恒等式最大绝对误差：2.047e-13。
- 训练奖励未参与跨臂正式排名。

## 3. Track B：多峰值 PPO rollout/update epoch（辅助诊断）

| 臂 | seed | 5 个 update | C_orig/epoch | C_orig/slot | C_sec incremental/epoch | C_sec incremental/slot | C_sec backlog/epoch |
|---|---:|---|---:|---:|---:|---:|---:|
| C1 | 0 | 479, 646, 762, 890, 990 | 113.80 | 0.8891 | 37811.39 | 295.40 | 20698.73 |
| C1 | 1 | 475, 582, 719, 838, 966 | 98.40 | 0.7688 | 37433.22 | 292.45 | 22301.83 |
| C1 | 2 | 343, 499, 650, 763, 899 | 83.64 | 0.6534 | 33313.84 | 260.26 | 20410.20 |
| B2 | 0 | 408, 516, 636, 740, 888 | 252.48 | 2.1765 | 49585.38 | 427.46 | 20273.76 |
| B2 | 1 | 412, 544, 728, 848, 968 | 216.75 | 1.8686 | 44625.28 | 384.70 | 17474.30 |
| B2 | 2 | 404, 528, 676, 844, 968 | 232.56 | 2.0049 | 46592.44 | 401.66 | 19742.10 |

三 seed 先各自对 5 点平均，再汇总：

| 指标 | C1 | B2 | C1−B2 |
|---|---:|---:|---:|
| C_orig_epoch | 98.61 | 233.93 | -135.32 |
| C_orig_epoch_per_slot | 0.77 | 2.02 | -1.25 |
| C_sec_epoch_incremental | 36186.15 | 46934.37 | -10748.22 |
| C_sec_epoch_incremental_per_slot | 282.70 | 404.61 | -121.90 |
| C_sec_epoch_backlog_area | 21136.92 | 19163.39 | 1973.53 |
| C_sec_epoch_backlog_area_per_slot | 165.13 | 165.20 | -0.07 |

峰值结果有明确选择偏差，也来自各策略访问到的不同训练状态，只能解释训练轨迹，不能替代固定 tape 测试。`C_sec_epoch_incremental` 使用本 epoch 首次结算任务的可精确归属增量时延；`C_sec_epoch_backlog_area` 使用探针起点到终点的 censored backlog-area 边界。没有把完整 DAG flowtime 归到最后一个 slot，也没有用 episode reward 除法近似。

## 4. 版本现实（AGENTS.md §3）

- 服务器实际 HEAD：`644227bb90cd5eb87b89e8cafb66af305f56bd4d`；Track A manifest 记录 dirty 29 行，完整列表见机器 JSON。
- Track A 脚本 git object：orchestrator `5b72029f361bda0180d5fcd334631e098b63f7d7`；evaluator `bdbb622c53d50da992beedce67e2e6e01ea67d74`；tape module `bc90f146a5d0a46bffa01f1a8427a1945f21d99e`；environment `458ce59b56edf679ac9e58d72cced2f7d18de7a1`。
- Track B B2 replay 脚本 git object：launcher `fbace8f1ed7190df513cfcd693a0209f2893126d`；runner `b6b20a465a70ae1b640d7f125f9d9b5cf2d566d8`；trainer `bf56e08984ae4c3fc4d254e9407a0b4bd79f2da1`；metrics `6220dc6c0bf01bae84981e97d79d6aae6e09d7cd`。
- Track B 修正 C1 replay 脚本 git object：launcher `0ad99d36be53ea228c9fde1e55eceb5c52363861`；runner `9cc508f6caf8c0cfd9f5dc43672695e60a05c6f3`；trainer `a9db978f4d2d0e4bea8aa0276f380cbd162eb6c6`；metrics `6220dc6c0bf01bae84981e97d79d6aae6e09d7cd`。
- C1/B2/C2 每个 run 的实际参数、源 result、train_dir、checkpoint 与源版本均逐 seed 保存在机器 JSON；没有根据命名猜路径。

## 5. 结论边界

- 正式胜负只按固定 tape 上的实际时延、任务能耗、移动能耗统一口径判断；不直接比较 C1/B2/C2 的训练 reward。
- completed-DAG flowtime 仅是辅助指标；主成本把未接纳或未完成 offer 按 horizon 截尾纳入。
- C2 的解释范围严格限定为已批准的两个对比，不新增奖励臂，不进入 HGNN vs MLP。
- 机器 JSON 保存所有 1800 条精简逐 tape 数值行及其服务器原始 JSON 路径；峰值候选、剔除原因、15×2 个峰值明细和原始 JSONL 路径也全部可追溯。

## 6. 原始产物

- Track A manifest：`/data2/zrj2025/uav-results/audits/20260908_C1_B2_C2_fair_eval_manifest.json`
- Track B 原 B2 launch manifest：`/data2/zrj2025/uav-results/audits/20260908_C1_B2_peak_epoch_launch_manifest.json`
- Track B 修正 C1 launch manifest：`/data2/zrj2025/uav-results/audits/20260909_C1_peak_epoch_diag_replay_fixed_launch_manifest.json`
- 原始 validation/test JSON、训练 JSONL、诊断 replay JSONL 的逐文件路径见机器可读结果。
