# 六臂场景重标定第一阶段门禁结果（2026-09-15）

**结论：HARD STOP / FAIL。** 六臂各 5 episode smoke 均完成，G1–G8 的实现/夹具门禁与 G9 的 flag、C1/C2 回归、rollout 形状检查均通过；但 G9 的同轨迹 B1/B2 账本闭合最大相对误差为 **0.48917915414021346**，大于 spec 要求的 `1e-6`。已触发 spec §6.3，未重跑、未近似降级、未启动任何正式训练或正式评测。

## 1. 服务器版本现实

- 执行副本：`/data2/zrj2025/HyperUAV-gating-20260915`
- 实际 HEAD：`e7056dada1b0c85057b12967a0525de2a814027c`
- `git status --porcelain`：0 行
- 静态检查：10 个改动文件 `py_compile` PASS；`git diff --check` PASS
- 执行设备：单卡 `CUDA_VISIBLE_DEVICES=0`；未占用多卡

改动脚本 git object：

| 文件 | git object |
|---|---|
| `environment/reward_redesign.py` | `3a2c63b95524244bdbdc96adfc175e6673b6af7b` |
| `scripts/train_clean_mainline.py` | `4c77c8e5d470cbe54fa6e3d0ceb27b5076b7654c` |
| `scripts/run_reward_redesign_arm.py` | `f2a6f0db4f04859ac7016e651b7c49e181a2daf6` |
| `scripts/orchestrate_fair_eval.py` | `f803a033fcf8063fe3cf19858d0e04321c30d04f` |
| `scripts/launch_six_arm_realign_production.py` | `db9de5fd845cabfb0f4520729231e96a357dbf08` |
| `scripts/run_fair_eval_batch.py` | `21d7a332958ae873dd2dfc0922b13ed828fd74e9` |
| `scripts/summarize_c1_b2_c2_fair_reevaluation.py` | `1fe2496a4cd2df54aa9b2dfba9f0a99295cf9b5a` |
| `scripts/export_fair_eval_tensorboard.py` | `fd7dba11f4d01b423f974dbf039b28bd1749a2fc` |
| `scripts/export_unified_train_tensorboard.py` | `515daabd2bc44ace9bddf0346a003cfdfcfc0c48` |
| `scripts/smoke_six_arm_realign_gating.py` | `b9377b6b728f179c50e180a7b4703ef41d4099b6` |

## 2. §5 resolved flag 实测表

以下值来自六个实际 server smoke 的 `result.json.resolved_flags`，不是从代码推断。

| 臂 | `offloading_eft_advantage` | `movement_position_advantage` | `forecast_enabled` | `offloading_forecast_advantage` | `teacher_anneal_total_updates` |
|---|:-:|:-:|:-:|:-:|---:|
| B1 | False | False | False | False | 0 |
| B2 | False | False | True | False | 0 |
| C2A | True | False | True | False | 2000 |
| C2B | False | True | True | False | 2000 |
| C2 | True | True | True | False | 2000 |
| C1 | True | True | False | False | 2000 |

结果：6×5 个格子全部匹配 spec。C1/C2 的五项值与修改前实际解析基线一致。

## 3. G1–G9

| Gate | 状态 | 实测结果 |
|---|---|---|
| G1 | PASS | C2A/C2B 均完成 5 episode；两个教师可独立开关，resolved flag 表全匹配。 |
| G2 | PASS | 六臂均产生 20 个 PPO update；每 episode 4 个 update；所有 rollout 均为 125 slot；500 episode 配置解析为 2000 update。 |
| G3 | PASS | 启动/编排入口要求显式 `--arms` 与 `--seeds`；缺参实测非零退出；批准矩阵为 6×3、seed 5/86/617。未启动该正式矩阵。 |
| G4 | PASS（实现门禁） | 编排选择协议固定为 joint；同一 locked checkpoint 用于 joint/forced_hover test；字段已改为 `joint_validation_J_per_offer_mean`。本轮按范围未生成正式 manifest。 |
| G5 | PASS（代码路径夹具） | 候选固定为 320/360/400/450/500；服务器临时夹具实际创建并读取五个候选路径。正式 checkpoint 尚未生成，本轮禁止正式训练。 |
| G6 | PASS（实现门禁） | `hover_action_ratio` 已提升到评测行顶层并进入 `TEST_METRICS`；汇总含 joint test 两层 CI、episode 300–500 训练值及 forced_hover=1 自检。 |
| G7 | PASS（数值夹具） | 2×2 数值夹具实测：移动老师平均主效应 4.0、卸载老师平均主效应 3.0、交互项 2.0；条件效应与两组规定措辞均已编码。 |
| G8 | PASS（数值夹具） | `--lambda-move-sweep` 默认实际解析并输出 0.10/0.04/0.0 三档及排名。 |
| G9 | **FAIL** | flag 表 PASS；C1/C2 回归 PASS；六臂 5 episode PASS；B1/B2 同轨迹账本闭合最大相对误差 **0.48917915414021346**，未达 `<1e-6`。 |

## 4. 六臂 smoke

| 臂 | 训练墙钟（秒） | JSONL 行数 | terminal episode | rollout slot |
|---|---:|---:|---:|---:|
| B1 | 185.829 | 20 | 5 | 125 |
| B2 | 215.361 | 20 | 5 | 125 |
| C2A | 213.406 | 20 | 5 | 125 |
| C2B | 208.585 | 20 | 5 | 125 |
| C2 | 202.535 | 20 | 5 | 125 |
| C1 | 175.886 | 20 | 5 | 125 |

全部单臂远低于 30 分钟硬停止线。每臂原始结果：

- `/data2/zrj2025/uav-results/audits/20260915_smoke_<ARM>_seed5/result.json`
- `/data2/zrj2025/uav-results/audits/20260915_smoke_<ARM>_seed5.log`
- 每臂具体 `train_metrics.jsonl` 路径见机器可读报告。

总 smoke 入口原计划写入 `/data2/zrj2025/uav-results/audits/20260915_six_arm_realign_gating_smoke.json`，但硬停止断言在写总 JSON 前触发，因此该服务器总文件不存在；异常原文为：

`AssertionError: hard stop: B1/B2 ledger closure error 0.48917915414021346`

## 5. 停止边界

- 触发条件：spec §6.3，账本闭合相对误差 `>=1e-6`。
- 未触发：flag 不符、C1/C2 回归变化、修改冻结决策、修改额外 environment 文件、单臂超 30 分钟、旧结果覆盖风险。
- 没有修改 §3 决定，没有修改 `environment/` 下其它文件，没有清理任何旧工作树或旧结果。
- 按硬停止要求，不对 0.489 的原因作修补性假设，不继续正式训练。下一步需用户/Claude 审核本次 diff 与账本定义后另行批准。
