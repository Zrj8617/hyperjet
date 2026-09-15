# 场景重标定后的六臂训练与公平评测（2026-09-15）

分支：`param-realign-20260914`（场景参数 commit `0aa78f0`，规模扫描 commit `2cb3e2c`）

## 0. 新窗口先读什么

1. 仓库根 `AGENTS.md`
2. `docs/research/HyperUAV_research_master_roadmap.md`
3. `docs/clean_load_calibration.md` 最后两节（场景重标定 + 规模上限扫描）
4. 本 spec 全文

## 1. 这次要回答什么

在**重标定后的新场景**下，用**完全相同的 PPO 超参、环境和训练预算**，比较奖励设计与教师配置的组合，得到一张可以进论文的主表。

四组对比：

| 对比 | 隔离的变量 |
|---|---|
| C2 − B2 | 教师包的总效应 |
| C2 − C2a | 飞行老师的净效应 |
| C2 − C2b | 卸载老师的净效应 |
| C2 − C1 | 奖励形式的净效应（教师配置相同） |
| B2 − B1 | 记账方式（分期 vs 一次性）的效应 |

**跨臂排名只看统一代价 J/offer，禁止用各臂的训练奖励数值排名。** 奖励公式不同，尺度不同，不是一把尺子。

不在本轮范围：A1/A2（增量式时延已有结论，不重训）、B2D/N0/N0-LOCAL/D1/D2（解析优势线已有确定负面结论）、HGNN vs MLP（下一阶段）。

## 2. 实验臂（6 臂 × 3 seed = 18 run）

seed = **5, 86, 617**，三个 seed 在所有臂间共用。

| 臂 | `--reward-redesign-arm` | 奖励 | `offloading_eft_advantage` | `movement_position_advantage` | `forecast_enabled` |
|---|---|---|:-:|:-:|:-:|
| B1 | `B1` | 真实 flowtime 记账 | ✗ | ✗ | ✗ |
| B2 | `B2` | 增量式三段账 | ✗ | ✗ | ✓ |
| C2a | `C2A` | 同 B2 | **✓** | ✗ | ✓ |
| C2b | `C2B` | 同 B2 | ✗ | **✓** | ✓ |
| C2 | `C2` | 同 B2 | ✓ | ✓ | ✓ |
| C1 | `C1` | 原始奖励 | ✓ | ✓ | ✗ |

两个老师的注入点（`marl_models/mappo/clean_trainer.py`）：

- 卸载老师 `:1403-1416`：`A = (1-w)·GAE + w·per_decision_eft_advantage`
- 飞行老师 `:1296-1309`：`A = (1-w)·GAE + w·per_uav_movement_advantage`
- 共用退火权重 `_teacher_weight()` `:340-351`

## 3. 统一训练参数（任何臂不得偏离）

### 3.1 回合与采样

| 参数 | 值 | 说明 |
|---|---|---|
| `--episodes` | 500 | 默认是 100，必须显式传 |
| `--max-steps-per-episode` | 500 | = `config.EPISODE_LENGTH` |
| `--rollout-horizon` | **125** | **本轮从 128 改为 125**，使 500 整除，所有 rollout 等长 |
| 每 episode 更新次数 | 4 | 500 ÷ 125 |
| 全程更新次数 | 2000 | 500 × 4 |
| 总环境步数 | 250,000 | 500 × 500 |
| `--num-envs` | 1 | reward-redesign 臂强制 |
| `--sampler-backend` | synchronous | reward-redesign 臂强制 |

### 3.2 PPO

| 参数 | 值 | 出处 |
|---|---|---|
| `--lr` | 3e-4 | `train_clean_mainline.py:902` |
| `--gamma` | 0.99 | `:903` |
| `--gae-lambda` | 0.95 | `:910` |
| `--clip-ratio` | 0.2 | `:911` |
| `--entropy-coef` | 0.01 | `:912` |
| `--value-coef` | 0.5 | `:913` |
| `--ppo-epochs` | 1 | `:1045` |
| `--normalize-value-targets` | 开（默认） | `:915` |
| minibatch | 无此参数，整 rollout 全批更新 | — |

> `config.DISCOUNT_FACTOR = 0.96`（`config.py:568`）是**死代码**，全仓库无人读取。实际 γ 来自 `--gamma`，默认 0.99。该行建议后续加废弃注释。

### 3.3 网络

| 参数 | 值 |
|---|---|
| `--task-encoder` | **mlp**（本轮全部 MLP，HGNN 留给下一阶段） |
| `--hidden-dim` | 128 |
| `--task-embedding-dim` | 64 |

### 3.4 教师退火（C1 / C2 / C2a / C2b）

| 参数 | 值 | 折算 |
|---|---|---|
| `teacher_anneal_total_updates` | 2000（自动 = `episodes × ceil(steps/horizon)`） | — |
| `teacher_anneal_hold_fraction` | 0.50 | update 1000 = **episode 250** 开始退火 |
| `teacher_anneal_end_fraction` | 0.60 | update 1200 = **episode 300** 权重归零 |

本轮**不动**退火时刻表。P1 的候选修法（退火后冻结 actor 学习率、拉长退火、保留残余权重）一律推迟，避免污染消融。

### 3.5 其他固定项

- `--no-dag-progress-potential-shaping`
- `--device cuda`

### 3.6 场景（分支 `param-realign-20260914` 的 `config.py`，不得覆盖）

`NUM_UAVS=5`、`NUM_UES=60`、`EPISODE_LENGTH=500`、`TIME_SLOT_DURATION=5.0`、`AREA 500×500`、
`DAG_BASE_ARRIVAL_PROB=0.0145`（热点 ×2.0，半径 150）、`DAG_MIN/MAX_TASKS=5/8`、`DAG_MAX_LEVELS=4`、`DAG_MAX_PARENTS=3`、
`INPUT_DATA_SIZE_MB_RANGE=(0.1,1.0)`、`OUTPUT_DATA_SIZE_MB_RANGE=(0.15,0.75)`、`TASK_CONSTANT_RANGE=(500_000,1_500_000)`、
`BASE_UPLOAD_BANDWIDTH_MBPS=[1.75,3.5,7.0]`、`BASE_DOWNLOAD_BANDWIDTH_MBPS=[3.5,7.0,14.0]`、`BANDWIDTH_LEVEL_PROBS=[0.3,0.5,0.2]`、
`UAV_COMPUTE_RATE_OPS_PER_SEC=1e9`（量纲 cycles/s）、`P_UAV_COMPUTE=50`、`P_UAV_TX=0.5`、`CLEAN_POWER_MOVE=100`、
`CLEAN_MAX_QUEUE_PER_UAV=16`、`CLEAN_UAV_MOVEMENT_SPEED=15.0`

B2 系奖励量纲基准（`reward_redesign.py:26-29`）：`flowtime_ref=500 s`、`λ_task=1.0 秒/焦`、`λ_move=0.10 秒/焦`。

## 4. 需要的代码改动（5 处，只加开关，不改任何已有臂的行为）

1. `environment/reward_redesign.py:7-19` — `REWARD_REDENOMINATION_ARMS` 加 `"C2A"`、`"C2B"`
2. `environment/reward_redesign.py:41` — `forecast_enabled` 集合加 `"C2A"`、`"C2B"`（用 B2 奖励，必须开估价器 B）
3. `scripts/train_clean_mainline.py:1434-1436` — 改为按 arm 分别设：

```python
if args.reward_redesign_arm in {"C1", "C2", "C2A", "C2B"}:
    args.offloading_eft_advantage = args.reward_redesign_arm != "C2B"
    args.movement_position_advantage = args.reward_redesign_arm != "C2A"
elif bool(args.offloading_eft_advantage):
    raise ValueError("reward-redesign arms cannot combine with EFT advantage")
```

4. `scripts/train_clean_mainline.py:248` — `_resolved_teacher_anneal_total_updates` 的白名单 `{"C1","C2"}` 扩为 `{"C1","C2","C2A","C2B"}`
5. `scripts/run_reward_redesign_arm.py:130` — `"--rollout-horizon", "128"` 改为 `"125"`

**不需要改奖励逻辑**：`reward_redesign.py:57` 的短路集合是 `{"N0","N0-LOCAL","A1","C1"}`，C2A/C2B 不在其中，会自动落到三段账 + 能耗分支，天然得到 B2 奖励。

**回归保护（必须验证并写进报告）**：改动后 `C1` 与 `C2` 解析出的 `offloading_eft_advantage` / `movement_position_advantage` 仍然都是 `True`，`B1`/`B2` 仍然都是 `False`。

## 5. 评测协议

### 5.1 tape

场景参数已变，**必须用 `scripts/generate_fair_eval_tapes.py` 重新生成**，不得复用 20260908 的 tape。

- validation tape ID **100–119**（20 条）
- test tape ID **200–249**（50 条）
- tape 与策略无关：offer 序列、DAG 属性、UE 轨迹预生成；生成过程不得读取策略动作或 active-DAG cap 状态
- 无法接纳的 offer 仍记为 offer 并标 rejected/blocked，不得从分母消失
- 6 个臂 × 3 seed 共用同一份 tape，必须逐值相同

### 5.2 两个协议都跑

`scripts/run_fair_eval_batch.py --protocol {joint,forced_hover}`，**每个 checkpoint 两个协议都要跑**。

- **主表 = `joint`**（完整确定性联合策略，无人机可飞）
- 附表 = `forced_hover`（冻结飞行，隔离卸载策略）

> 口径变更说明：2026-09-08 spec 曾定 `forced_hover` 为主协议，理由是与 leverage check / Stage-1 一致。本轮新增 C2a/C2b 两个飞行老师消融臂，forced_hover 会把飞行老师的效应抹为零，故主协议改为 `joint`。这是**本轮的有意变更**，不是笔误。

### 5.3 checkpoint 选择

- 每个训练 seed **独立**在 validation tape 上选 J/offer 最低的 checkpoint，平局取更早的
- 锁定后在 test tape 上**只测一次**
- 同时评 final checkpoint（episode 500）
- validation 选点与 final 两套结果都要报

### 5.4 统一代价

```
J_episode   = [ Σ_{G∈offers} L_G + 1.0·E_task + 0.10·E_move ] / 500
J_per_offer = J_episode / max(N_offer, 1)

L_G = C_G − a_G      若 G 在 horizon 内完成回传
    = T_end − a_G    若 G 未完成或未被接纳（截尾）
```

越低越好。不使用任何臂的训练奖励、bonus、coverage 或 forecast 值。

### 5.5 必报指标

每臂 × 每 seed × 每协议 × 每 checkpoint：

- `J_per_offer`、delay 分量、task energy、move energy
- admission = N_admitted/N_offer、conditional = N_completed/N_admitted、end-to-end = N_completed/N_offer
- **`hover_ratio`（本轮新增，必报）** — 用于判定 C2a 是否发生 hover collapse
- 两层 95% bootstrap CI：先重采样 3 个训练 seed，再在每 seed 内重采样 test tape
- 每组配对差必须报 3/3、2/3、1/3 的 seed 方向，**不得把 50 个 tape 当成独立训练重复**

## 6. 产物与命名

- run 命名：`20260915_<ARM>_seed{5,86,617}`，ARM ∈ {B1, B2, C2A, C2B, C2, C1}
- 训练产物根：`/data2/zrj2025/uav-results/`（已有目录，不新建）
- 评测产物根：`/data2/zrj2025/uav-results/audits/20260915_six_arm_fair_eval`
- TensorBoard view 根：`/data2/zrj2025/uav-results/tensorboard_views/reward_redesign/`
- 报告：`docs/superpowers/reports/2026-09-15-scenario-realign-six-arm-training-results.md` + 同名 `-result.json`

每份报告必须记录：服务器实际 HEAD、dirty 行数、每个脚本的 git object hash、每个 run 的实际超参（从 run 记录解析，**不得从 `config.py` 推断**）。

## 7. 执行顺序

1. 实施第 4 节的 5 处代码改动，跑回归检查（C1/C2/B1/B2 的 flag 解析值不变）
2. 128-slot smoke：6 个臂各跑 1 个短 run，确认 C2A/C2B 能启动、两个 flag 按预期分别生效、三段账字段有值
3. 重新生成 fair-eval tape，验证 6 臂重放逐值相同
4. 正式训练 18 个 run
5. validation 选点 → 冻结清单 → test 只测一次，两个协议都跑
6. 汇总、bootstrap、写报告
7. 结果追加进 `docs/research/HyperUAV_research_master_roadmap.md`

## 8. 完成判据与必须停止的情况

完成必须同时满足：

- 18 个 run 全部跑满 500 episode，无中断重启混入
- 6 臂的 PPO 超参、场景参数、训练预算逐项一致（报告中逐项列出实际值）
- validation/test 分离，test 只测一次
- `joint` 与 `forced_hover` 两个协议都完成
- 所有排名基于 J/offer，报告中没有用训练奖励数值排名
- 所有结论可追溯到 JSON 原始行

遇到以下情况**立即停止并报告，不做近似降级**：

- 无法让 6 个臂重放逐值相同的外生 tape
- C1 或 C2 的两个教师 flag 解析值与改动前不一致（回归失败）
- 任一 run 的实际超参与本 spec 第 3 节不符
- C2A 或 C2B 的三段账诊断字段全为 0（说明 `forecast_enabled` 没生效）
- 单臂 3 个 seed 中有 run 崩溃且无法用相同参数复现

## 9. 已知限制（写进 roadmap，本轮不处理）

**GAE 有效前瞻远短于 DAG 寿命。**

```
GAE 有效前瞻 = 1/(1 − γλ) = 1/(1 − 0.99×0.95) ≈ 17 个时隙
新场景 DAG flowtime：greedy ≈ 39 时隙，random ≈ 105 时隙
```

优势估计基本看不到 DAG 完成的那一刻，差额全靠 critic 补，而 critic EV 已知很差。这为"逐决策卸载信用分配不可行"提供了一个量化解释，也解释了教师（直接给每个决策打分、完全绕过时序信用）为何效果显著。

候选旋钮（下一阶段）：λ 0.95 → 0.99，有效前瞻 17 → 50 时隙。只减少 GAE 偏差、增加方差，不改目标函数。本轮不动，因为一次只改一个变量。

**其他未处理项**：B2 三段账的量级分解审计（P1，读日志即可）；random 策略下的规模上限；训练墙钟随规模的增长。

## 10. 给 Codex 的提示词（可直接复制）

```
读仓库根 AGENTS.md、docs/research/HyperUAV_research_master_roadmap.md、
docs/clean_load_calibration.md 的最后两节，再完整阅读
docs/superpowers/specs/2026-09-15-scenario-realign-six-arm-training.md，按该 spec 执行。

分支 param-realign-20260914。本次是新场景参数下的六臂重训 + 公平评测：
B1 / B2 / C2a / C2b / C2 / C1，每臂 3 个 seed（5, 86, 617），共 18 个 run，全部 MLP 编码器。

先做 spec 第 4 节的 5 处代码改动（只加开关，不改任何已有臂的行为），
并跑回归检查：C1/C2 的 offloading_eft_advantage 与 movement_position_advantage
仍然都是 True，B1/B2 仍然都是 False。回归不过就停，不要绕过。

然后按 spec 第 7 节的顺序执行：smoke → 重新生成 fair-eval tape → 18 个正式 run →
validation 选点冻结 → test 只测一次 → joint 和 forced_hover 两个协议都跑 → 汇总写报告。

固定要求：
1. 每个 run 跑满 500 episode，不得中途重启混入。
2. 用满 7 张卡最快跑完，你自己检查显存并分配，可一卡多进程。
3. run 命名含年月日：20260915_<ARM>_seed{5,86,617}；接 TensorBoard，只保留判断
   "是否学会"和"是否收敛"的字段，不要把诊断字段全量导出。
4. 结果存已有目录：训练产物 /data2/zrj2025/uav-results/，评测产物
   /data2/zrj2025/uav-results/audits/20260915_six_arm_fair_eval，不新建根目录。
5. 挂上服务器后停止监控，返回预计完成时间，我据此定定时任务来取结果。
6. 尽可能节省额度，不要在小地方反复试。
7. 触发 spec 第 8 节任一硬停止条件就停下报告，不要自行判断是否可以降级。
8. 报告必须记录服务器实际 HEAD、dirty 行数、每个脚本的 git object hash，
   以及每个 run 的实际超参（从 run 记录解析，不得从 config.py 推断）。

特别注意三条：
- 主协议是 joint，不是 forced_hover。这是本轮的有意变更，理由写在 spec 5.2。
- hover_ratio 是本轮新增的必报指标，用来判定 C2a 是否 hover collapse。
- 跨臂排名只能用 J/offer，禁止用任何臂的训练奖励数值排名。

一路自主执行，中途不要问我。结果写：
docs/superpowers/reports/2026-09-15-scenario-realign-six-arm-training-results.md
docs/superpowers/reports/2026-09-15-scenario-realign-six-arm-training-result.json
```
