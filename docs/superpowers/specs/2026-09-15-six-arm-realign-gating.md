# 六臂场景重标定 —— 第一阶段门禁（只改代码 + smoke）

日期：2026-09-15
分支：`param-realign-20260914`
上游 commit：`0aa78f0`（场景参数重标定）、`2cb3e2c`（规模上限扫描）

## 0. 本 spec 的范围

**只做代码改动和 smoke 验收，不开任何正式训练 run。**

18 个正式 run（6 臂 × 3 seed）的训练 spec 在本 spec 验收通过之后另写。
第一阶段几乎不消耗机时；把门禁前置，是为了避免 flag 设错之后在一周机时之后才发现
（参照 D1 臂因 η 指令自相矛盾白跑 3 个 run 的教训）。

## 1. 背景

场景参数已按两篇参考文献重新标定并通过负载 Gate（见 `docs/clean_load_calibration.md`）。
下一步是在新场景下重跑六个奖励/教师臂。2026-09-15 的代码审查指出：当前代码
**不足以启动**这批实验——实验臂未实现、评估编排硬编码旧的三臂与 seed 0/1/2、
主协议与选点协议不一致。本 spec 逐条复核了这些问题（§2 给出 file:line 证据），
并把用户已拍板的决定固化为不可自行变更的约束（§3）。

## 2. 已核实的问题（全部经代码核对）

| # | 问题 | 证据 |
|---|---|---|
| P1 | `C2A` / `C2B` 不在臂列表中，传入会被 argparse 拒绝 | `environment/reward_redesign.py:7-19` |
| P2 | 两个教师在入口层被硬绑定，无法单独开关 | `scripts/train_clean_mainline.py:1434-1436` |
| P3 | 退火参数白名单只认 C1/C2 | `scripts/train_clean_mainline.py:248` |
| P4 | `--rollout-horizon` 写死 128 | `scripts/run_reward_redesign_arm.py:130` |
| P5 | 编排脚本写死 C1/B2/C2 与 `_wait_for_c2` 逻辑 | `scripts/orchestrate_fair_eval.py:44,50,53` |
| P6 | 编排脚本写死 `for seed in range(3)` 与旧 run 日期 | `scripts/orchestrate_fair_eval.py:93,150,155` |
| P7 | **选点用 forced-hover validation，与 joint 主表不一致** | `scripts/orchestrate_fair_eval.py:96` |
| P8 | 候选 checkpoint 只有 ep 50–250，全部落在教师权重 = 1 的阶段 | `scripts/orchestrate_fair_eval.py:17` |
| P9 | 汇总脚本只支持三臂三对比 | `scripts/summarize_c1_b2_c2_fair_reevaluation.py:14` |
| P10 | `hover_action_ratio` 嵌在 `policy_metrics` 里，未进 `TEST_METRICS` | `scripts/run_fair_eval_batch.py:112`、`summarize_...py:17-37` |
| P11 | 移动能耗是 500 J 定额（动一下就扣），λ_move=0.10 → 50 等价秒/架/时隙 | `environment/env.py:717` |

关于 P11 的量级：新场景每时隙约完成 0.78 个 DAG、单 DAG 延迟约 196 s，
即延迟代价约 153 等价秒/时隙。一架无人机移动一个时隙就是 50 等价秒（约延迟代价的 1/3），
五架全动 250 秒则超过延迟代价。**悬停在现有权重下是理性选择**，而且训练奖励与裁判 J
用的是同一个权重，靠飞行取胜的策略会被裁判自身扣分。本轮不修改 λ_move（会污染消融），
改为在汇总侧做零成本敏感性分析（G8）。

已确认**不是**问题：`--checkpoint-interval` 默认 10（`scripts/train_clean_mainline.py:1015`），
全程保存 50 个 checkpoint，因此 320/360/400/450/500 均存在，G5 可直接实施。

## 3. 用户已拍板的决定（不得自行变更）

| 项 | 值 |
|---|---|
| 实验臂 | `B1`、`B2`、`C2A`、`C2B`、`C2`、`C1`（大写，CLI 与结果目录统一） |
| 不含 | `A1`、`A2`（增量式已有结论，不收敛，不重训） |
| 训练 seed | **5、86、617**（六臂共用同一组） |
| `--rollout-horizon` | **125**（500 ÷ 125 = 4，无零头；总 update 仍为 2000） |
| 主协议 | **deterministic joint-policy**（主表） |
| 次协议 | `forced_hover`（卸载隔离诊断，附表） |
| 候选 checkpoint | **320 / 360 / 400 / 450 / 500**（全部位于教师归零之后） |
| 教师退火 | `hold=0.50`、`end=0.60`，不变 |
| 统一代价 J 权重 | `λ_task=1.0` 秒/焦、`λ_move=0.10` 秒/焦、`flowtime_ref=500`，不变 |
| 任务编码器 | 全部 `mlp`（HGNN vs MLP 是下一阶段，本轮编码器必须统一） |
| 其余 PPO 参数 | 沿用现有默认值，六臂逐字节相同 |

## 4. 门禁项

### G1 —— C2A / C2B 训练开关

改动四处，只加开关，不改任何已有臂的行为：

1. `environment/reward_redesign.py:7-19`：`REWARD_REDENOMINATION_ARMS` 加 `"C2A"`、`"C2B"`
2. `environment/reward_redesign.py:41`：`forecast_enabled` 集合加 `"C2A"`、`"C2B"`
3. `scripts/train_clean_mainline.py:1434-1436`：改为按臂分别设置
   ```python
   if args.reward_redesign_arm in {"C1", "C2", "C2A", "C2B"}:
       args.offloading_eft_advantage = args.reward_redesign_arm != "C2B"
       args.movement_position_advantage = args.reward_redesign_arm != "C2A"
   ```
4. `scripts/train_clean_mainline.py:248`：`_resolved_teacher_anneal_total_updates` 的白名单
   `{"C1","C2"}` 扩为 `{"C1","C2","C2A","C2B"}`

注意 `reward_redesign.py:57` 的短路集合 `{"N0","N0-LOCAL","A1","C1"}` **不要动**：
C2A/C2B 不在其中，会自动落到三段账分支，天然获得 B2 奖励。

**验收**：§5 的 resolved flag 表逐格实测通过。

### G2 —— rollout-horizon 改 125

`scripts/run_reward_redesign_arm.py:130` 的 `"--rollout-horizon", "128"` 改为 `"125"`，
或改成 CLI 透传（默认 125）。

**验收**：实测每 episode 4 次更新、全程 2000 次更新、四个 rollout 等长无零头。

### G3 —— 六臂 / 新 seed / 新 run 的编排与启动

- `scripts/orchestrate_fair_eval.py`：臂列表、seed 列表、run 目录解析全部参数化，
  移除 `_wait_for_c2` 这类针对旧批次的特例逻辑
- 生产启动脚本：生成 6 臂 × 3 seed，**不得覆盖任何旧结果目录**
- TensorBoard 导出（`export_fair_eval_tensorboard.py`、`export_unified_train_tensorboard.py`）
  扩展 arm 与 comparison 列表

**验收**：不传任何臂/seed 时脚本报错而非回退到旧默认值（防止静默用错配置）。

### G4 —— 选点改用 joint validation

`scripts/orchestrate_fair_eval.py:96` 的 `"forced_hover"` 改为 `"joint"`。

流程固定为：**joint validation 选出唯一 checkpoint → 锁定 → 同一个 checkpoint 上
分别跑 joint test 与 forced_hover test**。禁止两个协议各选各的 checkpoint。

**验收**：manifest 中 `selection_source` 指向 joint validation 文件；
字段名 `forced_hover_validation_J_per_offer_mean` 相应改名。

### G5 —— 候选 checkpoint 改为教师归零之后

`scripts/orchestrate_fair_eval.py:17`：
`CANDIDATE_EPISODES = (320, 360, 400, 450, 500)`

理由：教师权重在 update 1200（= episode 300）归零。旧候选 50–250 全部位于
教师全开阶段，选出来的只能回答"谁模仿老师模仿得好"，回答不了"撤掉老师后还剩多少收益"
——这正是本轮实验的核心问题。`final_ep0500` 仍单独作为不经选择的结果报告。

**验收**：五个候选 checkpoint 文件均存在且被实际读取。

### G6 —— hover ratio 进入汇总

`run_fair_eval_batch.py:112` 把 `policy_metrics.hover_action_ratio` 提到顶层，
并加入 `summarize_*.py` 的 `TEST_METRICS`。需要报告：

- joint test 的 `hover_action_ratio`（每 seed 均值 + 两层 bootstrap 95% CI）
- 训练后 40% 阶段（episode 300–500）的 hover ratio
- forced_hover 协议下该值恒为 1，仅作协议自检，不参与方法比较

理由：`C2A`（只有卸载老师、无飞行老师、B2 奖励里没有任何正向位置信号）是
hover collapse 的高危配置。这个指标是判定它是否塌陷的直接证据。

### G7 —— 2×2 主效应与交互项

汇总脚本的对比列表替换为：

```
移动老师平均主效应 = 0.5 × [(C2B − B2) + (C2 − C2A)]
卸载老师平均主效应 = 0.5 × [(C2A − B2) + (C2 − C2B)]
交互效应           = C2 − C2A − C2B + B2
```

单独的 `C2 − C2A`、`C2 − C2B` 仍然报告，但必须标注为**条件效应**
（"在已有另一个老师的前提下增加本老师的效果"），不得称为净效应。

另两组对比的措辞：

- `C2 − C1`：称为**完整奖励定义的效应**。C1 与 C2 的差异不止记账时序，
  还包含 `+8` 完成奖励、覆盖整形 `+0.5·q_t` 和不同的参考尺度。
- `B2 − B1`：称为**同一实际代价的分期记账效应**，前提是 G9 的账本闭合断言通过。

全部对比均报告两层 bootstrap 95% CI 与 3/3 · 2/3 · 1/3 的 seed 方向。

### G8 —— λ_move 敏感性分析（离线，零机时）

J 的三个分量已分开记录（`TEST_METRICS` 含 `J_move_energy_component`），
因此可用**同一批 test 结果**离线重算不同 λ_move 下的 J/offer，不需重跑任何评测。

汇总脚本增加 `--lambda-move-sweep`，默认 `0.10,0.04,0.0`，输出三档下的
J/offer 与臂间排名。

判读规则写进报告：三档下排名一致则结论稳健；排名翻转则说明"谁赢"由这个人为权重
决定，该事实必须在结论中明示。

### G9 —— smoke 与回归断言

5-episode smoke（六臂各一次，单卡即可），必须同时通过：

1. §5 的 resolved flag 表逐格实测匹配
2. **回归保护**：`C1` 与 `C2` 解析出的 `offloading_eft_advantage`、
   `movement_position_advantage`、`teacher_anneal_total_updates`、`forecast_enabled`、
   `offloading_forecast_advantage` 与本次改动**前**逐字节相同
3. **账本闭合**：固定一条轨迹，断言 B2 的
   `initial_cost + assignment_cost + correction_cost` 与 B1 记录的整回合真实 flowtime
   相对误差 < 1e-6
4. 六臂均能正常完成 5 个 episode 并写出 `train_metrics.jsonl`

## 5. Resolved flag 验收表（必须实测填写，不得从代码推断）

| 臂 | `offloading_eft_advantage` | `movement_position_advantage` | `forecast_enabled` | `offloading_forecast_advantage` | `teacher_anneal_total_updates` |
|---|:-:|:-:|:-:|:-:|---:|
| B1 | False | False | False | False | 0 |
| B2 | False | False | True | False | 0 |
| C2A | **True** | **False** | True | False | 2000 |
| C2B | **False** | **True** | True | False | 2000 |
| C2 | True | True | True | False | 2000 |
| C1 | True | True | False | False | 2000 |

说明：`forecast_enabled=True` 只用于计算 B2 的增量式三段账奖励，
**不等于**把解析 ΔΦ 直接塞给 actor —— 后者由 `offloading_forecast_advantage` 控制，
该值由 `train_clean_mainline.py:1607-1609` 硬编码为 `arm in {"N0","N0-LOCAL","B2D"}`，
六臂全部为 False。因此 C2 系在教师归零后使用的是普通的 slot 级 GAE。

`teacher_anneal_total_updates = 2000` 来自 `episodes(500) × ceil(steps(500)/horizon(125))`，
折算为 update 1–1000 权重 1、1001–1199 线性退火、1200 起为 0，
即 episode 250 开始退火、episode 300 归零。

## 6. 硬停止条件（触发即停止并报告，不得自行决策）

1. §5 任一格实测值与表中不符
2. G9 的回归断言失败（C1/C2 行为发生任何变化）
3. 账本闭合相对误差 ≥ 1e-6
4. 需要修改 §3 中任何一项决定
5. 需要改动 `environment/` 下除 `reward_redesign.py` 以外的任何文件
   （本 spec 不允许改变环境行为）
6. smoke 单臂耗时超过 30 分钟
7. 发现任何旧结果目录有被覆盖的风险

## 7. 交付物

- `docs/superpowers/reports/2026-09-15-six-arm-realign-gating-results.md`
- `docs/superpowers/reports/2026-09-15-six-arm-realign-gating-result.json`

报告必须包含：§5 实测表、G1–G9 逐条通过/未通过、改动文件清单与每个文件的
git object hash、本地 HEAD 与 dirty 行数、smoke 日志路径、以及触发的硬停止条件（如有）。

## 8. 给 Codex 的提示词

```
读仓库根 AGENTS.md 和 docs/research/HyperUAV_research_master_roadmap.md，
再完整阅读 docs/superpowers/specs/2026-09-15-six-arm-realign-gating.md，按该 spec 执行。

当前分支 param-realign-20260914。本轮**只改代码 + 跑 5-episode smoke，不开任何正式训练
run，不占用多卡机时**——因此 spec 里没有 500 轮次、七卡分配和服务器 ETA 的要求，
但命名规范（含年月日）、结果写入已有目录不新建、节省额度、记录服务器 HEAD / dirty 行数 /
每个改动脚本的 git object hash 这几条仍然适用。

要做的是 spec §4 的九个门禁 G1–G9。核心是四件事：
1) 新增 C2A / C2B 两个实验臂，让两个教师（卸载 EFT、飞行位置）能独立开关，
   且 C1 / C2 的行为逐字节不变；
2) rollout-horizon 从 128 改为 125；
3) 评估编排去掉对旧三臂和 seed 0/1/2 的硬编码，选点改用 joint validation，
   候选 checkpoint 改为 320/360/400/450/500；
4) 汇总侧加 hover ratio、2×2 平均主效应与交互项、λ_move 离线敏感性分析。

§3 的决定是用户拍板的，不得自行变更，包括：六臂名称用大写、seed 用 5/86/617、
rollout-horizon 125、joint 为主协议、候选 checkpoint 320-500、教师退火时刻表不变、
J 的三个权重不变、编码器全部 mlp。

完成后必须逐格实测填写 spec §5 的 resolved flag 表——**实测，不要从代码推断**，
并跑通 §4 G9 的三条断言（flag 表匹配、C1/C2 回归不变、B1 与 B2 账本闭合误差 < 1e-6）。

一路自主执行，中途不要问我。触发 spec §6 任一硬停止条件就停下并报告，禁止近似降级、
禁止自行修改 §3 的决定。结果写：
docs/superpowers/reports/2026-09-15-six-arm-realign-gating-results.md
docs/superpowers/reports/2026-09-15-six-arm-realign-gating-result.json
```

## 9. 本 spec 之后

验收通过后另写训练 spec（18 个正式 run）。届时的评测工作量预估：

- joint validation：18 run × 5 候选 × 20 条 tape = 1800 个评测 episode
- test：18 run × 2 checkpoint（selected + final）× 2 协议 × 50 条 tape = 3600 个
- 合计约 5400 个 500-slot 评测 episode

固定 tape 必须按新场景**重新生成**，目录另起，绝不复用旧场景的 100–119 / 200–249 文件；
manifest 记录实际实验 commit 与完整场景参数；replay 前校验 UAV/UE 数、时隙、
任务属性、带宽与算力配置。
