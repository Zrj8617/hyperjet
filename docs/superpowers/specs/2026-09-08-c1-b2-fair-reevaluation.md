# C1 vs B2 公平重评与峰值 epoch 诊断（2026-09-08）

> 状态：用户已批准。
> 范围：只使用现有 C1/B2 模型做重评；除“旧日志不足时的最小记录性复跑”外，不重训新策略、不增加奖励臂。
> 本文件是聚焦的实验 spec 和新窗口执行说明，不是第二份 master roadmap。活动状态仍以
> `docs/research/HyperUAV_research_master_roadmap.md` 为唯一事实源。

## 0. 新窗口先读什么

按顺序完整阅读：

1. 仓库根 `AGENTS.md`；
2. `docs/research/HyperUAV_research_master_roadmap.md`；
3. 本文件；
4. `docs/superpowers/specs/2026-09-06-reward-redesign-experiment-plan.md`；
5. `docs/superpowers/specs/2026-09-07-missing-arms-and-reeval.md`；
6. `docs/superpowers/reports/2026-09-07-reward-redesign-formal-results.md`；
7. `docs/superpowers/reports/2026-09-07-missing-arms-and-reeval-results.md`。

不要根据 TensorBoard 曲线截图猜参数或指标语义；以 `result.json`、`actual_parameters`、
训练 JSONL 和实际代码为准。

## 1. 这次到底要回答什么

只回答 C1 和 B2 之间的两个问题：

1. **正式公平比较**：在完全相同、与策略无关的外生任务流上，C1 和 B2 哪个模型产生更低的实际时延与能耗综合代价？
2. **师兄提出的峰值 epoch 检查**：分别在 C1 的“老师完全开启”阶段和 B2 的等训练预算阶段取多个奖励峰值，累计每个完整训练 epoch（本项目中即一个 PPO rollout/update）的实际时延与能耗，平均后哪个更低？

第二问是训练轨迹诊断，受“挑峰值”和策略访问到不同状态的影响；它必须报告，但不能推翻或替代第一问的固定任务流独立测试结论。

本次不回答：老师是否具有独立因果效应、B2 奖励加老师会怎样、HGNN 是否优于 MLP。回答这些问题需要新训练臂，超出本 spec。

## 2. 名词：episode、epoch、checkpoint

- **环境 episode**：一次完整的 500-slot 环境运行。
- **PPO rollout/update epoch**：收集一段最多 128 slot 的 rollout，然后执行一次 PPO update。正式训练参数 `ppo_epochs=1`，因此本文“峰值 epoch”特指这一个完整 rollout/update 窗口，不是单个 slot，也不是 500-slot episode。
- **checkpoint**：保存的策略参数。现有正式 run 通常每 10 个环境 episode 保存一次 checkpoint。

不要再把以下三个量混在一起：

- C1 第二轮 TensorBoard 的 `train/episode_reward`：完整环境 episode 累计奖励；
- B2 第一轮同名 TensorBoard 标签：实际是 PPO rollout reward total，标签语义错误；
- 原始 JSONL 中的 `reward`：只代表该更新最后一个 slot 的奖励，不能当成整个 epoch。

## 3. 两组实验究竟是什么

### 3.1 C1：原始奖励 + 退火老师

C1 的环境奖励与 A1 原始奖励相同。每个 slot 的奖励为：

\[
r^{C1}_t =
-0.5\sum_{i\in K_t}w_i\min\!\left(\frac{\Delta t_i}{60},10\right)
-0.05\frac{E^{task}_t}{250}
-0.08\frac{E^{move}_t}{2500}
+8N^{done}_t
+0.5q_t .
\]

其中：

- \(K_t\)：本 slot 第一次到达“奖励可结算完成状态”的任务集合；一个任务只结算一次；
- \(w_i=1\)（关键路径任务），否则 \(w_i=0.5\)；
- 入口任务的 \(\Delta t_i\) 从 DAG 到达时刻算起；其余任务从最晚父任务计算完成时刻算起；
- 除以 60 是把秒换成相对 60 秒参考值的无量纲数；10 是奖励计算的长尾截断，不改变日志中的真实时延；
- \(E^{task}_t\) 是任务计算、通信和结果回传能耗；250 J 是参考量；
- \(E^{move}_t\) 是所有 UAV 的移动能耗；2500 J 是参考量；
- \(N^{done}_t\) 是本 slot 真正完成回传的 DAG 数；
- \(q_t\) 是当前 ready 任务源位置被至少一架 UAV 覆盖的比例。

C1 的特殊之处不在环境奖励，而在 actor 的训练 advantage：

- 前 50% updates：老师权重 \(\alpha=1\)。offloading 使用 EFT anchored advantage，movement 使用服务质心 advantage；
- 中间 10% updates：\(\alpha\) 从 1 线性退火到 0；
- 后 40% updates：\(\alpha=0\)，只用原始奖励产生的 slot GAE。

500 episode、每 episode 500 slot、rollout horizon 128 时总计约 2000 updates。老师完全开启阶段约为 update 1--1000（约 episode 1--250）；实际边界必须读取日志中的 `teacher_weight`/`teacher_phase`，不能只按 episode 猜。

### 3.2 B2：完整增量时延账本 + 能耗，普通 slot GAE

B2 没有 EFT 老师、没有 movement 质心老师，也没有直接注入 actor 的解析 \(\Delta\Phi\) advantage。actor 只通过环境奖励和普通 slot GAE 学习。

完整 episode 的 B2 奖励账本为：

\[
R^{B2}=-\frac{1}{500}
\left[
\sum_G(\hat T_G^0-a_G)
+\sum_k\Delta\Phi_k
+\sum_G(\bar C_G-\hat T_G^{last})
+1.0E_{task}
+0.10E_{move}
\right].
\]

- \(a_G\)：DAG 到达时间；
- \(\hat T_G^0\)：DAG 刚进入系统时的乐观预计完成时间；
- \(\Delta\Phi_k\)：第 \(k\) 次卸载决策前后，所有活跃 DAG 预计完成时间总和的变化；
- \(\hat T_G^{last}\)：DAG 最近一次 forecast；
- \(\bar C_G\)：真实回传完成时间；未完成 DAG 在 episode 结束时进行截尾校正；
- 能耗系数的单位是秒/J，使时延与能耗先换成同一“等价秒”量纲，再除以 500 秒参考值。

Forecast 对未调度任务使用乐观拓扑 DP，对已经调度的任务使用固定计划时间。它是奖励估计器，不是 actor 的额外输入，也不是老师。B2 的 actor 输入输出结构和普通 MLP actor 相同：输入当前任务/UAV/队列等状态，输出合法 UAV 的概率分布；采样动作后由环境给奖励，再由 GAE 更新策略。

### 3.3 为什么 C1 和 B2 的训练奖励不能直接比

C1 含 `+8` DAG 完成奖励和正的覆盖塑形；B2 只有负的时延/能耗成本。C1 到 3000、B2 接近 0 都不表示 C1 必然更好。两者的奖励零点、缩放和组成不同，必须用同一个独立测试函数比较真实后果。

## 4. 已有证据（只能当基线，不能冒充本次正式答案）

现有 3 模型 seed × 20 环境 seed、forced-hover、deterministic masked argmax 结果：

| 配置 | 完成率 | 平均 DAG flowtime | throughput | EFT agreement |
|---|---:|---:|---:|---:|
| C1 | 0.918 [0.897, 0.937] | 182 s [144, 222] | 0.1286 [0.1162, 0.1412] | 0.775 |
| B2 | 0.813 [0.789, 0.837] | 488 s [424, 553] | 0.0700 [0.0629, 0.0776] | 0.467 |

这些数字表面上明显偏向 C1，但尚有两个关键缺口：

1. 相同 seed 不等于相同工作负载。更快的策略会更早释放 UE active-DAG cap，从而接纳更多 DAG；现有评估没有固定独立于策略的 offer tape。
2. 既有 paired bootstrap 没有预注册 C1−B2 这一对的完整统一代价比较，而且只有 3 个训练 seed。

此前从训练轨迹各挑一个最好看的 update，得到过探索性样例：

- C1 teacher-on：`20260907_C1_seed0`，update 890，episode 222，global slot 111256，128 slots，原生 rollout reward 1088.209；
- B2：`20260906_B2_seed2`，update 676，episode 168，global slot 84500，116 slots，原生 rollout reward −1.974。

按原始奖励的时延+能耗惩罚部分重算：

| 单峰样例 | 时延成本 | 任务能耗成本 | 移动能耗成本 | 合计 | 每 slot 合计 |
|---|---:|---:|---:|---:|---:|
| C1 teacher-on | 86.155 | 3.719 | 0.272 | 90.147 | 0.704 |
| B2 | 207.485 | 1.380 | 0.048 | 208.913 | 1.801 |

C1 样例的原始任务能耗是 18596.97 J、移动能耗是 8500 J；B2 分别是 6900.22 J 和 1500 J。C1 的任务能耗约为 B2 的 2.70 倍，总实际能耗约为 3.23 倍。该样例显示 C1 用更多能耗换来了明显更低的时延惩罚，但它是单点、跨 seed、事后挑峰，禁止写成正式胜负。本次多峰值结果也应先用这些数值回归检查旧计算脚本的口径。

## 5. Track A：正式公平比较（主结论）

### 5.1 不允许改变的评估条件

- 只加载已有 C1/B2 checkpoints，不更新参数；
- MLP 架构不变；
- offloading 使用 deterministic masked argmax；
- 主协议 forced-hover，与 leverage check / Stage-1 一致；
- 次协议运行完整 deterministic joint policy，用于观察 movement 后果；
- 500 offer slots，默认不 drain；如增加 drain，只能作为另列的次要敏感性分析；
- C1、B2 必须使用同一份预生成外生 tape：DAG offer、DAG 属性、UE 轨迹和相关外生随机量一致；
- tape 生成不得读取策略动作或当前是否达到 active-DAG cap。无法接纳的 offer 仍记录为 offer 和 rejected/blocked，不能从分母消失。

同 seed 但仍由策略实时驱动到达过程，不算通过固定 tape 条件，也不允许作为降级替代。

### 5.2 checkpoint 选择与测试集合

每个训练 seed 独立选点，不允许从 3 个 seed 中只留下最好的一条。

**比较 A：C1 老师完全开启阶段，等训练预算**

- 候选点：episode 50、100、150、200、250 的 checkpoint；
- C1 候选点还必须由实际日志确认 `teacher_weight=1`；若边界点不是 1，则剔除；
- B2 使用完全相同的 global-slot/update 上限和候选预算；
- 对每个模型 seed，在固定 validation tapes（建议 ID 100--119，共 20 个）上选择平均统一代价最低的 checkpoint；平局取更早的 checkpoint。

**比较 B：最终模型**

- 双方直接使用 episode 500 / `latest.pt`；
- 不经验证集选择。

锁定 checkpoint 后，使用从未参与选点的固定 test tapes（建议 ID 200--249，共 50 个）只做一次正式测试。若实现前发现既有项目已经冻结了另一套不重叠 seed/tape 约定，优先沿用既有约定并在 report 记录，不得事后根据结果换集合。

### 5.3 统一测试代价

正式比较使用实际结果，不使用 C1 的 bonus/coverage，也不使用 B2 的 forecast：

\[
J_{episode}=\frac{
\sum_{G\in offers}L_G
+1.0\,E_{task}
+0.10\,E_{move}
}{500},
\qquad
J_{per\ offer}=\frac{J_{episode}}{\max(N_{offer},1)}.
\]

其中：

\[
L_G=\begin{cases}
C_G-a_G,& G\text{ 在 horizon 内完成回传};\\
T_{end}-a_G,& G\text{ 未完成或未被接纳}.
\end{cases}
\]

解释：1.0 秒/J 的任务能耗权重和 0.10 秒/J 的移动能耗权重把能耗换成“等价秒”；再除以 500 秒只改变数值尺度。**越低越好。**

主统计量为 `J_per_offer` 的 C1−B2 配对差；同时完整报告：

- `J_episode`、`J_per_offer` 及其 delay/task-energy/move-energy 三项；
- `N_offer`、`N_admitted/N_offer`、`N_completed/N_admitted`、`N_completed/N_offer`；
- completed DAG flowtime（仅作辅助，不能忽略未完成 DAG）；
- throughput、任务总能耗、移动总能耗、每完成 DAG 能耗；
- 每个模型 seed 的单独结果，不能只给总平均。

### 5.4 统计方法

- 所有差值定义为 C1−B2；`J`、时延、能耗为负值有利，完成率/throughput 为正值有利；
- tape 内严格配对；
- 给出均值、中位数、配对 95% bootstrap 区间；
- 同时给出两层 bootstrap：先重采样 3 个模型 seed，再在每个模型 seed 内重采样 test tapes；
- 由于只有 3 个训练 seed，必须报告 3/3、2/3 或 1/3 seed 方向，不能把环境 seed 当成新的独立训练重复；
- 不做“区间没跨零即老师因果有效”的表述，因为 C1 与 B2 同时改变了奖励和信用通路。

## 6. Track B：师兄的多峰值 epoch 法（辅助诊断）

### 6.1 峰值怎样选，避免只挑一根最好看的尖刺

每个模型 seed 分别选择 5 个点，共每臂 15 个峰值 epoch：

- C1：只允许 `teacher_weight=1` 的 update；
- B2：只允许不超过 C1 老师完全开启阶段末尾的相同 update/global-slot 预算；
- 峰值依据各臂自己的 `ppo_rollout_reward_total`，数值越大越好；B2 为负值时即越接近 0 越高；
- 先找局部极大值，再按 reward 从高到低选择；任意两个入选点至少间隔 100 updates，防止同一段平台被重复采五次；
- 不跨模型 seed 比较谁的原生 reward 更大，也不从三个 seed 中只挑一个 seed；
- 保存全部候选峰、剔除原因和最终 5 点列表，禁止看统一代价后重新挑点。

若某 seed 不足 5 个满足间隔的局部极大值，则按间隔约束选择 reward 最高的非局部峰值补足，并明确标记。

### 6.2 “整个 epoch”的累计范围

每个点累计该 PPO update 实际消费的整个 rollout，通常为 128 slots；episode 尾部可能少于 128。必须记录 `rollout_slot_count`，并同时报告：

- epoch 总成本；
- 每 slot 成本；
- 每个完成/结算任务成本；
- 该 epoch 的起止 episode、起止 global slot、teacher weight；
- 时延、任务能耗、移动能耗原始值和加权值。

不能使用 JSONL 的单个 `reward` 字段，它只是 rollout 最后一个 slot。

### 6.3 两套加权值都输出，主次必须写清

为了能复核过去的“原始奖励惩罚部分”算法，先输出：

\[
C^{orig}_{epoch}=
0.5\sum_{t}\sum_{i\in K_t}w_i\min\!\left(\frac{\Delta t_i}{60},10\right)
+0.05\frac{E^{task}_{epoch}}{250}
+0.08\frac{E^{move}_{epoch}}{2500}.
\]

它排除 C1 的 `+8` 完成 bonus 和 `+0.5q_t` 覆盖塑形，也排除 B2 的 forecast ledger，因此可直接解释原始 reward 中“付出了多少时延+能耗惩罚”。越低越好。

另外输出与正式 Track A 一致的等价秒口径：

\[
C^{sec}_{epoch}=D^{actual}_{epoch}+1.0E^{task}_{epoch}+0.10E^{move}_{epoch}.
\]

`D_actual_epoch` 必须来自 epoch 内真实发生/累计的时延量，不得使用 forecast \(\hat T\) 或 \(\Delta\Phi\)。如果跨 epoch 的 DAG 无法仅凭既有日志唯一分摊真实 DAG flowtime，则必须同时给出两种边界清楚的量：

1. 本 epoch 首次结算任务的 incremental delay（可精确归属于该 epoch）；
2. 在固定 128-slot 探针起点存在的所有 DAG，从探针起点到结束的实际 censored backlog-area/flowtime increment。

不要把完整 DAG 最终 flowtime 全归给它碰巧完成的那个 epoch。

### 6.4 多点汇总

先在每个模型 seed 内对 5 个峰值取平均，再对 3 个模型 seed 汇总；不能把 15 个点直接当成 15 个独立训练 seed。输出：

- 每个峰值点的明细表；
- 每 seed 的 5 点平均；
- 两臂 3-seed 平均与 C1−B2 差值；
- 每 slot 标准化结果；
- 峰值位置和 C1 teacher weight；
- bootstrap 只用于描述峰值/tape 采样波动，结论注明存在选择偏差。

### 6.5 旧数据不足时允许的最小复跑

先审计既有 `train_metrics.jsonl` 和 checkpoint。已知：

- `ppo_rollout_reward_total` 是整个 update 的 reward 总和，可用于选峰；
- 第一轮 B2 TensorBoard 的 `train/episode_reward` 标签错误，不可作为原始数据；
- 旧日志很可能没有每个 rollout 的完整真实 delay/task-energy/move-energy 聚合，也没有所有逐 slot 轨迹。

若所需字段不足，允许：

1. 增加纯诊断、RNG-neutral 的 rollout 聚合器，只累计已有 `slot_record/info`，不参与 reward、state、action、advantage 或 loss；
2. 先用固定 checkpoint + 固定 tape 做 1 个 128-slot smoke，验证三项之和、slot 数、teacher phase 和落盘 schema；
3. 若必须复现训练峰值，则只重跑到 update 1000（老师完全开启阶段末尾，约 250 episodes），不跑满 500 episodes；C1/B2 使用原正式参数和相同 seed 0/1/2；
4. 新字段至少包括每 update 的 slot 数、原始 incremental delay、截断次数、任务/移动原始能耗、两套加权成本、bonus、coverage、B2 ledger 各项和 teacher weight；
5. gate OFF 必须逐值复现旧行为。用同 seed 短 smoke 对比动作、slot reward、loss 与旧代码，确认记录器不改变 RNG 或训练；
6. 新复跑只能命名为 diagnostic replay，不得覆盖 `20260906_B2_*` 或 `20260907_C1_*`。

禁止用“只记录最后一个 slot”“把 C1 episode reward 除以四”“从 TensorBoard 平滑曲线反推分量”等近似代替缺失数据。

## 7. 现有资产与服务器路径

服务器工作副本：

```text
/data2/zrj2025/HyperUAV-reward-redesign
```

已有结果根目录（必须复用，不新建第二个结果根）：

```text
/data2/zrj2025/uav-results/audits
```

C1 正式 runs：

```text
/data2/zrj2025/uav-results/audits/20260907_C1_seed0
/data2/zrj2025/uav-results/audits/20260907_C1_seed1
/data2/zrj2025/uav-results/audits/20260907_C1_seed2
```

B2 正式 runs：

```text
/data2/zrj2025/uav-results/audits/20260906_B2_seed0
/data2/zrj2025/uav-results/audits/20260906_B2_seed1
/data2/zrj2025/uav-results/audits/20260906_B2_seed2
```

每个 run 的 `result.json` 给出实际 `train_dir` 和 `checkpoint`。不得猜测嵌套时间戳目录；从 `result.json` 解析。预期存在 `checkpoints/latest.pt` 和每 10 episode checkpoint，但执行前要列出并核对。

可复用脚本：

```text
scripts/reevaluate_reward_redesign_checkpoint.py
scripts/run_reward_redesign_reevaluation_batch.py
scripts/eval_clean_mainline.py
scripts/summarize_missing_arms_and_reeval.py
```

现有重评脚本只有“20 seed + forced-hover + 实时到达”的旧协议。它可以复用模型加载和确定性评估部分，但**尚未满足固定 offer tape、未接纳 offer 分母、统一代价和验证/测试隔离**。不要仅重跑该脚本就宣称完成本 spec。

## 8. 实现与执行顺序

1. 只读审计 6 个 run 的 `result.json`、参数、checkpoint 列表、训练 JSONL schema；
2. 设计并 smoke 固定外生 tape，证明 C1/B2 每个 tape 的 offer ID/属性/UE 轨迹逐值相同；
3. 实现统一实际代价与 offer/admission/completion 统计；
4. Track A：validation 选点，冻结选择清单，再跑独立 test；
5. Track B：从原始 `ppo_rollout_reward_total` 冻结每 seed 5 个峰值；审计字段，必要时才做最小 diagnostic replay；
6. 生成机器可读 JSON 和人类可读报告；
7. 按 `AGENTS.md` §3 记录实际 HEAD、dirty 文件摘要、用到脚本的 `git hash-object`；参数必须来自 run 实际值；
8. 停止。不要启动新奖励训练臂，不进入 HGNN vs MLP。

## 9. 产物

建议服务器原始产物均放在既有 audits 根下，使用不会覆盖旧实验的前缀：

```text
20260908_C1_B2_fair_eval_*
20260908_C1_B2_peak_epoch_*
```

仓库报告：

```text
docs/superpowers/reports/2026-09-08-c1-b2-fair-reevaluation-results.md
docs/superpowers/reports/2026-09-08-c1-b2-fair-reevaluation-result.json
```

机器 JSON 至少包含：协议、tape 标识、checkpoint 选择、逐 model-seed/逐 tape 原始结果、统一代价分量、配对差值、bootstrap 结果、峰值清单、版本记录和任何 protocol deviation。

## 10. 完成判据与必须停止的情况

完成必须同时满足：

- Track A 的固定 tape validation/test 与两种策略协议完成；
- Track B 每臂每 seed 5 个峰值点及完整 epoch 分量完成；
- 没有用原生 C1/B2 reward 数值直接排名；
- 所有结论都能追溯到 JSON 原始行；
- 报告明确区分正式公平结论与峰值诊断。

遇到以下情况停止并报告，不做近似降级：

- 无法让两臂重放逐值相同的外生 offer/mobility tape；
- checkpoint 或实际参数无法确认；
- 所需真实时延无法从旧日志恢复，且 diagnostic replay 也无法定义清楚归属；
- 诊断记录器改变 RNG、动作、奖励或 loss；
- 发现现有 C1/B2 checkpoint 对应代码/参数与报告记录不一致。

## 11. 新窗口可直接复制的提示词

```text
读仓库根 AGENTS.md、docs/research/HyperUAV_research_master_roadmap.md，
再完整阅读 docs/superpowers/specs/2026-09-08-c1-b2-fair-reevaluation.md，按该 spec 执行。

本次只评估已有 C1 与 B2 checkpoints，不训练新奖励臂。必须完成两条通路：
1）主结论：固定、与策略无关的外生 offer/mobility tape；每个 model seed 独立用 validation 选 checkpoint，锁定后在独立 test tapes 上比较；同时跑 forced-hover deterministic masked argmax 主协议和完整 deterministic joint-policy 次协议；用 spec 中 J_episode/J_per_offer 的实际时延+任务能耗+移动能耗统一口径，并报告 offer/admission/conditional completion/end-to-end completion。
2）师兄峰值法：C1 只取 teacher_weight=1 阶段，B2 用相同训练预算；每个 model seed 选 5 个至少间隔 100 updates 的原生 rollout-reward 峰值，累计完整 PPO rollout/update epoch 的真实时延和能耗，分别计算 C_orig_epoch 与 C_sec_epoch；先每 seed 对 5 点平均，再汇总 3 seeds。它是辅助诊断，不得替代主结论。

先审计旧 JSONL。字段够就直接计算；字段不够时，只增加 RNG-neutral 诊断聚合并做最小复跑：先 128-slot smoke，确有必要才按原参数重跑到 update 1000/约 250 episodes，绝不跑满 500、不覆盖旧 run。禁止用最后一个 slot、TensorBoard 平滑值或 episode_reward 除法近似 epoch 分量。

复用服务器 /data2/zrj2025/HyperUAV-reward-redesign 与已有结果根 /data2/zrj2025/uav-results/audits。C1 为 20260907_C1_seed{0,1,2}，B2 为 20260906_B2_seed{0,1,2}；实际 checkpoint 和参数从各 run/result.json 解析，不要猜。

一路自主执行，中途不要问我。若触发 spec 第 10 节的硬停止条件才停下报告，禁止近似降级。结果写：
docs/superpowers/reports/2026-09-08-c1-b2-fair-reevaluation-results.md
docs/superpowers/reports/2026-09-08-c1-b2-fair-reevaluation-result.json
按 AGENTS.md §3 记录实际 HEAD、dirty、脚本 git object 和 run 实际参数。做完即停，不新增实验臂，不进入 HGNN vs MLP。
```
