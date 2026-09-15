# HyperUAV 研究上下文交接文档

> **用途**：把当前对话窗口的必要上下文交接给新窗口。
> **生成时间**：2026-09-08
> **代码版本**：`HEAD = aae17ee`（工作区 dirty，大量未提交修改）
> **代码位置**：`D:\CodeFile\HyperUAV-qly`（设备 `desktop-mnqnchi`）

---

## 0. 一分钟速读

- **课题**：HyperUAV —— 多无人机 MEC 系统中的动态 DAG 任务卸载，用 HGNN + MAPPO。
- **论文核心主张**：在环境/PPO/奖励/信用分配/评测**完全相同**的条件下，HGNN/超图表示是否优于一个**公平对照的 MLP**。
- **当前阶段**：奖励重构实验（21 个 run）已跑完并完成 20-seed 评测。**C1（EFT 教师 + 退火）是唯一显著胜出的配置**，但其训练稳定性有已知问题（见 §6）。
- **下一步**：解决 §7 的 P0 疑点后，用 C1 配方启动 HGNN vs MLP 正式对比。

---

## 1. 协作分工与纪律

| 角色 | 职责 |
|---|---|
| **Claude**（本窗口） | 方案设计、代码审查、结果分析、写给 Codex 的提示词 |
| **Codex**（本地 CLI） | 实现、跑实验、返回结果报告 |
| **用户（Caiyurun）** | 审批算法改动和阶段推进 |

**单一事实来源**：`docs/research/HyperUAV_research_master_roadmap.md`（Living Charter，v1.2）。不建第二份 master roadmap。

**版本纪律**（用户定的，必须遵守）：
- 本地 = 编辑源
- 服务器 = 执行副本
- GitHub = 已冻结版本
- **任何正式实验必须记录：实际 HEAD、dirty files 列表、诊断脚本 hash**

**已作废的约束**：不再需要以"师兄同意"为推进条件（用户 2026-09 明确指示）。

**给 Codex 写提示词时的固定要求**（用户提出，每次都要带上）：
1. 每个实验跑 500 轮次
2. 用满 7 张卡最快跑完，Codex 自己检查显存并分配（可一卡多进程）
3. 规范化命名含年月日 + 接 TensorBoard，只保留判断"是否学会"和"是否收敛"的字段
4. 结果存已有目录，不新建目录
5. 挂上服务器后停止监控并返回预计完成时间，据此定定时任务
6. 尽可能节省额度，不要浪费时间在小地方

---

## 2. 用户最初要验证的 4 件事（必须始终保留）

这是整条实验线的锚点，**任何方案调整都不能让这 4 条丢失**：

1. **统一量纲**——时延（秒）和能耗（焦耳）单位不同，用系数统一
2. **只把时延改成增量式**，其余不动，和最原始的实验做对比 → **A1 vs A2**
3. **把最原始的奖励删减到只剩时延 + 能耗**，再把时延改成增量式 → **B1 vs B2**
4. **先给 EFT 学习信号和移动信号，之后关掉**，看能不能学会 → **C1**

---

## 3. 系统与算法（已核对代码的事实）

### 3.1 场景常量（`config.py`）

```
AREA_WIDTH/HEIGHT = 500          NUM_UAVS = 5            NUM_UES = 60
EPISODE_LENGTH = 500             TIME_SLOT_DURATION = 5.0 s
HOTSPOT_RADIUS = 150.0           DAG_BASE_ARRIVAL_PROB = 0.0145
DAG_HOTSPOT_ARRIVAL_MULTIPLIER = 2.0
DAG_MIN_TASKS = 5                DAG_MAX_TASKS = 8
UAV_COMPUTE_RATE_OPS_PER_SEC = 1_000_000
CLEAN_MAX_QUEUE_PER_UAV = 16     CLEAN_UAV_MOVEMENT_SPEED = 15.0
CLEAN_POWER_MOVE = 100.0
```

> ⚠️ **重要教训**：`config.py` 里的 `DISCOUNT_FACTOR = 0.96`，但实际 run 用的是 **γ = 0.99**。
> **实验参数以 run 的实际记录为准，绝不能从 config 推断。** 我曾因此算错过一次残差量级。

### 3.2 两类智能体

- **飞行智能体（movement actor）**：参数共享，每架无人机一个实例
- **卸载智能体（offloading actor）**：**单一的、任务中心的中心化策略**——它遍历"就绪任务"列表，不是遍历无人机

> **推论（关键）**：卸载的信用分配问题是**时序/顺序**问题，**不是多智能体**问题。
> 这一点决定了所有 counterfactual credit 类方案的适用性。

- **奖励是全局共享的**（`env.py`）：`reward_per_uav = reward_total / len(uavs)`，每架无人机拿一样的分。
- **卸载动作没有 no-op**——只在结构性原因下跳过（任务不在图快照里 / 无候选行 / 无合法候选）。

### 3.3 两个"排队时间估价器"（容易混淆，务必区分）

| | **估价器 A**：候选评估 | **估价器 B**：全局预测 DP |
|---|---|---|
| 文件 | `environment/assignment.py` | `environment/forecast.py` |
| 算什么 | 单个任务派给单架无人机的时间/能耗分解 | 全局所有未完成 DAG 的预测完成时间 Φ 及增量 ΔΦ |
| 用在哪 | **策略网络的输入特征** | **奖励（B2）或优势（B2D/N0）** |
| 时序 | 决策**之前** | 决策**之后**（`scorer()` → `argmax` → `delta_for_reservation()`） |
| **部署时** | **必须有，框架的一部分** | **不需要，纯训练脚手架** |
| 开销 | 可忽略 | **训练墙钟 20.15%**，部署 0 |

**估价器 A 的特征维度**（这是策略实际看到的东西）：
- pair 特征 8 维：`[传输时间, 传输能耗, 排队时间, 计算时间, 计算能耗, 增量时延, 回传时间, 回传能耗]`
- dynamic 特征 7 维：`[x, y, 队列长度, 剩余槽位, 还要等多久空闲, 排队工作量, 本时隙已分配数]`

> **注意**：pair 特征第 6 维就是**"增量时延"**（`estimated_finish_time − now`，归一化用 `x/(x+160)`）。
> 也就是说 **"增量"概念本来就在策略观测里**，B2 只是把它也搬进了奖励。这可能是 B2 收益有限的原因之一。

**估价器 B 的机制**（贪心 + 乐观的影子调度 DP）：
- 影子时钟 `queue_clock[u]` 从真实已排产的 `compute_finish_time` 初始化，**故意不含 return time**
- 未分配任务：`ready[u] = max over 父 ( min over 父UAV ( finish + 传输 ) )` ← **乐观**
- `finish = max(ready, local_queue) + compute_time`，取 `argmin` 占位 ← **贪心**
- DAG 完成时间 = `max over sinks ( min over UAV )`
- `Φ = Σ_{未完成 DAG} T̂_G`；`ΔΦ` 拆成 `target`（本 DAG）+ `cross`（挤占别的 DAG）
- 传输模型简化：`data_mb × 8 × (1 + (d/100)²) / bandwidth`，不调真实通信模块

### 3.4 已知的场景问题（声明了但未强制执行）

- **C7（MIN_UAV_SEPARATION）**：`_apply_clean_movement` 只检查地图边界，没有强制最小间距
- **C8（DAG_TASK_UAV_MAX_DISTANCE）**：`is_assignment_legal` 只检查 ready / 有效UAV / 未被预留 / 队列容量，**没有通信可达性检查**（代码注释自己写了 "T7 should extend this helper with communication reachability"）
- 无人机算力异质性参数存在但未使用
- UE 速度矛盾：`UE_GM_MAX_SPEED = 0.6 < UE_WALK_SPEED_MEAN = 1.2`
- `task_execution.py` 在分配时把 `uav_available_time` 设为 `compute_finish`，回程时间稍后才补进去 → 同时隙的后续任务可能被排进"算完但还没飞回来"的窗口

---

## 4. 奖励公式（每一组的确切定义）

### 4.1 最原始奖励（= A1 = C1 的奖励部分）

```
r_t = −0.5  · Σ_{k∈K_t} w_k · min(d_k / 60, 10)     时延惩罚
      −0.05 · Σ_{k∈K_t} (e_k / 250)                  计算能耗
      −0.08 · (E_move,t / 2500)                      飞行能耗
      +8.0  · n_completed_DAG,t                      DAG 完成奖励
      +0.5  · q_t                                    覆盖整形
```

- `K_t` = 本时隙**新结算**的子任务集合，由 `task.reward_settled` 保证只算一次
- `w_k` = 关键路径 1.0 / 非关键 0.5
- `/60` = `CLEAN_REWARD_TIME_REF`，clip 到 10
- `/250` = `CLEAN_REWARD_TASK_ENERGY_REF` = `P_UAV_COMPUTE × 5.0`（一架无人机满功率算一个时隙）
- `/2500` = `CLEAN_REWARD_MOVE_ENERGY_REF` = `5 × 100 × 5.0`（5 架全飞一个时隙）
- `q_t` 覆盖整形：`ENABLE_MOVEMENT_POSITION_SHAPING = True` **本来就是开着的，不是新加的**

> ⚠️ **常见误解**：原始奖励**已经归一化了**（三个 ref 常数早就存在）。
> "统一量纲"这轮改的是**换参考尺度和权重**（时延 ref 60 → 500），**不是从无到有引入归一化**。
> 因此 A1/A2 的差异**不能**归因于"加了归一化"。

### 4.2 重构后的量纲基准（`environment/reward_redesign.py`）

```
flowtime_ref_seconds       = 500.0   # 一条 DAG 典型 flowtime
lambda_task_seconds/joule  = 1.0
lambda_move_seconds/joule  = 0.10    # D2 组特例：0.04
```

### 4.3 各 arm 的定义

| Arm | 奖励定义 | 用估价器 B？ | 解析优势？ |
|---|---|---|---|
| **A1** | 原始奖励，原样透传 | ✗ | ✗ |
| **A2** | `原始 − step_time_penalty + time_reward`（只换时延项为增量式） | ✓ | ✗ |
| **B1** | `time_reward − task_cost − move_cost`，时延用**真实 flowtime**（完成时一次记） | ✗ | ✗ |
| **B2** | 同 B1 结构，时延用**增量式三段账**（见下） | ✓ | ✗ |
| **B2D** | 同 B2，但 `−ΔΦ/500` 经 batch 标准化后**直接当优势**注入 actor loss | ✓ | ✓ |
| **C1** | 奖励 = 原始奖励（不变）；**加 EFT 教师信号 + 移动信号，中途退火撤掉** | ✗ | ✗ |
| **D1** | （已作废，见 §8 错误记录） | ✓ | ✗ |
| **D2** | 同 B1，`λ_move = 0.04` | ✗ | ✗ |
| **N0** | 原始奖励 + `−ΔΦ/500` 当解析优势 | ✓ | ✓ |
| **N0-LOCAL** | 同 N0，但只用 `target`（本 DAG）不用 `cross` | ✓ | ✓ |

### 4.4 B2 的"增量式三段账"（望远镜求和）

```
timing_cost = initial_cost + assignment_cost + correction_cost
time_reward = −timing_cost / 500
```

| 项 | 何时记 | 记什么 |
|---|---|---|
| `initial_cost` | DAG **首次出现**，仅一次（`admitted` 集合） | `T̂_G − arrival_time`（预测的整条 flowtime） |
| `assignment_cost` | **每次分配** | `Σ ΔΦ` |
| `correction_cost` | DAG **真正完成** | `真实 return_complete_time − 最后一次预测锚点`；episode 末未完成的按 `now − 锚点` 补 |

> ⚠️ **2026-09-15 更正：下面这两句原话是错的，已证伪，保留原文仅为留痕。**
> ~~三段加起来恰好等于真实 flowtime，与 B1 的 episode 总回报数学上相等。
> → B2 不是换目标函数，只是把同一笔账从"完成时一次记"改成"分期摊到每次决策"。~~
>
> **实测：B1/B2 同轨迹账本最大相对误差 `0.4891791541`（要求 <1e-6），恒等式从未成立。**
> 要闭合必须满足 `Σ ΔΦ_G = anchor_G − T̂⁰_G`，即预测值的**全部**漂移都由分配决策造成；但预测
> 还会因 UE 移动、新 DAG 到达改变排队、以及影子 DP 本身的**贪心 + 乐观**假设而移动
> （`environment/forecast.py`，影子时钟故意不含 return time），这部分漂移**完全没有被记账**。
> 因此 **闭合缺口 = 未被记账的预测漂移**，`b2_total ≈ 0.51 × b1_total`。
>
> **推论更正**：B2 与 B1 是**两个不同的目标函数**，不是同一目标的两种记账方式；
> 曾据此推出的"B1/B2 优化同一目标、两者都差 ⇒ 问题在目标函数本身"**作废**；
> `B2 − B1` 不得称为"同一实际代价的分期记账效应"。
> 同时这**坐实了**下面那条原标注"尚未用数据坐实"的机制假说，且证据更硬。
>
> 证据：`docs/superpowers/reports/2026-09-15-six-arm-realign-gating-results.md`

**B2 失败的机制假说（尚未用数据坐实）**：`initial_cost` 量级是几百秒且**与策略完全无关**（纯看到达了什么活儿），而 `assignment_cost`（真正携带决策质量的部分）只有几秒到几十秒 → 运气信号淹没本事信号。
**验证方法**：`initial_cost / assignment_cost / correction_cost` 三项在 `_diagnostics` 里是分开记的，读日志即可，不用重跑。**这个检查还没做。**

### 4.5 C1 的教师退火时刻表

```python
teacher_anneal_hold_fraction = 0.50   # 前 50% 更新，教师权重恒为 1.0
teacher_anneal_end_fraction  = 0.60   # 到 60% 时线性降到 0
progress = (update_step + 1) / total
w = 1.0                        if progress <= 0.50
  = (0.60 − progress) / 0.10   if 0.50 < progress < 0.60
  = 0.0                        if progress >= 0.60
```

**折算到 500 episode × 500 step/ep = 250k step**：
- episode 250（125k step）开始退火
- **episode 300（150k step）教师权重归零** ← TensorBoard 上那个断崖就在这里

---

## 5. 实验结果总账（20-seed 评测，唯一可跨组比较的裁判）

> **只有环境物理指标可跨组比较。** 训练奖励曲线不能跨组比——各组奖励定义不同，不是一把尺子。

| 配置 | Completion | Flowtime ↓ | Throughput ↑ |
|---|---|---|---|
| **C1** | **0.9181** | **182.4** | **0.1286** |
| Stage-1 参考（100 ep） | 0.8911 | 246.5 | 0.1152 |
| **A2** | 0.8677 | 304.7 | 0.1055 |
| EFT-greedy 参考 | 0.8413 | 331.7 | 0.0971 |
| **A1**（最原始奖励） | 0.8478 | 367.2 | 0.0922 |
| **B2** | 0.8131 | 487.6 | 0.0700 |
| **B2D** | 0.7850 | 559.9 | 0.0626 |
| **N0** | 0.7818 | 573.4 | 0.0614 |
| **N0-LOCAL** | 0.7835 | 578.4 | 0.0610 |
| **B1** | 0.7688 | 598.5 | 0.0591 |
| Random 参考 | 0.7702 | 624.2 | 0.0586 |
| **D2** | 0.7523 | 662.8 | 0.0545 |

**结论**：
- 对用户目标 2（A1 vs A2）：**增量式有效**，flowtime 367.2 → 304.7
- 对用户目标 3（B1 vs B2）：**删到只剩时延+能耗后两者都很差**，B2 好于 B1 但都不如 A1
- 对用户目标 4（C1）：**EFT 教师 + 退火是唯一明显胜出的机制**
- 所有解析优势方案（N0 / N0-LOCAL / B2D）**全部失败**，且 N0 vs N0-LOCAL 无差异 → **跨 DAG 外部性不是失败原因**

---

## 6. 已确立的负面结论（这是本课题最硬的证据链）

### 6.1 「针对长时程全局回报的逐决策卸载信用分配，在本环境中不可行」

三条独立证据：
1. **Phase 4A oracle**：只能分辨 63/270 = **23%** 的真值对
2. **奖励密度审计 + PBRS 训练探针**：PBRS（Ng et al. 1999）因望远镜性质**保持回报不变**，因此不可能给信噪比本就极低的价值函数增加信息；实测 critic EV **变差**
3. **解析 ΔΦ（N0）**：熵降到 0.15（策略很确定）但性能 ≈ 随机 → **锐利但方向错**

### 6.2 但卸载策略的杠杆是**大**的

EFT-greedy vs Random：flowtime **−46.87%**，throughput **+65.82%**，**20/20 seed 全胜**。
→ 问题不是"卸载不重要"，而是"从全局回报学不到卸载"。
→ **解法：启发式锚定的信用（heuristic-anchored credit）**，即 C1。

### 6.3 方法论结论（值得写进论文）

- **熵下降 ≠ 学会了**——网络可以非常自信地做蠢事
- **critic EV 不能给策略排序**
- **训练奖励曲线不能当裁判**——它测的是"奖励怎么定义的"，不是"策略多好"
  - C1：训练曲线**难看**（震荡）但评测**好**
  - B2：训练曲线**好看**（平稳收敛）但评测**差**
  - B2 收敛得平稳，恰恰是"奖励与策略无关"的**预测结果**，不是反例

### 6.4 「C1 还算不算强化学习？」

算。理由：教师是**被退火掉的**，最终策略是纯网络推理。这是标准的课程学习（curriculum learning）。
同理，估价器 B 和 critic 一样是训练期组件——**没人会说 PPO 因为训练用了 critic 就部署依赖 critic**。
分界线是：**部署的策略需不需要它**。答案是不需要。

---

## 7. 未决问题（按优先级）

### P0 —— C1 的 checkpoint 疑点（**必须先解决**）

三个 seed 的**最终训练奖励差 7 倍**（374 / 2684 / 1214），但**评测几乎相同**（0.920 / 0.916 / 0.917）。

"训练随机采样 vs 评测确定性 masked argmax"只能解释一部分，7 倍太大了。

**需要 Codex 确认：评测用的是哪个 checkpoint？**
如果评测用的是训练中途的最优 checkpoint 而非最终 checkpoint，那"C1 学会了"这个结论要打折扣——真正好的策略可能只存在于撤教师之前。

**这是当前证据链上最大的洞。HGNN vs MLP 正式对比必须等这个查清楚再启动**，否则整个对比建立在一个不确定的配方上。

### P0 —— C1 撤教师后的震荡

**已向用户承认："决定性胜出"在稳定性上说过头了。** 用户从 TensorBoard 图上看出的震荡是真实的。

诚实的表述应该是：
- 断崖在 150k step，**精确对应 episode 300 教师权重归零**
- A1 和 C1 **共用同一个奖励定义**，所以 12.6（A1 后期均值）vs 1682.1（C1 后期均值）是可比的
- 即使 C1 震荡的**谷底**（约 300–500）也**远高于** A1 的 12.6 → **从"优秀"掉到"良好"，没掉回基线**
- **机制**：撤教师后唯一剩下的梯度信号是基于回报的优势——而 §6.1 已证明它在本环境里基本是噪声 → 策略在噪声梯度下随机游走

**建议修法（推荐 1+2 组合）**：
1. 退火结束时把 offloading actor 的学习率降到 0（冻结）
2. 退火更慢：`hold` 0.50 → 0.70，`end` 0.60 → 0.95
3. 保留残余教师权重 0.05–0.1（代价：削弱"完全撤掉老师"这个论文卖点）

### P1 —— B2 失败机制的坐实

读日志即可，不用重跑：拉 `initial_cost / assignment_cost / correction_cost` 的量级比例。若 `initial_cost` 占 80%+ 则机制坐实（见 §4.4）。

### P1 —— HGNN vs MLP 正式对比

spec 已存在：`docs/superpowers/specs/2026-09-06-hgnn-vs-mlp-formal-comparison.md`
要求：**≥5 seed**、**容量匹配**、用 C1 配方（待 P0 解决后确定最终配方）。

### P2 —— 场景问题
见 §3.4：C7 / C8 未强制、算力异质性未用、UE 速度矛盾。

### P3 —— 杂项
仓库里有个 `.__wtest__` 残留文件，`device_bash` 默认无删除权限。

---

## 8. 我（Claude）在本轮犯过的错误 —— 新窗口请引以为戒

用户明确担心："我主要会担心你随着聊天上下文变长会产生幻觉或者忘记关键信息"。以下每一条都要记住：

| # | 错误 | 教训 |
|---|---|---|
| 1 | 用 `config.py` 的 γ=0.96 算残差，实际 run 是 γ=0.99。`0.99^499 = 0.00664`，残差 ≈ 8.42 ≈ **一整个 DAG bonus** | **实验参数以 run 实际值为准，不能信 config** |
| 2 | PBRS 探针选错势函数——用了"DAG 完成进度"（一个不携带质量信息的计数器） | 势函数必须**测量决策质量**（师兄的 Φ = −T̂_G） |
| 3 | 时序门限用错分母——把 10% 门限设在 `forecast/collection` 上，但 collection 只有 ~8ms/slot | 重测为 `forecast/总训练墙钟 = 20.15%`；**并承认 10% 这个数本身是我拍的、过严**，已放宽到 30% |
| 4 | 夸大"拖延"风险——用 holding-cost 替换了用户的"不增量"定义 | 卸载 actor **没有 no-op**，任务时长物理有界，该 exploit 大概率不可达。**已撤回，恢复用户原定义** |
| 5 | η 指令自相矛盾，**浪费 3 个 run** —— spec 定义 D1 为 η=0，后续提示词写成 "η 固定为 1"，导致 D1 与 B2 逐字节相同 | **被浪费掉的恰恰是唯一能诊断 ΔΦ 为何失败的那一组** |
| 6 | A1 基线不可比 —— 我标 A1"已有，不重跑"，复用 Stage-1 OFF 组（100 ep，20-seed 评测）去比 A2（500 ep，1-seed 评测）。**用户抓到了**："那也就是说一跑的所有实验里面并没有跑我最原始奖励的那个版本" | **用户目标 2 当时根本没被测过。** 已在匹配条件下重跑 A1 |
| 7 | "决定性胜出"过度断言 —— 只看最终评测数字，没交代训练稳定性。**用户从 TensorBoard 图上抓到** | 报结论必须同时报稳定性 |
| 8 | 「策略网络输入里没有排队估价」—— 对估价器 B 成立，**对估价器 A 不成立**。用户追问"你确定吗"才查出来 | 见 §3.3。**"排队时间估价器"这个词有歧义，回答前先问清是哪一个** |

**行为准则**：
- 任何关于代码的断言，**先读代码再说**，不要凭印象
- 用户挑战时（"你确定吗"），**去查，不要重复断言**
- 承认错误时直接说，不要绕

---

## 9. 关键文件索引

### 代码
| 路径 | 内容 |
|---|---|
| `config.py` | 场景 + 算法参数（618 行）—— **但实验参数以 run 记录为准** |
| `environment/metrics.py` | `calculate_step_reward`，原始奖励 |
| `environment/env.py` | 全局共享奖励；`_apply_clean_movement`（943 行） |
| `environment/assignment.py` | **估价器 A**；`is_assignment_legal`；`build_offloading_candidate_components` |
| `environment/forecast.py` | **估价器 B**：`ForecastContext`（240 行，untracked） |
| `environment/reward_redesign.py` | 各 arm 的奖励拼装；`RewardRedesignLedger` |
| `environment/task_execution.py` | L151 分配时设 `uav_available_time`；L351 `_maybe_start_return` |
| `marl_models/mappo/clean_offloading_actor.py` | 卸载 actor；`act()` 的决策—预测时序在此 |
| `marl_models/mappo/clean_trainer.py` | 教师退火（`_teacher_weight`）；解析优势注入点 |
| `scripts/train_clean_mainline.py` | 训练主入口；`_collect_clean_slot`（L3439） |
| `scripts/eval_clean_mainline.py` | 评测入口 —— **`forecast` 出现 0 次** |
| `scripts/offloading_policy_gate.py` | 评测期动作选择 —— **无 forecast** |

### 文档
| 路径 | 内容 |
|---|---|
| `AGENTS.md` | Codex 入口：角色、单一事实来源、任务循环、版本纪律、10 条工程原则、禁做实验清单 |
| `docs/research/HyperUAV_research_master_roadmap.md` | **Living Charter（唯一事实来源）**，v1.2 |
| `docs/research/HyperUAV_problem_formulation.md` | (P1) 目标 + 约束 C1–C8，标注了 C7/C8 声明但未执行 |
| `docs/superpowers/specs/` | 各实验 spec |
| `docs/superpowers/reports/` | Codex 返回的报告 |

**相关 spec**：
- `2026-09-04-reward-density-diagnostic.md`
- `2026-09-05-shaped-reward-training-probe.md`（v3）
- `2026-09-05-offloading-leverage-check.md`
- `2026-09-06-stage1-eft-anchored-credit-probe.md`
- `2026-09-06-hgnn-vs-mlp-formal-comparison.md` ← **下一步要用的**
- `2026-09-06-reward-redesign-experiment-plan.md`
- `2026-09-07-missing-arms-and-reeval.md`

### 已知的协议偏差（第一轮实验，Codex 自报）
1. 每个 checkpoint 只在 **1 个匹配 seed** 上评测（Stage-1 用的是 20 seed）
2. B2D 实现为 batch 标准化的 `−ΔΦ/500` 解析优势，**不是**预注册的 micro-critic 版本
3. **3 个 TensorBoard 标量记的是 rollout/slot 值但标了 episode/total 名** —— 跨轮次比这几条曲线要小心

> TensorBoard 横轴是 **environment step（时隙）**，不是 episode。20k step = 40 episode；500 episode = 250k step。
> `rollout_reward` = 128 时隙片段；`episode_reward` = 完整 500 时隙 episode（比值 ≈ 3.9）。

---

## 10. 给新窗口的开场建议

1. 先读 `docs/research/HyperUAV_research_master_roadmap.md`（Living Charter）
2. 处理 §7 的 **P0（C1 checkpoint 疑点）** —— 给 Codex 写一个读日志的核查提示词，不用重跑
3. P0 澄清后，决定 C1 的最终配方（是否加冻结/慢退火）
4. 再启动 HGNN vs MLP 正式对比

**不要做的事**：
- 不要在没查代码的情况下断言代码行为
- 不要从 `config.py` 推断已跑实验的参数
- 不要跨组比训练奖励曲线
- 不要重新引入"等师兄同意"作为推进条件
