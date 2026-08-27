# HyperUAV 研究主线与诊断路线图（Living Research Charter）

**版本：v1.0**  
**日期：2026-08-27**  
**用途：作为 HyperUAV 后续所有实验、代码修改和新对话窗口的“主线锚点”。**  
**当前研究分支：** `codex/phase3ab-phase4a-research-snapshot-20260826`  
**GitHub 已提交锚点：** `7c02c258cfc88659bf7021f22bedfab6902bc383`  
**注意：** Scheme-B2 strict semantic common-random 的 3 个新增文件与结果在本路线图形成时尚未提交，应先独立提交为研究里程碑。

---

## 0. 如何使用这份文档

这份文档不是一次性的总结，而是**长期维护的研究宪章 + 决策日志**。

后续每开启一个新窗口、让 Codex 做一次较大修改、或准备开启新实验前，都先读取本文件。任何新建议都必须先放回本文的诊断树中，不能因为最近一次讨论出现一个新想法，就直接改变整个研究方向。

每次阶段结束只更新四类内容：

1. **已确认事实**：有实验或代码证据支持；
2. **开放假设**：仍待验证，不写成结论；
3. **当前唯一下一步**：一次只解决一个主问题；
4. **决策日志**：记录为什么继续、冻结或放弃某条路线。

推荐把 Markdown 版本提交到仓库，例如：

`docs/research/HyperUAV_research_master_roadmap.md`

Word 版本用于阅读、汇报和交接；**仓库内 Markdown 版本建议作为唯一 canonical source**，因为最方便 ChatGPT/Codex 在任何窗口重新读取和版本比较。

---

# 1. 原始研究目标（不可因诊断过程而漂移）

HyperUAV 的核心研究问题不是“给每个 task 找一个固定最优 UAV”，也不是把强化学习改造成监督分类器。

核心目标应保持为：

> **在动态 DAG 任务与 UAV 移动/卸载联合决策场景中，利用 HGNN/超图表示任务之间的高阶依赖和关系，通过端到端强化学习获得更好的长期系统性能，并与不使用超图结构的 MLP 基线进行公平比较。**

长期系统性能至少包括：

- DAG completion / completion rate；
- DAG flowtime / task delay / critical-path delay；
- task communication + computation energy；
- UAV movement energy；
- queue / backlog / load balance 等系统状态。

### 1.1 HGNN 在研究中的正确角色

HGNN 是**状态表示器/结构编码器**。它应该帮助 actor/critic理解：

- task 的父子依赖；
- 多跳依赖；
- 关键路径与未来任务释放；
- task 之间的资源竞争/属性关系；
- 当前一个 offloading 决策对未来 DAG 的连锁影响。

最终希望证明的是：

`同样的环境 + 同样的 reward + 同样的 RL 训练框架` 下，HGNN 因为利用了结构信息而优于 MLP。

### 1.2 “best UAV”只允许作为局部诊断概念

Scheme-B 中“某个 decision 下哪个 UAV 长期结果更好”表示的是局部 counterfactual：

`Q(s, a=UAV_j)` 在某个具体状态下的长期后果。

它**不表示某架 UAV 全局上永远最好**，更不应自动转化成 actor 的监督标签。

### 1.3 当前 UAV 异构性

当前 clean 主线的计算执行使用统一的：

`UAV_COMPUTE_RATE_OPS_PER_SEC = 1,000,000`

因此当前 5 架 UAV 的差异主要来自位置、队列、可用时间、通信距离和前序任务位置，而不是不同算力。旧配置中存在 `UAV_COMPUTING_CAPACITY[]` 等异构变量，但 clean 执行主线并未据此设置每架 UAV 的计算速度。

**当前不应为了“制造 action 差异”而临时加入 UAV 异构性。** 若论文最终需要异构 UAV，应在基础系统可学习之后作为独立 scenario dimension 引入。

---

# 2. 到目前为止已经确认的事实

## 2.1 原始 MLP/HGNN PPO 基线没有显示正常策略学习

现有实验长期出现：

- offloading entropy 接近 1；
- movement/offloading preference 缺乏稳定筛选；
- critic explained variance 接近 0；
- reward/策略表现没有形成可信的持续改善。

因此目前不能把“HGNN 没优势”当作结论，因为 MLP 基线本身也未证明该环境 + reward + PPO 是可学习的。

## 2.2 给 actor 低噪声、action-specific 的“老师信号”后可以形成 preference

EFT offloading 信号、movement position/centroid 类信号曾使 actor 明显降低 entropy、形成动作偏好。

正确解释是：

> actor 网络、optimizer 和 PPO 更新链条并非完全失效；当 signal 低噪声且直接对应动作时，策略能够学习。

**不能据此推出：actor 必须有监督标签。** PPO 原本就应从环境 reward 中学习。

## 2.3 Scheme C / action-value 路线没有形成可信长期监督

此前 action-conditioned Q / Scheme-C 的多轮诊断表明：短视 action-value 标签与真正长期环境后果并不稳定一致；直接把这种信号推给 actor 曾使 reward 恶化。因此该路线目前冻结，不作为当前主线。

## 2.4 Scheme-B 证明“单个 offloading action 有长期真实后果”

通过 same-snapshot branching，只改变一个 `task -> UAV` 决策，可以观察到后续几十个 slot 的 DAG completion / return 出现明显差异。

这证明：

> offloading action 不是无关紧要的；原始环境里确实存在长期 action consequence。

但它没有证明 reward 一定正确，也没有证明问题一定只是 credit assignment。

## 2.5 H20 不是可靠长期 horizon

严格实验中，许多 target DAG 的 completion 发生在 H≈46–66，部分 branch 到 H100 仍未完成。严格 CRN 条件下，H20 排名与 completion-scale 的 `G_common` 排名仍非常不稳定。

因此：

> 不再把 H20/H30/H40 等固定短 horizon 当作正式 oracle。

## 2.6 Scheme-B2 strict semantic common-random 已基本通过工程门禁

当前结果：

- semantic key：`(future_slot, ue_id, subsystem)`；
- 覆盖 mobility / arrival / DAG generation；
- shared semantic checks：32,582；
- semantic mismatches：0；
- 未识别 environment RNG：0；
- serial/spawn：5/5 完全一致；
- 5 个 decision 中 4 个在 H100 内所有 UAV branches 都完成；
- `task_8` 的 UAV3/UAV4 到 cap 仍未完成。

因此 Scheme-B2 应被冻结为**诊断基础设施**。

**重要修正：** 旧 H20 实验在 H20 内本来就 RNG 对齐，所以“旧 H20 ranking 与新 CRN H20 ranking 不同”不能证明旧 H20 被 RNG divergence 污染；它更多提示 action ranking 对未来随机 realization 可能敏感。真正能证明 H20 太短的是**同一 CRN rollout 内的 G20_CRN vs G_common 不稳定**。

---

# 3. 当前不能混淆的三个概念

## 3.1 Reward design：系统想优化什么

Reward 回答：

> “系统做成什么样算好？”

当前 clean reward 主要由：

- task incremental-delay penalty；
- task computation/communication/return energy penalty；
- UAV movement energy penalty；
- completed DAG bonus；
- 可选 movement position shaping；

组成。

如果 reward 越高却对应更差的 completion/flowtime/energy，那么是 **reward alignment/design failure**。

## 3.2 Credit assignment：最终结果应该算到哪个动作头上

Credit 回答：

> “这个长期结果是谁造成的？”

HyperUAV 同时存在：

- **temporal credit**：一个 offloading action 的关键后果可能 40–60 个 slot 后才出现；
- **within-slot multi-action credit**：同一 slot 有 5 个 movement actions + 多个 offloading decisions，共享较粗的 slot-level signal。

这是一类普遍 RL 问题；“decision-specific credit”只是针对 HyperUAV 结构的具体描述，不意味着必须额外造标签。

## 3.3 Representation：HGNN 是否提供了 MLP 没有的信息

即使 reward 和 credit 都正确，如果当前 task/DAG 场景不需要高阶结构，HGNN也可能与 MLP 差不多。

当前 MLP raw task feature 已经包含：

- DAG level；
- 是否入口/出口；
- predecessor 数；
- successor 数；
- ready/pending；
- input/output/operation/bandwidth 等。

因此 MLP 并非“完全看不到结构”。HGNN 的增量信息主要是**具体连接关系及高阶 hyperedge 关系**。当前 DAG 仅 5–8 tasks、最多 4 levels，是否足以放大 HGNN 优势仍待最终验证。

---

# 4. 当前主要竞争性假设（全部保留，暂不押注）

以下不是结论，而是待验证的根因候选。

## H1. Environment/load process 存在策略反馈耦合

当前每个 UE active-DAG cap 默认为 1：

`DAG 完成更快 -> UE 更早释放 -> 更早重新具备 arrival eligibility -> 生成更多工作负载`

因此一个“更好的策略”可能因为完成更多 DAG 而被环境投喂更多新 DAG，进而承担更多 delay/energy penalty。

这是 Scheme-B RNG divergence 中已经实际观察到的机制，必须检查它是否会污染训练/评估可比性。

## H2. Reward 方向或权重/时序可能有问题

当前 reward 时间结构高度不对称：

- movement energy：当前 slot 即时；
- movement position shaping（当前 config 为 ON）：当前 slot dense；
- task delay/energy：task reward-completed 时才结算；
- completed DAG bonus：整个 DAG 真正完成时才结算。

可能问题包括：

- dense shaping 压过真正长期目标；
- reward components 相互抵消；
- reward 的高低与 completion/flowtime/energy 排名不一致；
- reward 方向对，但反馈太晚、太稀疏。

## H3. 长期 credit propagation 困难

当前训练参数约为：

- `gamma = 0.99`；
- `GAE lambda = 0.95`；
- critic EV 过去接近 0。

Scheme-B 显示重要后果经常 50 slot 左右才出现。50 slot 后 reward 的普通 discount `0.99^50` 仍约 0.605，但纯 GAE trace 的 `(γλ)^50 = (0.99*0.95)^50` 约为 0.047。

因此当 critic 本身不能很好 bootstrap 时，几十 slot 后的 TD 信号直接回传到早期动作会非常弱。

这使 credit assignment 成为合理嫌疑，但不能单独解释全部问题。

## H4. Global gradient clipping / critic->HGNN 梯度干扰

当前 baseline：

- `max_grad_norm = 0.5`；
- HGNN + movement actor + offloading actor + critic 在 baseline 中一起做 global norm clipping；
- optimizer 主要是单一 Adam 参数组（除非 actor LR scale 被显式改变）。

可能出现：critic gradient 很大 -> global clip scale 很小 -> actor/HGNN 本来较弱的梯度一起被压缩。

同时 critic EV≈0 时，value loss 仍可能通过共享 HGNN 改 representation。

好消息是代码已经记录：

- `grad_pre_clip_*`；
- `grad_post_clip_*`；
- `grad_clip_scale`；
- HGNN actor/value gradient decomposition / cosine（相关诊断路径）。

因此这一项应优先“读日志验证”，而不是盲改 clipping。

## H5. 场景/action discretization 可能让策略自然震荡

当前：

- map：500m x 500m；
- UAV speed：15m/s；
- slot：5s；
- 每次 movement step = 75m；
- UAV coverage radius = 100m。

一步移动已经是 coverage radius 的 75%，也是地图边长的 15%。对离散四方向动作而言，这个粒度可能过粗，使 movement 出现 overshoot/来回切换。

## H6. Scenario 参数存在一致性/可解释性问题

当前值得核对：

- `UE_WALK_SPEED_MEAN = 1.2m/s`，但 `UE_GM_MAX_SPEED = UE_MAX_DIST / TIME_SLOT = 3/5 = 0.6m/s`；实际更新会 clip 到 0.6m/s，参数语义不一致；
- clean queue limit 为 `CLEAN_MAX_QUEUE_PER_UAV=16`，仓库里还有旧的 `DAG_MAX_QUEUE_PER_UAV=8`，调参时容易改错变量；
- runtime `REWARD_COMPLETED_DAG_WEIGHT=8.0`，但部分 CLI help 文案仍写“baseline remains 2.0”，需要以运行时 config/checkpoint 为准；
- clean 主线与 legacy config 并存，必须避免引用旧变量解释当前实验。

## H7. 当前 workload 可能偏重，且与 active cap 强耦合

当前：

- 60 UEs；
- hotspot radius 150m，占 500x500 地图面积约 28.3%；
- base arrival prob = 0.0145/slot；
- hotspot multiplier = 2；
- DAG = 5–8 tasks。

若所有 UE 都空闲且位置近似均匀，粗略期望每 slot 约 1.1 个 DAG offer，约等于 7 个新 tasks/slot；实际 steady-state 会被 active-DAG cap 强烈压低。

这不证明 load 一定过重，但必须用实际日志检查 arrival admitted / blocked / queue / completion capacity 的平衡。

---

# 5. 当前总路线：从“可学习性地基”重新往上搭

**原则：先证明环境和基础 RL 可学习，再研究 HGNN 是否更好。**

## Stage R0 — 冻结与版本化 Scheme-B2

目标：把已经通过的 semantic CRN infrastructure 固定下来，避免后面改动污染。

动作：

- 提交 `clean_counterfactual_oracle_common_random.py`；
- 提交 `phase4_scheme_b2_common_random_completion.py`；
- 提交 `smoke_phase4_common_random.py`；
- 保存结果 JSON；
- 记录 commit hash。

完成后：

> 暂停 B3 multi-CRN、oracle dataset、Q label、actor correction。

Scheme-B2 保留为后续 reward/environment 诊断工具。

---

## Stage R1 — Environment + Reward Sanity Audit（下一阶段优先）

**先不训练。先回答环境和 reward 是否合理。**

### R1-A 场景负载审计

固定若干相同 seed / scenario，至少比较：

- random policy；
- 简单可解释 heuristic（如 nearest / load-aware / EFT，仅作为评估基准，不作为 teacher）；
- 当前 deterministic checkpoint policy（若需要）。

记录：

- generated DAG count；
- admitted arrival / blocked arrival；
- active DAG count；
- ready/pending tasks；
- UAV queue length；
- completion rate；
- flowtime；
- throughput；
- energy；
- action executed/invalid rate。

核心问题：

> 更强策略是否因为更快完成任务而系统性生成更多新 DAG，从而让“episode reward/系统指标”比较失去相同 workload 基础？

如果存在强反馈，再决定是否需要**仅用于诊断**的固定 workload / arrival tape。

### R1-B Reward component 对齐审计

利用已有日志和 Scheme-B2 counterfactual 分支，把 return 拆成：

- `G_time`；
- `G_task_energy`；
- `G_movement_energy`；
- `G_DAG_bonus`；
- `G_movement_position`；
- `G_total`。

同时对照：

- target DAG completion time；
- overall completion rate；
- DAG flowtime；
- critical-path delay；
- total/energy-per-DAG；
- queue/backlog。

### R1 判定

- 若 `G_total` 排序基本与真正关心指标一致：reward 方向暂时保留，重点转向 learnability/credit/optimization；
- 若 `G_total` 与核心系统指标明显反向或被单一分量支配：正式进入 reward redesign；
- 若最终方向一致但早期 reward 几乎无区分：属于 reward timing/density + credit propagation 联合问题。

---

# 6. Reward 方案怎么试：允许 AI 提方案，但不允许“试彩票”

师兄建议“用 AI 生成多个 reward 方案”是可行的，但必须变成**受控 reward ablation**。

正确用法：AI 只负责提出少量、理论含义明确的 reward hypotheses；实验只改 reward，不同时改环境、网络或 optimizer。

最多先设计 3 类：

### Reward-A：当前 reward（control）

完整保留当前定义，作为唯一对照组。

### Reward-B：目标对齐的 dense 方案

前提是 R1 发现当前 reward 太稀疏/太晚。可考虑把 backlog/age/DAG progress 等作为 dense signal，但必须逐项解释它与最终 flowtime/completion 的关系，不能为了让 entropy 降低而随便加分。

### Reward-C：potential-based shaping 候选

尽量用“状态进展 potential difference”提供中间信号，在标准条件下减少改变最优策略的风险。具体 potential 必须来自系统目标，而不是人工 teacher 的动作答案。

### Reward 实验规则

- 固定 MLP；
- 固定 PPO；
- 固定 seed/scenario；
- 固定训练预算；
- 每次只改 reward；
- 同时看 reward 曲线和真实系统指标；
- 不以“entropy 下降”单独判断成功；
- 不从十几个 AI reward 中挑最好看的一个。

---

# 7. Stage R2 — MLP Learnability Gate

这是后续 HGNN 实验的地基。

## 7.1 为什么先看 MLP

如果简单 MLP + PPO 在合理环境/奖励下都无法显著优于 random policy，那么 HGNN 不收敛没有解释价值。

**MLP 不收敛不能直接证明 environment 错，但说明应先诊断 environment/reward/PPO，而不是继续改 HGNN。**

## 7.2 “MLP 学会”的门槛

不要求 entropy 必须降到很低。至少应同时满足：

- 真实系统指标持续优于 random；
- reward 与系统指标方向一致；
- policy 不再长期保持完全均匀/无差别；
- 训练后期表现相对稳定；
- 多 seed 至少不出现完全相反的结论；
- critic 至少开始解释一部分 return，或有明确证据说明 actor 在 critic 较弱时仍能改善。

## 7.3 若 full MLP 仍学不会：按复杂度阶梯拆问题

一次只简化一个维度：

1. **固定 movement，只训练 offloading**；
2. 若仍不学，检查固定/更可控 arrival 或较低 workload 的诊断环境；
3. 若简化环境可学，再逐步恢复 movement、mobility、原 arrival/load；
4. 找到“从可学变成不可学”的第一层复杂性。

这比一次性大改 reward、scene、network 更有诊断价值。

---

# 8. Stage R3 — Gradient / Optimizer 审计

这一阶段优先“读已有诊断”，不是改代码。

## 8.1 Gradient clipping

当前 baseline global clip 为 0.5。需要统计整个训练过程：

- `grad_clip_scale` 分布；
- `grad_pre_clip_hgnn/movement/offloading/critic`；
- `grad_post_clip_*`。

判定：

- 若大多数 update `clip_scale≈1`：clipping 基本排除；
- 若长期频繁 `clip_scale << 1`，尤其 critic 占主导：再设计 module-wise clipping / separate optimizer 的单变量实验。

不应在没有日志证据前直接调大 max_grad_norm。

## 8.2 critic -> HGNN 梯度污染

相比“先单独预训练 HGNN”，当前更干净的实验是：

`baseline shared HGNN gradients` vs `--detach-critic-hgnn`

后者只阻断 critic value-loss 对 HGNN 的梯度，但 actor 仍然端到端通过 HGNN 学习。

如果 detach 后 HGNN/actor明显更稳定，说明 critic representation gradient 是重要问题。

---

# 9. 是否“只训练 HGNN / 先训练好 HGNN”

当前不作为主线。

原因：HGNN只是 encoder，本身没有天然训练标签。如果单独预训练，需要额外定义：

- graph reconstruction；
- critical path prediction；
- future delay/completion prediction；
- contrastive/self-supervised objective；
- 或其他任务。

这会把论文方法改成“预训练 HGNN + RL”，与原始“端到端 RL 让 HGNN 为调度学习结构表示”的主张不同。

因此：

- **主线：端到端 HGNN + PPO；**
- **诊断：可以 detach critic->HGNN；**
- **预训练 HGNN：只有在后续明确形成新的科学假设时再做，不作为当前救收敛手段。**

---

# 10. Stage R4 — HGNN End-to-End Gate

只有在 MLP learnability gate 通过后再进入。

固定：

- same environment；
- same reward；
- same PPO；
- same seeds；
- same training budget。

比较：

- MLP；
- current HGNN；
- 必要时结构变体/超边消融。

## 10.1 如果 MLP 能学，HGNN不能学

优先检查：

- HGNN 梯度尺度；
- critic->HGNN 干扰；
- oversmoothing；
- hyperedge type/normalization；
- HGNN输出是否真正进入 actor useful features。

## 10.2 如果 MLP/HGNN都能学，但效果接近

这时才讨论：

- DAG 5–8 tasks 是否太小；
- MLP raw feature 是否已经编码过多结构统计；
- workload 是否不需要具体高阶关系；
- 是否需要更结构敏感的 DAG / larger graph；
- 是否需要作为**独立研究设定**加入 UAV heterogeneity。

不能为了让 HGNN 赢而随意改变 scene；所有改变必须有论文问题上的理由，并重新做公平 MLP baseline。

---

# 11. 当前场景快速审计清单

在正式改 reward 或模型前，必须核对：

### Load / arrival
- hotspot 占比与实际 hotspot UE 数；
- eligible arrival rate；
- admitted vs blocked arrival；
- active-DAG cap 对 workload 的反馈；
- steady-state generated/completed DAG 平衡。

### Task / DAG
- DAG size 5–8 是否给 HGNN 足够结构信息；
- task compute time 与 5s slot 的尺度关系；
- communication time 是否成为绝对主导；
- queue 16 是否频繁接近 cap；
- critical/noncritical reward 权重是否合理。

### Movement
- 75m/slot 是否过粗；
- boundary-blocked action 比例；
- hover ratio；
- movement position shaping 的实际量级与频率；
- coverage radius 100m 与 movement step 75m 的关系。

### UE mobility
- 1.2m/s mean vs 0.6m/s max 的配置冲突；
- service waiting speed scale = 0.2 是否符合设定。

### Reward
- 各分量均值、std、非零频率；
- 哪个分量主导总 reward；
- component cancellation；
- reward 与 completion/flowtime/energy 的 rank correlation。

### Optimization
- critic EV；
- clip scale；
- module grad norms；
- HGNN actor/value grad cosine；
- actor logits/probability spread；
- invalid/candidate rejection 是否限制真实探索。

---

# 12. 当前明确“不做”的事情

在 R1/R2 完成前，不做：

- 不继续扩展 Scheme-B3 multi-CRN 作为 actor label 路线；
- 不构建 formal oracle dataset；
- 不再把 Q 当 teacher 推给 actor；
- 不因为 actor 不收敛就直接加入异构 UAV；
- 不先预训练 HGNN 再解释成原始端到端方法；
- 不一次同时改 reward + arrival + movement + optimizer；
- 不以 entropy 降低作为唯一成功标准；
- 不把“老师实验能收敛”解释成“RL 必须有老师”；
- 不在 MLP baseline 尚未证明可学时讨论 HGNN优劣结论。

---

# 13. 实验设计强制模板（防止路线漂移）

以后每次让 Codex 开新实验，必须先写清楚下面 6 项：

1. **Hypothesis**：这次只验证什么？
2. **Single change**：相对 control 唯一改了什么？
3. **Frozen items**：哪些环境/模型/seed/训练参数绝对不变？
4. **Primary metrics**：用什么真实系统指标判断，而不只是 loss/entropy？
5. **Pass/Fail gate**：什么结果支持/反对假设？
6. **Stop rule**：跑到哪里必须停，不允许自动进入下一阶段？

若一次实验无法用这 6 项说明，就先不要跑。

---

# 14. 当前唯一推荐下一步

**先提交 Scheme-B2，然后做 R1：Environment + Reward Sanity Audit。**

这个阶段的目的不是“马上找一个新 reward”，而是用最小成本回答：

> 1. 当前 workload / arrival 是否对不同策略公平且合理？  
> 2. 当前 reward 高低是否与我们真正想优化的 completion / flowtime / energy 同方向？  
> 3. 哪些 reward components 在时间和尺度上主导训练？

只有 R1 有结果后，才决定下一步是：

- reward redesign；
- MLP learnability pilot；
- credit/GAE/critic；
- gradient clipping；
- 或 environment complexity ladder。

---

# 15. 当前决策日志

### 2026-08-27 — 研究主线重置

**决定：** 不再默认“credit assignment 是唯一根因”。

**保留的竞争性解释：**

- reward design/timing；
- environment/load feedback；
- temporal + within-slot credit；
- critic/GAE propagation；
- gradient clipping / shared HGNN gradient conflict；
- movement/action discretization；
- representation/scenario structure不足。

**决定：** Scheme-B2 作为诊断工具冻结，不扩展成监督 teacher 主线。

**决定：** 先验证 MLP 基础可学习性，再评价 HGNN。

**决定：** HGNN 主线仍坚持端到端 RL；`detach-critic-hgnn` 可作为干净诊断，不默认采用独立 HGNN 预训练。

**决定：** AI 生成 reward 方案可以使用，但只能作为少量理论假设的生成器，必须做受控 ablation。

---

# 16. 新窗口启动协议

以后新开 ChatGPT/Codex 窗口时，第一句话可直接使用：

> 请先读取 `HyperUAV_research_master_roadmap.md`，把它作为当前研究主线的 canonical source。不要根据最近一条讨论自行改变路线。先告诉我：当前处于哪个 Stage、已冻结哪些路线、当前唯一下一步是什么。如果新证据与路线图冲突，先指出冲突并建议更新路线图，不要直接跳阶段。

每完成一个阶段，把：

- 结果摘要；
- 关键 JSON/log 路径；
- commit hash；
- Pass/Fail；
- 下一阶段决策；

追加到本文件的“决策日志”。

这样即使对话变长或换窗口，也可以从版本化文档恢复主线，而不是依赖聊天记忆。
