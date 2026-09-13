# HyperUAV 研究主线与诊断路线图（Living Research Charter）

**版本：v1.2**
**日期：2026-09-04**
**用途：作为 HyperUAV 后续所有实验、代码修改和新对话窗口的“主线锚点”。**  
**当前研究分支：** `codex/phase3ab-phase4a-research-snapshot-20260826`（历史；当前工作以本地 HEAD 为准）
**当前代码基准 commit：** 本地 HEAD `e9dbe96`（dirty，含未提交实验代码）；已冻结基准 `a950187`(Decision-Q v2) / `9b18477`(Scheme-B2)
**R0 状态：** CLOSED / PASS。Scheme-B2 strict semantic common-random 已冻结、提交并推送。  
**当前 Stage：** Phase 4A — all-legal-action 长视野 counterfactual ranking（纯 audit，未训练）
**注意：** 正式实验前须按 `AGENTS.md` §3 记录 HEAD + dirty files + 脚本版本；协作规约见仓库根 `AGENTS.md`。

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

## 2.6 Scheme-B2 strict semantic common-random 已冻结并版本化

当前结果：

- semantic key：`(future_slot, ue_id, subsystem)`；
- 覆盖 mobility / arrival / DAG generation；
- shared semantic checks：32,582；
- semantic mismatches：0；
- 未识别 environment RNG：0；
- serial/spawn：5/5 完全一致；
- 5 个 decision 中 4 个在 H100 内所有 UAV branches 都完成；
- `task_8` 的 UAV3/UAV4 到 cap 仍未完成。

Scheme-B2 已在 commit `9b1847774cb99f55d24cb99cc216d38a78cb23fc` 冻结并推送，服务器 smoke（semantic determinism / skip isolation / generation isolation / serial-spawn）全部 PASS，三项 `py_compile` 与 `git diff --check` PASS。后续只把它作为**诊断基础设施**，不扩展成 actor teacher 主线。

**重要修正：** 旧 H20 实验在 H20 内本来就 RNG 对齐，所以“旧 H20 ranking 与新 CRN H20 ranking 不同”不能证明旧 H20 被 RNG divergence 污染；它更多提示 action ranking 对未来随机 realization 可能敏感。真正能证明 H20 太短的是**同一 CRN rollout 内的 G20_CRN vs G_common 不稳定**。

## 2.7 已确认：当前环境存在 policy-dependent workload feedback

R1-A 原生环境 pilot 与 R1-A2 strict-CRN control 均显示：

`更高 DAG completion → UE 更早解除 active-DAG cap → admitted/generated DAG 更多 → blocked fraction 更低`

R1-A2 使用 Scheme-B2 semantic CRN 控制 mobility / arrival / DAG generation 的外生随机性：

- shared semantic checks：129,010；
- mismatches：0；
- unrecognized environment RNG：0；
- 6/6 paired comparisons 均保持 `completed↑、admitted/generated↑、blocked_fraction↓`；
- strict-CRN 下 admitted 增量为 +48 至 +202，completed 增量为 +49 至 +219。

因此 H1 已从“开放假设”升级为**已确认的环境结构性质**：

> 当前 clean environment 中，policy performance 会内生改变实际 admitted workload。

Native 与 strict-CRN 的效应量明显不同，说明 global RNG-stream divergence 会影响效应幅度，但不推翻该方向性结论。

该结论**不等于环境设计错误**。它意味着以后做策略公平比较时，必须区分：
- 相同 arrival mechanism；
- 相同潜在外生随机 realization；
- 实际 admitted workload 是否因策略完成速度而不同。

R1-A / R1-A2 诊断脚本与结果路径（当前尚未版本化）：
- `scripts/diagnose_r1a_environment_load_feedback.py`
- `logs/r1a_environment_load_feedback.json`
- `scripts/diagnose_r1a2_environment_load_feedback_strict_crn.py`
- `logs/r1a2_environment_load_feedback_strict_crn.json`


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

## H1. Environment/load process 策略反馈耦合 — 已确认

该项已由 R1-A + R1-A2 strict-CRN control 确认，不再作为开放根因假设。

确认的结构链为：

`DAG 完成更快 -> UE 更早释放 -> 更早重新具备 arrival eligibility -> admitted/generated workload 增加`

后续问题不是“它是否存在”，而是：
- 训练/评估时如何公平报告 policy-dependent admitted workload；
- reward 与 completion/flowtime/energy 的关系是否被该反馈放大或扭曲。

当前不因这一性质直接修改 environment。

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

## Stage R0 — 冻结与版本化 Scheme-B2 — CLOSED / PASS

目标：把已经通过的 semantic CRN infrastructure 固定下来，避免后面改动污染。

**状态：完成。最终 commit：`9b1847774cb99f55d24cb99cc216d38a78cb23fc`。**

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

### R1-A 场景负载审计 — CLOSED / PASS

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

**结果：已确认存在 policy-dependent workload feedback。**

Native R1-A：3 seeds × 3 fixed policies × 500 slots；6/6 paired comparisons 均为 completed↑、admitted/generated↑、blocked fraction↓。

R1-A2 strict-CRN：129,010 shared checks，0 mismatch，0 unrecognized RNG；6/6 comparisons 保持同方向。RNG divergence 会改变效应幅度，但不改变结论。

因此 R1-A 关闭。当前进入 R1-B。

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

### 长时实验执行规则（节省 Codex 额度）

服务器上的耗时实验默认采用“启动即退出”模式：

- Codex 只负责启动后台任务、确认一次进程已启动；
- 返回 PID、日志路径、结果路径、启动时间与 ETA；
- 禁止持续 `tail -f`、循环 `ps`、定时轮询或等待实验完成；
- 用户在 ETA 后单独发起一次结果检查；
- 若无历史 wall-clock 数据，先做短 benchmark 估算 seconds/slot 或 seconds/run，再启动正式任务并退出。

---

# 14. 当前唯一推荐下一步

**执行 R1-B：Reward component / timing audit。**

R1-A 已确认 workload 会随 policy performance 内生变化。R1-B 现在只回答：

> 1. 当前总 reward 与 completion / flowtime / energy 等真实系统目标是否方向一致？  
> 2. 哪些 reward components 在量级、非零频率和时间位置上主导训练？  
> 3. 是否存在 component cancellation、过度稀疏/延迟或 dense shaping 压过长期目标？

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

### 2026-08-27 — R0 关闭

**R0：PASS / CLOSED。**

- Scheme-B2 strict semantic CRN 已提交并推送；
- commit：`9b1847774cb99f55d24cb99cc216d38a78cb23fc`；
- 服务器 smoke 4/4 PASS；
- py_compile / git diff --check PASS；
- Scheme-B2 后续只作为诊断基础设施。

### 2026-08-27 — R1-A / R1-A2 关闭

**结论：policy-dependent workload feedback confirmed under strict CRN。**

- Native R1-A：6/6 paired comparisons 同方向；
- R1-A2：129,010 shared semantic checks，0 mismatch，0 unrecognized RNG；
- strict-CRN 下 6/6 comparisons 均为 completed↑、admitted/generated↑、blocked_fraction↓；
- RNG-stream divergence 会影响效应量，但不是该现象的根因；
- 不据此直接修改 environment；
- 当前 Stage 转入 **R1-B Reward component / timing audit**。

**工程状态：** R1-A / R1-A2 诊断脚本与 JSON 结果尚未 commit/push，后续应独立版本化，不与 reward 修改混在同一 commit。

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


---

## 状态更新 2026-09-04（v1.2，Phase 1–4A 同步）

> 本节把 charter 从 R1-B 推进到当前 Phase 4A。下方历史小节（R0/R1 等）保留不改。

### 版本现实快照（写作本节时）
- 本地 HEAD：`e9dbe96`（Document Phase 1 local counterfactual credit audit）。
- 本地工作树：dirty，约 243 个 modified/untracked（含 `config.py`、docs、诊断脚本、实验性 GAE-target 改动）。
- 已冻结基准：`a950187`(Decision-Q v2)、`9b18477`(Scheme-B2)。
- ⚠️ 状态歧义：本地含未提交的 “GAE-target minimal replacement” 与后续诊断脚本；正式实验前按 `AGENTS.md` §3 记录版本。

### 自 R1-B 以来已证实结论（decision log）
- Decision-V / Decision-GAE：critic EV≈0 → CLOSED（`c37d89b`/`a7d6bb0`）。
- Decision-Q v1→v2：v2 只改 actor advantage 缩放为 `/std(critic_targets)`（`a950187`）；数值 pathology 修复，但 entropy 仍 collapse、Q EV↑ 而 ranking 不同步 → **Q EV 不能当 ranking 可信度指标**。
- same-slot recursive bootstrap：E_πQ(next) vs true V(next) Spearman −0.209、bootstrap/target error corr 0.941 → **正式 CLOSED**。
- GAE-target minimal replacement（当前工作树中 `clean_offloading_decision_q_credit.py` 用 `compute_smdp_decision_gae` 的 targets）：Q EV 转负、spread 膨胀、ranking 不改善 → **实验性版本，非最终主线**。
- Phase 1 `C_local=r_real−r_cf`：feasibility PASS（16/27 非零、action-dependent），但**不作训练 target**。
- Boundary critic Phase 2A：held-out EV≈−0.01、corr≈0.04（只学到均值）→ 当前不可作 long-term anchor，暂停。
- Phase 2B-lite：C_local↔truth ordering 31.3%、Spearman −0.375 → 即时 counterfactual 太短视。
- Phase 3 horizon：C_H 在大 H 出现信号（H=80 ordering 77.8%），但**曲线非单调、单决策方差大、仅 3/27 CI 排除 0** → 缺的是 horizon，但尚不能直接训练。

### 当前唯一下一步
- **Phase 4A（纯 audit，不训练、不改主线）**：all-legal-action 长视野 counterfactual ranking。

### Claude 审核意见（待用户批准，spec 未冻结）
- 27 决策样本量偏小、C_H 曲线非单调 = 低功效信号 → Phase 4A 定稿前需决定**样本量**。
- 建议 audit 内顺手算 **centered counterfactual baseline**（`C_H(a)−mean_a' C_H`），零额外成本。
- `is_assignment_legal` **缺可达性约束**会改变 Phase 4A 所排的 legal action set → 需先决定是否加 `DAG_TASK_UAV_MAX_DISTANCE` 过滤，否则 audit 之后要重做。

### 未决元问题（Stage 1 收尾的真正门）
长视野 counterfactual 能否成为**低方差、随策略一致、成本可承受**的训练信号；否则回到 estimand 定义 / state sufficiency，而非再换第 6 个 target。

### Phase 4A 结果（2026-09-04，已完成，Codex 执行）
- 判决：all-legal-action 长视野 counterfactual **不能**形成稳定完整 ranking → **不具备进入 Decision-Q supervision 的条件**。
- 单-alternative 严重高估：H=80 Top-1 77.8%(2-action) → all-5-action 仅 33.3%(9/27，随机基线 20%)。
- 最优 horizon = **H=60**（非 H=80）：all-legal pairwise 66.3%、truth-resolved pair acc 74.6%、root std 低于 H=80；H=80 噪声更大、跨 seed Top-1 不稳(44/11/44%)。
- 关键证据：**oracle 自身只解析了 63/270 (23%) 的 truth pair ordering**；actual-relative credits 仅 ~13%(最优 H=60) CI 排除 0 → **目标信号本身低 SNR，非单纯 credit 估计器问题**。
- 工程门禁全过（censor=0、semantic mismatch=0、参数逐 tensor 不变、optimizer step=0、无训练、无主线改动、无提交）。
- JSON（服务器）：`/data2/zrj2025/uav-results/audits/boundary-anchored-decision-credit/phase4a_multi_alternative/formal/phase4a_multi_alternative_credit_audit.json`

### 决策门（section 17 触发）
Phase 4A 未达稳定 ranking → **停止设计第 6 个 Decision-Q target**。回到更上游三选一：credit estimand 定义 / **reward 信号密度与时延** / state sufficiency。

### 新假设（提议，待用户 + 师兄批准；触及"勿改 reward"规则，故不擅自执行）
- reward 很可能是"**对齐正确但可学习性差**"：R1-B 只验证了 alignment（reward 能否正确排序策略），没验证 learnability（actor-critic 能否在合理时间内从中抽出 per-decision 信号）。这是两种不同性质。
- 代码事实：clean 主线 reward 下，**单个 subtask 完成没有任何正向信号**（只有 time+energy 小惩罚），唯一正奖励是延迟 46–66 slot 的整 DAG 完成 (+8) → 典型 sparse-delayed，正是 critic EV≈0 / policy 不稳的上游根因。
- 提议：potential-based progress shaping（Φ = DAG 完成进度，Ng 1999，**objective-preserving，不改最优策略**）把长视野 credit 变成近即时信号。先做**纯诊断**（复用 Phase 4A 同 27 决策/8 roots/exact branches）：把 C_H 改成对 shaped/progress return 计算，看 pair 可解析率与 all-legal ranking 稳定性是否显著上升；若显著上升再议是否进训练。

### reward 密度诊断结果（2026-09-04,已完成,Claude 已独立复核 report+diff → 判 ACCEPT-as-FAIL）
- 判决:**FAIL**(按预注册门槛)。progress shaping 确实把正 reward transition 从 40%→88% 变密,但没稳定改善 all-legal credit ranking(H=60 反退化;credit CI 可解析率最好仅 ~13%→16–18%,仍 >82% 无法排除 0)。
- Codex 执行严谨:telescoping 自检误差 3.4e-13、gate-OFF 逐值复现 Phase 4A、optimizer=0、版本双记录(服务器 `a950187` / 本地 `e9dbe96`)。Claude 复核确认。
- **关键限制(Codex 与 Claude 独立同得)**:PBRS 的 discounted return 满足 telescoping → 本诊断只改了 return 的 horizon 边界项 `γ^H·ΔΦ(s_H)`,**结构上无法测试"dense reward 是否帮助训练期 critic/GAE 学习"**。它证伪的是"shaping 能救 credit-ranking 审计",**不是**"reward 密度与学习无关"。
- 综合 Phase 4A(oracle 仅解析 23% truth pair)+本结果:**证据指向"对最终延迟 return 做 per-decision 信用本身低 SNR"** → critic EV≈0 更可能因 return 内在高方差(cause b),而非 state 不足(cause a)。

### 修订后下一步建议(Claude;与 spec 预注册的"转 state/estimand"有偏差,理由如上;待用户+师兄批准)
- 证据更支持 **return 侧杠杆**,优先级:
  1. **shaped-reward 小训练探针**——最便宜,直接测 audit 测不到的 Question B:dense reward 能否让 critic EV>0、actor entropy 稳定。单变量、gate-OFF 复现、短训练。触及训练+reward → 需师兄批。
  2. **credit estimand 重定义**——把信用打到"该决策触及的 DAG 的完成结果/时延"(局部、高 SNR),而非固定 H 窗口的全局 return。
  3. state sufficiency——证据支持最弱,暂降级。

### shaped-reward 训练探针结果（2026-09-05,已完成 → FAIL）
- **primary gate 全败**:shaped-EV OFF `0.0087` → ON `-0.0316`(3/3 seed 变差);offloading entropy OFF `0.99980` / ON `0.99995`,两组都近乎完全均匀(actor 未形成策略);de-shaped-EV `0.436` 经查为假象(Φ 同时进 prediction 与 GAE target),护栏正常起作用。
- 系统指标 3/3 同向改善(completion +3.0%、flowtime −12.6%、throughput +12.4%),但**被下述残差污染,不能干净归因**。
- **Claude 更正自己的错误**:v3 spec 中"F=0 at done 只留 ~1e-9 残差"**算错了**。误用 config 的 `DISCOUNT_FACTOR=0.96`,而该 run 实际 `γ=0.99`;且低估 Φ(s_{T-1})(≈1268)。真实残差 `γ^499=0.00664 × Φ ≈ 8.42`(范围 6.28–11.10),**约等于一次 +8 的 DAG 完成奖励**。故本次并非严格 policy-invariant PBRS,末端多了一个与策略相关的进度目标。Codex 复核发现,Claude 确认。教训:**实验参数以 run 实际值为准,不能信 config**。
- **关键理论结论（本轮最大收获）**:**PBRS 是 return 守恒的(telescoping),只在 episode 内重分配奖励,不给 critic 增加任何关于长期价值的新信息。** 因此"不改目标的进度塑形"在理论上就无法拯救一个"return 本身低信噪比"的 critic。→ **reward 密度这条杠杆(在 paper-safe 的 PBRS 形式下)已关闭。**
- **三条独立证据现已收敛到同一根因**:Phase 4A(oracle 仅解析 23% 动作对)+ reward 密度审计(telescoping→排序不变)+ 本探针(PBRS→critic 不改善)⇒ **把卸载决策的信用打到"长视野全局 return"上,本质低 SNR,任何不改目标的手段都绕不过。**

### 当前唯一下一步(2026-09-05 起)
- **offloading leverage 前置检查**(纯评估,无训练):见 `docs/superpowers/specs/2026-09-05-offloading-leverage-check.md`。
- 逻辑:Phase 4A 说的是**单个决策**的 credit 低 SNR;但**整条策略**的 leverage 可以很大(单步小差异跨 500 slot 会累积),两者不矛盾。本检查测的是 **policy-level leverage**——正是论文前提所依赖的量。
- 若 policy-level leverage 大而 per-decision credit 低 SNR ⇒ 结论是"不该做 per-decision credit",而应走 EFT-guided / DAG-local estimand 这类**聚合信号**路线。

### offloading leverage 前置检查结果（2026-09-05,已完成 → PASS / LARGE LEVERAGE）
- **EFT greedy vs random**:completion **+9.23%**(19/20 seed)、avg flowtime **−46.87%**(20/20)、throughput **+65.82%**(20/20)、energy/DAG −13.99%、avg queue −18.63%、load-balance CV −38.96%。
- **greedy vs eft_worst 跨度**同样大:completion 11.01%、flowtime 45.00%、throughput 67.31%。
- 口径:forced-hover、deterministic common-seed closed-loop evaluation(非 strict semantic-tape CRN),存在 policy-induced workload feedback:greedy 平均完成 242.85 个 DAG vs random 146.45,**且吸收了更多 admitted workload**。→ 这**强化**而非削弱结论:greedy 在扛更多负载的同时把 flowtime 砍掉近一半。
- **论文前提成立**:卸载策略对系统表现有决定性影响。

### 综合结论（Stage 1 方向定案）
1. **单决策 credit 低 SNR(Phase 4A)与整条策略 leverage 极大(本检查)同时成立且不矛盾** —— 单步差异跨 500 slot 累积。
2. ⇒ **不再做 per-decision return-based credit**;改用**聚合 / 启发式锚定**信号。
3. **严峻事实**:当前训练出的 actor entropy ≈ 0.9998,行为等价于 leverage check 的 `random` 臂 —— 一个平凡贪心白送的 47% flowtime 改善,RL 系统一分未取。Stage 1 卡点严重程度首次被量化。
4. **前期"失败"转化为论文资产**:Phase 4A(oracle 仅 23% 可解析)+ PBRS 探针(信息守恒救不动 critic)正是**论证"必须采用启发式锚定信用"的方法论证据**,可写成论文的方法论章节,而非废弃弯路。
5. **leverage check 顺带提供了测量尺**:`random` 与 `eft_greedy` 两个参考点,使"学到了多少"首次可量化。

### 当前唯一下一步
**Stage-1 EFT-anchored offloading credit 短训练**:见 `docs/superpowers/specs/2026-09-06-stage1-eft-anchored-credit-probe.md`。

### Stage-1 EFT-anchored credit 短训练结果（2026-09-06,已完成 → **PASS**,commit `644227b`）
- **突破**:ON entropy `0.9998 → 0.54–0.60`(3/3 seed),**offloading actor 首次形成真实策略**;OFF 仍 `0.9998`(基线确认不学)。
- **ON vs OFF**:completion +15.6%、flowtime −57.8%、throughput +86.7%。⚠️ 此组**不可作为论文贡献数字**(OFF 等价 random 臂)。
- **ON vs EFT-greedy(论文该报的数)**:completion **+5.9%**、flowtime **−25.7%**、throughput **+18.6%**;3/3 checkpoint 三项全面超过。
- **agreement rate 76.88%** —— 学到 EFT 方向但非硬复制;约 23% 决策主动偏离老师**且结果更好** ⇒ 学到了启发式看不见的信息。
- 门禁全过:6/6 训练、400 updates/run、140/140 评估、初始参数/optimizer/RNG 逐值一致、单变量检查通过、20-seed leverage 口径校准误差 0。
- **解释边界(Codex,正确)**:本次是 **MLP + EFT teacher**;超过 greedy **不能**证明学到超图结构,需下一阶段 HGNN vs MLP 正式对照。

### ★ Stage 1 完成 ★
"建立一个 credit 可靠、真的会学的 MLP clean baseline" —— **达成**。历经 5 个 credit 机制证伪后,结论定案:**per-decision return-based credit 在本环境不可行;启发式锚定(EFT)信用是可行且公平的机制**。前期负结果(Phase 4A oracle 23%、PBRS 信息守恒)构成论文方法论章节的论证基础。

### 当前唯一下一步
**HGNN vs MLP 正式对照(论文核心命题)**:见 `docs/superpowers/specs/2026-09-06-hgnn-vs-mlp-formal-comparison.md`。

### reward-redesign 前置标定（2026-09-06,检查点触发停止,Claude 已独立复核）
- **停止原因**:forecast/collection = **48.76%** > 10% 门槛。21×500 episode 未启动(检查点按设计生效,省下大量算力)。
- **ΔΦ 分布**(seed0 / 20 slot / 91 决策,样本小):零比例 20.88%;`|ΔΦ|` p25/p50/p75 = 0.025 / **1.423** / 5.573 s;p90/p95/p99 = 8.203 / 10.319 / 34.185 s;mean positive 4.746 s。
- **Claude 复核发现的三点(Codex 未指出)**:
  1. **门槛用错了分母**。forecast 绝对耗时仅 **0.85 ms/决策**;collection 仅 8 ms/slot——环境本身极快,所以任何附加计算占 collection 的比例都会显得很大。但**训练总墙钟包含 PPO 更新/GPU 前后向**,collection 只是其中一小部分。应改用 **forecast / 总训练墙钟**重新度量,而非 forecast / collection。
  2. **cross-DAG 效应大于 target-DAG**:cross-DAG mean delta **2.617 s** > target-DAG mean delta **1.139 s**。⇒ 一个卸载决策的影响**主要落在别的 DAG 上**;**η=1(全局)是必须的**,η=0(local)会丢掉约 70% 的信号。D1 从"省成本选项"改为"证明全局必要性"的消融。
  3. **affected DAG count 均值仅 2.198**(不是预估的 ~38)⇒ `used_uavs` 过滤非常有效;global 相对 local 只贵约 2.2×,**降级到 local 并不能解决耗时问题**。
- **Claude 修正此前判断**:增量项 `ΣΔΦ` 每 DAG 量级仅约个位数秒,而 DAG flowtime 约 332 s ⇒ **regret-only 账本只覆盖了时延目标的很小一部分**。GPT 坚持"必须补初始项"是对的,Claude 此前"先跳过初始项"的判断**不成立,予以撤回**。

### reward-redesign 正式结果（2026-09-07,21/21 run 完成,Claude 裁决:**描述性,非正式验收**）
| 臂 | Completion | Flowtime | Throughput | Eval entropy |
|---|---:|---:|---:|---:|
| N0(全局 ΔΦ 解析优势) | 0.7560 | 682.3 | 0.05227 | **0.152** |
| A2(原奖励+增量时延) | **0.8228** | **416.3** | **0.07640** | 0.999 |
| B1(精简,不增量) | 0.7379 | 665.3 | 0.05173 | 0.986 |
| B2 / D1(精简,增量) | 0.7681 | 607.1 | 0.05613 | 0.845 |
| B2D(实为解析 ΔΦ 优势) | 0.7497 | 655.0 | 0.05280 | **0.126** |
| D2(λ_move=0.04) | 0.6861 | 761.9 | 0.04360 | 0.986 |
| *参考* eft_greedy | 0.841 | **331.7** | 0.0971 | — |
| *参考* Stage-1 EFT-anchored | ~0.891 | **~246** | ~0.115 | 0.55 |

**核心结论(entropy 是训练期统计,不受评估协议缺陷影响,故此条稳健)**
1. **纯改奖励(A2/B1/B2/D2)entropy 全部停在 0.845–0.999 ⇒ 卸载策略仍未学起来。** 增量奖励假设**未获支持**。
2. **ΔΦ 解析优势(N0/B2D)把 entropy 压到 0.13–0.15,系统指标反而更差** ⇒ **"策略变尖锐" ≠ "学到正确排序"**。Claude"结构感知信号应优于短视 EFT"的假设**被证伪**。
3. **没有任何臂达到 eft_greedy,更未达 Stage-1 EFT-anchored。** 迄今唯一有效机制仍是 **EFT-anchored credit**。
4. 待 20-seed 复核的假设:A2 领先很可能来自**保留了 +8 完成奖励与覆盖塑形**,而非增量时延本身;B 组删掉 +8 后系统指标全线下滑。
5. `λ_move=0.10` 把移动率压到 **0.55%**(设计目标 m_ref=0.30);`0.04` 恢复到 **35.5%** ⇒ **把 0.04 改成 0.10 是过度惩罚**。

**协议缺陷(Codex 指出,Claude 确认)**
- 每 checkpoint 仅评估 **1 个 matching-seed episode**(Stage-1 为 ×20 seeds)⇒ 方差极大,**不能作正式排名**。
- B2D 实现偏离预注册:实为 batch 标准化的 `−ΔΦ/500` **解析优势**,非逐决策 reward + variable-discount GAE + 微状态 critic ⇒ 与 N0 同机制,应更名"B2 + analytic ΔΦ advantage"。
- TensorBoard 3 个标量口径与标签语义不符;报告已从原始 JSONL 重算,结论不受影响。

**Claude 自身两处错误**
1. **D1 白跑 3 个 run**:spec 定义 D1 为 η=0(local),但 timing-followup 提示词写了"η 固定为 1",指令冲突 ⇒ D1 与 B2 逐值一致。**恰恰这一臂本该回答"跨 DAG 外部性是否被过度加权"**,信息被自己毁掉。
2. **N0 是 Claude 提的臂,已被数据证伪。**

### ⚠️ 重要更正（2026-09-07,用户指出,Claude 确认）
**A1 基线从未在可比条件下运行过。** Claude 在方案中把 A1 标为"已有,不重跑",直接复用 Stage-1 的 OFF 臂,但两者**不可比**：
- Stage-1 OFF：**100 episodes**,每 checkpoint × **20 seeds** 评估
- 本批 A2：**500 episodes**,每 checkpoint × **1 seed** 评估

⇒ **用户原始需求第 2 点(原始奖励 vs 时延改增量)实际上从未被测试过。** 这是 Claude 的方案错误。

**仍然成立的结论**(不依赖 A1)：A2 entropy=0.999 ⇒ 增量奖励未让卸载策略学起来;N0/B2D 尖锐但更差 ⇒ ΔΦ 解析优势被证伪;无任何臂达到 eft_greedy。
**已失效的结论**：任何"增量相对原始奖励改善了多少"的系统指标比较。

**同时确认:用户原始需求第 4 点(EFT 老师退火)从未运行。** 原因是 Claude 将其排在"等 A/B 结果后再定退火目标"。但 A/B 全部失败反而使该实验更有价值——EFT-anchored 已知有效(entropy 0.55、超 greedy),"撤掉老师能否维持"可直接退火回**原始奖励**回答,不依赖任何失败的奖励臂。

**修正后的下一批**：A1(500ep,3 run)+ C1(老师退火,3 run)+ N0-local(3 run)+ 全部 checkpoint 的 20-seed 补评(不重训)。

## ★★ 2026-09-07:C1(教师退火)取得决定性成功 —— Stage 1 真正完成 ★★
540 个评估 cell(27 checkpoints × 20 env seeds),配对 bootstrap。

| 配置 | Completion | Flowtime ↓ | Throughput |
|---|---:|---:|---:|
| **C1(教师退火)** | **0.918** | **182 s** | **0.1286** |
| Stage-1 EFT-anchor(仅 100ep,参考线) | 0.891 | 246 s | 0.1152 |
| A2(原奖励+增量时延) | 0.868 | 305 s | 0.1055 |
| A1(最原始奖励) | 0.848 | 367 s | 0.0922 |
| EFT-greedy | 0.841 | 332 s | 0.0971 |
| N0-local / N0 | 0.784 / 0.782 | 578 / 573 s | 0.0610 / 0.0614 |
| Random | 0.770 | 624 s | 0.0586 |

**核心结论**
1. **C1 学住了,且教师撤除后没有回弹。** entropy 退火期 `0.245` → 撤除后仅回升到 `0.317`,最后 20% 三 seed `0.351/0.379/0.332`,**未回到 ≈0.999**。
2. **C1 对 A1(同 500ep、同协议,可比)全面显著**:completion `+0.070 [0.052, 0.089]`、flowtime `−185 s [−238, −136]`、throughput `+0.0364 [0.0300, 0.0429]`,CI 均排除 0;三 seed 极稳(0.920/0.916/0.917)。
3. **flowtime 比 EFT-greedy 低 45%**(182 vs 332),且最后 40% 训练**已无教师**。
4. **用户原始需求第 2 点答案**:A2 平均优于 A1 ⇒ 增量式时延账本**有效**,但 seed 0 反转 ⇒ 存在明显 seed 敏感性。
5. **用户原始需求第 4 点答案:成立。** 教师退火是本项目迄今最有效的机制。
6. **ΔΦ 解析 actor-credit 路线正式关闭**:N0-local 与 N0 无可检测差异(所有 CI 跨 0)⇒ 去掉跨 DAG 外部性**没有**救回 ΔΦ;Claude 的"外部性被过度加权"假设**亦被证伪**。
7. **两个诊断指标同时作废**:entropy 下降 ≠ 学会(N0/N0-local/B2D entropy 0.16–0.17 但性能接近 random);**critic EV 也不能用于策略排名**(C1 late critic EV 仅 `0.010`,低于多个失败臂)。这**回溯性地印证了关闭 critic 中心路线的决定**。

**边界**:C1 超过 Stage-1 anchor,但后者仅 100 episode,**只能作参考线,不是同预算 treatment 对比**。C1 vs A1 才是干净的可比对照。

**论文叙事的关键转变**:教师只用于**前半程课程/热启动**,后 40% 完全撤除,最终策略在**纯环境奖励**下运行并继续改进,且超过教师 45%。这使"这还算 RL 吗"的质疑基本消解 —— 这是标准的 curriculum / teacher annealing,可发表。

### ★ Stage 1 完成(第二次,且这次是干净的)★
**当前唯一下一步:HGNN vs MLP 正式对照,采用 C1 配方。**
