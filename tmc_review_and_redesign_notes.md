# TMC 评估、重设计方案与当前行动顺序整理

> 说明：本文只做汇总整理与分区标注。各部分正文保持来源内容，不做改写或合并判断。

## 目录

1. [评估 A：Codex 审稿报告](#评估-a-codex-审稿报告)
2. [评估 B：附件审稿报告](#评估-b-附件审稿报告)
3. [方案 A：Codex 重设计方案](#方案-a-codex-重设计方案)
4. [方案 B：附件重设计方案](#方案-b-附件重设计方案)
5. [最新建议：当前大致执行顺序](#最新建议当前大致执行顺序)

---

## 评估 A：Codex 审稿报告

Part 1 [The Review Report]

Summary: 本文试图面向动态多 UAV DAG 卸载提出“按时隙重建任务超图 + 移动/卸载联合策略 + PPO”的框架，但当前代码和实验更像一个尚未完成的研究原型，核心的超图主张尚未被有效注入决策或实验证明。

Strengths: 第一，问题设定有实际价值：移动 UE、随机 DAG 到达、UAV 队列、跨 UAV 传输、sink 回传等要素被统一到一个较完整的仿真环境中，比只做静态 DAG 调度更接近 TMC 关注的移动边缘计算场景。

第二，`TemporaryReservationState` 将同一时隙内的多任务联合卸载从组合动作分解为顺序决策，并在每次选择后更新临时队列、可用时间和工作量。这是一个清晰且有用的系统建模贡献，解决了“所有 ready task 基于同一旧队列状态做 EFT 估计”的不一致问题。

第三，代码中已有较强的实验纪律：冻结 tape、配对评估、闭环 episode、invalid assignment 检查、baseline harness 和若干诊断报告。这些基础设施如果用于最终实验，会显著提高可复现性和统计可信度。

Weaknesses (Critical): 最核心的问题是“超图为主角”的论文定位目前没有成立。`environment/graph_builder.py` 中 DAG dependency hyperedge 是二节点父子边，本质上是普通图边；k-hop hyperedge 很容易被多层 GNN 近似；attribute similarity 是目前唯一较像高阶关系的部分；KaHyPar partition 又更像外部划分特征而非问题内生的移动卸载关系。更严重的是，当前图中没有 UAV 节点，因此无法表达代码里真正存在的高阶关系：同一 DAG 子任务共址带来的跨 UAV 传输节省，以及多个 ready task 竞争同一 UAV 队列容量的 n 元约束。若投稿声称“超图捕获 DAG 高阶依赖从而改进 UAV 卸载”，审稿人会要求看到超边基数分布、同参数 GNN 对照、逐类超边消融；当前证据不足以支撑该主张。

第二，超图表示对卸载决策的注入位置很弱。`clean_offloading_actor.py` 中同一个 task embedding 被 `np.repeat` 到该任务的所有候选 UAV 行上，因此对“选哪一架 UAV”而言，HGNN 输出在同一次候选比较中是常量。它只能间接调节 scorer 如何解释 UAV 动态特征和 pair features，不能产生候选相关的 `z_{task,uav}` 表示。若没有 encoder-zero、MLP/GNN/HGNN 同参数对照证明该常量上下文确实改变策略，超图模块很可能被审稿人视为装饰性组件。

第三，当前训练阶段与论文主张不一致。文档和代码显示 Stage 1 使用 MLP 编码器、固定 movement，并以 `-EFT regret` 作为目标；这只能学习逼近 greedy-EFT，不能证明 RL、超图或联合移动-卸载优于强启发式。B1 结果更进一步显示：当 EFT 排序几乎学到完美后，闭环 `completed_dag_count` 反而下降约 15%。这不是小的调参问题，而是说明短视 EFT 代理目标与最终 DAG 吞吐/流时目标存在冲突。完整 MAPPO、HGNN、movement actor、centralized critic 的端到端结果目前没有形成可投稿证据链。

第四，baseline 矩阵还不够 TMC。已有 random、greedy-EFT、若干 HEFT 变体和启发式诊断，但缺少至少两个关键对照：同参数普通 GNN/GAT 与 DAG scheduling/MEC 领域强基线，如 HEFT 完整版本、GNN+DRL 类方法、在线改造的相关超图调度方法。没有这些对照，即便本方法在自建环境中优于 random/greedy，也难以说明社区贡献，而只能说明工程系统可运行。

Rating: 4/10。当前工作有扎实系统原型和若干有价值的建模组件，但作为 TMC 论文，核心超图贡献、端到端方法有效性和强基线实验尚未闭环，属于有潜力但当前应拒稿的状态。

Part 2 [Strategic Advice]

问题根源：第一类问题是方法定位与实际实现错位。论文想让“任务超图”成为主贡献，但当前真正有价值的高阶结构在 task-UAV 关系中，而不是只在 task-task 图中。共址收益、资源竞争、队列容量这些移动边缘卸载的核心结构都需要 UAV 或候选对参与表示；仅在 task 之间建超边，很难解释为什么普通 GNN 不够。

第二类问题是训练目标错位。Stage 1 的 `-EFT regret` 是一个短视 imitation/proxy objective，成功学到它并不意味着闭环性能提高。B1 结果已经把这一点实证化：越接近 EFT-greedy，闭环 DAG 完成数反而越差。因此不能把 Stage 1 当成论文性能结果，只能把它写成诊断：短视 EFT 在本环境中有结构性局限，从而引出长视野 RL 或高阶资源耦合建模的必要性。

第三类问题是实验论证尚未达到顶刊要求。这不是“再多跑几个 seed”能解决的表述问题，而是 baseline 和 ablation 还没有围绕核心 claim 组织起来。TMC 审稿人会问三个问题：为什么需要超图而不是 GNN？为什么需要 RL 而不是 HEFT/greedy/queue-aware heuristic？为什么需要联合 movement 和 offloading？目前每个问题都只有部分证据。

可救性判断：超图注入弱、缺少 task-UAV 高阶关系，这是方法层面的结构性缺陷，不能靠补充现有结果弥补。必须改建模或显著降格论文主张。Stage 1 目标错位也是结构性问题，但它可以转化为正面动机：把“EFT-greedy 被证伪”写成引出最终方法的实验事实。baseline 不全、消融不全、指标选择混乱属于可在修订期内解决的问题，但工作量较大。

行动指南：首先不要再把当前 Stage 1 包装成最终方法。论文叙事应改为：先证明短视 EFT 代理目标不足，再提出长视野、资源耦合感知的策略。主指标优先使用 `completed_dag_count`、DAG flowtime、admitted backlog、energy per completed DAG；`dag_completion_rate` 只能辅助，因为 generated DAG count 会受 active-DAG cap 和策略性能影响。

其次，必须做一个最小但决定性的超图有效性验证：encoder-zero、MLP、同参数 GNN、当前 HGNN、typed HGNN 在同一冻结 tape 上配对比较，并报告超边基数分布和逐类超边消融。如果当前 HGNN 与 MLP/GNN 差异不显著，就不要投稿“任务超图是核心贡献”。

第三，建议优先实现 task-UAV 关系的轻量验证，而不是直接重写完整异构 HGNN。先在候选 pair features 中加入前瞻性共址/资源竞争特征，例如 successor output volume、successor likely co-location gain、当前 UAV 竞争强度、同 DAG 未完成任务在各 UAV 的分布。若这些手工特征能显著超过 greedy-EFT，再把它们抽象为异构 task-UAV 超图；否则说明环境里可学空间不足，继续堆 HGNN 风险很高。

第四，baseline 至少补齐 HEFT/queue-aware greedy、同参数 GNN/GAT、一个 GNN+DRL DAG 调度类方法，以及 no-movement/fixed-movement、no-reservation、no-hypergraph、no-typed-hyperedge、no-centralized-critic 等消融。所有对比必须使用同一冻结 tape、同一场景集合、配对 bootstrap 或置信区间；否则 TMC 审稿人会把性能差异归因于场景方差。

Rebuttal 策略上，应避免声称“已有超图方法优于 greedy”或“完整 MAPPO 已验证”。更稳妥的攻击面是：承认短视 EFT 不是最终目标；强调当前系统发现了 EFT 在闭环 DAG 吞吐上的局限；将最终贡献收敛到“动态在线 DAG 卸载中的高阶资源耦合建模与顺序预留决策”。如果不能在修订前完成 task-UAV 高阶建模和强 baseline，建议不要投 TMC。

---

## 评估 B：附件审稿报告

# Part 1 [The Review Report]

## Summary

本文面向多 UAV 辅助边缘计算下的在线 DAG 任务卸载，提出（i）逐时隙基于活跃任务集重建的任务超图表示与 HGNN 编码、（ii）带临时资源预留的时隙内顺序卸载分解（把 $M^N$ 联合动作降为 $N$ 次选择）、（iii）"超图编码—UAV 移动—顺序卸载—集中式价值学习"的 MAPPO 框架；实际提交的证据仅覆盖第（ii）项在固定悬停、MLP 编码器下的单阶段诊断。

## Strengths

**S1. 顺序临时预留机制在形式和实现上都是扎实的。** `TemporaryReservationState`（`environment/assignment.py:46-110`）在每次选择后立即更新队列长度、可用时间与工作量，下一个任务基于更新后的状态重算候选与 EFT，从而避免了同一时隙内多个任务基于同一份陈旧队列状态做出互相冲突的完成时间估计。更值得肯定的是估计器与执行器的逐项对齐：决策端 `estimate_offloading_candidate`（`assignment.py:295-419`）与执行端 `_build_schedule_record`（`task_execution.py:206-310`）在"入口上传 / 跨 UAV 中转 / 排队等待 / 计算 / sink 回传"五段上使用完全相同的公式与服务位置快照。同类工作中"策略看到的 EFT"与"环境真正发生的时间线"不一致是常见暗坑，本文没有这个问题。

**S2. 可复现性与统计口径的工程化程度高于本领域平均水平。** 冻结场景 tape + checkpoint/场景/采样随机数的 SHA-256、预注册判据、以 `(策略, seed, tape episode)` 为单位的配对 bootstrap、跑批期间的源码钉死与批后 `sha256sum -c` 校验（`docs/superpowers/reports/2026-08-08-...round3-results.md` §5），这套东西写进实验章节是实打实的加分项。作者对指标口径的自我纠正同样可信：拒绝用 `dag_completion_rate`（分母 `generated_dag_count` 依策略变化，实测 random 比 RL 少生成 45.12 个 DAG）、指出 `average_dag_flowtime` 只在已完成 DAG 上取平均因而存在幸存者偏差、拒绝呈现单条训练回报曲线。这些判断都是对的，而且大多数投稿不会主动说。

**S3. 消融所需的编码器阶梯已经就位。** `build_clean_task_encoder`（`marl_models/hgnn/clean_incidence.py:44-88`）以统一接口提供 MLP / mean-incidence HGNN / 对称归一化加权 HGNN / typed-gated HGNN 四档，且 typed 变体的类型权重经 softplus 后做均值归一化以保证与无类型基线的尺度可比。"无图 → 普通超图 → 区分超边类型"这条审稿人最想看的阶梯，基础设施成本已经付过了。

## Weaknesses (Critical)

**W1. 论文声称的主要贡献在实验上是空的，而且训练入口把它写死排除了。** Stage 1 唯一的训练脚本 `scripts/train_decision_ppo_bandit_gate.py` 第 95 行硬编码 `encoder_type="mlp"`，第 163 行 `env.apply_movement({})` 强制全程悬停。因此贡献 1（超图表示）与贡献 3（联合框架，含移动 actor、集中式 critic、GAE、boundary transition）没有任何对应实验，`runs/` 下也没有完整方法的产物。当前状态下摘要中三分之二的主张是无支撑的。

**W2. 超边在默认配置下退化为普通图，"高阶建模"这一核心论证站不住。** 我在你们的默认参数下跑了 200 个时隙（greedy-EFT rollout，KaHyPar 关闭以避免子进程），统计 21,220 条超边：**72.4% 的超边基数为 2**。分解来看，`E_DAG` 12,118 条全部是二节点（占总数 57%，等价普通边）；`E_khop` 8,311 条、均值基数 3.23、p90 为 5——考虑到 `DAG_MIN_TASKS=5, DAG_MAX_TASKS=8, DAG_MAX_LEVELS=4` 而 `KHOP_K=2`，k-hop 邻域已接近覆盖整个 DAG，`E_khop` 实质上是"同 DAG"指示边；真正的多节点超边只有 `E_attr`，每时隙仅 3.96 条、均值基数 19.19，而同期活跃任务均值只有 75.91——即每条属性超边覆盖约四分之一的图，在 mean 或对称归一化聚合下与全局平均池化几乎没有区别。一端是普通边、另一端是全局池化，中间缺少真正承载高阶结构的尺度。这不是表述问题，是超边构造本身的问题：`ATTRIBUTE_HYPEREDGE_CLUSTER_NUM=4` 对 76 个节点做 k-means，产物必然是巨簇。

**W3. 图中没有 UAV 节点，因此本问题里唯一真正的高阶关系无法被表达。** `CleanGraphBuilder.build` 第 148 行 `del uavs, executor` 直接丢弃 UAV 信息，超图是纯任务图。而这个问题里真正超可加、成对边表达不了的两类关系——同一 DAG 若干任务共址所节省的跨 UAV 传输（`assignment.py:373-375` 父子同址时 `continue`，收益随共址任务数呈 $O(k^2)$）、以及同时隙争抢同一架 UAV 剩余槽位的 ready-task 组（`remaining_slots`，cap=16 的 $n$ 元约束）——都是 task↔UAV 关系。当前表示无法承载它们。

需要一处**纠正内部文档的表述**：项目文档（`CLAUDE.md`、positioning doc）声称"任务嵌入对候选不变，因此编码器输出不能直接改变选哪架 UAV"。这个说法不成立。`clean_offloading_actor.py:151` 用 `np.repeat` 复制 $z_j$ 后，打分器是三层 MLP $f([z_j, x_i])$，$z_j$ 是条件输入而非常数偏置。我用同结构随机初始化 MLP 做了 2000 次数值检验：候选特征完全相同、仅更换 $z_j$，argmax 改变的概率是 **58.9%**。所以表示路径是通的。准确的缺陷陈述应该是：$z_j$ 只能对 15 维候选差异（7 维 UAV 动态 + 8 维配对）做**任务级重加权**，无法注入任何 UAV 侧结构信息——这是表达能力上界问题，不是"架构 bug"。把它写成 bug 会让读者以为改一行就能修好，实际上要改的是图的节点集合。

**W4. Stage 1 的目标函数把 greedy-EFT 设成了它自己的最优解，"RL 优于 greedy"因而逻辑不自洽。** `clean_decision_ppo_bandit.py:171-175`：`rewards = -max(EFT_a - min_legal_EFT, 0) / scale`，逐决策 bandit，无跨决策 bootstrapping。在顺序预留的状态演化下，这个目标的逐点最优策略**恰好就是** greedy-EFT。而 round3 报告称 RL 在拥堵档比 greedy 多完成 11.5 个 DAG。也就是说，报告的增益完全来自策略**没有优化好自己的目标**（`deterministic_margin20_accuracy` 仅 0.64）。审稿人会立刻问：把训练做得更好，这个增益会不会消失？在现有框架下这个问题无法回答，因为 Stage 1 没有任何机制去表达"以短期 EFT 换长期吞吐"的偏好。

**W5. 拥堵侧增益缺少最该做的那个控制基线。** 现有基线集只有 deterministic `greedy_eft`、两个 HEFT 排序变体和 uniform `random`（`environment/heuristic_policies.py:462-464`）。我在同一环境（20 个独立场景 × 200 slot，按 greedy 完成数分难/易各 10 档）补了几个对照，主指标为 `completed_dag_count`，配对 bootstrap 8000 次：

| 策略 | 均值 | Δ vs greedy | CI95 | Δ 拥堵半区 | Δ 宽松半区 |
| --- | --- | --- | --- | --- | --- |
| greedy-EFT | 111.25 | 0 | — | 0 | 0 |
| **eftband_q5**（EFT 最优 +5 s 容差带内选队列最短） | 99.00 | −12.25 | [−24.3, −0.4] | **+4.50** | −29.00 |
| eftband_q15 | 81.50 | −29.75 | [−42.5, −17.1] | −11.10 | −48.40 |
| softmax(−EFT), T=10 | 70.80 | −40.45 | [−53.9, −27.6] | −15.80 | −65.10 |
| softmax(−EFT), T=25 | 58.55 | −52.70 | [−66.0, −39.8] | −28.30 | −77.10 |
| shortest-queue | 59.50 | −51.75 | [−64.5, −39.6] | −28.50 | −75.00 |
| random | 47.70 | −63.55 | [−77.1, −50.5] | −38.60 | −88.50 |

两个结论。其一对你们有利：**温度匹配的随机化 greedy 在两个半区都大幅变差**，所以"RL 的增益只是随机扰动带来的隐式负载分散"这个解释可以排除。其二是问题所在：一个五行的启发式 `eftband_q5` 就复现了你们归因给学习的那个**签名**——拥堵侧为正、宽松侧为负、全局接近打平（你们报告 +11.5 / −17.1 / +0.17）。在把这类"EFT 容差带 + 负载均衡 tie-break"启发式扫进基线集之前，拥堵侧的 +11.5 不能归因于学习到的结构。（我的场景不是你们的冻结 tape，量级不可直接对比，但符号与不对称模式的一致性已足以构成质疑。）

**W6. 通信与能耗模型对 TMC 而言过于简化，而这直接动摇"联合通信与计算"的问题设定。** 四个具体点：(a) clean 主线的速率模型是 `rate = B · 1/(1+(d/100)²)`（`comm_model.clean_effective_rate_mbps`），没有 SNR、没有路径损耗指数、没有 A2G LoS/NLoS 概率模型；距离用 `clean_distance_2d` 只取平面分量，`config.UAV_ALTITUDE=100` 在 clean 路径中完全未被使用。仓库里那套基于 Shannon 的 `calculate_ue_uav_rate` 只服务旧路径。(b) 没有带宽争用：同一时隙上传到同一架 UAV 的多个任务各自拿满 `base_upload_bandwidth_mbps`；旧函数里的 `bandwidth_per_ue = BANDWIDTH_EDGE / num_associated_ues` 在 clean 路径没有对应实现。(c) `task_execution.py:274-284` 把传输串行接在 `uav_available_time` 之后，即传输与计算共用同一个串行资源，UAV 实际是一个服务时间等于"传输+计算"的单服务台。(a)(b)(c) 合起来意味着环境里唯一被真正竞争的资源是计算队列，通信侧既无容量约束也无干扰耦合。(d) 能耗全部是常数功率×时间（`P_UAV_COMPUTE=50 W`、`CLEAN_POWER_MOVE=100 W`），没有 CPU 频率立方模型，也没有旋翼无人机的推进功率-速度模型，因此"兼顾能耗"的目标在优化上几乎等价于"缩短时间"。

**W7. 与最直接对标工作的差异未被实验界定。** 四类超边与 HyperJet (INFOCOM'25) 的 sequence dependency / attribute similarity / attribute correlation 高度重合。你们自己识别出的两条真差异——逐时隙重建的在线超图快照、异构 task-UAV 高阶关系——前者没有任何对照实验（最直接的是：每 $T$ 个时隙才重建 vs 每时隙重建的指标曲线），后者尚未实现。

**W8. 以下属于修订期内可解决的工程/表述问题，与上面的结构性缺陷严重程度不同，但审稿人会逐条问。** `graph_builder._build_task_features` 用 `max_operation = max(当前活跃任务的 num_operation)` 做归一化，参考尺度逐时隙漂移，同一任务在不同时隙拿到不同特征值，应改用 `config` 的固定上界；任务特征第 11、12 维是 `is_ready` 与 `1 − is_ready`，完全冗余；`is_assignment_legal`（`assignment.py:127-160`）的 docstring 明写 "T7 should extend this helper with communication reachability" 且 `del service_positions`，即可行域完全不含通信可达性，UAV 移动只通过距离影响传输时间、不影响候选集合，这会显著削弱移动决策的意义；`UAV_COMPUTE_RATE_OPS_PER_SEC` 是全局常量（异构的 `config.UAV_COMPUTING_CAPACITY` 未接入 clean 主线），导致同一决策内所有候选的 `compute_time` 完全相同、零判别信息，卸载问题退化为纯排队均衡；KaHyPar 分区超边每 5 slot spawn 子进程重算，但其输入本身由 k-hop 与属性超边导出，冗余度需要以"与 attr 边的重叠率"形式报告。

## Rating

**3/10（Reject）**——按当前证据成稿投 TMC 的判断。理由一句话：论文声称的三项贡献中，超图表示与联合框架零实验支撑且被训练入口显式排除，超边在默认配置下 72.4% 退化为二节点普通边，唯一被训练的模块其目标函数的最优解恰是被对比的 greedy 基线，而拥堵侧增益可被一个五行启发式复现其定性签名。

若 roadmap 的 Phase A/B/C/D 按计划完成（异构 task-UAV 超边使高阶结构可影响逐候选打分、补齐 HEFT / 同参数量 GNN / GAT / GA-DRL 基线、完整 MAPPO 与逐类超边消融、并修正通信模型），凭现有的顺序预留机制与可复现性基础设施，**5.5–6.5（Borderline to Weak Accept）是现实目标**。要到 8 分以上，需要证明高阶结构带来的增益无法被 W5 那类简单启发式复现。

---

# Part 2 [Strategic Advice]

## 一、问题根源

W1、W2、W3 是同一个根源的三种表现：**超图被放在了一个它无法影响决策质量的位置上。** 你们的决策是"给定任务 $j$，在 5 架 UAV 中选一架"，这个决策的判别信息全部在 UAV 侧；而超图建的是纯任务图，输出的 $z_j$ 只能做任务级条件。于是超边构造被迫在"任务之间还有什么关系"里找题材，找出来的四类里两类是普通边（`E_DAG`、大部分 `E_khop`）、一类是全局池化（`E_attr`）、一类是前者的导出物（`E_part`）。**先天缺陷在于图的节点集合，不在于 HGNN 的层设计。** 顺着当前架构继续调超边类型、加注意力、换归一化，收益会一直很小，而且做完消融才会发现——那时已经花掉几个月。

W4、W5 的根源是**目标函数与所声称的贡献错位**。你们想论证的是"RL 能拿到 greedy 短视拿不到的收益"，但 Stage 1 训的是"模仿逐决策 EFT 最优"，两者的最优解方向相反。于是出现了一个尴尬的实证结构：策略越接近它的训练目标，就越接近 greedy，声称的优势就越小；现在的优势恰恰来自它没学好。这不是可以靠"多跑几个 seed"或"补一张表"解决的表述问题。

W6 的根源是**环境是为算法调试搭的，不是为 TMC 的问题设定搭的**。`1/(1+(d/100)²)` 是一个便于调参、单调、无奇点的替代函数，在工程上是合理选择；但 TMC 的审稿人第一眼看的就是系统模型章节，看到没有 SNR、没有 LoS 概率、UAV 高度不进公式、多用户不分带宽，会直接怀疑整个结论的适用性。

W8 那一组则纯粹是"没来得及做"，不涉及方法。

## 二、可救性判断

**结构性、补实验救不回来的（必须改代码/改设定）：**

- **W3（图中无 UAV 节点）**——必须改。这是 Phase B2，也是相对 HyperJet 唯一有分量的方法差异。改动范围是：图的节点集合、incidence 矩阵的类型体系、以及 actor 输入从 $z_j$ 变成 $z_{ji}$，进而 Stage 1 的 checkpoint 全部作废。这个代价必须现在付，越晚越贵。
- **W4（bandit 目标的最优解是 greedy）**——Stage 1 作为诊断工具是合格的，作为论文方法不成立。最终方法必须是有 bootstrapping 的时序 credit assignment（你们的 `clean_ppo.py` 里 GAE 与集中式 critic 已经写好，只是没跑）。论文里 Stage 1 只能出现在"训练流程"或附录，不能出现在主结果表里。
- **W6(a)(b)(c)（信道与资源模型）**——(a) 换 Shannon + A2G LoS 概率模型是纯替换工作，一两天；(b) 加带宽共享需要在 `estimate_offloading_candidate` 与 `_build_schedule_record` 两处同步改并重新校准负载，是中等改动；(c) 把传输与计算解耦成两个资源，会改变 EFT 公式与 `TemporaryReservationState` 的语义，是较大改动。(a) 必须做，(b) 强烈建议做（它同时为超图带来一类真正的资源竞争高阶关系），(c) 可以在论文中显式声明为建模假设并给出理由，未必要改。

**修订期内可解决的：**

- W1（缺实验）——纯执行问题，按 roadmap 走。
- W2（超边退化）——把 `ATTRIBUTE_HYPEREDGE_CLUSTER_NUM` 从 4 提到与活跃任务数挂钩（例如令平均基数落在 5–15）、把 `KHOP_K` 与 DAG 深度解耦、或直接删掉 `E_DAG`（它就是普通边，删了反而干净）。关键是**必须在论文里报告超边基数分布表**，这是审稿人一定会查的。
- W5（缺控制基线）——纯工程量，`heuristic_policies.py` 的 `SelectPolicy` 协议已经在了，加两个类即可。
- W7（与 HyperJet 的差异）——需要一个对照实验，不需要改架构。
- W8 全部——一到两天。

## 三、行动指南

**立刻做（本周，零训练成本，且能改变后续几个月的走向）：**

第一件是**打乱消融**（roadmap A2 的主变体），用现有 checkpoint 在冻结 tape 上重评：把任务 $i$ 的嵌入换成随机另一个任务的嵌入，数值分布不变、只破坏对应关系。注意 W3 里那条纠正——不要预期"打乱后完全不变"，$z_j$ 是有效条件输入，它一定会有影响。真正要读的判据是**影响的量级**：如果打乱后 `deterministic_margin20_accuracy` 下降不足 2pp 而闭环 `completed_dag_count` 落在噪声内，说明任务级条件所能贡献的判别力上限很低，Phase B2 的 UAV 节点改造就是必须的，而且理由是"表达能力上界"而非"架构 bug"——后一种说法在 rebuttal 里会被反驳。

第二件是**把 W5 表里的 `eftband_qδ` 族扫进基线**，$\delta \in \{2, 5, 10, 20\}$ s，tie-break 分别用队列长度、预留工作量、剩余槽位三种，在你们的冻结 tape 上按难度分层报告。这个实验的价值是双向的：如果 RL 在拥堵侧仍显著超过最好的 `eftband`，你们就有了一个真正干净的卖点（并且可以直接写进 rebuttal）；如果没超过，你们现在就知道了，而不是在完整 MAPPO 训完之后。

**架构改造（Phase B2，核心工作量）：** 在图中加入 5 个 UAV 节点，新增两类超边——共址超边（同一 DAG 中若分置会产生跨 UAV 传输的任务组 + 候选 UAV）与资源竞争超边（同时隙争抢同一架 UAV 剩余槽位的 ready-task 组 + 该 UAV），并让编码器输出逐候选嵌入 $z_{ji}$ 接到 scorer 的候选行上。论证形式可以直接借用无线超图文献（D2D 频谱分配、pilot assignment）已被领域接受的那套："多个弱效应累积成强效应，成对边说不了 A+B+C 超阈值"——你们的共址收益（$O(k^2)$ 超可加）与队列容量约束（$n$ 元）与之结构同构。**闸门要预注册**：改造后重跑打乱消融，必须出现明显下降，否则说明表示仍然没接进决策。

**基线补齐（可与架构改造并行）：** P0 是 HEFT（这个领域不带 HEFT 会被直接质疑，你们已有 `HeftUpwardRankOrderPolicy`，但需要把它做成完整的 rank_u 调度而不只是排序策略）与**同参数量普通 GNN**（把超图退化为二节点展开图，其余完全一致——这是证明超图必要性的唯一方式）。P1 是 GAT 版本与 GA-DRL 类 GNN+DRL。P2 是 HyperJet 的在线改造版（每时隙重解一次）作为强上界。

**论文写作上的降攻击面策略：** 贡献一改写为"**动态异构任务-UAV 超图**"，把"逐时隙随 active-task 集合重建"和"task↔UAV 高阶关系"两点写死，不要写笼统的"用超图"。系统模型章节必须换成 Shannon + A2G LoS 概率模型，并明确声明带宽共享与传输/计算是否解耦——如果保留串行假设，就在模型章节给出一句理由（例如单收发链路、半双工），主动交代比被审稿人发现好得多。实验章节按**负载分层**报告主结果而非只报全局均值——你们自己已经测出全局打平是拥堵侧 +11.5 与宽松侧 −17.1 相消的假象，主动分层呈现既更诚实也更有信息量。当前 outline 的 §8"当前边界"那一节的诚实是整份材料里最好的部分，改写成论文的 Limitations 章节保留，期刊审稿对这种诚实是有回报的。

**最后一条判断：** 现在最不该做的事，是在不改图节点集合的前提下继续优化 Stage 1 的指标。归一化变换（B1）值得做，因为它是廉价且可单独验证的；但它救不了 W2/W3/W4。Phase A 的三个结果到齐之前不要动架构代码这条纪律是对的，只是要确保"到齐"包括上面那个 `eftband` 基线——headroom 测试如果不含负载均衡类启发式，它给出的"有可赢空间"结论是不可靠的。

---

## 方案 A：Codex 重设计方案

我会把论文彻底从“任务超图 + RL”重设计成：

**动态异构任务-UAV超图用于在线 DAG 卸载中的高阶资源耦合决策**

核心不是证明“我用了超图”，而是证明：**在多 UAV 在线 DAG 卸载里，真正难点是 task-UAV 之间的高阶资源耦合；普通图、短视 EFT 和单任务决策都表达不充分。**

**1. 论文主线**

我会把论文的问题定义改成：

给定移动 UE 连续产生 DAG 任务，多 UAV 提供通信、排队和计算资源。每个时隙系统必须联合决定：

1. UAV 服务位置；
2. ready task 的处理顺序；
3. 每个 ready task 卸载到哪架 UAV；
4. 如何在当前任务收益和未来 DAG 完成收益之间权衡。

论文的核心判断是：

**逐任务最小化 EFT 是短视的，因为它只看当前 task 的完成时间，而不看它对后续子任务共址收益、UAV 队列竞争、DAG 完成奖励和未来 backlog 的影响。**

所以方法必须显式建模两类高阶结构：

1. **共址高阶关系**：同一 DAG 中一组相关子任务如果放到同一 UAV，可避免多次跨 UAV 传输，这不是单条边收益的简单相加。
2. **资源竞争高阶关系**：同一时隙多个 ready task 竞争同一 UAV 的剩余队列和服务能力，这是 n 元容量约束，不是 pairwise edge 能完整表达的。

这两类关系都天然涉及 UAV，所以图必须是 **task-UAV 异构超图**，不是当前的纯 task 超图。

**2. 论文贡献重新设计**

我会写成三项贡献。

**Contribution 1：动态异构任务-UAV超图建模。**  
每个时隙构建一个 active task + UAV 的异构超图。节点包括 task 节点和 UAV 节点；超边包括：

1. DAG dependency edge：父子依赖，作为基础结构；
2. co-location hyperedge：候选 UAV 与一组存在潜在共址收益的 DAG 子任务形成超边；
3. resource-competition hyperedge：同一 UAV 与当前竞争其队列容量的 ready tasks 形成超边；
4. temporal backlog hyperedge：长期积压 DAG、接近 sink 的 DAG、以及对应 UAV 资源之间的耦合关系；
5. optional attribute hyperedge：任务属性相似性，但它只能作为辅助，不再作为主角。

这能把当前最弱的地方变成论文最强的地方：不是“任务之间有高阶关系”，而是“任务和 UAV 资源之间有高阶关系”。

**Contribution 2：候选相关的超图决策表示。**  
当前 `z_j` 对所有 UAV 候选是同一个 task embedding，表达能力弱。我会改成对每个 task-UAV 候选生成：

\[
z_{j,i} = \text{HGNN}(task_j, uav_i, \mathcal{H}_t)
\]

也就是说 actor 打分不再是：

\[
f([z_j, u_i, p_{j,i}])
\]

而是：

\[
f([z_{j,i}, u_i, p_{j,i}])
\]

这样超图真的能影响“选哪架 UAV”，而不是只做任务级上下文调制。

**Contribution 3：长视野顺序卸载策略。**  
保留 `TemporaryReservationState`，这是当前代码里最扎实的东西。论文应强调：

传统一次性联合动作空间是 \(M^N\)，不可训练；独立逐任务决策又会基于旧队列状态做互相冲突的估计。本文用顺序决策 + 临时资源预留，把动作空间降为 \(N\) 次 UAV 选择，同时保持后续候选估计与先前选择一致。

但训练目标必须从 Stage 1 的 `-EFT regret` 换成真正闭环目标：

\[
r_t = \alpha \cdot \Delta \text{completed DAG}
- \beta \cdot \text{DAG flowtime}
- \gamma \cdot \text{backlog}
- \eta \cdot \text{energy}
\]

用 centralized critic + GAE 做时序 credit assignment。Stage 1 只能作为 warm-up 或 diagnostic，不能作为主实验。

**3. 方法章节结构**

我会这样组织方法：

1. **System Model**  
移动 UE、DAG 到达、UAV 位置、通信、排队、计算、sink return。这里必须把通信模型升级到 TMC 能接受的程度：Shannon rate、A2G LoS/NLoS、UAV 高度、带宽共享至少要有清楚假设。

2. **Online DAG Offloading Problem**  
定义状态、动作、约束和优化目标。重点写清楚：目标不是单 task EFT，而是长期 DAG throughput / flowtime / backlog / energy trade-off。

3. **Heterogeneous Task-UAV Hypergraph Construction**  
这是论文核心。给出节点、超边、超边基数分布、更新频率、每类超边的物理意义。

4. **Candidate-Conditioned HGNN Encoder**  
说明如何从异构超图产生 \(z_{j,i}\)，并进入 offloading scorer。

5. **Sequential Reservation-Based Policy**  
保留当前顺序预留机制，作为工程与算法贡献。

6. **Training Objective**  
完整 MAPPO，不再主打 bandit imitation。Stage 1 只放到训练初始化或附录。

**4. 实验设计**

实验必须围绕三个问题设计，而不是堆指标。

**Q1：为什么不是 greedy-EFT？**  
必须比较：

1. random；
2. greedy-EFT；
3. HEFT；
4. EFT-band + shortest-queue tie-break；
5. shortest-queue；
6. DAG-aware heuristic；
7. load-balancing heuristic；
8. 本方法。

这里 `eftband_qδ` 很关键。因为如果一个五行启发式就能复现你的收益，论文主张会被打穿。

**Q2：为什么不是普通 GNN？**  
比较：

1. MLP；
2. pairwise GNN；
3. GAT；
4. task-only HGNN；
5. heterogeneous task-UAV HGNN；
6. typed heterogeneous HGNN。

必须同参数量或报告参数量，并做配对统计。

**Q3：高阶关系到底哪类有效？**  
消融：

1. no co-location hyperedge；
2. no resource-competition hyperedge；
3. no temporal backlog hyperedge；
4. no attribute hyperedge；
5. no UAV node；
6. no TemporaryReservationState；
7. fixed UAV movement；
8. no movement actor；
9. greedy order vs learned order；
10. per-slot rebuild vs every-K-slot rebuild。

**5. 主指标**

我会把主指标改成：

1. `completed_dag_count`；
2. DAG throughput；
3. admitted incomplete backlog；
4. average DAG flowtime，但必须说明 survivor bias；
5. p90/p95 DAG flowtime；
6. energy per completed DAG；
7. invalid assignment rate；
8. average UAV queue length；
9. load balance CV。

`dag_completion_rate` 只能辅助，因为 generated DAG count 会随策略变化。

**6. 最关键的图表**

论文里我会放这些图：

1. **超边基数分布图**：证明不是一堆二节点边。
2. **拥堵/中等/宽松负载分层结果**：不要只报全局均值。
3. **greedy-EFT vs 本方法的单场景轨迹图**：展示队列演化、DAG 完成、backlog。
4. **超边消融柱状图**：证明 co-location 和 resource-competition 是主要收益来源。
5. **普通 GNN vs 异构 HGNN 曲线**：证明超图不可替代。
6. **EFT-band baseline 扫描曲线**：证明收益不是简单 tie-break 可以复现。

**7. 论文标题建议**

我会考虑这些标题方向：

1. **Heterogeneous Hypergraph Learning for Online DAG Offloading in Multi-UAV Edge Computing**
2. **Learning High-Order Task-Resource Coupling for Dynamic UAV-Assisted DAG Offloading**
3. **Beyond Greedy EFT: Hypergraph-Based Long-Horizon Scheduling for Multi-UAV Edge Computing**

我最推荐第二个，因为它不把卖点写成“用了超图”，而是写成“学习高阶任务-资源耦合”。

**8. 最不该再写的东西**

我不会再把当前论文写成：

1. “我们提出任务超图表示 DAG 依赖”；
2. “HGNN 帮助 UAV 卸载”；
3. “Stage 1 证明 RL 有效”；
4. “EFT regret 下降说明方法好”；
5. “超图 + UAV + RL 是创新组合”。

这些都会被审稿人打。

正确说法应该是：

1. greedy-EFT 是一个强但短视的局部代理；
2. 我们实验证明，仅把 EFT 学到更准不一定提升闭环 DAG 吞吐；
3. 因此需要建模跨任务、跨 UAV、跨时隙的高阶资源耦合；
4. 异构 task-UAV 超图是表达这些关系的合适结构；
5. 顺序预留机制让这种高阶表示可以落到可训练的在线决策上。

**9. 最小可投稿版本**

如果时间有限，我会做一个收敛版本：

1. 不追求复杂 movement actor，先固定 UAV 或使用简单移动策略；
2. 主打 offloading；
3. 实现 task-UAV 异构超图；
4. 补齐 HEFT、GNN、GAT、EFT-band baseline；
5. 用 20-50 个冻结场景做配对统计；
6. 证明异构 HGNN 在拥堵负载下显著优于所有启发式和普通 GNN。

这样比现在直接做完整 MAPPO 更稳。因为 TMC 更看重主张是否闭环，而不是组件是否很多。

---

## 方案 B：附件重设计方案

先说结论：**要重新设计，动的不该是网络，而是问题本身。**

你们现在的环境里，UAV 同质、通信不竞争、传输与计算共用一个串行服务台、没有截止期、任务顺序被冻结、每个 UE 最多一个活跃 DAG。把这些合起来看，它等价于"把作业分派到 5 台相同的服务器上最小化流时间"——list scheduling 在这类问题上有 $(2-1/m)$ 的近似保证，贪心天然接近最优。你们自己的三个实验已经从三个方向确认了这件事：HEFT 排序改掉 61% 的逆序对，完成数点估计 −1.2；RL 与 greedy 全局打平；我补的负载均衡 tie-break 也只在拥堵侧捞回 4.5 个。**不是方法没调好，是这个问题里没有值得学的东西。** 在这样的环境上继续做表示学习，消融表出来一定是平的。

下面是我会怎么重来。

## 一、环境层：先造出结构

五处改动，每一处都对应一个"贪心必然吃亏"的机制，而不是为了复杂而复杂。

**1. 引入截止期（最高优先级）。** 给每个 DAG 一个相对截止期，主指标改为"按时完成的 DAG 数"。这一改，greedy-EFT 的短视变成**可证明**的而不是希望性的：EFT 只看完成时间，不看松弛，于是它会把算力浪费在注定超时的任务上，也会让松弛紧的任务排在松弛松的后面。EDF/least-slack 与 EFT 之间的张力立刻产生了可学习的权衡，而"任务的紧迫度 = DAG 剩余关键路径 vs 剩余松弛"天然需要 DAG 级前瞻——正好是你们想论证 RL 优于贪心的那个理由。

**2. UAV 异构。** 把 `config.UAV_COMPUTING_CAPACITY` 真正接进 clean 主线，让 `compute_time` 在候选间有差异。现在这一维是零信息，改动成本近乎为零，收益是决策空间从"纯排队均衡"变成广义指派。

**3. 通信变成受限资源，并与计算解耦。** 每架 UAV 一个总带宽 $B_i$，同时隙对它的并发传输按并发数分摊；同时把无线收发与 CPU 拆成两条流水线，A 的传输可以与 B 的计算重叠。这两件事一起做，"联合通信与计算"才是真的——现在的模型里唯一被竞争的资源只有计算队列，标题里的 joint 是空的。副产物是：**"同时分配给 UAV $i$ 的这组任务共享 $B_i$"本身就是一条 $n$ 元约束**，成对边表达不了，这是超图最硬的一个立足点。

**4. 加入 defer / 不分配动作。** 现在只要有合法候选就必须分配，等于没有准入控制。有了截止期以后，"这个任务已经注定超时，不要浪费槽位"是一个巨大的收益来源，而它在当前动作空间里根本无法表达。

**5. 通信可达性进可行域，让移动真的有意义。** `is_assignment_legal` 里那句 `del service_positions` 必须去掉，超出通信范围的 UAV 不是合法候选。配合 Shannon + A2G LoS 概率信道（把 `UAV_ALTITUDE` 真正用起来），UAV 移动才会改变可行集而不只是微调传输时间，"移动与卸载联合"才成立。

**6.（可选，但对超图故事很有价值）共享输入数据 / 模型缓存。** 不同 DAG 的任务共用同一份模型或数据块，同址时一次传输服务全组。这让共址收益变成显式超可加，而不是靠父子传输那点二阶效应硬撑。

## 二、动手前的闸门：先量 headroom，再写方法

这是我认为整个项目最该补上的一条纪律。**把新环境缩小到可解规模**（3 架 UAV、10 个 UE、20 个时隙），用 MILP 或穷举求出离线最优，测 greedy-EFT / EDF / 最好的启发式与最优的差距。

- gap < 3%：这个配置没救，调参数（加载、收紧截止期、加大异构度）直到有 gap。
- gap > 15%：可以开工，而且论文里可以直接写"启发式距最优 X%，我们的方法回收了其中 Y%"——这是审稿人最认的表述方式。

**这个闸门要在写任何方法代码之前过。** 你们现在的困境本质上就是跳过了这一步。

## 三、表示层：异构超图，每条超边都要有物理语义

节点集合改成**任务 ∪ UAV**。这是最关键的一处结构性改动——现在 `graph_builder.build` 第 148 行 `del uavs, executor` 直接把 UAV 扔了，于是超图无论怎么设计都只能给出任务级条件。

超边只保留有 $n$ 元语义、且基数天然 ≥3 的：

| 超边 | 成员 | 为什么是高阶 |
| --- | --- | --- |
| **竞争超边** $E_{\text{cont}}(i,t)$ | 本时隙候选集包含 UAV $i$ 的全部 ready 任务 + 节点 $i$ | "这组任务的总需求不能超过 $i$ 的剩余槽位/带宽"是 $n$ 元约束，拆成两两边就丢掉了约束本身 |
| **共址超边** $E_{\text{col}}(D,i)$ | DAG $D$ 中分置会产生跨 UAV 传输的任务组 + 节点 $i$ | 共址收益随组大小呈 $O(k^2)$，超可加 |
| **松弛超边** $E_{\text{slack}}(\tau)$ | 截止期紧迫度落在同一档的任务 | "这批都很急，服务不完"是集合级的容量-需求关系 |
| **覆盖超边** $E_{\text{cov}}(i)$ | 当前在 UAV $i$ 通信范围内的 UE/任务 + 节点 $i$ | 移动一次同时改变整条超边，是移动决策的自然载体 |
| **数据超边** $E_{\text{data}}(c)$（可选） | 共享同一缓存块的任务 | 一次传输服务全组，超可加 |

删掉 `E_DAG`（它就是普通边，DAG 依赖放到一条独立的轻量 DAG-GNN 分支或作为异构图的一种边类型，不要包装成超边）；删掉 `E_part`（KaHyPar 的输入是你们自己的 k-hop 和属性边，"对自己的超边跑一遍分区器"不构成建模贡献，而且每 5 slot spawn 子进程的开销毫无回报）；`E_attr` 如果保留，簇数必须与活跃任务数挂钩，让平均基数落在 5–15，而不是现在的 4 个簇 / 基数 19（活跃任务才 76，等于四分之一张图做全局池化）。

**论文里"为什么必须是超图"的那句话，我会这么写：** 对任务 $j$ 选哪架 UAV，取决于**还有谁在抢这架 UAV**；而"抢同一架 UAV 的那组任务"就是一条超边。贪心-EFT 只看到 $i$ 当前的队列状态，看不到待决策的竞争集合——这正是它在拥堵侧吃亏的机理。这个论证的好处是**可证伪**：增益应该集中在 $|\text{ready set}|$ 相对容量大的时隙，而你们已经观测到增益确实与拥堵度相关（$r=+0.526$）。把它做成一张"增益 vs 竞争强度"的图，比任何全局均值都有说服力。

## 四、决策层：保留你们最好的资产，补两处

**顺序临时预留必须留下。** 它是你们目前唯一形式清晰、实现干净、估计器与执行器逐项对齐的机制，也是唯一能独立成为贡献二的东西。

在它之上补两件：

**增量重编码。** 加入 UAV 节点后，每完成一次预留，$i$ 的队列/可用时间就变了，时隙初的嵌入随即过期。全图重跑太贵，但只更新 5 个 UAV 节点很便宜——从它们的关联超边聚合一次消息即可。这本身就是一个干净的技术点：**顺序预留下的增量再编码**，而且它给了消融一个明确对象（开/关）。

**打分头改成逐候选。** 输入从 $[z_j \| u_i \| p_{ji}]$ 变成包含 $z_{ji}$ 的形式（cross-attention 或 task-UAV 超边的边嵌入）。顺带纠正一处：你们文档里"任务嵌入对候选不变所以编码器不能改变选择"的说法是错的——我用同结构随机 MLP 测过，仅换 $z_j$、候选特征不变时 argmax 有 58.9% 的概率改变，$z_j$ 是有效的条件输入。真正的缺陷是它**只能做任务级重加权、无法携带 UAV 侧结构信息**。这个区别在 rebuttal 里很要命，别把它写成 bug。

## 五、训练：单阶段，别再用 EFT regret

直接上时隙 MDP 上的 MAPPO + GAE，时隙内的顺序决策作为 micro-step，共享集中式 critic——你们 `clean_ppo.py` 里这些件都写好了，只是没跑过。

**奖励**：按时完成 +1、超时按归一化 tardiness 扣、能耗小权重。**绝对不要**再用 `−(EFT_a − min EFT)` 这种逐决策 bandit——它的最优解逐点等于 greedy-EFT，你拿它训出来的策略去和 greedy 比，赢了说明没学好，输了说明没学会，两头都不能写进论文。

需要 warm start 的话，就**显式地**做 HEFT/greedy 模仿预训练，并在论文里明说"我们从 HEFT 初始化，RL 在其上提升 X%"。这是干净的叙事，比现在这种"训练目标是模仿贪心、结果声称超过贪心"的结构安全得多。

## 六、实验：基线和分层，比方法本身更决定命运

基线最少这几条：random、greedy-EFT、**EFT 容差带 + 负载均衡 tie-break**（我那组 `eftband_qδ` 已经证明它能复现你们归因给学习的定性签名，不放进去这篇必被抓）、HEFT、EDF/least-slack、小规模上的 MILP 最优或 LP 下界、同参数量普通 GNN、GAT、GA-DRL 类、HyperJet 在线改造版。

消融按"结构性决策"逐项来：节点集合（纯任务图 vs 任务+UAV）、逐类超边、增量重编码开关、defer 动作开关、顺序预留 vs 同时隙并行决策（用陈旧状态）。最后这一条是你们机制贡献的直接证据，现在还没有人做过。

呈现上有一条我会当成硬规定：**主结果按负载/拥堵度分层报告，不报全局均值。** 你们自己已经测出全局打平是拥堵侧 +11.5 与宽松侧 −17.1 相消出来的假象。主动分层既更诚实，信息量也大得多；而且如果方法真的靠"看见竞争集合"起作用，分层图就是最直接的机理证据。

冻结 tape、SHA-256、预注册判据、配对 bootstrap 全部保留——这是你们相对同类投稿最稀缺的优势，写进实验章节。

## 七、代价与退路

坦率说：这个重设计会让 Stage 1 的全部结果和 checkpoint 作废。但它们本来也已经作废了——roadmap 里已经把 Stage 1 从"预训练产物"降级为"诊断工具"，而诊断结论（clip 天花板、训练量、表示路径）都已经拿到了。现在扔掉的是沉没成本，继续在旧环境上做的才是新成本。

工程量的现实估计：环境改造 3–4 周（截止期 + 异构 + 带宽竞争 + defer + 可达性），headroom 闸门 1 周，表示与决策层改造 3–4 周，基线 2–3 周（可并行），完整训练与消融 2–3 个月。和 roadmap 原本的 6–9 个月是同一量级——因为原计划里 Phase D 大概率会撞上平消融然后返工。

**如果不愿意重做环境**，还有一条诚实的退路：把它写成基准与评测论文——"多 UAV MEC 下 DAG 卸载策略的可复现基准，以及超图/GNN 方法在何种负载区间不优于调好的启发式"。你们的冻结 tape + 配对 bootstrap + 预注册判据这套东西，恰好是这类论文最核心的资产，而"在 X 区间无效、在 Y 区间有效"本身就是对社区有价值的结论。这条路风险低得多，但天花板也低一些（TMC 上大概 5–6 分）。

要不要我把这份重设计整理成 `docs/superpowers/specs/` 下的正式设计文档，按你们的日期命名约定？

---

## 最新建议：当前大致执行顺序

我建议按“先判定问题是否值得做，再改环境，再改表示，再训练”的顺序走。不要先写新的 HGNN 或完整 MAPPO，那会很容易把几个月投进一个 headroom 不足的问题里。

**第 0 阶段：冻结当前结论**

先把现有 Stage 1 定性为诊断工具，不再当论文主结果推进。

当前已经可以确定：

1. `-EFT regret` 目标不能作为最终方法目标；
2. task-only hypergraph 不足以支撑 TMC 论文；
3. 顺序临时预留机制值得保留；
4. 冻结 tape、paired bootstrap、guardrail 这些评估基础设施值得保留；
5. 当前环境太接近同质服务器 list scheduling，greedy-EFT headroom 不明确。

这一阶段不要再优化 Stage 1 checkpoint。

**第 1 阶段：做 headroom gate**

目标：先证明“这个问题里确实有 greedy 学不到的空间”。

做两个东西：

1. 在当前环境上补 `eftband_qδ` baseline：  
   `δ ∈ {2, 5, 10, 20}` 秒，在 EFT 最优附近的候选里，用 queue length / queued workload / remaining slots 做 tie-break。

2. 做小规模离线最优或近似最优：  
   缩小到例如 2-3 UAV、10-20 UE、20-50 slot，用 MILP、穷举或强启发式 lower/upper bound，比较 greedy-EFT、EDF、HEFT、eftband 与 oracle gap。

判据：

- 如果 greedy/eftband 距最优 gap < 3%-5%，当前环境不要继续写方法。
- 如果 gap > 10%-15%，说明问题有可学习空间，可以继续。
- 如果只有拥堵区有 gap，就把后续实验主线改成“负载分层”。

**第 2 阶段：改环境，而不是先改网络**

按优先级改：

1. **deadline / slack**  
   给每个 DAG 加 relative deadline。主指标改成 `on-time completed DAG count`、tardiness、completed DAG count、backlog。

2. **defer / reject 动作**  
   ready task 不再必须分配。允许策略选择暂缓或放弃明显不可救的任务。

3. **通信可达性**  
   `is_assignment_legal` 必须使用 service position。超出通信范围的 UAV 不是合法候选。这样 movement 才真正有意义。

4. **通信模型升级**  
   至少用 Shannon + A2G LoS/NLoS + UAV altitude。先不必做到极复杂，但不能再是纯 `1/(1+(d/100)^2)`。

5. **带宽竞争**  
   同一 UAV 同时服务多个传输时共享带宽。这个会自然产生 resource-competition hyperedge。

6. **UAV 异构算力**  
   接入 `config.UAV_COMPUTING_CAPACITY`，让候选 UAV 的 compute time 有真实差异。

7. **通信与计算解耦**  
   如果时间允许，把无线传输资源和 CPU 队列拆开。这个改动较大，可以放在 2.5 阶段。

第 2 阶段结束后，重新跑 headroom gate。只有确认 greedy/HEFT/eftband 仍有明显 gap，才进入方法。

**第 3 阶段：重建 baseline harness**

在新环境里先不要训练模型，先把启发式跑全：

1. random；
2. greedy-EFT；
3. EDF；
4. least-slack-first；
5. HEFT；
6. EFT-band + queue tie-break；
7. EFT-band + workload tie-break；
8. shortest-queue；
9. deadline-aware greedy；
10. 小规模 oracle / MILP upper bound。

这一步的目的不是发论文，而是防止以后模型收益被一个五行启发式吃掉。

**第 4 阶段：改图表示**

确认新环境有 headroom 后，再动图。

把 `CleanGraphBuilder` 从 task-only 改成 task-UAV heterogeneous hypergraph。

节点：

1. task nodes；
2. UAV nodes；
3. 可选 UE / data-cache nodes，先不必加。

核心超边：

1. **competition hyperedge**：同一时隙竞争 UAV `i` 的 ready tasks + UAV `i`；
2. **co-location hyperedge**：同一 DAG 中放在 UAV `i` 会产生共址收益的任务组 + UAV `i`；
3. **slack hyperedge**：deadline/slack 相近、共同紧迫的一组任务；
4. **coverage hyperedge**：UAV `i` 当前覆盖范围内的 UE/tasks + UAV `i`；
5. attribute hyperedge 只作为辅助，并控制平均基数在合理范围，例如 5-15。

删除或降级：

- `E_DAG` 不再包装成“超边贡献”，作为普通 DAG edge 或 DAG-GNN 分支；
- `E_part` 暂停，不要把 KaHyPar 作为主方法；
- `E_khop` 谨慎使用，避免变成“同 DAG 大池化”。

**第 5 阶段：改 actor 注入方式**

从当前：

\[
f([z_j, u_i, p_{ji}])
\]

改成候选相关：

\[
f([z_{j,i}, u_i, p_{ji}])
\]

也就是每个 task-UAV candidate 都要有自己的表示。可以用：

1. task-UAV hyperedge embedding；
2. task node 与 UAV node cross-attention；
3. heterogeneous HGNN 后取 task-UAV pair representation。

这一阶段必须做两个快速消融：

1. shuffle task embedding；
2. zero encoder output。

如果消融几乎不影响结果，说明图还是没接到决策上。

**第 6 阶段：保留并升级顺序临时预留**

保留 `TemporaryReservationState`，但要适配新资源：

1. queue slots；
2. CPU available time；
3. bandwidth reservation；
4. deadline/defer 状态；
5. communication reachability；
6. UAV coverage state。

然后加一个可选技术点：**增量重编码**。  
每次 reserve 后，只更新相关 UAV 节点和它关联的 competition/co-location hyperedges，不必全图重跑。这个可以成为论文里的一个方法贡献。

**第 7 阶段：训练完整 MDP，不再训练 EFT imitation**

直接用 MAPPO + GAE。

奖励建议：

\[
r_t =
+ w_c \cdot \text{on-time completed DAG}
- w_l \cdot \text{tardiness}
- w_b \cdot \text{backlog}
- w_e \cdot \text{energy}
- w_d \cdot \text{dropped/deferred penalty}
\]

Stage 1 如果保留，只作为 warm start：

- 明确写成 HEFT/greedy imitation pretraining；
- 主结果必须来自 RL fine-tuning；
- 不要再用 `-EFT regret` 当论文主目标。

**第 8 阶段：完整实验矩阵**

主表：

1. heuristics；
2. MLP；
3. same-parameter GNN；
4. GAT；
5. task-only HGNN；
6. task-UAV heterogeneous HGNN；
7. full method。

消融表：

1. no UAV nodes；
2. no competition hyperedge；
3. no co-location hyperedge；
4. no slack hyperedge；
5. no coverage hyperedge；
6. no defer；
7. no reservation；
8. no movement；
9. no incremental re-encoding；
10. per-slot rebuild vs every-K-slot rebuild。

报告方式：

- 必须分拥堵/中等/宽松；
- 必须分 tight-deadline / loose-deadline；
- 所有比较用 frozen tape + paired bootstrap；
- 主指标用 on-time completed DAG、completed DAG count、tardiness、backlog、energy per completed DAG；
- `dag_completion_rate` 只做辅助。

**最现实的执行顺序**

如果按周排：

1. **第 1 周**：补 `eftband_qδ`，做当前环境 headroom gate。
2. **第 2-4 周**：改环境：deadline、defer、通信可达性、UAV 异构、带宽竞争。
3. **第 5 周**：新环境 headroom gate + heuristic baseline 全跑。
4. **第 6-8 周**：实现 task-UAV hetero hypergraph + candidate representation。
5. **第 9 周**：encoder zero/shuffle、MLP/GNN/HGNN smoke 与小规模评估。
6. **第 10-12 周**：跑 MAPPO + GAE，修 critic、reward scale、训练稳定性。
7. **第 13-16 周**：完整 baseline、消融、负载分层、论文图表。

**一句话优先级**

先做 `eftband + oracle/headroom`，再改环境制造真实决策张力，再做 task-UAV 异构超图，最后才训练 MAPPO。现在最不该做的是继续优化现有 Stage 1 或在 task-only 超图上堆网络。

---

## 补充：两个重设计方案的相同之处与不同之处

> 说明：本节用于单独整理“方案 A：Codex 重设计方案”和“方案 B：附件重设计方案”的对应关系。

### 相同之处

1. **都认为必须重设计，而不是修补当前 task-only HGNN。**  
   方案 A 认为当前纯 task 超图无法表达真正有价值的 task-UAV 高阶关系；方案 B 也指出当前 `graph_builder.build` 直接 `del uavs, executor`，导致超图只能输出任务级条件，问题根在节点集合。

2. **都认为当前最有价值的机制是顺序临时预留。**  
   方案 A 建议保留 `TemporaryReservationState`，把它作为贡献之一；方案 B 也明确说“顺序临时预留必须留下”，认为它是目前唯一形式清晰、实现干净、估计器与执行器对齐的机制。

3. **都认为 Stage 1 的 `-EFT regret` 不能作为主方法。**  
   方案 A 认为 Stage 1 只能学 greedy-EFT，不能证明 RL 优于 greedy；方案 B 进一步指出该目标的最优解就是 greedy-EFT，赢了说明没学好，输了说明没学会，两头都不能作为主论文结果。

4. **都认为最终训练应走真正的 MDP / MAPPO + GAE。**  
   方案 A 建议用 centralized critic + GAE 做长期 DAG throughput / flowtime / backlog 优化；方案 B 也建议直接上时隙 MDP，时隙内顺序决策作为 micro-step，共享 centralized critic。

5. **都认为必须加入 task-UAV 异构超图。**  
   方案 A 建议节点包括 task 与 UAV，输出候选相关表示 `z_{j,i}`；方案 B 同样要求节点集合改成 task + UAV，并让打分头从 `[z_j, u_i, p_{ji}]` 变成包含 `z_{ji}` 的逐候选输入。

6. **都认为超边必须有物理意义，不能再包装普通边。**  
   方案 A 建议重点建 co-location hyperedge、resource-competition hyperedge、temporal backlog hyperedge；方案 B 要求保留天然 n 元关系：竞争超边、共址超边、slack 超边、覆盖超边、可选数据缓存超边，并建议删除 `E_DAG` 和 `E_part`。

7. **都认为 baseline 必须补强，尤其是启发式。**  
   方案 A 建议加入 HEFT、GNN/GAT、GNN+DRL、HyperJet 在线改造、queue-aware heuristic；方案 B 更具体地要求加入 `EFT 容差带 + 负载均衡 tie-break`，也就是 `eftband_qδ`。

8. **都认为结果必须按负载/拥堵程度分层报告。**  
   方案 A 建议不要只报全局均值，应看拥堵/中等/宽松负载；方案 B 更强硬地要求主结果必须按负载/拥堵度分层，因为全局打平可能掩盖拥堵侧收益和宽松侧损失相抵。

9. **都认为 TMC 需要更可信的通信模型。**  
   方案 A 提到系统模型要升级到 Shannon rate、A2G LoS/NLoS、UAV 高度、带宽共享；方案 B 也要求 Shannon + A2G LoS、通信可达性、带宽竞争、通信与计算解耦。

### 不同之处

1. **最大差异：方案 A 主要重设方法，方案 B 先重设问题环境。**  
   方案 A 默认当前问题还有可救空间，只要把表示从 task-only 改成 task-UAV 异构超图，并把训练从 EFT imitation 改成长视野 RL，就能形成论文。  
   方案 B 认为当前环境本身太容易，接近同质服务器 list scheduling：UAV 同质、通信不竞争、传输与计算共用串行服务台、没有 deadline、任务顺序被冻结、每个 UE 最多一个 active DAG。它认为继续在这个环境上做表示学习，消融大概率是平的。

2. **方案 B 把 deadline / slack 放到最高优先级，方案 A 没有。**  
   方案 A 的主指标偏 DAG throughput、flowtime、backlog、energy。  
   方案 B 认为必须引入 DAG relative deadline，把主指标改成按时完成 DAG 数；有 deadline 后，EFT、EDF、least-slack 之间才产生真实张力，RL 才有必要学习“救谁、放弃谁、先服务谁”。

3. **方案 B 强调 defer / reject 动作，方案 A 没有重点展开。**  
   方案 A 保留 ready task 顺序卸载，但没有把 defer 作为核心动作。  
   方案 B 认为当前“只要有合法 UAV 就必须分配”等于没有 admission control。加入 deadline 后，defer/reject 可以避免把资源浪费在注定超时的任务上，是制造长期决策收益的重要来源。

4. **方案 B 要求先做 headroom gate，方案 A 更偏直接设计方法。**  
   方案 B 要求在小规模环境中用 MILP、穷举或强上下界比较 greedy/EDF/HEFT/eftband 与离线最优的 gap：gap 太小就先调环境，不写新方法。  
   方案 A 也要求强 baseline，但没有把 MILP/headroom gate 放到方法设计之前作为硬门槛。

5. **方案 B 对通信资源建模更具体。**  
   方案 A 提到通信模型应升级。  
   方案 B 具体要求每架 UAV 有总带宽 `B_i`、并发传输共享带宽、无线收发与 CPU 拆成两条流水线、通信可达性进入合法候选集、movement 改变可行集。

6. **方案 B 强调 UAV 异构算力，方案 A 没有把它放核心。**  
   方案 B 要求接入 `config.UAV_COMPUTING_CAPACITY`，让 `compute_time` 在候选 UAV 间产生真实差异，从而把问题从纯排队均衡变成广义指派。  
   方案 A 主要强调 task-UAV 高阶关系，没有把异构算力列为核心环境改动。

7. **方案 B 对旧超边设计更激进。**  
   方案 A 建议将 `E_DAG` 降级为基础结构，不再作为高阶贡献；attribute 可作为辅助。  
   方案 B 建议直接删除 `E_DAG` 和 `E_part`，`E_attr` 若保留也必须控制平均基数，避免变成全局池化。

8. **方案 B 新增“增量重编码”作为潜在技术点。**  
   方案 A 提出候选相关 `z_{j,i}`，但没有展开顺序预留后 embedding 如何更新。  
   方案 B 指出加入 UAV 节点后，每次 reserve 都会改变 UAV 队列/可用时间，时隙初始 embedding 会过期，因此建议只更新相关 UAV 节点及其关联超边，形成“顺序预留下的增量重编码”。

9. **方案 B 提供一条退路：基准与评测论文。**  
   方案 A 只围绕重设计主论文。  
   方案 B 提出如果不愿重做环境，可以把工作转成 benchmark/evaluation paper：多 UAV MEC 中 DAG 卸载策略的可复现基准，以及超图/GNN 方法在哪些负载区间不优于调好的启发式。

10. **方案 A 更强调异构 task-UAV 超图本身作为论文主角；方案 B 认为必须先让问题具备学习空间。**  
    方案 A 的中心是“学习高阶任务-资源耦合”。  
    方案 B 的中心是“先改环境制造真实决策张力，再让异构超图和 RL 上场”。

### 综合判断

如果目标是继续沿当前代码快速收敛一个方法原型，方案 A 更直接：把 task-only 图改为 task-UAV 异构超图，改 actor 注入方式，改训练目标，再补 baseline。

如果目标是真正冲 TMC，方案 B 更稳：先用 deadline、defer、通信竞争、UAV 异构、通信可达性和 headroom gate 证明问题里确实有 greedy 学不到的结构，再设计 task-UAV 异构超图和 MAPPO。

综合两者后的优先级是：

1. 先做 `eftband + oracle/headroom`；
2. 再改环境制造真实决策张力；
3. 然后做 task-UAV 异构超图；
4. 再做候选相关 `z_{j,i}` 和增量重编码；
5. 最后训练 MAPPO + GAE 并补完整 baseline / 消融。
