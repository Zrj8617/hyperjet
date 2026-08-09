# HyperUAV 当前论文思路（精简版）

## 1. 研究问题

本文研究动态热点场景下的多 UAV 辅助边缘计算。移动 UE 持续产生具有依赖关系的 DAG 任务，多架 UAV 同时承担数据传输、任务排队与计算。目标是在有限队列和计算资源下，联合决定 UAV 服务位置与 DAG 子任务卸载位置，以提高完成的 DAG 数量、降低 DAG 完成时延，并兼顾通信、计算和移动能耗。

本文的重点不是一般的“超图 + UAV + 强化学习”组合，而是解决两个具体问题：

1. 如何表示多个 DAG 子任务之间的依赖、局部高阶关系与资源关联；
2. 如何将一个时隙内指数规模的多任务联合分配，转化为可训练且资源估计自洽的顺序决策。

## 2. 系统模型

系统包含 5 架 UAV、60 个移动 UE 和一个固定于 episode 内的热点区域。地图大小为 $500\,\mathrm{m}\times500\,\mathrm{m}$，每个物理时隙为 5 s。热点只改变 UE 的 DAG 到达概率；UE 按移动模型更新位置。每个 UE 同时最多保留一个未完成 DAG。

每个 DAG 由若干具有有向依赖关系的子任务组成。子任务依次经历依赖等待、就绪、输入传输、UAV 排队、计算、结果传输和完成状态。入口任务从 UE 上传输入；非入口任务需要获取前驱任务输出；若父子任务位于不同 UAV，则产生 UAV 间传输；sink 任务完成后还需将结果返回 UE，之后整个 DAG 才算完成。

每架 UAV 具有位置、有限队列和可用计算时间。当前 clean 主线采用统一计算速率，单 UAV 队列上限为 16。每个时隙的联合动作由两部分构成：

- UAV 移动动作：`hover / +x / -x / +y / -y`；
- 对冻结 ready-task 集合中的每个任务选择一个合法 UAV。

环境奖励由任务增量时延惩罚、任务能耗惩罚、UAV 移动能耗惩罚和 DAG 完成奖励组成。论文的主要性能指标应以 `completed_dag_count`、DAG flowtime 和积压为主，能耗作为辅助指标。

## 3. 任务超图模型

每个时隙以所有 active unfinished tasks 为节点，构建任务超图快照。当前 clean 代码包含四类关系：

- DAG dependency：直接父子依赖，当前实现为二节点超边；
- k-hop dependency：同一 DAG 内的多跳局部任务组；
- attribute similarity：输入量、输出量、计算量和通信属性相似的任务组；
- partition hyperedge：基于 k-hop 与属性超边的 KaHyPar 分区结果。

节点使用 12 维任务特征，包括上下行带宽、输入/输出数据量、计算量、拓扑层级、入口/出口标志、父子数量以及 ready/pending 状态。由关联矩阵 $H_t$ 表示节点与超边关系，再通过 incidence-based HGNN 执行“节点—超边—节点”消息传播，得到任务嵌入：

$$
Z_t=\operatorname{HGNN}(X_t,H_t).
$$

代码同时支持 MLP、普通 incidence HGNN、标准加权 HGNN和 typed-gated HGNN，便于完成“无图—普通超图—区分超边类型”的消融比较。

## 4. 算法框架

整体采用集中训练、分散执行的联合策略。一个时隙内严格按以下顺序运行：

### 4.1 状态准备与超图编码

先更新 UE 位置和 DAG 到达，刷新任务生命周期，然后冻结本时隙 ready-task 集合。基于 active tasks 构建 `GraphSnapshot`，HGNN 每个时隙仅前向一次，得到固定的任务嵌入。

### 4.2 UAV 移动决策

共享 movement actor 使用 UAV 自身状态，并分别对 ready-task 与 pending-task 嵌入做注意力聚合，为每架 UAV 输出五个移动动作。边界 mask 直接作用于动作 logits。移动完成后的 UAV service position 用于本时隙全部通信和卸载估计。

### 4.3 顺序卸载与临时资源预留

冻结的 ready tasks 按稳定键顺序处理。对任务 $j$ 和候选 UAV $i$，actor 输入由以下部分拼接：

$$
[z_j\,\|\,u_i^{\mathrm{dyn}}\,\|\,p_{ji}],
$$

其中 $z_j$ 是任务嵌入，$u_i^{\mathrm{dyn}}$ 是 7 维动态 UAV 状态，$p_{ji}$ 是 8 维任务—UAV 配对特征，包含传输、排队、计算、预计完成时延及相应能耗。非法或队列已满的候选通过 mask 排除。

核心机制是 `TemporaryReservationState`：每选择一个 UAV，就立即在临时状态中更新队列长度、预计可用时间和工作量，再为下一个任务重新计算候选特征。这样将原本 $M^N$ 的联合分配动作分解为 $N$ 次 UAV 选择，同时避免多个任务都基于同一份旧队列状态做出相互冲突的 EFT 估计。全部决策结束后再统一提交 executor。

### 4.4 集中式价值学习

集中式 critic 使用全局任务嵌入汇总、各 UAV 状态、ready/pending 数量和队列统计估计 $V(s_t)$。完整方法拟使用 PPO 与 GAE 联合更新 HGNN、movement actor、offloading actor 和 critic：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),
$$

再由 GAE 构造 advantage。执行阶段只需要共享 actor 和当前局部/任务上下文。

## 5. 当前训练路线

当前实现采用“先验证卸载排序，再进入完整联合策略”的分阶段路线：

1. **Stage 1：卸载排序预训练/诊断。** 固定 UAV 移动，使用 MLP 任务编码器和决策级 PPO，以负 EFT regret 训练共享候选 scorer。该阶段只验证 actor 能否学会近似 greedy-EFT 排序，不代表最终超图方法，也不能证明 RL 优于 greedy。
2. **Stage 1 闭环复评。** 在冻结 tape 上比较 deterministic、不同温度 sampled policy，并用 checkpoint、场景和采样随机数的稳定 SHA-256 保证可复现。当前正在执行加长训练，以区分“训练强度不足”和“pair feature 归一化截断”两种原因。
3. **完整方法。** 只有 Stage 1 门禁明确后，再将任务编码器替换为 HGNN，引入 movement actor、集中式 critic、boundary transition 和 GAE，训练完整 MAPPO。当前尚未授权或完成这一阶段。

## 6. 预期贡献

论文可将贡献归纳为三点：

1. 面向动态多 UAV DAG 卸载，构建随 active-task 集合更新的任务超图表示，并区分依赖、局部高阶关系、属性相似和分区关系；
2. 提出带临时资源预留的时隙内顺序卸载机制，将组合动作空间分解为线性决策序列，同时保持候选 EFT 与先前决策一致；
3. 构建“超图编码—UAV 移动—顺序卸载—集中式价值学习”的联合框架，并通过冻结场景 tape 进行可复现的配对策略比较。

## 7. 实验设计

主要基线建议包括 random、greedy-EFT、无图 MLP、普通 GNN/基础 HGNN以及 typed-gated HGNN。核心消融包括：去除超图、去除不同超边类型、去除临时预留、固定 UAV 移动和不同负载水平。

报告指标优先级为：完成 DAG 数、DAG flowtime、未完成 DAG/队列积压、完成子任务数、总奖励和能耗。不同方法应使用同一冻结 tape，以 `policy + seed + tape episode` 为比较单位做配对 bootstrap；由于动作会改变后续状态，闭环结果只能称为匹配初始场景，而不能称为逐决策严格反事实。

## 8. 当前边界与论文表述

- 当前正式 Stage 1 checkpoint 使用的是 MLP，不是超图编码器；
- Stage 1 的目标是模仿 greedy-EFT 排序，其上界就是 greedy，论文最终优势必须来自 DAG 级前瞻、移动决策或更有效的结构表示；
- 当前 40 s 的 queue-wait/incremental-delay 归一化存在明显饱和，尚在区分训练不足与特征天花板；
- 当前代码中的 DAG dependency 超边是二节点关系，因此不能单独作为“高阶建模”贡献，超图有效性必须依靠真实多节点超边及严格消融证明；
- 在完整 MAPPO 和 HGNN 实验完成前，不应宣称联合强化学习或超图方法已经优于 greedy。
