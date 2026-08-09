# 超图为什么没起作用：注入点分析、文献定位与三条路线

日期：2026-08-07
针对问题：(1) 没找到高质量的"超图+UAV+RL"文章 (2) 高阶关系是不是从代码里来的、之前加超边为什么没成功 (3) 要不要异构 UAV、要不要加 UAV 节点

---

## 1. 最重要的发现：超图在当前架构下**结构性地无法影响卸载决策**

这可能就是你"之前试过但没成功"的原因，而且它跟超边怎么设计无关。

### 1.1 证据

`marl_models/mappo/clean_offloading_actor.py:148-156`（诊断副本在 `stage1_temperature_diagnostic.py:188-189`）：

```python
task_embedding_np = task_embeddings_tensor[int(task_idx)].detach().cpu().numpy().reshape(1, -1)
candidate_features_np = np.concatenate(
    [
        np.repeat(task_embedding_np, dynamic_features_np.shape[0], axis=0),   # ← 逐行复制
        dynamic_features_np,   # 7 维
        pair_features_np,      # 8 维
    ],
    axis=1,
)
```

**同一个任务的所有候选行，前 64 维（task embedding）完全相同。**

编码器（无论是 MLP 还是 HGNN）的全部输出，在一次决策内对所有候选是一个**常数向量**。

### 1.2 后果

scorer 打分是 `score_j = f(embed, dynamic_j, pair_j)`。因为 `embed` 对 j 不变：

- **超图不能直接改变"选哪架 UAV"**。它只能作为上下文，非线性地改变
  "对这个任务，应该更看重队列长度还是更看重距离"——即调制那 15 维的加权方式。
- 更糟的是，**决策顺序也不受策略控制**：`_ready_sort_key`（`assignment.py:416-428`）
  按 `(DAG 到达时间, dag_id, 拓扑序号, task_id)` 冻结排序。
  所以更好的任务表示连"先处理哪个任务"都改变不了。

也就是说，在当前设计里，超图的**唯一**作用面是"任务条件化的 UAV 特征加权"。
这个作用面非常窄，能贡献的性能提升自然也非常有限。

> **加超边没效果，最可能不是超边设计不好，是注入点不对。**

这个解释是可证伪的：把 encoder 输出整体乘以 0（等价于完全不用图），
再跑一次 Stage 1，如果 margin20 准确率几乎不变，就证实了上面的判断。
这个实验几乎零成本，**建议作为下一步的第 0 项**。

### 1.3 这也解释了为什么 Stage 1 用 `mlp` 编码器就能到 96–98%

`FROZEN_CHECKPOINTS` 对应的 checkpoint 强制 `cfg["encoder"] == "mlp"`
（`stage1_temperature_diagnostic.py:160-161`），也就是**完全不用任何图结构**，
未饱和区照样 96–98%。

这不是巧合：EFT 排序的信息本来就全在那 15 维里（`incremental_delay` 是精确充分统计量），
图/超图对这个任务**本来就没有信息可加**。

---

## 2. 我提的两类高阶关系确实来自你的代码

不是我凭空设计的，两处机制在 simulator 里都已经实现了，只是没进入表示。

### 2.1 共址（co-location）：`assignment.py:347-372`

```python
for parent_id in task.predecessors:
    ...
    if int(parent.assigned_uav) == int(uav_id):
        continue                      # ← 父任务同址，传输时间直接跳过，为 0
    ...
    transfer_time += parent_transfer_time
```

**父子同址传输代价为零。** 于是把一个 DAG 里 k 个相关任务放在同一架 UAV 上，
省下的传输是 O(k²) 级的（所有跨机对都消失），而不是 k 条两两边的简单相加。

**这就是超可加性（superadditivity），是超图立论的教科书形态**：
"三个任务放一起省的传输 > 三条两两边之和"，成对图无法表达这个收益结构。

### 2.2 资源竞争：`assignment.py:47-109` + `:129-155`

`TemporaryReservationState` 在时隙内逐个决策地更新
`queue_lengths` / `available_times` / `queued_workloads`，
而 `is_assignment_legal` 用 `remaining_slots(uav_id) > 0`（cap = 16）判定合法性。

同一时隙内争抢同一架 UAV 的那一组 ready task 构成一个 **n 元约束**：
"这一组的总量不能超过剩余槽位"。拆成两两边会丢掉"总量超限"这个约束本身。

### 2.3 但是——这两类关系都是 task↔UAV 关系

**而你的图里没有 UAV 节点。** 所以在当前图结构下，这两类超边
**在数学上无法表达**。这不是实现难度问题，是表达能力问题。

这直接引出第 4 节。

---

## 3. 文献定位：空白是真的，但比你想的窄

我检索了三个方向，结论是**每一条边都不空白，只有三者交叉可能薄**。

| 方向 | 状态 | 对你的意义 |
| --- | --- | --- |
| **GNN + DRL 做 DAG 调度 / MEC 卸载** | **成熟**。GA-DRL（GNN 增强 DRL 做车载云 DAG 调度）、Edge Generation Scheduling（IEEE TC）、task graph offloading via DRL（FGCS） | **这些就是你的直接竞争基线**。审稿人会要求你和 GNN 版本对比 |
| **超图用于无线资源分配** | **成熟**，但多为经典超图着色而非 HGNN。D2D 频谱分配、pilot assignment 都用超图建模累积干扰 | 立论模板已经现成 |
| **HGNN + 多智能体决策** | **已有**。"Pairwise is Not Enough: HGNN for Multi-Agent Pathfinding"、IJCAI 2024 多通道超图上的分层 RL | 说明 HGNN+MARL 这条路本身是通的 |

### 3.1 无线超图文献给了你现成的论证模板

这个领域用超图的**标准理由**是：

> 多个单独很弱的干扰源叠加起来构成强干扰，
> 成对图只能表达"A 干扰 B"，无法表达"A+B+C 合起来超过阈值"。

**结构和你的共址收益完全同构**：单独看每条父子边收益很小，
但一组任务共址的总收益是超可加的，成对边表达不了。

**建议直接借用这个论证形式**——它在这个领域已经被接受了，你不需要重新说服审稿人
"为什么超图"，只需要说服"为什么在你的问题里也成立"。

### 3.2 关于"没找到高质量文章"

空白有三种可能，需要分清：

1. **真的没人做** —— 机会
2. **没有好理由，所以好研究者没做** —— 陷阱
3. **换了术语已经做了** —— 风险（例如"高阶交互建模"、"集合函数"、"注意力集合编码"）

从检索结果看，你这个交叉更接近 1 和 2 之间。
**关键是：审稿人不会因为"没人做过"给分，只会问"为什么需要"。**
所以空白本身不是护城河，§2.1 那个超可加性论证才是。

---

## 4. 关于异构 UAV 和 UAV 节点

### 4.1 异构算力：我收回这个建议，你的反驳是对的

理由：

- 它给论文加了一个**必须单独论证和消融**的维度，而你的主题不是异构性；
- 我提它只是为了制造候选间的判别信号，但**更好的信号来源是 DAG 结构**
  （共址 + DAG 级优先级），不需要引入新实体。

保持同质算力。（顺带说明：`config.py:382` 有现成的 `UAV_COMPUTING_CAPACITY`
异构参数，但 clean 主线没用；只有 `environment/uavs.py` 等旧路径在用。
如果将来要开，改动很小——但现在不要开。）

### 4.2 UAV 节点：这个躲不掉，但可以先不建图

你担心得对——加 UAV 节点会显著提高复杂度。但如 §2.3 所述，
**没有 UAV 节点，我提的两类超边根本无法表达。**

三条路线，成本递增。**强烈建议先走 A：**

#### 路线 A：不动图，用手工候选特征验证机制（最便宜，先做这个）

在现有 15 维判别特征上加 1–2 维**前瞻性**特征，直接注入到能影响排序的位置：

- `pending_successor_output_mb`：该任务尚未调度的后继任务的输出数据总量
  （衡量"如果后继放在别处，未来会付多少传输代价"）
- `predecessors_on_uav_j` / `successors_likely_on_uav_j`：候选相关的共址程度

注意一个关键点：**当前任务自己的共址收益已经在 EFT 里了**
（`assignment.py:354-355` 已经算进 `transfer_time`），所以 greedy-EFT 不缺这部分。
**greedy 缺的是未来收益**——它不知道"把这个任务放 j 上，会让它的后继也倾向于放 j，从而省下更多"。
所以特征必须是**前瞻的**，否则和 greedy 完全等价，加了也没用。

判据：加了这些特征之后，策略能否在冻结 tape 上**配对显著地超过 greedy-EFT**。

- **能** → 机制真实，值得投入建超图（超图的作用就是把手工特征换成学出来的表示）
- **不能** → 超图也不会有增益，**省下几个月**

#### 路线 B：加 UAV 节点，让超边能表达 task-UAV 关系

`NUM_UAVS=5`，相对活跃任务节点（可达 480）计算量可忽略，
主要成本在代码（incidence 构造、异质节点类型、embedding 拼接口径）。
这是超图能起作用的**最低要求**。

#### 路线 C：完全重做为 task-UAV 二部超图 + 竞争/共址超边

最贵，只有在 A 验证机制成立、B 跑通之后才值得。

---

## 5. 关于"先收敛再一个一个加超边"

**方向完全正确**，这也是你一贯的 gate 纪律。但"收敛"的定义需要改：

**不要等 Stage 1 到 100%——它已经收敛了**（未饱和区 96–98%，30 个 update 达成）。
而且 Stage 1 的目标就是 greedy 本身，做到 100% 也不产出论文结果。

正确的递进顺序（每步都有预声明 gate，失败就停）：

| 步 | 动作 | 通过条件 | 失败意味着 |
| --- | --- | --- | --- |
| **0** | 把 encoder 输出置零重跑 Stage 1 | 准确率明显下降 | 若几乎不变 → 证实注入点问题，图当前无用 |
| **1** | 实测当前配置的 ρ 与队列演化 | 队列走平 | ρ>1 → 先修工作点 |
| **2** | 调 `DAG_BASE_ARRIVAL_PROB` 到 ρ∈[0.7,0.95] | seed 方差显著下降 | — |
| **3** | **手工 DAG-aware 启发式 vs greedy-EFT**（不训练任何模型） | 配对 bootstrap 下界 > 0 | **超不过 → 环境里没有值得学的东西，必须先加结构** |
| **4** | 路线 A：手工前瞻特征 + 重训 Stage 1 | 超过 greedy | 不超过 → 超图也不会超 |
| **5** | 路线 B：加 UAV 节点 + 竞争/共址超边 | 超过路线 A | — |
| **6** | 消融：超图 vs 同参数量 GNN vs MLP | 超图显著更好 | 否则论文没有超图故事 |

**第 3 步是整个项目的分水岭，而且完全不需要训练模型。**
它回答的是"这个环境里到底有没有 RL 可赢的空间"——
这个问题比 clip、比超边设计、比收敛诊断都重要一个数量级。

顺带一提：第 6 步的"MLP 档"你**已经有了**——Stage 1 用的就是 `mlp` 编码器。
`build_clean_task_encoder(encoder_type=...)` 已经支持切换，消融基础设施是现成的。

---

## 参考文献

- [GA-DRL: Graph Neural Network-Augmented Deep Reinforcement Learning for DAG Task Scheduling over Dynamic Vehicular Clouds](https://arxiv.org/pdf/2307.00777)
- [Edge Generation Scheduling for DAG Tasks Using Deep Reinforcement Learning (IEEE TC)](https://ieeexplore.ieee.org/abstract/document/10382409/)
- [Task graph offloading via deep reinforcement learning in mobile edge computing (FGCS)](https://www.sciencedirect.com/science/article/abs/pii/S0167739X24001638)
- [Radio Resource Allocation for Device-to-Device Underlay Communication Using Hypergraph Theory](https://arxiv.org/pdf/1604.03246)
- [Structured Hypergraphs in Cellular Mobile Communication Systems](https://dl.acm.org/doi/fullHtml/10.1145/3571306.3571335)
- [Pairwise is Not Enough: Hypergraph Neural Networks for Multi-Agent Pathfinding](https://arxiv.org/pdf/2602.06733)
- [Hierarchical Reinforcement Learning on Multi-Channel Hypergraph Neural Networks (IJCAI 2024)](https://www.ijcai.org/proceedings/2024/0232.pdf)
- [Higher-Order Learning with Graph Neural Networks via Hypergraph Encodings (NeurIPS 2025)](https://openreview.net/forum?id=oeMK0Js4lq)
- [Trajectory-Aware Offloading Decision in UAV-Aided Edge Computing: A Comprehensive Survey](https://pmc.ncbi.nlm.nih.gov/articles/PMC10975722/)
