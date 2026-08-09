# 可插拔启发式策略评估框架 —— 第 1 轮：复述 + 最小实现规格

日期：2026-08-07
状态：**规格待确认**（第 1 轮，不含任何实现代码）
只读来源：`CLAUDE.md`、`scripts/run_stage1_temperature_followup.py`、
`environment/stage1_temperature_diagnostic.py`、`environment/stage1_temperature_tape.py`、
`environment/assignment.py`、`marl_models/mappo/clean_slot_orchestrator.py`、
`environment/dag_tasks.py:754-780`、`environment/comm_model.py:85-127`、`config.py`（常量核对）。

---

## 0. 先说三个阻塞项

### 0.1 本地没有 tape，也没有 static corpus（**阻塞第 2 轮的执行，不阻塞规格**）

`D:\CodeFile\HyperUAV` 下不存在任何 tape 目录（无 `manifest.json`），
也不存在 `static_corpus/records.jsonl`。`runs/` 是空目录。
`analysis_inbox/` 里只有 `tier2/static_corpus_load_summary.json`（聚合摘要，没有逐条记录）
和 `summarize_static_corpus.py`。

后果：

- **验证锚点 1（greedy_eft_uav_id 逐条一致）现在无法执行。**
  需要你提供 `20260806_formal_v1/static_corpus/records.jsonl`。
  只需要 `slot_index == 0 且 decision_order == 0` 的行，
  按 `(evaluation_scenario_seed, stable_task_id)` 去重后每个场景最多 1 条，20 个场景就是 ≤20 行。
  可以用 `grep '"slot_index":0' | grep '"decision_order":0'` 过滤后拷进 `analysis_inbox/`。
- **框架跑起来本身也需要 tape。** 但 tape 是**完全确定性**的：
  `stage1_temperature_tape.generate_scenario_shard(i)` 只依赖 `_keyed_seed`，不依赖全局 RNG。
  仓库里已有 `scripts/generate_stage1_temperature_tape.py`。
  → **请确认**：第 2 轮我可以在本地用该脚本重新生成一条 tape（新目录，create-only，不动任何已有文件），
  还是你会直接提供服务器上那条 tape 的路径？
  如果本地重生成，`logical_tape_sha256` 应当与服务器一致，我会在 run_manifest 里记录以便你比对。

### 0.2 一处与你的描述不符（**请确认，我没有自行推断**）

你写的钩子签名是：

```python
SelectPolicy(task, candidate_uav_ids, candidate_mask, estimates, state_view, task_manager, env) -> int
```

`estimates` 里 **不合法候选的 `estimated_finish_time` 是 `0.0`，不是 `+inf`**
（`estimate_offloading_candidate` 在 `legal=False` 分支直接返回默认值 0.0 的
`OffloadingCandidateEstimate`，见 `assignment.py:329-334`）。
所以 "取 EFT 最小" **必须先按 mask 过滤**，否则会稳定地选中非法候选。
现有 corpus 生成代码就是这么做的（`stage1_temperature_diagnostic.py:220`，先 `np.flatnonzero(mask)`）。
我会照此实现，此处只是明确告知，不是要改你的规格。

另：`candidate_mask` 不只反映"队列有空位"这类硬约束，
还包含 `estimate_offloading_candidate` 内部的降级（`job is None`、前驱未完成/未分配/仍 ready、
父 UAV 不存在）。也就是说 **mask 是合法性的唯一真值来源**，不需要额外判断。

### 0.3 一个会影响 HEFT 的环境事实（**新发现，请确认我的理解**）

`run_stage1_temperature_followup.py:62` 调用的是 `env.apply_movement({})`，
而 `env.apply_movement` 的文档说"未提供动作的无人机默认悬停"。
也就是说 **Stage 1 冻结评估里 5 架 UAV 全程不移动**，
服务位置就是 shard 里 `uav_position_uniforms × AREA` 的初始抽样，200 个时隙恒定。

这对 HEFT 很重要：`c(t,s)` 需要的 UAV 两两距离在整个 episode 内是**精确已知且恒定**的，
不需要用面积统计量去近似。见第 5 节。

---

## 1. 复述（一）：时隙生命周期怎么走

每个 episode 是 `for slot_index in range(200)`，每个时隙严格四步：

**① `prepare_slot_state(env=env, graph_builder=builder)`**
（`clean_slot_orchestrator.py:142`）内部先调 `env.prepare_slot_state()`：

- `_prepared_slot_open` 守卫 —— 同一时隙只能 prepare 一次。
- `_time_step` **先自增**。因此主循环第 `k` 次迭代里
  `env.current_time_seconds == (k+1) * 5.0`，而 tape 的下标是 `slot_index = _time_step - 1 = k`。
  两者相差 1，是有意的，不是 bug。
- 推进 UE 位置：`_advance_ues_for_slot(k)` 用 `shard["ue_mobility_standard_normals"][k]`。
- 处理 DAG 到达：`_process_clean_dag_arrivals()` 用 `shard["arrival_uniforms"][k]`
  和 `potential_template_at(shard, k, ue_id)`；受 `max_active_dags_per_ue=1` 限制，
  被挡住的记 `arrival_blocked_count`。
- 刷新 ready 状态，然后 **冻结** ready 集合 → `frozen_ready_task_ids`（第 3 节详述）。
- 建图（`CleanGraphBuilder.build`）→ `graph_snapshot`；建 critic 非图输入。

返回 `CleanPreparedSlotState`，其中 `frozen_ready_task_ids` 是本时隙不可变的决策集合，
`graph_snapshot.task_id_to_idx` 用来把 task 映射到嵌入行（本框架不用嵌入，但仍需用它做
"任务是否在图里"的存在性检查，以复刻原循环的 skip 语义）。

**② `env.apply_movement({})`** —— 全体悬停，冻结本时隙服务位置。
必须在 ready 集合冻结之后调用（有 `_prepared_slot_open` 守卫）。

**③ 逐任务卸载决策** —— 顺序决策 + 顺序预留，见第 2、3 节。
产出 `CleanAssignmentBuffer` 和 `skips`。

**④ `env.commit_and_advance(assignment_buffer=..., offloading_skip_count=...)`**
返回 `(obs, rewards, done, info)`。运行脚本只取第 4 个 `info`（记作 `latest`），
累加 `latest["step_reward"]`。调用后当前时隙关闭，可以 prepare 下一个。

episode 结束后调 `env.arrival_identity_metrics()`，
它内部断言 `arrival_admitted_count == generated_dag_count`，不等就抛 `AssertionError`。

---

## 2. 复述（二）：候选特征和 EFT 在哪里、怎么算的

入口是 `assignment.py:227 build_offloading_candidate_components(...)`，返回 5 元组：

```
dynamic  [M, 7]  UAV 动态特征
pair     [M, 8]  任务-UAV 配对特征
mask     [M]     bool 合法掩码
uav_ids  list[int]   按 uav.id 升序
estimates list[OffloadingCandidateEstimate]
```

**M = 全部 UAV 数（5），不是"合法候选数"**。非法候选也占一行，
`pair` 全零、`legal=False`、`estimated_finish_time=0.0`。行序 = `sorted(uavs, key=id)`。

逐候选的 EFT 由 `assignment.py:300 estimate_offloading_candidate(...)` 计算：

1. **合法性**（`is_assignment_legal`）：task 非空且 `is_ready`；`uav_id` 有效；
   `task_id` 不在 `state_view.reserved_task_ids`（本时隙未被预留过）；
   `executor.is_task_scheduled(task_id)` 为假；`state_view.remaining_slots(uav_id) > 0`（队列上限 16）。
   不合法 → 直接返回零配对特征、`legal=False`。

2. **传输时间 `transfer_time`**：
   - 入口任务（`not task.predecessors`）：从 UE 服务位置上传到目标 UAV 服务位置，
     `_clean_tx_seconds(task.input_data_size_mb, job.base_upload_bandwidth_mbps, d)`；
     `predecessor_ready_time = current_time_seconds`。
   - 非入口任务：遍历所有前驱。任一前驱 `finish_time is None` / `assigned_uav is None` /
     仍 `is_ready` → **降级为 `legal=False`**。
     前驱与本候选**同机则代价为 0**（`continue`）；异机则累加
     `_clean_tx_seconds(parent.output_data_size_mb, job.base_upload_bandwidth_mbps, d(parent_uav, target))`。
     注意：**UAV→UAV 也用 upload 带宽**。
     `predecessor_ready_time = max(所有前驱 finish_time)`。

3. **时间推进**：
   ```
   available_time      = state_view.available_times[uav]   # ← 顺序预留会改这个
   queue_waiting_time  = max(available, pred_ready, t) - t
   transfer_ready_time = max(t, available, pred_ready) + transfer_time
   compute_time        = task.num_operation / UAV_COMPUTE_RATE_OPS_PER_SEC
   compute_finish_time = transfer_ready_time + compute_time
   return_time         = sink 任务才有：UAV→UE，用 base_download_bandwidth_mbps；否则 0
   estimated_finish_time (EFT) = compute_finish_time + return_time
   ```
   `compute_time` 在同一决策的所有候选间完全相同（UAV 同质），零判别信息 —— 与 `CLAUDE.md` 一致。
   传输时间是**串在 UAV 可用之后**的（先 max 再加 transfer），这是既有建模选择，不动。

4. `estimated_queued_workload = task.num_operation`。

**通信模型**（`comm_model.py`）是精确闭式：
```
rate(bw, d)  = bw / (1 + (d/100)^2)
tx(mb, bw, d) = mb * 8 / rate = mb * 8 * (1 + (d/100)^2) / bw     [秒]
```
即**传输时间对距离是严格二次函数**。这一点在第 5 节 `c(t,s)` 的口径选择上是决定性的。

**顺序预留**（`TemporaryReservationState`）：
`from_executor(uavs, executor)` 建快照；每选定一个候选后立刻
`reserve(task_id, uav_id, estimated_available_time=estimate.estimated_finish_time,
estimated_queued_workload=estimate.estimated_queued_workload)`，
它会 `queue_lengths += 1`、`slot_assigned_counts += 1`、`queued_workloads += workload`、
`available_times[uav] = max(旧值, EFT)`。下一个任务因此基于**更新后的状态**重算候选与 EFT。
这就是把 `M^N` 降为 N 次选择的机制，框架必须原样保留。

---

## 3. 复述（三）：任务处理顺序由谁决定

由**环境**决定，不由策略决定。

`env.prepare_slot_state()` 内部调用 `assignment.freeze_ready_tasks(task_manager)`：

```python
sorted(task_manager.get_ready_tasks(), key=lambda t: _ready_sort_key(t, task_manager))
_ready_sort_key -> (job.arrival_time, str(dag_id), topological_index, str(task_id))
```

`topological_index` 显式判 `None`（合法的 0 不会被当成缺失），缺失才退回任务 ID 数字后缀。
结果作为 `frozen_ready_task_ids` 交给策略层，策略层只能**遍历**它。

运行脚本再把 ID 还原成 `TaskNode`（`env.task_manager.get_task(tid)`，过滤 `None`），
`enumerate` 的下标就是 `decision_order`。

**本框架的定位**：`_ready_sort_key` 一个字符都不改。
OrderPolicy 在**新脚本内部**对还原后的 `list[TaskNode]` 重排，
默认实现是恒等（保持 `_ready_sort_key` 顺序）。主线不受影响。

**由此得到验证锚点 1 的适用边界**：`decision_order` 是**重排后**的下标。
`slot_index == 0 且 decision_order == 0` 只有在 **OrderPolicy 为恒等**时
才与 corpus 指向同一个任务。所以锚点 1 只对 `identity + greedy_eft` 成立，
对 HEFT 排序不成立（第 3 轮里我会改成按 `stable_task_id` 对齐而非按 `decision_order`）。

另外 corpus 只在 `count_nonzero(mask) >= 2` 时写记录，
所以某些 (场景, slot 0, order 0) 可能根本没有 corpus 行 —— 属正常，不算失配。

---

## 4. 复述（四）：`closed_loop/episodes.jsonl` 的字段结构

来自 `run_stage1_temperature_followup.py:75`，单行 JSON，
`json.dumps(..., ensure_ascii=False, sort_keys=True, allow_nan=False)`：

| 字段 | 来源 | 说明 |
| --- | --- | --- |
| `schema` | 常量 | `"stage1_temperature_closed_loop_episode_v1"` |
| `phase` | CLI | `pilot` / `formal` |
| `training_seed` | checkpoint 注册表 | **策略基线不适用** |
| `checkpoint_sha256` | checkpoint | **策略基线不适用** |
| `logical_tape_sha256` | `manifest["logical_tape_sha256"]` | tape 身份 |
| `episode_index` | = `scenario_index` | |
| `evaluation_scenario_seed` | `shard["evaluation_scenario_seed"]` | `424242 + episode_index` |
| `sampling_replicate` | CLI | **策略基线不适用**（确定性策略） |
| `temperature` | CLI | **策略基线不适用** |
| `physical_slots` | 常量 200 | |
| `active_dag_cap` | 常量 1 | |
| `queue_cap` | 常量 16 | |
| `encoder` | 常量 `"mlp"` | **策略基线不适用** |
| `actor_uav_feature_dim` | 常量 7 | **策略基线不适用** |
| `episode_reward_total` | 累加 `latest["step_reward"]` | |
| `completed_dag_count` | `int(latest["completed_dag_count"])` | **主指标** |
| `generated_dag_count` | `arrival["generated_dag_count"]` | 依赖策略，别当分母 |
| `admitted_dag_count` | `arrival["admitted_dag_count"]` | 恒等于上一行 |
| `arrival_blocked_count` | `arrival["arrival_blocked_count"]` | |
| `dag_completion_rate` | `latest.get(...)` | 可能为 `None` |
| `average_dag_flowtime` | `latest.get(...)` | 可能为 `None` |
| `avg_uav_queue_length` | `latest.get(...)` | 可能为 `None` |
| `admitted_incomplete_backlog` | `generated - completed`（**脚本里现算的，不是 info 字段**） | |
| `invalid_assignment_count` | `int(latest.get(..., 0))` | |
| `finite` | `math.isfinite(reward_total)` | |

注意 `dag_completion_rate` / `average_dag_flowtime` / `avg_uav_queue_length`
取自**最后一个时隙**的 `info`，是 episode 累计量的快照，不是逐时隙均值。

**本框架的输出约定（提案）**：字段名与上表**逐字一致**，以便直接进现有配对 bootstrap；
`training_seed` / `checkpoint_sha256` / `sampling_replicate` / `temperature` / `encoder` /
`actor_uav_feature_dim` 六项**保留字段名但置为 `null`**（不删，保证列对齐）；
`schema` 改为 `"heuristic_policy_closed_loop_episode_v1"`；
新增 `policy_name`（如 `"identity+greedy_eft"`）、`order_policy`、`select_policy`、
`policy_config_sha256`（策略参数的规范哈希，用于可重复性追踪）。
`phase` 沿用 CLI。**这一条如果你希望改成"直接删掉不适用字段"，请在确认时说明。**

---

## 5. HEFT 的 `c(t,s)` 口径 —— **待你拍板**

### 5.1 事实基础

- `tx(mb, bw, d) = mb * 8 * (1 + (d/100)^2) / bw`，**对 d 严格二次**。
- 因此 `E[tx]` 与 `tx(E[d])` **不相等**，且前者恒大（Jensen）。
- 本评估中 5 架 UAV **全程悬停不动**（见 0.3），位置由 shard 固定，两两距离精确已知。
- 父子**同机代价为 0**（代码里 `continue`）。
- 父→子传输用 `job.base_upload_bandwidth_mbps`（每个 DAG 一档：20/50/100 Mbps，概率 0.3/0.5/0.2）。

若把面积当成 500×500 均匀分布，数值参考（200 万次蒙特卡洛）：

| 口径 | 距离因子 `1 + (d/100)^2` |
| --- | --- |
| `E[1 + (d/100)^2]`（时间的期望） | **9.332** |
| `1 + (E[d]/100)^2`（在平均距离处取时间） | 7.794 |

两者差 **−16.5%**。这就是最容易悄悄写歪的地方。

### 5.2 四个候选方案

| 方案 | 定义 | 评价 |
| --- | --- | --- |
| **A. `c ≡ 0`** | 退化为 `dag_tasks.py:_mark_critical_path` 的 `dp` | 零成本、可直接复用；作为**敏感性对照**很有价值，但不是 HEFT |
| **B. 有序异机 UAV 对上的时间均值（推荐）** | `c(t,s) = (1/(M(M−1))) · Σ_{a≠b} tx(t.output_mb, bw_up, d(a,b))` = `t.output_mb·8/bw · (1 + mean_{a≠b} d(a,b)²/10000)` | 用本 episode 5 架 UAV 的**真实固定位置**，精确、无近似；对齐 HEFT "跨处理器取平均"的原意；期望取在**时间**上，避开 Jensen 陷阱 |
| **C. B + 计入同机** | 对全部 `M²` 有序对取均值，对角线为 0 → 恰好等于 `0.8 × B`（M=5） | 显式建模 1/M 的同机概率。与 B 只差一个常数因子 0.8，**对纯排序几乎无影响**（除非与 `w(t)` 的相对权重变了 —— 而它确实变了，见下） |
| **D. 面积统计量（位置无关）** | 用 `E[d²] = (W²+H²)/6 = 83333` → 因子 9.333 | 与 episode 无关，跨场景可比；但忽略了本 episode UAV 的具体构型 |

（**E. 在平均距离处取时间** = 因子 7.794 —— 明确**不推荐**，仅列出以说明差异。）

**我的推荐：方案 B**，理由是精确（位置恒定，不需要任何近似）、
期望取在时间上而非距离上、且最贴近 HEFT 原始定义。
B 与 C 的比值是常数 0.8，但因为 `rank_u = w + max(c + rank_u)` 里 `w` 不缩放，
0.8 的差异**会**改变 `c` 与 `w` 的相对权重，从而可能改变排序 —— 所以这不是无关紧要的选择。
如果你更看重"考虑同机可能性"的物理直觉，就选 C。

### 5.3 两个附带的次级问题（一并请你确认）

**(a) sink 的回传要不要进 `rank_u`？**
标准 HEFT 里 `rank_u(sink) = w(sink)`。但本项目 DAG **必须等 sink 结果回传到 UE 才算完成**。
- (i) `rank_u(sink) = w(sink)`（HEFT 标准，推荐）
- (ii) `rank_u(sink) = w(sink) + 回传时间估计`（用 `bw_down` 和 UAV→UE 的平均距离）
(ii) 更贴合本环境的完成判据，但偏离 HEFT 标准。**我倾向 (i)，把 (ii) 留作变体。**

**(b) 跨 DAG 比较 `rank_u` 是否有意义？**
标准 HEFT 只在**单个 DAG 内**排序。但本时隙的 ready 集合混着多个 DAG，
而不同 DAG 的 `bw_up` 档位（20/50/100）和任务规模都不同，
`rank_u` 的量纲会系统性偏向低带宽 DAG。
- (i) 直接按 `rank_u` 降序（你的规格原文，推荐先这样做）
- (ii) 按 `rank_u / rank_u(该 DAG 的入口任务最大值)` 归一化后降序
**我倾向先做 (i)**，把 (ii) 记为已知偏差、留作第 4 轮的变体。请确认。

### 5.4 实现结构（无论选哪个口径都一样）

复用 `_mark_critical_path` 的逆拓扑序结构：
```python
for task in sorted(job_tasks, key=lambda t: (-t.level, t.task_id)):
    succ = [c for c in task.successors if c in task_map]
    if not succ:
        rank[task.task_id] = w(task)                       # 或 w + return（选项 a-ii）
    else:
        rank[task.task_id] = w(task) + max(c(task, s) + rank[s] for s in succ)
```
`-level` 降序保证子节点先算完（与既有 `dp` 完全同构）。
按 `dag_id` 缓存；因为 UAV 位置恒定，缓存在整个 episode 内有效。
**若将来开启 UAV 移动，此缓存必须改为逐时隙失效** —— 会在代码里写死断言。

---

## 6. 最小实现规格

### 6.1 新增文件清单（**只新增，零修改**）

| 文件 | 轮次 | 内容 |
| --- | --- | --- |
| `environment/heuristic_policies.py` | 2 | 钩子协议 + 策略实现 + 无 torch 的决策循环 `act_with_heuristic_policy` |
| `scripts/run_heuristic_policy_baseline.py` | 2 | 评估入口（CLI、tape 校验、200 时隙主循环、create-only 输出） |
| `scripts/smoke_heuristic_policy_baseline.py` | 2 | 验证锚点 2–7 |
| `environment/heuristic_policies.py`（追加） | 3 | `HeftUpwardRankOrderPolicy` + `compute_upward_ranks` |
| `scripts/smoke_heuristic_policy_heft.py` | 3 | HEFT 专项用例（手工 DAG 的 rank 闭式核对、`c≡0` 退化到 `dp`、逆拓扑序正确性） |
| 本文档 | 1 | 规格 |

文件名沿用你的建议，只把第 3 轮的 smoke 拆成独立文件（符合 `smoke_<名字>.py` 约定）。

### 6.2 接口签名

```python
# environment/heuristic_policies.py

@dataclass(frozen=True, slots=True)
class DecisionContext:
    """一次卸载决策的只读上下文，供 SelectPolicy 做确定性派生随机数用。"""
    evaluation_scenario_seed: int
    slot_index: int
    stable_task_id: str
    decision_order: int


class OrderPolicy(Protocol):
    name: str
    def __call__(
        self, *,
        frozen_ready_tasks: list[TaskNode],
        task_manager: DAGTaskManager,
        env: Any,
    ) -> list[TaskNode]: ...
    # 契约：返回值必须是入参的一个排列（同长度、同元素、无重复）——运行时断言


class SelectPolicy(Protocol):
    name: str
    def __call__(
        self, *,
        task: TaskNode,
        candidate_uav_ids: list[int],
        candidate_mask: np.ndarray,          # bool[M]
        estimates: list[OffloadingCandidateEstimate],
        state_view: TemporaryReservationState,
        task_manager: DAGTaskManager,
        env: Any,
        context: DecisionContext,
    ) -> int: ...
    # 契约：返回下标 i 必须满足 candidate_mask[i] is True——运行时断言
```

**与你原文的两处偏离，请确认**：

1. 改成**关键字参数**（`*,`）。理由：位置参数在 7–8 个形参时极易错位，
   而且第 3 轮加 HEFT 时可能还要扩参数。
2. 给 `SelectPolicy` 加了 `context: DecisionContext`。理由：`CLAUDE.md` 要求
   "需要严格复现的地方用 keyed 哈希派生随机数，不要依赖全局 RNG 状态"，
   random 对照策略必须有一个稳定的 key 来源。确定性策略（greedy-EFT）忽略它即可。

具体实现：

```python
class IdentityOrderPolicy:      name = "identity"
class HeftUpwardRankOrderPolicy: name = "heft_upward_rank"   # 第 3 轮

class GreedyEFTSelectPolicy:    name = "greedy_eft"
    # legal = np.flatnonzero(candidate_mask)
    # min_eft = min(estimates[i].estimated_finish_time for i in legal)
    # 取 eft == min_eft（精确浮点相等，与 corpus 生成代码一致）的下标中 uav_id 最小者
class RandomSelectPolicy:       name = "random"
    # seed = sha256(canonical_json(["heuristic_random_v1", name, seed, slot, task_id, order]))
    # index = legal[seed % len(legal)]
```

决策循环（`act_with_heuristic_policy`）**照抄 `act_with_temperature` 的骨架**，
逐行对应关系：

| `act_with_temperature` | 本框架 |
| --- | --- |
| `import torch` / embedding 张量 | **删除** |
| `for decision_order, task in enumerate(frozen_ready_tasks)` | `enumerate(order_policy(...))` |
| `task_idx = graph_snapshot.task_id_to_idx.get(...)`；`None → skips += 1` | **保留**（存在性检查，不用嵌入） |
| `build_offloading_candidate_components(...)` | **原样保留** |
| `dynamic.shape[0] == 0 or not mask.any() → skips += 1` | **原样保留** |
| `np.repeat` 拼特征 + `actor.scorer` 打分 | **删除** |
| `keyed_temperature_sample(...)` | `select_policy(...) -> int` |
| `assignments.append(...)` + `reservation.reserve(...)` | **原样保留** |
| `record_static` 分支 | 改为可选的决策级 trace（见 6.4） |

返回 `(CleanAssignmentBuffer, list[decision_record], skips)`。

### 6.3 运行脚本 CLI

```
--tape-dir PATH            必填，manifest.json 所在目录
--output-dir PATH          必填，create-only（已存在则 FileExistsError）
--policies NAME [NAME...]  必填，取值 identity+greedy_eft / identity+random / heft+greedy_eft
--scenario-indices INT...  必填
--max-physical-slots 200   默认 200，非 200 直接报错
--record-decisions         默认开启（锚点 1 需要），--no-record-decisions 可关
--phase {pilot,formal}     默认 formal，仅写进输出
```
**无 `--device`，全文件不 import torch**（smoke 里断言 `"torch" not in sys.modules`）。

### 6.4 输出布局（全部 create-only）

```
<output-dir>/
  closed_loop/episodes.jsonl     # 每 (policy, scenario) 一行，字段见 §4
  decisions/records.jsonl        # 可选；每条决策一行
  run_manifest.json              # 各产物 sha256、logical_tape_sha256、策略清单、常量回显
```

`decisions/records.jsonl` 单行字段（为锚点 1 和调试服务）：
```
policy_name, evaluation_scenario_seed, episode_index, slot_index,
stable_task_id, decision_order, candidate_uav_ids, candidate_mask,
eft, selected_uav_id, selected_index, legal_candidate_count
```
故意与 corpus 的 `stable_task_id` / `slot_index` / `decision_order` /
`candidate_uav_ids` / `eft` 同名同序，便于直接 join。

### 6.5 验证方案

| # | 锚点 | 检查方式 | 前置条件 |
| --- | --- | --- | --- |
| 1 | greedy-EFT 与 corpus 一致 | 读 corpus，取 `slot_index==0 且 decision_order==0`，按 `(scenario_seed, stable_task_id)` 去重；与我方 `decisions/records.jsonl` 同键 join；断言 `selected_uav_id == greedy_eft_uav_id` **逐条**相等。另额外断言 `eft` 数组逐元素 `== `（bit 级），这能同时验证 EFT 计算路径完全一致 | **需要 corpus 文件（见 §0.1）**；拿不到则 skip 并在 run_manifest 里写 `anchor1_status="skipped_no_corpus"` |
| 2 | `invalid_assignment_count == 0` | 每行 episode 断言 | — |
| 3 | `arrival_identity_metrics()` 不抛 | 直接调用；额外断言 `admitted == generated` | — |
| 4 | 每 episode 正好 200 时隙 | 计数器 + 断言 `prepared.slot_index` 序列 == `range(1,201)`（注意 `_time_step` 先自增） | — |
| 5 | 可重复性 | 同 `(policy, scenario)` 连跑两次到两个 create-only 目录，断言 `episodes.jsonl` 与 `decisions/records.jsonl` 的 **SHA-256 完全相等** | — |
| 6 | greedy-EFT ≫ random | 在 ≥3 个场景上跑两策略，按场景配对比较 `completed_dag_count`；预注册判据：**每个场景 greedy ≥ random，且均值提升 > 10%**；不满足就报告不通过、不改阈值 | — |
| 7 | 无 torch | `assert "torch" not in sys.modules` | — |
| 8 | 恒等 OrderPolicy 无副作用 | 断言 `[t.task_id for t in identity(...)] == prepared.frozen_ready_task_ids`（过滤 None 后） | — |
| 9 | 只新增文件 | `git status --porcelain --ignore-cr-at-eol` 只出现新增（`??`），无 `M` | — |

第 3 轮追加：

| # | 检查 |
| --- | --- |
| 10 | 手工构造一个 4 节点 DAG，闭式核对 `rank_u` 每个节点的值 |
| 11 | 令 `c ≡ 0`，断言 `rank_u` 与 `_mark_critical_path` 的 `dp` **逐任务相等**（这是最强的结构正确性证据） |
| 12 | 断言 HEFT OrderPolicy 输出是 `frozen_ready_tasks` 的严格排列，且排序键是**全序**（无平局歧义）：`(-rank_u, arrival_time, dag_id, topological_index, task_id)` |
| 13 | 断言 rank 缓存在 UAV 位置变化时会失效（构造一次位置扰动，断言抛错或重算） |

---

## 7. 需要你确认的清单（第 2 轮开始前）

1. **corpus**：能否提供 `20260806_formal_v1/static_corpus/records.jsonl`
   （或其中 `slot_index==0 && decision_order==0` 的若干行）到 `analysis_inbox/`？
2. **tape**：本地用 `scripts/generate_stage1_temperature_tape.py` 重新生成，还是你提供路径？
3. **§0.2** 关于 `estimates` 中非法候选 EFT 为 `0.0` 的事实，确认我的处理（先按 mask 过滤）无误。
4. **§0.3** 确认"Stage 1 评估中 UAV 全程悬停"这一理解正确。
5. **§4** 输出字段：不适用字段"保留字段名置 null" vs "直接删除"，选哪个？
6. **§5.2** `c(t,s)` 口径：**A / B（推荐）/ C / D**？
7. **§5.3(a)** sink 回传是否进 `rank_u`：(i) 不进（推荐）/ (ii) 进？
8. **§5.3(b)** 跨 DAG 直接比 `rank_u`：(i) 直接比（推荐）/ (ii) 归一化？
9. **§6.2** 钩子签名改关键字参数 + 新增 `DecisionContext`，是否接受？

**第 1 轮到此为止，等你确认后再进第 2 轮。**

---

## 8. 第 2 轮实现记录（2026-08-07）

### 8.1 tape SHA 校验 —— **通过**

本地按 `scripts/generate_stage1_temperature_tape.py` 的完全等价路径重新生成 20 个 shard，
用真实的 `build_manifest` 组装 manifest：

```
logical_tape_sha256 = cf0086e047e33931a867ff94104f4615627164d1afb29bbd6b7c9b133bbfacf4
expected            = cf0086e047e33931a867ff94104f4615627164d1afb29bbd6b7c9b133bbfacf4   -> MATCH
```

结论：本地代码与生成 `20260806_formal_v1` 正式 tape 时**逐 bit 一致**。
tape 实测约 1.43 GB / 每 shard 约 71 MB / 生成约 4.5 分钟。

### 8.2 与第 1 轮规格的偏离（均已实现）

| 项 | 第 1 轮规格 | 第 2 轮实际 | 原因 |
| --- | --- | --- | --- |
| 锚点 4 断言 | `prepared.slot_index` 序列 == `range(1,201)` | == `range(0,200)` | 读代码后确认 `env.prepare_slot_state()` 里 `slot_index = self._time_step` 取的是**自增前**的值；自增只影响 `current_time_seconds`。第 1 轮写错了 |
| `--record-decisions` | 默认开启 | **默认关闭** | 逐决策记录很大（200 slot 约 966 条/episode）。锚点 1 用 `--record-decisions --record-decisions-max-slot 0` 打开 |
| torch 断言 | `assert "torch" not in sys.modules` | 改为记录 `torch_present_transitively` + 断言 `cuda_initialized is False` | `marl_models.mappo.clean_slot_orchestrator` 顶层有 `try: import torch`，装了 torch 的机器上必然被动进入 sys.modules，原断言会误报 |
| smoke 结构 | 单次运行 | 加 `--sections A,B,C,D,R,G,E` 与 `--shard-dir` | 便于分段运行；`--shard-dir` 直接复用已有 tape 的 shard，省去每次 12s/shard 的重新生成 |
| B/C 测试位置 | slot 0 | 先用 greedy 推进 `--warmup-slots`（默认 40）再测 | slot 0 只有 1 个 DAG / 2 个 ready 任务，覆盖度太弱；warm-up 后是 6 个 DAG / 21 个 ready 任务 |

### 8.3 新增的实现决策

- **`--partition-hyperedges {on,off}`，默认 `off`**。E_part 走 KaHyPar spawn 子进程，
  但启发式策略**没有任何编码器读 incidence 矩阵**，所以关掉是纯提速、零语义影响。
  smoke 的 D8 用例会在装了 kahypar 的机器上实测「开/关结果完全一致」；未安装时 SKIP。
- **`--validate-shards`，默认关闭**。manifest 已对每个 shard 钉死 `size_bytes + sha256`，
  重复结构解析会把约 1.4 GB 全部 parse 进内存。
- **确定性策略只跑 `policy_replicate=0`**。`_resolve_replicates` 只对 `*+random` 展开多 replicate。
- **rank_u 统一口径**：非 HEFT 策略用 `include_return=true` 的独立缓存记录 rank_u，
  字段 `rank_u_convention` 标明尺子；HEFT 策略记录它自己实际用的那把。

### 8.4 验证结果

| 锚点 | 用例 | 结果 |
| --- | --- | --- |
| — | A1 `c(t,s)` 因式分解精确（1e-12） | PASS |
| — | A2 有序 20 对均值 == 无序 10 对均值 | PASS |
| — | B1 无后继任务集合 == `job.sink_task_ids`（6 DAG） | PASS |
| — | B2 upward rank DP == 穷举路径枚举（76 组） | PASS |
| — | B3 `c==0` 退化为 `_mark_critical_path` 的 dp（38 任务） | PASS |
| 8 | C1 恒等 OrderPolicy == `frozen_ready_task_ids`（21 任务） | PASS |
| — | C2 HEFT 排序：排列 + 全序 + rank_u 降序 | PASS |
| 2 | D1 `invalid_assignment_count == 0` | PASS |
| 3 | D2 `arrival_identity_metrics()` 不抛（admitted == generated == 158） | PASS |
| 4 | D3 正好 200 时隙，`slot_index` 0..199 | PASS |
| — | D4 greedy 恒取最小合法 EFT，并列取最小 uav_id（966 决策） | PASS |
| — | D5 每条决策都记录 rank_u + 带宽档位 | PASS |
| 5 | D6 逐 bit 可重复（run_episode 级 + 完整 runner 级两处都验） | PASS |
| — | D7 `policy_replicate` 确实改变 random 流 | PASS |
| — | D8 `--partition-hyperedges off` 语义惰性 | SKIP（本机无 kahypar） |
| 6 | D9 greedy ≫ random：greedy `[147,84,61,161,149]` vs random `[58,36,42,69,59]`，逐场景全胜，均值 +128.0% | PASS |
| 1 | E1 与 static corpus 的 `greedy_eft_uav_id` 逐条一致 | **SKIP（缺 `analysis_inbox/corpus_slot0_anchor.jsonl`）** |
| 9 | 只新增文件（4 个新文件全部 `??`，零 `M`） | PASS |

`uav_mean_distinct_pair_factor` 实测：episode 0 为 **8.720**，
而 500×500 均匀分布的面积统计量是 9.332 —— 用真实位置而非面积近似确有 6.6% 的差别。

### 8.5 第 3 轮的实际剩余工作

`#12` 要求所有策略都记录 rank_u，这把 upward rank 的全部机制提前拉进了第 2 轮，
`HeftUpwardRankOrderPolicy` 因此顺带完成并已被 C2 覆盖。
**但 HEFT 尚未跑过任何闭环评估** —— 第 3 轮的实质内容是：
`heft_ret0/heft_ret1 + greedy_eft` 的闭环跑批、与 `identity+greedy_eft` 的配对比较、
以及顺序重合度分析（用 decision 记录里的 rank_u）。
