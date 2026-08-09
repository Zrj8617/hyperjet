# 交给新窗口 Claude 的提示词：启发式基线评估框架

> 整块复制给新窗口。仓库 `D:\CodeFile\HyperUAV`（需先授予文件夹访问）。
> 项目约定已写在根目录 `CLAUDE.md`，本提示词不再重复。

---

## 任务

在 HyperUAV 项目里构建一个**可插拔的启发式策略评估框架**：
在已冻结的场景 tape 上运行**不训练**的启发式策略，
产出与现有闭环评估**同构**的结果文件，以便直接做配对比较。

第一个实例是 greedy-EFT（用于验证框架正确性），随后是 HEFT。

**不训练任何模型，不加载任何 checkpoint，不需要 GPU。**

仓库根目录的 `CLAUDE.md` 已包含项目约定、环境语义、关键常量和文件地图，
**请先读它，不要再自行全仓探索。**

---

## 协作规则（硬性）

分三轮，**每轮结束必须停下来等我确认，不要自行进入下一轮**。

**第 1 轮 —— 只读不写。**
读完下面「必读文件」后，用你自己的话复述这四件事：

1. 时隙生命周期是怎么走的
2. 候选特征和 EFT 在哪里、怎么算的
3. 任务处理顺序由谁决定
4. `closed_loop/episodes.jsonl` 的字段结构

然后写一份**最小实现规格**（接口签名 + 新增文件清单 + 验证方案）。
**这一轮不要写任何实现代码。**

**第 2 轮 —— 实现框架 + greedy-EFT + smoke 测试。** 必须通过全部「验证锚点」。

**第 3 轮 —— 加 HEFT。**

第 4 轮及以后（追加更多启发式）等我另行说明。

**如果任何一轮发现我给的信息与代码不符，立刻停下来告诉我，不要自行推断补齐。**

> ⚠️ 另有训练任务可能正在同一仓库上运行。
> **只新增文件，不要修改任何已有文件。**
> 如果你认为必须修改某个已有文件才能完成，停下来先问我。

---

## 必读文件（只读这些）

| 文件 | 读什么 |
| --- | --- |
| `scripts/run_stage1_temperature_followup.py` | **最重要**。时隙主循环、输出目录结构、`closed_loop/episodes.jsonl` 字段 |
| `environment/stage1_temperature_diagnostic.py` | `Stage1TemperatureDiagnosticEnv`；`act_with_temperature` 是你要仿写的决策循环 |
| `environment/stage1_temperature_tape.py` | tape 加载与校验（`load_scenario_shard`、`validate_manifest`） |
| `environment/assignment.py` | `build_offloading_candidate_components`、`estimate_offloading_candidate`、`TemporaryReservationState`、`CleanAssignmentBuffer`、`freeze_ready_tasks`、`_ready_sort_key` |
| `marl_models/mappo/clean_slot_orchestrator.py` | `prepare_slot_state` 的签名与返回值 |
| `environment/dag_tasks.py` 第 754–780 行 | `_mark_critical_path`。其中的 `dp` 递推**已经是一个不含通信代价的 upward rank**，HEFT 可直接复用该结构 |

---

## 要构建什么

一个评估脚本 + 一个策略模块，核心是**两个可替换的钩子**：

```python
# 钩子 1：决定本时隙任务的处理顺序
OrderPolicy(frozen_ready_tasks, task_manager, env) -> list[TaskNode]
#   默认实现 = 恒等（保持 freeze_ready_tasks 给出的顺序）

# 钩子 2：为单个任务选一架 UAV，返回候选数组里的下标
SelectPolicy(task, candidate_uav_ids, candidate_mask, estimates,
             state_view, task_manager, env) -> int
#   estimates[i].estimated_finish_time 就是 EFT
#   只允许在 candidate_mask 为 True 的下标里选
```

主循环**照抄** `act_with_temperature` 的结构，但把
「编码器 + scorer + 温度采样」替换为这两个钩子，并且：

- **删掉所有 torch / 编码器 / checkpoint 相关代码**
- **保留顺序预留语义**：每选一个立刻 `state_view.reserve(...)`，下一个任务基于更新后的状态重算
- 保留合法性掩码；掩码全 False 时计入 skip

> **关于任务顺序**：`CLAUDE.md` 里写了决策顺序由 `_ready_sort_key` 冻结——
> 那是**现有主线**的约束，**不许改动 `_ready_sort_key` 本身**。
> 本框架通过 OrderPolicy 钩子在框架内部重排，属于新增能力，不影响主线。

### 输出

每个 episode 一行 JSON，**字段与 `closed_loop/episodes.jsonl` 保持一致**
（`completed_dag_count`、`generated_dag_count`、`dag_completion_rate`、
`average_dag_flowtime`、`avg_uav_queue_length`、`admitted_incomplete_backlog`、
`arrival_blocked_count`、`episode_reward_total`、`invalid_assignment_count` 等），
额外加一个 `policy_name` 字段。目的是能直接进现有的配对 bootstrap 分析。

---

## 验证锚点（第 2 轮必须全部通过）

1. **策略无关锚点**：现有 static corpus 每条记录都有 `greedy_eft_uav_id` 字段。
   在 `slot_index == 0` 且 `decision_order == 0` 的那些决策上，
   你的 greedy-EFT 选择必须与之**逐条一致**——
   只有这个位置的状态完全由 tape 决定、与策略无关，之后就因策略而分叉了。
   corpus 路径由我提供；若拿不到，跳过此项并明确说明。
2. `invalid_assignment_count == 0`
3. `env.arrival_identity_metrics()` 不抛异常（`admitted == generated`）
4. 每个 episode 正好 200 个时隙
5. **可重复性**：同一 `(策略, 场景)` 连跑两次，输出逐 bit 相同
6. 顺手加一个 random 选择策略做健全性对照，greedy-EFT 的平均完成数应显著高于它

---

## HEFT 规格（第 3 轮）

标准 upward rank：

```
rank_u(t) = w(t) + max_{s ∈ succ(t)} ( c(t,s) + rank_u(s) )
rank_u(sink) = w(sink)
```

在本项目里：

- `w(t) = task.num_operation / config.UAV_COMPUTE_RATE_OPS_PER_SEC`
  （UAV 算力同质，所以这是精确值，不需要跨处理器取平均）
- `c(t,s)`：父子异机时的传输代价。UAV 同质但位置不同。
  **具体取哪个平均口径，请在第 1 轮规格里提出方案让我确认，不要自行拍板。**
- `dag_tasks.py:_mark_critical_path` 里的 `dp` 递推就是 `c ≡ 0` 的特例，
  按同样的逆拓扑序结构实现即可。

**适配方式已定**：用 `rank_u` 降序决定**本时隙 ready-task 的处理顺序**（即 OrderPolicy），
UAV 选择仍用 EFT 最小、tie-break 取最小 `uav_id`。

---

## 交付物

- **第 1 轮**：复述 + 最小规格（Markdown，不写代码）
- **第 2 轮**：`scripts/run_heuristic_policy_baseline.py`、
  `environment/heuristic_policies.py`、`scripts/smoke_heuristic_policy_baseline.py`
  （文件名可在第 1 轮规格里提更好的建议）
- **第 3 轮**：HEFT 的 OrderPolicy 实现 + 对应 smoke 用例

**每轮结束请停下来。**
