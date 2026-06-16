# zrj_3 Deprecated Entrypoints

`zrj_3` 当前唯一 clean mainline 验收入口是：

```bash
python scripts/smoke_clean_env.py
```

以下旧入口暂时不作为 `zrj_3 clean mainline` 的验收入口。原因是它们仍依赖旧环境主路径、旧 HGNN snapshot schema、旧 score-head teacher、旧 deadline/slack/task_type 诊断或旧 static scheduler compare 逻辑。

## Deprecated Scripts

### `scripts/train_assignment_mappo.py`

暂时不适配 clean mainline。

原因：

- 依赖旧 assignment MAPPO 与 HGNN encoder 输入；
- 依赖旧 candidate / action execution 记录；
- 仍包含 task_type、candidate count、旧 ablation 配置等统计；
- clean Env 当前只保证 reset、UE movement、hotspot DAG arrival 和最小 step。

替代计划：

- 后续新建 clean train script；
- 等 clean graph_builder、task_execution、reward 和 MAPPO 输入 schema 稳定后再迁移训练入口。

### `scripts/static_scheduler_compare.py`

暂时不适配 clean mainline。

原因：

- 依赖旧 service-domain、resource-competition、critical-support 等超边消融；
- 依赖 deadline/slack 风险判断；
- 依赖旧 static scheduler / score-head 对比字段；
- clean mainline 已废弃旧空间超边、旧 resource/critical-support 超边和 deadline-driven 主线。

替代计划：

- 后续新建 clean eval script；
- baseline 设计在实验跑通后单独收口。

### `scripts/pretrain_score.py`

暂时不适配 clean mainline。

原因：

- 依赖旧 score-head pretraining；
- 依赖旧 HGNN scheduler 和 snapshot schema；
- teacher score 当前仍与旧候选估计、deadline/slack、task_type 诊断存在耦合。

替代计划：

- clean HGNN snapshot schema 稳定后，重新设计 clean pretraining 或直接进入 RL/eval 流程。

## Deprecated Model Paths

### 旧 score-head pretraining

暂时不作为 clean mainline 的训练入口。

原因：

- 旧监督标签来自旧 task execution / graph schema；
- 与 clean DAG 属性、clean critical path、clean reward 尚未对齐。

### 旧 HGNN snapshot schema

暂时不作为 clean mainline schema。

原因：

- 旧 schema 包含 service-domain、resource-competition、critical-support、candidate-scarce、task-type 等字段；
- clean mainline 只保留 DAG dependency、k-hop dependency hyperedge、attribute similarity hyperedge。

## Current Acceptance Rule

这些脚本暂时不作为 `zrj_3 clean mainline` 的验收入口。

`zrj_3` 当前验收入口是：

```bash
python scripts/smoke_clean_env.py
```

后续会新建 clean train/eval scripts。
