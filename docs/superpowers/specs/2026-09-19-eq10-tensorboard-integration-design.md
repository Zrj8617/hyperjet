# 2026-09-19 EQ10 TensorBoard 接入设计

## 目标

在 20260918 四臂 × 双编码器正式评估完成后，把 24 个训练 run 及其 validation/test 结果接入现有 reward-redesign TensorBoard。可见 run 名沿用“日期 + 训练代号 + seed”，不展示 `MLP` 或 `EQ10`；HGNN 仅增加既有的 `TYPED_GATED_HGNN` 标记。

## 可见命名

- MLP：`20260918_<ARM>_seed<SEED>`
- HGNN：`20260918_TYPED_GATED_HGNN_<ARM>_seed<SEED>`
- 跨 run 汇总：`20260918_SUMMARY`

其中 `<ARM>` 为 B2/C2A/C2B/C2，`<SEED>` 为 5/86/617。服务器原始目录 `20260918_<ARM>_<ENCODER>_EQ10_seed<SEED>` 不移动、不改名。

## 数据布局

每个可见训练 run 使用独立组合事件目录：

1. 以只读符号链接引用原始训练 event 文件，保留全部既有训练 tags；
2. 新写一个 evaluation event 文件，追加：
   - `validation/epXXXX/<metric>`：5 个候选 checkpoint × 20 条 validation tape；
   - `validation/mean/<metric>`：横轴为 checkpoint episode；
   - `selection/*`：选中 episode 与 validation 主指标；
   - `joint|forced_hover / budget_selected|final_ep0500 / <metric>`：沿用旧正式评估 tag 结构，横轴为 test tape ID；
   - `sensitivity/lambda_0p1|0p7|1p0/<metric>`：同一批 joint/budget-selected test 行离线重算。
3. `20260918_SUMMARY` 记录跨 arm/encoder 的汇总标量；详细统计仍以机器可读结果和正式报告为事实源。

## 发布流程

1. 只在评估 manifest 为 `completed`、selection 已锁定、全部 120 个 validation 和 96 个 test JSON 完整时开始导出。
2. 先写入新的 staging 根目录，不触碰现有 TensorBoard view。
3. 用 EventAccumulator 验证 24 个 run 的训练与评估 tag、点数和 step 语义。
4. 验证通过后，在现有 `/data2/zrj2025/uav-results/tensorboard_views/reward_redesign` 下创建 25 个新链接（24 run + 1 summary）；目标名已存在则停止，不覆盖。
5. 校验 TensorBoard 进程和本机 16008 tunnel，返回可点击链接。

## 失败边界

- 评估未完成、任一批次缺失、tag 点数不符、目标名冲突或 TensorBoard view 不可访问时不发布半成品。
- 不修改原始训练 event、评估 JSON、20260915 旧链接或旧导出目录。
