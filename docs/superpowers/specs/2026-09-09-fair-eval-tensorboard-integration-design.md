# C1 / B2 / C2 公平评估 TensorBoard 接入设计（2026-09-09）

**Status:** APPROVED by user

## 目标

把 `2026-09-08-c1-b2-fair-reevaluation-result.json` 中已经完成并通过审计的 Track A 固定-tape 正式评估接入现有 `http://127.0.0.1:16008/`，同时保持训练曲线与评估结果的语义隔离。

## 数据映射

- 新增 9 个 TensorBoard run：`FormalEval/{C1,B2,C2}/seed{0,1,2}`。
- 横轴使用独立 test tape 的 `tape_id`（200--249），不冒充训练 step 或 episode。
- tag 使用 `{protocol}/{checkpoint_label}/{metric}`：
  - `forced_hover/budget_selected/*` 是正式主协议；
  - `forced_hover/final_ep0500/*` 是最终 checkpoint 敏感性分析；
  - `joint/budget_selected/*` 和 `joint/final_ep0500/*` 是完整 deterministic joint-policy 次协议。
- 导出统一成本、实际成本分量、offer/admission/completion 计数与比例，以及已有的流时、吞吐和能耗辅助指标。
- Track B 多峰值 epoch 诊断不接入 `FormalEval`，防止辅助诊断替代正式评估。

## 部署

- 在结果根下生成持久事件目录 `audits/20260908_C1_B2_C2_fair_eval/tensorboard_formal_eval/<arm>/seed<seed>`。
- 在既有持久 TensorBoard view 中建立对应层级软链接；不修改或覆盖原 30 个训练 run。
- 复用服务器 loopback TensorBoard 与本机 SSH tunnel。

## 验证

- EventAccumulator 必须读到恰好 9 个新 run，每个 run 恰好包含 4 个协议/checkpoint 组合、50 个 test-tape 点/标量 tag。
- TensorBoard API 必须列出 9 个 `FormalEval` run，且主协议核心 tag 可见。
- 本机 `http://127.0.0.1:16008/` 页面可打开、无前端控制台错误，并能看到正式评估分组。
