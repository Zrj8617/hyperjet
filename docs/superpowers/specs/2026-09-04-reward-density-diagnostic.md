# mini-spec: reward 密度诊断（2026-09-04）

**状态：** APPROVED　**执行：** Codex　**类型：** 纯诊断（不改训练 / 不改环境目标）

## 核心（4 行）
- **目标**：验证"即时正信号太稀疏 / DAG 完成奖励太延迟"是否是 offloading credit 学不出的根因。
- **做什么**：复用 Phase 4A 同一批 27 决策 / 8 roots / exact branches，唯一改动是把 return 从原始 reward 换成 **progress-shaped return**，其余不动。
- **看什么数**：all-legal pair 可解析率（原 ~23%）、all-legal Top-1（原 33%，随机 20%）、credit CI 排除 0 比例（原最优 H=60 ~13%）、最优 horizon（原 H=60）。
- **通过**：这几个数在 shaped return 下**显著上升** → reward 密度是主因，进第 2 步（shaped 训练，需师兄批）；否则 reward 不是主因，转 state / estimand。

## Codex 执行要点
- shaped return 定义（potential-based，Ng 1999，**不改最优策略/环境目标 → 属诊断，不属 reward redesign**）：
  `r'(s→s') = r(s→s') + γ·Φ(s') − Φ(s)`，
  `Φ(s) = REWARD_COMPLETED_DAG_WEIGHT × Σ_{active DAG} (该 DAG 已完成子任务数 / 该 DAG 总子任务数)`。
- 需在 branch rollout 里**逐 step 记录每个 active DAG 的完成进度**以构造 Φ（Phase 4A 若只存了 reward_sequence，则补记 progress_sequence）；不重跑环境语义，只在同一批 exact branches 上重新计分。
- 用 shaped return 重算 Phase 4A 的同一套指标，并列出与原始值的对比。
- **单变量 / RNG-neutral**：shaping 关闭时结果必须与 Phase 4A 逐值复现（gate-OFF 复现）。

## 工程门禁（沿用 Phase 4A）
纯 audit；无训练；无主线改动；optimizer step = 0；censor / semantic mismatch / unrecognized RNG 全为 0；参数逐 tensor 不变。

## 结果产物
- report：`docs/superpowers/reports/2026-09-04-reward-density-diagnostic-results.md`
- JSON：`/data2/zrj2025/uav-results/audits/boundary-anchored-decision-credit/reward_density_diagnostic/...json`
- 版本记录（AGENTS.md §3）：`git rev-parse HEAD` + `git status --porcelain` + 脚本 `git hash-object`

## 不做
不改训练、不改环境 reward、不接入 actor/critic、不动 state / PPO、不提交主线。
