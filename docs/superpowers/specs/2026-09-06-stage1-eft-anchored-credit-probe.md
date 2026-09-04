# mini-spec: Stage-1 EFT-anchored offloading credit 短训练（2026-09-06）

**状态：** APPROVED（用户）　**执行：** Codex　**类型：** 短训练探针

## 核心（4 行）
- **目标**：leverage 已确认极大,现在验证 offloading actor **能否真的学起来** —— 用已知可行的 EFT-anchored credit,看 entropy 是否离开均匀、系统指标能否从 `random` 水平走向/超过 `eft_greedy`。这是 **Stage 1 的收尾判据**。
- **做什么**：复用仓库**现有**的 EFT guidance / auxiliary 机制(如 `marl_models/mappo/clean_eft_auxiliary.py`,或历史 EFT regret advantage —— 二选一并在 report 写明用了哪个、是否替换了 PPO 的 return-based advantage),接到 offloading head。env / PPO / state / reward 其余全不动。3 seeds,short gate。
- **看什么数**：① offloading entropy 轨迹(能否从 ~0.9998 明显下降);② 系统指标(completion / flowtime / throughput)对照 leverage check 的 **`random` ↔ `eft_greedy` 参考带**;③ **学出的策略是否超过 `eft_greedy`**;④ 与 `eft_greedy` 的**动作一致率(agreement rate)**。
- **通过**：entropy 明显下降 **且** 系统指标至少达到 `eft_greedy` 水平(理想:超过) → Stage 1 打通,进 HGNN vs MLP 正式对照;落在 random 与 greedy 之间 → 报数再议;完全不动 → actor 侧存在结构问题,另开诊断。

## Codex 执行要点
- **参考带必须复用** 2026-09-05 leverage check 的 `random` / `eft_greedy` 数值,且**评估口径一致**(同 forced-hover 或同一移动设定、同 seed 口径),否则"学到多少"无法量化。
- **单变量**:只开 EFT-anchored credit;LR / GAE / entropy 系数 / 网络 / state 全不动;gate-OFF 严格复现现基线。
- **关键补充指标 —— agreement rate**:学出的策略与 `eft_greedy` 的逐决策动作一致率。用于区分"单纯模仿 EFT"与"学到了 EFT 之外的东西"。**这是论文差异化的关键**:EFT greedy 是**myopic** 的(不考虑 DAG 依赖结构、后继任务的跨 UAV 传输代价、整条 DAG 的负载影响),而超图表示正是为捕捉这些关系设计的 —— 若学出的策略既超过 greedy 又与其一致率不高,那正是 HGNN 的故事。
- 3 seeds,short gate 规模;最终用 deterministic evaluation 出对照数。

## 工程门禁
freeze commit;记录版本(HEAD + dirty + 脚本 hash,**以 run 实际参数为准,不要信 config**);服务器后台跑,返回 PID / log / result path。

## 结果产物
- report：`docs/superpowers/reports/2026-09-06-stage1-eft-anchored-credit-probe-results.md`
- result.json + 曲线:entropy vs update、系统指标 vs `random`/`eft_greedy` 参考带、agreement rate。

## 不做
不接 Phase 4A credit 分支、不动 state / reward / PPO 结构、不跑正式长训练、本阶段**暂不做 HGNN vs MLP**。
