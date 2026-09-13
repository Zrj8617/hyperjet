# mini-spec: offloading leverage 前置检查（2026-09-05）

**状态：** APPROVED（用户）　**执行：** Codex　**类型：** 纯评估（无训练、无主线改动）

## 核心（4 行）
- **目标**：回答"卸载决策到底有没有 leverage" —— 固定移动策略下,好的卸载启发式比随机卸载在**系统指标**上强多少。这决定后续 credit 工作值不值得做,以及**论文前提是否成立**。
- **做什么**：同一环境、同一 seeds(CRN)、**同一移动策略**,只换卸载策略跑三档:`random`(均匀合法) / `eft_greedy`(最早预计完成) / `eft_worst`(最晚预计完成,用于框出 leverage 上下界)。**不训练**。
- **看什么数**：主 = DAG completion rate、avg DAG flowtime、throughput;辅 = energy/completed DAG、avg queue length、load balance。报 `eft_greedy vs random` 的相对差,以及 `best vs worst` 的跨度。
- **判读**：gap 大(flowtime 改善 ≥15% 或 completion 相对 ≥10%) → 卸载有 leverage,继续做 credit(转 estimand 重定义);gap 小(<5%) → 卸载 leverage 有限,**论文前提需重构**;中间 → 报数再议。

## Codex 执行要点
- **复用现成 harness**(`scripts/run_clean_policy_baseline.py` / `scripts/static_scheduler_compare.py` / `scripts/eval_clean_mainline.py` 里最合适的一个),不新造环境。
- EFT 分数**直接用** `environment/assignment.py: estimate_offloading_candidate` 已经算好的 `estimated_finish_time`:合法候选里取**最小**=greedy,取**最大**=worst。
- **移动策略三档必须完全相同**(冻结 hover 或现有 centroid 启发式,二选一,在 report 里写明选了哪个)。
- **配对**:≥5 个环境 seed,三档共用同一批 seed(CRN);deterministic evaluation。
- **无训练**:optimizer step = 0,参数逐 tensor 不变。

## 为什么现在做这个
Phase 4A 证明的是**单个决策**的 credit 低 SNR;但**整条策略**的 leverage 可以很大(单步小差异跨 500 slot 累积),两者不矛盾。本检查测 **policy-level leverage**。若"policy-level leverage 大 + per-decision credit 低 SNR",结论就是**不该继续做 per-decision credit**,而该走聚合信号路线(EFT-guided / DAG-local estimand)。

## 结果产物
- report：`docs/superpowers/reports/2026-09-05-offloading-leverage-check-results.md`
- result.json + 三档 × 各指标对比表;版本记录(HEAD + dirty + 脚本 hash)。

## 不做
不训练、不改环境 / reward / state / PPO、不接入任何 credit 分支、不改主线。
