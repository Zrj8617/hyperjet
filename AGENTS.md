# AGENTS.md — HyperUAV 协作规约（Codex 入口，长期稳定规则）

> 本文件只放**长期稳定的规则**：角色分工、单一事实源、任务交接约定、版本纪律、工程原则、禁止事项。
> **不放阶段性结论**（那些进 Living Charter），避免本文越来越臃肿。
> 任何 agent（Codex / Claude / 人）开始工作前，先读本文件 + 下面指定的单一事实源。

## 0. 单一事实源（Single Source of Truth）

- **唯一活动状态文档 = `docs/research/HyperUAV_research_master_roadmap.md`（Living Research Charter）。**
- 每个新窗口 / 每次较大改动 / 每次开实验前，**先读它**。
- **不允许新建第二份 master roadmap / handoff。** 其它类 handoff 文件（如 `docs/progress_report.md`）一律降级为指向本 charter 的指针。
- 状态更新只写进 charter，采用 append 一个带日期的“状态更新”小节，不覆盖历史决策日志。

## 1. 角色分工

- **Claude**：方案推理、论文叙事、实验设计与统计审核、写 spec、独立核对 Codex 的 diff/结果、维护 charter。
- **Codex**：源码核查、实现、服务器/本地执行、数据统计、版本一致性检查、写 report。
- **用户（Caiyurun）**：批准算法变更、批准是否进入下一阶段。**算法层面的改动和阶段推进必须经用户批准。**

## 2. 任务交接循环（轻量，复用 superpowers）

1. Claude 写 `docs/superpowers/specs/<date>-<name>-design.md`（**实验前冻结定义**：目标、单一变量、精确文件、gate-off 行为、样本量、指标、**停止/通过条件**、结果产物路径）。用户批准后冻结。
2. Codex 实现并执行；把结果写 `docs/superpowers/reports/<date>-<name>-results.md`（结果、结论边界、**原始 JSON/log 路径**）+ 一个机器可读 `result.json`。
3. Claude 独立用 `git diff` + 读 result/log 核对 → 裁决（accept/reject/next）→ 回填 charter。
4. 模板见 `docs/superpowers/specs/_TEMPLATE-spec.md` 与 `docs/superpowers/reports/_TEMPLATE-report.md`。

## 3. 版本纪律（强制）

- **本地 = 编辑源；服务器 = 执行副本；GitHub = 已冻结版本。** 三者可能不一致，以本文约定的记录为准。
- **任何正式实验，report 里必须记录：**
  - 实际 `git rev-parse HEAD`；
  - `git status --porcelain`（dirty files 列表，或其摘要 + 行数）；
  - 用到的诊断/训练脚本的版本标识（`git hash-object <script>` 或 commit + 是否 dirty）。
- 目的：杜绝“服务器/GitHub 是 A、本地是 B、且本地/服务器还有一致的未提交代码”这类状态歧义。
- 清理工作树前先确认哪些实验脚本/改动要保留；**不要直接清理未提交代码**。

## 4. 工程原则（沿用，硬约束）

1. 单变量实验。
2. gate OFF 必须严格复现旧行为。
3. RNG-neutral（诊断/采样 RNG 独立，不污染 environment/training RNG）。
4. 不在 `clean_trainer._loss()` 继续堆更多 credit branch。
5. Branch oracle 只做诊断，绝不作为训练 label（除非正式设计明确允许）。
6. 重要路线先 freeze commit。
7. decisive experiment 结束前不做大重构。
8. 长实验服务器后台运行，启动后返回 PID / log path / result path，不做 tail 循环。
9. deterministic evaluation 优先于直接比较 training trajectory。
10. Stage 1（可靠的 MLP clean baseline offloading credit）未完成前，**不进入 HGNN vs MLP 最终结论**。
11. **文档声明不是已验证事实。** 任何来自 docs/（交接文档、roadmap、旧 spec/report）或他人报告的
    **定量断言**，在被用作推论前提、写进新 spec、或作为 gate/断言条件之前，**必须先读代码或读日志
    验证**。验证不了的，必须在使用处显式标注为「假设，未坐实」，并且不得据此推出结论。
    本条对 Claude、Codex 和用户同等适用。
    起因：2026-09-15 门禁 G9 失败——「B2 三段账与 B1 实际 flowtime 数学上相等」这一写在交接文档
    §4.4 的声明被当作事实沿用，据此推出过「B1/B2 优化同一目标函数」的结论，并被写成硬停止断言；
    实测相对误差 0.489，该恒等式从未成立。详见
    `docs/superpowers/reports/2026-09-15-six-arm-realign-gating-results.md`。

## 5. 禁止重复的实验（已完成对应证伪，除非有新证据不得重开）

- 再验证 same-slot recursive bootstrap；
- 再跑 GAE target replacement；
- 再调 Decision-V / Decision-GAE；
- 用 Q EV 判断 action ranking 是否学会；
- 再验证 single-root vs multi-root；
- 再做 reward ablation / redesign；
- 再跑 vanilla PPO learnability gate；
- 靠调 entropy/LR 拖延 collapse；
- 扩大旧 Q-v2 训练；
- 直接把 C_local 接入训练；
- 直接把 H=80 单-alternative 均值接入训练；
- 直接修改 183 维 state（无直接 aliasing 证据前）；
- 扩大/加深 Boundary critic 硬救它；
- 现在讨论 HGNN vs MLP 最终优劣。

## 6. 同步约定

- Claude 通过本地文件夹桥直接读取本仓库；Codex 直接读写本仓库。
- 结果一律写到约定的固定路径，Claude 自行读取核对，尽量不经人肉搬运。
- 用户只需触发 Codex 任务并告知“已完成/已启动”，以及批准阶段推进。
