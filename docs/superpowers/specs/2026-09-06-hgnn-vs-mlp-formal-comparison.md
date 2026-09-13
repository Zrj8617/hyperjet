# mini-spec: HGNN vs MLP 正式对照（2026-09-06）· 论文核心命题

**状态：** APPROVED（用户）　**执行：** Codex　**类型：** 正式对照训练

## 核心（4 行）
- **目标**：回答论文核心问题 —— 在**完全相同**的 env / PPO / reward / **EFT-anchored credit** / 训练预算 / 评估口径下,**超图表示是否优于公平 MLP**。
- **做什么**：唯一变量 = **task encoder**(MLP vs 选定的一个 HGNN 变体作 headline)。**参数量容量对齐**。≥5 seeds,训练预算长于 probe 的 400 updates(跑到指标平台期)。
- **看什么数**：主 = completion / flowtime / throughput,对照 `random` / `eft_greedy` / `MLP` 参考带;辅 = entropy 轨迹、**agreement rate(各自 vs eft_greedy)**、energy/DAG、load balance;**加分层分析**(见下)。
- **判读**：HGNN 在主指标上稳定优于容量对齐 MLP(多数 seed 同向 + 效应量/CI) → 论文核心结论成立;若持平 → **如实报负结果**,转 hyperedge 类型消融查是否某类超边有效。

## Codex 执行要点
- **容量对齐是硬要求**：MLP baseline 的参数量需与 HGNN 近似,**report 中必须列出两者 param count**。否则"HGNN 赢"可被解释为"参数更多" —— 这是审稿人第一个会问的问题。
- **唯一变量 = encoder**：env / PPO 超参 / reward / credit 机制 / 训练预算 / 评估口径 / seeds 全部一致;做一次配置逐项 diff,除 encoder 外应为空,并在 report 贴出。
- **分层分析(论文关键,别省)**：按"结构敏感度"分层报告性能 —— 例如按 **DAG 深度、任务后继数、该决策是否涉及跨 UAV 父节点传输**。**HGNN 的优势应集中在结构敏感的决策上**;能指出"赢在哪里"远强于只报"赢了多少"。
- **agreement rate 对比**：HGNN 与 MLP 各自与 `eft_greedy` 的动作一致率。若 **HGNN 偏离 greedy 更多且结果更好**,这就是"超图捕捉到 EFT 与扁平 MLP 都看不见的结构信息"的直接证据 —— 论文最有力的机制性证据。
- **HGNN 变体**：从 `TASK_ENCODER_CHOICES` 里选一个作 headline(其余留作后续消融),report 写明选了哪个及理由。
- **统计**：≥5 seeds;报均值 ± CI 与效应量,不只报点估计;明确说明 3-5 seeds 的结论强度边界。

## 工程门禁
freeze commit;版本记录(HEAD + dirty + 脚本 hash,**参数以 run 实际值为准**);服务器后台跑,返回 PID / log / result path。

## 结果产物
- report：`docs/superpowers/reports/2026-09-06-hgnn-vs-mlp-formal-comparison-results.md`
- result.json + 曲线:主指标 vs 参考带、entropy、agreement rate、分层分析表。

## 不做
本阶段不做 hyperedge 类型消融(下一阶段)、不改 credit 机制、不改 env / reward / PPO 结构。
