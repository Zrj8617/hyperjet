# HyperUAV 实验索引

## 当前阶段

Stage A：验证 HGNN / 超边结构在 pair feature 之外是否有独立贡献。

当前主线：
limited-pair feature 图贡献诊断。

当前不做：
- RL
- 联合训练
- 大规模 attribution
- 继续复杂 mixed hyperedge 调参

---

## 当前有效实验

| ID | 实验 | pair_feature_mode | ablation | seeds | checkpoint 目录 | 状态 | 结论 |
|---|---|---|---|---|---|---|---|
| E001 | limited no_hyperedge pretrain | limited | no_hyperedge | 42/43/44 | Result_Hyperuav/limited_pair_no_hyperedge_* | done | 内部小 eval 正常 |
| E002 | limited safe_hyperedge_only pretrain | limited | safe_hyperedge_only | 42/43/44 | Result_Hyperuav/limited_pair_safe_hyperedge_only_* | done | 内部小 eval 略好于 no_hyperedge |
| E003 | limited no_pair_hyperedge_score_feature pretrain | limited | no_pair_hyperedge_score_feature | 42/44 | Result_Hyperuav/limited_pair_no_pair_* | incomplete | 缺 seed43，暂不进主结论 |

---

## 下一步实验

正式 static compare：

比较：
- limited no_hyperedge
- limited safe_hyperedge_only

设置：
- NUM_UES=40
- seed=42/43/44
- episodes=100
- steps=200
- pair_feature_mode=limited
- no_attribution
- runtime_bounded_guard
- runtime_finish_tolerance=0.1
- no rescore

判断标准：
如果 limited safe_hyperedge_only 稳定高于 limited no_hyperedge，说明精简超边在 limited-pair 条件下具有弱结构贡献。
如果二者仍然差不多，说明当前图结构/超边还没有形成稳定独立收益。
