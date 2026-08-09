# 交给 Codex 的执行提示词（P0：Stage 1 长训练 + 冻结 tape 复评）

> 下面整块内容直接复制给 Codex。已写死所有参数与改动点，不需要它探索代码库。

---

## TASK

执行 Stage 1 的 P0 实验：**在完全不改变模型输入、网络结构、奖励和环境的前提下，
把训练强度提高约 33 倍，然后在已冻结的 tape 上复评。**

目的：验证一个预注册假设——
`deterministic_margin20_accuracy` 当前为 0.643/0.661/0.672，
若训练充分后仍停在 0.75 以下，则确认瓶颈是 pair 特征的
`clip(incremental_delay / 40, 0, 1)` 截断；若升到 0.90 以上，则该假设被推翻。

**严格执行范围内的改动，不要重构、不要探索无关代码、不要写范围外的测试。**

---

## 背景事实（已核实，不需要再验证）

- `optimizer_step_count` 每个 update 恒为 3（`ppo_epochs` 决定，chunk 是梯度累积不是独立步）
- 原训练全程仅 **90** 次参数更新、约 6000 个决策，模型约 36800 参数
- 全部 30 个 update、3 个 seed：`clip_fraction` 恒为 0.0000，`approx_kl` ≈ 1e-4
- 评估侧 `normalized_entropy` = 0.985–0.993（均匀 = 1.0），采样策略几乎随机
- 结论：策略基本没离开初始化，训练强度不足是独立于 clip 的第二个主因

---

## 一、代码改动（共 3 处，全部是最小改动）

### 改动 1 —— `scripts/train_decision_ppo_bandit_gate.py`，函数 `_validate_args`

解除对 PPO epoch 数的硬限制：

```python
# 原
    if int(args.ppo_epochs) != 3:
        raise ValueError("Stage 1 requires exactly three PPO epochs")
# 改为
    if int(args.ppo_epochs) <= 0:
        raise ValueError("ppo-epochs must be positive")
```

其余校验全部保留，**特别是 `DECISION_BANDIT_REGRET_SCALE` 的断言不得改动**。

### 改动 2 —— `environment/stage1_temperature_diagnostic.py`

保留现有 `FROZEN_CHECKPOINTS` **原封不动**（保证 `20260806_formal_v1` 仍可复现），
在其后新增：

```python
# 训练完成后由改动 2b 填入：{seed: (relative_path, sha256, completed_update)}
FROZEN_CHECKPOINTS_LONG: dict[int, tuple[str, str, int]] = {}

CHECKPOINT_SETS = {
    "formal_v1": FROZEN_CHECKPOINTS,
    "long_v1": FROZEN_CHECKPOINTS_LONG,
}
```

修改 `load_frozen_checkpoint` 签名与两处硬编码：

```python
def load_frozen_checkpoint(
    checkpoint_path, *, training_seed, device="cuda",
    registry=None, expected_completed_update=30,
):
    ...
    registry = FROZEN_CHECKPOINTS if registry is None else registry
    if seed not in registry:
        raise ValueError("checkpoint training seed is not frozen")
    entry = registry[seed]
    expected_relative, expected_hash = entry[0], entry[1]
    ...
    # 元数据校验里把字面量 30 换成参数
    or int(payload.get("completed_update", -1)) != int(expected_completed_update)
```

**其余逻辑（SHA-256 校验、路径后缀校验、`encoder == "mlp"`、`graph_dim == 12`）一律保留。**

### 改动 2b —— 训练结束后回填

对每个 seed 的 `checkpoints/checkpoint_update_0300.pt` 计算 SHA-256，
填入 `FROZEN_CHECKPOINTS_LONG`，`completed_update` 填 `300`。

### 改动 3 —— `scripts/run_stage1_temperature_followup.py`

新增参数并改造 checkpoint 遍历：

```python
parser.add_argument("--checkpoint-set", choices=("formal_v1", "long_v1"), default="formal_v1")
```

```python
# 原
for training_seed,(relative_path,_) in FROZEN_CHECKPOINTS.items():
    checkpoint=args.checkpoint_root/relative_path
    encoder,actor,checkpoint_meta=load_frozen_checkpoint(checkpoint,training_seed=training_seed,device=args.device)
# 改为
registry = CHECKPOINT_SETS[args.checkpoint_set]
for training_seed, entry in registry.items():
    relative_path = entry[0]
    expected_update = entry[2] if len(entry) > 2 else 30
    checkpoint = args.checkpoint_root / relative_path
    encoder, actor, checkpoint_meta = load_frozen_checkpoint(
        checkpoint, training_seed=training_seed, device=args.device,
        registry=registry, expected_completed_update=expected_update,
    )
```

把 `--checkpoint-set` 写进 `run_manifest.json` 的 summary 字典。

---

## 二、绝对不许改的东西

- `_normalize_pair_features`、`CLEAN_NORM_AVAIL_TIME_REF`、`CLEAN_NORM_PAIR_TIME_REF` 及任何 `CLEAN_NORM_*`
- `DECISION_BANDIT_REGRET_SCALE`、reward 任何部分、`CLEAN_REWARD_*`
- `entropy_coef`（保持 **0.0**，不要加熵奖励，当前问题是熵过高不是过低）
- actor 输入维度、`CleanOffloadingActor`、`SharedOffloadingCandidateScorer`、encoder 结构
- `hidden_dim=128`、`task_embedding_dim=64`、`encoder="mlp"`
- 环境、容量、`DAG_BASE_ARRIVAL_PROB`、`CLEAN_MAX_QUEUE_PER_UAV`、`active_dag_cap`
- 冻结 tape、`_ready_sort_key`、PPO/critic/GAE 任何路径
- 已有目录 `logs/stage1_temperature_followup/20260806_formal_v1/` 与 `runs/phase4_p0_baseline_200slot/`
- **不要进入 Stage 2**

---

## 三、要跑的实验

### E0 —— 计时探针（先跑，用于估算总耗时）

单个 seed，`--pilot`（自动限制为 2 个 update）：

```
--group S1-B --seed 42 --pilot --updates 2 --ppo-epochs 10 --slots-per-update 256 --run-name stage1_timing_probe
```

记录单 update 墙钟时间，外推 300 个 update 的总耗时。
**若单 seed 预计超过 12 小时，先停下来报告，不要直接开跑 E1。**

### E1 —— 长训练（3 个 seed）

对 `seed ∈ {42, 86, 1042}` 各跑一次，其余参数完全相同：

```
--group S1-B --seed <SEED>
--updates 300
--ppo-epochs 10
--slots-per-update 256
--lr 3e-4                 # 本轮不动学习率
--clip-ratio 0.2
--max-grad-norm 0.5
--chunk-decisions 64
--task-embedding-dim 64
--hidden-dim 128
--completed-dag-weight 16.0
--checkpoint-updates 0,30,100,200,300
--run-name stage1_long_v1
```

优化步数：300 × 10 = **3000**（原 90，33 倍）。
环境交互：300 × 256 = **76800** slot（原 3840，20 倍）。

**中途中止条件**（任一连续 5 个 update 成立即停并报告）：
- `approx_kl` > 0.05
- `clip_fraction` > 0.35
- `behavior.raw_eft_regret_mean` 相比前 20 个 update 均值上升超过 50%

### E2 —— 冻结 tape 复评

回填 `FROZEN_CHECKPOINTS_LONG` 后，用**与 `20260806_formal_v1` 完全相同的 tape**跑：

```
--phase formal --checkpoint-set long_v1
--tape-dir <与 20260806_formal_v1 相同>
--temperatures 1.0 0.75 0.5 0.25
--sampling-replicates 0 1 2 3 4
--scenario-indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
--max-physical-slots 200
--output-dir logs/stage1_temperature_followup/<新目录>
```

随后用 `scripts/analyze_stage1_temperature_followup.py` 生成分析 JSON。

### E3 —— 负载与饱和重测（离线，零 GPU）

用 `analysis_inbox/summarize_static_corpus.py`（已存在，不要改）
对 E2 新产出的 `static_corpus/records.jsonl` 生成汇总，与旧汇总对照。

---

## 四、预注册判据（跑之前就固定，不许事后调整）

### 主判据 —— 决定下一步

`deterministic_margin20_accuracy`（3 个 seed）：

| 结果 | 结论 | 下一步 |
| --- | --- | --- |
| **< 0.75** | clip 天花板确认 | 授权进入归一化修改设计 |
| **≥ 0.90** | clip 假设被推翻 | **停止**归一化计划，重新分析 |
| 0.75 – 0.90 | 部分成立 | 停下来报告，不要自行决定 |

### 辅助判据

1. **集中度**：`normalized_entropy` 应显著低于 0.985；`max_action_probability` 应显著高于 0.36
2. **闭环**：T=1.0 的 `completed_dag_count` 应达到或超过旧 T=0.25 水平
   （旧值按 checkpoint 为 56.15 / 62.07 / 63.06）
3. **训练健康度**：`clip_fraction` 应进入 0.05–0.20，`approx_kl` 应进入 0.005–0.02
   - **若两者仍 ≈ 0** → 步子依然太小，**这是本轮唯一允许的第二次调整**：
     `--lr 1e-3` 重跑 E1，其余参数不变
   - 若 `clip_fraction` > 0.3 或 `approx_kl` > 0.05 → 步子过大，降到 `--lr 1e-4`
4. **温度效应缩小是预期结果**：旧 T=0.25 相对 T=1.0 有 +19.9% 吞吐，
   训练充分后这个差距应显著缩小甚至消失。**这是好事，不要当成回归。**

---

## 五、要交付的东西

1. 三个 seed 的 `updates.jsonl` 与 `config.json`
2. E2 的 `run_manifest.json`、`analysis/*.json`、`closed_loop/episodes.jsonl`
3. E3 的新旧负载汇总对照
4. 一份 `summary.json`，至少包含：

```jsonc
{
  "optimizer_step_count_total": 0,          // 每 seed
  "deterministic_margin20_accuracy": {},    // 每 checkpoint
  "normalized_entropy": {},
  "max_action_probability": {},
  "completed_dag_count_by_temperature": {},
  "clip_fraction_final": {},
  "approx_kl_final": {},
  "saturation_rate_overall": {},            // 新 vs 旧
  "min_legal_incremental_delay_p50": {},    // 新 vs 旧
  "primary_verdict": "clip_ceiling_confirmed | clip_hypothesis_rejected | inconclusive"
}
```

5. 用 `completed_dag_count`（不是 `dag_completion_rate`）做主指标，
   新旧对比走冻结 tape 的**配对**比较
   （`generated_dag_count` 依赖策略，会污染完成率的分母）

---

## 六、结束条件

主判据落定并产出 `summary.json` 后**立即停止**。
不要自行开始归一化修改、不要碰超图、不要进入 Stage 2、不要调 reward 或环境参数。
