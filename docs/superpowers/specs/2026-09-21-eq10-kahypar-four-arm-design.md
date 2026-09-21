# EQ10 四臂 KaHyPar-on 三 seed 配对实验（2026-09-21）

**状态：** FROZEN（用户 2026-09-21 批准）  
**执行：** Codex  
**基线：** 2026-09-18 EQ10 四臂 × 双编码器实验中的
`typed_gated_hgnn`、KaHyPar-off 单元

## 1. 目标

在 2026-09-18 的统一能耗价格、场景、训练预算、老师矩阵、HGNN 容量和固定 tape 全部不变的
条件下，只把 KaHyPar partition hyperedges 从 OFF 改为 ON，测量 KaHyPar 对 B2/C2A/C2B/C2
四个臂的固定-tape 系统指标影响。

本实验回答的是完整 treatment `KaHyPar partition hyperedges ON − OFF`，不把观察到的效果进一步
拆分归因给 partition 算法、type gate 或某个单独门控权重。

## 2. 单一变量与配对基线

新增 treatment：

```text
task_encoder = typed_gated_hgnn
enable_kahypar = true
```

配对基线为现有 20260918 同 arm、同 seed 的：

```text
task_encoder = typed_gated_hgnn
enable_kahypar = false
```

除 `enable_kahypar`、由其派生的 partition/type-3 超边运行元数据，以及 run/output 标识外，
实际配置逐项 diff 必须为空。gate OFF 继续严格复现 20260918 行为；不得修改默认开关。

## 3. 实验矩阵

- arms：`B2 / C2A / C2B / C2`
- seeds：`5 / 86 / 617`
- encoder：全部 `typed_gated_hgnn`
- KaHyPar：全部开启
- 正式训练：4 arms × 3 seeds = **12 run**
- smoke：4 arms × seed 5 × 5 episodes = **4 cell**

正式 run 名称：

```text
20260921_<ARM>_TYPED_GATED_HGNN_KAHYPAR_EQ10_seed<SEED>
```

不得覆盖、重命名或写入 20260915/20260918 的任何训练或评测目录。

## 4. 冻结配置

### 4.1 奖励与老师

- `flowtime_ref = 500 s`
- `lambda_task = 1.0 s/J`
- `lambda_move = 1.0 s/J`
- B2：无老师
- C2A：卸载 EFT 老师开，飞行老师关
- C2B：卸载老师关，飞行位置老师开
- C2：两个老师都开
- C2A/C2B/C2：`teacher_anneal_total_updates=2000`，hold=0.50，end=0.60
- B2：`teacher_anneal_total_updates=0`

### 4.2 训练

- episodes 500；每 episode 500 slot
- rollout horizon 125；每 episode 4 update；全程 2000 update
- num envs 1；synchronous sampler
- lr 3e-4；gamma 0.99；GAE lambda 0.95；clip 0.2
- entropy coef 0.01；value coef 0.5；PPO epochs 1
- normalize value targets 开；checkpoint interval 10
- shared hidden dim 128；task embedding dim 64
- typed-gated encoder hidden dim 128
- `--no-dag-progress-potential-shaping`
- RNG-neutral encoder comparison 保持开启，reference MLP encoder hidden dim 773；共享 actor/critic
  初始化必须与 20260918 的容量对齐构造口径一致

### 4.3 场景与超边

- 场景逐项沿用 `param-realign-20260914` 的 20260918 实际配置
- DAG dependency、k-hop、attribute hyperedges 保持开启
- 本实验唯一新增：KaHyPar partition hyperedges 开启，正式训练和每个评测 batch 必须实际观察到
  type 3 / partition hyperedges
- 不改 environment、reward、PPO、state、teacher schedule 或其它超边定义

## 5. 实现边界

新增独立 smoke 与 production launcher，不修改已冻结的 20260918 launcher 的矩阵或命名。允许对通用
fair-eval orchestrator 做仅增加 20260921 显式 treatment/prefix 的兼容扩展；20260915 与 20260918
入口解析和行为必须保持不变。

所有新入口必须：

- 要求显式输出根、manifest 和 GPU 列表；
- 目标存在时拒绝覆盖；
- 记录 HEAD、dirty 列表/行数和相关脚本 git object；
- 从实际 `result.json` 回填配置，不从 `config.py` 推断正式 run 参数；
- 长任务后台启动后立即返回 PID/log/result path，不做 tail 循环。

## 6. Smoke 与硬门禁

正式训练前并行执行四个 5-episode smoke cell。全部满足才可启动 12 个正式 run：

1. 4/4 status completed，每 cell 20 update；无 NaN/Inf；
2. task encoder 为 `typed_gated_hgnn`，`enable_kahypar=true`；
3. 四臂 `lambda_task=lambda_move=1.0`；
4. B2/C2A/C2B/C2 的老师 flag 与第 4.1 节逐格一致；
5. 与对应 20260918 KaHyPar-off 单元的非 treatment 配置 diff 为空；
6. 参数量与 20260918 typed-gated HGNN 一致；
7. 共享模块初始化/RNG-neutral 检查通过；
8. KaHyPar 未出现 `degraded_*`、circuit-open、NaN/Inf 或无 type-3 的情况；
9. 旧结果目录无覆盖风险，服务器执行树 clean。

任一门禁失败立即停止，不启动剩余正式任务，不做近似降级。

正式训练期间任一 run 崩溃、checkpoint 缺失、配置不符或 KaHyPar 健康门禁失败，则该 run 标记 FAIL；
不得静默换参数重跑。

## 7. GPU 调度

启动前读取一次 `nvidia-smi`。优先使用完全空闲 GPU，再在显存有安全余量的 GPU 上按 20260918
实测密度安排多进程；不挤占显存不足或已有高负载且无安全余量的任务。目标是最短 makespan，
不是机械平均分配。

正式 launcher 接受显式、可重复 GPU assignment。全部后台进程挂起后停止监控并返回 PID、GPU、
log、result 和基于 smoke/历史墙钟的 ETA。

## 8. 固定 tape 评测

复用 20260915/20260918 已校验的 tape：

```text
/data2/zrj2025/uav-results/audits/20260915_six_arm_realign_tapes
```

- validation IDs 100–119
- test IDs 200–249
- joint validation 在 checkpoint 320/360/400/450/500 中按 `J_EQ10/offer` 最低选点，平局取早
- 所有 selection locks 落盘后才允许读取 test payload
- 同一 selected checkpoint 跑 joint 与 forced-hover test
- final episode 500 另行跑 joint 与 forced-hover
- 主表为 selected joint；其余为附表/稳定性检查
- 工作量：60 validation batch（1200 episode）+ 48 test batch（2400 episode）= 3600 episode

统一裁判：

```text
J_EQ10 = (actual censored delay + task energy + move energy) / 500
J_per_offer = J_EQ10 / max(N_offer, 1)
```

## 9. 指标与统计

主比较按同 arm、同 seed、同 tape 严格配对：

```text
B2_KaHyPar_ON  − B2_KaHyPar_OFF
C2A_KaHyPar_ON − C2A_KaHyPar_OFF
C2B_KaHyPar_ON − C2B_KaHyPar_OFF
C2_KaHyPar_ON  − C2_KaHyPar_OFF
```

报告：

- J/offer 及 delay/task-energy/move-energy 分量；
- admission、conditional completion、end-to-end completion；
- completed-DAG flowtime、throughput；
- hover ratio、m_action、m_effective；
- 每类超边 gate weight 的训练后 40%均值/标准差；
- KaHyPar health/status counts；
- 两层 bootstrap 95% CI 和 3/3、2/3、1/3 seed 方向；
- 每个 treatment 的 selected checkpoint 与 validation 分数；
- ON/OFF训练墙钟和吞吐；
- 每个 encoder treatment 内的 2×2 教师效应，作为机制描述。

本实验仍只有 3 个训练 seed，只能作为 KaHyPar 初步消融；不得升级为最终 HGNN headline 结论。

## 10. TensorBoard

训练结束后按既有统一命名规范导出：

```text
20260921_KAHYPAR_<ARM>_seed<SEED>
```

并增加 `20260921_KAHYPAR_SUMMARY`。不得混写或覆盖 20260915/20260918 event 目录。

## 11. 结果产物

- spec：`docs/superpowers/specs/2026-09-21-eq10-kahypar-four-arm-design.md`
- report：`docs/superpowers/reports/2026-09-21-eq10-kahypar-four-arm-results.md`
- machine-readable：`docs/superpowers/reports/2026-09-21-eq10-kahypar-four-arm-result.json`
- smoke：`/data2/zrj2025/uav-results/audits/20260921_eq10_kahypar_four_arm_smoke.json`
- launch manifest：`/data2/zrj2025/uav-results/audits/20260921_eq10_kahypar_four_arm_training_launch.json`
- training runs：`/data2/zrj2025/uav-results/audits/20260921_<ARM>_TYPED_GATED_HGNN_KAHYPAR_EQ10_seed<SEED>`
- fair eval：`/data2/zrj2025/uav-results/audits/20260921_eq10_kahypar_four_arm_fair_eval`
- fair-eval manifest：`/data2/zrj2025/uav-results/audits/20260921_eq10_kahypar_four_arm_fair_eval_manifest.json`

## 12. 不做

- 不重跑 MLP；不重跑 KaHyPar-off HGNN；
- 不新增 B1/C1；
- 不改能耗系数、teacher schedule、PPO、场景、tape或容量；
- 不做新的超边定义、KaHyPar参数搜索或事后择优；
- 不根据训练 reward 排名；
- 不在本轮下 HGNN vs MLP 最终结论。
