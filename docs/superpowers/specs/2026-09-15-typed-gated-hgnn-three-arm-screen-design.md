# SPEC: 2026-09-15 typed-gated HGNN 三臂配对筛选

**状态：** FROZEN（用户 2026-09-15 批准；明确取消 smoke，直接运行 9 个正式 run）
**提出与批准：** 用户　**设计与执行：** Codex
**基线协议：** `docs/superpowers/specs/2026-09-15-six-arm-realign-training.md`

## 1. 目标与结论边界

在新场景、同一奖励/教师臂和同一训练协议下，初步回答：

> 把任务表示从独立 MLP 换成启用四类超边的完整 `typed_gated_hgnn` 后，B2、C1、C2
> 三个臂的固定 tape 系统指标是否出现一致的配对变化？

本轮是 **3-seed 筛选实验**，不是容量对齐的正式 HGNN vs MLP headline 实验：

- 可以报告每个臂内 `typed_gated_hgnn + KaHyPar − mlp` 的配对效应、CI 和 seed 方向；
- 不得据此宣布 HGNN 最终优于或不如 MLP；
- 不得把观察到的效应单独归因给 type gate、类型权重或 KaHyPar partition；
- 不得跨 B2/C1/C2 比较训练奖励；
- 若结果值得继续，再单独设计至少 5 seeds、容量对齐的正式验证。

## 2. 实验矩阵

只新增 9 个训练 run：

| 奖励/教师臂 | task encoder | seeds | run 数 |
|---|---|---|---:|
| B2 | `typed_gated_hgnn` | 5、86、617 | 3 |
| C1 | `typed_gated_hgnn` | 5、86、617 | 3 |
| C2 | `typed_gated_hgnn` | 5、86、617 | 3 |

配对 MLP control 直接使用正在执行的第二阶段六臂实验中相同 arm、相同 seed 的 9 个 run，
只读、不重跑、不覆盖。C2A、C2B、B1 不增加 HGNN run。

## 3. 单一 treatment 与冻结项

### 3.1 单一 treatment

```text
MLP control: task_encoder=mlp, enable_kahypar=false
HGNN treatment: task_encoder=typed_gated_hgnn, enable_kahypar=true
```

完整 treatment 使用现有四类超边 type ID：DAG dependency、k-hop、attribute、KaHyPar
partition；四类权重从相同值初始化，并通过现有 typed weight + residual gate 实现学习。本轮不修改
HGNN 数学定义、不增加层数、不做超边消融。

MLP 前向明确忽略 incidence matrix 和 hyperedge type IDs，因此不重跑正式 MLP control。本轮把
HGNN 的 encoder、typed weighting、residual gate 与 KaHyPar partition 共同定义为一个完整 treatment，
不声称“只开启 KaHyPar 对 MLP 的训练轨迹必然逐值无影响”。

### 3.2 冻结训练参数

下列参数必须与对应 MLP run 的**实际记录**相同，而不是从 `config.py` 或旧文档推断：

- episodes：500；每 episode 500 slot；
- rollout horizon：125；
- seeds：5、86、617；
- `num_envs=1`，`sampler_backend=synchronous`；
- lr 3e-4，gamma 0.99，GAE lambda 0.95，clip ratio 0.2；
- entropy/value coef：0.01 / 0.5；PPO epochs：1；value target normalization 开；
- hidden dim 128，task embedding dim 64；
- C1/C2 teacher anneal raw CLI 必须保持 `teacher_anneal_total_updates=null`；训练器按实际
  `500 × ceil(500/125)` 解析为 2000 updates；resolved hold/end 为 0.50/0.60；
- B2 无卸载教师、无飞行教师；C1/C2 的 resolved reward/teacher flags 保持原定义；
- DAG progress potential shaping 关闭；checkpoint interval 10；
- 场景参数、环境、奖励、actor、critic、optimizer、评估协议全部冻结。

launcher 门禁分别比较 raw CLI 与 resolved teacher clock，不把 `null` 和 2000 当作同一层字段。
“同参数”指除上述表示 treatment 外，相同训练超参数和相同网络宽度；不声称 MLP 与 HGNN 的
trainable parameter count 相同。报告必须列出 encoder 和全模型的实际 trainable parameter count。

## 4. 最小实现设计

### 4.1 runner 开关

在 `scripts/run_reward_redesign_arm.py` 增加显式 `--task-encoder` 与 `--enable-kahypar`；默认仍为
`mlp` / false，并原样传给 `train_clean_mainline.py`。默认不传时，现有
B1/B2/C2A/C2B/C2/C1 行为必须逐字节保持不变。HGNN launcher 必须显式传
`--task-encoder typed_gated_hgnn --enable-kahypar`。

### 4.2 RNG-neutral 初始化

当前训练器在 task encoder 之后构造 movement actor 和 centralized critic；不同 encoder 参数量
会消耗不同数量的 Torch RNG，从而连带改变这些共享模块的初始权重。HGNN 比较路径必须隔离这一
副作用：

- MLP 默认路径完全不改；
- typed-gated 比较路径在全局 RNG 流中消耗与原 MLP encoder 相同的初始化序列；
- typed-gated encoder 本身在独立、可复现的 Torch RNG context 中初始化，精确冻结为
  `typed_encoder_seed = training_seed`；
- 同 seed 下，offloading actor、movement actor、critic 等所有共享模块的初始 state dict 必须
  与原 MLP 构造路径逐张量相等；不新增 hash/SHA 校验逻辑。
- 完成全部模型构造后，HGNN 路径的全局 Torch RNG state 必须与 MLP 路径逐值相同。

具体实现应采用最小、显式 gate，不能改变仓库中其它 HGNN run 的默认初始化语义。

### 4.3 KaHyPar checkpoint 与评估恢复

训练 config snapshot 已记录 `experiment_controls.enable_kahypar`，但当前
`checkpoint_experiment_controls()` 没有把它返回给评估链。最小修正必须：

- 在 checkpoint control 解析中校验并返回布尔字段 `enable_kahypar`；
- resume 时拒绝 requested/saved KaHyPar 开关不一致；
- `scripts/run_fair_eval_batch.py` 在创建 `CleanGraphBuilder` 前，把运行时 config 恢复为 checkpoint
  中的 `enable_kahypar=true`，并在输出中记录实际值和 partition status counts；
- MLP 默认 checkpoint 缺省语义仍为 false，旧 checkpoint 评估行为不变；
- validation 与 test 均由 checkpoint 恢复该值，禁止依赖服务器 `config.py` 默认值。

### 4.4 独立 launcher

新增 `scripts/launch_20260915_typed_gated_hgnn_three_arm_screen.py`，只生成本矩阵的 9 个
target，不扩写现有六臂 MLP launcher。launcher 必须：

- 在启动前读取对应 MLP run 的实际 controls，并逐字段核对冻结项；
- 检查目标目录均不存在；任何一个已存在即停止，不做覆盖或自动换名；
- 记录服务器实际 HEAD、dirty 文件列表/行数，以及 runner、launcher、trainer、HGNN 实现、
  reward ledger、评估/汇总脚本的 `git hash-object`；
- 接 TensorBoard；启动后只确认一次 PID 存活，不持续监控。

### 4.5 三臂评估与配对汇总

新增两个 dated 文件，不修改现有六臂 orchestrator 的输入契约：

1. `scripts/orchestrate_20260915_typed_gated_hgnn_three_arm_screen.py`
   - 固定输入：HGNN launch manifest、tape directory、arms `B2,C1,C2`、seeds `5,86,617`、
     run prefix `20260915_TYPED_GATED_HGNN`；
   - 只解析 `20260915_TYPED_GATED_HGNN_<ARM>_seed<SEED>`；
   - selection 前允许读取共享 `manifest.json` 的 schema、scene、version、splits 元数据与 validation
     条目，并只打开 validation payload；此时不得校验/解析 test 条目对应的文件，更不得打开 test
     payload；
   - validation 完成后先原子写每 run selection lock；所有 locks 落盘后才校验 test split/条目并打开
     test payload；
   - 输出中央 selection manifest 与每 run validation/test JSON。
2. `scripts/summarize_20260915_typed_gated_hgnn_three_arm_screen.py`
   - 固定输入：HGNN selection/evaluation manifest、只读 MLP 六臂 evaluation manifest、两组 launch
     manifest；
   - 只做同 arm、同 seed、同 tape ID、同协议的 HGNN−MLP 配对；
   - 输出第 10 节的独立 Markdown/JSON 报告，不写入 MLP 结果目录。

准确输出路径冻结为：

```text
/data2/zrj2025/uav-results/audits/20260915_typed_gated_hgnn_three_arm_fair_eval/
  orchestrator_manifest.json
  selection_manifest.json
  <ARM>/seed<SEED>/validation/checkpoint_ep<EP>.json
  <ARM>/seed<SEED>/selection_lock.json
  <ARM>/seed<SEED>/test/selected_joint.json
  <ARM>/seed<SEED>/test/selected_forced_hover.json
  <ARM>/seed<SEED>/test/final_ep0500_joint.json
  <ARM>/seed<SEED>/test/final_ep0500_forced_hover.json
```

orchestrator 启动时记录 `shared_tape_manifest_read=true`，但在 selection locks 全部落盘前必须保持
`test_tape_payload_read_count=0`；锁定后才校验共享 manifest 中的 test split/条目，随后打开 test
payload 并启动 test jobs。不得声称共享 manifest 在 selection 前未读取；任何 payload 顺序违例硬停止。

## 5. 结果隔离与命名

沿用已有服务器结果根 `/data2/zrj2025/uav-results/audits`，不新建另一棵结果目录树。

新增 run 目录严格命名为：

```text
20260915_TYPED_GATED_HGNN_B2_seed5
20260915_TYPED_GATED_HGNN_B2_seed86
...
20260915_TYPED_GATED_HGNN_C2_seed617
```

因此它们与 MLP 的 `20260915_<ARM>_seed<SEED>` 目录物理隔离。统一元数据：

- launch manifest：`/data2/zrj2025/uav-results/audits/20260915_typed_gated_hgnn_three_arm_screen_launch.json`
- launcher log：`/data2/zrj2025/uav-results/audits/20260915_typed_gated_hgnn_three_arm_screen_launcher.log`
- 每 run 日志：同一结果根下 `<RUN_NAME>.log`
- 每 run 结果：`<RUN_DIR>/result.json`
- TensorBoard：`<RUN_DIR>/tensorboard/`

不得写入、重命名或删除任何 MLP run 目录。

## 6. GPU 调度

- 当前六臂 MLP 正式训练优先；本轮不得与其争抢已占用的 7 张 GPU。
- 所有训练和评估只在服务器执行，本地只做静态检查和编辑。
- 不修改 MLP 正在使用的服务器执行副本；HGNN 使用独立执行 worktree
  `/data2/zrj2025/HyperUAV-typed-gated-hgnn-20260915`。
- 实施时先做一次服务器 PID/显存检查。
- 若 MLP 仍在运行，使用独立 dated queue launcher 等待那些已记录的 MLP PID 结束后再启动，
  不由 Codex 循环轮询；返回 queue PID、log path、最终 result paths 和基于现有 run 的 ETA。
- GPU 释放后用满 7 张卡：前 7 个 run 各占一卡，剩余 2 个放到显存余量最大的卡上；实际分配
  以启动时 `nvidia-smi` 为准并写入 manifest。
- 9 个正式 run 挂上后停止监控。

## 7. 评估协议

复用第二阶段六臂实验为同一新场景生成的固定 tape；这是配对设计所需的共同外生 realization，
不得另生成一套 HGNN 专属 tape，也不得使用更旧场景的 tape。replay 前必须从实际 tape manifest
重新校验场景参数。

这批 tape 的 generation HEAD 为 `2450a40`，HGNN execution HEAD 必然是新的 frozen implementation
commit；两者允许不同，但不得删除版本兼容校验。新 orchestrator 必须执行以下替代门禁：

- tape manifest 的 generation worktree dirty 列表必须为空；
- manifest schema、scene parameters、validation/test split 和每个已打开 payload 的
  `generation_context` 必须相互一致并符合本 spec；
- tape manifest 记录的 `config.py`、`scripts/generate_fair_eval_tapes.py`、
  `environment/exogenous_tape.py` 三个 git object 必须分别等于 frozen implementation commit 中对应
  文件的 git object；
- generation HEAD 只作为 provenance 记录，不要求等于 HGNN execution HEAD；
- selection 前只对 validation split/条目/payload 执行上述校验；test split/条目/payload 的校验严格
  延后到所有 selection locks 落盘之后。

每个 HGNN run 独立执行：

1. 只用 joint validation tape 100–119，在 checkpoint 320/360/400/450/500 中最小化
   `J_per_offer`，平局取更早 checkpoint；
2. 写入 selection lock 后，才允许读取 test tape 200–249；
3. 对锁定 checkpoint 和 `final_ep0500` 分别跑 joint 与 forced_hover test；
4. test 结果不得反向改变 checkpoint。

`J_episode`、`J_per_offer`、三档 lambda_move 离线敏感性、hover ratio 和 bootstrap 口径全部复用
第二阶段六臂训练 spec，不改变权重或定义。

每个 validation/test batch 必须从 checkpoint 解析出
`task_encoder=typed_gated_hgnn, enable_kahypar=true`，并把两者写入 batch JSON；任一不符立即停止。

## 8. 门禁与硬停止

用户明确取消任何 episode smoke。正式启动前只执行不推进环境、不做 PPO update 的静态/构造级
preflight：

1. MLP 默认 runner 的 resolved argv 与改动前基准一致；
2. 同 seed 的 MLP/typed 构造中，除 task encoder 外的共享模块初始参数逐张量相同；
3. typed encoder seed 等于 training seed，且完整模型构造后的全局 Torch RNG state 与 MLP 路径相同；
4. B2/C1/C2 的 resolved reward/teacher flag 与已通过的门禁表逐格一致；raw teacher CLI 均为
   `null`，C1/C2 resolved clock 为 2000，B2 无教师；
5. 目标目录全部不存在，MLP control 目录只读且完整；
6. 服务器工作树、脚本对象版本和实际启动参数完整记录。

首次实际前向、反向、checkpoint round-trip、type 3 观测与 KaHyPar 健康检查均发生在 9 个正式 run
中，不另跑 smoke；任一失败按下述硬停止规则处理，不得启动替代 run。

KaHyPar 健康门禁适用于 9 个正式训练 run 及每一个 validation/test batch：

- 分别记录完整 `kahypar_partition_status_counts`、`kahypar_circuit_open` 和观察到 type 3 partition
  hyperedge 的 slot/episode 数；
- 任一 `degraded_*` count 大于 0、`kahypar_circuit_open=true`，或整个 run/batch 从未观察到 type 3，
  该 run/batch 立即标记 FAIL，不得计入技术 PASS；训练 FAIL 后不得进入评估，validation FAIL 后不得
  selection，test FAIL 后不得汇总；
- `disabled` 只允许对应活动任务数不足 2 的 slot；`no_base_hyperedges` 与 `cache_interval` 保持现有
  正常语义，并分别计数，不把它们误判为降级。

出现以下任一情况立即停止，不近似降级：

- 任一冻结参数或 resolved flag 不匹配；
- 共享模块初始化不一致；
- 需要修改 `environment/`、奖励定义、教师退火表或固定 tape；
- tape manifest 场景校验失败，或 test tape 在 selection lock 前被读取；
- typed-gated 前后向出现 NaN/Inf、正式 run 崩溃或 checkpoint 缺失；
- 任一正式训练或评估 batch 的 KaHyPar 健康门禁失败；
- 存在覆盖/混写 MLP 目录的风险；
- 当前 MLP 训练仍占满 GPU 且安全排队无法建立。

实现及本地静态检查通过后直接提交 frozen implementation commit，并同步到上述独立服务器 worktree。
服务器只执行同样的静态/构造级 preflight；满足 `HEAD == frozen implementation commit` 且
`git status --porcelain` 为空后，直接启动 9 个正式 run。正式 manifest 记录该 commit 和所有改动/
执行脚本的 git object hash。

技术 PASS 定义为：上述门禁全部通过、9/9 正式训练完成且所有规定 checkpoint/评测结果完整。
科学结果不设置为了“让 HGNN 过关”的单一阈值；根据三个 arm 内的配对 `J_per_offer`、真实系统
指标、CI 和 seed 方向归类为 promising / mixed / no improvement。只有结果在至少两个臂上方向稳定、
且没有由 completion/flowtime/energy trade-off 明显推翻时，才建议进入容量对齐的正式验证；是否
推进仍由用户批准。

## 9. 汇总与判读

主比较只在 arm 内进行：

```text
B2_typed − B2_mlp
C1_typed − C1_mlp
C2_typed − C2_mlp
```

每项报告：

- `J_per_offer` 及 delay/task-energy/move-energy 分量；
- admission、conditional、end-to-end completion；
- completed DAG flowtime 均值/中位数、throughput；
- joint hover ratio、forced-hover 自检；
- episode 300–500 的训练稳定性与 entropy，仅作机制描述；
- `grad_clip_scale` 分布，以及 encoder/movement actor/offloading actor/critic 的 pre/post-clip gradient
  norm；用于区分表示变化与全局 gradient clipping 比例变化，不改优化算法；
- 两层 bootstrap 95% CI 和 3/3、2/3、1/3 seed 方向；
- HGNN/MLP 参数量和训练 wall-clock/throughput。

三臂方向一致可表述为“typed-gated 在本次 3-seed 筛选中表现一致”；方向不一致应按臂分别报告。
无论结果如何，不得把 3-seed、非容量对齐结果升级为正式 HGNN 优劣结论，不得把效果单独归因给
type gate、类型权重或 KaHyPar，也不得在三个臂或三个 HGNN 变体中事后择优。

## 10. 产物

与 MLP 报告分开写：

- report：`docs/superpowers/reports/2026-09-15-typed-gated-hgnn-three-arm-screen-results.md`
- machine-readable：`docs/superpowers/reports/2026-09-15-typed-gated-hgnn-three-arm-screen-result.json`
- 原始训练与 TensorBoard：保存于第 5 节的独立 HGNN run 目录
- validation/test JSON、orchestrator manifest 和 selection locks：保存于第 4.5 节的独立 fair-eval
  目录

报告必须记录实际服务器版本、dirty 状态、脚本 git object hash、9 个 run 的实际参数、PID/log/
result path、checkpoint 选择、完整评估表、配对效应及本节规定的结论边界。

## 11. 明确不做

- 不重跑或覆盖现有 MLP control；
- 不增加 B1、C2A、C2B 的 HGNN 臂；
- 不跑 `current_mean_hgnn` 或 `standard_weighted_hgnn`；
- 不改 environment、reward、teacher schedule、PPO 或场景；
- 不运行任何额外 episode smoke；
- 不做超边类型消融、参数量匹配或新增 seeds；
- 不用 test 选 checkpoint；
- 不从本轮结果下 HGNN vs MLP 最终论文结论。
