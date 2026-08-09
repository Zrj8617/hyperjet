# HyperUAV zrj_3 会话交接文档(Phase 4 进行中)

生成:2026-07-12。供下一个工作窗口无缝接续。当前 HEAD:`effb93e`(未推送)。

## 0. 协作模式与工作规则

- 三方协作:用户中转,Codex 与 Claude 互相交叉评审。**每轮先出方案/分析,用户确认后才动代码。**
- 铁律:最小 diff;单变量受控实验;诊断先行再改机制;每个 phase/修复独立 commit;stale smoke 修复必须单独 commit;接口改名破坏式、不留兼容 alias;归一化/物理常量只在 config 单点声明;每个修复先有会失败的测试或可复现数字。
- Gate 判定以 deterministic eval 为准,不用训练采样曲线;corr(reward, completion) 只作诊断不作通过标准。

## 1. 基础设施陷阱(必读)

1. **D:\CodeFile\HyperUAV 挂载同步会截断文件**,已发生 3 次(env.py 两次、clean_trainer.py 尾部一次)。尾部截断可能仍通过 ast 语法检查(如 `_jsonable` 丢了最后两行导致所有日志布尔变 null)。对策:写文件后 sleep+双读 hash 校验;验证在 `/tmp` 副本上跑;检查文件尾部标记(`if __name__` / `return value`)。
2. **挂载盘上 git index 不稳定**(曾损坏)。工作法:`export GIT_INDEX_FILE=/tmp/hyperuav_index && git read-tree HEAD` 后 add/commit;默认 index 过期时 `rm .git/index && git reset -q` 刷新。`core.filemode=false` 已设。
3. 沙盒无 torch(pytorch 源被代理 403)、无 kahypar;torch 验证只能在训练机(cuda,/data2/zrj2025/HyperUAV)做。
4. 沙盒后台进程约 2 分钟被回收;长批量任务用"完成标记跳过 + 反复拉起"的可续跑脚本。
5. `__pycache__` 可能陈旧,验证前清掉。

## 2. 场景与算法(定型部分)

500x500m、5 UAV、60 UE、500 slot x 5s、hotspot r=150 每 episode 采样一次;DAG 5-8 task 2-4 层,单 UE 单 active DAG;incidence HGNN(E_DAG/khop/attr/KaHyPar)+ 共享 movement actor(5 离散动作)+ 共享 offloading actor(顺序决策+reservation)+ 单 centralized critic + slot-level shared advantage PPO。
Phase 3 定参:DAG_BASE_ARRIVAL_PROB=0.0145, INPUT 0.75-14MB, OUTPUT 0.6-10.5MB, TASK_CONSTANT 6-60。
Phase 4 P1 定参:CLEAN_REWARD_TIME_REF=60.0, CLEAN_REWARD_TIME_CLIP=10.0(只 clip reward 的 norm_time,不动 metrics 原始值)。

## 3. 阶段史与关键数字

- **原始审查四大问题**:1) slot 序号当秒用(executor 每 slot 只推进 1s)-> Phase 1 修复,smoke_clean_time_units 红转绿;2) 负载不可行 -> Phase 3 标定;3) pair feature 被 /2500s 压扁 -> Phase 2 收口到 CLEAN_NORM_* 常量;4) legality 无通信距离约束(spec 要求,**至今未实现**,backlog)。
- **Phase 4 P0 基线**(14 seeds,runs/phase4_p0_baseline_200slot{,_drain}/):
  - 200-slot 无 drain:**random 0.5261±0.0989,greedy 0.7850±0.0991**(训练曲线对照锚点)。
  - drain<=300:两基线完成率均 1.0 -> **drain 口径判别指标是 flowtime(greedy 188s vs random 566s)与 drain 长度(62 vs 143)**,不是 completion。
- **P1 复跑 2x2**(runs/phase4_p1_rl_fixed/,freeze/learned x seed 42/1042,100ep x 200slot,ppo_epochs=3):
  - completion last20:freeze 0.538/0.513,learned 0.553/0.540 —— 全部约等于 random,gate 未过。
  - 数值修复成功:value_loss 4e5->约240,grad_norm 1e6->约2400,returns -4500->-350。
  - EV(每 episode 末条口径)-0.40~+0.13,critic 只学到均值。
- **机制诊断(修正后共识)**:
  - reward 仍"完成即罚":全轨迹罚奖比约 4.9(120-slot random 实测),边界采样 6.5-8.4;每 DAG 净 reward 为负有算术下界支持。待 episode 累计口径(C1' 新日志)最终定量。
  - **已撤回的推断**:"clip 压死 actor 学习率 6e-8"(Adam 对全局常数缩放近似不变);"HGNN 占总梯度 0.01%"(pre/post-clip 口径混用,post-clip 实占约 34%)。
  - 现行嫌疑:1) reward 结构错位 -> advantage 与目标近乎正交;2) shared slot-level advantage 对 10-20 个动作的信噪比低;3) HGNN 干路可能被 value 梯度主导(待 cosine 数据)。

## 4. 当前 git 状态(未推送,历史已重写)

    effb93e  Commit 2': freeze/learned RNG 对齐 + smoke_clean_freeze_rng_pairing
    227850c  Commit 1': 纯诊断 + 行为中性 smoke(含审查修复)
    7493fa5  修复同步截断的 env.py(a111f46 曾把截断版提交进历史)
    a111f46  中文注释快照
    71907c1  merge origin/zrj_3(服务器侧 torch checkpoint 修复等)

注意:旧 71f189f/3a4a49b 已被重写废弃;服务器若 fetch 过旧 commit 需强制对齐。工作区干净(仅 runs/ untracked)。

## 5. C1'/C2' 新增日志字段(下一窗口分析要用的字段名)

- episode 累计:episode_{reward,time_penalty,dag_bonus,task_energy_penalty,movement_energy_penalty}_so_far(中途边界)/ _total(终点边界)+ episode_terminal_record。
- horizon 末 counterfactual:unsettled_task_count / unsettled_dag_count / unsettled_delay_seconds_sum_estimate / unsettled_norm_time_cost_estimate(永不进 reward)。
- ppo_diagnostics(嵌套 dict):grad_pre_clip_{movement,offloading,critic,hgnn,global}、grad_post_clip_*、grad_clip_scale、grad_norms_epoch(=最后 epoch);hgnn_actor_grad_norm / hgnn_value_grad_norm / hgnn_actor_value_cosine / hgnn_decomposition_epoch(=0,每 hgnn_grad_decomposition_interval(默认5)次 update 一次,graph-free 时 norm=0、cosine=None);rollout_{offloading,movement}_entropy_normalized_mean(按 n_valid 归一化,n_valid<=1 排除)+ 平均候选数。
- 既有(P0):ppo_returns_mean/std, ppo_value_pred_mean, ppo_explained_variance, avg_uav_queue_length, active_dags, frozen_ready_task_count, offloading_skipped_no_candidate, movement_frozen。

## 6. 下一步(三方已共识的顺序)

1. **训练机**:跑 smoke_clean_diag_neutrality.py、smoke_clean_freeze_rng_pairing.py、smoke_clean_training_loop.py(torch 分支)+ 一个 --smoke 短训验证新字段;然后 push。
2. **Deterministic eval**:4 个 run 的 ep20/40/60/80/100 checkpoints,双口径——200-slot fixed completion(对 random 0.526 / greedy 0.785)+ drain flowtime(对 188s/566s)。
3. **定 w_c**:从新日志读 episode 累计罚奖比,REWARD_COMPLETED_DAG_WEIGHT 候选 10(可能 10-16),不预设锁死。
4. **第二实验臂**:由 hgnn_actor/value_grad_norm + cosine 决定(候选 value_coef 下调 / critic detach 干路;分模块 clip 优先级已因 Adam 缩放不变性下调)。
5. **受控实验**:w=2/global(已有 4 run 可复用)+ 补 w=w* 组;freeze/learned 现已 RNG 对齐可严格 A/B。
- Gate:deterministic 200-slot completion >=0.60 且 > random+1sigma、趋势单调;drain flowtime 显著 <566s。

## 7. 更远 backlog(策略 gate 通过后)

kahypar 安装(**急**,当前一切结果只能标 no-KaHyPar/degraded)、legality 通信可达性(spec 要求)、空间 task features + hotspot 观测 + movement shaping(只进 improved 配置,不进 baseline,除非用户改 spec)、critic value normalization(若 EV 持续 <0.3)、可视化 Phase 2-4(TensorBoard->Streamlit)、main 实验 300-500ep x 500slot x 3-5 seeds、final 5 seeds。

## 8. 数据位置

本地:runs/phase4_p0_baseline_200slot{,_drain}/(14 seed 基线,DIAG_JSON 行)、runs/phase4_p1_rl_fixed/(4 个 RL run)、runs/phase3*(旧标定)。服务器:/data2/zrj2025/HyperUAV/runs|logs/。标定记录:docs/clean_load_calibration.md。
