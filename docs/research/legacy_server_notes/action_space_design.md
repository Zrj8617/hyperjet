# 缝合方案的动作空间设计

## 当前Multi-UAV的动作空间

### 动作维度: 2D连续动作
```python
ACTION_DIM: int = 2  # (delta_x, delta_y) 范围 [-1, 1]
```

**含义**:
- 输出: 2维向量 (Δx, Δy)
- 归一化到 [-1, 1]
- 实际移动距离 = 向量模长 × UAV_SPEED × TIME_SLOT_DURATION

**优点**:
- 简单直观
- 连续动作空间,适合DDPG/TD3/SAC
- 灵活性高

**缺点**:
- 只控制轨迹,缺少任务卸载决策
- 缓存策略是启发式算法(GDSF),不是学习的

---

## 🔧 缝合后的动作空间设计

### 方案一: 混合动作空间 (推荐)

#### 动作组成:
```python
# 1. 轨迹控制 (连续动作)
trajectory_action = [delta_x, delta_y]  # 范围 [-1, 1]

# 2. 任务卸载决策 (离散动作)
offloading_action = [
    offload_locally,      # 0/1: 是否本地处理
    offload_to_neighbor,  # 0/1: 是否卸载到邻居无人机
    offload_to_mbs,       # 0/1: 是否卸载到宏基站
]

# 3. 缓存决策 (离散动作,可选)
cache_action = file_id_to_cache  # 范围 [0, NUM_FILES-1]
```

#### 总动作维度:
```python
ACTION_DIM = 2 (轨迹) + 3 (卸载) + 1 (缓存) = 6
```

#### 实现方式:
```python
# 使用混合动作空间的Actor网络
class HybridActor(nn.Module):
    def __init__(self, obs_dim, continuous_dim=2, discrete_dims=[3, NUM_FILES]):
        super().__init__()
        self.shared = nn.Sequential(...)

        # 连续动作头 (轨迹)
        self.continuous_head = nn.Sequential(
            nn.Linear(hidden_dim, continuous_dim),
            nn.Tanh()  # 输出 [-1, 1]
        )

        # 离散动作头 (卸载决策)
        self.offloading_head = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )

        # 离散动作头 (缓存决策)
        self.cache_head = nn.Sequential(
            nn.Linear(hidden_dim, NUM_FILES),
            nn.Softmax(dim=-1)
        )
```

**优点**:
- 联合优化轨迹和任务卸载
- 端到端学习,不依赖启发式算法
- 更强的优化能力

**缺点**:
- 动作空间复杂,训练难度增加
- 需要混合动作空间的算法(如SAC-Discrete)

---

### 方案二: 纯连续动作空间 (简化版)

#### 动作组成:
```python
# 只控制轨迹,其他决策用启发式算法
trajectory_action = [delta_x, delta_y]  # 范围 [-1, 1]

# 卸载决策: 基于规则
# - 如果本地缓存有 → 本地处理
# - 如果邻居有 → 卸载到最近邻居
# - 否则 → 卸载到MBS

# 缓存决策: GDSF算法(保持不变)
```

**优点**:
- 简单,易于训练
- 可以直接使用现有的MADDPG/MATD3
- 快速验证超图卷积的效果

**缺点**:
- 优化能力受限
- 无法学习最优卸载策略

---

### 方案三: 分层动作空间 (高级)

#### 两层决策:
```python
# 高层决策 (每T步执行一次)
high_level_action = [
    target_region_x,  # 目标区域中心x
    target_region_y,  # 目标区域中心y
    service_priority  # 服务优先级 [0, 1]
]

# 低层决策 (每步执行)
low_level_action = [
    delta_x,          # 朝向目标的移动
    delta_y,
    offload_decision  # 卸载决策
]
```

**优点**:
- 符合实际决策层次
- 高层规划,低层执行
- 适合长时间范围优化

**缺点**:
- 实现复杂
- 需要分层强化学习算法(HRL)

---

## 🌟 推荐方案: 方案一(混合动作空间)

### 理由:
1. **研究价值高**: 联合优化是研究热点
2. **性能最优**: 端到端学习优于启发式
3. **可行性强**: 有成熟的混合动作空间算法

### 具体实现建议:

#### 1. 算法选择
```python
# 推荐: MATD3 + 混合动作空间
# 或: MASAC (SAC天然支持混合动作)

MODEL = "hybrid_matd3"  # 新增模型
```

#### 2. 动作空间定义
```python
# config.py
ACTION_DIM_CONTINUOUS = 2      # 轨迹 (delta_x, delta_y)
ACTION_DIM_OFFLOADING = 3      # 卸载决策 (local, neighbor, mbs)
ACTION_DIM_CACHE = NUM_FILES   # 缓存决策 (可选,初期可以不加)

# 简化版: 只加卸载决策
ACTION_DIM_TOTAL = 2 + 3 = 5
```

#### 3. 观测空间扩展
```python
# 当前观测
obs = [
    own_pos,           # 自身位置
    own_cache,         # 缓存状态
    neighbor_pos,      # 邻居位置
    ue_requests        # 用户请求
]

# 扩展观测 (加入超图信息)
obs_extended = [
    own_pos,
    own_cache,
    neighbor_pos,
    neighbor_cache,    # 新增: 邻居缓存状态
    ue_requests,
    hypergraph_adj,    # 新增: 超图邻接矩阵
    group_id           # 新增: 所属分组ID
]
```

---

## 🔄 与超图卷积的结合

### 超图构建:
```python
# 节点: 无人机
nodes = [UAV_0, UAV_1, ..., UAV_N]

# 超边: 根据不同关系构建
hyperedges = [
    # 超边1: 覆盖同一区域的无人机
    [UAV_0, UAV_2, UAV_3],

    # 超边2: 服务相同用户的无人机
    [UAV_1, UAV_4],

    # 超边3: 缓存相似内容的无人机
    [UAV_0, UAV_1, UAV_5]
]
```

### 超图卷积增强观测:
```python
# 原始观测
obs_raw = get_local_observation(uav_i)

# 超图卷积
obs_enhanced = HGCN(obs_raw, hypergraph_adj)

# 输入到Actor网络
action = Actor(obs_enhanced)
```

---

## 📊 对比实验设计

### 消融实验:
1. **Baseline**: Multi-UAV (Attention + 纯轨迹控制)
2. **Variant 1**: Multi-UAV + HGCN (超图卷积 + 纯轨迹控制)
3. **Variant 2**: Multi-UAV + 混合动作空间 (Attention + 轨迹+卸载)
4. **Full Model**: Multi-UAV + HGCN + 混合动作空间 (完整版)

### 评估指标:
- 平均延迟
- 能耗
- 公平性(JFI)
- 离线率
- 收敛速度

---

## 🚀 实施路线图

### 阶段1: 验证超图卷积 (1-2周)
```python
# 保持动作空间不变,只替换注意力机制
ACTION_DIM = 2  # 纯轨迹控制
Network = HGCN  # 替换Attention
```

### 阶段2: 扩展动作空间 (2-3周)
```python
# 添加卸载决策
ACTION_DIM = 2 + 3  # 轨迹 + 卸载
Network = HGCN
```

### 阶段3: 完整优化 (1-2周)
```python
# 添加缓存决策(可选)
ACTION_DIM = 2 + 3 + 1  # 轨迹 + 卸载 + 缓存
Network = HGCN + Dynamic Grouping
```

---

## 💡 关键技术点

### 1. 混合动作空间的训练技巧
```python
# 分别计算损失
loss_continuous = MSE(predicted_trajectory, target_trajectory)
loss_discrete = CrossEntropy(predicted_offloading, target_offloading)

# 加权组合
total_loss = loss_continuous + lambda * loss_discrete
```

### 2. 超图动态更新
```python
# 每隔一定步数重新聚类
if step % CLUSTERING_INTERVAL == 0:
    new_groups = DynamicSpectralClustering(state_history)
    hypergraph = build_hypergraph(new_groups)
```

### 3. 奖励函数调整
```python
# 原始奖励
reward = ALPHA_3*log(fairness) - ALPHA_1*log(latency)
         - ALPHA_2*log(energy) - ALPHA_4*log(offline_rate)

# 新增卸载相关奖励
reward += ALPHA_5 * offloading_success_rate  # 卸载成功率
reward -= ALPHA_6 * inter_uav_communication_cost  # 无人机间通信开销
```

---

## 📝 总结

**最佳方案**:
- **算法**: 多智能体强化学习 (MARL)
- **动作空间**: 混合动作空间 (轨迹 + 卸载决策)
- **网络结构**: HGCN + 动态分组
- **实施策略**: 分阶段,先验证超图卷积,再扩展动作空间

这样的设计既有研究创新性,又具备工程可行性!
