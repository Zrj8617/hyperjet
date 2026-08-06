# NumPy 除了提供数组，也负责按固定种子生成场景中的随机参数。
import numpy as np

# ===================== 训练参数设置 =====================
# 选择使用的多智能体强化学习模型
MODEL: str = "mappo"  # 可选模型: 'maddpg', 'matd3', 'mappo', 'masac', 'attention_<model>', 'random'
# 随机种子，保证实验可复现
SEED: int = 42
# 设置numpy库的随机种子
np.random.seed(SEED)

# ===================== zrj_3 clean 主线：环境规模 =====================
# 地图单位是米；一个回合包含 EPISODE_LENGTH 个时隙，每个时隙持续 TIME_SLOT_DURATION 秒。
AREA_WIDTH: int = 500
AREA_HEIGHT: int = 500
NUM_UAVS: int = 5
NUM_UES: int = 60
EPISODE_LENGTH: int = 500
TIME_SLOT_DURATION: float = 5.0
# 旧训练入口仍读取这个名字；clean 主线直接使用 EPISODE_LENGTH。
STEPS_PER_EPISODE: int = EPISODE_LENGTH

# ===================== zrj_3 clean 主线：热点与任务到达 =====================
# 热点内 UE 的 DAG 到达概率为“基础概率 × 热点倍率”。
HOTSPOT_RADIUS: float = 150.0
DAG_BASE_ARRIVAL_PROB: float = 0.0145
DAG_HOTSPOT_ARRIVAL_MULTIPLIER: float = 2.0

# ===================== zrj_3 clean 主线：UE 移动 =====================
# 等待 DAG 服务的 UE 会把正常步行速度乘以下面的降速比例。
UE_WALK_SPEED_MEAN: float = 1.2
UE_SERVICE_WAITING_SPEED_SCALE: float = 0.2

# ===================== zrj_3 clean 主线：DAG 结构 =====================
# 控制每个 DAG 的任务数、层数和单任务最大父节点数；BASE_UNIT_BYTES 用于估算运算量。
DAG_MIN_TASKS: int = 5
DAG_MAX_TASKS: int = 8
DAG_MAX_LEVELS: int = 4
DAG_MAX_PARENTS: int = 3
BASE_UNIT_BYTES: int = 10 * 1024

# ===================== zrj_3 clean 主线：任务属性 =====================
# 每个任务会在这些范围内随机采样输入、输出、常数和复杂度类型。
INPUT_DATA_SIZE_MB_RANGE: tuple[float, float] = (0.75, 14.0)
OUTPUT_DATA_SIZE_MB_RANGE: tuple[float, float] = (0.6, 10.5)
TASK_CONSTANT_RANGE: tuple[int, int] = (6, 60)
TASK_COMPLEXITY_PROBS: dict[str, float] = {
    "n": 0.2,
    "nlogn": 0.7,
    "nlog2n": 0.1,
}

# ===================== zrj_3 clean 主线：链路带宽 =====================
# 创建 DAG 时按 BANDWIDTH_LEVEL_PROBS 选一档基础上下行带宽，单位为 Mbps。
BASE_UPLOAD_BANDWIDTH_MBPS: list[float] = [20.0, 50.0, 100.0]
BASE_DOWNLOAD_BANDWIDTH_MBPS: list[float] = [50.0, 100.0, 200.0]
BANDWIDTH_LEVEL_PROBS: list[float] = [0.3, 0.5, 0.2]

# ===================== zrj_3 clean 主线：算力、能耗与队列 =====================
# 功率单位为瓦特，计算速率单位为每秒运算次数；这些值直接影响完成时间和能耗奖励。
P_UAV_COMPUTE: float = 50.0
P_UE_TX: float = 0.5
P_UAV_TX: float = 0.5
CLEAN_POWER_MOVE: float = 100.0
UAV_COMPUTE_RATE_OPS_PER_SEC: float = 1_000_000.0
CLEAN_MAX_QUEUE_PER_UAV: int = 16

# ===================== zrj_3 clean 主线：UAV 移动 =====================
# 每个时隙从悬停和四个平面方向中选一个动作，实际距离=速度×时隙秒数。
CLEAN_MOVEMENT_ACTIONS: tuple[str, ...] = ("hover", "+x", "-x", "+y", "-y")
CLEAN_MOVEMENT_ACTION_DIM: int = len(CLEAN_MOVEMENT_ACTIONS)
CLEAN_MOVEMENT_HOVER_ACTION: str = "hover"
CLEAN_UAV_MOVEMENT_SPEED: float = 15.0

# ===================== zrj_3 clean 主线：特征归一化 =====================
# 所有特征构建器都从这里读取统一参考尺度，不能在函数里临时写一个更大的分母，
# 否则候选特征会被压得接近 0，卸载策略就看不出候选间的差别。
CLEAN_NORM_PAIR_TIME_REF: float = 4.0 * TIME_SLOT_DURATION  # link/compute time scale (s)
CLEAN_NORM_AVAIL_TIME_REF: float = 8.0 * TIME_SLOT_DURATION  # queue-wait / availability scale (s)
CLEAN_NORM_PAIR_ENERGY_REF: float = P_UAV_COMPUTE * TIME_SLOT_DURATION  # per-task energy scale (J)
CLEAN_NORM_QUEUE_WORKLOAD_REF: float = (
    UAV_COMPUTE_RATE_OPS_PER_SEC * TIME_SLOT_DURATION * CLEAN_MAX_QUEUE_PER_UAV
)  # per-UAV queued ops scale
CLEAN_NORM_ACTIVE_TASK_REF: float = float(NUM_UES * DAG_MAX_TASKS)  # max possible active tasks

# ===================== zrj_3 clean 主线：超图 =====================
# CleanGraphBuilder 只读取下面四个 ENABLE_* 开关来控制四类超边：
#   E_DAG   <- ENABLE_DAG_DEPENDENCY_EDGES
#   E_khop  <- ENABLE_KHOP_DEPENDENCY_HYPEREDGES
#   E_attr  <- ENABLE_ATTRIBUTE_HYPEREDGES
#   E_part  <- ENABLE_KAHYPAR_PARTITION_HYPEREDGES
# 做 clean 消融实验时只改这些开关。文件后面的旧 `USE_*` 超边开关属于旧管线，
# 改它们不会影响 clean 主线。
ENABLE_DAG_DEPENDENCY_EDGES: bool = True
ENABLE_KHOP_DEPENDENCY_HYPEREDGES: bool = True
ENABLE_ATTRIBUTE_HYPEREDGES: bool = True
ATTRIBUTE_HYPEREDGE_UPDATE_INTERVAL: int = 5
ATTRIBUTE_HYPEREDGE_CLUSTER_NUM: int = 4
KHOP_K: int = 2
ENABLE_KAHYPAR_PARTITION_HYPEREDGES: bool = False  # 收敛调优阶段暂不启用（2026-08-06 决策）；Linux 服务器装 kahypar==1.3.7 后开启
KAHYPAR_PARTITION_UPDATE_INTERVAL: int = 5
KAHYPAR_DEGRADED_EXPERIMENT_LABEL: str = "no-KaHyPar / degraded"
KAHYPAR_INI_RELATIVE_PATH: str = "third_party/kahypar/cut_rKaHyPar_sea20.ini"
KAHYPAR_SEED: int = 0
KAHYPAR_EPSILON: float = 0.03
KAHYPAR_WORKER_TIMEOUT_SECONDS: float = 10.0
KAHYPAR_WORKER_TERMINATE_GRACE_SECONDS: float = 1.0
KAHYPAR_WORKER_KILL_GRACE_SECONDS: float = 1.0
KAHYPAR_MAX_CONSECUTIVE_FAILURES: int = 3

# ===================== zrj_3 clean 主线：奖励 =====================
# 时间和能耗权重越大，惩罚越重；完成 DAG 权重越大，完成奖励越高。
# 调优v8: 系统性收敛方案 - 强化任务信号 + 适度移动惩罚
REWARD_TIME_WEIGHT: float = 0.5
REWARD_ENERGY_WEIGHT: float = 0.05
REWARD_MOVEMENT_ENERGY_WEIGHT: float = 0.08  # 中高惩罚，抑制无效横跳但允许必要移动
REWARD_COMPLETED_DAG_WEIGHT: float = 8.0   # 大幅提高完成奖励，让任务信号主导
CRITICAL_TASK_WEIGHT: float = 1.0
NONCRITICAL_TASK_WEIGHT: float = 0.5
# 时间参考值和截断上限只用于奖励归一化，避免长队列产生的极端时延压过 DAG 完成奖励。
# 指标日志仍保留原始时延和流时间，不会被截断。
CLEAN_REWARD_TIME_REF: float = 60.0
CLEAN_REWARD_TIME_CLIP: float = 10.0
CLEAN_REWARD_TASK_ENERGY_REF: float = P_UAV_COMPUTE * TIME_SLOT_DURATION
CLEAN_REWARD_MOVE_ENERGY_REF: float = NUM_UAVS * CLEAN_POWER_MOVE * TIME_SLOT_DURATION

# --- 可选的位置塑形奖励（已开启，鼓励无人机移动覆盖用户） ---------------------
# 打开后，UAV 覆盖更多就绪任务源位置会得到额外奖励。
# 调优v8: 位置塑形奖励保持，但权重适中，避免诱导无效移动
ENABLE_MOVEMENT_POSITION_SHAPING: bool = True
REWARD_MOVEMENT_POSITION_WEIGHT: float = 0.5  # 适中奖励，覆盖即得分但不过度诱导

# 日志记录频率（单位：回合）
LOG_FREQ: int = 1
# 图像保存频率（单位：步长）
IMG_FREQ: int = 1000
# 测试阶段日志记录频率（单位：回合）
TEST_LOG_FREQ: int = 1
# 测试阶段图像保存频率（单位：步长）
TEST_IMG_FREQ: int = 100

# ===================== 仿真环境参数 =====================
# 宏基站坐标位置（X, Y, Z），单位：米
MBS_POS: np.ndarray = np.array([350.0, 350.0, 30.0], dtype=np.float32)
# 用户设备每时间片最大移动距离，单位：米
UE_MAX_DIST: float = 3.0
# 用户高斯-马尔可夫移动模型记忆因子
UE_GM_ALPHA: float = 0.85
# 用户高斯-马尔可夫移动模型平均速度，单位：米/秒
UE_GM_MEAN_SPEED: float = UE_WALK_SPEED_MEAN
# 用户高斯-马尔可夫移动模型速度扰动标准差，单位：米/秒
UE_GM_SPEED_SIGMA: float = 0.3
# 用户高斯-马尔可夫移动模型方向扰动标准差，单位：弧度
UE_GM_THETA_SIGMA: float = 0.30
# 用户高斯-马尔可夫移动模型最小速度，单位：米/秒
UE_GM_MIN_SPEED: float = 0.0
# 用户高斯-马尔可夫移动模型最大速度，单位：米/秒
UE_GM_MAX_SPEED: float = UE_MAX_DIST / TIME_SLOT_DURATION
# 用户高斯-马尔可夫移动模型边界处理方式
UE_BOUNDARY_MODE: str = "reflect"

# ===================== 高密度热点/任务聚集场景参数 =====================
# 是否启用UE热点初始化与热点任务到达增强
ENABLE_UE_HOTSPOTS: bool = False  # Deprecated: clean mainline uses one map hotspot region, not hotspot UE identity.
# 热点数量
NUM_HOTSPOTS: int = 0  # Deprecated: clean mainline samples one episode hotspot outside UE class.
# 初始分布在热点内的UE比例
HOTSPOT_UE_RATIO: float = 0.0  # Deprecated: fixed-ratio hotspot UE model is forbidden in clean mainline.
# UE围绕热点中心采样的标准差，单位：米
HOTSPOT_STD: float = 60.0  # Deprecated.
# 热点UE的DAG到达概率倍率
HOTSPOT_DAG_ARRIVAL_MULTIPLIER: float = DAG_HOTSPOT_ARRIVAL_MULTIPLIER  # Deprecated alias.

# ===================== 阶段一 DAG 任务参数 =====================
# 是否启用阶段一动态DAG底层模块
ENABLE_DYNAMIC_DAG: bool = True
# 是否启用阶段一任务执行主路径
ENABLE_PHASE_ONE_EXECUTION: bool = True
# 是否保留旧版基于MBS的请求处理路径
ENABLE_LEGACY_REQUEST_PIPELINE: bool = False
# 每个UE每个时隙生成一个新DAG的概率
DAG_ARRIVAL_PROB: float = DAG_BASE_ARRIVAL_PROB  # Deprecated alias.
# 单个DAG任务节点数量范围；当前主线收口到中等规模DAG，降低噪声但保留依赖结构
# DAG最大拓扑层数
DAG_MAX_TASK_LEVELS: int = DAG_MAX_LEVELS  # Deprecated alias.
# DAG_MAX_PARENTS is defined in the clean DAG group above and kept under the
# same name for old imports.
# 任务输入/输出大小范围
DAG_MIN_INPUT_SIZE: int = 1 * 10**5
DAG_MAX_INPUT_SIZE: int = 2 * 10**6
DAG_MIN_OUTPUT_SIZE: int = 5 * 10**4
DAG_MAX_OUTPUT_SIZE: int = 8 * 10**5
# 任务计算量范围（CPU cycles）
DAG_MIN_CPU_CYCLES: int = 2 * 10**8
DAG_MAX_CPU_CYCLES: int = 6 * 10**9
# 任务截止期相对当前时隙的偏移范围
DAG_MIN_DEADLINE_OFFSET: int = 5
DAG_MAX_DEADLINE_OFFSET: int = 40
# DAG任务类型。Deprecated: clean mainline no longer uses preprocess/compute/aggregation semantics.
TASK_TYPE_PREPROCESS: int = 0
TASK_TYPE_COMPUTE: int = 1
TASK_TYPE_AGGREGATION: int = 2
NUM_DAG_TASK_TYPES: int = 3
# preprocess/sensing任务属性范围
DAG_TYPE0_INPUT_RANGE: tuple[int, int] = (1_000_000, 3_000_000)
DAG_TYPE0_OUTPUT_RANGE: tuple[int, int] = (300_000, 1_000_000)
DAG_TYPE0_CPU_RANGE: tuple[int, int] = (500_000_000, 2_000_000_000)
DAG_TYPE0_DEADLINE_RANGE: tuple[int, int] = (12, 40)
# compute-heavy inference任务属性范围
DAG_TYPE1_INPUT_RANGE: tuple[int, int] = (300_000, 1_500_000)
DAG_TYPE1_OUTPUT_RANGE: tuple[int, int] = (50_000, 400_000)
DAG_TYPE1_CPU_RANGE: tuple[int, int] = (2_000_000_000, 8_000_000_000)
DAG_TYPE1_DEADLINE_RANGE: tuple[int, int] = (10, 35)
# aggregation/output-heavy任务属性范围
DAG_TYPE2_INPUT_RANGE: tuple[int, int] = (200_000, 1_200_000)
DAG_TYPE2_OUTPUT_RANGE: tuple[int, int] = (500_000, 1_500_000)
DAG_TYPE2_CPU_RANGE: tuple[int, int] = (800_000_000, 3_000_000_000)
DAG_TYPE2_DEADLINE_RANGE: tuple[int, int] = (8, 30)
# 图构建阶段保留的ready任务数量
DAG_MAX_READY_TASKS_FOR_GRAPH: int = 12
# 协同超边局部构造：任务数 top-M
DAG_COLLAB_TOP_M_TASKS: int = 5
# 协同超边局部构造：候选UAV数 top-K
DAG_COLLAB_TOP_K_UAVS: int = 3
# 资源竞争超边：每个任务只把前K个候选UAV视为真实竞争对象
DAG_RESOURCE_COMPETITION_TOP_K_UAVS: int = 2
# task-UAV可执行边的最远距离阈值
DAG_TASK_UAV_MAX_DISTANCE: float = 220.0
# 单个UAV在阶段一中允许保留的最大排队任务数
DAG_MAX_QUEUE_PER_UAV: int = 8
# critical hyperedge使用的slack阈值
DAG_CRITICAL_SLACK_THRESHOLD: int = 8
# critical hyperedge单条最多任务数，避免全局urgent大超边过平滑
DAG_CRITICAL_MAX_SIZE: int = 8
# 任务属性超边：每簇最多任务数
DAG_ATTRIBUTE_TOP_M_TASKS: int = 5
# 任务属性超边：按属性邻域构造时保留的簇数
DAG_ATTRIBUTE_MAX_GROUPS: int = 4
# deadline允许的粗判容忍量
DAG_MAX_DEADLINE_TOLERANCE: int = 2
# candidate policy：strict保持原过滤逻辑；expanded放宽部分软约束用于候选诊断
CANDIDATE_POLICY_MODE: str = "strict"
ENABLE_CANDIDATE_REJECTION_LOGGING: bool = False
EXPANDED_CANDIDATE_MAX_DISTANCE: float = 500.0
EXPANDED_CANDIDATE_MAX_QUEUE: int = 16
EXPANDED_CANDIDATE_DEADLINE_TOLERANCE: int = 20
EXPANDED_CANDIDATE_KEEP_DEADLINE_VIOLATORS: bool = True
EXPANDED_CANDIDATE_KEEP_QUEUE_OVER_LIMIT: bool = True
EXPANDED_CANDIDATE_KEEP_DISTANCE_OVER_STRICT_LIMIT: bool = True
# 阶段一任务节点特征维度
DAG_TASK_FEATURE_DIM: int = 11

# ===================== 阶段一奖励参数 =====================
PHASE_ONE_FINISH_REWARD: float = 2.0
PHASE_ONE_DEADLINE_PENALTY: float = 2.5
PHASE_ONE_ENERGY_PENALTY: float = 0.002
PHASE_ONE_INVALID_PENALTY: float = 0.5
# 阶段一RL轻量DAG job级奖励塑形：保留子任务级reward，同时追加完整DAG完成/失败增量项
USE_PHASE_ONE_DAG_REWARD_SHAPING: bool = True
PHASE_ONE_DAG_SUCCESS_REWARD: float = 6.0
PHASE_ONE_DAG_ON_TIME_BONUS: float = 4.0
PHASE_ONE_DAG_FAILURE_PENALTY: float = 6.0
# Stage B专用MAPPO移动奖励：只给移动 actor 短因果链反馈，DAG指标继续logging
USE_STAGE_B_MOVEMENT_REWARD: bool = False
STAGE_B_READY_COVERAGE_REWARD: float = 0.3
STAGE_B_ASSIGNED_COVERAGE_REWARD: float = 1.2
STAGE_B_READY_COVERAGE_DELTA_REWARD: float = 0.0
STAGE_B_ASSIGNED_COVERAGE_DELTA_REWARD: float = 0.0
STAGE_B_ASSIGNED_DISTANCE_REWARD: float = 0.0
STAGE_B_FEASIBLE_EDGE_REWARD: float = 0.1
STAGE_B_PROGRESS_REWARD: float = 0.3
STAGE_B_LOCAL_FINISH_REWARD: float = 0.0
STAGE_B_LOCAL_ON_TIME_FINISH_REWARD: float = 0.0
STAGE_B_MOVE_ENERGY_PENALTY: float = 0.2
STAGE_B_COLLISION_PENALTY: float = 1.5
STAGE_B_BOUNDARY_PENALTY: float = 1.5

# ===================== 阶段一后半段：HGNN/Score 参数 =====================
# 是否启用图编码后的 score assignment
USE_HGNN_SCORE_ASSIGNMENT: bool = False
# 预训练 graph scheduler checkpoint 路径；为空表示不加载
HGNN_SCORE_CHECKPOINT: str = ""
# 若 score assignment 不可用或无分数时，是否回退到启发式调度
SCORE_FALLBACK_TO_HEURISTIC: bool = True
# 是否只让HGNN score接管高风险ready task；普通ready task继续走fallback
USE_SELECTIVE_HGNN_SCORING: bool = False
# 诊断开关：是否在每个ready task分配前重新构图并重新HGNN打分，避免同一step内顺序分配使用过期score
USE_HGNN_PER_TASK_RESCORING: bool = False
# 运行时保护：score只能在不明显慢于heuristic的候选中重排序
USE_SCORE_RUNTIME_BOUNDED_GUARD: bool = False
SCORE_RUNTIME_FINISH_TOLERANCE: float = 0.1
# 诊断开关：只允许score与heuristic一致的决策生效，分歧全部回退heuristic
USE_SCORE_AGREEMENT_ONLY: bool = False
# selective scoring高风险判据：DAG/task剩余slack阈值
SELECTIVE_HGNN_SLACK_THRESHOLD: int = DAG_CRITICAL_SLACK_THRESHOLD
# selective scoring上下文风险窗口：critical/successor信号只在DAG slack进入该窗口时触发
SELECTIVE_HGNN_CONTEXT_SLACK_MULTIPLIER: float = 2.0
# selective scoring高风险判据：候选UAV数量稀缺阈值
SELECTIVE_HGNN_CANDIDATE_THRESHOLD: int = 2
# selective scoring高风险判据：所属DAG已接近闭环的完成比例阈值
SELECTIVE_HGNN_COMPLETION_THRESHOLD: float = 0.6
# selective scoring高风险判据开关
USE_STRICT_SELECTIVE_HGNN_SCORING: bool = False
SELECTIVE_HGNN_USE_CRITICAL_PATH: bool = True
SELECTIVE_HGNN_USE_CANDIDATE_SCARCITY: bool = True
SELECTIVE_HGNN_USE_SUCCESSOR_UNLOCK: bool = True
SELECTIVE_HGNN_USE_DAG_COMPLETION: bool = True
# 消融实验开关：是否启用阶段一超边编码
USE_PHASE_ONE_HYPEREDGES: bool = True
# 消融实验开关：是否启用旧口径的协同/资源联合超边组
USE_COLLABORATIVE_HYPEREDGES: bool = True
# 消融实验开关：是否启用局部服务域超边
USE_SERVICE_DOMAIN_HYPEREDGES: bool = True
# 消融实验开关：是否启用共享候选资源竞争超边
USE_RESOURCE_COMPETITION_HYPEREDGES: bool = True
# 消融实验开关：是否启用关键阶段超边
USE_CRITICAL_HYPEREDGES: bool = True
# 消融实验开关：是否启用关键任务 support mixed hyperedge
USE_CRITICAL_SUPPORT_HYPEREDGES: bool = True
# 消融实验开关：是否启用任务属性超边
USE_ATTRIBUTE_HYPEREDGES: bool = False
# 消融实验开关：是否启用计算密集属性风险超边
USE_COMPUTE_ATTRIBUTE_HYPEREDGES: bool = False
# 消融实验开关：是否启用通信密集属性风险超边
USE_COMMUNICATION_ATTRIBUTE_HYPEREDGES: bool = False
# 消融实验开关：是否启用候选资源稀缺属性风险超边
USE_CANDIDATE_SCARCE_ATTRIBUTE_HYPEREDGES: bool = False
# 消融实验开关：task_type属性超边仅作为后续可选接口，默认关闭
USE_TASK_TYPE_ATTRIBUTE_HYPEREDGES: bool = False
# 消融实验开关：是否启用 task-UAV pair feature；关闭时保留维度但置零
USE_TASK_UAV_PAIR_FEATURES: bool = True
# task-UAV pair feature模式：full保留全部9维；limited置零答案型EFT字段；none整段置零
TASK_UAV_PAIR_FEATURE_MODE: str = "full"
# task-UAV基础pair feature维度：上传/迁移/计算/排队/deadline/父节点迁移/队列/距离
BASE_TASK_UAV_PAIR_FEATURE_DIM: int = 9
# 是否把超边参与摘要直接拼入score head的pair feature；关闭时保留维度但置零
USE_PAIR_HYPEREDGE_SCORE_FEATURES: bool = True
# pair-level超边摘要维度：
# service_pair, resource_pair, critical_pair, service_uav_ratio,
# resource_uav_ratio, critical_uav_ratio, service_task_ratio, resource_task_ratio
PAIR_HYPEREDGE_SCORE_FEATURE_DIM: int = 8
# HGNN 隐层维度
HGNN_HIDDEN_DIM: int = 64
# HGNN 层数
HGNN_NUM_LAYERS: int = 2
# 任务与 UAV embedding 维度
TASK_EMB_DIM: int = 64
UAV_EMB_DIM: int = 64
# score 预训练参数
SCORE_PRETRAIN_MODE: str = "top1"  # 可选: "top1", "ranking", "soft", "bounded_ranking"
SCORE_PRETRAIN_LR: float = 1e-3
SCORE_PRETRAIN_EPOCHS: int = 5
SCORE_PRETRAIN_EPISODES: int = 3
SCORE_PRETRAIN_STEPS_PER_EPISODE: int = 60
SCORE_PRETRAIN_ACTION_MODE: str = "zero"  # 可选: "zero", "random"
SCORE_RANKING_MARGIN: float = 0.05
SCORE_RANKING_TOP1_WEIGHT: float = 0.2
SCORE_SOFT_TARGET_TAU: float = 0.2
SCORE_BOUNDED_RANKING_FINISH_TOLERANCE: float = 0.1
# ===================== Assignment-only RL 参数 =====================
# 最小化任务卸载RL分支：默认关闭，不影响原supervised score-head主路径
USE_RL_ASSIGNMENT: bool = False
RL_ASSIGNMENT_USE_HGNN_ENCODER: bool = True
RL_ASSIGNMENT_TRAIN_ENCODER: bool = False
RL_ASSIGNMENT_LOAD_ENCODER_CHECKPOINT: bool = True
RL_ASSIGNMENT_GAMMA: float = 0.99
RL_ASSIGNMENT_CLIP_EPS: float = 0.2
RL_ASSIGNMENT_ENTROPY_COEF: float = 0.01
RL_ASSIGNMENT_VALUE_COEF: float = 0.5
RL_ASSIGNMENT_LR: float = 3e-4
RL_ASSIGNMENT_MAX_GRAD_NORM: float = 0.5
RL_ASSIGNMENT_UPDATE_EPOCHS: int = 4
RL_ASSIGNMENT_MINIBATCH_SIZE: int = 64

# ===================== 无人机参数 =====================
# 无人机飞行高度，单位：米
UAV_ALTITUDE: int = 100
# 无人机飞行速度，单位：米/秒
UAV_SPEED: float = 15.0
# 无人机存储容量，单位：字节
UAV_STORAGE_CAPACITY: np.ndarray = np.random.choice(np.arange(40 * 10**6, 80 * 10**6, 10**6), size=NUM_UAVS).astype(np.int64)
# 无人机计算能力，单位：周期/秒
UAV_COMPUTING_CAPACITY: np.ndarray = np.random.choice(np.arange(5 * 10**9, 20 * 10**9, 10**9), size=NUM_UAVS).astype(np.int64)
# 无人机感知范围半径，单位：米
UAV_SENSING_RANGE: float = 300.0
# A2A单跳通信最大可用距离，单位：米
A2A_MAX_RANGE: float = UAV_SENSING_RANGE
# A2A链路建立与调度控制开销，单位：秒
A2A_CTRL_OVERHEAD: float = 0.05
# 无人机服务覆盖半径，单位：米
UAV_COVERAGE_RADIUS: float = 100.0
# 无人机间最小安全距离，单位：米
MIN_UAV_SEPARATION: float = 200.0
# 断言：确保无人机存储容量大于0
assert np.all(UAV_STORAGE_CAPACITY > 0)
# 断言：确保无人机计算能力大于0
assert np.all(UAV_COMPUTING_CAPACITY > 0)
# 断言：确保覆盖半径不会导致无人机重叠
assert UAV_COVERAGE_RADIUS * 2 <= MIN_UAV_SEPARATION
# 断言：确保感知范围大于最小安全距离
assert UAV_SENSING_RANGE >= MIN_UAV_SEPARATION

# ===================== 碰撞与惩罚机制 =====================
# 解决无人机碰撞的迭代次数
COLLISION_AVOIDANCE_ITERATIONS: int = 20
# 发生碰撞的惩罚值
COLLISION_PENALTY: float = 10.0
# 超出边界的惩罚值
BOUNDARY_PENALTY: float = 10.0
# 请求未被服务的时延惩罚值
NON_SERVED_LATENCY_PENALTY: float = 20.0
# 单个无人机最大邻居数量
MAX_UAV_NEIGHBORS: int = NUM_UAVS - 1
# 单个无人机最大关联服务用户数
MAX_ASSOCIATED_UES: int = min(30, NUM_UES // NUM_UAVS + 10)
# 断言：邻居数量范围合法
assert MAX_UAV_NEIGHBORS >= 1 and MAX_UAV_NEIGHBORS <= NUM_UAVS - 1
# 断言：关联用户数量范围合法
assert MAX_ASSOCIATED_UES >= 1 and MAX_ASSOCIATED_UES <= NUM_UES

# 无人机移动功耗，单位：瓦特
POWER_MOVE: float = 100.0
# 无人机悬停功耗，单位：瓦特
POWER_HOVER: float = 80.0
# 无人机初始/最大电量，单位：焦耳
UAV_ENERGY_CAPACITY: float = 1_000_000.0

# ===================== 请求相关参数 =====================
# 服务类型数量
NUM_SERVICES: int = 25
# 内容类型数量
NUM_CONTENTS: int = 50
# 文件总数（服务+内容）
NUM_FILES: int = NUM_SERVICES + NUM_CONTENTS
# 每字节数据所需CPU计算周期
CPU_CYCLES_PER_BYTE: np.ndarray = np.random.randint(2000, 4000, size=NUM_SERVICES)
# 文件大小，单位：字节
FILE_SIZES: np.ndarray = np.random.randint(10**6, 5 * 10**6, size=NUM_FILES).astype(np.int64)
# 请求最小输入数据大小，单位：字节
MIN_INPUT_SIZE: int = 1 * 10**6
# 请求最大输入数据大小，单位：字节
MAX_INPUT_SIZE: int = 5 * 10**6
# Zipf分布参数（请求热度分布）
ZIPF_BETA: float = 0.8
# CPU能耗计算系数
K_CPU: float = 1e-27

# ===================== 缓存策略参数 =====================
# 缓存更新间隔，单位：时间片
T_CACHE_UPDATE_INTERVAL: int = 50
# GDSF缓存平滑因子
GDSF_SMOOTHING_FACTOR: float = 0.75

# ===================== 概率缓存参数 =====================
# 文件平均大小
AVG_FILE_SIZE: float = float(np.mean(FILE_SIZES))
# 概率缓存gamma参数
PROB_GAMMA: float = 0.5

# ===================== 通信参数 =====================
# 通信信道增益常数乘积
G_CONSTS_PRODUCT: float = 2.2846 * 1.42 * 1e-4
# 数据传输功率，单位：瓦特
TRANSMIT_POWER: float = 0.5
# 加性高斯白噪声功率
AWGN: float = 1e-13
# 无人机间通信带宽，单位：赫兹
BANDWIDTH_INTER: int = 20 * 10**6
# 边缘通信带宽，单位：赫兹
BANDWIDTH_EDGE: int = 40 * 10**6
# 回传链路带宽，单位：赫兹
BANDWIDTH_BACKHAUL: int = 10 * 10**6

# ===================== 无线能量传输(WPT)参数 =====================
# 用户设备电池最大容量，单位：焦耳
UE_BATTERY_CAPACITY: float = 100.0
# 用户设备电池低电量阈值
UE_CRITICAL_THRESHOLD: float = 0.3 * UE_BATTERY_CAPACITY
# 无线能量传输发射功率
WPT_TRANSMIT_POWER: float = 500.0 * 1e6
# 能量收集效率（60%）
WPT_EFFICIENCY: float = 0.6
# 用户设备静态功耗，单位：瓦特
UE_STATIC_POWER: float = 0.05

# ===================== 奖励函数权重参数 =====================
# 奖励公式：时延权重（负向，惩罚时延）
ALPHA_1 = 1.0
# 奖励公式：能耗权重（负向，惩罚能耗）
ALPHA_2 = 0.4
# 奖励公式：公平性权重（正向，鼓励公平）
ALPHA_3 = 2.0
# 奖励公式：离线率权重（负向，惩罚断电）
ALPHA_4 = 50.0
# 奖励缩放系数，防止数值过大
REWARD_SCALING_FACTOR: float = 0.01

# ===================== 神经网络输入输出维度 =====================
# 旧请求/缓存管线观测维度：自身位置(2)+缓存状态(NUM_FILES)
LEGACY_SELF_OBS_DIM: int = 2 + NUM_FILES
# 旧请求/缓存管线UE观测维度：相对位置(2)+请求类型/大小/ID(3)+电量(1)
LEGACY_UE_OBS_DIM: int = 2 + 3 + 1
# 旧请求/缓存管线邻居观测维度：相对位置(2)
LEGACY_NEIGHBOR_OBS_DIM: int = 2

# 阶段一专用观测开关：去掉旧own_cache 75维，只保留UAV移动控制需要的局部摘要
USE_PHASE_ONE_DEDICATED_OBS: bool = True
# MAPPO紧凑观测开关：阶段一RL接入时只给UAV移动控制所需的局部统计摘要
USE_MAPPO_COMPACT_OBS: bool = True
# 阶段一自身观测维度：位置(2)+队列/忙闲/电量/邻居数/算力(5)
PHASE_ONE_SELF_OBS_DIM: int = 7
# 阶段一邻居观测维度：相对位置(2)+邻居队列长度/忙闲(2)
PHASE_ONE_NEIGHBOR_OBS_DIM: int = 4
# 阶段一任务摘要维度：相对位置(2)+slack/input/level/state(4)
PHASE_ONE_TASK_OBS_DIM: int = 6
# MAPPO紧凑局部任务/压力摘要维度
MAPPO_COMPACT_LOCAL_OBS_DIM: int = 22

_USE_PHASE_ONE_OBS_DIM: bool = (
    ENABLE_DYNAMIC_DAG
    and ENABLE_PHASE_ONE_EXECUTION
    and not ENABLE_LEGACY_REQUEST_PIPELINE
    and USE_PHASE_ONE_DEDICATED_OBS
)
_USE_MAPPO_COMPACT_OBS_DIM: bool = _USE_PHASE_ONE_OBS_DIM and USE_MAPPO_COMPACT_OBS
# 以下三个名称保持给现有模型/attention模块使用，默认随当前实验管线选择活动obs维度
SELF_OBS_DIM: int = PHASE_ONE_SELF_OBS_DIM if _USE_PHASE_ONE_OBS_DIM else LEGACY_SELF_OBS_DIM
UE_OBS_DIM: int = (
    MAPPO_COMPACT_LOCAL_OBS_DIM
    if _USE_MAPPO_COMPACT_OBS_DIM
    else PHASE_ONE_TASK_OBS_DIM
    if _USE_PHASE_ONE_OBS_DIM
    else LEGACY_UE_OBS_DIM
)
NEIGHBOR_OBS_DIM: int = PHASE_ONE_NEIGHBOR_OBS_DIM if _USE_PHASE_ONE_OBS_DIM else LEGACY_NEIGHBOR_OBS_DIM
OBS_DIM_SINGLE: int = (
    SELF_OBS_DIM + (MAX_UAV_NEIGHBORS * NEIGHBOR_OBS_DIM) + UE_OBS_DIM
    if _USE_MAPPO_COMPACT_OBS_DIM
    else SELF_OBS_DIM + (MAX_UAV_NEIGHBORS * NEIGHBOR_OBS_DIM) + (MAX_ASSOCIATED_UES * UE_OBS_DIM)
)
# 动作维度：移动角度+移动距离
ACTION_DIM: int = 2
# 神经网络隐藏层维度
MLP_HIDDEN_DIM: int = 128

# ===================== 优化器参数 =====================
# Actor网络学习率
ACTOR_LR: float = 9e-4
# Critic网络学习率
CRITIC_LR: float = 8e-4
# 折扣因子γ
DISCOUNT_FACTOR: float = 0.96
# 目标网络更新因子τ
UPDATE_FACTOR: float = 0.012
# 梯度裁剪最大范数（防止梯度爆炸）
MAX_GRAD_NORM: float = 0.5
# 策略高斯分布最大对数标准差
LOG_STD_MAX: float = 2
# 策略高斯分布最小对数标准差
LOG_STD_MIN: float = -20
# 极小值，防止除0错误
EPSILON: float = 1e-9

# ===================== 离线策略算法参数 =====================
# 经验回放池大小
REPLAY_BUFFER_SIZE: int = 10**6
# 训练批次大小
REPLAY_BATCH_SIZE: int = 128
# 初始随机探索步数
INITIAL_RANDOM_STEPS: int = 5000
# 网络更新频率
LEARN_FREQ: int = 10

# ===================== 高斯探索噪声参数 =====================
# 初始噪声尺度
INITIAL_NOISE_SCALE: float = 0.2
# 最小噪声尺度
MIN_NOISE_SCALE: float = 0.01
# 噪声衰减率
NOISE_DECAY_RATE: float = 0.995

# ===================== MATD3算法专用参数 =====================
# 策略延迟更新频率
POLICY_UPDATE_FREQ: int = 2
# 目标策略平滑噪声标准差
TARGET_POLICY_NOISE: float = 0.25
# 噪声裁剪范围
NOISE_CLIP: float = 0.5

# ===================== MAPPO算法专用参数 =====================
# 采样轨迹长度
PPO_ROLLOUT_LENGTH: int = STEPS_PER_EPISODE
# GAE优势函数参数λ
PPO_GAE_LAMBDA: float = 0.95
# 每次更新迭代轮数
PPO_EPOCHS: int = 10
# PPO训练批次大小
PPO_BATCH_SIZE: int = 200
# PPO裁剪阈值ε
PPO_CLIP_EPS: float = 0.2
# 熵正则项系数
PPO_ENTROPY_COEF: float = 0.01

# ===================== MASAC算法专用参数 =====================
# 自适应温度系数α学习率
ALPHA_LR: float = 3e-4

# ===================== 注意力机制参数 =====================
# 注意力隐藏层维度
ATTN_HIDDEN_DIM: int = 64
# 注意力头数
ATTN_NUM_HEADS: int = 8
# 断言：确保隐藏层维度能被头数整除
assert ATTN_HIDDEN_DIM % ATTN_NUM_HEADS == 0, f"ATTN_HIDDEN_DIM ({ATTN_HIDDEN_DIM}) must be divisible by ATTN_NUM_HEADS ({ATTN_NUM_HEADS})"
