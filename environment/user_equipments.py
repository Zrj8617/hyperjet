import config
import numpy as np


class UE:
    def __init__(self, ue_id: int) -> None:
        """创建一个 UE，并随机生成它的初始位置、速度和移动方向。"""
        self.id: int = ue_id
        self.pos: np.ndarray = self._sample_uniform_position()
        self.speed: float = self._sample_walk_speed()
        self.theta: float = float(np.random.uniform(0.0, 2.0 * np.pi))
        self.velocity: np.ndarray = self._velocity_from_polar()
        self.service_waiting: bool = False
        self.active_dag_id: str | None = None

    def reset_episode_state(self) -> None:
        """开始新回合时重新采样移动状态，并清除尚未完成的 DAG 标记。"""
        self.pos = self._sample_uniform_position()
        self.speed = self._sample_walk_speed()
        self.theta = float(np.random.uniform(0.0, 2.0 * np.pi))
        self.velocity = self._velocity_from_polar()
        self.service_waiting = False
        self.active_dag_id = None

    def update_position(self, *, commit_position: bool = True) -> None:
        """按高斯—马尔可夫模型推进 UE 一个时隙的位置。

        正在等待服务的 UE 会主动降速；移动越过地图边界时采用镜像反弹。
        """
        alpha = float(config.UE_GM_ALPHA)
        noise_scale = np.sqrt(max(1.0 - alpha**2, 0.0))
        speed_noise = noise_scale * float(config.UE_GM_SPEED_SIGMA) * float(np.random.normal())
        theta_noise = noise_scale * float(config.UE_GM_THETA_SIGMA) * float(np.random.normal())

        # 保留上一时隙的运动趋势，同时加入少量速度和方向扰动。
        walk_speed = alpha * self.speed + (1.0 - alpha) * float(config.UE_WALK_SPEED_MEAN) + speed_noise
        walk_speed = float(np.clip(walk_speed, config.UE_GM_MIN_SPEED, config.UE_GM_MAX_SPEED))
        self.theta = float((self.theta + theta_noise) % (2.0 * np.pi))

        # UE 有活动 DAG 时降低移动速度，表示用户在等待服务期间不会快速离开。
        speed_scale = float(config.UE_SERVICE_WAITING_SPEED_SCALE) if self.service_waiting else 1.0
        effective_speed = walk_speed * speed_scale
        self.velocity = self._velocity_from_polar(speed=effective_speed)
        next_pos = self.pos[:2] + self.velocity * float(config.TIME_SLOT_DURATION)
        reflected_position = self._reflect_position(next_pos)
        if bool(commit_position):
            self.pos[:2] = reflected_position
        self.speed = walk_speed

    def is_inside_hotspot(self, hotspot_center: np.ndarray | None, hotspot_radius: float | None) -> bool:
        """判断当前 UE 是否位于本回合的圆形热点区域内。"""
        if hotspot_center is None or hotspot_radius is None:
            return False
        center = np.asarray(hotspot_center, dtype=np.float32).reshape(-1)[:2]
        distance = float(np.linalg.norm(self.pos[:2] - center))
        return distance <= float(hotspot_radius)

    def get_arrival_probability(self, hotspot_center: np.ndarray | None, hotspot_radius: float | None) -> float:
        """返回 UE 在当前时隙产生新 DAG 的概率。

        热点内的用户使用放大后的到达概率，最终结果会限制在 0 到 1 之间。
        """
        arrival_prob = float(config.DAG_BASE_ARRIVAL_PROB)
        if self.is_inside_hotspot(hotspot_center, hotspot_radius):
            arrival_prob *= float(config.DAG_HOTSPOT_ARRIVAL_MULTIPLIER)
        return float(np.clip(arrival_prob, 0.0, 1.0))

    def enter_service_waiting(self, dag_id: str) -> None:
        """记录 UE 当前正在等待指定 DAG 完成服务。"""
        self.active_dag_id = str(dag_id)
        self.service_waiting = True

    def release_service_waiting(self, dag_id: str) -> None:
        """指定 DAG 完成后解除等待；ID 不匹配时不修改状态。"""
        if self.active_dag_id != dag_id:
            return
        self.active_dag_id = None
        self.service_waiting = False

    def sync_service_waiting(self, active_dag_ids: list[str]) -> None:
        """Synchronize the compatibility UE marker from authoritative DAG state."""
        normalized = [str(active_dag_id) for active_dag_id in active_dag_ids]
        self.active_dag_id = normalized[-1] if normalized else None
        self.service_waiting = bool(normalized)

    def _sample_uniform_position(self) -> np.ndarray:
        """在矩形地图内均匀采样一个地面三维坐标，z 轴固定为 0。"""
        return np.array(
            [
                np.random.uniform(0.0, float(config.AREA_WIDTH)),
                np.random.uniform(0.0, float(config.AREA_HEIGHT)),
                0.0,
            ],
            dtype=np.float32,
        )

    def _sample_walk_speed(self) -> float:
        """在允许范围内随机采样 UE 的初始步行速度。"""
        upper = max(float(config.UE_WALK_SPEED_MEAN) * 2.0, float(config.UE_GM_MIN_SPEED))
        return float(np.random.uniform(float(config.UE_GM_MIN_SPEED), upper))

    def _velocity_from_polar(self, speed: float | None = None) -> np.ndarray:
        """把当前速度和方向角换成二维速度向量。"""
        actual_speed = self.speed if speed is None else float(speed)
        return np.array([np.cos(self.theta), np.sin(self.theta)], dtype=np.float32) * actual_speed

    def _reflect_position(self, position: np.ndarray) -> np.ndarray:
        """把越界位置镜像回地图内，并同步修正速度方向。"""
        reflected = np.array(position, dtype=np.float32)
        bounds = np.array([float(config.AREA_WIDTH), float(config.AREA_HEIGHT)], dtype=np.float32)
        for dim in range(2):
            while reflected[dim] < 0.0 or reflected[dim] > bounds[dim]:
                if reflected[dim] < 0.0:
                    reflected[dim] = -reflected[dim]
                    self.velocity[dim] = abs(self.velocity[dim])
                elif reflected[dim] > bounds[dim]:
                    reflected[dim] = 2.0 * bounds[dim] - reflected[dim]
                    self.velocity[dim] = -abs(self.velocity[dim])
        self.theta = float(np.arctan2(self.velocity[1], self.velocity[0]) % (2.0 * np.pi))
        return reflected
