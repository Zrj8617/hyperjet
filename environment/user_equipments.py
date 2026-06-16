import config
import numpy as np


class UE:
    def __init__(self, ue_id: int) -> None:
        self.id: int = ue_id
        self.pos: np.ndarray = self._sample_uniform_position()
        self.speed: float = self._sample_walk_speed()
        self.theta: float = float(np.random.uniform(0.0, 2.0 * np.pi))
        self.velocity: np.ndarray = self._velocity_from_polar()
        self.service_waiting: bool = False
        self.active_dag_id: str | None = None

    def reset_episode_state(self) -> None:
        self.pos = self._sample_uniform_position()
        self.speed = self._sample_walk_speed()
        self.theta = float(np.random.uniform(0.0, 2.0 * np.pi))
        self.velocity = self._velocity_from_polar()
        self.service_waiting = False
        self.active_dag_id = None

    def update_position(self) -> None:
        alpha = float(config.UE_GM_ALPHA)
        noise_scale = np.sqrt(max(1.0 - alpha**2, 0.0))
        speed_noise = noise_scale * float(config.UE_GM_SPEED_SIGMA) * float(np.random.normal())
        theta_noise = noise_scale * float(config.UE_GM_THETA_SIGMA) * float(np.random.normal())

        walk_speed = alpha * self.speed + (1.0 - alpha) * float(config.UE_WALK_SPEED_MEAN) + speed_noise
        walk_speed = float(np.clip(walk_speed, config.UE_GM_MIN_SPEED, config.UE_GM_MAX_SPEED))
        self.theta = float((self.theta + theta_noise) % (2.0 * np.pi))

        speed_scale = float(config.UE_SERVICE_WAITING_SPEED_SCALE) if self.service_waiting else 1.0
        effective_speed = walk_speed * speed_scale
        self.velocity = self._velocity_from_polar(speed=effective_speed)
        next_pos = self.pos[:2] + self.velocity * float(config.TIME_SLOT_DURATION)
        self.pos[:2] = self._reflect_position(next_pos)
        self.speed = walk_speed

    def is_inside_hotspot(self, hotspot_center: np.ndarray | None, hotspot_radius: float | None) -> bool:
        if hotspot_center is None or hotspot_radius is None:
            return False
        center = np.asarray(hotspot_center, dtype=np.float32).reshape(-1)[:2]
        distance = float(np.linalg.norm(self.pos[:2] - center))
        return distance <= float(hotspot_radius)

    def get_arrival_probability(self, hotspot_center: np.ndarray | None, hotspot_radius: float | None) -> float:
        arrival_prob = float(config.DAG_BASE_ARRIVAL_PROB)
        if self.is_inside_hotspot(hotspot_center, hotspot_radius):
            arrival_prob *= float(config.DAG_HOTSPOT_ARRIVAL_MULTIPLIER)
        return float(np.clip(arrival_prob, 0.0, 1.0))

    def enter_service_waiting(self, dag_id: str) -> None:
        self.active_dag_id = str(dag_id)
        self.service_waiting = True

    def release_service_waiting(self, dag_id: str) -> None:
        if self.active_dag_id != dag_id:
            return
        self.active_dag_id = None
        self.service_waiting = False

    def _sample_uniform_position(self) -> np.ndarray:
        return np.array(
            [
                np.random.uniform(0.0, float(config.AREA_WIDTH)),
                np.random.uniform(0.0, float(config.AREA_HEIGHT)),
                0.0,
            ],
            dtype=np.float32,
        )

    def _sample_walk_speed(self) -> float:
        upper = max(float(config.UE_WALK_SPEED_MEAN) * 2.0, float(config.UE_GM_MIN_SPEED))
        return float(np.random.uniform(float(config.UE_GM_MIN_SPEED), upper))

    def _velocity_from_polar(self, speed: float | None = None) -> np.ndarray:
        actual_speed = self.speed if speed is None else float(speed)
        return np.array([np.cos(self.theta), np.sin(self.theta)], dtype=np.float32) * actual_speed

    def _reflect_position(self, position: np.ndarray) -> np.ndarray:
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
