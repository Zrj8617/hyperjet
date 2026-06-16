import config
import numpy as np


class UE:
    all_ids: np.ndarray
    global_ranks: np.ndarray
    id_to_rank_map: dict[int, int]
    global_probs: np.ndarray
    hotspot_centers: np.ndarray

    @classmethod
    def initialize_ue_class(cls) -> None:
        """Initialize legacy request-popularity state.

        Clean mainline does not use hotspot UE identity. `hotspot_centers` stays
        empty and is kept only so old imports do not crash.
        """
        cls.all_ids = np.arange(config.NUM_FILES)
        cls.global_ranks = np.arange(1, config.NUM_FILES + 1)
        np.random.shuffle(cls.global_ranks)
        cls.id_to_rank_map = dict(zip(cls.all_ids, cls.global_ranks))
        zipf_denom: float = np.sum(1 / cls.global_ranks**config.ZIPF_BETA)
        cls.global_probs = (1 / cls.global_ranks**config.ZIPF_BETA) / zipf_denom
        cls.hotspot_centers = np.zeros((0, 2), dtype=np.float32)

    def __init__(self, ue_id: int) -> None:
        self.id: int = ue_id
        # Deprecated compatibility fields. Clean mainline hotspot is a map
        # region sampled by Env.reset(), never a UE identity.
        self.is_hotspot: bool = False
        self.hotspot_id: int | None = None

        self.pos: np.ndarray = self._sample_uniform_position()
        self.battery_level: float = np.random.uniform(0.6, 1.0) * config.UE_BATTERY_CAPACITY

        self.current_request: tuple[int, int, int] = (0, 0, 0)
        self.latency_current_request: float = 0.0
        self.assigned: bool = False

        self.service_waiting: bool = False
        self.active_dag_id: str | None = None

        self.speed: float = float(
            np.random.uniform(config.UE_GM_MIN_SPEED, max(config.UE_WALK_SPEED_MEAN * 2.0, config.UE_WALK_SPEED_MEAN))
        )
        self.theta: float = float(np.random.uniform(0.0, 2.0 * np.pi))
        self.velocity: np.ndarray = self._velocity_from_polar()

        self._successful_requests: int = 0
        self.service_coverage: float = 0.0

    def reset_episode_state(self, *, uniform_position: bool = True) -> None:
        if uniform_position:
            self.pos = self._sample_uniform_position()
        self.service_waiting = False
        self.active_dag_id = None
        self.assigned = False
        self.current_request = (0, 0, 0)
        self.latency_current_request = 0.0
        self.speed = float(
            np.random.uniform(config.UE_GM_MIN_SPEED, max(config.UE_WALK_SPEED_MEAN * 2.0, config.UE_WALK_SPEED_MEAN))
        )
        self.theta = float(np.random.uniform(0.0, 2.0 * np.pi))
        self.velocity = self._velocity_from_polar()
        self._successful_requests = 0
        self.service_coverage = 0.0

    def update_position(self) -> None:
        alpha = config.UE_GM_ALPHA
        noise_scale = np.sqrt(max(1.0 - alpha**2, 0.0))
        speed_noise = noise_scale * config.UE_GM_SPEED_SIGMA * float(np.random.normal())
        theta_noise = noise_scale * config.UE_GM_THETA_SIGMA * float(np.random.normal())
        base_speed = alpha * self.speed + (1.0 - alpha) * config.UE_WALK_SPEED_MEAN + speed_noise
        base_speed = float(np.clip(base_speed, config.UE_GM_MIN_SPEED, config.UE_GM_MAX_SPEED))
        self.theta = float((self.theta + theta_noise) % (2.0 * np.pi))
        scale = config.UE_SERVICE_WAITING_SPEED_SCALE if self.service_waiting else 1.0
        effective_speed = max(base_speed * float(scale), 0.0)
        self.velocity = self._velocity_from_polar(speed=effective_speed)
        next_pos = self.pos[:2] + self.velocity * config.TIME_SLOT_DURATION
        self.pos[:2] = self._reflect_position(next_pos)
        # Store the unscaled walking state so service-waiting does not
        # permanently slow the underlying Gaussian-Markov process.
        self.speed = base_speed

    def is_inside_hotspot(self, hotspot_center: np.ndarray | None, hotspot_radius: float | None) -> bool:
        if hotspot_center is None or hotspot_radius is None:
            return False
        center = np.asarray(hotspot_center, dtype=np.float32).reshape(-1)[:2]
        return bool(float(np.linalg.norm(self.pos[:2] - center)) <= float(hotspot_radius))

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

    def generate_request(self) -> None:
        """Deprecated legacy cache/request generator. Not used by clean mainline."""
        if self.battery_level < config.UE_CRITICAL_THRESHOLD:
            self.current_request = (2, 0, 0)
            self.latency_current_request = 0.0
            self.assigned = False
            return

        req_id: int = np.random.choice(UE.all_ids, p=UE.global_probs)
        req_type: int = 0 if req_id < config.NUM_SERVICES else 1
        req_size: int = np.random.randint(config.MIN_INPUT_SIZE, config.MAX_INPUT_SIZE) if req_type == 0 else 0
        self.current_request = (req_type, req_size, req_id)
        self.latency_current_request = 0.0
        self.assigned = False

    def update_service_coverage(self, current_time_step_t: int) -> None:
        if self.assigned and self.latency_current_request <= config.TIME_SLOT_DURATION:
            self._successful_requests += 1

        assert current_time_step_t > 0
        self.service_coverage = self._successful_requests / current_time_step_t

    def _sample_uniform_position(self) -> np.ndarray:
        pos_xy = np.array(
            [np.random.uniform(0, config.AREA_WIDTH), np.random.uniform(0, config.AREA_HEIGHT)],
            dtype=np.float32,
        )
        return np.array([pos_xy[0], pos_xy[1], 0.0], dtype=np.float32)

    def _velocity_from_polar(self, speed: float | None = None) -> np.ndarray:
        actual_speed = self.speed if speed is None else float(speed)
        return np.array([np.cos(self.theta), np.sin(self.theta)], dtype=np.float32) * float(actual_speed)

    def _reflect_position(self, position: np.ndarray) -> np.ndarray:
        reflected = np.array(position, dtype=np.float32)
        bounds = np.array([config.AREA_WIDTH, config.AREA_HEIGHT], dtype=np.float32)
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

    def update_battery(self, harv_energy: float, ue_transmit_time: float) -> None:
        consumed_energy: float = config.UE_STATIC_POWER * config.TIME_SLOT_DURATION
        consumed_energy += config.TRANSMIT_POWER * ue_transmit_time
        self.battery_level = min(config.UE_BATTERY_CAPACITY, self.battery_level - consumed_energy + harv_energy)
        self.battery_level = max(0.0, self.battery_level)
