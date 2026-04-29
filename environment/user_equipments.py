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
        cls.all_ids = np.arange(config.NUM_FILES)
        cls.global_ranks = np.arange(1, config.NUM_FILES + 1)
        np.random.shuffle(cls.global_ranks)
        cls.id_to_rank_map = dict(zip(cls.all_ids, cls.global_ranks))
        zipf_denom: float = np.sum(1 / cls.global_ranks**config.ZIPF_BETA)
        cls.global_probs = (1 / cls.global_ranks**config.ZIPF_BETA) / zipf_denom
        cls.hotspot_centers = cls._init_hotspot_centers()

    @classmethod
    def _init_hotspot_centers(cls) -> np.ndarray:
        if not config.ENABLE_UE_HOTSPOTS or config.NUM_HOTSPOTS <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.stack(
            [
                np.array(
                    [np.random.uniform(0, config.AREA_WIDTH), np.random.uniform(0, config.AREA_HEIGHT)],
                    dtype=np.float32,
                )
                for _ in range(config.NUM_HOTSPOTS)
            ],
            axis=0,
        )

    def __init__(self, ue_id: int) -> None:
        self.id: int = ue_id
        self.is_hotspot: bool = self._is_hotspot_ue(ue_id)
        self.hotspot_id: int | None = self._assign_hotspot_id(ue_id) if self.is_hotspot else None
        self.pos: np.ndarray = self._init_position()
        self.battery_level: float = np.random.uniform(0.6, 1.0) * config.UE_BATTERY_CAPACITY

        self.current_request: tuple[int, int, int] = (0, 0, 0)
        self.latency_current_request: float = 0.0
        self.assigned: bool = False

        self.speed: float = float(np.random.uniform(config.UE_GM_MIN_SPEED, config.UE_GM_MAX_SPEED))
        self.theta: float = float(np.random.uniform(0.0, 2.0 * np.pi))
        self.velocity: np.ndarray = self._velocity_from_polar()

        self._successful_requests: int = 0
        self.service_coverage: float = 0.0

    def _is_hotspot_ue(self, ue_id: int) -> bool:
        if not config.ENABLE_UE_HOTSPOTS or len(UE.hotspot_centers) == 0:
            return False
        hotspot_count = int(round(config.NUM_UES * config.HOTSPOT_UE_RATIO))
        return ue_id < max(0, min(config.NUM_UES, hotspot_count))

    def _assign_hotspot_id(self, ue_id: int) -> int | None:
        if len(UE.hotspot_centers) == 0:
            return None
        return int(ue_id % len(UE.hotspot_centers))

    def _init_position(self) -> np.ndarray:
        if self.is_hotspot and self.hotspot_id is not None:
            center = UE.hotspot_centers[self.hotspot_id]
            pos_xy = np.random.normal(center, config.HOTSPOT_STD, size=2).astype(np.float32)
            pos_xy = np.clip(pos_xy, [0.0, 0.0], [config.AREA_WIDTH, config.AREA_HEIGHT])
        else:
            pos_xy = np.array(
                [np.random.uniform(0, config.AREA_WIDTH), np.random.uniform(0, config.AREA_HEIGHT)],
                dtype=np.float32,
            )
        return np.array([pos_xy[0], pos_xy[1], 0.0], dtype=np.float32)

    def update_position(self) -> None:
        alpha = config.UE_GM_ALPHA
        noise_scale = np.sqrt(max(1.0 - alpha**2, 0.0))
        speed_noise = noise_scale * config.UE_GM_SPEED_SIGMA * float(np.random.normal())
        theta_noise = noise_scale * config.UE_GM_THETA_SIGMA * float(np.random.normal())
        self.speed = alpha * self.speed + (1.0 - alpha) * config.UE_GM_MEAN_SPEED + speed_noise
        self.speed = float(np.clip(self.speed, config.UE_GM_MIN_SPEED, config.UE_GM_MAX_SPEED))
        self.theta = float((self.theta + theta_noise) % (2.0 * np.pi))
        self.velocity = self._velocity_from_polar()
        next_pos = self.pos[:2] + self.velocity * config.TIME_SLOT_DURATION
        self.pos[:2] = self._reflect_position(next_pos)

    def generate_request(self) -> None:
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

    def _velocity_from_polar(self) -> np.ndarray:
        return np.array([np.cos(self.theta), np.sin(self.theta)], dtype=np.float32) * float(self.speed)

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
        self.speed = float(np.clip(np.linalg.norm(self.velocity), config.UE_GM_MIN_SPEED, config.UE_GM_MAX_SPEED))
        self.theta = float(np.arctan2(self.velocity[1], self.velocity[0]) % (2.0 * np.pi))
        return reflected

    def update_battery(self, harv_energy: float, ue_transmit_time: float) -> None:
        consumed_energy: float = config.UE_STATIC_POWER * config.TIME_SLOT_DURATION
        consumed_energy += config.TRANSMIT_POWER * ue_transmit_time
        self.battery_level = min(config.UE_BATTERY_CAPACITY, self.battery_level - consumed_energy + harv_energy)
        self.battery_level = max(0.0, self.battery_level)
