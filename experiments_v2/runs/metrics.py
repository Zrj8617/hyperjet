"""训练、评估和基线共用的 episode 指标统计。"""

import numpy as np

from methods_v2.mappo.reward import compute_total_delay


ENERGY_FIELDS = ("compute_energy_j", "tx_energy_j", "return_energy_j", "flight_energy_j")
SEGMENT_SLOTS = 20


def direction_cosines(snapshot, flight):
    """计算飞行方向与关联且被覆盖 UE 重心方向的余弦。"""
    values = []
    for uav, action in enumerate(flight):
        associated = (snapshot.manager_uav == uav) & snapshot.coverage[:, uav]
        if not associated.any():
            values.append(None)
            continue
        direction = (
            snapshot.ue_positions[associated, :2].mean(axis=0)
            - snapshot.uav_positions[uav, :2]
        )
        denominator = np.linalg.norm(action) * np.linalg.norm(direction)
        values.append(float(np.dot(action, direction) / denominator) if denominator > 0 else None)
    return values


class FlightStats:
    """按二十个时隙一段累计飞行和覆盖统计。"""

    FIELDS = (
        "covered_share", "speed_mps", "user_cosine",
        "lost_share", "crowded_share", "empty_hotspot_share",
    )

    def __init__(self, settings, hotspot_centers, hotspot_radius_m):
        self.settings = settings
        self.hotspot_centers = np.asarray(hotspot_centers)
        self.hotspot_radius_m = hotspot_radius_m
        segments = settings.scheduling.arrival_slots // SEGMENT_SLOTS
        self.counts = {field: np.zeros((segments, 2)) for field in self.FIELDS}

    def add(self, snapshot, flight, positions_after):
        """累计当前时隙的六项原始分子和分母。"""
        segment = snapshot.slot // SEGMENT_SLOTS
        uav_count = len(snapshot.uav_positions)
        covered_ids = set(snapshot.candidates)
        self.counts["covered_share"][segment] += (len(covered_ids), len(snapshot.new_dags))

        distance = np.linalg.norm(
            positions_after[:, :2] - snapshot.uav_positions[:, :2], axis=1,
        ).sum()
        duration = uav_count * self.settings.scheduling.slot_duration_s
        self.counts["speed_mps"][segment] += (distance, duration)

        cosines = [value for value in direction_cosines(snapshot, flight) if value is not None]
        self.counts["user_cosine"][segment] += (sum(cosines), len(cosines))

        associated = np.asarray([
            np.sum((snapshot.manager_uav == uav) & snapshot.coverage[:, uav])
            for uav in range(uav_count)
        ])
        self.counts["lost_share"][segment] += (np.sum(associated == 0), uav_count)

        distances = np.linalg.norm(
            positions_after[:, None, :2] - self.hotspot_centers[None, :, :], axis=2,
        )
        hotspot_counts = (distances <= self.hotspot_radius_m).sum(axis=0)
        self.counts["crowded_share"][segment] += (
            np.maximum(hotspot_counts - 1, 0).sum(), uav_count,
        )
        self.counts["empty_hotspot_share"][segment] += (
            np.sum(hotspot_counts == 0), len(self.hotspot_centers),
        )

    @classmethod
    def _report(cls, counts):
        rows = []
        for segment in range(next(iter(counts.values())).shape[0]):
            row = {"start_slot": segment * SEGMENT_SLOTS}
            for field in cls.FIELDS:
                numerator, denominator = counts[field][segment]
                row[field] = float(numerator / denominator) if denominator > 0 else None
            rows.append(row)
        return rows

    def report(self):
        """返回当前 episode 的十段统计。"""
        return self._report(self.counts)

    @classmethod
    def combine(cls, items):
        """先合并多个 episode 的分子分母，再计算十段比例。"""
        counts = {
            field: sum((item.counts[field] for item in items), np.zeros_like(items[0].counts[field]))
            for field in cls.FIELDS
        }
        return cls._report(counts)


class EpisodeMetrics:
    """按覆盖归属累计 DAG、放置和能耗指标。"""

    def __init__(self):
        self.offers = self.completed = self.invalid = 0
        self.covered_offers = self.covered_completed = 0
        self.uncovered_offers = self.uncovered_completed = 0
        self.delay_s = self.covered_delay_s = self.uncovered_delay_s = 0.0
        self.covered_tasks = self.task_ue = self.task_manager = self.task_neighbor = 0
        self.reward = 0.0
        self.energy = {name: 0.0 for name in ENERGY_FIELDS}

    def add(self, snapshot, result, placements):
        """用动作前快照划分覆盖归属，并累计本时隙结果。"""
        covered_ids = set(snapshot.candidates)
        self.offers += len(result.offers)
        self.completed += len(result.completed_dags)
        self.invalid += result.invalid_placements
        self.delay_s += compute_total_delay(result)
        for offer in result.offers:
            if offer.dag_id in covered_ids:
                self.covered_offers += 1
            else:
                self.uncovered_offers += 1
        for dag in result.completed_dags:
            delay = dag.finish_time_s - dag.arrival_time_s
            if dag.dag_id in covered_ids:
                self.covered_completed += 1
                self.covered_delay_s += delay
            else:
                self.uncovered_completed += 1
                self.uncovered_delay_s += delay
        for dag in result.timed_out_dags:
            delay = dag.finish_time_s - dag.arrival_time_s
            if dag.dag_id in covered_ids:
                self.covered_delay_s += delay
            else:
                self.uncovered_delay_s += delay
        dags = {dag.id: dag for dag in snapshot.new_dags}
        for (dag_id, _), node in placements:
            manager = int(snapshot.manager_uav[dags[dag_id].ue_id])
            self.covered_tasks += 1
            if node is None:
                self.task_ue += 1
            elif node == manager:
                self.task_manager += 1
            else:
                self.task_neighbor += 1
        for name in ENERGY_FIELDS:
            self.energy[name] += getattr(result, name)

    def report(self):
        """返回 episode 日志所需的主指标、解释指标和原始计数。"""
        return dict(
            episode_return=self.reward,
            offers=self.offers,
            completed=self.completed,
            timed_out=self.offers - self.completed,
            total_delay_s=self.delay_s,
            on_time_rate=self.completed / self.offers,
            average_delay_s=self.delay_s / self.offers,
            covered_offers=self.covered_offers,
            covered_completed=self.covered_completed,
            covered_share=self.covered_offers / self.offers,
            covered_on_time_rate=self.covered_completed / self.covered_offers,
            covered_average_delay_s=self.covered_delay_s / self.covered_offers,
            uncovered_offers=self.uncovered_offers,
            uncovered_completed=self.uncovered_completed,
            uncovered_on_time_rate=self.uncovered_completed / self.uncovered_offers,
            uncovered_average_delay_s=self.uncovered_delay_s / self.uncovered_offers,
            covered_tasks=self.covered_tasks,
            covered_task_ue_share=self.task_ue / self.covered_tasks,
            covered_task_manager_share=self.task_manager / self.covered_tasks,
            covered_task_neighbor_share=self.task_neighbor / self.covered_tasks,
            invalid_placements=self.invalid,
            **self.energy,
        )


def summarize(episodes):
    """DAG 和子任务按总数聚合，回报与能耗取 episode 均值。"""
    offers = sum(item.offers for item in episodes)
    completed = sum(item.completed for item in episodes)
    covered_offers = sum(item.covered_offers for item in episodes)
    uncovered_offers = sum(item.uncovered_offers for item in episodes)
    covered_completed = sum(item.covered_completed for item in episodes)
    uncovered_completed = sum(item.uncovered_completed for item in episodes)
    covered_tasks = sum(item.covered_tasks for item in episodes)
    return dict(
        episode_return=float(np.mean([item.reward for item in episodes])),
        offers=offers,
        completed=completed,
        timed_out=offers - completed,
        total_delay_s=sum(item.delay_s for item in episodes),
        on_time_rate=completed / offers,
        average_delay_s=sum(item.delay_s for item in episodes) / offers,
        covered_offers=covered_offers,
        covered_completed=covered_completed,
        covered_share=covered_offers / offers,
        covered_on_time_rate=covered_completed / covered_offers,
        covered_average_delay_s=sum(item.covered_delay_s for item in episodes) / covered_offers,
        uncovered_offers=uncovered_offers,
        uncovered_completed=uncovered_completed,
        uncovered_on_time_rate=uncovered_completed / uncovered_offers,
        uncovered_average_delay_s=sum(item.uncovered_delay_s for item in episodes) / uncovered_offers,
        covered_tasks=covered_tasks,
        covered_task_ue_share=sum(item.task_ue for item in episodes) / covered_tasks,
        covered_task_manager_share=sum(item.task_manager for item in episodes) / covered_tasks,
        covered_task_neighbor_share=sum(item.task_neighbor for item in episodes) / covered_tasks,
        invalid_placements=sum(item.invalid for item in episodes),
        **{name: float(np.mean([item.energy[name] for item in episodes])) for name in ENERGY_FIELDS},
    )
