"""计算 ERS 排名并提供沿环境路由求和的参考时长。"""

from environment_v2.communication.routing import Route, Router
from environment_v2.contracts import Placements, Snapshot
from environment_v2.settings import Settings


class ERS:
    """用快照、路由和配置估计时长，只排序，不选择执行节点。"""

    def __init__(self, router: Router, settings: Settings):
        """保存环境的只读路由计算器和配置。"""
        self.router = router
        self.settings = settings

    def compute_duration(self, cycles: float, node: int | None) -> float:
        """返回指定执行节点的单核计算参考时长，单位 s。"""
        frequency = (self.settings.user.cpu_frequency_hz if node is None
                     else self.settings.uav.cpu_frequency_hz)
        return cycles / frequency

    def _route_duration(self, snapshot: Snapshot, route: Route, bits: float) -> float:
        return sum(self.router.transfer_duration_s(
            hop, bits, snapshot.ue_positions, snapshot.uav_positions,
        ) for hop in route.hops)

    def upload_duration(
        self, snapshot: Snapshot, ue_id: int, node: int | None, bits: float,
    ) -> float:
        """返回入口输入沿上传路径的参考时长，单位 s。"""
        route = self.router.upload(
            ue_id, node, bits, snapshot.ue_positions, snapshot.uav_positions,
        )
        return self._route_duration(snapshot, route, bits)

    def predecessor_duration(
        self, snapshot: Snapshot, ue_id: int, source: int | None,
        target: int | None, bits: float,
    ) -> float:
        """返回前驱结果沿执行节点间路径的参考时长，单位 s。"""
        route = self.router.predecessor(
            source, target, ue_id, bits, snapshot.ue_positions, snapshot.uav_positions,
        )
        return self._route_duration(snapshot, route, bits)

    def return_duration(
        self, snapshot: Snapshot, ue_id: int, node: int | None, bits: float,
    ) -> float:
        """返回 sink 回传的参考时长，关闭回传时为零，单位 s。"""
        if not self.settings.scheduling.result_return:
            return 0.0
        route = self.router.back(
            node, ue_id, bits, snapshot.ue_positions, snapshot.uav_positions,
        )
        return self._route_duration(snapshot, route, bits)

    def rank_tasks(self, snapshot: Snapshot) -> dict[tuple[int, int], float]:
        """计算全部待决策子任务的向上 rank，不计入口上传。"""
        ranks = {}
        for dag in snapshot.new_dags:
            if dag.id not in snapshot.candidates:
                continue
            candidates = snapshot.candidates[dag.id]
            seconds_per_cycle = sum(
                self.compute_duration(1.0, node) for node in candidates
            ) / len(candidates)
            seconds_per_bit = sum(
                self.predecessor_duration(snapshot, dag.ue_id, source, target, 1.0)
                for source in candidates for target in candidates
            ) / len(candidates) ** 2
            successors = [[] for _ in range(dag.num_tasks)]
            for parent, child in dag.edges:
                successors[parent].append(child)
            for node in sorted(range(dag.num_tasks), key=lambda i: dag.levels[i], reverse=True):
                ranks[dag.id, node] = float(dag.cpu_cycles[node]) * seconds_per_cycle + max(
                    (float(dag.output_bits[node]) * seconds_per_bit + ranks[dag.id, child]
                     for child in successors[node]), default=0.0,
                )
        return ranks


def sort_placements(placements: Placements, ranks: dict[tuple[int, int], float]) -> Placements:
    """按 rank 降序统一排序，相等时按 DAG 编号、子任务编号升序。"""
    return sorted(placements, key=lambda placement: (-ranks[placement[0]], placement[0]))
