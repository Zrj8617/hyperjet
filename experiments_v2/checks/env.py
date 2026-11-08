"""打印固定输入、分布对照、随机运行量级和随机与贪心差距。"""
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config as reference_config
from environment.dag_tasks import DAGTaskManager
from environment.user_equipments import UE as ReferenceUE
from environment_v2 import simulator
from environment_v2.entities.region import Region
from environment_v2.entities.user import UE
from environment_v2.settings import load_settings
from environment_v2.workload.arrival import ArrivalProcess
from environment_v2.workload.dag_generator import generate_dag
from methods_v2.baselines.policies import choose_random, choose_greedy, sample_flight
from methods_v2.ers import ERS

def summarize(values):
    """返回样本数、均值、最小值、分位数和最大值。"""
    return f"n={len(values)} " + " ".join(f"{v:.6g}" for v in
        [np.mean(values), *np.quantile(values, [0, .05, .5, .95, 1])])

class ArrivalProbe(np.random.Generator):
    """保存到达过程实际使用的概率，仍执行原伯努利采样。"""
    def binomial(self, n, p):
        """记录概率后交给原生成器抽样。"""
        self.probabilities = p.copy()
        return super().binomial(n, p)

class TraceRuntime(simulator.EventRuntime):
    """在每个时隙算完后保存该时隙记录。"""
    def run(self, dags, placements):
        """执行原运行器并保存完整结果。"""
        TraceRuntime.records = super().run(dags, placements)
        return TraceRuntime.records

def main():
    """按当前 TOML 运行一次性检查，只打印，不作统计裁决。"""
    settings = load_settings(ROOT / "config")
    d, u, r = settings.dag, settings.user, settings.region
    count, bits_per_mib = 20000, 8 * 1024**2
    np.random.seed(42)
    rng = np.random.default_rng(42)
    parameters = dict(DAG_MIN_TASKS=d.min_tasks, DAG_MAX_TASKS=d.max_tasks,
        DAG_MAX_LEVELS=d.max_levels, DAG_MAX_PARENTS=d.max_parents, BASE_UNIT_BYTES=d.base_unit_bits / 8,
        INPUT_DATA_SIZE_MB_RANGE=(d.min_input_bits / bits_per_mib, d.max_input_bits / bits_per_mib),
        OUTPUT_DATA_SIZE_MB_RANGE=(d.min_output_bits / bits_per_mib, d.max_output_bits / bits_per_mib),
        TASK_CONSTANT_RANGE=(int(np.ceil(d.min_task_constant)), int(np.floor(d.max_task_constant))),
        TASK_COMPLEXITY_PROBS=dict(n=d.linear_probability, nlogn=d.log_probability, nlog2n=d.squared_log_probability),
        AREA_WIDTH=r.side_length_m, AREA_HEIGHT=r.side_length_m, TIME_SLOT_DURATION=u.move_duration_s,
        UE_WALK_SPEED_MEAN=u.mean_speed, UE_GM_ALPHA=u.gm_alpha, UE_GM_SPEED_SIGMA=u.speed_sigma,
        UE_GM_THETA_SIGMA=u.heading_sigma, UE_GM_MIN_SPEED=0., UE_GM_MAX_SPEED=np.inf,
        DAG_BASE_ARRIVAL_PROB=u.arrival_probability, DAG_HOTSPOT_ARRIVAL_MULTIPLIER=r.arrival_multiplier)
    reference_config.__dict__.update(parameters)
    print("reference 参数:", parameters, flush=True)
    center = np.array([r.side_length_m / 2] * 2)
    region = Region(center[None, :], r.hotspot_radius_m)
    distances = np.array([0., r.hotspot_radius_m - 1, r.hotspot_radius_m, r.hotspot_radius_m + 1, 2 * r.hotspot_radius_m])
    positions = np.column_stack((center[0] + distances, np.full(5, center[1]), np.zeros(5)))
    probe = ArrivalProbe(np.random.PCG64(43))
    arrival = ArrivalProcess(settings, region, SimpleNamespace(arrival=probe, dag=rng))
    arrivals = np.zeros(5)
    for slot in range(count):
        for dag in arrival.generate(slot, positions):
            arrivals[dag.ue_id] += 1
    for i, (distance, region_id) in enumerate(zip(distances, region.get_region_ids(positions))):
        expected = u.arrival_probability * (r.arrival_multiplier if distance <= region.radius_m else 1)
        reference = ReferenceUE(i)
        reference.pos = positions[i].copy()
        probability = reference.get_arrival_probability(center, region.radius_m)
        print(f"固定点 distance_m={distance:g} region={region_id}/{int(distance <= region.radius_m)} "
              f"p={probe.probabilities[i]:g}/{expected:g} equal={probe.probabilities[i] == expected} "
              f"arrival_rate={arrivals[i] / count:.6f} reference_p={probability:g}")
    print("reference 实际到达率需整个环境的准入流程，本次只打印当前到达率及 reference 理论概率。")
    samples, comparison = defaultdict(list), defaultdict(list)
    manager = DAGTaskManager()
    for index in range(count):
        dag = generate_dag(index, 0, 0., d, rng)
        manager._tasks.clear()
        levels = []
        for level, size in enumerate(manager._sample_level_sizes()):
            keys = [str(len(manager._tasks) + j) for j in range(size)]
            for key in keys:
                manager._tasks[key] = manager._create_task_node(key, str(index), 0, level, np.zeros(2), 0.)
            levels.append(keys)
        manager._connect_levels(levels)
        tasks = list(manager._tasks.values())
        parents = np.bincount([child for _, child in dag.edges], minlength=dag.num_tasks)
        for data, values in ((samples, ([dag.num_tasks], [max(dag.levels) + 1], parents[parents > 0],
                dag.input_bits[parents == 0], dag.output_bits, [sum(dag.cpu_cycles)])),
                (comparison, ([len(tasks)], [len(levels)], [len(t.predecessors) for t in tasks if t.predecessors],
                [t.input_data_size_mb * bits_per_mib for t in tasks if not t.predecessors],
                [t.output_data_size_mb * bits_per_mib for t in tasks], [sum(t.num_operation for t in tasks)]))):
            for name, values in zip(("tasks", "levels", "parents", "input_bits", "output_bits", "dag_cycles"), values):
                data[name].extend(values)
    print("分布列: n mean min p05 p50 p95 max; 左=当前，右=reference")
    for name in samples:
        print(name, summarize(samples[name]), "|", summarize(comparison[name]), flush=True)
    print("位移对照: 相同初速度/方向；reference 不等待服务且关闭速度上限；当前热点减速保留。")
    for i in (0, 4):
        speeds, headings = rng.uniform(0, 2 * u.mean_speed, count), rng.uniform(0, 2 * np.pi, count)
        users = UE(np.tile(positions[i], (count, 1)), speeds.copy(), headings.copy())
        users.move(u, r, region, rng)
        displacement = []
        for speed, heading in zip(speeds, headings):
            reference = ReferenceUE(0)
            reference.pos, reference.speed, reference.theta = positions[i].copy(), speed, heading
            reference.update_position()
            displacement.append(np.linalg.norm(reference.pos - positions[i]))
        print(f"displacement_m region={int(i == 0)}", summarize(np.linalg.norm(users.positions - positions[i], axis=1)),
              "|", summarize(displacement), flush=True)
    simulator.EventRuntime = TraceRuntime
    timing = defaultdict(list)
    for seed, episode in ((42, 0), (42, 1), (43, 0)):
        results, trajectories = [], []
        for policy in (choose_random, choose_greedy):
            scene = simulator.Simulator(settings)
            snapshot = scene.reset(seed, episode)
            ers = ERS(scene.router, settings)
            flight = np.random.default_rng(np.random.SeedSequence([seed, episode, 101]))
            placement = np.random.default_rng(np.random.SeedSequence([seed, episode, 102]))
            offered = completed = timed_out = 0
            trajectory = [snapshot.uav_positions.copy()]
            while snapshot is not None:
                placements = policy(snapshot, ers, placement) if policy is choose_random else policy(snapshot, ers)
                result, following = scene.step(sample_flight(snapshot, flight), placements)
                trajectory.append(snapshot.uav_positions.copy())
                offered += len(result.offers)
                completed += len(result.completed_dags)
                timed_out += len(result.timed_out_dags)
                if policy is choose_random:
                    events = defaultdict(list)
                    for record in TraceRuntime.records.values():
                        for transfer in record.transfers:
                            for job in transfer.jobs:
                                timing["hop_s"].append(job.finish_time_s - job.start_time_s)
                                timing["link_queue_s"].append(job.start_time_s - job.source_ready_s)
                        for task in record.tasks.values():
                            timing["cpu_queue_s"].append(task.start_time_s - task.ready_time_s)
                            if task.ready_time_s < task.start_time_s:
                                node = ("ue", record.dag.ue_id) if task.execution_node is None else ("uav", task.execution_node)
                                events[node].extend(((task.ready_time_s, 1), (task.start_time_s, -1)))
                    timing["queue_peak"].append(max([0] + [max(np.cumsum([delta for _, delta in sorted(points)]))
                                                          for points in events.values()]))
                snapshot = following
            results.append(completed / offered)
            trajectories.append(np.array(trajectory))
            print(f"seed={seed} episode={episode} {policy.__name__} slots={len(trajectory)-1} offers={offered} "
                  f"on_time_rate={results[-1]:.6f} timed_out={timed_out}", flush=True)
        print(f"gap_pp={100 * (results[1]-results[0]):.6f} same_flight={np.array_equal(*trajectories)}")
    for name, values in timing.items():
        print(name, summarize(values))

if __name__ == "__main__":
    output = ROOT / "output/checks/env"
    output.mkdir(parents=True, exist_ok=True)
    log = output / "check.log"
    with log.open("w", encoding="utf-8") as stream, redirect_stdout(stream):
        main()
    print(log.read_text(encoding="utf-8"), end="")
    print(log.read_text(encoding="utf-8"), end="")
