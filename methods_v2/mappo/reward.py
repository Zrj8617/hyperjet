"""按全部到达 DAG 的实际完成时延计算团队奖励。"""


def compute_total_delay(result):
    """统计按时及超时 DAG 的实际时延，非法分配立即报错。"""
    if result.invalid_placements != 0:
        raise ValueError(f"invalid_placements={result.invalid_placements}")
    return sum(dag.finish_time_s - dag.arrival_time_s
               for dag in (*result.completed_dags, *result.timed_out_dags))


def compute_reward(result, deadline_s, expected_arrivals):
    """团队奖励为负的总归一化时延，以固定期望到达数缩放。"""
    return -compute_total_delay(result) / (expected_arrivals * deadline_s)
