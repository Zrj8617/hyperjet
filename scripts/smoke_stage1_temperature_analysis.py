from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from environment.stage1_temperature_analysis import checkpoint_guardrail,combined_guardrail,deterministic_reachability
def main()->int:
    ceiling=deterministic_reachability({"sampled_mean_EFT_regret":10.0,"deterministic_mean_EFT_regret":6.0,"deterministic_margin20_accuracy":.8}); assert not ceiling["reachable"]
    base={"completed_dag_count":100,"dag_completion_rate":.8,"episode_reward_total":100,"average_dag_flowtime":100,"admitted_incomplete_backlog":10}; a=dict(base,completed_dag_count=92); b=dict(base,average_dag_flowtime=115); c=dict(base)
    combined=combined_guardrail({"a":checkpoint_guardrail(base,a),"b":checkpoint_guardrail(base,b),"c":checkpoint_guardrail(base,c)}); assert not combined["pass"]
    catastrophic=dict(base,completed_dag_count=80); assert not combined_guardrail({"a":checkpoint_guardrail(base,catastrophic),"b":checkpoint_guardrail(base,c),"c":checkpoint_guardrail(base,c)})["pass"]
    print("PASS smoke_stage1_temperature_analysis"); return 0
if __name__=="__main__": raise SystemExit(main())
