from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import config
import numpy as np
from environment.stage1_temperature_tape import instantiate_potential_template, potential_template_at
from environment.user_equipments import UE
from environment.stage1_temperature_diagnostic import Stage1TemperatureDiagnosticEnv
from environment.stage1_temperature_tape import generate_scenario_shard
def main()->int:
    np.random.seed(123); first=UE(0); state=np.random.get_state(); first.update_position(); expected=(first.pos.copy(),first.speed,first.theta)
    np.random.set_state(state); speed_draw=float(np.random.normal()); theta_draw=float(np.random.normal()); np.random.seed(123); second=UE(0); second.update_position(speed_standard_normal=speed_draw,theta_standard_normal=theta_draw); assert np.allclose(second.pos,expected[0]) and second.speed==expected[1] and second.theta==expected[2]
    with patch.object(config,"NUM_UES",2),patch.object(config,"NUM_UAVS",2):
        shard=generate_scenario_shard(0,num_ues=2,num_uavs=2); shard["arrival_uniforms"][0]=[0.0,1.0]; env=Stage1TemperatureDiagnosticEnv(scenario_shard=shard); env.reset(); job=instantiate_potential_template(env.task_manager,potential_template_at(shard,0,0),source_pos=env.ues[0].pos[:2],arrival_time=0.0); env.ues[0].enter_service_waiting(job.dag_id); env.prepare_slot_state(); funnel=env._latest_arrival_funnel; assert env.time_step==1 and funnel["arrival_attempt_count"]==2 and funnel["arrival_blocked_count"]==1 and funnel["arrival_draw_count"]==1 and funnel["arrival_sampled_event_count"]==0
    print("PASS smoke_stage1_temperature_env"); return 0
if __name__=="__main__": raise SystemExit(main())
