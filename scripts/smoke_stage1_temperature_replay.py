from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from environment.stage1_temperature_analysis import replay_static_record
def main()->int:
    row={"checkpoint_sha256":"ab"*32,"evaluation_scenario_seed":424242,"episode_index":0,"slot_index":0,"stable_task_id":"x","decision_order":0,"sampling_replicate":0,"candidate_uav_ids":[0,1],"candidate_mask":[True,True],"raw_logits":[2.0,1.0],"eft":[30.0,10.0],"gumbels":[0.1,0.2]}
    before=repr(row); outputs=[replay_static_record(row,t) for t in (1.0,.75,.5,.25)]; assert repr(row)==before and all(value["deterministic_uav_id"]==0 for value in outputs); print("PASS smoke_stage1_temperature_replay"); return 0
if __name__=="__main__": raise SystemExit(main())
