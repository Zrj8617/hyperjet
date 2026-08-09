from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import config
from environment.stage1_temperature_tape import generate_scenario_shard,validate_scenario_shard
def main()->int:
    with patch.object(config,"NUM_UES",2),patch.object(config,"NUM_UAVS",2): shard=generate_scenario_shard(0,num_ues=2,num_uavs=2)
    validate_scenario_shard(shard); text=repr(shard); assert "arrival_bits" not in text and "source_pos" not in text and "arrival_time" not in text; print("PASS smoke_stage1_temperature_tape"); return 0
if __name__=="__main__": raise SystemExit(main())
