from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from environment.stage1_temperature_diagnostic import FROZEN_CHECKPOINTS,load_frozen_checkpoint
def main()->int:
    parser=argparse.ArgumentParser(); parser.add_argument("--checkpoint-root",type=Path,default=ROOT); parser.add_argument("--device",default="cpu"); args=parser.parse_args()
    for seed,(relative,_) in FROZEN_CHECKPOINTS.items():
        _,_,metadata=load_frozen_checkpoint(args.checkpoint_root/relative,training_seed=seed,device=args.device); assert metadata["encoder_strict_load_pass"] and metadata["resolved_task_feature_dim"]==12
    print("PASS smoke_stage1_temperature_closed_loop"); return 0
if __name__=="__main__": raise SystemExit(main())
