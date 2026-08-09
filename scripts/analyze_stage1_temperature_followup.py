from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from environment.stage1_temperature_analysis import ANALYSIS_SCHEMA, checkpoint_guardrail, classify, combined_guardrail, deterministic_reachability, group_static_rows, static_temperature_gate
from environment.stage1_temperature_sampling import file_sha256

def _jsonl(path:Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip(): yield json.loads(line)

def main()->int:
    parser=argparse.ArgumentParser(); parser.add_argument("--run-dir",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    if args.output.exists(): raise FileExistsError("analysis output is create-only")
    manifest=json.loads((args.run_dir/"run_manifest.json").read_text(encoding="utf-8")); closed=list(_jsonl(args.run_dir/"closed_loop"/"episodes.jsonl")); replay=list(_jsonl(args.run_dir/"static_replay"/"records.jsonl"))
    expected=48 if manifest["phase"]=="pilot" else 1200
    checksum_pass=(file_sha256(args.run_dir/"static_corpus"/"records.jsonl")==manifest["static_corpus_sha256"] and file_sha256(args.run_dir/"static_replay"/"records.jsonl")==manifest["static_replay_sha256"] and file_sha256(args.run_dir/"closed_loop"/"episodes.jsonl")==manifest["closed_loop_sha256"] and all(row["source_corpus_sha256"]==manifest["static_corpus_sha256"] for row in replay))
    technical=bool(manifest.get("technical_pass")) and checksum_pass and len(closed)==expected and len(replay)==int(manifest["static_replay_records"]) and all(row["physical_slots"]==200 and row["finite"] and row["invalid_assignment_count"]==0 and row["evaluation_scenario_seed"]==424242+row["episode_index"] for row in closed)
    static=group_static_rows(replay); hashes=sorted({row["checkpoint_sha256"] for row in replay}); reachability={checkpoint:deterministic_reachability(static[(checkpoint,1.0)]) for checkpoint in hashes}
    closed_means={}
    for checkpoint in hashes:
        for temperature in (1.0,0.75,0.5,0.25):
            selected=[row for row in closed if row["checkpoint_sha256"]==checkpoint and row["temperature"]==temperature]; closed_means[(checkpoint,temperature)]={metric:sum(float(row[metric]) for row in selected if row.get(metric) is not None)/len([row for row in selected if row.get(metric) is not None]) for metric in ("completed_dag_count","dag_completion_rate","episode_reward_total","average_dag_flowtime","admitted_incomplete_backlog") if any(row.get(metric) is not None for row in selected)}
    guardrails={}
    for temperature in (0.75,0.5,0.25): guardrails[str(temperature)]=combined_guardrail({checkpoint:checkpoint_guardrail(closed_means[(checkpoint,1.0)],closed_means[(checkpoint,temperature)]) for checkpoint in hashes})
    gates={} if manifest["phase"]=="pilot" else {str(t):static_temperature_gate(replay,t) for t in (0.75,0.5,0.25)}
    moderate=manifest["phase"]=="formal" and any(gates[str(t)]["common_pass"] and guardrails[str(t)]["pass"] for t in (0.75,0.5)); hard=manifest["phase"]=="formal" and gates["0.25"]["common_pass"] and guardrails["0.25"]["pass"]
    classification="technical_only" if manifest["phase"]=="pilot" and technical else classify(technical_pass=technical,reachability_by_checkpoint=reachability,moderate_pass=moderate,hard_pass=hard)
    result={"schema":ANALYSIS_SCHEMA,"phase":manifest["phase"],"technical_pass":technical,"static_by_checkpoint_temperature":{f"{key[0]}@{key[1]}":value for key,value in static.items()},"deterministic_reachability":reachability,"static_temperature_gates":gates,"closed_loop_guardrails":guardrails,"classification":classification,"classification_note":"diagnostic classification only; no deployment temperature is selected","deployment_temperature_selected":False}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False),encoding="utf-8"); print(json.dumps({"technical_pass":technical,"classification":result["classification"]},sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
