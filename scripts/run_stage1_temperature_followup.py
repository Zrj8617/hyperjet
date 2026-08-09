from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from environment.graph_builder import CleanGraphBuilder
from environment.stage1_temperature_analysis import replay_static_record
from environment.stage1_temperature_diagnostic import CHECKPOINT_SETS, Stage1TemperatureDiagnosticEnv, act_with_temperature, load_frozen_checkpoint
from environment.stage1_temperature_sampling import FROZEN_TEMPERATURES, canonical_sha256, file_sha256
from environment.stage1_temperature_tape import load_scenario_shard, validate_manifest
from marl_models.mappo.clean_slot_orchestrator import prepare_slot_state

SWEEP_TEMPERATURES = (1.0,)


def _args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description="Frozen checkpoint-only Stage 1 temperature follow-up")
    parser.add_argument("--phase", choices=("pilot","formal","sweep"), required=True); parser.add_argument("--tape-dir", type=Path, required=True); parser.add_argument("--checkpoint-root", type=Path, default=ROOT); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-set", choices=("formal_v1", "long_v1", "b1_v1", "b1_sweep"), default="formal_v1")
    parser.add_argument("--temperatures", nargs="+", type=float, default=list(FROZEN_TEMPERATURES)); parser.add_argument("--sampling-replicates", nargs="+", type=int, required=True); parser.add_argument("--scenario-indices", nargs="+", type=int, required=True); parser.add_argument("--max-physical-slots", type=int, default=200)
    parser.add_argument("--checkpoint-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--checkpoint-update", type=int, default=None)
    parser.add_argument("--allow-shard", action="store_true", help="Run a create-only subset for parallel formal evaluation; merge before analysis.")
    return parser.parse_args()


def _validate_controls(args: argparse.Namespace) -> tuple[tuple[int,...],tuple[int,...]]:
    scenarios = tuple(args.scenario_indices); replicates = tuple(args.sampling_replicates)
    if args.phase == "sweep":
        if tuple(args.temperatures) != SWEEP_TEMPERATURES or replicates != (0,1,2) or int(args.max_physical_slots) != 200:
            raise ValueError("sweep controls do not match frozen plan")
        if args.allow_shard:
            if not scenarios or any(value not in tuple(range(20)) for value in scenarios):
                raise ValueError("sweep shard scenario controls do not match frozen plan")
        elif scenarios != tuple(range(20)):
            raise ValueError("sweep controls do not match frozen plan")
        return scenarios, replicates
    expected_scenarios = (0,1) if args.phase == "pilot" else tuple(range(20)); expected_replicates = (0,1) if args.phase == "pilot" else tuple(range(5))
    if args.allow_shard:
        if tuple(args.temperatures) != FROZEN_TEMPERATURES or int(args.max_physical_slots) != 200: raise ValueError("runner controls do not match frozen phase")
        if not scenarios or any(value not in expected_scenarios for value in scenarios): raise ValueError("shard scenario controls do not match frozen phase")
        if not replicates or any(value not in expected_replicates for value in replicates): raise ValueError("shard replicate controls do not match frozen phase")
    elif tuple(args.temperatures) != FROZEN_TEMPERATURES or scenarios != expected_scenarios or replicates != expected_replicates or int(args.max_physical_slots) != 200: raise ValueError("runner controls do not match frozen phase")
    return scenarios, replicates


def _write_line(handle: Any, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)+"\n"); handle.flush()


def main() -> int:
    import torch
    args=_args(); scenarios, replicates=_validate_controls(args)
    output=args.output_dir.resolve()
    if output.exists(): raise FileExistsError("output directory is create-only")
    output.mkdir(parents=True)
    tape_root=args.tape_dir.resolve(); manifest=json.loads((tape_root/"manifest.json").read_text(encoding="utf-8")); validate_manifest(manifest,root=tape_root)
    corpus_dir=output/"static_corpus"; replay_dir=output/"static_replay"; closed_dir=output/"closed_loop"; analysis_dir=output/"analysis"
    for path in (corpus_dir,replay_dir,closed_dir,analysis_dir): path.mkdir()
    corpus_path=corpus_dir/"records.jsonl"; replay_path=replay_dir/"records.jsonl"; closed_path=closed_dir/"episodes.jsonl"
    seen_static:set[tuple[Any,...]]=set(); rows_written=0; static_written=0; checkpoint_metadata={}
    with corpus_path.open("x",encoding="utf-8",newline="\n") as corpus_handle, closed_path.open("x",encoding="utf-8",newline="\n") as closed_handle:
        source_registry = CHECKPOINT_SETS[args.checkpoint_set]
        if args.checkpoint_set == "b1_sweep":
            if args.checkpoint_update is None:
                raise ValueError("b1_sweep requires --checkpoint-update")
            seeds = (42, 86, 1042) if args.checkpoint_seeds is None else tuple(args.checkpoint_seeds)
            registry = {seed: (*source_registry[(seed, int(args.checkpoint_update))], int(args.checkpoint_update)) for seed in seeds}
        else:
            if args.checkpoint_update is not None:
                raise ValueError("--checkpoint-update is only valid with b1_sweep")
            registry = source_registry
        requested_seeds = tuple(registry.keys()) if args.checkpoint_seeds is None else tuple(args.checkpoint_seeds)
        if not requested_seeds or any(seed not in registry for seed in requested_seeds): raise ValueError("checkpoint seed is not frozen in the selected set")
        for training_seed in requested_seeds:
            entry = registry[training_seed]
            relative_path=entry[0]; expected_update=entry[2] if len(entry)>2 else 30
            checkpoint=args.checkpoint_root/relative_path; encoder,actor,checkpoint_meta=load_frozen_checkpoint(checkpoint,training_seed=training_seed,device=args.device,registry=registry,expected_completed_update=expected_update)
            checkpoint_metadata[str(training_seed)]=checkpoint_meta
            for scenario_index in scenarios:
                record=manifest["shards"][scenario_index]; shard=load_scenario_shard(tape_root/record["path"]); scenario_seed=int(shard["evaluation_scenario_seed"])
                for replicate in replicates:
                    for temperature in tuple(args.temperatures):
                        env=Stage1TemperatureDiagnosticEnv(scenario_shard=shard); builder=CleanGraphBuilder(); env.reset(); builder.reset(); reward_total=0.0; latest:dict[str,Any]={}; episode_static=[]
                        try:
                            for slot_index in range(200):
                                prepared=prepare_slot_state(env=env,graph_builder=builder); env.apply_movement({})
                                features=torch.as_tensor(np.array(prepared.graph_snapshot.task_features,dtype=np.float32,copy=True),dtype=torch.float32,device=torch.device(args.device))
                                with torch.no_grad(): embeddings=encoder(features)
                                ready=[env.task_manager.get_task(task_id) for task_id in prepared.frozen_ready_task_ids]; ready=[task for task in ready if task is not None]
                                assignments,static_records,skips=act_with_temperature(actor=actor,frozen_ready_tasks=ready,task_embeddings=embeddings,graph_snapshot=prepared.graph_snapshot,task_manager=env.task_manager,uavs=env.uavs,executor=env.executor,current_time_seconds=env.current_time_seconds,uav_service_positions=env.uav_service_positions,ue_service_positions=env.ue_service_positions,ues=env.ues,temperature=temperature,checkpoint_sha256=checkpoint_meta["checkpoint_sha256"],evaluation_scenario_seed=scenario_seed,slot_index=slot_index,sampling_replicate=replicate,record_static=(temperature==1.0))
                                _,_,_,latest=env.commit_and_advance(assignment_buffer=assignments,offloading_skip_count=skips); reward_total+=float(latest["step_reward"]); episode_static.extend(static_records)
                        finally: builder.close()
                        if temperature==1.0:
                            for static in episode_static:
                                key=tuple(static[name] for name in ("checkpoint_sha256","evaluation_scenario_seed","slot_index","stable_task_id","decision_order","sampling_replicate"))
                                if key in seen_static: raise ValueError("duplicate static record key")
                                seen_static.add(key); static["record_sha256"]=canonical_sha256(static); _write_line(corpus_handle,static); static_written+=1
                        arrival=env.arrival_identity_metrics(); completed=int(latest.get("completed_dag_count",0)); generated=int(arrival["generated_dag_count"])
                        row={"schema":"stage1_temperature_closed_loop_episode_v1","phase":args.phase,"training_seed":training_seed,"checkpoint_sha256":checkpoint_meta["checkpoint_sha256"],"logical_tape_sha256":manifest["logical_tape_sha256"],"episode_index":scenario_index,"evaluation_scenario_seed":scenario_seed,"sampling_replicate":replicate,"temperature":temperature,"physical_slots":200,"active_dag_cap":1,"queue_cap":16,"encoder":"mlp","actor_uav_feature_dim":7,"episode_reward_total":reward_total,"completed_dag_count":completed,"generated_dag_count":generated,"admitted_dag_count":arrival["admitted_dag_count"],"arrival_blocked_count":arrival["arrival_blocked_count"],"dag_completion_rate":latest.get("dag_completion_rate"),"average_dag_flowtime":latest.get("average_dag_flowtime"),"avg_uav_queue_length":latest.get("avg_uav_queue_length"),"admitted_incomplete_backlog":generated-completed,"invalid_assignment_count":int(latest.get("invalid_assignment_count",0)),"finite":bool(math.isfinite(reward_total))}
                        _write_line(closed_handle,row); rows_written+=1
    expected=len(requested_seeds)*len(scenarios)*len(replicates)*len(tuple(args.temperatures))
    if rows_written!=expected: raise AssertionError(f"closed-loop row count {rows_written} != {expected}")
    corpus_sha256=file_sha256(corpus_path)
    with replay_path.open("x",encoding="utf-8",newline="\n") as replay_handle, corpus_path.open(encoding="utf-8") as corpus_handle:
        for line in corpus_handle:
            if not line.strip(): continue
            static=json.loads(line)
            if canonical_sha256({key:value for key,value in static.items() if key!="record_sha256"})!=static["record_sha256"]: raise ValueError("static record checksum mismatch")
            for replay_temperature in tuple(args.temperatures):
                replay=replay_static_record(static,replay_temperature); replay["source_corpus_sha256"]=corpus_sha256; _write_line(replay_handle,replay)
    summary={"schema":"stage1_temperature_run_manifest_v1","phase":args.phase,"checkpoint_set":args.checkpoint_set,"checkpoint_update":args.checkpoint_update,"technical_pass":True,"partial_shard":bool(args.allow_shard),"checkpoint_seeds":[int(seed) for seed in requested_seeds],"scenario_indices":[int(value) for value in scenarios],"sampling_replicates":[int(value) for value in replicates],"temperatures":[float(value) for value in tuple(args.temperatures)],"closed_loop_rows":rows_written,"static_corpus_records":static_written,"static_replay_records":static_written*len(tuple(args.temperatures)),"static_corpus_sha256":corpus_sha256,"static_replay_sha256":file_sha256(replay_path),"closed_loop_sha256":file_sha256(closed_path),"checkpoint_metadata":checkpoint_metadata,"logical_tape_sha256":manifest["logical_tape_sha256"],"pairing_limitation":"matched starts and keyed noise are not strict counterfactual pairs after trajectories diverge","deployment_temperature_selected":False}
    (output/"run_manifest.json").write_text(json.dumps(summary,indent=2,sort_keys=True),encoding="utf-8"); print(json.dumps(summary,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
