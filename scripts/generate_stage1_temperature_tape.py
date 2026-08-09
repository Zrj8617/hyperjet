from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from environment.stage1_temperature_tape import FORMAL_EPISODES, build_manifest, generate_scenario_shard, save_json_create_only, validate_manifest

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--validate-only", action="store_true"); args = parser.parse_args()
    root, manifest_path = args.output_dir.resolve(), args.output_dir.resolve() / "manifest.json"
    if args.validate_only:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")); validate_manifest(manifest, root=root); print(json.dumps({"status":"PASS","logical_tape_sha256":manifest["logical_tape_sha256"]}, sort_keys=True)); return 0
    if manifest_path.exists() or (root.exists() and any(root.iterdir())): raise FileExistsError("tape generation is create-only")
    paths=[]
    for index in range(FORMAL_EPISODES):
        shard=generate_scenario_shard(index); path=root/"shards"/f"episode_{index:02d}.json"; save_json_create_only(shard,path); paths.append(path)
    manifest=build_manifest(paths,root=root); save_json_create_only(manifest,manifest_path); validate_manifest(manifest,root=root)
    print(json.dumps({"status":"PASS","logical_tape_sha256":manifest["logical_tape_sha256"]}, sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
