import glob
import json


directories = glob.glob("logs/decision_ppo_bandit/*_stage1_formal_*")
cells = []
for directory in directories:
    with open(directory + "/summary.json", encoding="utf-8") as handle:
        summary = json.load(handle)
    with open(directory + "/updates.jsonl", encoding="utf-8") as handle:
        updates = [json.loads(line) for line in handle if line.strip()]
    cells.append((summary, updates))

assert len(cells) == 6
assert all(
    summary["technical_pass"]
    and summary["completed_updates"] == 30
    and summary["global_slot"] == 3840
    for summary, _updates in cells
)
assert all(all(row["finite"] for row in updates) for _summary, updates in cells)
assert all(
    all(row["update"]["optimizer_step_count"] == 0 for row in updates)
    for summary, updates in cells
    if summary["group"] == "S1-A"
)
assert all(
    all(
        row["update"]["optimizer_step_count"]
        == (0 if row["update"]["empty_actor_batch"] else 3)
        for row in updates
    )
    for summary, updates in cells
    if summary["group"] == "S1-B"
)
identities = {
    (summary["seed"], summary["group"]): summary["initialization_identity"]
    for summary, _updates in cells
}
assert all(
    identities[(seed, "S1-A")] == identities[(seed, "S1-B")]
    for seed in (42, 86, 1042)
)
print("TECHNICAL_AUDIT_PASS")
