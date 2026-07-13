# Clean deterministic evaluation correctness design

## Scope

This change is limited to the clean deterministic evaluation entrypoint and its smoke coverage. It does not change training, rewards, model architecture, checkpoints, or environment dynamics.

## Evaluation protocol

Each evaluation episode has two consecutive phases:

1. **Arrival phase:** execute exactly `--arrival-steps` slots with normal DAG arrivals unless the environment terminates earlier.
2. **Drain phase:** disable new DAG arrivals and continue execution for at most `--max-drain-steps`, stopping early when all active DAGs finish.

Immediately after the final arrival-phase slot and before the first drain slot, snapshot the environment metrics. The primary 200-slot completion anchor is:

`arrival_DAG_completion_rate = arrival_completed_DAG_count / max(arrival_generated_DAG_count, 1)`

The per-episode JSONL row and aggregate summary will add:

- `arrival_generated_DAG_count`
- `arrival_completed_DAG_count`
- `arrival_DAG_completion_rate`
- `arrival_active_DAG_count`
- `arrival_active_task_count`

Existing unprefixed metrics remain drain-final metrics for backward compatibility. In particular, `Average_DAG_flowtime` remains the flowtime after drain.

Aggregate arrival completion is computed from aggregate counts, not as an unweighted mean of episode rates:

`sum(arrival_completed_DAG_count) / max(sum(arrival_generated_DAG_count), 1)`

## Frozen movement semantics

Add `--freeze-movement` to the evaluation CLI. When enabled, every UAV receives the configured hover action in every arrival and drain slot. The movement actor is not sampled for action selection. Offloading remains deterministic masked argmax and is otherwise unchanged.

The evaluation config, per-episode metrics, and aggregate summary record `movement_frozen`. Frozen runs must report a hover-only movement distribution and zero mean displacement apart from any environment-level numerical tolerance.

Freeze-trained checkpoints will be evaluated with `--freeze-movement`; learned-movement checkpoints will be evaluated without it.

## Checkpoint loading

Clean checkpoints are trusted project-generated artifacts that include optimizer and NumPy state. Load them explicitly with `torch.load(..., weights_only=False)` so PyTorch 2.6 does not silently switch to weights-only loading. If the installed PyTorch predates the `weights_only` parameter, retry without that keyword.

No generic untrusted checkpoint loading is introduced; the config and summary identify the checkpoint path used.

## Smoke coverage

Extend the existing eval entrypoint smoke to verify:

- CLI parsing and config recording of `--freeze-movement`;
- arrival snapshot fields and count-weighted aggregation;
- preservation of existing drain-final fields;
- frozen movement forces hover without sampling the movement actor;
- trusted checkpoint loading passes `weights_only=False` and retains the old-PyTorch fallback;
- summary schema remains valid for zero-completion and max-drain cases.

The smoke must be behavior-local and must not require a production checkpoint.

## Validation and rollout

Run the eval smoke and the relevant clean evaluation/training-loop smoke suite locally. Create one final eval-only correctness commit containing this design, implementation, and smoke changes. Push `zrj_3`, fast-forward the server, rerun all 20 ep20/40/60/80/100 checkpoints using 20 paired evaluation seeds, 200 arrival slots, and up to 500 drain slots, then analyze arrival completion and drain flowtime separately.

## Non-goals

- No change to `w_c`, `value_coef`, optimizer, PPO, HGNN, or reward accounting.
- No deletion or rewriting of previous evaluation outputs.
- No interpretation of the earlier freeze evaluation as a valid frozen-movement anchor.
