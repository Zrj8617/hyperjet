decision_gae_3seed_results

Contents:
- code_snapshot/: pre-launch HEAD, git status, diff, diff stat, modified-file list.
- shared_seed{0,1,2}/ and decision_seed{0,1,2}/: full train_metrics.jsonl,
  run_summary.json, resolved config.json, and four deterministic evaluations.
- eval/update_XXXX.json: aggregate eval summary for 10 deterministic episodes.
- eval/update_XXXX_episodes.jsonl: original per-episode evaluation metrics.
- paired_initialization_check.json: direct module/optimizer/RNG equality result.
- paired_config_comparison.json: resolved-config semantic comparison.
- training_metrics_field_audit.json: actual diagnostic-key inventory.
- manifest.json: run and evaluation provenance.

Evaluation protocol matches the prior EFT bridge:
10 episodes, 200 arrival steps, up to 300 drain steps, evaluation seed equal
to the training seed, actor-argmax deterministic offloading, deterministic
learned movement actor. No checkpoint .pt files are included.
