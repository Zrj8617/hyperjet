Decision-GAE-v2 3-seed result bundle.

Each seed directory contains the complete train_metrics.jsonl, run_summary.json,
resolved config.json, and four deterministic checkpoint evaluation JSON files.
Each evaluation JSON includes aggregate summary plus all 10 raw episode metrics.

Evaluation protocol matches the prior EFT bridge: 10 episodes, 200 arrival
steps, up to 300 drain steps, eval seed equal to training seed, deterministic
actor-argmax offloading, and deterministic learned movement.

The evaluator does not emit episode reward. The convenience reward field is
therefore null and explicitly marked unavailable; training rewards are preserved
in full in train_metrics.jsonl. No checkpoint .pt files are included.
