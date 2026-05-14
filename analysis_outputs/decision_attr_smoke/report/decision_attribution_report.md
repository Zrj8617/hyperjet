# Decision Attribution Report

## Key MVP Metrics

- score_selected_rate: 0.5244
- disagreement_rate: 0.2317
- mean_delta_planned_finish(score selected): 0.1759
- pct_delta_planned_finish_gt_0(score selected): 0.4419
- mean_delta_deadline_margin(score selected): -0.1759
- task_drop_rate disagreement vs agree: 0.0000 vs 0.0000
- DAG failure rate any_disagreement vs no_disagreement: 0.0000 vs 0.0000
- DAG failure rate critical_disagreement vs no_disagreement: 0.0000 vs 0.0000

## Evidence-Based Diagnosis

The current degradation is most consistent with: **teacher bias, student ranking instability, gate hardest-case exposure, static baseline strength**.

This is a diagnostic hint, not a proof. Use the CSV files to decide whether to change teacher constraints, ranking imitation, or selective gate thresholds.

## Tables

Detailed CSV outputs are written next to this report:

- decision_attribution_summary.csv
- decision_attribution_by_bucket.csv
- decision_attribution_task_outcomes.csv
- decision_attribution_dag_effects.csv
- decision_attribution_dag_groups.csv
- decision_attribution_gate_quality.csv
