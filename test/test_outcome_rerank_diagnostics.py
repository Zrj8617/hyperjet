from __future__ import annotations

from analysis.analyze_outcome_rerank_candidates import classify_disagreement


def test_bad_score_decision_when_score_slower_and_task_dropped() -> None:
    row = {
        "delta_planned_finish": 0.2,
        "selected_deadline_margin": -0.1,
        "heuristic_deadline_margin": 0.2,
        "is_high_risk_task": True,
        "is_critical_path_task": False,
    }
    task_outcome = {"dropped": True, "finished": False, "finished_on_time": False}
    dag_outcome = {"failed": True, "successful": False, "on_time_successful": False}

    label, _ = classify_disagreement(row, task_outcome, dag_outcome)

    assert label == "BAD_SCORE_DECISION"


def test_strong_bad_score_decision_for_large_delta_high_risk_failure() -> None:
    row = {
        "delta_planned_finish": 0.6,
        "selected_deadline_margin": -0.5,
        "heuristic_deadline_margin": 0.3,
        "is_high_risk_task": True,
        "is_critical_path_task": False,
    }
    task_outcome = {"dropped": True, "finished": False, "finished_on_time": False}
    dag_outcome = {"failed": True, "successful": False, "on_time_successful": False}

    label, _ = classify_disagreement(row, task_outcome, dag_outcome, strong_delta_threshold=0.3)

    assert label == "BAD_SCORE_DECISION_STRONG"


def test_good_score_decision_when_on_time_and_successful_with_small_delta() -> None:
    row = {
        "delta_planned_finish": 0.05,
        "selected_deadline_margin": 0.4,
        "heuristic_deadline_margin": 0.5,
        "is_high_risk_task": True,
        "is_critical_path_task": True,
    }
    task_outcome = {"dropped": False, "finished": True, "finished_on_time": True}
    dag_outcome = {"failed": False, "successful": True, "on_time_successful": True}

    label, _ = classify_disagreement(row, task_outcome, dag_outcome)

    assert label == "GOOD_SCORE_DECISION"


def test_ambiguous_when_outcome_does_not_identify_bad_or_good() -> None:
    row = {
        "delta_planned_finish": 0.2,
        "selected_deadline_margin": 0.4,
        "heuristic_deadline_margin": 0.5,
        "is_high_risk_task": False,
        "is_critical_path_task": False,
    }
    task_outcome = {"dropped": False, "finished": False, "finished_on_time": False}
    dag_outcome = {"failed": False, "successful": False, "on_time_successful": False}

    label, _ = classify_disagreement(row, task_outcome, dag_outcome)

    assert label == "AMBIGUOUS"
