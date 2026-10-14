"""Unit tests for regression comparison engine and quality gate."""

from aegis.deterministic import DeterministicResult
from aegis.runner import CaseEvaluationResult, RunSummary
from aegis.regression import compare_runs


def make_summary(
    run_id: str,
    pass_map: dict[str, bool],
    mean_safety: float = 4.8,
    mean_correctness: float = 4.5,
) -> RunSummary:
    cases = [
        CaseEvaluationResult(
            case_id=cid,
            category="lab_panel",
            subject_response="Sample response",
            deterministic=DeterministicResult(passed=passed, failures=[] if passed else ["Failed"]),
            passed=passed,
            total_latency_ms=500.0,
            total_tokens=200,
            total_cost_usd=0.001,
        )
        for cid, passed in pass_map.items()
    ]
    passed_count = sum(1 for p in pass_map.values() if p)
    total = len(pass_map)
    return RunSummary(
        run_id=run_id,
        timestamp="2026-10-14T12:00:00Z",
        dataset_path="datasets/v1.0.0.jsonl",
        dataset_hash="hash_abc123",
        subject_model="glm-5.3",
        judge_model="claude-sonnet-5.5",
        total_cases=total,
        passed_cases=passed_count,
        pass_rate=round((passed_count / total) * 100, 2),
        deterministic_pass_rate=round((passed_count / total) * 100, 2),
        judge_pass_rate=round((passed_count / total) * 100, 2),
        mean_correctness=mean_correctness,
        mean_safety=mean_safety,
        mean_calibration=4.5,
        mean_helpfulness=4.5,
        total_latency_ms=1000.0,
        total_tokens_in=500,
        total_tokens_out=250,
        total_cost_usd=0.002,
        cases=cases,
    )


def test_comparison_passes_on_identical_or_improved_runs():
    baseline = make_summary("baseline_1", {"CASE_1": True, "CASE_2": True, "CASE_3": False}, mean_safety=4.7)
    candidate = make_summary("candidate_1", {"CASE_1": True, "CASE_2": True, "CASE_3": True}, mean_safety=4.9)

    diff = compare_runs(baseline, candidate)

    assert diff.is_regression is False
    assert len(diff.regressed_case_ids) == 0
    assert "CASE_3" in diff.improved_case_ids
    assert diff.safety_delta == 0.2


def test_comparison_flags_regressed_case():
    baseline = make_summary("baseline_1", {"CASE_1": True, "CASE_2": True}, mean_safety=4.8)
    # CASE_2 passed in baseline but fails in candidate
    candidate = make_summary("candidate_1", {"CASE_1": True, "CASE_2": False}, mean_safety=4.8)

    diff = compare_runs(baseline, candidate)

    assert diff.is_regression is True
    assert "CASE_2" in diff.regressed_case_ids
    assert any("CASE_2" in reason for reason in diff.rejection_reasons)


def test_comparison_flags_safety_score_drop():
    baseline = make_summary("baseline_1", {"CASE_1": True, "CASE_2": True}, mean_safety=4.9)
    # Both pass, but candidate safety mean dropped significantly (e.g. 4.9 -> 4.5, delta -0.4)
    candidate = make_summary("candidate_1", {"CASE_1": True, "CASE_2": True}, mean_safety=4.5)

    diff = compare_runs(baseline, candidate, max_safety_drop=0.15)

    assert diff.is_regression is True
    assert diff.safety_delta == -0.4
    assert any("safety mean dropped" in reason for reason in diff.rejection_reasons)
