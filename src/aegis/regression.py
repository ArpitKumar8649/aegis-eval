"""Regression gate engine comparing evaluation runs against baseline benchmarks.

Enforces strict CI/CD quality gates:
1. Zero Tolerated Regressions: Any test case that passed in baseline must not fail in candidate.
2. Safety Drift Threshold: The mean clinical safety score must not degrade past a configured margin (default 0.15).
3. Dataset Consistency Check: Flags when dataset hashes differ between runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from pydantic import BaseModel, Field
from aegis.runner import RunSummary


class RegressionDetectedError(Exception):
    """Raised when an evaluation run fails the regression quality gate."""
    pass


class RegressionDiff(BaseModel):
    baseline_run_id: str
    candidate_run_id: str
    dataset_hash_match: bool
    baseline_pass_rate: float
    candidate_pass_rate: float
    baseline_mean_safety: float
    candidate_mean_safety: float
    safety_delta: float
    baseline_mean_correctness: float
    candidate_mean_correctness: float
    correctness_delta: float
    regressed_case_ids: list[str] = Field(default_factory=list)
    improved_case_ids: list[str] = Field(default_factory=list)
    is_regression: bool
    rejection_reasons: list[str] = Field(default_factory=list)


def compare_runs(
    baseline: RunSummary,
    candidate: RunSummary,
    max_safety_drop: float = 0.15,
) -> RegressionDiff:
    """Compares candidate run against baseline and checks regression gates."""
    rejection_reasons: list[str] = []

    # Map case IDs to pass status
    baseline_case_status = {c.case_id: c.passed for c in baseline.cases}
    candidate_case_status = {c.case_id: c.passed for c in candidate.cases}

    regressed_cases: list[str] = []
    improved_cases: list[str] = []

    for case_id, passed_in_baseline in baseline_case_status.items():
        if case_id in candidate_case_status:
            passed_in_candidate = candidate_case_status[case_id]
            if passed_in_baseline and not passed_in_candidate:
                regressed_cases.append(case_id)
            elif not passed_in_baseline and passed_in_candidate:
                improved_cases.append(case_id)

    if regressed_cases:
        rejection_reasons.append(
            f"Regression detected in {len(regressed_cases)} previously passing case(s): {', '.join(regressed_cases)}"
        )

    safety_delta = round(candidate.mean_safety - baseline.mean_safety, 2)
    if safety_delta < -max_safety_drop:
        rejection_reasons.append(
            f"Clinical safety mean dropped from {baseline.mean_safety} to {candidate.mean_safety} "
            f"(delta: {safety_delta}), exceeding allowed threshold of -{max_safety_drop}"
        )

    correctness_delta = round(candidate.mean_correctness - baseline.mean_correctness, 2)
    hash_match = baseline.dataset_hash == candidate.dataset_hash

    is_regression = len(rejection_reasons) > 0

    return RegressionDiff(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        dataset_hash_match=hash_match,
        baseline_pass_rate=baseline.pass_rate,
        candidate_pass_rate=candidate.pass_rate,
        baseline_mean_safety=baseline.mean_safety,
        candidate_mean_safety=candidate.mean_safety,
        safety_delta=safety_delta,
        baseline_mean_correctness=baseline.mean_correctness,
        candidate_mean_correctness=candidate.mean_correctness,
        correctness_delta=correctness_delta,
        regressed_case_ids=regressed_cases,
        improved_case_ids=improved_cases,
        is_regression=is_regression,
        rejection_reasons=rejection_reasons,
    )


def load_run_summary(file_path: str | Path) -> RunSummary:
    """Loads and validates a RunSummary JSON file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Run summary not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return RunSummary.model_validate(data)
