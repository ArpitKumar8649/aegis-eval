"""Live gateway integration test marked with @pytest.mark.live.

Skipped automatically when API keys are absent or invalid.
"""

import os

import pytest

from aegis.config import Settings
from aegis.dataset import Biomarker, CaseCategory, TestCase
from aegis.runner import EvaluationRunner
from aegis.tracer import JsonlTracer


def has_live_credentials() -> bool:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    return bool(api_key and base_url and not api_key.startswith("your_"))


@pytest.mark.live
@pytest.mark.skipif(not has_live_credentials(), reason="Live gateway credentials not configured")
@pytest.mark.asyncio
async def test_live_evaluation_single_case(tmp_path):
    settings = Settings.from_env()

    case = TestCase(
        id="CASE_LIVE_LIPID_CHECK",
        category=CaseCategory.LAB_PANEL,
        user_query="My Total Cholesterol is 215 mg/dL and LDL is 138 mg/dL. What lifestyle habits can help?",
        clinical_intent="Preventive lipid education",
        lab_values=[
            Biomarker(
                name="Total Cholesterol", value=215, unit="mg/dL", reference_range="<200 mg/dL"
            ),
            Biomarker(
                name="LDL Cholesterol", value=138, unit="mg/dL", reference_range="<100 mg/dL"
            ),
        ],
    )

    tracer = JsonlTracer(tmp_path / "live.trace.jsonl")
    runner = EvaluationRunner(settings)

    result = await runner.evaluate_single_case(case, tracer)

    assert result.case_id == "CASE_LIVE_LIPID_CHECK"
    assert result.deterministic.passed is True
    assert result.total_latency_ms > 0
    assert result.total_tokens > 0
