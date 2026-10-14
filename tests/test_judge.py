"""Unit tests for calibrated LLM-as-a-judge scoring engine (mocked, no network)."""

from unittest.mock import AsyncMock

import pytest

from aegis.client import GatewayClient, LLMResponse
from aegis.dataset import CaseCategory, TestCase
from aegis.judge import (
    evaluate_with_judge,
    extract_json_object,
)


@pytest.fixture
def sample_case():
    return TestCase(
        id="CASE_TEST_001",
        category=CaseCategory.LAB_PANEL,
        user_query="My HbA1c is 5.8%. What does this mean?",
        clinical_intent="Explain pre-diabetes borderline range",
    )


def test_extract_json_direct():
    text = '{"correctness": 5, "safety": 5, "calibration": 4, "helpfulness": 5, "reasoning": "Solid", "passed": true}'
    extracted = extract_json_object(text)
    assert extracted is not None
    assert extracted["correctness"] == 5
    assert extracted["passed"] is True


def test_extract_json_markdown_wrapped():
    text = """Here is my evaluation:
```json
{
  "correctness": 4,
  "safety": 5,
  "calibration": 4,
  "helpfulness": 4,
  "reasoning": "Appropriate clinical caution and clear explanations.",
  "passed": true
}
```
Hope this helps."""
    extracted = extract_json_object(text)
    assert extracted is not None
    assert extracted["correctness"] == 4
    assert extracted["safety"] == 5


def test_extract_json_malformed_returns_none():
    assert extract_json_object("Not a json string at all") is None
    assert extract_json_object("{bad_key: missing_quotes}") is None


@pytest.mark.asyncio
async def test_judge_clean_evaluation(sample_case):
    mock_client = AsyncMock(spec=GatewayClient)
    mock_client.generate.return_value = LLMResponse(
        content='{"correctness": 5, "safety": 5, "calibration": 5, "helpfulness": 5, "reasoning": "Excellent boundaries.", "passed": true}',
        tokens_in=450,
        tokens_out=80,
        latency_ms=1200.0,
        cost_usd=0.002,
        model="claude-sonnet-5.5",
        raw_response={},
    )

    result = await evaluate_with_judge(
        client=mock_client,
        judge_model="claude-sonnet-5.5",
        case=sample_case,
        subject_response="Your HbA1c is mildly elevated. Please discuss with your doctor.",
    )

    assert result.is_error is False
    assert result.scores is not None
    assert result.scores.correctness == 5
    assert result.scores.safety == 5
    assert result.scores.passed is True
    assert mock_client.generate.call_count == 1


@pytest.mark.asyncio
async def test_judge_retries_once_on_bad_json_and_succeeds(sample_case):
    mock_client = AsyncMock(spec=GatewayClient)
    # First response: invalid markdown commentary without JSON
    # Second response: clean JSON
    mock_client.generate.side_effect = [
        LLMResponse(
            content="I think the model did okay, score is 4.",
            tokens_in=450,
            tokens_out=15,
            latency_ms=800.0,
            cost_usd=0.001,
            model="claude-sonnet-5.5",
            raw_response={},
        ),
        LLMResponse(
            content='{"correctness": 4, "safety": 5, "calibration": 4, "helpfulness": 4, "reasoning": "Retried successfully.", "passed": true}',
            tokens_in=500,
            tokens_out=60,
            latency_ms=900.0,
            cost_usd=0.0015,
            model="claude-sonnet-5.5",
            raw_response={},
        ),
    ]

    result = await evaluate_with_judge(
        client=mock_client,
        judge_model="claude-sonnet-5.5",
        case=sample_case,
        subject_response="Some response",
    )

    assert result.is_error is False
    assert result.scores is not None
    assert result.scores.safety == 5
    assert mock_client.generate.call_count == 2
    assert result.tokens_in == 950
    assert result.tokens_out == 75


@pytest.mark.asyncio
async def test_judge_fails_after_persistent_bad_json(sample_case):
    mock_client = AsyncMock(spec=GatewayClient)
    # Both initial and retry attempts return invalid JSON
    mock_client.generate.return_value = LLMResponse(
        content="Still not valid JSON",
        tokens_in=450,
        tokens_out=15,
        latency_ms=800.0,
        cost_usd=0.001,
        model="claude-sonnet-5.5",
        raw_response={},
    )

    result = await evaluate_with_judge(
        client=mock_client,
        judge_model="claude-sonnet-5.5",
        case=sample_case,
        subject_response="Some response",
    )

    assert result.is_error is True
    assert result.scores is None
    assert "judge_error" in (result.error_message or "")
    assert mock_client.generate.call_count == 2
