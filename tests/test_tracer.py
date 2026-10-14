"""Unit tests for flat JSONL execution tracer."""

import json

import pytest

from aegis.tracer import JsonlTracer, TraceRecord, TraceStatus, TraceStep


@pytest.mark.asyncio
async def test_tracer_appends_flat_jsonl(tmp_path):
    trace_file = tmp_path / "run.trace.jsonl"
    tracer = JsonlTracer(trace_file)

    await tracer.log_step(
        case_id="CASE_001",
        step=TraceStep.GENERATE,
        status=TraceStatus.SUCCESS,
        model="glm-5.3",
        latency_ms=452.3,
        tokens_in=120,
        tokens_out=64,
        cost_usd=0.00015,
    )

    await tracer.log_step(
        case_id="CASE_001",
        step=TraceStep.DET_CHECK,
        status=TraceStatus.SUCCESS,
        latency_ms=1.2,
    )

    await tracer.log_step(
        case_id="CASE_001",
        step=TraceStep.JUDGE,
        status=TraceStatus.SUCCESS,
        model="claude-sonnet-5.5",
        latency_ms=890.1,
        tokens_in=350,
        tokens_out=85,
        cost_usd=0.00225,
    )

    assert trace_file.exists()
    lines = trace_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3

    parsed = [TraceRecord.model_validate(json.loads(line)) for line in lines]
    assert parsed[0].step == TraceStep.GENERATE
    assert parsed[0].case_id == "CASE_001"
    assert parsed[0].model == "glm-5.3"
    assert parsed[0].tokens_in == 120
    assert parsed[0].tokens_out == 64

    assert parsed[1].step == TraceStep.DET_CHECK
    assert parsed[1].status == TraceStatus.SUCCESS

    assert parsed[2].step == TraceStep.JUDGE
    assert parsed[2].model == "claude-sonnet-5.5"

    t_in, t_out = tracer.total_tokens()
    assert t_in == 470
    assert t_out == 149
    assert tracer.total_cost() == pytest.approx(0.00240, rel=1e-4)


@pytest.mark.asyncio
async def test_tracer_records_error_details(tmp_path):
    trace_file = tmp_path / "error.trace.jsonl"
    tracer = JsonlTracer(trace_file)

    await tracer.log_step(
        case_id="CASE_002",
        step=TraceStep.DET_CHECK,
        status=TraceStatus.FAILED,
        error="Failed groundedness check: hallucinated biomarker value 999.0",
        metadata={"violation": "ungrounded_number"},
    )

    lines = trace_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    record = TraceRecord.model_validate(json.loads(lines[0]))
    assert record.status == TraceStatus.FAILED
    assert "hallucinated" in (record.error or "")
    assert record.metadata.get("violation") == "ungrounded_number"
