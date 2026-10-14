"""Flat JSONL execution tracer for Aegis-Eval.

Logs exactly one flat record per step (generate, det_check, judge) with
timing, token counts, cost, status, and error details.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class TraceStep(StrEnum):
    GENERATE = "generate"
    DET_CHECK = "det_check"
    JUDGE = "judge"


class TraceStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


class TraceRecord(BaseModel):
    case_id: str = Field(..., description="Test case identifier")
    step: TraceStep = Field(..., description="Pipeline execution step")
    model: str | None = Field(default=None, description="Model identifier if LLM step")
    latency_ms: float = Field(default=0.0, description="Step duration in milliseconds")
    tokens_in: int = Field(default=0, description="Prompt input tokens")
    tokens_out: int = Field(default=0, description="Completion output tokens")
    cost_usd: float = Field(default=0.0, description="Calculated or reported cost in USD")
    status: TraceStatus = Field(..., description="Outcome of step")
    error: str | None = Field(default=None, description="Error message if failed or errored")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO 8601 UTC timestamp",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary step metadata")


class JsonlTracer:
    """Thread-safe append-only flat JSONL trace logger."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._records: list[TraceRecord] = []

    async def log_step(
        self,
        case_id: str,
        step: TraceStep | str,
        status: TraceStatus | str,
        model: str | None = None,
        latency_ms: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceRecord:
        """Appends a single flat trace record to the JSONL trace log."""
        record = TraceRecord(
            case_id=case_id,
            step=TraceStep(step),
            status=TraceStatus(status),
            model=model,
            latency_ms=round(latency_ms, 2),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=round(cost_usd, 6),
            error=error,
            metadata=metadata or {},
        )

        line = record.model_dump_json() + "\n"

        async with self._lock:
            self._records.append(record)
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(line)

        return record

    @property
    def records(self) -> list[TraceRecord]:
        return list(self._records)

    def total_tokens(self) -> tuple[int, int]:
        total_in = sum(r.tokens_in for r in self._records)
        total_out = sum(r.tokens_out for r in self._records)
        return total_in, total_out

    def total_cost(self) -> float:
        return round(sum(r.cost_usd for r in self._records), 6)
