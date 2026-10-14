"""Asynchronous LLM Gateway client for Aegis-Eval.

Communicates with the TokenForge Anthropic-compatible gateway using
httpx.AsyncClient with timeout handling, token usage tracking, and cost calculation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from aegis.config import Settings

# Estimated pricing per 1M tokens (input / output) for standard gateway models
MODEL_PRICING_PER_MILLION: dict[str, tuple[float, float]] = {
    "glm-5.3": (0.15, 0.30),
    "claude-sonnet-5.5": (3.00, 15.00),
    "claude-opus-5.5": (15.00, 75.00),
    "gpt-6-astra": (2.50, 10.00),
    "kimi-k3": (0.60, 2.40),
}


@dataclass(frozen=True)
class LLMResponse:
    content: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_usd: float
    model: str
    raw_response: dict[str, Any]


class GatewayClient:
    """Client for dispatching chat completions to the TokenForge gateway."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.base_url = settings.base_url
        self.api_key = settings.api_key
        self.timeout = settings.timeout_seconds
        self._external_client = client

    def _calculate_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        input_rate, output_rate = MODEL_PRICING_PER_MILLION.get(model.lower(), (1.00, 3.00))
        cost = (tokens_in * input_rate + tokens_out * output_rate) / 1_000_000.0
        return round(cost, 6)

    async def generate(
        self,
        prompt: str,
        model: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Dispatches an inference request to the Anthropic-compatible /v1/messages endpoint."""
        endpoint = f"{self.base_url}/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt

        start_time = time.perf_counter()

        client = self._external_client or httpx.AsyncClient(timeout=self.timeout)
        try:
            resp = await client.post(endpoint, headers=headers, json=payload)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code != 200:
                raise RuntimeError(
                    f"Gateway returned HTTP {resp.status_code} on {endpoint}: {resp.text}"
                )

            data = resp.json()

            # Extract text content from Anthropic content block format
            content_blocks = data.get("content", [])
            content_text = ""
            if isinstance(content_blocks, list):
                for block in content_blocks:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content_text += block.get("text", "")
            elif isinstance(content_blocks, str):
                content_text = content_blocks
            else:
                content_text = str(content_blocks)

            usage = data.get("usage", {})
            tokens_in = usage.get("input_tokens", 0)
            tokens_out = usage.get("output_tokens", 0)

            cost = self._calculate_cost(model, tokens_in, tokens_out)

            return LLMResponse(
                content=content_text.strip(),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=elapsed_ms,
                cost_usd=cost,
                model=model,
                raw_response=data,
            )
        finally:
            if self._external_client is None:
                await client.aclose()
