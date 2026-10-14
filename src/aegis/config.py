"""Configuration loader and validator for Aegis-Eval."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""

    pass


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    subject_model: str
    judge_model: str
    max_concurrency: int = 4
    timeout_seconds: float = 60.0

    @classmethod
    def from_env(
        cls,
        env_path: str | Path | None = None,
        load_env_file: bool = True,
    ) -> "Settings":
        """Loads settings from environment or .env file."""
        if load_env_file:
            if env_path is not None:
                load_dotenv(env_path, override=True)
            else:
                load_dotenv(override=False)

        base_url = (os.getenv("LLM_BASE_URL") or "").strip().rstrip("/")
        api_key = (os.getenv("LLM_API_KEY") or "").strip()
        subject_model = (os.getenv("SUBJECT_MODEL") or "").strip()
        judge_model = (os.getenv("JUDGE_MODEL") or "").strip()

        if not base_url:
            raise ConfigError("LLM_BASE_URL is not set in environment or .env")
        if not api_key:
            raise ConfigError("LLM_API_KEY is not set in environment or .env")
        if not subject_model:
            raise ConfigError("SUBJECT_MODEL is not set in environment or .env")
        if not judge_model:
            raise ConfigError("JUDGE_MODEL is not set in environment or .env")

        # Crucial safeguard: An evaluator model must never grade its own outputs
        if subject_model.lower() == judge_model.lower():
            raise ConfigError(
                f"JUDGE_MODEL ('{judge_model}') cannot be identical to SUBJECT_MODEL ('{subject_model}'). "
                "An independent model must evaluate outputs to eliminate self-evaluation bias."
            )

        max_concurrency_raw = os.getenv("MAX_CONCURRENCY", "4")
        try:
            max_concurrency = int(max_concurrency_raw)
            if max_concurrency < 1:
                max_concurrency = 1
        except ValueError:
            max_concurrency = 4

        timeout_raw = os.getenv("REQUEST_TIMEOUT_SECONDS", "60")
        try:
            timeout_seconds = float(timeout_raw)
        except ValueError:
            timeout_seconds = 60.0

        return cls(
            base_url=base_url,
            api_key=api_key,
            subject_model=subject_model,
            judge_model=judge_model,
            max_concurrency=max_concurrency,
            timeout_seconds=timeout_seconds,
        )
