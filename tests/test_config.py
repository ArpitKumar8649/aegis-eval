"""Unit tests for configuration loading and validation."""

import pytest

from aegis.config import ConfigError, Settings


def test_settings_valid(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://tokenforge.ai.studio/")
    monkeypatch.setenv("LLM_API_KEY", "tf_live_test_key")
    monkeypatch.setenv("SUBJECT_MODEL", "glm-5.3")
    monkeypatch.setenv("JUDGE_MODEL", "claude-sonnet-5.5")
    monkeypatch.setenv("MAX_CONCURRENCY", "8")
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "45")

    settings = Settings.from_env(load_env_file=False)
    assert settings.base_url == "https://tokenforge.ai.studio"
    assert settings.api_key == "tf_live_test_key"
    assert settings.subject_model == "glm-5.3"
    assert settings.judge_model == "claude-sonnet-5.5"
    assert settings.max_concurrency == 8
    assert settings.timeout_seconds == 45.0


def test_settings_identical_models_rejected(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://tokenforge.ai.studio")
    monkeypatch.setenv("LLM_API_KEY", "tf_live_test_key")
    monkeypatch.setenv("SUBJECT_MODEL", "glm-5.3")
    monkeypatch.setenv("JUDGE_MODEL", "glm-5.3")

    with pytest.raises(ConfigError) as exc_info:
        Settings.from_env(load_env_file=False)
    assert "JUDGE_MODEL" in str(exc_info.value)
    assert "cannot be identical" in str(exc_info.value)


@pytest.mark.parametrize(
    "missing_var",
    ["LLM_BASE_URL", "LLM_API_KEY", "SUBJECT_MODEL", "JUDGE_MODEL"],
)
def test_settings_missing_required_variable(monkeypatch, missing_var):
    vars_dict = {
        "LLM_BASE_URL": "https://tokenforge.ai.studio",
        "LLM_API_KEY": "tf_live_test_key",
        "SUBJECT_MODEL": "glm-5.3",
        "JUDGE_MODEL": "claude-sonnet-5.5",
    }
    vars_dict.pop(missing_var)

    for k, v in vars_dict.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv(missing_var, raising=False)

    with pytest.raises(ConfigError) as exc_info:
        Settings.from_env(load_env_file=False)
    assert missing_var in str(exc_info.value)
