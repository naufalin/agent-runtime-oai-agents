"""Unit tests for application settings (config.py)."""

import pytest

from agent_runtime.config import Settings


def test_settings_defaults():
    """Settings should have sane defaults when env vars are absent."""
    s = Settings(
        _env_file=None,
        agent_runtime_bearer_token="test",
    )
    assert s.openai_model == "gpt-5.4-nano"
    assert s.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert s.openrouter_model == "z-ai/glm-5.2"
    assert s.agent_runtime_model_provider == "openai"


def test_settings_from_env(monkeypatch):
    """Settings should pick up env vars."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/testdb")
    monkeypatch.setenv("AGENT_RUNTIME_BEARER_TOKEN", "secret-token")

    s = Settings(_env_file=None)
    assert s.openai_api_key == "sk-test-123"
    assert s.openai_model == "gpt-4o"
    assert s.database_url == "postgresql://localhost/testdb"
    assert s.agent_runtime_bearer_token == "secret-token"


def test_settings_openrouter_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key-abc")
    monkeypatch.setenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-pro")

    s = Settings(_env_file=None, agent_runtime_bearer_token="t")
    assert s.openrouter_api_key == "or-key-abc"
    assert s.openrouter_model == "deepseek/deepseek-v4-pro"


def test_settings_tinyfish_env(monkeypatch):
    monkeypatch.setenv("TINYFISH_API_KEY", "tf-key-xyz")

    s = Settings(_env_file=None, agent_runtime_bearer_token="t")
    assert s.tinyfish_api_key == "tf-key-xyz"


def test_settings_model_provider_override(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_MODEL_PROVIDER", "openrouter")

    s = Settings(_env_file=None, agent_runtime_bearer_token="t")
    assert s.agent_runtime_model_provider == "openrouter"
