"""Config loads with defaults and exposes the provider's active key."""

import os
from importlib import reload

import src.config as config_module


def test_default_provider_is_huggingface(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    reload(config_module)
    assert config_module.settings.llm_provider == "huggingface"


def test_active_api_key_routes_to_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    reload(config_module)
    assert config_module.settings.active_api_key == "test-key"
    # Reset for other tests.
    monkeypatch.delenv("LLM_PROVIDER")
    monkeypatch.delenv("GROQ_API_KEY")
    reload(config_module)
