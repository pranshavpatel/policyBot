"""Unit tests for agent/llm.py's provider selection — the config flip
between Groq and any OpenAI-compatible local server (Ollama, vLLM,
llama.cpp). Doesn't call out to a real server (CI has no GPU/Ollama), so
this only tests that the right client class gets constructed with the
right settings — see the README's "Local LLM" section for how this was
verified end-to-end against a real local model instead."""
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

import agent.llm as llm_module
from agent.llm import get_llm


def test_defaults_to_groq(monkeypatch):
    monkeypatch.setattr(llm_module, "LLM_PROVIDER", "groq")
    llm = get_llm()
    assert isinstance(llm, ChatGroq)


def test_local_provider_uses_openai_compatible_client(monkeypatch):
    monkeypatch.setattr(llm_module, "LLM_PROVIDER", "local")
    monkeypatch.setattr(llm_module, "LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setattr(llm_module, "LOCAL_LLM_MODEL", "qwen3:8b")
    monkeypatch.setattr(llm_module, "LOCAL_LLM_API_KEY", "ollama")
    llm = get_llm()
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "qwen3:8b"
    assert str(llm.openai_api_base) == "http://localhost:11434/v1"


def test_temperature_is_passed_through(monkeypatch):
    monkeypatch.setattr(llm_module, "LLM_PROVIDER", "local")
    llm = get_llm(temperature=0.7)
    assert llm.temperature == 0.7


def test_local_provider_bounds_every_call(monkeypatch):
    # Regression: with no token cap and no timeout, one runaway qwen3
    # generation hung a real eval run and, by queueing every later request
    # behind it, the whole server (see config.py's LOCAL_LLM_* comment).
    monkeypatch.setattr(llm_module, "LLM_PROVIDER", "local")
    monkeypatch.setattr(llm_module, "LOCAL_LLM_MAX_TOKENS", 1234)
    monkeypatch.setattr(llm_module, "LOCAL_LLM_TIMEOUT_SECONDS", 42.0)
    llm = get_llm()
    assert llm.max_tokens == 1234
    assert llm.request_timeout == 42.0
    assert llm.max_retries == 0
