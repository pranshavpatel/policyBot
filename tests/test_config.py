"""Unit tests for config.py's parsed settings (not simple os.getenv reads,
which don't need testing)."""
import importlib
import os


def test_cors_origins_defaults_cover_local_dev_and_vercel(monkeypatch):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    import config
    importlib.reload(config)
    assert "http://localhost:5173" in config.CORS_ORIGINS
    assert "https://pranshavpatel.vercel.app" in config.CORS_ORIGINS


def test_cors_origins_parses_comma_separated_env_var(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example.com, https://b.example.com")
    import config
    importlib.reload(config)
    assert config.CORS_ORIGINS == ["https://a.example.com", "https://b.example.com"]
    # restore default for any test that runs after this one in the same process
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    importlib.reload(config)


def test_cors_origins_ignores_empty_entries(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example.com,,")
    import config
    importlib.reload(config)
    assert config.CORS_ORIGINS == ["https://a.example.com"]
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    importlib.reload(config)


def test_embed_threads_defaults_to_one(monkeypatch):
    monkeypatch.delenv("EMBED_THREADS", raising=False)
    import config
    importlib.reload(config)
    assert config.EMBED_THREADS == 1
