"""Configuration contract tests."""

from src import config


def test_ollama_base_url_uses_canonical_environment_name():
    assert config.OLLAMA_URL == "http://mock-ollama.invalid:11434"
