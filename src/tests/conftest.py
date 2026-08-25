"""Pytest safety defaults: no real provider credentials or network calls."""

import os

import pytest


# Loaded before test modules, so imports cannot discover real secrets.
os.environ["GOOGLE_API_KEY"] = "mock-google-key"
os.environ["COHERE_API_KEY"] = "mock-cohere-key"
os.environ["OLLAMA_BASE_URL"] = "http://mock-ollama.invalid:11434"
os.environ.setdefault("TEST_DOCUMENT_PATH", "")


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests that may contact configured model providers",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "network: test requires an explicitly configured model provider"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return

    skip_network = pytest.mark.skip(reason="use --run-network to enable provider calls")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)
