"""Pytest configuration and shared fixtures for wandb-mcp-server tests."""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def fake_api_key():
    """Provide a deterministic fake API key for unit tests."""
    return "fake_api_key_for_testing_1234567890"


@pytest.fixture
def server_url():
    """Get the server URL from environment or use default."""
    return os.environ.get("MCP_TEST_SERVER_URL", "http://localhost:7860")
